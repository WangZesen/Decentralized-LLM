"""Frozen, resumable 20M AdamW data-reuse campaign using the ordinary tuning controller."""

import argparse
import copy
import csv
import fcntl
import hashlib
import itertools
import json
import math
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from decentralized_llm.config import Config, SlurmLauncher, load_config, save_config
from decentralized_llm.data.cache import TokenCache, file_digest, training_boundaries
from decentralized_llm.data.sampling import GlobalSampler, sampling_plan
from decentralized_llm.launch.preflight import validate
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.controller import (
    check_result,
    execute,
    load_state,
    outstanding_allocations,
    provenance,
    save_state,
    validate_cached,
)
from decentralized_llm.tuning.jobs import Backend
from decentralized_llm.tuning.spec import Record, TuningSpec, expand, fingerprint, numeric_recipe, trial_id

if __package__:
    from scripts.report_style import METHOD_COLORS, METHOD_ORDER
    from scripts.tune_50m import accounting, boundary_probes, confirmed_candidates, seeded_state
else:
    from report_style import METHOD_COLORS, METHOD_ORDER
    from tune_50m import accounting, boundary_probes, confirmed_candidates, seeded_state

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "runs/p-tuning/20m-data-reuse-20261102"
DEFAULT_REFERENCES = PROJECT / "runs/update-20261025/data/selected-adamw.csv"
SEEDS = (42, 43, 44)
METHOD_NAMES = {"sync": "sync", "cmp": "complete", "exp": "exponential", "ring": "ring"}
METHODS = {
    METHOD_NAMES[method]: "complete" if method in {"sync", "cmp"} else f"one_peer_{METHOD_NAMES[method]}"
    for method in METHOD_ORDER
}
COLORS = {METHOD_NAMES[method]: METHOD_COLORS[method] for method in METHOD_ORDER}
FIELDS = ("lr", "beta1", "beta2")
CONTROLLER = {"max_jobs": 25, "jobs_per_allocation": 1, "max_retries": 1, "poll_seconds": 30}
BUDGET = {"core": 1056, "boundary": 288, "controls": 144, "upper_bound": 1488}


def read_json(path: Path) -> Record:
    return json.loads(path.read_text())


@contextmanager
def lock_file(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def condition_key(batch: int, method: str, gamma: int, reuse: int) -> str:
    return f"b{batch}/{method}/ef{gamma}/reuse{reuse}"


def relocated_trial(original: Record, study: Path) -> Record:
    """Resolve the documented move into batch directories without editing old states."""
    trial = copy.deepcopy(original)
    attempt = trial["attempts"][-1]
    old = Path(attempt["directory"])
    if not (old / "run/result.json").is_file():
        trials_root = next((p for p in old.parents if p.name == "trials"), None)
        if trials_root is None:
            raise ValueError(f"Cannot locate historical run: {old}")
        relocated = study.parent / trials_root.parent.name / "trials" / old.relative_to(trials_root)
        if not (relocated / "run/result.json").is_file():
            raise ValueError(f"Cannot locate historical run: {old}")
        attempt["directory"] = str(relocated.resolve())
    return trial


def historical_config(raw: Record) -> Config:
    normalized = copy.deepcopy(raw)
    distributed = normalized["distributed"]
    if "error_feedback" in distributed:
        gamma = float(distributed.pop("error_feedback"))
        if distributed.get("error_feedback_gamma", gamma) != gamma:
            raise ValueError("Conflicting historical feedback fields")
        distributed["error_feedback_gamma"] = gamma
    return Config.model_validate(normalized)


def historical_result(trial: Record) -> Record:
    """Read the pre-gamma schema explicitly; never relax current-run authentication."""
    run = Path(trial["attempts"][-1]["directory"]) / "run"
    actual = historical_config(yaml.safe_load((run / "config.yaml").read_text()))
    intended = historical_config(trial["config"])
    if numeric_recipe(actual) != numeric_recipe(intended) or actual.runtime.seed != trial["seed"]:
        raise ValueError("Historical raw configuration does not match its trial")
    metadata, result = read_json(run / "run.json"), read_json(run / "result.json")
    expected = trial["provenance"]
    if metadata["cache"] != expected["cache_identity"] or any(
        metadata["environment"][key] != expected[key]
        for key in ("source_sha256", "lock_sha256", "python_version", "torch", "numpy")
    ):
        raise ValueError("Historical result provenance mismatch")
    loss = result["validation"]["loss"]
    if (
        result["cursor"] != training_boundaries(intended, intended.model.parameter_count)[-1]
        or result["validation"]["split"] != "full"
        or type(loss) not in (int, float)
        or not math.isfinite(loss)
    ):
        raise ValueError("Historical result is not a finite full-budget evaluation")
    return result


def reference_candidates(index: Path, project: Path) -> list[Record]:
    """Authenticate only designated references, never substitute a different gamma winner."""
    with index.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["model"] == "20m" and r["batch"] in ("64", "128")]
    expected = set(itertools.product((64, 128), METHODS, (0, 1)))
    found: set[tuple[int, str, int]] = set()
    references = []
    for row in rows:
        batch, method, gamma = int(row["batch"]), row["method"], float(row["gamma"])
        key = (batch, method, int(gamma))
        if gamma not in (0, 1) or key not in expected or key in found:
            raise ValueError(f"Unexpected or duplicate reference: {key}")
        found.add(key)
        report_path = project / row["report"]
        state_path = report_path.with_name("state.json")
        state = load_state(state_path.parent)
        trials = [
            relocated_trial(t, state_path.parent)
            for t in state["trials"].values()
            if t["candidate_id"] == row["id"] and t["seed"] in SEEDS
        ]
        if state["status"] != "complete" or sorted(t["seed"] for t in trials) != list(SEEDS):
            raise ValueError(f"Reference needs a completed three-seed study: {state_path}")
        evidence = []
        recipes = []
        losses = []
        for trial in sorted(trials, key=lambda t: t["seed"]):
            if trial["status"] != "succeeded":
                raise ValueError(f"Reference trial is not successful: {trial['id']}")
            result = historical_result(trial)
            cfg = historical_config(trial["config"])
            if (
                cfg.model.parameter_count != 20403520
                or cfg.model.context_length != 512
                or cfg.training.batch_tokens != batch * 512
                or cfg.training.data_reuse_factor != 1
                or cfg.training.tokens_per_parameters != 20
                or cfg.training.epoch_tokens_per_parameters != 0.5
                or cfg.training.num_accumulation != 1
                or cfg.optimizer.name != "adamw"
                or cfg.optimizer.grad_clip != 1
                or cfg.optimizer.weight_decay != 0.1
                or cfg.lr_schedule.name != "cosine"
                or cfg.lr_schedule.warmup_tokens_per_parameters != 1
                or cfg.distributed.error_feedback_gamma != gamma
                or cfg.distributed.mode != ("ddp" if method == "sync" else "decentralized")
                or cfg.distributed.world_size != (1 if method == "sync" else 8)
                or (method != "sync" and cfg.distributed.topology != METHODS[method])
                or cfg.distributed.scheme != "awc"
                or not result["validation"].get("validation_complete")
                or result != trial["result"]
                or result["validation"]["loss"] != trial["loss"]
            ):
                raise ValueError(f"Reference does not match the approved recipe: {trial['id']}")
            recipes.append(numeric_recipe(cfg))
            losses.append(result["validation"]["loss"])
            run = Path(trial["attempts"][-1]["directory"]) / "run"
            evidence.append(
                {
                    "trial_id": trial["id"],
                    "seed": trial["seed"],
                    "loss": trial["loss"],
                    "session_seconds": result.get("session_seconds"),
                    "files": {
                        str(run / name): file_digest(run / name) for name in ("config.yaml", "run.json", "result.json")
                    },
                }
            )
        if any(recipe != recipes[0] for recipe in recipes) or not math.isclose(
            statistics.mean(losses), float(row["mean"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"Reference summary does not match raw results: {state_path}")
        references.append(
            {
                "batch": batch,
                "method": method,
                "gamma": int(gamma),
                "candidate_id": row["id"],
                "config": historical_config(trials[0]["config"]).model_dump(mode="json"),
                "trials": evidence,
                "files": {str(p): file_digest(p) for p in (state_path, report_path)},
            }
        )
    if found != expected:
        raise ValueError(f"Missing references: {expected - found}")
    return sorted(references, key=lambda r: (r["batch"], list(METHODS).index(r["method"]), r["gamma"]))


def campaign_config(reference: Record, reuse: int, root: Path, project: Path) -> Config:
    """Normalize placement for fresh runs, retaining the reference optimizer exactly."""
    raw = Config.model_validate(reference["config"]).model_dump(mode="json")
    key = condition_key(reference["batch"], reference["method"], reference["gamma"], reuse)
    raw["data"].update(cache_dir=str((project / "data/c4").resolve()), loader_mode="auto")
    raw["training"].update(data_reuse_factor=reuse, checkpoint_policy="none", log_every=50)
    raw["lr_schedule"] = {"name": "cosine", "warmup_tokens_per_parameters": 1.0, "min_lr_ratio": 0.0}
    raw["distributed"].update(
        nodes=1,
        gpus_per_node=1 if reference["method"] == "sync" else 8,
        topology=METHODS[reference["method"]],
        scheme="awc",
        error_feedback_gamma=float(reference["gamma"]),
    )
    raw["runtime"].update(
        seed=42,
        training_backend="packed",
        execution_backend="gh200",
        device="cuda",
        amp=True,
        compile="auto",
        compile_scope="model",
        compile_mode="default",
        cpu_affinity="none",
        fused_optimizer=True,
        fp32_precision=None,
        output_dir=str(root / key / "base-run"),
        compile_cache_dir=str((project / ".cache/compile").resolve()),
    )
    raw["evaluation"] = {"subset_blocks": 1024, "batch_size": 128, "seed": 12345, "fp32_precision": None}
    raw["launcher"].update(
        kind="slurm",
        partition="gpu",
        time="00:30:00",
        gpu_type="nvidia_gh200_120gb",
        environment_script=str(root / "source/environment.sh"),
        network=None,
        environment_overlays=[],
    )
    return Config.model_validate(raw)


def local_grid(config: Config) -> Record:
    opt = config.optimizer
    return {
        "optimizer.lr": [opt.lr / 2, opt.lr, opt.lr * 2],
        "optimizer.beta1": [opt.beta1**2, opt.beta1, opt.beta1**0.5],
        "optimizer.beta2": [opt.beta2**4, opt.beta2, opt.beta2**0.25],
    }


def phase_spec(base_path: Path, config: Config, *, joint: bool) -> Record:
    search = local_grid(config) if joint else {"optimizer.lr": [config.optimizer.lr]}
    return {
        "base": {"configs": [str(base_path)]},
        "seed_start": 42,
        "screening_seeds": 1 if joint else 3,
        "phases": [{"name": "joint" if joint else "fixed", "search": search, "top_k": 3 if joint else 1}],
        "confirmation": {"seeds": 3},
        "controller": dict(CONTROLLER),
    }


def plan_campaign(root: Path, index: Path, project: Path) -> Record:
    references = reference_candidates(index, project)
    conditions = []
    # Controls first, then reuse 2 and 4; each condition occupies one campaign slot.
    for reuse in (1, 2, 4):
        for reference in references:
            cfg = campaign_config(reference, reuse, root, project)
            key = condition_key(reference["batch"], reference["method"], reference["gamma"], reuse)
            conditions.append(
                {
                    "key": key,
                    "batch": reference["batch"],
                    "method": reference["method"],
                    "gamma": reference["gamma"],
                    "reuse": reuse,
                    "directory": str(root / key),
                    "reference_id": reference["candidate_id"],
                    "config": cfg.model_dump(mode="json"),
                    "grid": local_grid(cfg) if reuse > 1 else None,
                }
            )
    return {
        "version": 1,
        "kind": "20m-data-reuse",
        "root": str(root),
        "project": str(project),
        "source": str(root / "source"),
        "index": str(index),
        "index_sha256": file_digest(index),
        "references": references,
        "conditions": conditions,
        "budget": BUDGET,
        "max_studies": 2,
        "max_jobs_per_study": 25,
        "seeds": list(SEEDS),
    }


def validate_plan(manifest: Record, *, prepared: bool) -> list[Record]:
    conditions = manifest["conditions"]
    expected = set(itertools.product((64, 128), METHODS, (0, 1), (1, 2, 4)))
    keys = [(r["batch"], r["method"], r["gamma"], r["reuse"]) for r in conditions]
    if len(keys) != 48 or set(keys) != expected or manifest["budget"] != BUDGET:
        raise ValueError("Campaign matrix or budget changed")
    if manifest["max_studies"] != 2 or manifest["max_jobs_per_study"] != 25 or manifest["seeds"] != list(SEEDS):
        raise ValueError("Campaign concurrency or seed policy changed")
    reports = []
    reference_map = {(r["batch"], r["method"], r["gamma"]): r for r in manifest["references"]}
    cache = TokenCache(Path(conditions[0]["config"]["data"]["cache_dir"]))
    cache.verify()
    if not cache.manifest["validation_complete"]:
        raise ValueError("Campaign requires the complete validation split")
    for row in conditions:
        cfg = Config.model_validate(row["config"])
        intended = campaign_config(
            reference_map[row["batch"], row["method"], row["gamma"]],
            row["reuse"],
            Path(manifest["root"]),
            Path(manifest["project"]),
        )
        if cfg != intended or row["grid"] != (local_grid(cfg) if row["reuse"] > 1 else None):
            raise ValueError(f"Condition changed: {row['key']}")
        if not prepared:
            # Validate against the existing setup; dry runs do not create a snapshot.
            assert isinstance(cfg.launcher, SlurmLauncher)
            cfg.launcher.environment_script = Path(manifest["project"]) / "scripts/environment.sh"
        spec = TuningSpec.model_validate(phase_spec(Path(row["directory"]) / "base.yaml", cfg, joint=row["reuse"] > 1))
        candidates = expand(cfg, spec.phases[0])
        if len(candidates) != (27 if row["reuse"] > 1 else 1):
            raise ValueError("Search grid has duplicate or missing points")
        report: Record = {}
        for candidate in candidates:
            report = validate(Config.model_validate(candidate["config"]), launch=True, verify=False)
        reports.append({"key": row["key"], "candidates": len(candidates), "preflight": report})
    limit = Backend().partition_limit("gpu")
    if limit is not None and limit < 30:
        raise ValueError("Partition cannot accommodate 30-minute trials")
    return reports


def paired_inputs(manifest: Record) -> list[Record]:
    """Check input identities across methods and explicitly verify nested data pools."""
    evidence = []
    rows = manifest["conditions"]
    for seed in SEEDS:
        initial_hashes = set()
        # Eight method/EF combinations, same model across all batching/reuse choices.
        for row in [r for r in rows if r["batch"] == 64 and r["reuse"] == 1]:
            cfg = Config.model_validate(row["config"])
            cfg.runtime.seed = seed
            digest = hashlib.sha256()
            for name, tensor in initialize_model(cfg).state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.numpy().tobytes())
            initial_hashes.add(digest.hexdigest())
        if len(initial_hashes) != 1:
            raise ValueError("Initialization differs across methods")
        for batch in (64, 128):
            cache = TokenCache(Path(rows[0]["config"]["data"]["cache_dir"]))
            pools: dict[int, np.ndarray] = {}
            budgets = set()
            samples = []
            for reuse in (1, 2, 4):
                configs = [
                    Config.model_validate(r["config"]) for r in rows if r["batch"] == batch and r["reuse"] == reuse
                ]
                samplers = []
                for cfg in configs:
                    cfg.runtime.seed = seed
                    budgets.add(tuple(training_boundaries(cfg, cfg.model.parameter_count)))
                    samplers.append(GlobalSampler.from_config(cache, cfg))
                if len({s.digest for s in samplers}) != 1:
                    raise ValueError("Sampling differs across methods")
                sampler = samplers[0]
                first = [int(sampler.block(i)) for i in range(32)]
                pool = (
                    sampler.permutation[: sampling_plan(cache, configs[0]).total_blocks] if reuse == 1 else sampler.pool
                )
                pools[reuse] = pool.copy()
                samples.append({"reuse": reuse, "sampler_digest": sampler.digest, "first_blocks": first})
                sampler.close()
            if not (
                np.array_equal(pools[4], pools[2][: len(pools[4])])
                and np.array_equal(pools[2], pools[1][: len(pools[2])])
            ):
                raise ValueError("Reuse pools are not nested")
            if len(budgets) != 1:
                raise ValueError("Training boundaries changed with reuse")
            evidence.append(
                {
                    "seed": seed,
                    "batch": batch,
                    "initial_weights_sha256": next(iter(initial_hashes)),
                    "sampling": samples,
                }
            )
    return evidence


def snapshot(project: Path, source: Path) -> Record:
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=project)
        .decode()
        .split("\0")
    )
    for name in sorted(set(names) - {""}):
        origin = project / name
        if origin.is_file():
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
    environment = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export DECENTRALIZED_PYTHON={shlex.quote(str(project))}/.venv-$(uname -m)/bin/python\n"
        f"export PYTHONPATH={shlex.quote(str(source / 'src') + ':' + str(source))}\n"
        "export PYTHONNOUSERSITE=1\n"
        "\"$DECENTRALIZED_PYTHON\" - <<'PY'\n"
        "import importlib.metadata, sys, tomllib\nfrom pathlib import Path\n"
        "assert sys.version_info[:2] == (3, 14)\n"
        f"project = tomllib.loads(Path({str(source / 'pyproject.toml')!r}).read_text())\n"
        "for requirement in project['project']['dependencies']:\n"
        "    name, expected = requirement.split('==')\n"
        "    assert importlib.metadata.version(name).split('+')[0] == expected\n"
        "PY\n"
    )
    (source / "environment.sh").write_text(environment)
    return {str(p.relative_to(source)): file_digest(p) for p in sorted(source.rglob("*")) if p.is_file()}


def verify_manifest(manifest: Record) -> None:
    source = Path(manifest["source"])
    for name, digest in manifest["source_files"].items():
        if file_digest(source / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for name, digest in manifest["frozen_files"].items():
        if file_digest(Path(name)) != digest:
            raise ValueError(f"Frozen campaign input changed: {name}")
    if manifest["contract_sha256"] != fingerprint({k: v for k, v in manifest.items() if k != "contract_sha256"}):
        raise ValueError("Campaign manifest changed")


def prepare(root: Path, index: Path, project: Path, *, dry_run: bool = False) -> Record:
    manifest_path = root / "campaign.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        verify_manifest(manifest)
        validate_plan(manifest, prepared=True)
        return manifest
    manifest = plan_campaign(root, index, project)
    checks = validate_plan(manifest, prepared=False)
    if dry_run:
        return {"dry_run": True, "conditions": len(checks), "tuning_studies": 32, "budget": BUDGET, "preflight": checks}
    if root.exists() and any(root.iterdir()):
        raise ValueError("Incomplete preparation exists; use a new empty campaign root")
    root.mkdir(parents=True, exist_ok=True)
    with lock_file(root / ".prepare.lock"):
        manifest["created"] = time.time()
        manifest["source_files"] = snapshot(project, Path(manifest["source"]))
        manifest["provenance"] = provenance(TokenCache(Path(manifest["conditions"][0]["config"]["data"]["cache_dir"])))
        manifest["paired_inputs"] = paired_inputs(manifest)
        frozen = {str(index): file_digest(index)}
        for reference in manifest["references"]:
            frozen.update(reference["files"])
            for trial in reference["trials"]:
                frozen.update(trial["files"])
        for row in manifest["conditions"]:
            directory = Path(row["directory"])
            directory.mkdir(parents=True)
            cfg = Config.model_validate(row["config"])
            save_config(cfg, directory / "base.yaml")
            for phase in ["fixed", "joint"] if row["reuse"] > 1 else ["fixed"]:
                phase_dir = directory / phase
                phase_dir.mkdir()
                spec = phase_spec(directory / "base.yaml", cfg, joint=phase == "joint")
                path = phase_dir / "spec.yaml"
                path.write_text(yaml.safe_dump(spec, sort_keys=False))
                frozen[str(path)] = file_digest(path)
            frozen[str(directory / "base.yaml")] = file_digest(directory / "base.yaml")
        manifest["frozen_files"] = frozen
        manifest["contract_sha256"] = fingerprint(manifest)
        atomic_json(manifest_path, manifest)
        atomic_json(root / "preflight.json", {"conditions": checks})
        verify_manifest(manifest)
    return manifest


def condition_states(row: Record) -> list[Record]:
    directory = Path(row["directory"])
    paths = [
        directory / "fixed/state.json",
        directory / "joint/state.json",
        *sorted(directory.glob("boundary/*/state.json")),
    ]
    return [load_state(path.parent) for path in paths if path.is_file()]


def collect_trials(states: list[Record], evidence: Record, *, authenticate: bool = False) -> tuple[Record, Record]:
    trials: Record = {}
    allocations: Record = {}
    for state in states:
        if state["provenance"] != evidence:
            raise ValueError("Mixed campaign provenance")
        if authenticate:
            if state["status"] != "complete" or outstanding_allocations(state):
                raise ValueError("Cannot reuse unfinished work")
            validate_cached(state)
        for key, original in state["trials"].items():
            trial = copy.deepcopy(original)
            config = Config.model_validate(trial["config"])
            candidate = {"id": fingerprint(numeric_recipe(config))}
            if (
                candidate["id"] != trial["candidate_id"]
                or trial["provenance"] != evidence
                or trial_id(candidate, trial["seed"], evidence) != key
                or key != trial["id"]
            ):
                raise ValueError("Trial identity mismatch")
            if trial["attempts"]:
                trial["directory"] = str(Path(trial["attempts"][0]["directory"]).parent.resolve())
            if trial["status"] == "succeeded" and not trial["result"]["validation"].get("validation_complete"):
                raise ValueError("Incomplete validation cannot enter the campaign report")
            if authenticate and trial["status"] == "succeeded":
                result = check_result(trial, trial["attempts"][-1])
                if result != trial["result"] or result["validation"]["loss"] != trial["loss"]:
                    raise ValueError("Cached result changed")
            if key in trials and trials[key] != trial:
                raise ValueError(f"Conflicting trial records: {key}")
            trials[key] = trial
        for key, allocation in state["allocations"].items():
            if key in allocations and allocations[key] != allocation:
                raise ValueError(f"Conflicting allocation records: {key}")
            allocations[key] = allocation
    return trials, allocations


def run_phase(directory: Path, cfg: Config, spec: Record, evidence: Record, history: list[Record]) -> Record:
    directory.mkdir(parents=True, exist_ok=True)
    spec_path = directory / "spec.yaml"
    if spec_path.exists():
        if TuningSpec.model_validate(yaml.safe_load(spec_path.read_text())) != TuningSpec.model_validate(spec):
            raise ValueError(f"Frozen phase changed: {directory}")
    else:
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    if not (directory / "state.json").exists():
        trials, allocations = collect_trials(history, evidence, authenticate=True)
        # Imports are confined to this condition: reuse factor is part of numeric_recipe.
        wanted = numeric_recipe(cfg)
        for trial in trials.values():
            actual = numeric_recipe(Config.model_validate(trial["config"]))
            for field in FIELDS:
                actual["optimizer"].pop(field)
            fixed = copy.deepcopy(wanted)
            for field in FIELDS:
                fixed["optimizer"].pop(field)
            if actual != fixed:
                raise ValueError("History does not belong to this condition")
        save_state(directory, seeded_state(spec, cfg.model_dump(mode="json"), evidence, trials, allocations))
    state = load_state(directory)
    if (
        state["provenance"] != evidence
        or state["base"] != cfg.model_dump(mode="json")
        or TuningSpec.model_validate(state["spec"]) != TuningSpec.model_validate(spec)
    ):
        raise ValueError("Saved phase differs from its frozen recipe")
    if state["status"] != "complete" or outstanding_allocations(state):
        execute("resume", directory, use_local_defaults=False)
        state = load_state(directory)
    if state["status"] != "complete" or outstanding_allocations(state):
        raise RuntimeError(f"Phase needs attention: {directory}: {state.get('reason')}")
    validate_cached(state)
    return state


def make_boundary_plan(core: Record, states: list[Record]) -> Record:
    trials, _ = collect_trials(states, core["provenance"], authenticate=True)
    candidates = confirmed_candidates(trials, core["provenance"])
    if not candidates:
        return {"winner": None, "probes": []}
    winner = candidates[0]
    probes = []
    for probe in boundary_probes({**core, "winner": winner}):
        config = copy.deepcopy(winner["config"])
        config["optimizer"][probe["field"].split(".")[1]] = probe["value"]
        # Do not retain the originating trial's output location or seed.
        config["runtime"]["output_dir"] = core["base"]["runtime"]["output_dir"]
        config["runtime"]["seed"] = 42
        probes.append({**probe, "config": config})
    if len(probes) > 3:
        raise ValueError("Boundary budget exceeded")
    return {"winner": winner, "probes": probes}


def run_condition(manifest: Record, row: Record) -> None:
    directory = Path(row["directory"])
    with lock_file(directory / ".condition.lock"):
        cfg = Config.model_validate(row["config"])
        base_path = directory / "base.yaml"
        evidence = manifest["provenance"]
        fixed = run_phase(directory / "fixed", cfg, phase_spec(base_path, cfg, joint=False), evidence, [])
        if row["reuse"] > 1:
            core = run_phase(directory / "joint", cfg, phase_spec(base_path, cfg, joint=True), evidence, [fixed])
            plan = make_boundary_plan(core, [fixed, core])
            plan_path = directory / "boundary-plan.json"
            if plan_path.exists():
                if read_json(plan_path) != plan:
                    raise ValueError("The frozen boundary plan changed")
            else:
                atomic_json(plan_path, plan)
            for probe in plan["probes"]:
                destination = directory / "boundary" / probe["field"].split(".")[1]
                destination.mkdir(parents=True, exist_ok=True)
                boundary = Config.model_validate(probe["config"])
                path = destination / "base.yaml"
                if path.exists():
                    if load_config(path) != boundary:
                        raise ValueError("Boundary configuration changed")
                else:
                    save_config(boundary, path)
                run_phase(
                    destination, boundary, phase_spec(path, boundary, joint=False), evidence, condition_states(row)
                )
        states = condition_states(row)
        trials, _ = collect_trials(states, evidence, authenticate=True)
        limit = 44 if row["reuse"] > 1 else 3  # 3 fixed + 26 new screening + 6 confirmation + 9 boundary.
        if len(trials) > limit:
            raise ValueError(f"Condition exceeded its run budget: {len(trials)} > {limit}")
        candidates = confirmed_candidates(trials, evidence)
        atomic_json(
            directory / "complete.json",
            {
                "finished": time.time(),
                "unique_trials": len(trials),
                "winner_id": candidates[0]["id"] if candidates else None,
            },
        )


def lane_conditions(manifest: Record, lane: int) -> list[Record]:
    if lane not in (0, 1):
        raise ValueError("There are exactly two campaign lanes")
    return manifest["conditions"][lane::2]


def run_lane(root: Path, manifest: Record, lane: int) -> None:
    # Stable lanes keep a failed/resumed study's outstanding jobs in its original slot.
    with lock_file(root / f".lane-{lane}.lock"):
        verify_manifest(manifest)
        if (
            provenance(TokenCache(Path(manifest["conditions"][0]["config"]["data"]["cache_dir"])))
            != manifest["provenance"]
        ):
            raise ValueError("Run from the campaign's frozen source and environment")
        for row in lane_conditions(manifest, lane):
            if not (Path(row["directory"]) / "complete.json").exists():
                run_condition(manifest, row)


def locked(path: Path) -> bool:
    try:
        with lock_file(path):
            return False
    except BlockingIOError:
        return True


def frozen_command(manifest: Record, arguments: list[str]) -> list[str]:
    source = Path(manifest["source"])
    command = (
        f"cd {shlex.quote(str(source))}\nsource {shlex.quote(str(source / 'environment.sh'))}\n"
        'exec "$DECENTRALIZED_PYTHON" scripts/tune_data_reuse.py ' + shlex.join(arguments)
    )
    return ["bash", "-c", command]


def supervise(root: Path, manifest: Record) -> Record:
    with lock_file(root / ".campaign.lock"):
        verify_manifest(manifest)
        validate_plan(manifest, prepared=True)
        children: dict[int, subprocess.Popen] = {}
        for lane in (0, 1):
            if locked(root / f".lane-{lane}.lock"):
                continue  # An orphaned live lane still owns its 25-job allowance.
            with (root / f"lane-{lane}.log").open("a") as log:
                children[lane] = subprocess.Popen(
                    frozen_command(manifest, ["run", "--root", str(root), "--lane", str(lane)]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        last_report = 0.0
        while True:
            active = any(child.poll() is None for child in children.values()) or any(
                locked(root / f".lane-{lane}.lock") for lane in (0, 1)
            )
            if time.monotonic() - last_report >= 60 or not active:
                write_report(root, manifest)
                last_report = time.monotonic()
            if not active:
                result = campaign_status(root, manifest)
                atomic_json(root / "supervisor-result.json", result)
                return result
            time.sleep(5)


def start(root: Path, manifest: Record, *, dry_run: bool = False, foreground: bool = False) -> Record:
    verify_manifest(manifest)
    if dry_run:
        return {"dry_run": True, "preflight": validate_plan(manifest, prepared=True), "budget": manifest["budget"]}
    if foreground:
        return supervise(root, manifest)
    session = "reuse-" + hashlib.sha256(str(root).encode()).hexdigest()[:10]
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
        command = shlex.join(frozen_command(manifest, ["run", "--root", str(root), "--foreground"]))
        command += " >> " + shlex.quote(str(root / "controller.log")) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    return {"status": "running", "session": session, "root": str(root), "max_outstanding_jobs": 50}


def campaign_status(root: Path, manifest: Record) -> Record:
    conditions = []
    all_trials: Record = {}
    all_allocations: Record = {}
    for row in manifest["conditions"]:
        states = condition_states(row)
        trials, allocations = collect_trials(states, manifest["provenance"])
        all_trials.update(trials)
        all_allocations.update(allocations)
        conditions.append(
            {
                "key": row["key"],
                "complete": (Path(row["directory"]) / "complete.json").exists(),
                "trials": dict(Counter(t["status"] for t in trials.values())),
                "phases": [
                    {"status": s["status"], "reason": s.get("reason"), "warning": s.get("warning")} for s in states
                ],
            }
        )
    complete = sum(row["complete"] for row in conditions)
    return {
        "root": str(root),
        "status": "complete" if complete == 48 else "incomplete",
        "completed_conditions": complete,
        "conditions": conditions,
        "trials": dict(Counter(t["status"] for t in all_trials.values())),
        "unique_trials": len(all_trials),
        "outstanding_allocations": sum(a["status"] in {"submitted", "submitting"} for a in all_allocations.values()),
    }


def candidate_result(candidate: Record | None, trials: Record) -> Record | None:
    if candidate is None:
        return None
    seeds = []
    for item in candidate["seeds"]:
        trial = trials[item["trial_id"]]
        result = check_result(trial, trial["attempts"][-1])
        if not result["validation"].get("validation_complete") or result != trial["result"]:
            raise ValueError("Winner raw result changed or validation is incomplete")
        seeds.append(
            {
                **item,
                "consensus_error": result["validation"]["consensus_error"],
                "run": str(Path(trial["attempts"][-1]["directory"]) / "run"),
            }
        )
    return {
        "candidate_id": candidate["id"],
        "config": candidate["config"],
        "seeds": seeds,
        "mean_loss": statistics.mean(s["loss"] for s in seeds),
        "std_loss": statistics.stdev(s["loss"] for s in seeds),
        "mean_consensus_error": statistics.mean(s["consensus_error"] for s in seeds),
    }


def paired_difference(left: Record | None, right: Record | None, field: str = "loss") -> Record | None:
    if left is None or right is None:
        return None
    a = {s["seed"]: s[field] for s in left["seeds"]}
    b = {s["seed"]: s[field] for s in right["seeds"]}
    if set(a) != set(SEEDS) or set(b) != set(SEEDS):
        raise ValueError("A comparison requires all three paired seeds")
    values = [a[seed] - b[seed] for seed in SEEDS]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "seeds": [{"seed": seed, "delta": value} for seed, value in zip(SEEDS, values, strict=True)],
    }


def comparison_rows(results: list[Record]) -> list[Record]:
    by_key = {(r["batch"], r["method"], r["gamma"], r["reuse"]): r for r in results}
    rows = []
    for result in results:
        baseline = by_key[result["batch"], result["method"], result["gamma"], 1]
        sync = by_key[result["batch"], "sync", result["gamma"], result["reuse"]]
        sync_baseline = by_key[result["batch"], "sync", result["gamma"], 1]
        benefit = paired_difference(result["fixed"], result["retuned"])
        for view in ("fixed", "retuned"):
            penalty = paired_difference(result[view], baseline[view])
            sync_penalty = paired_difference(sync[view], sync_baseline[view])
            relative = paired_difference(penalty, sync_penalty, "delta")
            rows.append(
                {
                    **{
                        k: result[k]
                        for k in (
                            "key",
                            "batch",
                            "method",
                            "gamma",
                            "reuse",
                            "complete",
                            "trial_statuses",
                            "boundary_edges",
                        )
                    },
                    "view": view,
                    "result": result[view],
                    "reuse_penalty": penalty,
                    "relative_degradation": relative,
                    "retuning_benefit": benefit,
                }
            )
    return rows


def report_data(root: Path, manifest: Record) -> Record:
    results = []
    all_trials: Record = {}
    all_allocations: Record = {}
    observed: dict[tuple[int, str], list[float]] = {}
    for row in manifest["conditions"]:
        states = condition_states(row)
        trials, allocations = collect_trials(states, manifest["provenance"])
        all_trials.update(trials)
        all_allocations.update(allocations)
        candidates = confirmed_candidates(trials, manifest["provenance"])
        center_id = fingerprint(numeric_recipe(Config.model_validate(row["config"])))
        fixed = candidate_result(next((c for c in candidates if c["id"] == center_id), None), trials)
        complete = (Path(row["directory"]) / "complete.json").exists()
        retuned = candidate_result(candidates[0], trials) if complete and candidates else None
        edges = []
        if retuned and row["reuse"] > 1:
            tested = copy.deepcopy(row["grid"])
            plan = read_json(Path(row["directory"]) / "boundary-plan.json")
            for probe in plan["probes"]:
                tested[probe["field"]].append(probe["value"])
            for field, values in tested.items():
                if retuned["config"]["optimizer"][field.split(".")[1]] in (min(values), max(values)):
                    edges.append(field)
        group = (row["batch"], "sync" if row["method"] == "sync" else "decentralized")
        observed.setdefault(group, []).extend(
            t["result"]["session_seconds"] / 60
            for t in trials.values()
            if t["status"] == "succeeded" and t["result"].get("session_seconds", 0) > 0
        )
        results.append(
            {
                **{k: row[k] for k in ("key", "batch", "method", "gamma", "reuse")},
                "complete": complete,
                "fixed": fixed,
                "retuned": retuned,
                "boundary_edges": edges,
                "trial_statuses": dict(Counter(t["status"] for t in trials.values())),
                "unique_trials": len(trials),
                "fixed_seed_statuses": [
                    {"seed": seed, "status": t["status"] if t else "not_started", "loss": t.get("loss") if t else None}
                    for seed in SEEDS
                    for t in [trials.get(trial_id({"id": center_id}, seed, manifest["provenance"]))]
                ],
            }
        )
    remaining_hours = 0.0
    projections = []
    for result in results:
        group = (result["batch"], "sync" if result["method"] == "sync" else "decentralized")
        samples = observed.get(group, [])
        origin = "campaign" if samples else "historical references"
        if not samples:
            samples = [
                t["session_seconds"] / 60
                for ref in manifest["references"]
                for t in ref["trials"]
                if ref["batch"] == group[0]
                and (ref["method"] == "sync") == (group[1] == "sync")
                and t.get("session_seconds", 0)
            ]
        minutes = statistics.median(samples) if samples else None
        terminal = sum(result["trial_statuses"].get(k, 0) for k in ("succeeded", "diverged"))
        remaining = 0 if result["complete"] else max(0, (3 if result["reuse"] == 1 else 44) - terminal)
        if minutes is not None:
            remaining_hours += remaining * minutes / 60
        projections.append(
            {
                "key": result["key"],
                "remaining_runs_upper_bound": remaining,
                "median_minutes": minutes,
                "timing_source": origin,
            }
        )
    usage = accounting([a["job_id"] for a in all_allocations.values() if a.get("job_id")])
    return {
        "updated": time.time(),
        "status": campaign_status(root, manifest),
        "budget": BUDGET,
        "conditions": results,
        "comparisons": comparison_rows(results),
        "accounting": usage,
        "retry_attempts": sum(max(0, len(t["attempts"]) - 1) for t in all_trials.values()),
        "remaining_session_gpu_hours_upper_estimate": remaining_hours,
        "runtime_projections": projections,
        "interpretation": [
            "Global batches 64/128; 20M/context 512; fixed 20 TPP; seeds 42, 43, 44; packed GH200 BF16.",
            "Final full-validation losses evaluate averaged decentralized weights; lower is better.",
            "Negative relative degradation means AWC's reuse penalty is smaller than synchronous training's.",
            "Reported seeds and validation data select winners; these are not independent test scores.",
            "Reuse-1 references are rerun from historical searches, not retuned with an equal search budget.",
            "Local training versus averaged-model validation is not a conventional generalization gap.",
            "GPU-hour projections exclude queueing and future retries; accounting records actual allocations.",
        ],
    }


def csv_rows(report: Record) -> list[Record]:
    rows = []
    for row in report["comparisons"]:
        result = row["result"] or {}
        config = result.get("config", {})
        seeds = {s["seed"]: s["loss"] for s in result.get("seeds", [])}
        out = {k: row[k] for k in ("batch", "method", "gamma", "reuse", "view", "complete")}
        out.update({k: result.get(k) for k in ("candidate_id", "mean_loss", "std_loss", "mean_consensus_error")})
        out.update({k: config.get("optimizer", {}).get(k) for k in FIELDS})
        out.update({f"seed_{seed}_loss": seeds.get(seed) for seed in SEEDS})
        out["boundary_edges"] = ",".join(row["boundary_edges"]) if row["view"] == "retuned" else ""
        for key in ("reuse_penalty", "relative_degradation", "retuning_benefit"):
            out[key] = row[key]["mean"] if row[key] else None
            out[key + "_std"] = row[key]["std"] if row[key] else None
            out.update({f"{key}_seed_{s['seed']}": s["delta"] for s in (row[key] or {}).get("seeds", [])})
        rows.append(out)
    return rows


def plot_report(root: Path, report: Record) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    for view, metric in itertools.product(("fixed", "retuned"), ("loss", "reuse_penalty", "relative_degradation")):
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        for axis, (batch, gamma) in zip(axes.flat, itertools.product((64, 128), (0, 1)), strict=True):
            for method in METHODS:
                rows = sorted(
                    [
                        r
                        for r in report["comparisons"]
                        if (r["batch"], r["gamma"], r["view"], r["method"]) == (batch, gamma, view, method)
                    ],
                    key=lambda r: r["reuse"],
                )
                values = [
                    (r["reuse"], r["result"]["mean_loss"], r["result"]["std_loss"])
                    if metric == "loss"
                    else (r["reuse"], r[metric]["mean"], r[metric]["std"])
                    for r in rows
                    if r["result"] and (metric == "loss" or r[metric])
                ]
                if values:
                    axis.errorbar(
                        [v[0] for v in values],
                        [v[1] for v in values],
                        yerr=[v[2] for v in values],
                        marker="o",
                        capsize=3,
                        label=method,
                        color=COLORS[method],
                    )
            axis.set(
                title=f"Global batch {batch}, gamma={gamma}",
                xlabel="Data reuse factor",
                ylabel="Nats/token",
                xticks=[1, 2, 4],
            )
            axis.grid(alpha=0.2)
            if metric != "loss":
                axis.axhline(0, color="black", linewidth=0.5)
            if axis.get_legend_handles_labels()[1]:
                axis.legend(fontsize=8)
        figure.suptitle(f"{view.capitalize()}: {metric.replace('_', ' ')} (mean ± sample SD; selection seeds)")
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(plots / f"{view}-{metric}.{suffix}", dpi=150)
        plt.close(figure)
    for condition in report["conditions"]:
        for view in ("fixed", "retuned"):
            result = condition[view]
            if not result:
                continue
            directory = root / condition["key"] / "curves" / view
            marker = directory / "candidate.json"
            if marker.exists() and read_json(marker).get("id") == result["candidate_id"]:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            cfg = Config.model_validate(result["config"])
            boundaries = training_boundaries(cfg, cfg.model.parameter_count)
            for training in (False, True):
                figure, axis = plt.subplots(figsize=(8, 4.5))
                for seed in result["seeds"]:
                    path = Path(seed["run"]) / ("metrics-rank-00000.jsonl" if training else "metrics.jsonl")
                    records = [json.loads(line) for line in path.read_text().splitlines()]
                    records = [
                        r
                        for r in records
                        if r.get("event") == ("train" if training else "validation")
                        and (training or r.get("split") == "subset")
                    ]
                    x = [
                        r["step"] * cfg.training.batch_tokens
                        if training
                        else boundaries[r["epoch"] - 1] * cfg.model.context_length
                        for r in records
                    ]
                    axis.plot(
                        x,
                        [r["local_loss" if training else "loss"] for r in records],
                        label=f"seed {seed['seed']}",
                        alpha=0.8,
                    )
                label = "Local training loss" if training else "Subset validation loss (averaged model)"
                axis.set(xlabel="Processed training tokens", ylabel=label, title=f"{condition['key']} / {view}")
                axis.legend()
                axis.grid(alpha=0.2)
                figure.tight_layout()
                for suffix in ("png", "svg"):
                    figure.savefig(directory / f"{'training' if training else 'validation-subset'}.{suffix}", dpi=150)
                plt.close(figure)
            atomic_json(marker, {"id": result["candidate_id"]})


def write_report(root: Path, manifest: Record) -> Record:
    with lock_file(root / ".report.lock"):
        report = report_data(root, manifest)
        atomic_json(root / "report.json", report)
        rows = csv_rows(report)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temporary = root / "results.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "results.csv")
        for condition in report["conditions"]:
            if condition["complete"] and condition["retuned"]:
                path = root / condition["key"] / "final-winner.yaml"
                save_config(Config.model_validate(condition["retuned"]["config"]), path)
        lines = [
            "# 20M data-reuse comparison",
            "",
            f"Completed conditions: {report['status']['completed_conditions']}/48. "
            f"Outstanding allocations: {report['status']['outstanding_allocations']}.",
            "",
            *[f"- {line}" for line in report["interpretation"]],
            "",
            f"Trial outcomes: `{report['status']['trials']}`. Retry attempts: {report['retry_attempts']}.",
            "",
            f"Allocated GPU-hours: {report['accounting'].get('gpu_hours')} "
            f"(accounting complete: {report['accounting']['complete']}). "
            f"Estimated remaining session GPU-hours, upper bound before retries: "
            f"{report['remaining_session_gpu_hours_upper_estimate']:.1f}.",
            "",
            "| Batch | Method | Gamma | Reuse | View | Full loss ± SD | Reuse penalty | "
            "Relative degradation | Retuning benefit |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            number = lambda value: "pending/unavailable" if value is None else f"{value:.6f}"  # noqa: E731
            lines.append(
                f"| {row['batch']} | {row['method']} | {row['gamma']} | {row['reuse']} | {row['view']} | "
                f"{number(row['mean_loss'])} ± {number(row['std_loss'])} | {number(row['reuse_penalty'])} | "
                f"{number(row['relative_degradation'])} | {number(row['retuning_benefit'])} |"
            )
        lines += ["", "## Search boundaries and unresolved conditions", ""]
        for condition in report["conditions"]:
            if condition["boundary_edges"] or not condition["complete"] or not condition["retuned"]:
                lines.append(
                    f"- `{condition['key']}`: complete={condition['complete']}; "
                    f"edges={condition['boundary_edges']}; outcomes={condition['trial_statuses']}."
                )
        lines += ["", "## Figures", ""]
        for view in ("fixed", "retuned"):
            for metric in ("loss", "reuse_penalty", "relative_degradation"):
                lines += [f"![{view} {metric}](plots/{view}-{metric}.png)", ""]
        plot_report(root, report)
        temporary = root / "report.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(root / "report.md")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "status", "report"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name in ("prepare", "run"):
            command.add_argument("--dry-run", action="store_true")
        if name == "prepare":
            command.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
        if name == "run":
            command.add_argument("--foreground", action="store_true")
            command.add_argument("--lane", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare(root, args.references.resolve(), PROJECT, dry_run=args.dry_run)
        print(
            json.dumps({"root": str(root), "prepared": not args.dry_run, "conditions": 48, "budget": result["budget"]})
        )
        return
    manifest = read_json(root / "campaign.json")
    if args.command == "run":
        if args.lane is not None:
            if args.dry_run:
                parser.error("--lane cannot be combined with --dry-run")
            run_lane(root, manifest, args.lane)
            return
        result = start(root, manifest, dry_run=args.dry_run, foreground=args.foreground)
    elif args.command == "status":
        result = campaign_status(root, manifest)
    else:
        verify_manifest(manifest)
        result = write_report(root, manifest)["status"]
    print(json.dumps(result))
    if args.command == "run" and args.foreground and result["status"] != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
