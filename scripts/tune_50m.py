"""Run the bounded 50M campaign through the existing tuning controller.

Run one instance per study. A separate report process reads atomic controller states;
it never submits work. Boundary candidates are fixed from the original core winner.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path

import yaml

from decentralized_llm.config import Config, save_config
from decentralized_llm.data.cache import TokenCache, training_boundaries
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.controller import (
    VERSION,
    check_result,
    execute,
    load_state,
    provenance,
    save_state,
    validate_cached,
)
from decentralized_llm.tuning.estimate import ranking
from decentralized_llm.tuning.outcomes import loss_value
from decentralized_llm.tuning.spec import (
    Record,
    TuningSpec,
    expand,
    fingerprint,
    numeric_recipe,
    search_points,
    trial_id,
)

FIELDS = ("optimizer.lr", "optimizer.beta1", "optimizer.beta2")


def grids(state: Record) -> dict[str, list[float]]:
    spec = TuningSpec.model_validate(state["spec"])
    base = Config.model_validate(state["base"])
    return {
        key: sorted(point["value"] for point in search_points(base, key, search))
        for phase in spec.phases
        for key, search in phase.search.items()
    }


def boundary_probes(state: Record) -> list[Record]:
    """At most one axial neighbor per coordinate, never recursive expansion."""
    if state.get("winner") is None:
        return []
    probes = []
    for field, values in grids(state).items():
        value = state["winner"]["config"]["optimizer"][field.split(".")[1]]
        if value not in (values[0], values[-1]):
            continue
        lower = value == values[0]
        if field == "optimizer.lr":
            neighbor = value / 2 if lower else value * 2
        else:
            factor = 2 if field == "optimizer.beta1" else 4
            neighbor = value ** (factor if lower else 1 / factor)
        neighbor = float(format(neighbor, ".15g"))
        if not math.isfinite(neighbor) or neighbor <= 0 or (field != "optimizer.lr" and neighbor >= 1):
            raise ValueError(f"Invalid boundary neighbor for {field}: {neighbor}")
        probes.append({"field": field, "value": neighbor, "side": "lower" if lower else "upper"})
    return probes


def boundary_spec(core: Record, config_path: Path, probe: Record) -> Record:
    return {
        "base": {"configs": [str(config_path.resolve())]},
        "seed_start": core["spec"]["seed_start"],
        "screening_seeds": 3,
        "phases": [{"name": "boundary", "search": {probe["field"]: [probe["value"]]}, "top_k": 1}],
        "confirmation": {"seeds": 3},
        "controller": core["spec"]["controller"],
    }


def ensure_complete(directory: Path, spec: Path) -> Record:
    state = load_state(directory) if (directory / "state.json").exists() else None
    if state is None or state["status"] != "complete":
        execute(
            "run" if state is None else "resume",
            directory,
            spec if state is None else None,
            dry_run=True,
            use_local_defaults=False,
        )
        result = execute(
            "run" if state is None else "resume",
            directory,
            spec if state is None else None,
            use_local_defaults=False,
        )
        if result["status"] != "complete":
            raise RuntimeError(f"Study stopped: {directory}: {result}")
        state = load_state(directory)
    base = Config.model_validate(state["base"])
    if state["provenance"] != provenance(TokenCache(base.data.cache_dir)):
        raise ValueError(f"Frozen provenance changed: {directory}")
    return state


def run_study(directory: Path, spec: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core = ensure_complete(directory, spec)
        probes = boundary_probes(core)
        plan = {"core_winner_id": (core.get("winner") or {}).get("id"), "probes": probes}
        plan_path = directory / "boundary-plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Core winner changed after the boundary plan was frozen")
        atomic_json(plan_path, plan)
        if core.get("winner"):
            config_path = directory / "boundary-base.yaml"
            save_config(Config.model_validate(core["winner"]["config"]), config_path)
            for probe in probes:
                destination = directory / "boundary" / probe["field"].split(".")[1]
                destination.mkdir(parents=True, exist_ok=True)
                specification = destination / "spec.yaml"
                if not specification.exists():
                    specification.write_text(yaml.safe_dump(boundary_spec(core, config_path, probe), sort_keys=False))
                result = ensure_complete(destination, specification)
                if result["provenance"] != core["provenance"]:
                    raise ValueError("Boundary result provenance differs from the core search")
        atomic_json(directory / "campaign-complete.json", {"finished": time.time(), **plan})


def study_result(directory: Path) -> tuple[Record, list[Record]]:
    """Combine complete three-seed candidates without changing controller reports."""
    path = directory / "state.json"
    if not path.exists():
        return {"name": directory.name, "status": "not_started", "winner": None}, []
    core = load_state(directory)
    states = [core]
    pending = core["status"] != "complete"
    failed = core["status"] in {"failed", "interrupted"}
    candidates = []
    if core["status"] == "complete" and core.get("winner"):
        candidates.extend((row, core, directory) for row in core["phases"][-1]["rankings"])
        for probe in boundary_probes(core):
            location = directory / "boundary" / probe["field"].split(".")[1]
            if not (location / "state.json").exists():
                pending = True
                continue
            boundary = load_state(location)
            if boundary["provenance"] != core["provenance"]:
                raise ValueError(f"Mixed provenance in {location}")
            states.append(boundary)
            pending |= boundary["status"] != "complete"
            failed |= boundary["status"] in {"failed", "interrupted"}
            if boundary["status"] == "complete" and boundary.get("winner"):
                candidates.append((boundary["winner"], boundary, location))
    candidates = [item for item in candidates if math.isfinite(loss_value(item[0]["mean_loss"]))]
    candidates.sort(key=lambda item: (loss_value(item[0]["mean_loss"]), item[0]["id"]))
    winner = None
    edges = []
    if candidates:
        row, owner, location = candidates[0]
        seeds = []
        for seed in row["seeds"]:
            trial = owner["trials"][seed["trial_id"]]
            result = trial["result"]
            if trial["status"] != "succeeded" or result["validation"]["split"] != "full":
                raise ValueError("Final candidate requires successful full-validation results")
            seeds.append(
                {
                    **seed,
                    "consensus_error": result["validation"]["consensus_error"],
                    "run": str(Path(trial["attempts"][-1]["directory"]) / "run"),
                }
            )
        if [seed["seed"] for seed in seeds] != [42, 43, 44]:
            raise ValueError("Expected exactly seeds 42, 43, 44")
        winner = {
            "id": row["id"],
            "optimizer": row["config"]["optimizer"],
            "mean_loss": row["mean_loss"],
            "std_loss": row["std_loss"],
            "seeds": seeds,
            "study": str(location),
            "mean_consensus_error": statistics.mean(seed["consensus_error"] for seed in seeds),
        }
        expanded = grids(core)
        for probe in boundary_probes(core):
            expanded[probe["field"]].append(probe["value"])
        for field, values in expanded.items():
            if row["config"]["optimizer"][field.split(".")[1]] in (min(values), max(values)):
                edges.append(field)
        if not pending and not failed:
            config = Config.model_validate(row["config"])
            config.runtime.output_dir = directory / "final-winner-run"
            config.runtime.seed = 42
            save_config(config, directory / "final-winner.yaml")
    trials = [trial for state in states for trial in state["trials"].values()]
    completed = [t["result"]["session_seconds"] for t in trials if t["status"] == "succeeded"]
    observed_minutes = statistics.mean(completed) / 60 if completed else None
    planned = 27 + (3 * len(boundary_probes(core)) if core["status"] == "complete" else 0)
    status = "failed" if failed else "running" if pending else "complete" if winner else "no_finite_winner"
    row = {
        "name": directory.name,
        "status": status,
        "core_status": core["status"],
        "reason": core.get("reason"),
        "outcomes": dict(Counter(t["status"] for t in trials)),
        "unique_trials": len(trials),
        "boundary_trials": sum(len(state["trials"]) for state in states[1:]),
        "planned_trials": planned,
        "observed_mean_session_minutes": observed_minutes,
        "estimated_training_session_gpu_hours": planned * (observed_minutes or 25.22) / 60,
        "winner": winner,
        "remaining_boundary_coordinates": edges,
    }
    return row, states


def accounting(job_ids: list[str]) -> Record:
    """Use allocation elapsed time, including retries, without double-counting srun steps."""
    if not job_ids:
        return {"gpu_hours": 0.0, "accounted_jobs": 0, "total_jobs": 0, "complete": True, "jobs": []}
    try:
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(sorted(set(job_ids))),
                "--format=JobIDRaw,State,ElapsedRaw,AllocTRES",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        return parse_accounting(result.stdout, job_ids)
    except (OSError, subprocess.SubprocessError) as error:
        return {"gpu_hours": None, "complete": False, "error": str(error), "total_jobs": len(set(job_ids))}


def parse_accounting(output: str, job_ids: list[str]) -> Record:
    wanted = set(job_ids)
    jobs = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or parts[0] not in wanted or not parts[2].isdigit():
            continue
        resources = dict(item.split("=", 1) for item in parts[3].split(",") if "=" in item)
        gpus = int(resources.get("gres/gpu", "0"))
        # Some Slurm versions report only the typed GPU TRES.
        if "gres/gpu" not in resources:
            gpus = sum(int(value) for key, value in resources.items() if key.startswith("gres/gpu:"))
        pending = parts[1].split()[0] in {"PENDING", "CANCELLED", "FAILED"} and int(parts[2]) == 0
        if not gpus and not pending:
            continue
        jobs[parts[0]] = {
            "job_id": parts[0],
            "state": parts[1],
            "elapsed_seconds": int(parts[2]),
            "gpus": gpus,
            "gpu_hours": int(parts[2]) * gpus / 3600,
        }
    return {
        "gpu_hours": sum(job["gpu_hours"] for job in jobs.values()),
        "accounted_jobs": len(jobs),
        "total_jobs": len(wanted),
        "complete": len(jobs) == len(wanted),
        "jobs": list(jobs.values()),
    }


def plot_winner(directory: Path, winner: Record) -> None:
    marker = directory / "winner-curves.json"
    if marker.exists() and json.loads(marker.read_text())["winner_id"] == winner["id"]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = directory / "winner-curves"
    plots.mkdir(exist_ok=True)
    for filename, metric, training in [
        ("validation", "loss", False),
        ("consensus", "consensus_error", False),
        ("training", "local_loss", True),
    ]:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for seed in winner["seeds"]:
            path = Path(seed["run"]) / ("metrics-rank-00000.jsonl" if training else "metrics.jsonl")
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records = [
                row
                for row in records
                if row.get("event") == ("train" if training else "validation")
                and (training or row.get("split") == "subset")
                and row.get(metric) is not None
            ]
            axis.plot(
                [row["step" if training else "epoch"] for row in records],
                [row[metric] for row in records],
                label=f"seed {seed['seed']}",
                alpha=0.8,
            )
        axis.set(xlabel="Optimizer update" if training else "Virtual epoch", ylabel=metric, title=directory.name)
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(plots / f"{filename}.png", dpi=150)
        plt.close(figure)
    atomic_json(marker, {"winner_id": winner["id"], "seeds": winner["seeds"]})


def write_campaign_report(root: Path) -> Record:
    manifest = json.loads((root / "campaign.json").read_text())
    if manifest.get("kind") == "extension":
        return write_extension_report(root, manifest)
    rows, jobs_by_study = [], {}
    for study in manifest["studies"]:
        directory = Path(study["directory"])
        row, states = study_result(directory)
        rows.append(row)
        jobs_by_study[row["name"]] = {
            allocation["job_id"]
            for state in states
            for allocation in state["allocations"].values()
            if allocation.get("job_id")
        }
        if row.get("winner"):
            plot_winner(directory, row["winner"])
    usage = accounting([job for jobs in jobs_by_study.values() for job in jobs])
    for row in rows:
        known = [job for job in usage.get("jobs", []) if job["job_id"] in jobs_by_study[row["name"]]]
        row["allocated_gpu_hours"] = sum(job["gpu_hours"] for job in known) if usage.get("jobs") is not None else None
        row["accounting_complete"] = usage.get("error") is None and len(known) == len(jobs_by_study[row["name"]])
    result = {"updated": time.time(), "studies": rows, "accounting": usage}
    atomic_json(root / "report.json", result)
    lines = [
        "# 50M tuning campaign",
        "",
        "Context 1024; global batch 65,536 tokens; 20 TPP; packed GH200; seeds 42–44.",
        "",
        "Each study allows 30 jobs, one GPU per allocation, 45 minutes per allocation, and one retry.",
        "",
        "Core budget: 216 runs; with one boundary round: at most 288, excluding infrastructure retries.",
        "",
        "| Study | Status | Trials | Boundary trials | Mean loss | Sample std | Allocated GPU-hours |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        winner = row.get("winner") or {}

        def number(value: float | None) -> str:
            return "—" if value is None else f"{value:.6f}"

        lines.append(
            f"| [{row['name']}]({row['name']}/report.md) | {row['status']} | {row.get('unique_trials', 0)} | "
            f"{row.get('boundary_trials', 0)} | {number(winner.get('mean_loss'))} | "
            f"{number(winner.get('std_loss'))} | {number(row.get('allocated_gpu_hours'))} |"
        )
    lines += [
        "",
        "GPU-hours come from Slurm allocation elapsed time, including failed attempts; queue time is excluded.",
        f"Accounting complete for known jobs: {usage['complete']}. Missing accounting is not zero usage.",
        "Session-time projections in report.json use each study's completed trials, or the 25.22-minute ring proxy.",
        "Winner statistics are provisional until the study status is complete. These are tuning results.",
        "",
    ]
    for row in rows:
        if row.get("winner"):
            name = row["name"]
            lines += [
                f"## {name}",
                "",
                f"Remaining boundary coordinates: {row['remaining_boundary_coordinates']}.",
                "",
                f"Mean consensus error: {row['winner']['mean_consensus_error']:.8g}.",
                "",
                f"[Final configuration]({name}/final-winner.yaml) (available after boundary checks).",
                "",
                f"![Validation]({name}/winner-curves/validation.png)",
                "",
                f"![Training]({name}/winner-curves/training.png)",
                "",
                f"![Consensus]({name}/winner-curves/consensus.png)",
                "",
            ]
    temporary = root / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(root / "report.md")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_recipe(config: Record) -> Record:
    recipe = numeric_recipe(Config.model_validate(config))
    for field in FIELDS:
        recipe["optimizer"].pop(field.split(".")[1])
    return recipe


def import_history(paths: list[Path], reference: Record) -> tuple[Record, Record, list[Record]]:
    """Authenticate existing artifacts without moving or relabelling any run."""
    trials, allocations, sources = {}, {}, []
    for path in paths:
        state = load_state(path.parent)
        if state["status"] != "complete" or state["provenance"] != reference["provenance"]:
            raise ValueError(f"History is incomplete or has different provenance: {path}")
        validate_cached(state)
        imported = []
        for key, original in state["trials"].items():
            if original["status"] not in {"succeeded", "diverged"}:
                raise ValueError(f"History contains an unresolved trial: {key}")
            config = Config.model_validate(original["config"])
            candidate = {"id": fingerprint(numeric_recipe(config))}
            if (
                original["provenance"] != reference["provenance"]
                or fixed_recipe(original["config"]) != fixed_recipe(reference["base"])
                or original["candidate_id"] != candidate["id"]
                or key != original["id"]
                or key != trial_id(candidate, original["seed"], reference["provenance"])
            ):
                raise ValueError(f"History configuration or identity mismatch: {key}")
            if original["status"] == "succeeded":
                result = check_result(original, original["attempts"][-1])
                if result != original["result"] or result["validation"]["loss"] != original["loss"]:
                    raise ValueError(f"History result changed: {key}")
                if not result["validation"].get("validation_complete"):
                    raise ValueError(f"History validation is incomplete: {key}")
            trial = copy.deepcopy(original)
            trial["directory"] = str(Path(trial["attempts"][0]["directory"]).parent.resolve())
            if key in trials and trial != trials[key]:
                raise ValueError(f"Conflicting history for trial {key}")
            trials[key] = trial
            imported.append(key)
        for key, allocation in state["allocations"].items():
            if allocation["status"] in {"planned", "submitting", "submitted"}:
                raise ValueError(f"History allocation is still active: {key}")
            if key in allocations and allocation != allocations[key]:
                raise ValueError(f"Conflicting allocation {key}")
            allocations[key] = copy.deepcopy(allocation)
        sources.append({"state": str(path.resolve()), "sha256": sha256(path), "trial_ids": sorted(imported)})
    return trials, allocations, sources


def snap_beta(value: float, field: str, trials: Record) -> float:
    counts = Counter(
        t["config"]["optimizer"][field]
        for t in trials.values()
        if abs(t["config"]["optimizer"][field] - value) <= 2e-15
    )
    return min(counts, key=lambda x: (-counts[x], abs(x - value), x)) if counts else float(format(value, ".15g"))


def extension_spec(base_path: Path, core: Record, trials: Record) -> Record:
    sync = core["base"]["distributed"]["mode"] == "ddp"
    topology = core["base"]["distributed"]["topology"]
    beta2_exponents = (
        [1 / 4, 1 / 16, 1 / 64]
        if sync
        else [1 / 64, 1 / 256, 1 / 1024]
        if topology == "complete"
        else [1 / 256, 1 / 1024, 1 / 4096]
    )
    search = {
        "optimizer.lr": [2.0**e for e in ([-9, -8, -7] if sync else [-8, -7, -6])],
        "optimizer.beta1": [
            snap_beta(0.9**e, "beta1", trials) for e in ([1 / 2, 1 / 4, 1 / 8] if sync else [1 / 4, 1 / 8, 1 / 16])
        ],
        "optimizer.beta2": [snap_beta(0.95**e, "beta2", trials) for e in beta2_exponents],
    }
    return {
        "base": {"configs": [str(base_path.resolve())]},
        "seed_start": 42,
        "screening_seeds": 1,
        "phases": [{"name": "joint", "search": search, "top_k": 3}],
        "confirmation": {"seeds": 3},
        "controller": {**core["spec"]["controller"], "max_jobs": 30, "jobs_per_allocation": 1, "max_retries": 1},
    }


def seeded_state(spec: Record, base: Record, evidence: Record, trials: Record, allocations: Record) -> Record:
    return {
        "version": VERSION,
        "id": uuid.uuid4().hex,
        "revision": 1,
        "created": time.time(),
        "spec": TuningSpec.model_validate(spec).model_dump(mode="json"),
        "base": copy.deepcopy(base),
        "provenance": copy.deepcopy(evidence),
        "trials": copy.deepcopy(trials),
        "allocations": copy.deepcopy(allocations),
        "phases": [],
        "status": "prepared",
        "winner": None,
        "reason": None,
    }


def confirmed_candidates(trials: Record, evidence: Record) -> list[Record]:
    candidates = {}
    for trial in trials.values():
        candidates[trial["candidate_id"]] = {"id": trial["candidate_id"], "config": trial["config"]}
    rows = []
    for candidate in candidates.values():
        result = ranking([candidate], [42, 43, 44], trials, evidence)
        if result and math.isfinite(loss_value(result[0]["mean_loss"])):
            rows.append(result[0])
    return sorted(rows, key=lambda row: (loss_value(row["mean_loss"]), row["id"]))


def preferred_winner(trials: Record, evidence: Record, incumbent: Record) -> Record:
    candidates = confirmed_candidates(trials, evidence)
    return candidates[0] if candidates and candidates[0]["mean_loss"] < incumbent["mean_loss"] else incumbent


def joint_budget(state: Record) -> Record:
    spec = TuningSpec.model_validate(state["spec"])
    candidates = expand(Config.model_validate(state["base"]), spec.phases[0])
    missing = [trial_id(c, 42, state["provenance"]) for c in candidates]
    missing = sorted(key for key in missing if key not in state["trials"])
    confirmation = sorted(
        (sum(trial_id(c, seed, state["provenance"]) not in state["trials"] for seed in (43, 44)) for c in candidates),
        reverse=True,
    )
    maximum = len(missing) + sum(confirmation[:3])
    if len(candidates) != 27 or maximum > 30:
        raise ValueError(f"Joint search exceeds the approved budget: {len(candidates)=}, {maximum=}")
    return {
        "limit": 30,
        "imported_trial_ids": sorted(state["trials"]),
        "joint_candidate_count": len(candidates),
        "screening_new_trial_ids": missing,
        "screening_reused": len(candidates) - len(missing),
        "confirmation_upper_bound": sum(confirmation[:3]),
        "core_new_upper_bound": maximum,
    }


def extension_states(directory: Path) -> list[Record]:
    return [
        load_state(path.parent) for path in [directory / "state.json", *sorted(directory.glob("boundary/*/state.json"))]
    ]


def merge_evidence(states: list[Record]) -> tuple[Record, Record]:
    trials, allocations = {}, {}
    for state in states:
        if state["provenance"] != states[0]["provenance"]:
            raise ValueError("Mixed extension provenance")
        for key, trial in state["trials"].items():
            trial = copy.deepcopy(trial)
            if trial["attempts"]:
                trial["directory"] = str(Path(trial["attempts"][0]["directory"]).parent)
            if key in trials and trial != trials[key]:
                raise ValueError(f"Conflicting extension trial {key}")
            trials[key] = trial
        allocations.update(state["allocations"])
    return trials, allocations


def plan_extension_boundaries(core: Record, trials: Record, incumbent: Record, budget: Record) -> Record:
    winner = preferred_winner(trials, core["provenance"], incumbent)
    probe_state = {**core, "winner": winner}
    reserved = set(trials) - set(budget["imported_trial_ids"])
    probes = []
    for probe in boundary_probes(probe_state):
        config = copy.deepcopy(winner["config"])
        config["optimizer"][probe["field"].split(".")[1]] = probe["value"]
        candidate = {"id": fingerprint(numeric_recipe(Config.model_validate(config))), "config": config}
        missing = [trial_id(candidate, seed, core["provenance"]) for seed in (42, 43, 44)]
        missing = [key for key in missing if key not in trials and key not in reserved]
        admitted = len(reserved) + len(missing) <= budget["limit"]
        probes.append({**probe, "candidate": candidate, "new_trial_ids": missing, "admitted": admitted})
        if admitted:
            reserved.update(missing)
    if len(reserved) > budget["limit"]:
        raise ValueError("Extension budget exceeded")
    return {"winner": winner, "probes": probes, "reserved_new_trial_ids": sorted(reserved)}


def run_extension_study(directory: Path) -> None:
    with (directory / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = json.loads((directory / "extension.json").read_text())
        budget = json.loads((directory / "budget.json").read_text())
        if budget["limit"] > 30:
            raise ValueError("Extension run limit exceeds 30")
        if sha256(directory / "spec.yaml") != metadata["spec_sha256"]:
            raise ValueError("Extension specification changed")
        core = ensure_complete(directory, directory / "spec.yaml")
        plan_path = directory / "boundary-plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            plan = plan_extension_boundaries(core, core["trials"], metadata["incumbent"], budget)
            atomic_json(plan_path, plan)
        budget.update(reserved_new_trial_ids=plan["reserved_new_trial_ids"], boundary_probes=plan["probes"])
        atomic_json(directory / "budget.json", budget)
        for probe in plan["probes"]:
            if not probe["admitted"]:
                continue
            destination = directory / "boundary" / probe["field"].split(".")[1]
            destination.mkdir(parents=True, exist_ok=True)
            if not (destination / "state.json").exists():
                trials, allocations = merge_evidence(extension_states(directory))
                # Keep imported artifact paths absolute in every seeded child state.
                for trial in trials.values():
                    if trial["attempts"]:
                        trial["directory"] = str(Path(trial["attempts"][0]["directory"]).parent)
                base = probe["candidate"]["config"]
                base_path = destination / "base.yaml"
                save_config(Config.model_validate(base), base_path)
                spec = boundary_spec(core, base_path, probe)
                (destination / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
                save_state(destination, seeded_state(spec, base, core["provenance"], trials, allocations))
            ensure_complete(destination, destination / "spec.yaml")
        trials, _ = merge_evidence(extension_states(directory))
        new = set(trials) - set(budget["imported_trial_ids"])
        if not new <= set(plan["reserved_new_trial_ids"]) or len(new) > budget["limit"]:
            raise ValueError("Unbudgeted extension trial")
        budget.update(actual_new_trial_ids=sorted(new), finished=time.time())
        atomic_json(directory / "budget.json", budget)
        atomic_json(directory / "campaign-complete.json", {"finished": time.time(), "new_trials": len(new)})


def extension_result(directory: Path) -> tuple[Record, Record]:
    metadata = json.loads((directory / "extension.json").read_text())
    budget = json.loads((directory / "budget.json").read_text())
    states = extension_states(directory)
    trials, allocations = merge_evidence(states)
    baseline = metadata["incumbent"]
    best = preferred_winner(trials, states[0]["provenance"], baseline)
    winner = copy.deepcopy(best)
    winner["optimizer"] = best["config"]["optimizer"]
    baseline_seeds = {s["seed"]: s["loss"] for s in baseline["seeds"]}
    for seed in winner["seeds"]:
        trial = trials[seed["trial_id"]]
        seed.update(
            run=str(Path(trial["attempts"][-1]["directory"]) / "run"),
            consensus_error=trial["result"]["validation"]["consensus_error"],
            loss_change_from_incumbent=seed["loss"] - baseline_seeds[seed["seed"]],
        )
    winner["mean_consensus_error"] = statistics.mean(s["consensus_error"] for s in winner["seeds"])
    complete = (directory / "campaign-complete.json").exists()
    failed = any(state["status"] in {"failed", "interrupted"} for state in states)
    plan_path = directory / "boundary-plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {"probes": []}
    expanded = grids(states[0])
    for probe in plan["probes"]:
        if probe["admitted"]:
            child = directory / "boundary" / probe["field"].split(".")[1] / "state.json"
            if child.exists() and load_state(child.parent)["status"] == "complete":
                expanded[probe["field"]].append(probe["value"])
    edges = [
        field
        for field, values in expanded.items()
        if winner["optimizer"][field.split(".")[1]] <= min(values)
        or winner["optimizer"][field.split(".")[1]] >= max(values)
    ]
    new = {key: t for key, t in trials.items() if key not in set(budget["imported_trial_ids"])}
    boundary_ids = {key for p in plan["probes"] if p["admitted"] for key in p["new_trial_ids"]}
    sessions = [t["result"]["session_seconds"] for t in new.values() if t["status"] == "succeeded"]
    if not sessions:
        sessions = [t["result"]["session_seconds"] for t in trials.values() if t["status"] == "succeeded"]
    observed = statistics.mean(sessions) / 60
    imported_jobs = set(metadata["imported_job_ids"])
    new_allocations = {key: a for key, a in allocations.items() if a.get("job_id") not in imported_jobs}
    retry_jobs = {a["job_id"] for t in new.values() for a in t["attempts"][1:] if a.get("job_id")}
    boundary_jobs = {
        a["job_id"] for key, t in new.items() if key in boundary_ids for a in t["attempts"] if a.get("job_id")
    }
    if complete and not failed:
        config = Config.model_validate(winner["config"])
        config.runtime.seed = 42
        config.runtime.output_dir = directory / "final-winner-run"
        save_config(config, directory / "final-winner.yaml")
        atomic_json(directory / "final-winner.json", winner)
    return {
        "name": directory.name,
        "status": "failed" if failed else "complete" if complete else "running",
        "core_status": states[0]["status"],
        "reason": next((s.get("reason") for s in states if s.get("reason")), None),
        "incumbent": baseline,
        "winner": winner,
        "mean_loss_change": winner["mean_loss"] - baseline["mean_loss"],
        "outcomes": dict(Counter(t["status"] for t in new.values())),
        "unique_trials": len(new),
        "imported_trials": len(budget["imported_trial_ids"]),
        "boundary_trials": len(set(new) & boundary_ids),
        "retry_attempts": sum(max(0, len(t["attempts"]) - 1) for t in new.values()),
        "retry_job_ids": sorted(retry_jobs),
        "boundary_job_ids": sorted(boundary_jobs),
        "budget_limit": budget["limit"],
        "budget_remaining": budget["limit"] - len(new),
        "core_new_upper_bound": budget["core_new_upper_bound"],
        "observed_mean_session_minutes": observed,
        "estimated_maximum_session_gpu_hours": budget["limit"] * observed / 60,
        "remaining_boundary_coordinates": edges,
        "skipped_boundary_probes": [p for p in plan["probes"] if not p["admitted"]],
    }, new_allocations


def write_extension_report(root: Path, manifest: Record) -> Record:
    rows, job_sets = [], {}
    for study in manifest["studies"]:
        directory = Path(study["directory"])
        row, allocations = extension_result(directory)
        rows.append(row)
        job_sets[row["name"]] = {a["job_id"] for a in allocations.values() if a.get("job_id")}
        plot_winner(directory, row["winner"])
    usage = accounting([job for jobs in job_sets.values() for job in jobs])
    for row in rows:
        jobs = [j for j in usage.get("jobs", []) if j["job_id"] in job_sets[row["name"]]]
        row["accounting_complete"] = "jobs" in usage and len(jobs) == len(job_sets[row["name"]])
        row["allocated_gpu_hours"] = sum(j["gpu_hours"] for j in jobs) if "jobs" in usage else None
        for category in ("retry", "boundary"):
            row[f"{category}_gpu_hours"] = (
                sum(j["gpu_hours"] for j in jobs if j["job_id"] in row[f"{category}_job_ids"])
                if "jobs" in usage
                else None
            )
    result = {"updated": time.time(), "kind": "extension", "studies": rows, "accounting": usage}
    atomic_json(root / "report.json", result)
    lines = [
        "# 50M joint tuning extension",
        "",
        "Context 1024; batch 65,536 tokens; 20 TPP; frozen packed GH200 execution; seeds 42–44.",
        "",
        "Budget: at most 30 new configuration/seed runs per study, 240 overall. Infrastructure retries are additional.",
        "30 concurrent jobs per study; one GPU and one trial per 45-minute allocation.",
        "",
        "| Study | Status | New runs | Boundary | Baseline loss | Best mean | Sample std | Change | GPU-hours |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        w = row["winner"]
        hours = "unknown" if row["allocated_gpu_hours"] is None else f"{row['allocated_gpu_hours']:.3f}"
        lines.append(
            f"| [{row['name']}]({row['name']}/report.md) | {row['status']} | {row['unique_trials']}/30 | "
            f"{row['boundary_trials']} | {row['incumbent']['mean_loss']:.6f} | {w['mean_loss']:.6f} | "
            f"{w['std_loss']:.6f} | {row['mean_loss_change']:+.6f} | {hours} |"
        )
    lines += [
        "",
        "Loss changes are relative to each historical three-seed incumbent; negative is better.",
        "Only complete three-seed candidates can replace an incumbent. Results are provisional while running.",
        "GPU-hours count new Slurm allocations, including retries, and exclude historical runs and queue time.",
        f"Accounting complete for known jobs: {usage['complete']}. Missing accounting is not zero usage.",
        "",
    ]
    for row in rows:
        name = row["name"]
        lines += [
            f"## {name}",
            "",
            f"Optimizer: `{json.dumps(row['winner']['optimizer'], sort_keys=True)}`.",
            "",
            f"Mean consensus error: {row['winner']['mean_consensus_error']:.8g}.",
            "",
            f"Boundary GPU-hours: {row['boundary_gpu_hours']}; retry GPU-hours: {row['retry_gpu_hours']} "
            "(retry spending overlaps its search phase).",
            "",
            f"Remaining boundary coordinates: {row['remaining_boundary_coordinates']}.",
            "",
            f"Skipped for budget: {[p['field'] for p in row['skipped_boundary_probes']]}.",
            "",
            "| Seed | Final full-validation loss | Change from incumbent |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {s['seed']} | {s['loss']:.8f} | {s['loss_change_from_incumbent']:+.8f} |"
            for s in row["winner"]["seeds"]
        ]
        lines += [
            "",
            f"[Final configuration]({name}/final-winner.yaml) (written when complete).",
            "",
            f"![Validation]({name}/winner-curves/validation.png)",
            "",
            f"![Training]({name}/winner-curves/training.png)",
            "",
            f"![Consensus]({name}/winner-curves/consensus.png)",
            "",
        ]
    temporary = root / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(root / "report.md")
    return result


def verify_extension_files(root: Path) -> Record:
    manifest = json.loads((root / "campaign.json").read_text())
    for filename, digest in manifest["frozen_files"].items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"Frozen campaign file changed: {filename}")
    return manifest


def controller_script(source: Path, environment: Path, wrapper: Path, args: list[str], log: Path) -> str:
    command = " ".join(shlex.quote(str(arg)) for arg in [str(wrapper), *args])
    trap = f'controller_status=$?; echo "$controller_status" > {shlex.quote(str(log.with_suffix(".exit")))}'
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(source))}\nsource {shlex.quote(str(environment))}\n"
        f"exec >> {shlex.quote(str(log))} 2>&1\ntrap {shlex.quote(trap)} EXIT\n"
        f'"$DECENTRALIZED_PYTHON" -u {command}\n'
    )


def prepare_extension(source_root: Path, output: Path) -> Record:
    manifest_path = output / "campaign.json"
    if manifest_path.exists():
        manifest = verify_extension_files(output)
        if manifest.get("kind") != "extension" or manifest["source_root"] != str(source_root):
            raise ValueError("Existing campaign has different inputs")
        return manifest
    original = json.loads((source_root / "campaign.json").read_text())
    source = Path(original["source"])
    environment = Path(original["environment_script"])
    frozen = {str(source / name): digest for name, digest in original["source_files"].items()}
    frozen[str(environment)] = original["environment_script_sha256"]
    for filename, digest in frozen.items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"Original campaign snapshot changed: {filename}")
    if len(original["studies"]) != 8:
        raise ValueError("Expected eight source studies")
    wrapper = output / "orchestration" / "tune_50m.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    if wrapper.exists() and sha256(wrapper) != sha256(Path(__file__)):
        raise ValueError("An unfinished preparation has different orchestration")
    if not wrapper.exists():
        wrapper.write_bytes(Path(__file__).read_bytes())
        wrapper.chmod(0o444)
    frozen[str(wrapper)] = sha256(wrapper)
    studies = []
    for study in original["studies"]:
        previous = Path(study["directory"])
        destination = output / study["name"]
        destination.mkdir(parents=True, exist_ok=True)
        paths = [previous / "state.json", *sorted(previous.glob("boundary/*/state.json"))]
        if study["name"] == "50m-sync-1w-1024c-64bs-20tpp":
            paths.append(source_root / "checks" / "sync-1w-beta1-p025-beta2-p00625-lr-m8" / "state.json")
        if not (destination / "extension.json").exists():
            if (destination / "state.json").exists():
                raise ValueError(f"Incomplete import requires inspection: {destination}")
            core = load_state(previous)
            trials, allocations, imports = import_history(paths, core)
            base = copy.deepcopy(core["base"])
            base["launcher"]["time"] = "00:45:00"
            base_path = destination / "base.yaml"
            save_config(Config.model_validate(base), base_path)
            spec = extension_spec(base_path, core, trials)
            specification = destination / "spec.yaml"
            specification.write_text(yaml.safe_dump(spec, sort_keys=False))
            state = seeded_state(spec, base, core["provenance"], trials, allocations)
            budget = joint_budget(state)
            incumbent = confirmed_candidates(trials, core["provenance"])[0]
            metadata = {
                "source_study": str(previous),
                "imports": imports,
                "incumbent": incumbent,
                "spec_sha256": sha256(specification),
                "imported_job_ids": sorted({a["job_id"] for a in allocations.values() if a.get("job_id")}),
            }
            atomic_json(destination / "budget.json", budget)
            save_state(destination, state)
            atomic_json(destination / "extension.json", metadata)
        controller = destination / "controller.sh"
        controller.write_text(
            controller_script(
                source,
                environment,
                wrapper,
                ["extend-study", "--output", str(destination)],
                destination / "controller.log",
            )
        )
        for filename in ("spec.yaml", "base.yaml", "extension.json", "controller.sh"):
            path = destination / filename
            frozen[str(path)] = sha256(path)
        studies.append({"name": study["name"], "directory": str(destination), "controller": str(controller)})
    monitor = output / "monitor.sh"
    monitor.write_text(
        controller_script(
            source, environment, wrapper, ["report", "--root", str(output), "--watch"], output / "monitor.log"
        )
    )
    frozen[str(monitor)] = sha256(monitor)
    manifest = {
        "kind": "extension",
        "created": time.time(),
        "source_root": str(source_root),
        "source": str(source),
        "environment_script": str(environment),
        "orchestration": str(wrapper),
        "studies": studies,
        "frozen_files": frozen,
        "new_trials_per_study": 30,
        "max_jobs_per_study": 30,
        "jobs_per_allocation": 1,
        "allocation_time": "00:45:00",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def validate_extension(root: Path, manifest: Record) -> Record:
    checks = []
    for study in manifest["studies"]:
        directory = Path(study["directory"])
        state = load_state(directory)
        base = Config.model_validate(state["base"])
        spec = TuningSpec.model_validate(state["spec"])
        if base.launcher.kind != "slurm":
            raise ValueError("Extension requires Slurm allocations")
        sync = "-sync-" in study["name"]
        topology = (
            "complete"
            if sync or "-cmp-" in study["name"]
            else "one_peer_ring"
            if "-ring-" in study["name"]
            else "one_peer_exponential"
        )
        expected = (
            base.model.parameter_count == 48507392
            and base.model.context_length == 1024
            and base.training.batch_tokens == 65536
            and base.training.tokens_per_parameters == 20
            and base.lr_schedule.warmup_tokens_per_parameters == 1
            and base.lr_schedule.name == "cosine"
            and base.training.checkpoint_policy == "none"
            and base.runtime.training_backend == "packed"
            and base.runtime.execution_backend == "gh200"
            and base.physical_gpu_count == 1
            and base.distributed.world_size == (1 if sync else 8)
            and base.distributed.topology == topology
            and base.distributed.error_feedback_gamma == (1.0 if "-ef-" in study["name"] else 0.0)
            and base.launcher.time == "00:45:00"
            and spec.controller.max_jobs == 30
            and spec.controller.jobs_per_allocation == 1
            and spec.controller.max_retries == 1
            and spec.screening_seeds == 1
            and spec.seed_start == 42
            and spec.confirmation.seeds == 3
            and spec.phases[0].top_k == 3
            and len(spec.phases) == 1
        )
        if not expected:
            raise ValueError(f"Extension recipe changed: {study['name']}")
        budget = json.loads((directory / "budget.json").read_text())
        # Authenticate imports again, including cached losses, before any submissions.
        metadata = json.loads((directory / "extension.json").read_text())
        for source in metadata["imports"]:
            if sha256(Path(source["state"])) != source["sha256"]:
                raise ValueError(f"Historical state changed: {source['state']}")
        dry_run = execute("resume", directory, dry_run=True, use_local_defaults=False)
        new = set(state["trials"]) - set(budget["imported_trial_ids"])
        if len(new) + dry_run["new_jobs"] > budget["limit"]:
            raise ValueError(f"Remaining joint search exceeds budget: {study['name']}")
        for phase in dry_run["phases"]:
            for allocation in phase["allocations"]:
                if allocation["time"] != "00:45:00" or allocation["jobs"] != 1:
                    raise ValueError("Allocation sizing changed")
        atomic_json(directory / "dry-run.json", dry_run)
        check = {
            "study": study["name"],
            "passed": True,
            "model_parameters": base.model.parameter_count,
            "training_tokens": training_boundaries(base, base.model.parameter_count)[-1] * base.model.context_length,
            "logical_workers": base.distributed.world_size,
            "physical_gpus": base.physical_gpu_count,
            "gamma": base.distributed.error_feedback_gamma,
            "topology": topology,
            "max_jobs": spec.controller.max_jobs,
            "allocation_time": base.launcher.time,
            "core_new_upper_bound": budget["core_new_upper_bound"],
            "screening_reused": budget["screening_reused"],
        }
        checks.append(check)
        print(json.dumps(check), flush=True)
    result = {"passed": True, "studies": checks, "core_new_upper_bound": sum(c["core_new_upper_bound"] for c in checks)}
    atomic_json(root / "validation.json", result)
    return result


def launch_extension(root: Path, manifest: Record) -> Record:
    socket = "tuning50m-extension-" + fingerprint(str(root))[:12]
    session = "campaign"
    if all((Path(s["directory"]) / "campaign-complete.json").exists() for s in manifest["studies"]):
        return {"status": "already_complete"}
    exists = subprocess.run(["tmux", "-L", socket, "has-session", "-t", session], capture_output=True).returncode == 0
    if not exists:
        subprocess.run(
            [
                "tmux",
                "-L",
                socket,
                "new-session",
                "-d",
                "-s",
                session,
                "-n",
                "report",
                "bash",
                str(root / "monitor.sh"),
            ],
            check=True,
        )
    windows = set(
        subprocess.check_output(
            ["tmux", "-L", socket, "list-windows", "-t", session, "-F", "#{window_name}"], text=True
        ).splitlines()
    )
    if "report" not in windows:
        subprocess.run(
            ["tmux", "-L", socket, "new-window", "-d", "-t", session, "-n", "report", "bash", str(root / "monitor.sh")],
            check=True,
        )
    for study in manifest["studies"]:
        name = study["name"].removeprefix("50m-").removesuffix("-1024c-64bs-20tpp")
        if name in windows or (Path(study["directory"]) / "campaign-complete.json").exists():
            continue
        subprocess.run(
            ["tmux", "-L", socket, "new-window", "-d", "-t", session, "-n", name, "bash", study["controller"]],
            check=True,
        )
    receipt = {
        "launched": time.time(),
        "tmux_socket": socket,
        "tmux_session": session,
        "studies": [s["name"] for s in manifest["studies"]],
    }
    atomic_json(root / "launch.json", receipt)
    return receipt


def extend_campaign(source_root: Path, output: Path, dry_run: bool) -> Record:
    if source_root == output or source_root.is_relative_to(output):
        raise ValueError("Extension output must be separate from historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".extension.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare_extension(source_root, output)
        validation = validate_extension(output, manifest)
        with (output / ".report.lock").open("a") as report_lock:
            try:
                fcntl.flock(report_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass  # The running report watcher already owns this output.
            else:
                write_extension_report(output, manifest)
        return validation if dry_run else launch_extension(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--watch", action="store_true")
    extension = commands.add_parser("extend")
    extension.add_argument("--source-root", type=Path, required=True)
    extension.add_argument("--output", type=Path, required=True)
    extension.add_argument("--dry-run", action="store_true")
    extension_study = commands.add_parser("extend-study")
    extension_study.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_study(args.output.resolve(), args.spec.resolve())
    elif args.command == "extend":
        print(json.dumps(extend_campaign(args.source_root.resolve(), args.output.resolve(), args.dry_run)), flush=True)
    elif args.command == "extend-study":
        verify_extension_files(args.output.resolve().parent)
        run_extension_study(args.output.resolve())
    else:
        args.root.mkdir(parents=True, exist_ok=True)
        with (args.root / ".report.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                result = write_campaign_report(args.root.resolve())
                print(json.dumps({row["name"]: row["status"] for row in result["studies"]}), flush=True)
                finished = all(row["status"] in {"complete", "no_finite_winner"} for row in result["studies"])
                if not args.watch or (finished and result["accounting"]["complete"]):
                    break
                time.sleep(60)


if __name__ == "__main__":
    main()
