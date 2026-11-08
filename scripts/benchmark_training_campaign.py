"""Prepare, extend and supervise real eight-GPU steady-training benchmarks."""

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from decentralized_llm.benchmarks.protocol import benchmark_bounds
from decentralized_llm.config import Config, SlurmLauncher, load_config, save_config
from decentralized_llm.data.cache import TokenCache, file_digest
from decentralized_llm.launch.preflight import validate
from decentralized_llm.launch.slurm import SubmissionRejected, scheduler_arguments, slurm_scripts, submit_batch
from decentralized_llm.runtime import atomic_json, source_digest
from decentralized_llm.tuning.jobs import ACTIVE, FAILED, Backend, process_identity, scheduler_status

Record = dict[str, Any]
DEFAULT_ROOT = Path("runs/benchmarks/steady-8gpu-20261022")
MODELS = ("20m", "50m", "90m")
MODEL_PRESETS = (*MODELS, "250m", "500m")
CONTEXTS = (512, 1024, 2048)
BATCHES = (32, 64, 128)
VARIANTS = (
    ("sync", "ddp", "complete"),
    ("complete", "decentralized", "complete"),
    ("ring", "decentralized", "one_peer_ring"),
    ("exponential", "decentralized", "one_peer_exponential"),
)
PILOTS = {("20m", 512, 32), ("90m", 2048, 128)}
MAX_JOBS = 8
TERMINAL = FAILED | {"COMPLETED", "REJECTED"}


@contextmanager
def working_directory(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def read_json(path: Path) -> Record:
    return json.loads(path.read_text())


def matrix(root: Path, cache: Path, project: Path | None = None) -> tuple[dict[str, Config], list[Record]]:
    """Build the entire fixed matrix without creating files or reading local defaults."""
    root, cache = root.resolve(), cache.resolve()
    project = project or Path(__file__).resolve().parents[1]
    configurations, allocations = {}, []
    name_prefix = "steady8-" + hashlib.sha256(str(root).encode()).hexdigest()[:10]
    for model in MODELS:
        base = load_config(
            [
                project / "configs/base.yaml",
                project / f"configs/models/{model}.yaml",
                project / "configs/setups/gh200-8gpu.yaml",
            ]
        )
        for context in CONTEXTS:
            for batch in BATCHES:
                for gamma in (0.0, 1.0):
                    group = f"{model}-c{context}-b{batch}-ef{int(gamma)}"
                    keys = []
                    for variant, mode, topology in VARIANTS:
                        key = f"{group}-{variant}"
                        raw = base.model_dump(mode="json")
                        raw["model"]["context_length"] = context
                        raw["data"].update(cache_dir=str(cache), loader_mode="resident")
                        raw["training"].update(
                            batch_tokens=context * batch,
                            num_accumulation=1,
                            checkpoint_policy="none",
                            log_every=1_000_000_000,
                        )
                        raw["runtime"].update(
                            training_backend="distributed",
                            execution_backend="gh200",
                            compile=True,
                            compile_scope="model",
                            amp=True,
                            seed=42,
                            deterministic=False,
                            fused_optimizer=True,
                            decentralized_optimizer="auto",
                            output_dir=str(root / "cases" / key),
                            compile_cache_dir=str((project / ".cache/compile").resolve()),
                        )
                        raw["distributed"].update(
                            mode=mode,
                            nodes=2,
                            gpus_per_node=4,
                            backend="nccl",
                            scheme="awc",
                            topology=topology,
                            error_feedback_gamma=gamma,
                        )
                        raw["benchmark"].update(
                            warmup=30,
                            steps=60,
                            windows=3,
                            min_seconds=15.0,
                            max_chunks=math.ceil(128 * 1024**2 / (60 * context * batch)),
                            profile=False,
                            wait_for_resident=True,
                        )
                        raw["launcher"].update(
                            time="00:07:00",
                            gpu_type="nvidia_gh200_120gb",
                            cpus_per_task=None,
                            memory=None,
                            environment_script=str(root / "source/environment.sh"),
                            environment_overlays=[str(root / "source/scripts/activate_cxi.sh")],
                        )
                        configurations[key] = Config.model_validate(raw)
                        keys.append(key)
                    folder = root / "allocations" / group
                    allocations.append(
                        {
                            "id": group,
                            "name": f"{name_prefix}-{len(allocations):03d}",
                            "pilot": (model, context, batch) in PILOTS,
                            "cases": keys,
                            "directory": str(folder),
                            "receipt_path": str(folder / "receipt.json"),
                            "state": "PLANNED",
                            "created": time.time(),
                        }
                    )
    return configurations, allocations


def plan_summary(configs: dict[str, Config], allocations: list[Record]) -> Record:
    return {
        "cases": len(configs),
        "allocations": len(allocations),
        "cases_per_allocation": 4,
        "pilot_allocations": [a["id"] for a in allocations if a["pilot"]],
        "max_jobs": MAX_JOBS,
        "nodes_per_allocation": 2,
        "gpus_per_allocation": 8,
        "allocation_time": "00:30:00",
        "case_time": "00:07:00",
        "timing_windows": 3,
        "minimum_window_seconds": 15,
        "batch_size_semantics": "global sequences per optimizer update",
        "accumulation": 1,
        "minimum_measurement_gpu_hours": len(configs) * 45 * 8 / 3600,
        "allocation_limit_gpu_hours": len(allocations) * 0.5 * 8,
    }


def verify_snapshot(manifest: Record) -> None:
    source = Path(manifest["source"])
    for relative, expected in manifest["source_files"].items():
        if file_digest(source / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    for row in manifest["cases"]:
        for attempt in [*row.get("history", []), row]:
            if file_digest(Path(attempt["config"])) != attempt["config_sha256"]:
                raise ValueError(f"frozen configuration changed: {row['key']}")
    for filename, expected in manifest["launch_files"].items():
        if file_digest(Path(filename)) != expected:
            raise ValueError(f"frozen launch artifact changed: {filename}")
    if file_digest(Path(manifest["cache"]) / "manifest.json") != manifest["cache_manifest_sha256"]:
        raise ValueError("verified token cache manifest changed")


def frozen_environment(source: Path, project: Path) -> str:
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"source {shlex.quote(str(source / 'scripts/toolchain.sh'))}\n"
        f'activate_toolchain {shlex.quote(str(source))} "$(uname -m)"\n'
        f"export DECENTRALIZED_PYTHON={shlex.quote(str(project))}/.venv-$(uname -m)/bin/python\n"
        f"export PYTHONPATH={shlex.quote(str(source / 'src'))}\nexport PYTHONNOUSERSITE=1\n"
        'export PATH="$(dirname "$DECENTRALIZED_PYTHON"):$PATH"\n'
        "\"$DECENTRALIZED_PYTHON\" - <<'PY'\n"
        "import importlib.metadata, sys, tomllib\nfrom pathlib import Path\n"
        "assert sys.version_info[:2] == (3, 14)\n"
        f"project = tomllib.loads(Path({str(source / 'pyproject.toml')!r}).read_text())\n"
        "for requirement in project['project']['dependencies']:\n"
        "    name, expected = requirement.split('==')\n"
        "    assert importlib.metadata.version(name).split('+')[0] == expected\nPY\n"
    )


def write_allocations(
    source: Path, configs: dict[str, Config], allocations: list[Record]
) -> tuple[list[Record], dict[str, str]]:
    """Freeze configs and sequential Slurm steps for newly planned allocations."""
    cases, launch_files = [], {}
    for allocation in allocations:
        folder = Path(allocation["directory"])
        folder.mkdir(parents=True)
        steps = []
        for key in allocation["cases"]:
            config = configs[key]
            directory = config.runtime.output_dir
            directory.mkdir(parents=True)
            filename, worker, step = directory / "config.yaml", directory / "worker.sh", directory / "step.sh"
            save_config(config, filename)
            with working_directory(source):
                batch_text, worker_text = slurm_scripts(
                    config, filename, worker, action="benchmark-training", step_time="00:07:00"
                )
            worker.write_text(worker_text)
            step.write_text(batch_text)
            for path in (worker, step):
                launch_files[str(path)] = file_digest(path)
            execution = directory / "execution.json"
            steps.append(
                {
                    "trial_id": key,
                    "attempt": 1,
                    "script": str(step),
                    "time_minutes": 7,
                    "stdout": str(directory / "stdout.log"),
                    "stderr": str(directory / "stderr.log"),
                    "execution_record": str(execution),
                }
            )
            model, context, batch, ef, variant = key.split("-", 4)
            cases.append(
                {
                    "key": key,
                    "allocation_id": allocation["id"],
                    "model": model,
                    "context_length": int(context[1:]),
                    "global_batch_size": int(batch[1:]),
                    "error_feedback_gamma": float(ef[2:]),
                    "variant": variant,
                    "directory": str(directory),
                    "execution_record": str(execution),
                    "config": str(filename),
                    "config_sha256": file_digest(filename),
                }
            )
        execution_manifest = folder / "steps.json"
        atomic_json(execution_manifest, {"allocation_id": allocation["id"], "steps": steps})
        script = folder / "submit.sh"
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"cd {shlex.quote(str(source))}\nsource {shlex.quote(str(source / 'environment.sh'))}\n"
            'exec "$DECENTRALIZED_PYTHON" -m decentralized_llm.tuning.slurm_runner '
            + shlex.quote(str(execution_manifest))
            + "\n"
        )
        config = configs[allocation["cases"][0]].model_copy(deep=True)
        assert isinstance(config.launcher, SlurmLauncher)
        config.launcher.time = "00:30:00"
        allocation.update(
            script=str(script),
            arguments=scheduler_arguments(config)
            + [f"--job-name={allocation['name']}", f"--output={folder}/slurm-%j.out", f"--error={folder}/slurm-%j.err"],
        )
        for path in (script, execution_manifest):
            launch_files[str(path)] = file_digest(path)
    return cases, launch_files


def prepare(root: Path, cache: Path, *, dry_run: bool = False) -> Record:
    root, cache = root.resolve(), cache.resolve()
    manifest_path = root / "campaign.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["cache"] != str(cache):
            raise ValueError("campaign already uses a different token cache")
        verify_snapshot(manifest)
        return manifest if not dry_run else {**manifest["plan"], "prepared": True, "root": str(root)}
    project = Path(__file__).resolve().parents[1]
    configs, allocations = matrix(root, cache, project)
    token_cache = TokenCache(cache)
    # The dry run needs only metadata and memory feasibility, and writes no files.
    for config in configs.values():
        checked = config.model_copy(deep=True)
        assert isinstance(checked.launcher, SlurmLauncher)
        checked.launcher.environment_script = project / "scripts/environment.sh"
        checked.launcher.environment_overlays = [project / "scripts/activate_cxi.sh"]
        validate(checked, verify=False, benchmark=True)
    summary = {
        **plan_summary(configs, allocations),
        "root": str(root),
        "cache": str(cache),
        "cache_identity": token_cache.manifest["identity"],
    }
    if dry_run:
        return summary
    token_cache.verify()  # Exactly once for a newly prepared campaign, never in each rank/job.
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    if source.exists():
        raise ValueError("incomplete campaign preparation already exists; choose a new root")
    source.mkdir()
    shutil.copytree(project / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (source / "scripts").mkdir()
    for filename in ("benchmark_training_campaign.py", "activate_cxi.sh", "toolchain.sh"):
        shutil.copy2(project / "scripts" / filename, source / "scripts" / filename)
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copy2(project / filename, source / filename)
    for settings in project.glob(".toolchain-*.sh"):
        shutil.copy2(settings, source / settings.name)
    (source / "environment.sh").write_text(frozen_environment(source, project))
    cases, launch_files = write_allocations(source, configs, allocations)
    manifest = {
        "version": 1,
        "created": time.time(),
        "root": str(root),
        "source": str(source),
        "source_sha256": source_digest(source / "src/decentralized_llm"),
        "lock_sha256": file_digest(source / "uv.lock"),
        "source_files": {str(p.relative_to(source)): file_digest(p) for p in sorted(source.rglob("*")) if p.is_file()},
        "launch_files": launch_files,
        "cache": str(cache),
        "cache_identity": token_cache.manifest["identity"],
        "cache_manifest_sha256": file_digest(cache / "manifest.json"),
        "cache_verified_at": time.time(),
        "plan": summary,
        "cases": cases,
        "allocations": allocations,
        "state": "PREPARED",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def extend_batches(root: Path, batches: list[int], *, dry_run: bool = False) -> Record:
    """Append batch sizes using frozen recipes, retaining all existing evidence."""
    if not batches or any(type(batch) is not int or batch <= 0 or batch % 8 for batch in batches):
        raise ValueError("batch sizes must be positive integers divisible by eight")
    if len(batches) != len(set(batches)):
        raise ValueError("batch sizes must be distinct")
    return extend_matrix(root, batches=batches, dry_run=dry_run)


def extend_models(root: Path, models: list[str], *, dry_run: bool = False) -> Record:
    """Append model presets across every registered context and batch size."""
    if not models or any(model not in MODEL_PRESETS for model in models):
        raise ValueError(f"models must be selected from {MODEL_PRESETS}")
    if len(models) != len(set(models)):
        raise ValueError("models must be distinct")
    return extend_matrix(root, models=models, dry_run=dry_run)


def extend_matrix(
    root: Path, *, batches: list[int] | None = None, models: list[str] | None = None, dry_run: bool = False
) -> Record:
    """Preflight and freeze one new matrix dimension without changing prior evidence."""
    if bool(batches) == bool(models):
        raise ValueError("extend exactly one dimension: batch sizes or models")
    root = root.resolve()
    lock_path = root / ".controller.lock"
    lock_context = (lock_path.open("r") if lock_path.exists() else nullcontext()) if dry_run else lock_path.open("a")
    with lock_context as lock:
        try:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("stop the campaign controller before extending its matrix") from None
        manifest = read_json(root / "campaign.json")
        verify_snapshot(manifest)
        existing = {case["key"]: case for case in manifest["cases"]}
        existing_models = sorted({case["model"] for case in existing.values()}, key=lambda name: int(name[:-1]))
        contexts = sorted({case["context_length"] for case in existing.values()})
        existing_batches = sorted({case["global_batch_size"] for case in existing.values()})
        source = Path(manifest["source"])
        configs: dict[str, Config] = {}
        allocations: list[Record] = []
        preflights: dict[str, Record] = {}
        templates: dict[str, str] = {}
        presets: dict[str, Record] = {}
        preset_contents: dict[str, bytes] = {}
        name_prefix = "steady8-" + hashlib.sha256(str(root).encode()).hexdigest()[:10]
        dimensions = (
            [(f"batch {batch}", list(product(existing_models, contexts, [batch]))) for batch in sorted(batches)]
            if batches
            else [
                (f"model {model}", list(product([model], contexts, existing_batches)))
                for model in sorted(models or [], key=lambda name: int(name[:-1]))
            ]
        )
        for dimension, combinations in dimensions:
            expected = {
                f"{model}-c{context}-b{batch}-ef{int(gamma)}-{variant}"
                for model, context, batch in combinations
                for gamma in (0.0, 1.0)
                for variant, _, _ in VARIANTS
            }
            if expected <= existing.keys():
                continue  # Repeated preparation never creates duplicate cases or jobs.
            if expected & existing.keys():
                raise ValueError(f"{dimension} has an incomplete matrix in the manifest")
            for model, context, batch in combinations:
                if models and model not in presets:
                    path = Path(__file__).resolve().parents[1] / f"configs/models/{model}.yaml"
                    content = path.read_bytes()
                    preset = yaml.safe_load(content)
                    if set(preset) != {"model"} or set(preset["model"]) != {"layers", "width", "heads", "ffn_width"}:
                        raise ValueError(f"model preset must contain only architecture dimensions: {path}")
                    preset_contents[model] = content
                    presets[model] = {
                        "path": str(path),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "architecture": preset["model"],
                    }
                for gamma in (0.0, 1.0):
                    group = f"{model}-c{context}-b{batch}-ef{int(gamma)}"
                    keys = []
                    for variant, _, _ in VARIANTS:
                        key = f"{group}-{variant}"
                        candidates = [
                            case
                            for case in existing.values()
                            if (case["context_length"], case["error_feedback_gamma"], case["variant"])
                            == (context, gamma, variant)
                            and (case["model"] == model if batches else case["global_batch_size"] == batch)
                        ]
                        if not candidates:
                            raise ValueError(f"no frozen template available for {key}")
                        template = max(
                            candidates,
                            key=lambda case: case["global_batch_size"] if batches else int(case["model"][:-1]),
                        )
                        raw = load_config(template["config"]).model_dump(mode="json")
                        if models:
                            raw["model"].update(presets[model]["architecture"])
                        raw["training"]["batch_tokens"] = context * batch
                        raw["runtime"]["output_dir"] = str(root / "cases" / key)
                        # Twice the original capacity avoids the observed short-window failures.
                        raw["benchmark"]["max_chunks"] = math.ceil(256 * 1024**2 / (60 * context * batch))
                        config = Config.model_validate(raw)
                        assert config.training.num_accumulation == 1 and config.physical_gpu_count == 8
                        assert config.runtime.training_backend == "distributed"
                        configs[key] = config
                        templates[key] = template["key"]
                        keys.append(key)
                    folder = root / "allocations" / group
                    pilot = (
                        (model, context) in {(existing_models[0], contexts[0]), (existing_models[-1], contexts[-1])}
                        if batches
                        else (context, batch)
                        in {(contexts[0], existing_batches[0]), (contexts[-1], existing_batches[-1])}
                    )
                    allocations.append(
                        {
                            "id": group,
                            "name": f"{name_prefix}-{len(manifest['allocations']) + len(allocations):03d}",
                            "pilot": pilot,
                            "cases": keys,
                            "directory": str(folder),
                            "receipt_path": str(folder / "receipt.json"),
                            "state": "PLANNED",
                            "created": time.time(),
                        }
                    )
        if configs:
            if any(allocation["state"] not in TERMINAL for allocation in manifest["allocations"]):
                raise ValueError("extension requires known terminal allocations; reconcile existing work first")
            for path in [
                *(c.runtime.output_dir for c in configs.values()),
                *(Path(a["directory"]) for a in allocations),
            ]:
                if path.exists():
                    raise ValueError(f"unpublished extension artifacts already exist: {path}")
            for key, config in configs.items():
                checked = validate(config, verify=False, benchmark=True)
                bounds = benchmark_bounds(config)
                preflights[key] = {
                    "max_chunks": config.benchmark.max_chunks,
                    "maximum_updates_per_window": bounds.chunk * config.benchmark.max_chunks,
                    "maximum_loader_updates": bounds.maximum_steps,
                    "loader_memory_bytes_per_rank": checked["memory_bytes_per_rank"],
                }
        preview = {
            "batch_sizes": sorted(batches) if batches else existing_batches,
            "models": sorted(models, key=lambda name: int(name[:-1])) if models else existing_models,
            "context_lengths": contexts,
            "model_presets": presets,
            "model_parameters": {key.split("-", 1)[0]: config.model.parameter_count for key, config in configs.items()},
            "added_cases": len(configs),
            "added_allocations": len(allocations),
            "total_cases": len(existing) + len(configs),
            "total_allocations": len(manifest["allocations"]) + len(allocations),
            "pilot_allocations": [a["id"] for a in allocations if a["pilot"]],
            "max_jobs": MAX_JOBS,
            "nodes_per_allocation": 2,
            "gpus_per_allocation": 8,
            "allocation_time": "00:30:00",
            "case_time": "00:07:00",
            "window_capacity_tokens": 256 * 1024**2,
            "training_source_sha256": manifest["source_sha256"],
            "template_cases": templates,
            "measurement_bounds": preflights,
        }
        if dry_run or not configs:
            return preview
        # Save the prior manifest before publishing any additions. Training sources and
        # existing config/report artifacts are never rewritten by an extension.
        extension_id = (
            "batches-" + "-".join(map(str, sorted(batches)))
            if batches
            else "models-" + "-".join(sorted(models or [], key=lambda name: int(name[:-1])))
        )
        archive = root / "extensions" / extension_id
        archive.mkdir(parents=True)
        atomic_json(archive / "campaign-before.json", manifest)
        cases, launch_files = write_allocations(source, configs, allocations)
        for model, content in preset_contents.items():
            path = archive / f"{model}.yaml"
            path.write_bytes(content)
            launch_files[str(path)] = file_digest(path)
        for case in cases:
            case.update(extension_id=extension_id, template_case=templates[case["key"]])
            if models:
                case.update(
                    model_preset=str(archive / f"{case['model']}.yaml"),
                    model_preset_sha256=presets[case["model"]]["sha256"],
                )
        manifest["cases"].extend(cases)
        manifest["allocations"].extend(allocations)
        manifest["launch_files"].update(launch_files)
        all_configs = {case["key"]: load_config(case["config"]) for case in manifest["cases"]}
        manifest["plan"].update(plan_summary(all_configs, manifest["allocations"]))
        manifest["extensions"] = [
            *manifest.get("extensions", []),
            {"id": extension_id, "created": time.time(), **preview},
        ]
        manifest["state"] = "EXTENSION_PREPARED"
        atomic_json(archive / "plan.json", preview)
        atomic_json(root / "campaign.json", manifest)
        return preview


def failure_details(directory: Path) -> tuple[str, str]:
    text = ""
    for path in (directory / "stderr.log", directory / "stdout.log"):
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 32768))
                text += handle.read().decode(errors="replace")
    for path in directory.glob("failure-rank-*.json"):
        text += path.read_text()
    lower = text.lower()
    if "out of memory" in lower or "outofmemoryerror" in lower:
        return "oom", "CUDA out of memory; requested batch and accumulation were preserved"
    if "nonfinite" in lower:
        return "nonfinite", "nonfinite training loss or gradient"
    duration_message = "bounded benchmark window did not reach its minimum duration"
    duration_errors = [
        line[line.index(duration_message) :].strip() for line in text.splitlines() if duration_message in line
    ]
    if duration_errors:
        reason = next(
            (line for line in duration_errors if "after " in line and " chunks)" in line), duration_errors[-1]
        )
        return "invalid", reason
    if "time limit" in lower or "timed out" in lower or "timeout" in lower:
        return "timeout", "benchmark step exceeded its time limit or a distributed operation timed out"
    return "failed", "benchmark process failed; see per-case stdout/stderr and execution record"


def case_allocation(case: Record, allocation: Record) -> Record:
    """Resolve the exact allocation attempt that produced this case's evidence."""
    expected = case.get("allocation_attempt", 1)
    for attempt in [allocation, *allocation.get("history", [])]:
        if attempt.get("attempt", 1) == expected:
            return attempt
    raise ValueError(f"allocation history is missing attempt {expected} for {case['key']}")


def case_result(case: Record, allocation: Record, manifest: Record) -> Record:
    """Accept only complete, matching eight-rank evidence and a clean step exit."""
    allocation = case_allocation(case, allocation)
    config = load_config(case["config"])
    bounds = benchmark_bounds(config)
    row = {key: case[key] for key in ("key", "model", "context_length", "global_batch_size", "variant")}
    row.update(
        parameters=config.model.parameter_count,
        error_feedback_gamma=config.distributed.error_feedback_gamma,
        topology=config.distributed.topology,
        mode=config.distributed.mode,
        local_batch_size=config.micro_batch_size,
        global_batch_tokens=config.training.batch_tokens,
        benchmark_warmup_updates=bounds.warmup,
        benchmark_updates_per_chunk=bounds.chunk,
        benchmark_max_chunks=config.benchmark.max_chunks,
        benchmark_windows=config.benchmark.windows,
        benchmark_min_seconds=config.benchmark.min_seconds,
        benchmark_max_updates_per_window=bounds.chunk * config.benchmark.max_chunks,
        benchmark_max_global_tokens_per_window=bounds.chunk
        * config.benchmark.max_chunks
        * config.training.batch_tokens,
        benchmark_maximum_updates=bounds.maximum_steps,
        benchmark_maximum_global_tokens=bounds.total_blocks * config.model.context_length,
        allocation_id=allocation["id"],
        job_id=allocation.get("job_id"),
        directory=case["directory"],
        status="pending",
        reason=None,
        global_tokens_per_second=None,
        per_gpu_tokens_per_second=None,
        max_peak_allocated_bytes=None,
        max_peak_reserved_bytes=None,
        sum_peak_allocated_bytes=None,
        sum_peak_reserved_bytes=None,
        window_min_tokens_per_second=None,
        window_max_tokens_per_second=None,
        window_cv=None,
        global_sequences_per_second=None,
        seconds_per_update=None,
        source_sha256=manifest["source_sha256"],
        extension_id=case.get("extension_id"),
        template_case=case.get("template_case"),
        model_preset=case.get("model_preset"),
        model_preset_sha256=case.get("model_preset_sha256"),
        config_sha256=case["config_sha256"],
        lock_sha256=manifest["lock_sha256"],
        cache_identity=manifest["cache_identity"],
        attempt=case.get("attempt", 1),
        allocation_attempt=allocation.get("attempt", 1),
        attempt_history=case.get("history", []),
        measurement_adjustment=case.get("measurement_adjustment"),
    )
    directory = Path(case["directory"])
    execution_path = Path(case["execution_record"])
    try:
        execution = read_json(execution_path) if execution_path.exists() else None
    except (OSError, ValueError) as error:
        return {**row, "status": "invalid", "reason": f"malformed step execution record: {error}"}
    if execution and any(
        execution.get(key) != value
        for key, value in {
            "allocation_id": allocation["id"],
            "trial_id": case["key"],
            "attempt": case.get("attempt", 1),
        }.items()
    ):
        return {**row, "status": "invalid", "reason": "step execution identity mismatch"}
    if execution and execution.get("state") == "FAILED":
        row["status"], row["reason"] = failure_details(directory)
        if row["status"] == "failed" and allocation["state"] == "TIMEOUT":
            row.update(status="timeout", reason="allocation exceeded its walltime")
        return row
    paths = [directory / f"benchmark-rank-{rank:05d}.json" for rank in range(8)]
    if not all(path.is_file() for path in paths):
        if allocation["state"] in TERMINAL:
            row.update(
                status="missing" if execution is None else "failed",
                reason="allocation ended without eight complete rank reports",
            )
            if allocation["state"] == "OUT_OF_MEMORY":
                row["status"] = "oom"
            elif allocation["state"] == "TIMEOUT":
                row["status"] = "timeout"
        elif allocation["state"] in {"SUBMITTING", "UNKNOWN"}:
            row.update(status="unknown", reason="submission or scheduler outcome remains unresolved")
        elif execution and execution.get("state") == "RUNNING":
            row["status"] = "running"
        return row
    try:
        reports = [read_json(path) for path in paths]
        errors = []
        expected = config.model_dump(mode="json")
        for rank, report in enumerate(reports):
            prefix = f"rank {rank}: "
            if report.get("rank") != rank or report.get("config") != expected:
                errors.append(prefix + "rank/configuration mismatch")
            if (
                report.get("world_size"),
                report.get("physical_world_size"),
                report.get("physical_gpu_count"),
                report.get("training_backend"),
            ) != (8, 8, 8, "distributed"):
                errors.append(prefix + "not eight physical distributed workers")
            if report.get("environment", {}).get("source_sha256") != manifest["source_sha256"]:
                errors.append(prefix + "source fingerprint mismatch")
            if report.get("environment", {}).get("lock_sha256") != manifest["lock_sha256"]:
                errors.append(prefix + "dependency lock mismatch")
            if report.get("cache_identity") != manifest["cache_identity"]:
                errors.append(prefix + "token cache mismatch")
            if report.get("slurm_job_id") != allocation.get("job_id"):
                errors.append(prefix + "allocation mismatch")
            policy = report.get("execution", {})
            if policy.get("backend") != "gh200" or policy.get("compile") is not True:
                errors.append(prefix + "optimized compiled GH200 execution was not used")
            if not report.get("resident_at_measurement") or not report.get("measurement", {}).get("valid"):
                errors.append(
                    prefix
                    + "measurement was not resident and valid: "
                    + "; ".join(report.get("measurement", {}).get("invalid_reasons", []))
                )
            windows = report.get("windows", [])
            if len(windows) != 3 or any(w.get("seconds", 0) < 15 or w.get("updates", 0) < 60 for w in windows):
                errors.append(prefix + "three complete 15-second measurement windows are required")
            if any(
                not math.isfinite(float(w.get("global_tokens_per_second", 0)))
                or w.get("global_tokens_per_second", 0) <= 0
                for w in windows
            ):
                errors.append(prefix + "invalid throughput")
            if windows != reports[0].get("windows"):
                # Compiler diagnostics may legitimately differ by rank; timing and token counts may not.
                timing_fields = (
                    "updates",
                    "seconds",
                    "global_tokens",
                    "global_sequences",
                    "global_tokens_per_second",
                    "seconds_per_update",
                )
                if [{k: w.get(k) for k in timing_fields} for w in windows] != [
                    {k: w.get(k) for k in timing_fields} for w in reports[0].get("windows", [])
                ]:
                    errors.append(prefix + "rank timing windows disagree")
            for window in windows:
                updates = window.get("updates", 0)
                tokens = updates * config.training.batch_tokens
                if (
                    updates % 60
                    or updates > 60 * config.benchmark.max_chunks
                    or window.get("global_tokens") != tokens
                    or not math.isclose(
                        window.get("global_tokens_per_second", 0) * window.get("seconds", 0), tokens, rel_tol=1e-10
                    )
                ):
                    errors.append(prefix + "window update/token accounting mismatch")
                seconds = window.get("seconds", 0)
                if seconds > 0 and updates > 0:
                    derived = {
                        "global_sequences": tokens // config.model.context_length,
                        "global_sequences_per_second": tokens / config.model.context_length / seconds,
                        "per_gpu_tokens_per_second": tokens / seconds / 8,
                        "seconds_per_update": seconds / updates,
                    }
                    if any(
                        not math.isclose(window.get(key, -1), value, rel_tol=1e-10) for key, value in derived.items()
                    ):
                        errors.append(prefix + "derived window throughput or update time mismatch")
            if any(
                not isinstance(report.get(key), int) or report[key] <= 0
                for key in ("peak_allocated_bytes", "peak_reserved_bytes")
            ):
                errors.append(prefix + "missing training memory peaks")
        placements = reports[0].get("rank_measurements", [])
        if any(report.get("rank_measurements") != placements for report in reports):
            errors.append("gathered rank evidence differs across rank reports")
        if len(placements) != 8 or sorted(p.get("rank", -1) for p in placements) != list(range(8)):
            errors.append("eight rank placement records are required")
        else:
            if len({p.get("gpu_uuid") for p in placements}) != 8:
                errors.append("eight distinct GPU UUIDs are required")
            for placement in placements:
                own = reports[placement["rank"]]
                identity = ("hostname", "local_rank", "gpu_uuid", "peak_allocated_bytes", "peak_reserved_bytes")
                if any(placement.get(key) != own.get(key) for key in identity):
                    errors.append(f"rank {placement['rank']}: gathered placement differs from its own report")
                network = placement.get("network_environment", {})
                required = {"NCCL_NET": "AWS Libfabric", "FI_PROVIDER": "cxi", "OFI_NCCL_PROTOCOL": "RDMA"}
                if any(network.get(key) != value for key, value in required.items()):
                    errors.append(f"rank {placement['rank']}: NCCL/CXI RDMA environment mismatch")
                libraries = placement.get("loaded_libraries", [])
                if any(
                    not any(name in path for path in libraries) for name in ("libnccl-net-ofi", "libfabric", "libcxi")
                ):
                    errors.append(f"rank {placement['rank']}: required NCCL/CXI libraries were not loaded")
            hosts = Counter(p.get("hostname") for p in placements)
            if len(hosts) != 2 or set(hosts.values()) != {4}:
                errors.append("placement must be exactly two hosts with four ranks each")
            for host in hosts:
                local = [p for p in placements if p.get("hostname") == host]
                if sorted(p.get("local_rank", -1) for p in local) != list(range(4)):
                    errors.append(f"invalid local ranks on {host}")
                uuids = {p.get("gpu_uuid") for p in local}
                if len(uuids) != 4 or None in uuids or "" in uuids:
                    errors.append(f"GPU placement is not unique on {host}")
                if any("GH200" not in str(p.get("gpu")) for p in local):
                    errors.append(f"non-GH200 placement on {host}")
        if errors:
            return {**row, "status": "invalid", "reason": "; ".join(errors)}
        if not execution or execution.get("state") != "COMPLETED" or execution.get("exit_code") != 0:
            return {
                **row,
                "status": "running" if allocation["state"] not in TERMINAL else "failed",
                "reason": "rank reports exist but the benchmark step has not exited cleanly",
            }
        rates = [float(window["global_tokens_per_second"]) for window in reports[0]["windows"]]
        median = statistics.median(rates)
        row.update(
            status="complete",
            global_tokens_per_second=median,
            per_gpu_tokens_per_second=median / 8,
            max_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in reports),
            max_peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in reports),
            sum_peak_allocated_bytes=sum(r["peak_allocated_bytes"] for r in reports),
            sum_peak_reserved_bytes=sum(r["peak_reserved_bytes"] for r in reports),
            window_min_tokens_per_second=min(rates),
            window_max_tokens_per_second=max(rates),
            window_cv=statistics.pstdev(rates) / statistics.mean(rates),
            global_sequences_per_second=median / config.model.context_length,
            seconds_per_update=statistics.median(w["seconds"] / w["updates"] for w in reports[0]["windows"]),
            hostnames=sorted({r["hostname"] for r in reports}),
            gpu_names=sorted({p["gpu"] for p in placements}),
            rank_memory=[
                {key: p[key] for key in ("rank", "hostname", "gpu_uuid", "peak_allocated_bytes", "peak_reserved_bytes")}
                for p in placements
            ],
            transport=[
                {key: p[key] for key in ("rank", "network_environment", "loaded_libraries")} for p in placements
            ],
            windows=reports[0]["windows"],
        )
        return row
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {**row, "status": "invalid", "reason": f"malformed rank report: {error}"}


def summarize(manifest: Record) -> Record:
    allocations = {a["id"]: a for a in manifest["allocations"]}
    rows = [case_result(case, allocations[case["allocation_id"]], manifest) for case in manifest["cases"]]
    counts = Counter(row["status"] for row in rows)
    pilot_ids = {a["id"] for a in manifest["allocations"] if a["pilot"]}
    pilot_rows = [row for row in rows if row["allocation_id"] in pilot_ids]
    pilot_complete = all(row["status"] == "complete" for row in pilot_rows)
    # A clean case remains valid if a different case failed later in the same allocation.
    pilot_clean = all(
        case_allocation(case, allocations[case["allocation_id"]])["state"] in TERMINAL
        for case in manifest["cases"]
        if case["allocation_id"] in pilot_ids
    )
    pilot_failed = any(
        row["status"] in {"invalid", "failed", "oom", "nonfinite", "missing", "timeout"} for row in pilot_rows
    )
    accepted_ooms = {
        row["key"]
        for row in pilot_rows
        if row["status"] == "oom"
        and any(
            all(record.get(key) == row.get(key) for key in ("key", "config_sha256", "attempt", "job_id"))
            for record in manifest.get("accepted_pilot_ooms", [])
        )
    }
    pilot_blocked = any(
        row["status"] in {"invalid", "failed", "oom", "nonfinite", "missing", "timeout"}
        and row["key"] not in accepted_ooms
        for row in pilot_rows
    )
    return {
        "updated": time.time(),
        "state": manifest.get("state"),
        "counts": dict(counts),
        "expected_cases": len(manifest["cases"]),
        "complete": bool(rows) and counts["complete"] == len(manifest["cases"]),
        "pilot_passed": pilot_complete and pilot_clean,
        "pilot_failed": pilot_failed,
        "pilot_gate_satisfied": pilot_clean
        and all(row["status"] == "complete" or row["key"] in accepted_ooms for row in pilot_rows),
        "pilot_blocked": pilot_blocked,
        "accepted_pilot_ooms": manifest.get("accepted_pilot_ooms", []),
        "source_sha256": manifest["source_sha256"],
        "allocation_attempts": [
            {key: attempt.get(key) for key in ("id", "attempt", "name", "job_id", "state", "directory")}
            for allocation in manifest["allocations"]
            for attempt in [*allocation.get("history", []), allocation]
        ],
        "controller_revisions": manifest.get("controller_revisions", []),
        "rows": rows,
    }


def accept_pilot_ooms(root: Path, keys: list[str], *, reason: str, dry_run: bool = False) -> Record:
    """Record explicit permission to continue past these terminal pilot OOMs without retrying them."""
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("accept-oom requires distinct --case names")
    if not reason.strip():
        raise ValueError("accept-oom requires a nonempty reason")
    root = root.resolve()
    lock_path = root / ".controller.lock"
    lock_context = (lock_path.open("r") if lock_path.exists() else nullcontext()) if dry_run else lock_path.open("a")
    with lock_context as lock:
        try:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("stop the campaign controller before accepting pilot OOMs") from None
        manifest = read_json(root / "campaign.json")
        verify_snapshot(manifest)
        summary = summarize(manifest)
        rows = {row["key"]: row for row in summary["rows"]}
        allocations = {a["id"]: a for a in manifest["allocations"]}
        additions = []
        for key in keys:
            if key not in rows:
                raise ValueError(f"unknown campaign case: {key}")
            row = rows[key]
            allocation = allocations[row["allocation_id"]]
            if not allocation["pilot"] or allocation["state"] not in TERMINAL or row["status"] != "oom":
                raise ValueError(f"only terminal pilot OOMs can be accepted: {key} ({row['status']})")
            identity = {field: row[field] for field in ("key", "config_sha256", "attempt", "job_id")}
            if any(
                all(record.get(field) == value for field, value in identity.items())
                for record in manifest.get("accepted_pilot_ooms", [])
            ):
                continue
            additions.append({**identity, "accepted_at": time.time(), "reason": reason})
        manifest["accepted_pilot_ooms"] = [*manifest.get("accepted_pilot_ooms", []), *additions]
        updated = summarize(manifest)
        preview = {
            "newly_accepted_cases": [record["key"] for record in additions],
            "accepted_pilot_ooms": updated["accepted_pilot_ooms"],
            "pilot_gate_satisfied": updated["pilot_gate_satisfied"],
            "pilot_blocked": updated["pilot_blocked"],
            "counts": updated["counts"],
            "remaining_allocations": [a["id"] for a in manifest["allocations"] if a["state"] == "PLANNED"],
            "max_jobs": MAX_JOBS,
        }
        if not dry_run and additions:
            manifest["state"] = "CONTINUATION_PREPARED"
            atomic_json(root / "campaign.json", manifest)
        return preview


def retry_cases(root: Path, keys: list[str], *, dry_run: bool = False, max_chunks_factor: int = 1) -> Record:
    """Retry failures, optionally extending measurement capacity after memory preflight."""
    root = root.resolve()
    if type(max_chunks_factor) is not int or max_chunks_factor < 1:
        raise ValueError("max_chunks_factor must be a positive integer")
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("retry requires distinct --case names")
    lock_path = root / ".controller.lock"
    # A dry run only reads an existing lock; it never creates an artifact.
    lock_context = (lock_path.open("r") if lock_path.exists() else nullcontext()) if dry_run else lock_path.open("a")
    with lock_context as lock:
        try:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("stop the campaign controller before preparing retries") from None
        manifest = read_json(root / "campaign.json")
        verify_snapshot(manifest)
        cases = {case["key"]: case for case in manifest["cases"]}
        allocations = {allocation["id"]: allocation for allocation in manifest["allocations"]}
        selected: dict[str, list[Record]] = {}
        configurations: dict[str, Config] = {}
        measurement_changes: dict[str, Record] = {}
        for key in keys:
            if key not in cases:
                raise ValueError(f"unknown campaign case: {key}")
            case = cases[key]
            allocation = allocations[case["allocation_id"]]
            if allocation["state"] not in TERMINAL:
                raise ValueError(
                    f"retry requires a known terminal allocation: {allocation['id']} ({allocation['state']})"
                )
            result = case_result(case, allocation, manifest)
            if result["status"] not in {"failed", "invalid", "missing", "oom", "nonfinite", "timeout"}:
                raise ValueError(f"case is not a terminal failure: {key} ({result['status']})")
            selected.setdefault(allocation["id"], []).append(case)
            config = load_config(case["config"])
            if max_chunks_factor != 1:
                previous_chunks = config.benchmark.max_chunks
                config.benchmark.max_chunks *= max_chunks_factor
                # Check every expanded horizon before creating any retry artifacts. The
                # immutable cache was verified at preparation; do not reread its tokens.
                preflight = validate(config, verify=False, benchmark=True)
                bounds = benchmark_bounds(config)
                measurement_changes[key] = {
                    "max_chunks_factor": max_chunks_factor,
                    "previous_max_chunks": previous_chunks,
                    "max_chunks": config.benchmark.max_chunks,
                    "maximum_updates_per_window": bounds.chunk * config.benchmark.max_chunks,
                    "maximum_loader_updates": bounds.maximum_steps,
                    "loader_memory_bytes_per_rank": preflight["memory_bytes_per_rank"],
                }
            configurations[key] = config
        preview = {
            "retry_cases": keys,
            "retry_allocations": [
                {
                    "id": key,
                    "attempt": allocations[key].get("attempt", 1) + 1,
                    "prior_job_id": allocations[key].get("job_id"),
                    "cases": [case["key"] for case in group],
                }
                for key, group in selected.items()
            ],
            "training_source_sha256": manifest["source_sha256"],
            "training_settings_changed": False,
            "measurement_changes": measurement_changes,
        }
        if dry_run:
            return preview
        source = Path(manifest["source"])
        for group_id, group in selected.items():
            allocation = allocations[group_id]
            old_run = copy.deepcopy({key: value for key, value in allocation.items() if key != "history"})
            old_run.setdefault("attempt", 1)
            number = old_run["attempt"] + 1
            history = [*allocation.get("history", []), old_run]
            folder = root / "allocations" / group_id / f"attempt-{number:03d}"
            if folder.exists():
                raise ValueError(f"retry artifacts already exist without a published attempt: {folder}")
            folder.mkdir()
            steps = []
            for case in group:
                previous = copy.deepcopy({key: value for key, value in case.items() if key != "history"})
                previous.setdefault("attempt", 1)
                previous.setdefault("allocation_attempt", 1)
                previous["job_id"] = case_allocation(case, allocation).get("job_id")
                outcome = case_result(case, allocation, manifest)
                previous.update(status=outcome["status"], reason=outcome["reason"])
                config = configurations[case["key"]]
                directory = root / "cases" / case["key"] / f"attempt-{number:03d}"
                directory.mkdir()
                config.runtime.output_dir = directory
                filename, worker, step = directory / "config.yaml", directory / "worker.sh", directory / "step.sh"
                save_config(config, filename)
                with working_directory(source):
                    batch_text, worker_text = slurm_scripts(
                        config, filename, worker, action="benchmark-training", step_time="00:07:00"
                    )
                worker.write_text(worker_text)
                step.write_text(batch_text)
                execution = directory / "execution.json"
                case.update(
                    attempt=previous["attempt"] + 1,
                    allocation_attempt=number,
                    history=[*case.get("history", []), previous],
                    directory=str(directory),
                    config=str(filename),
                    config_sha256=file_digest(filename),
                    execution_record=str(execution),
                )
                if case["key"] in measurement_changes:
                    case["measurement_adjustment"] = measurement_changes[case["key"]]
                steps.append(
                    {
                        "trial_id": case["key"],
                        "attempt": case["attempt"],
                        "script": str(step),
                        "time_minutes": 7,
                        "stdout": str(directory / "stdout.log"),
                        "stderr": str(directory / "stderr.log"),
                        "execution_record": str(execution),
                    }
                )
                for path in (worker, step):
                    manifest["launch_files"][str(path)] = file_digest(path)
            execution_manifest = folder / "steps.json"
            atomic_json(execution_manifest, {"allocation_id": group_id, "steps": steps})
            script = folder / "submit.sh"
            script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"cd {shlex.quote(str(source))}\nsource {shlex.quote(str(source / 'environment.sh'))}\n"
                'exec "$DECENTRALIZED_PYTHON" -m decentralized_llm.tuning.slurm_runner '
                + shlex.quote(str(execution_manifest))
                + "\n"
            )
            base_name = allocation.get("base_name", allocation["name"])
            name = f"{base_name}-a{number:02d}"
            config = load_config(group[0]["config"])
            assert isinstance(config.launcher, SlurmLauncher)
            config.launcher.time = "00:30:00"
            updated = {
                "id": group_id,
                "name": name,
                "base_name": base_name,
                "pilot": allocation["pilot"],
                "cases": allocation["cases"],
                "active_cases": [case["key"] for case in group],
                "attempt": number,
                "history": history,
                "directory": str(folder),
                "created": time.time(),
                "state": "PLANNED",
                "receipt_path": str(folder / "receipt.json"),
                "script": str(script),
                "arguments": scheduler_arguments(config)
                + [f"--job-name={name}", f"--output={folder}/slurm-%j.out", f"--error={folder}/slurm-%j.err"],
            }
            allocation.clear()
            allocation.update(updated)
            for path in (script, execution_manifest):
                manifest["launch_files"][str(path)] = file_digest(path)
        manifest["state"] = "RETRY_PREPARED"
        atomic_json(root / "campaign.json", manifest)
        return preview


def reconcile(manifest: Record, *, persist_receipts: bool = True) -> None:
    outstanding = [a for a in manifest["allocations"] if a["state"] in {"SUBMITTING", "SUBMITTED", "UNKNOWN"} | ACTIVE]
    if not outstanding:
        return
    statuses = scheduler_status(outstanding)
    for allocation in outstanding:
        status = statuses.get(allocation["name"])
        if status is None:
            allocation["state"] = "UNKNOWN"
            continue  # Absence from accounting is never authorization to resubmit.
        allocation.update(status)
        receipt = Path(allocation["receipt_path"])
        if persist_receipts and not receipt.exists():
            atomic_json(
                receipt, {"job_id": allocation["job_id"], "cluster": allocation.get("cluster"), "recovered": True}
            )


def submit_ready(root: Path, manifest: Record, summary: Record) -> None:
    outstanding = sum(a["state"] in {"SUBMITTING", "SUBMITTED", "UNKNOWN"} | ACTIVE for a in manifest["allocations"])
    gate_satisfied = summary.get("pilot_gate_satisfied", summary["pilot_passed"])
    pilot_blocked = summary.get("pilot_blocked", summary["pilot_failed"])
    for allocation in manifest["allocations"]:
        if outstanding >= MAX_JOBS:
            break
        if allocation["state"] != "PLANNED" or (not allocation["pilot"] and not gate_satisfied):
            continue
        if pilot_blocked and not (allocation["pilot"] and allocation.get("attempt", 1) > 1):
            continue
        allocation.update(state="SUBMITTING", submission_intent_at=time.time())
        atomic_json(root / "campaign.json", manifest)
        try:
            receipt = submit_batch(allocation["arguments"], Path(allocation["script"]))
        except SubmissionRejected as error:
            allocation.update(state="REJECTED", error=str(error))
            atomic_json(root / "campaign.json", manifest)
            # Fail closed before submitting further allocations after an explicit rejection.
            return
        except Exception:
            allocation["state"] = "UNKNOWN"
            atomic_json(root / "campaign.json", manifest)
            raise
        atomic_json(Path(allocation["receipt_path"]), receipt)
        allocation.update(receipt, state="SUBMITTED")
        atomic_json(root / "campaign.json", manifest)
        outstanding += 1


def controller(root: Path, *, poll_seconds: float = 30, once: bool = False) -> Record:
    root = root.resolve()
    with (root / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("a controller already owns this campaign") from None
        manifest = read_json(root / "campaign.json")
        verify_snapshot(manifest)
        if not os.environ.get("account", "").strip():
            raise ValueError("submission requires the lowercase account environment variable")
        for executable in ("sbatch", "srun", "scontrol", "squeue", "sacct"):
            if shutil.which(executable) is None:
                raise ValueError(f"missing scheduler executable: {executable}")
        first_config = load_config(manifest["cases"][0]["config"])
        assert isinstance(first_config.launcher, SlurmLauncher)
        limit = Backend().partition_limit(first_config.launcher.partition)
        if limit is not None and limit < 30:
            raise ValueError("partition MaxTime is smaller than the 30-minute allocation")
        atomic_json(
            root / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "birth": process_identity(os.getpid()),
                "script": str(Path(__file__).resolve()),
                "script_sha256": file_digest(Path(__file__)),
            },
        )
        while True:
            try:
                reconcile(manifest)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                manifest.update(state="SCHEDULER_UNAVAILABLE", scheduler_error=str(error))
                atomic_json(root / "campaign.json", manifest)
                if once:
                    return summarize(manifest)
                time.sleep(poll_seconds)
                continue
            manifest.pop("scheduler_error", None)
            summary = summarize(manifest)
            if summary["pilot_blocked"]:
                manifest["state"] = "PILOT_FAILED"
                submit_ready(root, manifest, summary)
            elif all(a["state"] in TERMINAL for a in manifest["allocations"]):
                manifest["state"] = "COMPLETED" if summary["complete"] else "FINISHED_WITH_FAILURES"
            else:
                manifest["state"] = "RUNNING" if summary["pilot_gate_satisfied"] else "PILOT"
                submit_ready(root, manifest, summary)
            atomic_json(root / "campaign.json", manifest)
            result = summarize(manifest)
            atomic_json(root / "status.json", result)
            active = any(a["state"] in {"SUBMITTING", "SUBMITTED", "UNKNOWN"} | ACTIVE for a in manifest["allocations"])
            if once or (not active and manifest["state"] in {"PILOT_FAILED", "COMPLETED", "FINISHED_WITH_FAILURES"}):
                if not once:
                    write_report(root, manifest)
                return result
            time.sleep(poll_seconds)


def detach(root: Path, poll_seconds: float, *, controller_script: Path | None = None) -> Record:
    root = root.resolve()
    process_file = root / "controller-process.json"
    if process_file.exists():
        previous = read_json(process_file)
        if previous.get("host") == socket.gethostname() and previous.get("birth") == process_identity(previous["pid"]):
            if previous.get("birth") is not None:
                return {**previous, "already_running": True, "log": str(root / "controller.log")}
    with (root / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("a controller already owns this campaign") from None
        manifest = read_json(root / "campaign.json")
        verify_snapshot(manifest)
        revisions = manifest.setdefault("controller_revisions", [])
        script = (
            Path(revisions[-1]["path"])
            if revisions
            else Path(manifest["source"]) / "scripts/benchmark_training_campaign.py"
        )
        if controller_script is not None:
            controller_script = controller_script.resolve()
            digest = file_digest(controller_script)
            style = controller_script.with_name("report_style.py")
            style_digest = file_digest(style) if style.exists() else None
            revision_digest = hashlib.sha256((digest + style_digest).encode()).hexdigest() if style_digest else digest
            script = root / "controllers" / revision_digest[:16] / "benchmark_training_campaign.py"
            if script.exists() and file_digest(script) != digest:
                raise ValueError("frozen controller revision changed")
            if not script.exists():
                script.parent.mkdir(parents=True)
                shutil.copy2(controller_script, script)
            if style_digest:
                frozen_style = script.with_name("report_style.py")
                if frozen_style.exists() and file_digest(frozen_style) != style_digest:
                    raise ValueError("frozen report style changed")
                if not frozen_style.exists():
                    shutil.copy2(style, frozen_style)
                manifest["launch_files"][str(frozen_style)] = style_digest
            if not revisions or revisions[-1]["path"] != str(script):
                revisions.append(
                    {
                        "path": str(script),
                        "sha256": digest,
                        "created": time.time(),
                        "training_source_sha256": manifest["source_sha256"],
                    }
                )
            manifest["launch_files"][str(script)] = digest
            atomic_json(root / "campaign.json", manifest)
    env = {**os.environ, "PYTHONPATH": str(Path(manifest["source"]) / "src"), "PYTHONNOUSERSITE": "1"}
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), "run", "--root", str(root), "--poll-seconds", str(poll_seconds)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    record = {
        "pid": process.pid,
        "host": socket.gethostname(),
        "birth": process_identity(process.pid),
        "log": str(root / "controller.log"),
        "script": str(script),
        "script_sha256": file_digest(script),
    }
    atomic_json(process_file, record)
    return record


def plots(root: Path, rows: list[Record]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    if __package__:
        from scripts.report_style import METHOD_ORDER
    else:
        from report_style import METHOD_ORDER

    batches = sorted({row["global_batch_size"] for row in rows})
    models = sorted({row["model"] for row in rows}, key=lambda name: int(name[:-1]))
    contexts = sorted({row["context_length"] for row in rows})
    method_keys = {"sync": "sync", "cmp": "complete", "exp": "exponential", "ring": "ring"}
    labels = [(method_keys[method], gamma) for gamma in (0.0, 1.0) for method in METHOD_ORDER]
    method_names = {"sync": "Sync", "complete": "AWC complete", "ring": "AWC ring", "exponential": "AWC exponential"}

    def display_status(row: Record) -> str:
        if row["status"] == "pending" and row.get("job_id") is None:
            return "not run"
        return {"complete": "valid", "oom": "OOM"}.get(row["status"], row["status"])

    counts = Counter(display_status(row) for row in rows)
    count_text = " · ".join(f"{count} {status}" for status, count in counts.items())
    notes = []
    if "not run" in counts:
        notes.append("Not run: not submitted")
    if "OOM" in counts:
        notes.append("OOM: no valid steady-state measurement")
    cmap = plt.get_cmap("viridis").with_extremes(bad="#F2F3F5")
    fields = (
        ("global_tokens_per_second", 1e6, "Global throughput (million tokens/s)", "throughput"),
        ("max_peak_allocated_bytes", 1024**3, "Largest GPU training peak allocated (GiB)", "memory-allocated"),
        (
            "max_peak_reserved_bytes",
            1024**3,
            "Reserved GPU peak, including retained startup cache (GiB)",
            "memory-reserved",
        ),
    )
    for field, divisor, title, name in fields:
        figure, axes = plt.subplots(
            len(models), len(contexts), figsize=(14, 4 * len(models)), constrained_layout=True, squeeze=False
        )
        for mi, model in enumerate(models):
            for ci, context in enumerate(contexts):
                axis = axes[mi, ci]
                values = np.full((8, len(batches)), np.nan)
                states = {}
                for row in rows:
                    if row["model"] != model or row["context_length"] != context:
                        continue
                    index = labels.index((row["variant"], row["error_feedback_gamma"]))
                    column = batches.index(row["global_batch_size"])
                    states[index, column] = display_status(row)
                    if row["status"] == "complete":
                        values[index, column] = row[field] / divisor
                image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap)
                for i in range(8):
                    for j in range(len(batches)):
                        if math.isfinite(values[i, j]):
                            label = f"{values[i, j]:.2f}"
                            red, green, blue, _ = image.cmap(image.norm(values[i, j]))
                            color = "black" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5 else "white"
                        else:
                            label, color = states.get((i, j), "not run"), "#555555"
                            if label not in {"not run", "pending", "running", "unknown"}:
                                axis.add_patch(
                                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#FBE6E2", edgecolor="none")
                                )
                                color = "#9C2F26"
                        axis.text(
                            j,
                            i,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                        )
                axis.set_title(f"{model.upper()}, context {context}")
                axis.set_xticks(range(len(batches)), batches)
                axis.set_xlabel("Global batch (sequences)")
                axis.set_yticks(range(8), [method_names[v] + (" + EF" if g else "") for v, g in labels])
                if np.isfinite(values).any():
                    figure.colorbar(image, ax=axis, shrink=0.75)
        heading = title + " · 8 GPUs across 2 nodes\n" + count_text
        if notes:
            heading += "\n" + "; ".join(notes)
        figure.suptitle(heading, fontsize=12)
        figure.savefig(root / f"{name}.png", dpi=160)
        figure.savefig(root / f"{name}.pdf")
        plt.close(figure)


def write_report(root: Path, manifest: Record) -> Record:
    result = summarize(manifest)
    result["measurement_scope"] = (
        "Resident C4 loading/H2D, forward/backward, clipping, optimizer and communication; "
        "after warmup; no evaluation/checkpoints/profiling. CUDA allocated/reserved peaks are "
        "PyTorch allocator statistics, not total device usage. Three windows are not independent jobs."
    )
    atomic_json(root / "report.json", result)
    rows = result["rows"]
    columns = list(dict.fromkeys(key for row in rows for key in row if key != "windows"))
    with (root / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows
        )
    plots(root, rows)
    return result


def use_frozen_source(root: Path) -> None:
    """Use the campaign's schema and launch code even if the checkout has advanced."""
    path = root / "campaign.json"
    if not path.exists():
        return
    source = Path(read_json(path)["source"]) / "src"
    imported = Path(sys.modules[Config.__module__].__file__ or "")
    if imported.resolve() != (source / "decentralized_llm/config.py").resolve():
        env = {**os.environ, "PYTHONPATH": str(source), "PYTHONNOUSERSITE": "1"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "report", "retry", "extend", "accept-oom"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--once", action="store_true", help="run one controller reconciliation/submission cycle")
    parser.add_argument("--batch-size", type=int, action="append", default=[], help="extend: new global batch size")
    parser.add_argument("--model", choices=MODEL_PRESETS, action="append", default=[], help="extend: new model preset")
    parser.add_argument("--case", action="append", default=[], help="case for retry or accept-oom; repeat as needed")
    parser.add_argument("--reason", help="accept-oom: record the explicit authorization to continue past these OOMs")
    parser.add_argument(
        "--max-chunks-factor",
        type=int,
        default=1,
        help="retry only: multiply each selected case's iteration limit after resident-memory preflight",
    )
    parser.add_argument(
        "--controller-script",
        type=Path,
        help="freeze this supervision script for run --detach, preserving the frozen training source",
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.action == "extend":
        if bool(args.batch_size) == bool(args.model):
            parser.error("extend requires either --batch-size or --model, not both")
    elif args.batch_size or args.model:
        parser.error("--batch-size and --model are only valid for extend")
    if args.action == "accept-oom":
        if not args.case or not args.reason or not args.reason.strip():
            parser.error("accept-oom requires --case and --reason")
    elif args.reason is not None:
        parser.error("--reason is only valid for accept-oom")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.max_chunks_factor < 1 or (args.max_chunks_factor != 1 and args.action != "retry"):
        parser.error("--max-chunks-factor must be positive and requires retry")
    if args.controller_script is not None and (args.action != "run" or not args.detach):
        parser.error("--controller-script requires run --detach")
    root = args.root.resolve()
    use_frozen_source(root)
    if args.action in {"prepare", "run"} and not (root / "campaign.json").exists():
        cache = args.cache or Path("data/c4")
        prepared = prepare(root, cache, dry_run=args.dry_run)
        if args.dry_run or args.action == "prepare":
            print(json.dumps(prepared.get("plan", prepared), indent=2))
            return
    manifest = read_json(root / "campaign.json")
    if args.cache is not None and str(args.cache.resolve()) != manifest["cache"]:
        parser.error("campaign already uses a different token cache")
    verify_snapshot(manifest)
    if args.action == "extend":
        result = (
            extend_models(root, args.model, dry_run=args.dry_run)
            if args.model
            else extend_batches(root, args.batch_size, dry_run=args.dry_run)
        )
    elif args.action == "retry":
        result = retry_cases(root, args.case, dry_run=args.dry_run, max_chunks_factor=args.max_chunks_factor)
    elif args.action == "accept-oom":
        assert isinstance(args.reason, str)
        result = accept_pilot_ooms(root, args.case, reason=args.reason, dry_run=args.dry_run)
    elif args.dry_run:
        result = {**manifest["plan"], "state": manifest["state"], "counts": summarize(manifest)["counts"]}
    elif args.action == "prepare":
        result = manifest["plan"]
    elif args.action == "run":
        result = (
            detach(root, args.poll_seconds, controller_script=args.controller_script)
            if args.detach
            else controller(root, poll_seconds=args.poll_seconds, once=args.once)
        )
    elif args.action == "status":
        # Read-only: status never competes with the controller's mutable state.
        snapshot = json.loads(json.dumps(manifest))
        try:
            reconcile(snapshot, persist_receipts=False)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            snapshot["scheduler_error"] = str(error)
        result = summarize(snapshot)
        if "scheduler_error" in snapshot:
            result["scheduler_error"] = snapshot["scheduler_error"]
    else:
        result = write_report(root, manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
