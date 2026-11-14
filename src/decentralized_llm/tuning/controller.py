"""Durable phase orchestration, trial validation, and revision replay."""

import copy
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from decentralized_llm.config import Config, initialize_local_defaults
from decentralized_llm.data.cache import TokenCache, training_boundaries
from decentralized_llm.launch.preflight import validate
from decentralized_llm.launch.slurm import SubmissionRejected
from decentralized_llm.runtime import atomic_json, dependency_digest, source_digest
from decentralized_llm.training.logs import read_config, read_failures, read_metadata, read_result
from decentralized_llm.tuning import local_gpus
from decentralized_llm.tuning.batching import (
    check_walltime,
    effective_config,
    exhausted_reason,
    group_trials,
    trial_minutes,
)
from decentralized_llm.tuning.estimate import estimate, ranking
from decentralized_llm.tuning.jobs import ACTIVE, FAILED, Backend
from decentralized_llm.tuning.outcomes import select_finite
from decentralized_llm.tuning.report import write_report
from decentralized_llm.tuning.spec import Record, TuningSpec, expand, numeric_recipe, read_spec, search_points, trial_id

VERSION = 3


def provenance(cache: TokenCache) -> Record:
    return {
        "cache_identity": cache.manifest["identity"],
        "source_sha256": source_digest(Path(__file__).resolve().parents[1]),
        "lock_sha256": dependency_digest(),
        "python_version": list(sys.version_info[:2]),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


@contextmanager
def study_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".controller.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another controller is already using this tuning directory") from error
        yield


def load_state(directory: Path) -> Record:
    state = json.loads((directory / "state.json").read_text())
    if state.get("version") not in {1, 2, VERSION}:
        raise ValueError("unsupported tuning state version")
    state["version"] = VERSION
    state.setdefault("allocations", {})
    return state


def save_state(directory: Path, state: Record) -> None:
    state["updated"] = time.time()
    atomic_json(directory / "state.json", state)


def check_run(trial: Record, attempt: Record) -> Config:
    """Authenticate the run independently of whether it reached full validation."""
    output = Path(attempt["directory"]) / "run"
    run = read_metadata(output)
    actual = read_config(output)
    intended = Config.model_validate(trial["config"])
    if numeric_recipe(actual) != numeric_recipe(intended) or actual.runtime.seed != trial["seed"]:
        raise ValueError("result configuration does not match the trial")
    expected = trial["provenance"]
    if run["cache"] != expected["cache_identity"]:
        raise ValueError("result cache identity mismatch")
    for key in ("source_sha256", "lock_sha256", "python_version", "torch", "numpy"):
        if run["environment"][key] != expected[key]:
            raise ValueError(f"result provenance mismatch: {key}")
    return intended


def check_result(trial: Record, attempt: Record) -> Record:
    output = Path(attempt["directory"]) / "run"
    result = read_result(output)
    intended = check_run(trial, attempt)
    if result["cursor"] != training_boundaries(intended, intended.model.parameter_count)[-1]:
        raise ValueError("result did not complete its training budget")
    loss = result["validation"]["loss"]
    if type(loss) not in (int, float) or not math.isfinite(loss) or result["validation"]["split"] != "full":
        raise ValueError("trial requires a finite final full-validation loss")
    return result


def check_divergence(trial: Record, attempt: Record) -> Record | None:
    """Accept only a structured numerical failure from an authenticated worker."""
    output = Path(attempt["directory"]) / "run"
    intended = Config.model_validate(trial["config"])
    records = []
    for record in read_failures(output):
        if record.get("kind") != "nonfinite_update":
            continue
        rank = record.get("rank")
        if type(rank) is not int or not 0 <= rank < intended.physical_world_size:
            raise ValueError("nonfinite worker rank mismatch")
        path = Path(record["file"])
        if path.name != "events.jsonl" and path.name != f"failure-rank-{rank:05d}.json":
            raise ValueError("nonfinite worker rank mismatch")
        if not all(isinstance(record.get(key), str) and record[key] for key in ("message", "traceback")):
            raise ValueError("nonfinite worker diagnostics are missing")
        progress = record.get("progress")
        if not isinstance(progress, dict) or any(
            type(progress.get(key)) is not int or progress[key] < 0 for key in ("cursor", "step", "completed_epochs")
        ):
            raise ValueError("invalid nonfinite worker progress")
        total = training_boundaries(intended, intended.model.parameter_count)[-1]
        if progress["cursor"] > total or progress["completed_epochs"] > intended.training.epoch_count:
            raise ValueError("nonfinite worker progress exceeds training budget")
        records.append(record)
    if not records:
        return None
    check_run(trial, attempt)
    return records[0]


def finish_failed_trial(trial: Record, attempt: Record, reason: str, *, allow_divergence: bool) -> None:
    if allow_divergence:
        try:
            record = check_divergence(trial, attempt)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason += f"; invalid divergence evidence: {error}"
        else:
            if record is not None:
                attempt.update(status="diverged", error=record["message"], finished=time.time())
                trial.update(status="diverged", loss="inf", divergence=record)
                return
    fail_trial(trial, attempt, reason)


def fail_trial(trial: Record, attempt: Record, reason: str) -> None:
    attempt.update(status="failed", error=reason, finished=time.time())
    trial["status"] = "failed"


def reconcile(state: Record, backend: Backend, grace: float) -> bool:
    active = [
        trial
        for trial in state["trials"].values()
        if trial["status"] in {"submitting", "submitted"} and "allocation_id" not in trial["attempts"][-1]
    ]
    allocations = [
        allocation
        for allocation in state.get("allocations", {}).values()
        if allocation["status"] in {"submitting", "submitted"}
    ]
    try:
        statuses = backend.poll([trial["attempts"][-1] for trial in active] + allocations)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        state["warning"] = str(error)
        return False
    state.pop("warning", None)
    certain = reconcile_allocations(state, statuses, grace)
    for trial in active:
        attempt = trial["attempts"][-1]
        status = statuses.get(attempt["name"], {"state": "UNKNOWN", "exit_code": None})
        attempt.update({key: status[key] for key in ("job_id", "cluster") if key in status})
        condition = status["state"]
        attempt["scheduler_state"] = condition
        if condition in ACTIVE:
            attempt["status"] = trial["status"] = "submitted"
        elif condition in FAILED or (
            condition == "COMPLETED" and status["exit_code"] is not None and status["exit_code"] != "0:0"
        ):
            finish_failed_trial(
                trial,
                attempt,
                f"job ended in {condition}, exit={status['exit_code']}",
                allow_divergence=condition in {"FAILED", "COMPLETED"} and status["exit_code"] == "1:0",
            )
        elif condition == "COMPLETED" and status["exit_code"] == "0:0":
            attempt.setdefault("terminal_seen", time.time())
            try:
                result = check_result(trial, attempt)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if time.time() - attempt["terminal_seen"] >= grace:
                    fail_trial(trial, attempt, f"invalid or missing result: {error}")
            else:
                attempt.update(status="succeeded", finished=time.time())
                trial.update(status="succeeded", loss=result["validation"]["loss"], result=result)
        else:
            certain = False
            state["warning"] = (
                f"Unresolved attempt {attempt['name']}; retaining its slot without resubmission. "
                "Check its scheduler receipt or local process on the original host. After confirming no job exists, "
                f"resume with --retry-unresolved {attempt['name']}."
            )
    return certain


def reconcile_allocations(state: Record, statuses: dict[str, Record], grace: float) -> bool:
    certain = True
    for allocation in state.get("allocations", {}).values():
        if allocation["status"] in {"planned", "abandoned", "rejected"}:
            continue
        if allocation["status"] in {"submitting", "submitted"}:
            status = statuses.get(allocation["name"], {"state": "UNKNOWN", "exit_code": None})
            allocation.update({key: status[key] for key in ("job_id", "cluster") if key in status})
            condition = status["state"]
            allocation.update(scheduler_state=condition, exit_code=status["exit_code"])
            if condition in ACTIVE:
                allocation["status"] = "submitted"
            elif condition in FAILED or condition == "COMPLETED" and status["exit_code"] is not None:
                allocation.update(status="finished", finished=time.time(), terminal_seen=time.time())
            else:
                certain = False
                state["warning"] = (
                    f"Unresolved allocation {allocation['name']}; retaining it without resubmission. "
                    f"After confirming the entire allocation has no live job, resume with "
                    f"--retry-unresolved {allocation['name']}."
                )
        for member in allocation["members"]:
            trial = state["trials"][member["trial_id"]]
            attempt = trial["attempts"][member["number"] - 1]
            if attempt["status"] not in {"submitting", "submitted"}:
                continue
            attempt.update({key: allocation[key] for key in ("job_id", "cluster") if key in allocation})
            attempt["scheduler_state"] = allocation.get("scheduler_state", "UNKNOWN")
            try:
                reconcile_step(trial, attempt, allocation, grace)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if allocation["status"] == "finished" and time.time() - allocation["terminal_seen"] >= grace:
                    # An unreadable start record cannot prove that a trial never ran.
                    attempt["started"] = True
                    fail_trial(trial, attempt, f"invalid step execution record: {error}")
                else:
                    certain = False
                    state["warning"] = f"Cannot reconcile allocation {allocation['name']}: {error}"
    return certain


def reconcile_step(trial: Record, attempt: Record, allocation: Record, grace: float) -> None:
    path = Path(attempt["execution_record"])
    record = None
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        pass
    if record is not None:
        identity = {"allocation_id": allocation["id"], "trial_id": trial["id"], "attempt": attempt["number"]}
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("step execution record identity mismatch")
        if type(record["started"]) is not bool or record["state"] not in {"RUNNING", "COMPLETED", "FAILED", "SKIPPED"}:
            raise ValueError("invalid step execution state")
        # A final launch-error record can establish that Popen failed after the
        # runner published its start intent. That attempt never consumed training.
        attempt["started"] = record["started"]
        attempt["step_state"] = record["state"]
        if record.get("started_at") is not None:
            attempt["started_at"] = record["started_at"]
        if record["state"] == "COMPLETED" and record["started"] and record.get("exit_code") == 0:
            attempt.setdefault("terminal_seen", time.time())
            try:
                result = check_result(trial, attempt)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if time.time() - attempt["terminal_seen"] >= grace:
                    fail_trial(trial, attempt, f"invalid or missing result: {error}")
            else:
                attempt.update(status="succeeded", finished=record.get("finished", time.time()))
                trial.update(status="succeeded", loss=result["validation"]["loss"], result=result)
            return
        if record["state"] == "FAILED":
            # A finished step keeps its own outcome if a later step or parent fails.
            finish_failed_trial(
                trial,
                attempt,
                f"step failed, exit={record.get('exit_code')}: {record.get('reason', '')}",
                allow_divergence=record["started"]
                and type(record.get("exit_code")) is int
                and record.get("exit_code") == 1,
            )
            return
    # A queued/unknown/requeued parent still owns all of its unstarted members.
    if allocation["status"] == "finished" and time.time() - allocation["terminal_seen"] >= grace:
        reason = record.get("reason", "step did not finish") if record else "no step execution record"
        fail_trial(trial, attempt, f"allocation ended in {allocation['scheduler_state']}: {reason}")


def new_display_id(identity: str, trials: Record) -> str:
    """Assign an immutable short alias without changing scientific identities."""
    occupied = {trial.get("display_id", trial["id"]) for trial in trials.values()}
    for length in range(12, len(identity) + 1, 4):
        alias = identity[:length]
        if alias not in occupied:
            return alias
    raise ValueError("trial identity already has a display alias")


def add_phase(state: Record, spec: TuningSpec, base: Config) -> Record:
    index = len(state["phases"])
    parents = None
    if index:
        previous = state["phases"][-1]
        parents = [row for row in previous["rankings"] if row["id"] in previous["selected"]]
    confirmation = index == len(spec.phases)
    if confirmation:
        assert parents is not None
        candidates, name, top_k = parents, "confirmation", 1
        definition = spec.confirmation.model_dump(mode="json")
    else:
        phase = spec.phases[index]
        candidates, name, top_k = expand(base, phase, parents), phase.name, phase.top_k
        definition = phase.model_dump(mode="json")
    record: Record = {
        "name": name,
        "candidates": candidates,
        "top_k": top_k,
        "definition": definition,
        "complete": False,
        "started": time.time(),
        "trials": {},
        "seeds": spec.seeds(confirmation),
        "new_trials": 0,
        "reused_trials": 0,
    }
    for candidate in candidates:
        identifiers = []
        for seed in record["seeds"]:
            key = trial_id(candidate, seed, state["provenance"])
            identifiers.append(key)
            if key in state["trials"]:
                record["reused_trials"] += 1
                continue
            config = copy.deepcopy(candidate["config"])
            config["runtime"]["seed"] = seed
            display_id = new_display_id(key, state["trials"])
            state["trials"][key] = {
                "id": key,
                "display_id": display_id,
                "directory": str(Path("trials") / name / display_id),
                "candidate_id": candidate["id"],
                "seed": seed,
                "config": config,
                "provenance": state["provenance"],
                "status": "pending",
                "attempts": [],
            }
            record["new_trials"] += 1
        record["trials"][candidate["id"]] = identifiers
    state["phases"].append(record)
    return record


def trial_directory(directory: Path, trial: Record) -> Path:
    if "directory" not in trial:
        # Existing attempts keep their physical home, including legacy flat layouts.
        root = (
            Path(trial["attempts"][0]["directory"]).parent if trial["attempts"] else directory / "trials" / trial["id"]
        )
        trial["directory"] = os.path.relpath(root, directory)
    return directory / trial["directory"]


def materialize_phase(directory: Path, state: Record, phase: Record) -> None:
    """Build phase views without moving artifacts or linking through another view."""
    view = directory / "trials" / phase["name"]
    view.mkdir(parents=True, exist_ok=True)
    for identifiers in phase["trials"].values():
        for key in identifiers:
            trial = state["trials"][key]
            if "directory" not in trial and not trial["attempts"]:
                trial["directory"] = str(Path("trials") / phase["name"] / key)
            root = trial_directory(directory, trial).resolve()
            root.mkdir(parents=True, exist_ok=True)
            link = view / trial.get("display_id", key)
            if link == root:
                continue
            if link.is_symlink():
                if link.resolve() != root:
                    raise ValueError(f"phase trial link points to a different directory: {link}")
            elif link.exists():
                raise ValueError(f"phase trial path already exists: {link}")
            else:
                link.symlink_to(os.path.relpath(root, view), target_is_directory=True)


def advance(directory: Path, state: Record, spec: TuningSpec, base: Config) -> list[Record]:
    while True:
        phase = state["phases"][-1] if state["phases"] else add_phase(state, spec, base)
        materialize_phase(directory, state, phase)
        if phase["complete"]:
            if not phase["selected"]:
                state.update(
                    status="complete", winner=None, reason=f"No finite-loss candidates remain in phase {phase['name']}."
                )
                return []
            if phase["name"] == "confirmation":
                state.update(status="complete", winner=select_finite(phase["rankings"], 1)[0], reason=None)
                return []
            phase = add_phase(state, spec, base)
            materialize_phase(directory, state, phase)
        ranks = ranking(phase["candidates"], phase["seeds"], state["trials"], state["provenance"])
        if ranks is not None:
            phase.update(
                complete=True,
                finished=time.time(),
                rankings=ranks,
                selected=[row["id"] for row in select_finite(ranks, phase["top_k"])],
            )
            save_state(directory, state)
            write_report(directory, state)
            logger.info("Selected {} combinations in phase {}", len(phase["selected"]), phase["name"])
            continue
        required = [state["trials"][key] for keys in phase["trials"].values() for key in keys]
        exhausted = [reason for trial in required if (reason := exhausted_reason(trial, spec.controller.max_retries))]
        if exhausted:
            state.update(status="failed", reason=exhausted[0])
            return []
        return [
            trial
            for trial in required
            if trial["status"] in {"pending", "failed"}
            and not (
                trial["attempts"]
                and state.get("allocations", {}).get(trial["attempts"][-1].get("allocation_id"), {}).get("status")
                in {"submitting", "submitted"}
            )
        ]


def prepare_attempt(directory: Path, state: Record, trial: Record) -> tuple[Config, Record]:
    number = len(trial["attempts"]) + 1
    destination = trial_directory(directory, trial) / f"attempt-{number:03d}"
    config = effective_config(trial, Config.model_validate(state["base"]))
    config.runtime.output_dir = destination / "run"
    # Fail deterministic configuration errors before publishing a submission intent.
    validate(config, launch=True, verify=False)
    attempt: Record = {
        "number": number,
        "directory": str(destination),
        "backend": config.launcher.kind,
        "name": f"tune-{state['id'][:12]}-{trial['id'][:20]}-{number}",
        "status": "submitting",
        "created": time.time(),
    }
    return config, attempt


def submit(
    directory: Path, state: Record, trial: Record, backend: Backend, *, gpu_ids: list[str] | None = None
) -> None:
    config, attempt = prepare_attempt(directory, state, trial)
    if gpu_ids is not None:
        # The durable intent owns its GPUs even if the controller dies during spawn.
        attempt.update(local_gpu_ids=gpu_ids, host=state["local_gpu_pool"]["host"])
    destination = Path(attempt["directory"])
    trial["attempts"].append(attempt)
    trial["status"] = "submitting"
    save_state(directory, state)
    try:
        destination.mkdir(parents=True)
        receipt = backend.submit(config, attempt)
    except SubmissionRejected as error:
        fail_trial(trial, attempt, str(error))
    else:
        attempt.update({key: receipt[key] for key in ("job_id", "cluster", "pid", "host", "birth") if key in receipt})
        attempt["status"] = trial["status"] = "submitted"
    finally:
        save_state(directory, state)


def plan_allocations(directory: Path, state: Record, spec: TuningSpec, base: Config, pending: list[Record]) -> None:
    allocations = state.setdefault("allocations", {})
    planned = {key for item in allocations.values() if item["status"] == "planned" for key in item["trial_ids"]}
    groups = group_trials(
        [trial for trial in pending if trial["id"] not in planned], base, spec.controller.jobs_per_allocation
    )
    phase = state["phases"][-1]["name"]
    for group in groups:
        identifier = uuid.uuid4().hex
        root = directory / "slurm" / phase / identifier
        allocations[identifier] = {
            "id": identifier,
            "name": f"tune-{state['id'][:12]}-allocation-{identifier[:20]}",
            "backend": "slurm",
            "phase": phase,
            "revision": state["revision"],
            "directory": str(root),
            "receipt_path": str(root / "receipt.json"),
            "stdout": str(root / "slurm-%j.out"),
            "stderr": str(root / "slurm-%j.err"),
            "status": "planned",
            "created": time.time(),
            "trial_ids": [trial["id"] for trial in group],
            "time_minutes": sum(trial_minutes(effective_config(trial, base)) for trial in group),
            "members": [],
        }


def submit_allocation(directory: Path, state: Record, allocation: Record, backend: Backend) -> None:
    trials = [state["trials"][key] for key in allocation["trial_ids"]]
    prepared = [prepare_attempt(directory, state, trial) for trial in trials]
    configs, attempts = [item[0] for item in prepared], [item[1] for item in prepared]
    for trial, attempt in zip(trials, attempts, strict=True):
        label = trial.get("display_id", trial["id"])
        root = Path(allocation["directory"]) / "steps" / f"{label}-attempt-{attempt['number']:03d}"
        attempt.update(
            allocation_id=allocation["id"],
            trial_id=trial["id"],
            started=False,
            stdout=str(root.with_suffix(".out")),
            stderr=str(root.with_suffix(".err")),
            execution_record=str(root.with_suffix(".json")),
        )
        trial["attempts"].append(attempt)
        trial["status"] = "submitting"
    allocation.update(
        status="submitting",
        created=time.time(),
        members=[
            {"trial_id": trial["id"], "number": attempt["number"]}
            for trial, attempt in zip(trials, attempts, strict=True)
        ],
    )
    save_state(directory, state)
    try:
        receipt = backend.submit_allocation(configs, allocation, attempts)
    except SubmissionRejected as error:
        allocation.update(status="rejected", error=str(error), finished=time.time())
        for trial, attempt in zip(trials, attempts, strict=True):
            fail_trial(trial, attempt, str(error))
    else:
        allocation.update({key: receipt[key] for key in ("job_id", "cluster") if key in receipt})
        allocation["status"] = "submitted"
        for trial, attempt in zip(trials, attempts, strict=True):
            attempt.update({key: receipt[key] for key in ("job_id", "cluster") if key in receipt})
            attempt["status"] = trial["status"] = "submitted"
    finally:
        save_state(directory, state)


def outstanding_allocations(state: Record) -> int:
    parents = sum(item["status"] in {"submitting", "submitted"} for item in state.get("allocations", {}).values())
    legacy = sum(
        trial["status"] in {"submitting", "submitted"} and "allocation_id" not in trial["attempts"][-1]
        for trial in state["trials"].values()
    )
    return parents + legacy


def active_local_attempts(state: Record) -> list[Record]:
    return [
        trial["attempts"][-1]
        for trial in state["trials"].values()
        if trial["status"] in {"submitting", "submitted"} and trial["attempts"][-1]["backend"] == "local"
    ]


def resolve_local_gpu_pool(spec: TuningSpec, base: Config, existing: Record | None, action: str) -> Record | None:
    """Select visible devices for new revisions; resume uses the saved UUIDs."""
    if base.launcher.kind != "local":
        if spec.controller.local_gpus is not None:
            raise ValueError("controller.local_gpus requires launcher.kind=local")
        return None
    devices = {base.runtime.device}
    uses_cuda = False
    for phase in spec.phases:
        if "runtime.device" in phase.search:
            devices = {
                point["value"] for point in search_points(base, "runtime.device", phase.search["runtime.device"])
            }
        elif "runtime.device" in phase.initial_values:
            devices = {phase.initial_values["runtime.device"]}
        uses_cuda |= "cuda" in devices
    if not uses_cuda:
        if spec.controller.local_gpus is not None:
            raise ValueError("controller.local_gpus requires CUDA trials")
        return None
    saved = existing.get("local_gpu_pool") if existing else None
    if action == "resume" and saved is not None:
        pool = local_gpus.restore_pool(saved, local_gpus.discover_gpus())
    else:
        pool = local_gpus.discover_gpus(spec.controller.local_gpus)
    if existing:
        # Also rejects live attempts from another host and overlapping reservations.
        local_gpus.available_gpus(pool, active_local_attempts(existing))
    return pool


def validate_local_capacity(config: Config, pool: Record | None) -> None:
    if config.launcher.kind == "local" and config.runtime.device == "cuda":
        capacity = len(pool["gpu_ids"]) if pool is not None else 0
        if config.physical_gpu_count > capacity:
            raise ValueError(
                f"local trial requires {config.physical_gpu_count} GPUs, "
                f"but the selected local GPU pool has {capacity}; "
                "select more GPUs with controller.local_gpus / CUDA_VISIBLE_DEVICES or reduce the trial GPU count"
            )


def submit_local(
    directory: Path, state: Record, spec: TuningSpec, base: Config, pending: list[Record], backend: Backend
) -> None:
    pool = state.get("local_gpu_pool")
    # Check the whole phase before launching any of it, including later search phases.
    for trial in pending:
        validate_local_capacity(effective_config(trial, base), pool)
    for trial in pending:
        active = active_local_attempts(state)
        if outstanding_allocations(state) >= spec.controller.max_jobs:
            break
        config = effective_config(trial, base)
        if config.runtime.device == "cpu":
            # CPU trials retain their sequential behavior, including mixed-device searches.
            if not outstanding_allocations(state):
                submit(directory, state, trial, backend)
            break
        assert pool is not None
        free = local_gpus.available_gpus(pool, active)
        if len(free) < config.physical_gpu_count:
            continue
        submit(directory, state, trial, backend, gpu_ids=free[: config.physical_gpu_count])
        if exhausted_reason(trial, spec.controller.max_retries):
            break


def validate_cached(state: Record) -> None:
    for trial in state["trials"].values():
        if trial["status"] in {"succeeded", "diverged"}:
            try:
                if trial["status"] == "diverged":
                    if trial.get("loss") != "inf" or check_divergence(trial, trial["attempts"][-1]) is None:
                        raise ValueError("cached divergence has no valid nonfinite evidence")
                else:
                    check_result(trial, trial["attempts"][-1])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"cached trial {trial['id']} no longer has a valid result: {error}") from error


def run_loop(directory: Path, state: Record, spec: TuningSpec, base: Config, backend: Backend) -> Record:
    state.update(status="running", reason=None)
    checked_phase = None
    partition_limit = None
    try:
        save_state(directory, state)
        write_report(directory, state)
        while state["status"] == "running":
            certain = reconcile(state, backend, spec.controller.result_grace_seconds)
            pending = advance(directory, state, spec, base)
            if state["status"] != "running":
                if state["status"] == "complete" and outstanding_allocations(state):
                    state["status"] = "running"
                    save_state(directory, state)
                    time.sleep(spec.controller.poll_seconds)
                    continue
                break
            if certain:
                if base.launcher.kind == "slurm":
                    phase = state["phases"][-1]["name"]
                    if checked_phase != phase:
                        partition_limit = backend.partition_limit(base.launcher.partition)
                        checked_phase = phase
                    plan_allocations(directory, state, spec, base, pending)
                    planned = [item for item in state["allocations"].values() if item["status"] == "planned"]
                    for allocation in planned:
                        check_walltime(
                            allocation["time_minutes"], partition_limit, base.launcher.partition, trial_minutes(base)
                        )
                    save_state(directory, state)
                    for allocation in planned[: max(0, spec.controller.max_jobs - outstanding_allocations(state))]:
                        submit_allocation(directory, state, allocation, backend)
                        if any(
                            exhausted_reason(state["trials"][key], spec.controller.max_retries)
                            for key in allocation["trial_ids"]
                        ):
                            break
                elif pending:
                    submit_local(directory, state, spec, base, pending, backend)
            save_state(directory, state)
            logger.info(
                "Tuning revision {}: phase {}, {} outstanding allocations/processes{}",
                state["revision"],
                state["phases"][-1]["name"],
                outstanding_allocations(state),
                f"; {state['warning']}" if state.get("warning") else "",
            )
            time.sleep(spec.controller.poll_seconds)
    except KeyboardInterrupt:
        state.update(status="interrupted", reason="Controller interrupted; submitted work is retained for resume.")
    except Exception as error:
        state.update(status="failed", reason=str(error))
        raise
    finally:
        save_state(directory, state)
        write_report(directory, state)
    return {
        "directory": str(directory),
        "status": state["status"],
        "revision": state["revision"],
        "reason": state["reason"],
    }


def execute(
    action: str,
    directory: Path,
    spec_path: Path | None = None,
    *,
    configs: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    retry_unresolved: list[str] | None = None,
    use_local_defaults: bool = True,
) -> Record:
    if (configs or overrides) and (action not in ("run", "extend") or spec_path is None):
        raise ValueError("base config overrides require run or extend with a tuning specification")
    directory = directory.resolve()
    if action == "run" and (directory / "state.json").exists():
        raise ValueError("tuning directory already contains state; use resume or extend")
    if action == "status":
        state = load_state(directory)
        counts: Record = {}
        for trial in state["trials"].values():
            counts[trial["status"]] = counts.get(trial["status"], 0) + 1
        allocation_counts: Record = {}
        for item in state["allocations"].values():
            allocation_counts[item["status"]] = allocation_counts.get(item["status"], 0) + 1
        return {key: state.get(key) for key in ("revision", "status", "reason", "warning", "updated")} | {
            "trials": counts,
            "allocations": allocation_counts,
            "outstanding_allocations": outstanding_allocations(state),
            "allocation_details": list(state["allocations"].values()),
            "local_gpu_pool": state.get("local_gpu_pool"),
            "local_gpu_assignments": [
                {key: attempt.get(key) for key in ("name", "host", "local_gpu_ids", "status")}
                for attempt in active_local_attempts(state)
            ],
        }
    if action == "report":
        with study_lock(directory):
            return write_report(directory, load_state(directory))
    existing = None if action == "run" else load_state(directory)
    if spec_path is not None and action in ("run", "extend"):
        if use_local_defaults:
            initialize_local_defaults(dry_run=dry_run)
        spec, base = read_spec(
            spec_path.resolve(), configs=configs, overrides=overrides, use_local_defaults=use_local_defaults
        )
    else:
        if existing is None:
            raise ValueError("a tuning specification is required")
        spec, base = TuningSpec.model_validate(existing["spec"]), Config.model_validate(existing["base"])
    cache = TokenCache(base.data.cache_dir)
    cache.verify()
    current_provenance = provenance(cache)
    if action == "resume" and existing is not None and current_provenance != existing["provenance"]:
        raise ValueError("cache or software provenance changed; use extend to create a new tuning revision")
    if existing is not None:
        validate_cached(existing)
    pool = resolve_local_gpu_pool(spec, base, existing, action)
    # All first-phase configurations are validated even for a dry run.
    for candidate in expand(base, spec.phases[0]):
        config = Config.model_validate(candidate["config"])
        config.runtime.output_dir = directory / "preflight"
        validate_local_capacity(config, pool)
        validate(config, launch=True, verify=False)
    if base.launcher.kind == "slurm":
        for executable in ("squeue", "sacct"):
            if shutil.which(executable) is None:
                raise ValueError(f"SLURM monitoring prerequisite missing: {executable}")
    backend = Backend()
    if dry_run:
        result = estimate(
            spec,
            base,
            current_provenance,
            existing["trials"] if existing else None,
            existing.get("allocations") if existing else None,
        )
        if base.launcher.kind == "slurm":
            limit = backend.partition_limit(base.launcher.partition)
            result["partition_max_minutes"] = limit
            for phase in result["phases"]:
                for allocation in phase["allocations"]:
                    if phase["allocations_exact"]:
                        check_walltime(allocation["time_minutes"], limit, base.launcher.partition, trial_minutes(base))
        elif pool is not None:
            result.update(local_gpu_pool=pool, local_max_jobs=spec.controller.max_jobs)
        return result
    with study_lock(directory):
        if action == "run":
            if (directory / "state.json").exists():
                raise ValueError("tuning directory already contains state; use resume or extend")
            state: Record = {
                "version": VERSION,
                "id": uuid.uuid4().hex,
                "revision": 1,
                "created": time.time(),
                "spec": spec.model_dump(mode="json"),
                "base": base.model_dump(mode="json"),
                "provenance": current_provenance,
                "trials": {},
                "allocations": {},
                "phases": [],
                "status": "running",
                "winner": None,
                "reason": None,
            }
        else:
            state = load_state(directory)
            if existing is None or state["updated"] != existing["updated"]:
                raise ValueError("tuning changed during preflight; retry the command")
            if action == "extend":
                archive = directory / "revisions" / f"{state['revision']:04d}"
                archive.mkdir(parents=True, exist_ok=True)
                atomic_json(archive / "state.json", state)
                for filename in ("report.md", "report.json", "winner.yaml"):
                    if (directory / filename).exists():
                        shutil.copy2(directory / filename, archive / filename)
                if (directory / "plots").is_dir():
                    shutil.copytree(directory / "plots", archive / "plots", dirs_exist_ok=True)
                (directory / "winner.yaml").unlink(missing_ok=True)
                for allocation in state["allocations"].values():
                    if allocation["status"] == "planned":
                        allocation["status"] = "abandoned"
                state.update(
                    revision=state["revision"] + 1,
                    spec=spec.model_dump(mode="json"),
                    base=base.model_dump(mode="json"),
                    provenance=current_provenance,
                    phases=[],
                    winner=None,
                )
        state["local_gpu_pool"] = pool
        if retry_unresolved:
            reconcile(state, backend, spec.controller.result_grace_seconds)
            for name in retry_unresolved:
                allocation = next((item for item in state["allocations"].values() if item["name"] == name), None)
                matches = [
                    trial
                    for trial in state["trials"].values()
                    if trial["attempts"] and trial["attempts"][-1]["name"] == name
                ]
                if len(matches) == 1 and matches[0]["attempts"][-1].get("allocation_id"):
                    allocation = state["allocations"][matches[0]["attempts"][-1]["allocation_id"]]
                if allocation is not None:
                    if allocation.get("scheduler_state") != "UNKNOWN" or allocation["status"] not in {
                        "submitting",
                        "submitted",
                    }:
                        raise ValueError(f"allocation {name} is not unresolved")
                    allocation.update(
                        status="finished", scheduler_state="CANCELLED", terminal_seen=time.time(), finished=time.time()
                    )
                    for member in allocation["members"]:
                        trial = state["trials"][member["trial_id"]]
                        attempt = trial["attempts"][member["number"] - 1]
                        if attempt["status"] in {"submitting", "submitted"}:
                            reconcile_step(trial, attempt, allocation, 0)
                    continue
                if len(matches) != 1 or matches[0]["attempts"][-1].get("scheduler_state") != "UNKNOWN":
                    raise ValueError(f"attempt {name} is not unresolved")
                trial = matches[0]
                fail_trial(
                    trial, trial["attempts"][-1], "Operator confirmed that the unresolved attempt has no live job."
                )
        save_state(directory, state)
        return run_loop(directory, state, spec, base, backend)
