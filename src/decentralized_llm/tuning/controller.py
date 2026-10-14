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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from decentralized_llm.cache import TokenCache, training_boundaries
from decentralized_llm.config import Config, load_config
from decentralized_llm.launch import SubmissionRejected
from decentralized_llm.preflight import validate
from decentralized_llm.runtime import atomic_json, dependency_digest, source_digest
from decentralized_llm.tuning.backends import ACTIVE, FAILED, Backend
from decentralized_llm.tuning.estimate import estimate, ranking
from decentralized_llm.tuning.report import write_report
from decentralized_llm.tuning.spec import Record, TuningSpec, expand, numeric_recipe, read_spec, trial_id

VERSION = 1


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
    if state.get("version") != VERSION:
        raise ValueError("unsupported tuning state version")
    return state


def save_state(directory: Path, state: Record) -> None:
    state["updated"] = time.time()
    atomic_json(directory / "state.json", state)


def check_result(trial: Record, attempt: Record) -> Record:
    output = Path(attempt["directory"]) / "run"
    result = json.loads((output / "result.json").read_text())
    run = json.loads((output / "run.json").read_text())
    actual = load_config(output / "config.yaml")
    intended = Config.model_validate(trial["config"])
    if numeric_recipe(actual) != numeric_recipe(intended) or actual.runtime.seed != trial["seed"]:
        raise ValueError("result configuration does not match the trial")
    expected = trial["provenance"]
    if run["cache"] != expected["cache_identity"]:
        raise ValueError("result cache identity mismatch")
    for key in ("source_sha256", "lock_sha256", "python_version", "torch", "numpy"):
        if run["environment"][key] != expected[key]:
            raise ValueError(f"result provenance mismatch: {key}")
    if result["cursor"] != training_boundaries(intended, intended.model.parameter_count)[-1]:
        raise ValueError("result did not complete its training budget")
    loss = result["validation"]["loss"]
    if type(loss) not in (int, float) or not math.isfinite(loss) or result["validation"]["split"] != "full":
        raise ValueError("trial requires a finite final full-validation loss")
    return result


def fail_trial(trial: Record, attempt: Record, reason: str) -> None:
    attempt.update(status="failed", error=reason, finished=time.time())
    trial["status"] = "failed"


def reconcile(state: Record, backend: Backend, grace: float) -> bool:
    active = [trial for trial in state["trials"].values() if trial["status"] in {"submitting", "submitted"}]
    try:
        statuses = backend.poll([trial["attempts"][-1] for trial in active])
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        state["warning"] = str(error)
        return False
    state.pop("warning", None)
    certain = True
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
            fail_trial(trial, attempt, f"job ended in {condition}, exit={status['exit_code']}")
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
            state["trials"][key] = {
                "id": key,
                "directory": str(Path("trials") / name / key),
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
            link = view / key
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
            if phase["name"] == "confirmation":
                state.update(status="complete", winner=phase["rankings"][0], reason=None)
                return []
            phase = add_phase(state, spec, base)
            materialize_phase(directory, state, phase)
        ranks = ranking(phase["candidates"], phase["seeds"], state["trials"], state["provenance"])
        if ranks is not None:
            phase.update(
                complete=True,
                finished=time.time(),
                rankings=ranks,
                selected=[row["id"] for row in ranks[: phase["top_k"]]],
            )
            save_state(directory, state)
            write_report(directory, state)
            logger.info("Selected {} combinations in phase {}", phase["top_k"], phase["name"])
            continue
        required = [state["trials"][key] for keys in phase["trials"].values() for key in keys]
        exhausted = [
            trial
            for trial in required
            if trial["status"] == "failed" and len(trial["attempts"]) >= spec.controller.max_retries + 1
        ]
        if exhausted:
            state.update(status="failed", reason=f"Trial {exhausted[0]['id']} exhausted its retry limit.")
            return []
        return [trial for trial in required if trial["status"] in {"pending", "failed"}]


def submit(directory: Path, state: Record, trial: Record, backend: Backend) -> None:
    number = len(trial["attempts"]) + 1
    destination = trial_directory(directory, trial) / f"attempt-{number:03d}"
    config = Config.model_validate(trial["config"])
    # An extension can change scheduler resources or relocate the same immutable cache.
    config.launcher = Config.model_validate(state["base"]).launcher
    config.data.cache_dir = Path(state["base"]["data"]["cache_dir"])
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


def validate_cached(state: Record) -> None:
    for trial in state["trials"].values():
        if trial["status"] == "succeeded":
            try:
                check_result(trial, trial["attempts"][-1])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"cached trial {trial['id']} no longer has a valid result: {error}") from error


def run_loop(directory: Path, state: Record, spec: TuningSpec, base: Config, backend: Backend) -> Record:
    state.update(status="running", reason=None)
    try:
        save_state(directory, state)
        write_report(directory, state)
        while state["status"] == "running":
            certain = reconcile(state, backend, spec.controller.result_grace_seconds)
            pending = advance(directory, state, spec, base)
            if state["status"] != "running":
                break
            active = sum(trial["status"] in {"submitted", "submitting"} for trial in state["trials"].values())
            maximum = 1 if base.launcher.kind == "local" else spec.controller.max_jobs
            if certain:
                for trial in pending[: max(0, maximum - active)]:
                    submit(directory, state, trial, backend)
                    if trial["status"] == "failed" and len(trial["attempts"]) > spec.controller.max_retries:
                        break
            save_state(directory, state)
            logger.info(
                "Tuning revision {}: phase {}, {} outstanding jobs{}",
                state["revision"],
                state["phases"][-1]["name"],
                sum(trial["status"] in {"submitted", "submitting"} for trial in state["trials"].values()),
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
    dry_run: bool = False,
    retry_unresolved: list[str] | None = None,
) -> Record:
    directory = directory.resolve()
    if action == "run" and (directory / "state.json").exists():
        raise ValueError("tuning directory already contains state; use resume or extend")
    if action == "status":
        state = load_state(directory)
        counts: Record = {}
        for trial in state["trials"].values():
            counts[trial["status"]] = counts.get(trial["status"], 0) + 1
        return {key: state.get(key) for key in ("revision", "status", "reason", "warning", "updated")} | {
            "trials": counts
        }
    if action == "report":
        with study_lock(directory):
            return write_report(directory, load_state(directory))
    existing = None if action == "run" else load_state(directory)
    if spec_path is not None:
        spec, base = read_spec(spec_path.resolve())
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
    # All first-phase configurations are validated even for a dry run.
    for candidate in expand(base, spec.phases[0]):
        config = Config.model_validate(candidate["config"])
        config.runtime.output_dir = directory / "preflight"
        validate(config, launch=True, verify=False)
    if base.launcher.kind == "slurm":
        for executable in ("squeue", "sacct"):
            if shutil.which(executable) is None:
                raise ValueError(f"SLURM monitoring prerequisite missing: {executable}")
    if dry_run:
        return estimate(spec, base, current_provenance, existing["trials"] if existing else None)
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
                state.update(
                    revision=state["revision"] + 1,
                    spec=spec.model_dump(mode="json"),
                    base=base.model_dump(mode="json"),
                    provenance=current_provenance,
                    phases=[],
                    winner=None,
                )
        backend = Backend()
        if retry_unresolved:
            reconcile(state, backend, spec.controller.result_grace_seconds)
            for name in retry_unresolved:
                matches = [
                    trial
                    for trial in state["trials"].values()
                    if trial["attempts"] and trial["attempts"][-1]["name"] == name
                ]
                if len(matches) != 1 or matches[0]["attempts"][-1].get("scheduler_state") != "UNKNOWN":
                    raise ValueError(f"attempt {name} is not unresolved")
                trial = matches[0]
                fail_trial(
                    trial, trial["attempts"][-1], "Operator confirmed that the unresolved attempt has no live job."
                )
        save_state(directory, state)
        return run_loop(directory, state, spec, base, backend)
