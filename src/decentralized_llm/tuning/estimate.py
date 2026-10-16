"""Read-only whole-study job estimates, exact when survivor identities do not matter."""

import math
import statistics

from decentralized_llm.config import Config
from decentralized_llm.tuning.batching import (
    allocation_failures,
    balanced_sizes,
    effective_config,
    format_time,
    group_trials,
    training_attempts,
    trial_minutes,
)
from decentralized_llm.tuning.spec import Record, TuningSpec, expand, field_value, search_points, trial_id


def ranking(candidates: list[Record], seeds: list[int], trials: Record, provenance: Record) -> list[Record] | None:
    rows = []
    for candidate in candidates:
        losses = []
        for seed in seeds:
            trial = trials.get(trial_id(candidate, seed, provenance))
            if trial is None or trial["status"] != "succeeded":
                return None
            losses.append({"seed": seed, "loss": trial["loss"], "trial_id": trial["id"]})
        values = [entry["loss"] for entry in losses]
        rows.append(
            {
                **candidate,
                "seeds": losses,
                "mean_loss": statistics.mean(values),
                "std_loss": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return sorted(rows, key=lambda row: (row["mean_loss"], row["id"]))


def estimate(
    spec: TuningSpec,
    base: Config,
    provenance: Record,
    trials: Record | None = None,
    allocations: Record | None = None,
) -> Record:
    trials = trials or {}
    parents: list[Record] | None = None
    known = True
    previous_fields: set[str] = set()
    previous_count = 1
    prior_initial: Record = {}
    initial_history: Record = {}
    simple = True
    planned: set[str] = set()
    rows: list[Record] = []
    limits = spec.controller.max_retries + 1
    compatible_history = any(trial.get("provenance") == provenance for trial in trials.values())
    varying_resources = False
    for phase in spec.phases:
        varying_resources |= bool({"distributed.nodes", "distributed.gpus_per_node"} & phase.search.keys())
        searches = {key: search_points(base, key, search) for key, search in phase.search.items()}
        initial = {key: field_value(base, key, value) for key, value in phase.initial_values.items()}
        revisited = bool(previous_fields & phase.search.keys())
        if revisited or any(key in initial_history and initial_history[key] != value for key, value in initial.items()):
            simple = False
        initial_history.update(initial)
        row: Record = {
            "phase": phase.name,
            "search": searches,
            "initial_values": phase.initial_values,
            "top_k": phase.top_k,
            "reused": 0,
            "already_submitted": 0,
            "new_jobs": 0,
            "additional_attempts_upper_bound": 0,
            "exact": known or (simple and not compatible_history),
            "_varying_resources": varying_resources,
        }
        if known:
            candidates = expand(base, phase, parents)
            row["candidates"] = len(candidates)
            count_known(row, candidates, spec.seeds(), trials, provenance, planned, limits)
            ranks = ranking(candidates, spec.seeds(), trials, provenance)
            if ranks is not None:
                parents = ranks[: phase.top_k]
            else:
                known = False
        else:
            # Without winner identities, revisited dimensions can collapse multiple parents.
            count = previous_count * math.prod(len(points) for points in searches.values())
            if phase.top_k > count:
                raise ValueError(f"{phase.name}: top_k exceeds even the candidate upper bound")
            row["candidates"] = count
            reuse = 0
            if (
                not revisited
                and all(
                    key in prior_initial and any(point["value"] == prior_initial[key] for point in points)
                    for key, points in searches.items()
                )
                and all(prior_initial.get(key) == value for key, value in initial.items())
            ):
                reuse = previous_count * spec.screening_seeds
            row["reused"] = reuse
            row["new_jobs"] = count * spec.screening_seeds - reuse
            row["additional_attempts_upper_bound"] = row["new_jobs"] * limits
        row["required_executions"] = row["candidates"] * spec.screening_seeds
        rows.append(row)
        previous_fields.update(phase.search)
        previous_count = phase.top_k
        prior_initial = initial
    row = {
        "phase": "confirmation",
        "candidates": previous_count,
        "required_executions": previous_count * spec.confirmation.seeds,
        "top_k": 1,
        "reused": 0,
        "already_submitted": 0,
        "new_jobs": 0,
        "additional_attempts_upper_bound": 0,
        "exact": known or not compatible_history,
        "_varying_resources": varying_resources,
    }
    if known:
        assert parents is not None
        count_known(row, parents, spec.seeds(True), trials, provenance, planned, limits)
    else:
        row["reused"] = previous_count * spec.screening_seeds
        row["new_jobs"] = previous_count * (spec.confirmation.seeds - spec.screening_seeds)
        row["additional_attempts_upper_bound"] = row["new_jobs"] * limits
    rows.append(row)
    for row in rows:
        allocation_estimate(row, spec, base)
    exact = all(row["exact"] for row in rows)
    total = sum(row["new_jobs"] for row in rows)
    attempts = sum(row["additional_attempts_upper_bound"] for row in rows)
    exhausted = [key for row in rows for key in row.get("exhausted_trials", [])]
    qualifier = "" if exact else "at most "
    new_allocations = sum(row["new_allocations"] for row in rows)
    allocation_exact = all(row["allocations_exact"] for row in rows)
    active_allocations = {
        attempt.get("allocation_id", attempt.get("name", key))
        for key, trial in trials.items()
        if trial["status"] in {"submitting", "submitted"}
        for attempt in trial["attempts"][-1:]
        if attempt.get("backend") == "slurm"
    }
    active_allocations.update(
        key for key, item in (allocations or {}).items() if item["status"] in {"submitting", "submitted"}
    )
    summary = f"Estimated total: {qualifier}{total} new jobs; up to {attempts} additional training attempts."
    if base.launcher.kind == "slurm":
        summary += (
            f" Initial grouping: {'' if allocation_exact else 'at most '}{new_allocations} new Slurm allocations."
        )
    return {
        "summary": summary,
        "exact": exact,
        "explanation": (
            "Counts assume all required trials eventually succeed. One configuration/seed execution is one job."
            if exact
            else "Conservative upper estimate: unknown winners can change deduplication or reuse of existing trials."
        ),
        "new_jobs": total,
        "already_submitted": sum(row["already_submitted"] for row in rows),
        "additional_attempts_upper_bound": attempts,
        "new_allocations": new_allocations,
        "allocations_exact": allocation_exact,
        "already_submitted_allocations": len(active_allocations),
        "additional_allocations_upper_bound": (
            attempts + sum(row["allocation_failures_upper_bound"] for row in rows)
            if base.launcher.kind == "slurm"
            else 0
        ),
        "completion_blocked": bool(exhausted),
        "exhausted_trials": exhausted,
        "phases": rows,
    }


def allocation_estimate(row: Record, spec: TuningSpec, base: Config) -> None:
    pending = row.pop("_pending", None)
    varying = row.pop("_varying_resources", False)
    remaining_failures = row.pop("_allocation_failures_remaining", 0)
    row.update(allocations=[], new_allocations=0, allocations_exact=row["exact"], allocation_failures_upper_bound=0)
    if base.launcher.kind != "slurm":
        return
    minutes = trial_minutes(base)
    if pending is not None:
        groups = group_trials(pending, base, spec.controller.jobs_per_allocation)
        for group in groups:
            config = effective_config(group[0], base)
            duration = sum(trial_minutes(effective_config(trial, base)) for trial in group)
            row["allocations"].append(
                {
                    "trial_ids": [trial["id"] for trial in group],
                    "jobs": len(group),
                    "nodes": config.distributed.nodes,
                    "gpus_per_node": config.distributed.gpus_per_node,
                    "time_minutes": duration,
                    "time": format_time(duration),
                }
            )
        row["allocation_failures_upper_bound"] = remaining_failures
    else:
        # Winner identities can affect resource compatibility. One allocation per trial is
        # a safe upper bound; never claim ceil(N / B) when the resource mix is unknown.
        maximum = 1 if varying else spec.controller.jobs_per_allocation
        row["allocations_exact"] &= not varying
        row["allocations"] = [
            {"jobs": size, "time_minutes": size * minutes, "time": format_time(size * minutes)}
            for size in balanced_sizes(row["new_jobs"], maximum)
        ]
        row["allocation_failures_upper_bound"] = row["new_jobs"] * (spec.controller.max_retries + 1)
    row["new_allocations"] = len(row["allocations"])
    row["max_allocation_time_minutes_upper_bound"] = (
        max((allocation["time_minutes"] for allocation in row["allocations"]), default=0)
        if pending is not None
        else min(spec.controller.jobs_per_allocation, row["new_jobs"]) * minutes
    )


def count_known(
    row: Record,
    candidates: list[Record],
    seeds: list[int],
    trials: Record,
    provenance: Record,
    planned: set[str],
    limits: int,
) -> None:
    row["_pending"] = []
    row["_allocation_failures_remaining"] = 0
    for candidate in candidates:
        for seed in seeds:
            key = trial_id(candidate, seed, provenance)
            trial = trials.get(key)
            if key in planned or trial is not None and trial["status"] == "succeeded":
                row["reused"] += 1
            else:
                attempts = training_attempts(trial) if trial else 0
                remaining = max(0, limits - attempts)
                if trial and allocation_failures(trial) >= limits:
                    remaining = 0
                if remaining:
                    row["_allocation_failures_remaining"] += max(
                        0, limits - (allocation_failures(trial) if trial else 0)
                    )
                if trial and trial["status"] in {"submitted", "submitting"}:
                    row["already_submitted"] += 1
                elif remaining:
                    row["new_jobs"] += 1
                    row["_pending"].append(trial or {"id": key, "config": candidate["config"], "attempts": []})
                else:
                    row.setdefault("exhausted_trials", []).append(key)
                row["additional_attempts_upper_bound"] += remaining
            planned.add(key)
