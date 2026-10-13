"""Read-only whole-study job estimates, exact when survivor identities do not matter."""

import math
import statistics

from decentralized_llm.config import Config
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


def estimate(spec: TuningSpec, base: Config, provenance: Record, trials: Record | None = None) -> Record:
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
    for phase in spec.phases:
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
    }
    if known:
        assert parents is not None
        count_known(row, parents, spec.seeds(True), trials, provenance, planned, limits)
    else:
        row["reused"] = previous_count * spec.screening_seeds
        row["new_jobs"] = previous_count * (spec.confirmation.seeds - spec.screening_seeds)
        row["additional_attempts_upper_bound"] = row["new_jobs"] * limits
    rows.append(row)
    exact = all(row["exact"] for row in rows)
    total = sum(row["new_jobs"] for row in rows)
    attempts = sum(row["additional_attempts_upper_bound"] for row in rows)
    exhausted = [key for row in rows for key in row.get("exhausted_trials", [])]
    qualifier = "" if exact else "at most "
    return {
        "summary": f"Estimated total: {qualifier}{total} new jobs; up to {attempts} additional submission attempts.",
        "exact": exact,
        "explanation": (
            "Counts assume all required trials eventually succeed. One configuration/seed execution is one job."
            if exact
            else "Conservative upper estimate: unknown winners can change deduplication or reuse of existing trials."
        ),
        "new_jobs": total,
        "already_submitted": sum(row["already_submitted"] for row in rows),
        "additional_attempts_upper_bound": attempts,
        "completion_blocked": bool(exhausted),
        "exhausted_trials": exhausted,
        "phases": rows,
    }


def count_known(
    row: Record,
    candidates: list[Record],
    seeds: list[int],
    trials: Record,
    provenance: Record,
    planned: set[str],
    limits: int,
) -> None:
    for candidate in candidates:
        for seed in seeds:
            key = trial_id(candidate, seed, provenance)
            trial = trials.get(key)
            if key in planned or trial is not None and trial["status"] == "succeeded":
                row["reused"] += 1
            else:
                attempts = len(trial["attempts"]) if trial else 0
                remaining = max(0, limits - attempts)
                if trial and trial["status"] in {"submitted", "submitting"}:
                    row["already_submitted"] += 1
                elif remaining:
                    row["new_jobs"] += 1
                else:
                    row.setdefault("exhausted_trials", []).append(key)
                row["additional_attempts_upper_bound"] += remaining
            planned.add(key)
