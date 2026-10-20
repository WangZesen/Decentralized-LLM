"""Pure allocation planning, Slurm walltime arithmetic, and retry accounting."""

import json
import math
import re

from decentralized_llm.config import Config, SlurmLauncher
from decentralized_llm.tuning.spec import Record


def time_minutes(value: str) -> int | None:
    """Slurm rounds seconds up; zero and UNLIMITED mean no time limit."""
    if value.upper() in {"UNLIMITED", "INFINITE"}:
        return None
    if not re.fullmatch(r"(?:\d+-)?\d+(?::\d+){0,2}", value):
        raise ValueError(f"invalid Slurm time: {value}")
    day, separator, clock = value.partition("-")
    parts = [int(part) for part in (clock if separator else day).split(":")]
    if separator:
        seconds = int(day) * 86400 + sum(part * unit for part, unit in zip(parts, (3600, 60, 1), strict=False))
    else:
        seconds = sum(part * unit for part, unit in zip(reversed(parts), (1, 60, 3600), strict=False))
        if len(parts) == 1:
            seconds *= 60
    return math.ceil(seconds / 60) or None


def format_time(minutes: int) -> str:
    days, remainder = divmod(minutes, 1440)
    hours, minute = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minute:02d}:00" if days else f"{hours:02d}:{minute:02d}:00"


def trial_minutes(config: Config) -> int:
    assert isinstance(config.launcher, SlurmLauncher)
    minutes = time_minutes(config.launcher.time)
    if minutes is None:
        raise ValueError("Slurm tuning allocations require finite, positive launcher.time")
    return minutes


def effective_config(trial: Record, base: Config) -> Config:
    config = Config.model_validate(trial["config"])
    config.launcher = base.launcher.model_copy(deep=True)
    config.data.cache_dir = base.data.cache_dir
    return config


def resource_key(config: Config) -> str:
    launcher = config.launcher.model_dump(mode="json", exclude={"time"})
    return json.dumps(
        {"launcher": launcher, "nodes": config.physical_nodes, "gpus": config.processes_per_node},
        sort_keys=True,
    )


def balanced_sizes(count: int, maximum: int) -> list[int]:
    if not count:
        return []
    groups = math.ceil(count / maximum)
    size, extra = divmod(count, groups)
    return [size + (index < extra) for index in range(groups)]


def group_trials(trials: list[Record], base: Config, maximum: int) -> list[list[Record]]:
    compatible: dict[tuple[str, bool], list[Record]] = {}
    for trial in trials:
        key = resource_key(effective_config(trial, base)), bool(trial["attempts"])
        compatible.setdefault(key, []).append(trial)
    result = []
    for group in compatible.values():
        offset = 0
        for size in balanced_sizes(len(group), maximum):
            result.append(group[offset : offset + size])
            offset += size
    return result


def check_walltime(minutes: int, limit: int | None, partition: str, individual: int) -> None:
    if limit is not None and minutes > limit:
        maximum = limit // individual
        suggestion = (
            f"set controller.jobs_per_allocation <= {maximum}"
            if maximum
            else "reduce launcher.time; even one trial exceeds the partition limit"
        )
        raise ValueError(
            f"Allocation requests {format_time(minutes)}, exceeding partition {partition} "
            f"MaxTime={format_time(limit)}; {suggestion}."
        )


def training_attempts(trial: Record) -> int:
    # Legacy/local attempts used the original submission-based budget.
    return sum("allocation_id" not in attempt or attempt.get("started", False) for attempt in trial["attempts"])


def allocation_failures(trial: Record) -> int:
    return sum(
        "allocation_id" in attempt and attempt["status"] == "failed" and not attempt.get("started", False)
        for attempt in trial["attempts"]
    )


def exhausted_reason(trial: Record, retries: int) -> str | None:
    if trial["status"] != "failed":
        return None
    if training_attempts(trial) > retries:
        return f"Trial {trial['id']} exhausted its retry limit."
    if allocation_failures(trial) > retries:
        return f"Trial {trial['id']} exhausted its allocation-failure limit before starting."
    return None
