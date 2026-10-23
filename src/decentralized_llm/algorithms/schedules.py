"""Global-token learning-rate schedules."""

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

from decentralized_llm.config import Config, WSDScheduleConfig


def warmup_updates(config: Config) -> int:
    """Round warmup TPP to nearest global updates, with ties up and at least one if positive."""
    tpp = config.lr_schedule.warmup_tokens_per_parameters
    if tpp == 0:
        return 0
    updates = Fraction(str(tpp)) * config.model.parameter_count / config.training.batch_tokens
    return max(1, math.floor(updates + Fraction(1, 2)))


def learning_rate(config: Config, consumed: int, total: int) -> float:
    """Warm up over the TPP budget rounded to whole global token batches, then decay."""
    cfg = config.lr_schedule
    lr = config.optimizer.lr
    warmup = warmup_updates(config) * config.training.batch_tokens
    if warmup > 0 and consumed <= warmup:
        return lr * consumed / warmup
    if isinstance(cfg, WSDScheduleConfig):
        if cfg.decay_fraction == 0:
            return lr
        if consumed >= total:
            return 0.0
        stable_end = max(warmup, (1 - cfg.decay_fraction) * total)
        if consumed <= stable_end:
            return lr
        progress = min(1.0, max(0.0, (consumed - stable_end) / (total - stable_end)))
        return lr * (1 - math.sqrt(progress))
    if warmup >= total:
        return lr * cfg.min_lr_ratio
    progress = min(1.0, max(0.0, (consumed - warmup) / (total - warmup)))
    return lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def learning_rates(config: Config, consumed: int, total: int) -> dict[str, float]:
    """Scheduled base rates before Muon's per-matrix RMS adjustment."""
    rates = {"lr": learning_rate(config, consumed, total)}
    if config.optimizer.name == "muon_adamw":
        rates.update(
            adamw_lr=rates["lr"],
            muon_lr=rates["lr"] * config.optimizer.muon_lr_ratio,
        )
    return rates


def set_learning_rates(groups: Iterable[dict[str, Any]], rates: dict[str, float]) -> None:
    for group in groups:
        group["lr"] = rates["muon_lr"] if group.get("optimizer_kind") == "muon" else rates["lr"]
