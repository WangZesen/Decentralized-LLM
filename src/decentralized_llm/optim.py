"""Upstream AdamW groups and token-based learning-rate schedules."""

import math

import torch
from torch import nn

from decentralized_llm.config import Config, WSDScheduleConfig


def make_optimizer(model: nn.Module, config: Config, device: torch.device) -> torch.optim.AdamW:
    cfg = config.optimizer
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    fused = config.runtime.fused_optimizer and device.type == "cuda" and not config.runtime.deterministic
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=fused,
        foreach=False if config.runtime.deterministic else None,
    )
    return optimizer


def learning_rate(config: Config, consumed: int, total: int) -> float:
    """Warm up for fixed optimizer updates, each consuming one global token batch."""
    cfg = config.lr_schedule
    lr = config.optimizer.lr
    warmup = cfg.warmup_steps * config.training.batch_tokens
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
