"""AdamW with optional local-update correction and token-based learning-rate schedules."""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from decentralized_llm.config import Config, WSDScheduleConfig
from decentralized_llm.parameter_storage import ParameterStorage, prepare_parameter_storage


class ErrorFeedbackAdamW(torch.optim.AdamW):
    """Apply twice the full local AdamW displacement minus its previous displacement.

    The caller supplies the optimizer's base weights (after mixing for AWC, before
    mixing for ATC). History is rank-local and includes decoupled weight decay.
    Both buffers use the live storage's padded layout and alternate roles, avoiding
    a separate history copy. Placement and parameter layout must stay fixed after
    optimizer construction, as for the ordinary optimizer and execution wrappers.
    """

    def __init__(self, params: list[dict[str, Any]], storage: ParameterStorage, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.storage = storage
        self.previous_update = torch.zeros_like(storage.flat)
        self.scratch = torch.empty_like(storage.flat)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # Evaluate a closure before taking the snapshot, with the usual grad mode.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.scratch.copy_(self.storage.flat)
        super().step()
        torch.sub(self.storage.flat, self.scratch, out=self.scratch)
        self.storage.flat.add_(self.scratch).sub_(self.previous_update)
        # Writes through the flat alias do not update Parameter version counters.
        torch.autograd.graph.increment_version(self.storage.parameters)
        self.previous_update, self.scratch = self.scratch, self.previous_update
        return loss

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["error_feedback"] = {"previous_update": self.previous_update}
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        feedback = state_dict.get("error_feedback")
        previous = feedback.get("previous_update") if isinstance(feedback, dict) else None
        if (
            not isinstance(previous, torch.Tensor)
            or previous.layout != torch.strided
            or previous.shape != self.previous_update.shape
            or previous.dtype != self.previous_update.dtype
        ):
            raise ValueError("missing or incompatible error_feedback previous_update")
        super().load_state_dict({key: value for key, value in state_dict.items() if key != "error_feedback"})
        self.previous_update.copy_(previous)


def make_optimizer(model: nn.Module, config: Config, device: torch.device) -> torch.optim.AdamW:
    storage = prepare_parameter_storage(model)
    cfg = config.optimizer
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    fused = config.runtime.fused_optimizer and device.type == "cuda" and not config.runtime.deterministic
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    options: dict[str, Any] = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": cfg.eps,
        "fused": fused,
        "foreach": False if config.runtime.deterministic else None,
    }
    if config.distributed.error_feedback:
        return ErrorFeedbackAdamW(groups, storage, **options)
    return torch.optim.AdamW(groups, **options)


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
