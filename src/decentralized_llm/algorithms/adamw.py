"""AdamW with optional local-update correction and token-based learning-rate schedules."""

import importlib.util
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from decentralized_llm.config import Config
from decentralized_llm.models.storage import ParameterStorage, prepare_parameter_storage


class ErrorFeedbackAdamW(torch.optim.AdamW):
    """Apply (1 + gamma) times the AdamW displacement minus gamma times its history.

    The caller supplies the optimizer's base weights (after mixing for AWC, before
    mixing for ATC). History is rank-local and includes decoupled weight decay.
    Both buffers use the live storage's padded layout and alternate roles, avoiding
    a separate history copy. Placement and parameter layout must stay fixed after
    optimizer construction, as for the ordinary optimizer and execution wrappers.
    """

    def __init__(
        self, params: list[dict[str, Any]], storage: ParameterStorage, *, feedback_gamma: float, **kwargs: Any
    ) -> None:
        super().__init__(params, **kwargs)
        self.storage = storage
        self.feedback_gamma = feedback_gamma
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
        self.storage.flat.add_(self.scratch, alpha=self.feedback_gamma).sub_(
            self.previous_update, alpha=self.feedback_gamma
        )
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


def optimizer_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    history = getattr(optimizer, "previous_update", None)
    scratch = getattr(optimizer, "scratch", None)
    return {
        "implementation": getattr(optimizer, "implementation", "reference"),
        "history_bytes": history.nbytes if history is not None else 0,
        "scratch_bytes": scratch.nbytes if scratch is not None else 0,
    }


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
    if (
        config.distributed.mode == "decentralized"
        and config.runtime.decentralized_optimizer == "auto"
        and fused
        and storage.flat.is_cuda
        and storage.flat.dtype == torch.float32
        and torch.version.cuda is not None
        and importlib.util.find_spec("triton") is not None
    ):
        from decentralized_llm.execution.cuda.adamw import FusedDecentralizedAdamW

        return FusedDecentralizedAdamW(
            groups, storage, feedback_gamma=config.distributed.error_feedback_gamma, **options
        )
    if config.distributed.error_feedback_gamma != 0:
        return ErrorFeedbackAdamW(groups, storage, feedback_gamma=config.distributed.error_feedback_gamma, **options)
    return torch.optim.AdamW(groups, **options)
