"""AdamW with optional local-update correction and token-based learning-rate schedules."""

import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
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


@dataclass
class _FusedGroup:
    group: dict[str, Any]
    parameters: list[nn.Parameter]
    offsets: list[int]
    metadata: torch.Tensor
    tiles: torch.Tensor
    coefficients: torch.Tensor
    gradient_pointers: tuple[int, ...] | None = None
    aligned: bool = False


class FusedDecentralizedAdamW(torch.optim.AdamW):
    """FP32 CUDA AdamW with optional local history and fused mixing arithmetic.

    Parameters retain their original identities and AdamW state dictionaries.
    Placement and parameter groups must remain fixed, as for parameter storage.
    """

    def __init__(
        self, params: list[dict[str, Any]], storage: ParameterStorage, *, feedback: bool, **kwargs: Any
    ) -> None:
        super().__init__(params, **kwargs)
        self.storage = storage
        self.previous_update = torch.zeros_like(storage.flat) if feedback else None
        self.scratch: torch.Tensor | None = None
        self._layouts: list[_FusedGroup] | None = None
        self.implementation = "triton"
        # Let GradScaler unscale/check gradients before calling this optimizer.
        self._step_supports_amp_scaling = False

    def _prepare_layouts(self) -> list[_FusedGroup]:
        from decentralized_llm.optimizer_kernels import BLOCK

        if self._layouts is None:
            offsets = dict(zip(self.storage.parameters, self.storage.offsets, strict=True))
            self._layouts = []
            for group in self.param_groups:
                params = group["params"]
                if not params:
                    continue
                tiles = [(i, start) for i, p in enumerate(params) for start in range(0, p.numel(), BLOCK)]
                self._layouts.append(
                    _FusedGroup(
                        group,
                        params,
                        [offsets[p] for p in params],
                        torch.empty((len(params), 6), dtype=torch.int64, device=self.storage.flat.device),
                        torch.tensor(tiles, dtype=torch.int64, device=self.storage.flat.device).reshape(-1, 2),
                        torch.empty((len(params), 2), device=self.storage.flat.device),
                    )
                )
        return self._layouts

    def _supported(self) -> bool:
        return all(
            not any(group.get(key, False) for key in ("amsgrad", "maximize", "differentiable", "capturable"))
            and not isinstance(group["lr"], torch.Tensor)
            and all(
                p.grad is None or (p.grad.layout == torch.strided and p.grad.is_contiguous()) for p in group["params"]
            )
            for group in self.param_groups
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_fused()
        return loss

    @torch.no_grad()
    def step_fused(self, *, base: torch.Tensor | None = None, base_divisor: int = 1, output_divisor: int = 1) -> None:
        """Consume an AWC sum, or pre-normalize ATC output for an in-place SUM."""
        if not self._supported():
            self._reference_step(base, base_divisor, output_divisor)
            return
        from decentralized_llm.optimizer_kernels import update

        for layout in self._prepare_layouts():
            pointers = tuple(-1 if p.grad is None else p.grad.data_ptr() for p in layout.parameters)
            if pointers != layout.gradient_pointers:
                rows = []
                for p, offset in zip(layout.parameters, layout.offsets, strict=True):
                    if p.grad is None:
                        rows.append((offset, p.numel(), 0, 0, 0, 0))
                        continue
                    state = self.state[p]
                    if not state:
                        state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    rows.append(
                        (
                            offset,
                            p.numel(),
                            p.grad.data_ptr(),
                            state["exp_avg"].data_ptr(),
                            state["exp_avg_sq"].data_ptr(),
                            state["step"].data_ptr(),
                        )
                    )
                # A fresh pinned allocation cannot be overwritten while its DMA is in
                # flight. PyTorch's pinned allocator records the copy's completion.
                host = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
                layout.metadata.copy_(host, non_blocking=True)
                layout.gradient_pointers = pointers
                layout.aligned = all(row[1] % 4 == 0 and all(address % 16 == 0 for address in row[2:5]) for row in rows)
            update(
                self.storage.flat,
                self.storage.flat if base is None else base,
                self.previous_update,
                layout.metadata,
                layout.tiles,
                layout.coefficients,
                layout.group,
                base_divisor,
                output_divisor,
                layout.aligned,
            )
        torch.autograd.graph.increment_version(self.storage.parameters)
        self.implementation = "triton"

    def _reference_step(self, base: torch.Tensor | None, base_divisor: int, output_divisor: int) -> None:
        if base is not None:
            self.storage.flat.copy_(base).div_(base_divisor)
        if self.previous_update is not None:
            if self.scratch is None:
                self.scratch = torch.empty_like(self.storage.flat)
            self.scratch.copy_(self.storage.flat)
        # Noncontiguous dense gradients need the strided PyTorch implementation.
        options = [(group.get("fused"), group.get("foreach")) for group in self.param_groups]
        try:
            for group in self.param_groups:
                group.update(fused=False, foreach=False)
            super().step()
        finally:
            for group, (fused, foreach) in zip(self.param_groups, options, strict=True):
                group.update(fused=fused, foreach=foreach)
        if self.previous_update is not None:
            assert self.scratch is not None
            torch.sub(self.storage.flat, self.scratch, out=self.scratch)
            self.storage.flat.add_(self.scratch).sub_(self.previous_update)
            self.previous_update.copy_(self.scratch)
        if output_divisor != 1:
            self.storage.flat.div_(output_divisor)
        torch.autograd.graph.increment_version(self.storage.parameters)
        self.implementation = "reference"

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self.previous_update is not None:
            state["error_feedback"] = {"previous_update": self.previous_update}
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        feedback = state_dict.get("error_feedback")
        previous = feedback.get("previous_update") if isinstance(feedback, dict) else None
        if self.previous_update is not None:
            if (
                not isinstance(previous, torch.Tensor)
                or previous.layout != torch.strided
                or previous.shape != self.previous_update.shape
                or previous.dtype != self.previous_update.dtype
            ):
                raise ValueError("missing or incompatible error_feedback previous_update")
        elif feedback is not None:
            raise ValueError("unexpected error_feedback history for ordinary AdamW")
        super().load_state_dict({key: value for key, value in state_dict.items() if key != "error_feedback"})
        for group in self.param_groups:
            group.update(fused=True, foreach=None)
            for p in group["params"]:
                state = self.state.get(p)
                if state:
                    state["step"] = state["step"].to(device=p.device, dtype=torch.float32)
        if self.previous_update is not None:
            assert isinstance(previous, torch.Tensor)
            self.previous_update.copy_(previous)
        self._layouts = None


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
        return FusedDecentralizedAdamW(groups, storage, feedback=config.distributed.error_feedback, **options)
    if config.distributed.error_feedback:
        return ErrorFeedbackAdamW(groups, storage, **options)
    return torch.optim.AdamW(groups, **options)


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
