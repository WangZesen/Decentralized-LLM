"""Fused per-process AdamW; kernels are imported only when stepping on CUDA."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from decentralized_llm.models.storage import ParameterStorage


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
        from decentralized_llm.execution.cuda.optimizer_kernels import BLOCK

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
        from decentralized_llm.execution.cuda.optimizer_kernels import update

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
