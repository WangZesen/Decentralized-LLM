"""Opt-in batched Muon updates with the canonical PyTorch optimizer state layout."""

from collections import defaultdict
from collections.abc import Callable
from functools import cache
from typing import Any, Literal

import torch
from torch.optim.optimizer import ParamsT

# PyTorch exposes this operation at runtime but omits it from its export stubs.
foreach_copy = torch._foreach_copy_  # pyright: ignore[reportPrivateImportUsage]

type Mode = Literal["batched", "compiled", "cudagraph"]


def batched_update(
    weights: torch.Tensor,
    gradients: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: torch.Tensor,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    ns_steps: int,
    coefficients: tuple[float, float, float],
    eps: float,
    adjustment: float,
) -> None:
    """Keep normalization and all five NS iterations independent for each matrix."""
    momentum_buffer.lerp_(gradients, 1 - momentum)
    update = gradients.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    x = update.bfloat16()
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    a, b, c = coefficients
    for _ in range(ns_steps):
        gram = x @ x.mT
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, polynomial, x, beta=a)
    # Compute scalar factors in FP64, like the reference's Python floats, then
    # round to the parameter dtype. Do not round the scaled update back to BF16.
    decay = (1 - lr * weight_decay).to(weights.dtype)
    scale = (-lr * adjustment).to(weights.dtype)
    weights.mul_(decay)
    weights.add_(x.to(weights.dtype) * scale)


@cache
def compiled_update() -> Callable[..., None]:
    # Explicit capture below owns state restoration and gradient staging.
    return torch.compile(
        batched_update,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )


class Bucket:
    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], group: dict[str, Any]):
        self.parameters = [p for p, _ in pairs]
        self.transposed = [p.shape[0] > p.shape[1] for p in self.parameters]
        self.parameter_views = [
            p.mT if transpose else p for p, transpose in zip(self.parameters, self.transposed, strict=True)
        ]
        self.state_views = [m.mT if transpose else m for (_, m), transpose in zip(pairs, self.transposed, strict=True)]
        self.weights = torch.empty(
            (len(pairs), *self.parameter_views[0].shape),
            device=self.parameters[0].device,
            dtype=self.parameters[0].dtype,
        )
        self.gradients = torch.empty_like(self.weights)
        self.momentum = torch.empty_like(self.weights)
        self.weight_views = list(self.weights.unbind())
        self.gradient_views = list(self.gradients.unbind())
        self.momentum_views = list(self.momentum.unbind())
        rows, cols = self.parameters[0].shape
        adjustment = group["adjust_lr_fn"]
        if adjustment == "match_rms_adamw":
            self.adjustment = 0.2 * max(rows, cols) ** 0.5
        elif adjustment == "spectral_unclamped":
            self.adjustment = (rows / cols) ** 0.5
        else:
            self.adjustment = max(1, rows / cols) ** 0.5
        self.options = (
            group["momentum"],
            group["weight_decay"],
            group["nesterov"],
            group["ns_steps"],
            group["ns_coefficients"],
            group["eps"],
            self.adjustment,
        )

    def stage_gradients(self) -> None:
        gradients = []
        for parameter, transpose in zip(self.parameters, self.transposed, strict=True):
            assert parameter.grad is not None
            gradients.append(parameter.grad.mT if transpose else parameter.grad)
        foreach_copy(self.gradient_views, gradients)

    def update(self, function: Callable[..., None], lr: torch.Tensor) -> None:
        foreach_copy(self.weight_views, self.parameter_views)
        foreach_copy(self.momentum_views, self.state_views)
        function(self.weights, self.gradients, self.momentum, lr, *self.options)
        foreach_copy(self.parameter_views, self.weight_views)
        foreach_copy(self.state_views, self.momentum_views)


class UpdatePlan:
    def __init__(self, params: list[torch.Tensor], buffers: list[torch.Tensor], group: dict[str, Any], mode: Mode):
        # Opposite orientations can share an RMS-matched bucket; other adjustment
        # rules depend on the original orientation and therefore keep it separate.
        groups = defaultdict(list)
        for p, buffer in zip(params, buffers, strict=True):
            shape = tuple(sorted(p.shape)) if group["adjust_lr_fn"] == "match_rms_adamw" else tuple(p.shape)
            groups[(shape, p.device, p.dtype)].append((p, buffer))
        self.buckets = [Bucket(pairs, group) for pairs in groups.values()]
        self.params, self.buffers, self.mode = params, buffers, mode
        self.device = params[0].device
        if any(p.device != self.device for p in params):
            raise ValueError("batched Muon requires one device per parameter group")
        self.lr = torch.zeros((), dtype=torch.float64, device=self.device)
        self.function = batched_update if mode == "batched" else compiled_update()
        self.graph: torch.cuda.CUDAGraph | None = None

    def compute(self) -> None:
        for bucket in self.buckets:
            bucket.update(self.function, self.lr)

    def capture(self) -> None:
        persistent = [*self.params, *self.buffers]
        saved = [tensor.detach().clone() for tensor in persistent]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.compute()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.compute()
            self.graph = graph
        finally:
            stream.synchronize()
            # Capture/warmup must not consume an optimization step.
            foreach_copy(persistent, saved)

    def step(self, lr: float | torch.Tensor) -> None:
        if isinstance(lr, torch.Tensor):
            self.lr.copy_(lr)
        else:
            self.lr.fill_(lr)
        for bucket in self.buckets:
            bucket.stage_gradients()
        if self.mode == "cudagraph":
            if self.graph is None:
                self.capture()
            assert self.graph is not None
            self.graph.replay()
            torch.autograd.graph.increment_version(self.params)
        else:
            self.compute()


class BatchedMuon(torch.optim.Muon):
    """PyTorch-compatible state, with fresh gradient staging for every graph replay."""

    def __init__(self, params: ParamsT, *, mode: Mode, **kwargs: Any):
        super().__init__(params, **kwargs)
        self.mode: Mode = mode
        self._plans: dict[int, tuple[tuple, UpdatePlan]] = {}
        if any(p.dtype not in (torch.float32, torch.float64) for g in self.param_groups for p in g["params"]):
            raise ValueError("batched Muon requires FP32 or FP64 parameters")
        if mode != "batched" and any(p.device.type != "cuda" for g in self.param_groups for p in g["params"]):
            raise ValueError("compiled and cudagraph Muon require CUDA parameters")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for index, group in enumerate(self.param_groups):
            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            buffers: list[torch.Tensor] = []
            self._init_group(group, params, grads, buffers)
            if not params:
                continue
            signature = (
                tuple(id(p) for p in params),
                tuple(id(b) for b in buffers),
                group["momentum"],
                group["weight_decay"],
                group["nesterov"],
                group["ns_steps"],
                tuple(group["ns_coefficients"]),
                group["eps"],
                group["adjust_lr_fn"],
            )
            cached = self._plans.get(index)
            if cached is None or cached[0] != signature:
                cached = (signature, UpdatePlan(params, buffers, group, self.mode))
                self._plans[index] = cached
            cached[1].step(group["lr"])
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Restored buffers have new addresses; captured graphs must be rebuilt.
        self._plans.clear()
