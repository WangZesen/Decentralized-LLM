# pyright: reportMissingParameterType=false
"""Gathered AdamW and consensus over canonical, independently owned worker arenas.

All committed writes follow one global, sticky finite latch. A failed worker
therefore prevents partial optimizer/consensus updates in every other worker.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from decentralized_llm.config import Config
from decentralized_llm.models.packed import PackedLlama

from .kernels import jit


@jit
def _gather(POINTERS, DESC, GRAD, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    block, worker = tl.program_id(0), tl.program_id(1)
    parameter = tl.load(DESC + block * 6)
    offset = tl.load(DESC + block * 6 + 1)
    destination = tl.load(DESC + block * 6 + 2)
    size = tl.load(DESC + block * 6 + 3)
    local_size = tl.load(DESC + block * 6 + 4)
    pointer = tl.load(POINTERS + parameter).to(tl.pointer_type(tl.float32))
    i = tl.arange(0, BLOCK)
    g = tl.load(pointer + worker * local_size + offset + i, i < size, other=0.0)
    tl.store(GRAD + worker * STRIDE + destination + i, g, i < size)


@jit
def _statistics(G, PARTIAL, STRIDE: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    block, worker = tl.program_id(0), tl.program_id(1)
    i = block * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + worker * STRIDE + i, i < STRIDE, other=0.0)
    tl.store(PARTIAL + worker * PARTS + block, tl.sum(g * g, 0))


@jit
def _finish_statistics(
    PARTIAL,
    LOSS,
    FINITE,
    SCALE,
    NORMS,
    PARTS: tl.constexpr,
    N: tl.constexpr,
    MAX_NORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    valid = tl.load(FINITE) != 0
    for worker in range(N):
        norm = tl.sqrt(tl.sum(tl.load(PARTIAL + worker * PARTS + i, i < PARTS, other=0.0), 0))
        loss = tl.load(LOSS + worker)
        valid = valid & (norm == norm) & (norm != float("inf"))
        valid = valid & (loss == loss) & (tl.abs(loss) != float("inf"))
    tl.store(FINITE, valid.to(tl.int32))
    if valid:
        for worker in range(N):
            norm = tl.sqrt(tl.sum(tl.load(PARTIAL + worker * PARTS + i, i < PARTS, other=0.0), 0))
            scale = 1.0 if MAX_NORM == 0.0 else tl.minimum(1.0, MAX_NORM / (norm + 1.0e-6))
            tl.store(SCALE + worker, scale)
            tl.store(NORMS + worker, norm)


@jit
def _coefficients(LR, STEP, FINITE, COEF, B1: tl.constexpr, B2: tl.constexpr, WD: tl.constexpr):
    if tl.load(FINITE) != 0:
        step = (tl.load(STEP) + 1).to(tl.float64)
        lr = tl.load(LR).to(tl.float64)
        c1 = 1.0 - libdevice.pow(tl.full((), B1, tl.float64), step) if B1 != 0.0 else 1.0
        c2 = 1.0 - libdevice.pow(tl.full((), B2, tl.float64), step) if B2 != 0.0 else 1.0
        tl.store(COEF, 1.0 - lr * WD)
        tl.store(COEF + 1, lr / c1)
        tl.store(COEF + 2, libdevice.sqrt(c2))


@jit
def _adam(
    P,
    G,
    M,
    V,
    HISTORY,
    DESC,
    SCALE,
    FINITE,
    COEF,
    STRIDE: tl.constexpr,
    FEEDBACK: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(FINITE) != 0:
        block, worker = tl.program_id(0), tl.program_id(1)
        start = tl.load(DESC + block * 6 + 2)
        size = tl.load(DESC + block * 6 + 3)
        decay = tl.load(DESC + block * 6 + 5)
        i = tl.arange(0, BLOCK)
        address = worker * STRIDE + start + i
        p = tl.load(P + address, i < size, other=0.0)
        g = tl.load(G + address, i < size, other=0.0) * tl.load(SCALE + worker)
        m = tl.load(M + address, i < size, other=0.0)
        v = tl.load(V + address, i < size, other=0.0)
        if B1 > 0.5:
            m = tl.fma(g - m, 1.0 - B1, m)
        else:
            m = tl.fma(m - g, B1, g)
        v = v * B2 + (g * g) * (1.0 - B2)
        denominator = tl.sqrt(v) / tl.load(COEF + 2) + EPS
        base = p
        p = p * tl.where(decay != 0, tl.load(COEF), 1.0)
        p = p - tl.load(COEF + 1) * m / denominator
        if FEEDBACK:
            delta = p - base
            previous = tl.load(HISTORY + address, i < size, other=0.0)
            p = p + delta - previous
            tl.store(HISTORY + address, delta, i < size)
        tl.store(P + address, p, i < size)
        tl.store(M + address, m, i < size)
        tl.store(V + address, v, i < size)


@jit
def _mix(
    P,
    SCRATCH,
    PEERS,
    STEP,
    GAMMA,
    FINITE,
    STRIDE: tl.constexpr,
    N: tl.constexpr,
    PHASES: tl.constexpr,
    COMPLETE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(FINITE) != 0:
        worker = tl.program_id(1)
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        own = tl.load(P + worker * STRIDE + i, i < STRIDE, other=0.0)
        if COMPLETE:
            mixed = tl.full((BLOCK,), 0.0, tl.float32)
            for peer in range(N):
                mixed += tl.load(P + peer * STRIDE + i, i < STRIDE, other=0.0)
            mixed /= N
        else:
            phase = tl.load(STEP) % PHASES
            peer = tl.load(PEERS + phase * N + worker)
            mixed = (own + tl.load(P + peer * STRIDE + i, i < STRIDE, other=0.0)) * 0.5
        gamma = tl.load(GAMMA)
        mixed = tl.where(gamma == 1.0, mixed, mixed * gamma + own * (1.0 - gamma))
        tl.store(SCRATCH + worker * STRIDE + i, mixed, i < STRIDE)


@jit
def _commit_mix(P, SCRATCH, FINITE, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(FINITE) != 0:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(P + i, tl.load(SCRATCH + i, i < SIZE, other=0.0), i < SIZE)


@jit
def _commit_metrics(
    FINITE,
    STEP,
    LOSS,
    LAST_LOSS,
    LOSS_SUM,
    NORMS,
    CLIPS,
    N: tl.constexpr,
    MAX_NORM: tl.constexpr,
    LOCAL_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(FINITE) != 0:
        i = tl.arange(0, BLOCK)
        loss = tl.load(LOSS + i, i < N, other=0.0)
        tl.store(LAST_LOSS + i, loss, i < N)
        tl.store(LOSS_SUM, tl.load(LOSS_SUM) + tl.sum(loss, 0) * LOCAL_TOKENS)
        if MAX_NORM > 0.0:
            norm = tl.load(NORMS + i, i < N, other=0.0)
            clips = tl.load(CLIPS + i, i < N, other=0)
            tl.store(CLIPS + i, clips + (norm > MAX_NORM).to(tl.int64), i < N)
        tl.store(STEP, tl.load(STEP) + 1)


class ArenaAdamW:
    """Complete independent updates, including consensus and device counters."""

    def __init__(self, model: PackedLlama, config: Config):
        self.model, self.config = model, config
        self.parameters = tuple(model.parameters())
        self.arena = model.parameter_storage
        if not self.arena.is_cuda or self.arena.dtype != torch.float32:
            raise ValueError("GH200 AdamW requires FP32 CUDA arenas")
        self.previous_update = torch.zeros_like(self.arena) if config.distributed.error_feedback else None
        self.first_moment_storage = torch.zeros_like(self.arena)
        self.second_moment_storage = torch.zeros_like(self.arena)
        self.gradients = torch.zeros_like(self.arena)
        self.scratch = torch.empty_like(self.arena) if config.distributed.mode == "decentralized" else None
        self.workers, self.stride = self.arena.shape
        self.block = 4096
        desc = [
            (
                index,
                offset,
                entry.start + offset,
                min(self.block, entry.numel - offset),
                entry.numel,
                int(len(entry.shape) >= 2),
            )
            for index, entry in enumerate(model.layout)
            for offset in range(0, entry.numel, self.block)
        ]
        self.descriptors = torch.tensor(desc, dtype=torch.int64, device=self.arena.device)
        self.pointers = torch.empty(len(self.parameters), dtype=torch.int64, device=self.arena.device)
        self.parts = triton.cdiv(self.stride, self.block)
        self.partial = self.arena.new_empty((self.workers, self.parts))
        self.scales = self.arena.new_ones(self.workers)
        self.norms = self.arena.new_zeros(self.workers)
        self.losses = self.arena.new_zeros(self.workers)
        self.coefficients = self.arena.new_empty(3)
        self.lr = torch.tensor(config.optimizer.lr, dtype=torch.float64, device=self.arena.device)
        self.gamma = self.arena.new_tensor(1.0)
        self.finite = torch.ones((), dtype=torch.int32, device=self.arena.device)
        self.completed = torch.zeros((), dtype=torch.int64, device=self.arena.device)
        self.clip_counts = torch.zeros(self.workers, dtype=torch.int64, device=self.arena.device)
        self.loss_sum = self.arena.new_zeros(())
        topology = config.distributed.topology if config.distributed.mode == "decentralized" else "complete"
        self.complete = topology == "complete"
        self.phases = (
            (self.workers - 1).bit_length()
            if topology == "one_peer_exponential"
            else 2
            if topology == "one_peer_ring"
            else 1
        )
        self.phases = max(1, self.phases)
        peers = [
            [
                (
                    (worker ^ (1 << phase))
                    if topology == "one_peer_exponential"
                    else (worker + (1 if (worker + phase) % 2 == 0 else -1)) % self.workers
                )
                for worker in range(self.workers)
            ]
            for phase in range(self.phases)
        ]
        self.peers = torch.tensor(peers, dtype=torch.int64, device=self.arena.device)
        self._gradient_owners = []
        self._capturing = False
        self._finalized = False

    def zero_grad(self):
        if self._finalized:
            raise RuntimeError("captured gradients are owned by the graph; replay the complete update")
        for parameter in self.parameters:
            parameter.grad = None

    def bind_gradients(self):
        gradients = [p.grad for p in self.parameters]
        if any(g is None or not g.is_contiguous() or g.dtype != torch.float32 for g in gradients):
            raise ValueError("every fused leaf requires a contiguous FP32 gradient")
        gradients = cast(list[torch.Tensor], gradients)
        self.pointers.copy_(torch.tensor([g.data_ptr() for g in gradients], dtype=torch.int64))
        self._gradient_owners = gradients

    def finalize_capture(self):
        if not self._capturing or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("finalize_capture must follow capture, outside its context")
        self.bind_gradients()
        self._capturing, self._finalized = False, True

    def mix(self):
        if self.scratch is None or self.workers == 1:
            return
        _mix[(self.parts, self.workers)](
            self.arena,
            self.scratch,
            self.peers,
            self.completed,
            self.gamma,
            self.finite,
            self.stride,
            self.workers,
            self.phases,
            self.complete,
            self.block,
            enable_fp_fusion=False,
        )
        _commit_mix[(triton.cdiv(self.arena.numel(), self.block),)](
            self.arena, self.scratch, self.finite, self.arena.numel(), self.block
        )

    def step(self, means):
        if torch.cuda.is_current_stream_capturing():
            self._capturing = True
        else:
            self.bind_gradients()
        cfg = self.config.optimizer
        maximum = cfg.grad_clip or 0.0
        _gather[(len(self.descriptors), self.workers)](
            self.pointers, self.descriptors, self.gradients, self.stride, self.block
        )
        _statistics[(self.parts, self.workers)](self.gradients, self.partial, self.stride, self.parts, self.block)
        _finish_statistics[(1,)](
            self.partial,
            means,
            self.finite,
            self.scales,
            self.norms,
            self.parts,
            self.workers,
            maximum,
            triton.next_power_of_2(self.parts),
        )
        _coefficients[(1,)](
            self.lr,
            self.completed,
            self.finite,
            self.coefficients,
            cfg.beta1,
            cfg.beta2,
            cfg.weight_decay,
            enable_fp_fusion=False,
        )
        decentralized = self.config.distributed if self.config.distributed.mode == "decentralized" else None
        if decentralized and decentralized.scheme == "awc":
            self.mix()
        _adam[(len(self.descriptors), self.workers)](
            self.arena,
            self.gradients,
            self.first_moment_storage,
            self.second_moment_storage,
            self.previous_update if self.previous_update is not None else self.arena,
            self.descriptors,
            self.scales,
            self.finite,
            self.coefficients,
            self.stride,
            self.previous_update is not None,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            self.block,
            enable_fp_fusion=False,
        )
        if decentralized and decentralized.scheme == "atc":
            self.mix()
        _commit_metrics[(1,)](
            self.finite,
            self.completed,
            means,
            self.losses,
            self.loss_sum,
            self.norms,
            self.clip_counts,
            self.workers,
            maximum,
            self.config.training.batch_tokens // self.workers,
            triton.next_power_of_2(self.workers),
        )

    def inspect(self, expected_step=None):
        finite, completed = int(self.finite.item()), int(self.completed.item())
        if not finite:
            raise FloatingPointError(f"nonfinite GH200 update after {completed} successful updates")
        if expected_step is not None and completed != expected_step:
            raise RuntimeError(f"GH200 counter mismatch: device={completed}, expected={expected_step}")
        return completed

    def _groups(self):
        cfg = self.config.optimizer
        return [
            dict(
                lr=float(self.lr.item()),
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
                weight_decay=decay,
                amsgrad=False,
                maximize=False,
                foreach=None,
                capturable=False,
                differentiable=False,
                fused=True,
            )
            for decay in (cfg.weight_decay, 0.0)
        ]

    def local_state_dict(self, worker) -> dict[str, Any]:
        step = self.completed.detach().float().cpu()
        state = {
            entry.name: dict(
                step=step.clone(),
                exp_avg=self.model.parameter_view(self.first_moment_storage, entry)[worker],
                exp_avg_sq=self.model.parameter_view(self.second_moment_storage, entry)[worker],
            )
            for entry in self.model.layout
        }
        groups = [
            dict(group, params=[entry.name for entry in self.model.layout if (len(entry.shape) >= 2) == (index == 0)])
            for index, group in enumerate(self._groups())
        ]
        return dict(state=state, param_groups=groups)

    def state_dict(self):
        self.inspect()
        return dict(
            version=1,
            previous_update=self.previous_update,
            layout=self.model.layout_metadata(),
            first_moment_storage=self.first_moment_storage,
            second_moment_storage=self.second_moment_storage,
            workers=[
                dict(steps=[self.completed.detach().float().clone() for _ in self.model.layout], groups=self._groups())
                for _ in range(self.workers)
            ],
        )

    @torch.no_grad()
    def load_state_dict(self, state):
        steps = []
        groups = []
        if (
            state.get("version") != 1
            or state["layout"] != self.model.layout_metadata()
            or len(state["workers"]) != self.workers
        ):
            raise ValueError("incompatible canonical GH200 optimizer layout")
        for name in ("first_moment_storage", "second_moment_storage"):
            if state[name].shape != self.arena.shape:
                raise ValueError("incompatible canonical moment arena shape")
        for worker in state["workers"]:
            if len(worker["steps"]) != len(self.model.layout):
                raise ValueError("incompatible canonical optimizer steps")
            steps.extend(float(value) for value in worker["steps"])
            groups.append(worker["groups"])
        if not steps or any(
            not math.isfinite(step) or step != steps[0] or step < 0 or step != int(step) for step in steps
        ):
            raise ValueError("GH200 requires equal nonnegative integral Adam steps on all parameters")
        cfg = self.config.optimizer
        for worker_groups in groups:
            for group, decay in zip(worker_groups, (cfg.weight_decay, 0.0), strict=True):
                if (
                    tuple(group["betas"]) != (cfg.beta1, cfg.beta2)
                    or group["eps"] != cfg.eps
                    or group["weight_decay"] != decay
                    or group["amsgrad"]
                    or group["maximize"]
                ):
                    raise ValueError("checkpoint optimizer settings disagree with GH200 configuration")
        previous = state.get("previous_update")
        if self.previous_update is not None:
            if not isinstance(previous, torch.Tensor) or previous.shape != self.arena.shape:
                raise ValueError("missing or incompatible error_feedback previous_update")
        elif previous is not None:
            raise ValueError("unexpected error_feedback previous_update")
        if self.previous_update is not None:
            self.previous_update.copy_(cast(torch.Tensor, previous))
        # Validation precedes every mutation; captured tensors retain their storage.
        for name in ("first_moment_storage", "second_moment_storage"):
            getattr(self, name).copy_(state[name])
        self.completed.fill_(int(steps[0]))
        self.lr.fill_(groups[0][0]["lr"])
        self.finite.fill_(1)
