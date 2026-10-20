# pyright: reportMissingParameterType=false
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportPossiblyUnboundVariable=false, reportIncompatibleMethodOverride=false
"""Paired SwiGLU derivatives: share loads for gate and up gradients.

Precision follows the pinned TorchInductor fused BF16 implementation: FP32
intermediates, BF16 output and projection gradients, FP32 master parameters.
This is not the eager per-operator BF16 rounding contract.
"""

import torch
from triton.language.extra.cuda import libdevice

from .kernels import jit, tl, triton


@jit
def _swiglu_fwd(X, Y, N: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = i // F * (2 * F) + i % F
    gate = tl.load(X + offset, i < N, other=0).to(tl.float32)
    up = tl.load(X + offset + F, i < N, other=0).to(tl.float32)
    silu = gate / (1.0 + libdevice.exp(-gate))
    tl.store(Y + i, silu * up, i < N)


@jit
def _swiglu_bwd(
    X,
    DY,
    DX,
    N: tl.constexpr,
    F: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    STAGES: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    # With a full grid this specializes to one iteration. A bounded grid
    # permits asynchronous staging of the next tile while computing this one.
    for tile in tl.range(tl.cdiv(N, GROUPS * BLOCK), num_stages=STAGES):
        i = (tile * GROUPS + tl.program_id(0)) * BLOCK + lane
        offset = i // F * (2 * F) + i % F
        gate = tl.load(X + offset, i < N, other=0).to(tl.float32)
        up = tl.load(X + offset + F, i < N, other=0).to(tl.float32)
        dy = tl.load(DY + i, i < N, other=0).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        # Retain Inductor's separate expressions for the derivative and SiLU;
        # sharing a differently rounded sigmoid would change BF16 outputs.
        dg = ((dy * up) * sigmoid) * (gate * (1.0 - sigmoid) + 1.0)
        du = dy * (gate / (1.0 + libdevice.exp(-gate)))
        tl.store(DX + offset, dg, i < N)
        tl.store(DX + offset + F, du, i < N)


class _PairedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, policy):
        width, n = packed.shape[-1] // 2, packed.numel() // 2
        output = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
        _swiglu_fwd[(triton.cdiv(n, 1024),)](packed, output, n, width, 1024)
        ctx.save_for_backward(packed)
        ctx.policy = policy
        return output

    @staticmethod
    def backward(ctx, dy):
        (packed,) = ctx.saved_tensors
        block, group_limit, stages = ctx.policy
        n, width = packed.numel() // 2, packed.shape[-1] // 2
        groups = min(triton.cdiv(n, block), group_limit) if group_limit else triton.cdiv(n, block)
        dx = torch.empty_like(packed)
        _swiglu_bwd[(groups,)](packed, dy.contiguous(), dx, n, width, block, groups, stages)
        return dx, None


def paired_swiglu(packed, *, policy=(1024, 0, 1)):
    if not packed.is_cuda or packed.dtype != torch.bfloat16 or not packed.is_contiguous():
        raise ValueError("paired SwiGLU requires contiguous CUDA BF16")
    if packed.ndim < 2 or packed.numel() == 0 or packed.shape[-1] not in (1792, 2816, 3584, 5120, 6912):
        raise ValueError("paired SwiGLU requires packed FFN width 896/1408/1792/2560/3456")
    block, groups, stages = policy
    if block not in (256, 512, 1024, 2048, 4096) or groups not in (0, 264, 528, 1056) or stages not in (1, 2, 3):
        raise ValueError("unsupported paired SwiGLU memory policy")
    return _PairedSwiGLU.apply(packed, policy)
