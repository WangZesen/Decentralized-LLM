# pyright: reportMissingParameterType=false
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportPossiblyUnboundVariable=false, reportIncompatibleMethodOverride=false
"""Hopper-specialized residual/RMSNorm/autocast forward and backward fusion.

The FP32 residual stream is preserved. Only the output consumed by the following
autocast linear layer is BF16. Backward merges the residual gradient before
casting to the BF16 projection gradient. Norm-weight gradients use deterministic
partial reductions, without atomics or inter-program synchronization.
"""

import torch

from .kernels import jit, tl, triton

if triton is not None:

    @jit
    def _residual_norm_fwd(
        X,
        RESIDUAL,
        W,
        H,
        Y,
        RSTD,
        M: tl.constexpr,
        D: tl.constexpr,
        WS: tl.constexpr,
        EPS: tl.constexpr,
        ADD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        worker = tl.program_id(1)
        X += worker * M * D
        H += worker * M * D
        Y += worker * M * D
        RSTD += worker * M
        W += worker * WS
        if ADD:
            RESIDUAL += worker * M * D
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, COLS)
        offsets = rows[:, None] * D + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < D)
        h = tl.load(X + offsets, mask, other=0).to(tl.float32)
        if ADD:
            h += tl.load(RESIDUAL + offsets, mask, other=0)
        weight = tl.load(W + cols, cols < D, other=0)
        rstd = tl.rsqrt(tl.sum(h * h, axis=1) / D + EPS)
        normalized = h * rstd[:, None]
        tl.store(H + offsets, h, mask)
        tl.store(Y + offsets, normalized * weight[None, :], mask)
        tl.store(RSTD + rows, rstd, rows < M)

    @jit
    def _residual_norm_bwd(
        H,
        W,
        RSTD,
        DH,
        DY,
        DX,
        DRESIDUAL,
        PARTIAL,
        M: tl.constexpr,
        D: tl.constexpr,
        WS: tl.constexpr,
        HAS_DH: tl.constexpr,
        HAS_DY: tl.constexpr,
        ADD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        worker = tl.program_id(1)
        H += worker * M * D
        W += worker * WS
        RSTD += worker * M
        DX += worker * M * D
        PARTIAL += worker * GROUPS * D
        if HAS_DH:
            DH += worker * M * D
        if HAS_DY:
            DY += worker * M * D
        if ADD:
            DRESIDUAL += worker * M * D
        group = tl.program_id(0)
        cols = tl.arange(0, COLS)
        weight = tl.load(W + cols, cols < D, other=0)
        dw = tl.full((ROWS, COLS), 0, tl.float32)
        # 256 independent groups keep the fixed-width Hopper kernels occupied
        # while bounding partial-gradient storage independently of token count.
        for start in range(group * ROWS, M, GROUPS * ROWS):
            rows = start + tl.arange(0, ROWS)
            offsets = rows[:, None] * D + cols[None, :]
            mask = (rows[:, None] < M) & (cols[None, :] < D)
            h = tl.load(H + offsets, mask, other=0)
            rstd = tl.load(RSTD + rows, rows < M, other=0)
            normalized = h * rstd[:, None]
            if HAS_DY:
                dy = tl.load(DY + offsets, mask, other=0).to(tl.float32)
                wdy = dy * weight[None, :]
                correction = tl.sum(normalized * wdy, axis=1) / D
                dx = (wdy - normalized * correction[:, None]) * rstd[:, None]
                dw += dy * normalized
            else:
                dx = tl.full((ROWS, COLS), 0, tl.float32)
            if HAS_DH:
                dx += tl.load(DH + offsets, mask, other=0).to(tl.float32)
            tl.store(DX + offsets, dx, mask)
            if ADD:
                tl.store(DRESIDUAL + offsets, dx, mask)
        tl.store(PARTIAL + group * D + cols, tl.sum(dw, axis=0), cols < D)

    @jit
    def _norm_weight_reduce(PARTIAL, DW, D: tl.constexpr, GROUPS: tl.constexpr, R: tl.constexpr):
        worker = tl.program_id(1)
        PARTIAL += worker * GROUPS * D
        DW += worker * D
        rows = tl.arange(0, R)
        cols = tl.program_id(0) * 32 + tl.arange(0, 32)
        values = tl.load(
            PARTIAL + rows[:, None] * D + cols[None, :],
            (rows[:, None] < GROUPS) & (cols[None, :] < D),
            other=0,
        )
        tl.store(DW + cols, tl.sum(values, axis=0), cols < D)


class _ResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, branch, residual, weight, eps, rows):
        width = branch.shape[-1]
        replicas = weight.shape[0] if weight.ndim == 2 else 1
        tokens = branch.numel() // width // replicas
        stride = weight.stride(0) if weight.ndim == 2 else width
        hidden = torch.empty_like(branch, dtype=torch.float32)
        output = torch.empty_like(branch, dtype=torch.bfloat16)
        rstd = torch.empty((replicas, tokens), device=branch.device, dtype=torch.float32)
        _residual_norm_fwd[(triton.cdiv(tokens, rows), replicas)](
            branch,
            residual,
            weight,
            hidden,
            output,
            rstd,
            tokens,
            width,
            stride,
            eps,
            residual is not None,
            rows,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(hidden, weight, rstd)
        ctx.rows, ctx.add, ctx.branch_dtype = rows, residual is not None, branch.dtype
        ctx.replicas, ctx.tokens, ctx.stride = replicas, tokens, stride
        ctx.set_materialize_grads(False)
        return hidden, output

    @staticmethod
    def backward(ctx, dh, dy):
        hidden, weight, rstd = ctx.saved_tensors
        # General autograd users may supply strided or broadcast gradients.
        dh = dh.contiguous() if dh is not None else None
        dy = dy.contiguous() if dy is not None else None
        width, tokens = hidden.shape[-1], ctx.tokens
        groups = min(triton.cdiv(tokens, ctx.rows), 256)
        dx = torch.empty_like(hidden, dtype=ctx.branch_dtype)
        dr = torch.empty_like(hidden) if ctx.add else None
        partial = torch.empty((ctx.replicas, groups, width), device=hidden.device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        _residual_norm_bwd[(groups, ctx.replicas)](
            hidden,
            weight,
            rstd,
            dh,
            dy,
            dx,
            dr,
            partial,
            tokens,
            width,
            ctx.stride,
            dh is not None,
            dy is not None,
            ctx.add,
            ctx.rows,
            triton.next_power_of_2(width),
            groups,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _norm_weight_reduce[(triton.cdiv(width, 32), ctx.replicas)](
            partial,
            dw,
            width,
            groups,
            triton.next_power_of_2(groups),
            num_warps=4,
        )
        return dx, dr, dw, None, None


def _validate_inputs(branch, residual, weight, rows):
    if triton is None:
        raise RuntimeError("GH200 fusion requires Triton")
    if not branch.is_cuda or branch.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("branch must be CUDA FP32 or BF16")
    if (
        branch.ndim < 2
        or branch.numel() == 0
        or branch.shape[-1] not in (320, 512, 640, 960, 1280)
        or not branch.is_contiguous()
    ):
        raise ValueError("GH200 fusion requires contiguous inputs of width 320/512/640/960/1280")
    expected = (branch.shape[-1],) if weight.ndim == 1 else (branch.shape[0], branch.shape[-1])
    if tuple(weight.shape) != expected or weight.dtype != torch.float32:
        raise ValueError("RMSNorm weight must be FP32 with the input width")
    if weight.device != branch.device or weight.stride(-1) != 1:
        raise ValueError("weight rows must be contiguous on the branch device")
    if residual is not None and (
        residual.shape != branch.shape
        or residual.device != branch.device
        or residual.dtype != torch.float32
        or not residual.is_contiguous()
    ):
        raise ValueError("residual must be matching contiguous FP32")
    if rows not in (1, 2, 4, 8):
        raise ValueError("rows must be 1, 2, 4, or 8")


def residual_norm(branch, residual, weight, eps=1e-5, *, rows=4):
    """Return (FP32 residual sum, BF16 weighted RMSNorm) with a fused VJP.

    Intended for first-order training of the supported model presets.
    The BF16 result must replace a normalization immediately followed by a BF16
    autocast linear operation, not a general FP32 RMSNorm output.
    """
    _validate_inputs(branch, residual, weight, rows)
    return _ResidualNorm.apply(branch, residual, weight, eps, rows)
