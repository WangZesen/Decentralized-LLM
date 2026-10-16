# pyright: reportMissingParameterType=false
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportPossiblyUnboundVariable=false, reportIncompatibleMethodOverride=false
"""Experimental latency-hiding residual/RMSNorm kernels for GH200.

The previous implementation stays available as a measured control. Policies
vary independent resident groups, software pipeline depth, and register
prefetching while retaining the same FP32 arithmetic and BF16 boundaries.
"""

import torch

from .gh200_fusions import _norm_weight_reduce, _residual_norm_fwd, _validate_inputs, residual_norm
from .kernels import jit, tl, triton


@jit
def _load_tile(
    H,
    RSTD,
    DY,
    DH,
    rows,
    cols,
    M: tl.constexpr,
    D: tl.constexpr,
    HAS_DY: tl.constexpr,
    HAS_DH: tl.constexpr,
    CACHE: tl.constexpr,
):
    offsets = rows[:, None] * D + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < D)
    h = tl.load(H + offsets, mask, other=0, cache_modifier=CACHE)
    rstd = tl.load(RSTD + rows, rows < M, other=0)
    if HAS_DY:
        # Preserve BF16 storage until consumption. The tuning records report
        # actual register allocation; a narrow type alone does not ensure it.
        dy = tl.load(DY + offsets, mask, other=0, cache_modifier=CACHE)
    else:
        dy = tl.full(h.shape, 0, tl.float32)
    if HAS_DH:
        dh = tl.load(DH + offsets, mask, other=0, cache_modifier=CACHE)
    else:
        dh = tl.full(h.shape, 0, tl.float32)
    return h, rstd, dy, dh


@jit
def _norm_bwd_pipeline(
    H,
    W,
    RSTD,
    DH,
    DY,
    DX,
    DR,
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
    STAGES: tl.constexpr,
    PREFETCH: tl.constexpr,
    CACHE: tl.constexpr,
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
        DR += worker * M * D
    group = tl.program_id(0)
    cols = tl.arange(0, COLS)
    weight = tl.load(W + cols, cols < D, other=0)
    dw = tl.full((ROWS, COLS), 0, tl.float32)
    if PREFETCH:
        rows = group * ROWS + tl.arange(0, ROWS)
        h, rstd, dy, dh = _load_tile(H, RSTD, DY, DH, rows, cols, M, D, HAS_DY, HAS_DH, CACHE)
    for start in tl.range(group * ROWS, M, GROUPS * ROWS, num_stages=STAGES):
        rows = start + tl.arange(0, ROWS)
        if PREFETCH:
            next_rows = rows + GROUPS * ROWS
            hn, rn, dyn, dhn = _load_tile(H, RSTD, DY, DH, next_rows, cols, M, D, HAS_DY, HAS_DH, CACHE)
        else:
            h, rstd, dy, dh = _load_tile(H, RSTD, DY, DH, rows, cols, M, D, HAS_DY, HAS_DH, CACHE)
        normalized = h * rstd[:, None]
        dy_float = dy.to(tl.float32)
        wdy = dy_float * weight[None, :]
        correction = tl.sum(normalized * wdy, axis=1) / D
        dx = (wdy - normalized * correction[:, None]) * rstd[:, None] + dh
        dw += dy_float * normalized
        offsets = rows[:, None] * D + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < D)
        tl.store(DX + offsets, dx, mask)
        if ADD:
            tl.store(DR + offsets, dx, mask)
        if PREFETCH:
            h, rstd, dy, dh = hn, rn, dyn, dhn
    tl.store(PARTIAL + group * D + cols, tl.sum(dw, axis=0), cols < D)


class _PipelineNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, branch, residual, weight, eps, policy):
        rows, groups, warps, stages, prefetch, cache = policy
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
        ctx.policy, ctx.add, ctx.branch_dtype = policy, residual is not None, branch.dtype
        ctx.replicas, ctx.tokens, ctx.stride = replicas, tokens, stride
        ctx.set_materialize_grads(False)
        return hidden, output

    @staticmethod
    def backward(ctx, dh, dy):
        hidden, weight, rstd = ctx.saved_tensors
        dh = dh.contiguous() if dh is not None else None
        dy = dy.contiguous() if dy is not None else None
        rows, group_limit, warps, stages, prefetch, cache = ctx.policy
        width, tokens = hidden.shape[-1], ctx.tokens
        groups = min(triton.cdiv(tokens, rows), group_limit)
        dx = torch.empty_like(hidden, dtype=ctx.branch_dtype)
        dr = torch.empty_like(hidden) if ctx.add else None
        partial = torch.empty((ctx.replicas, groups, width), device=hidden.device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        _norm_bwd_pipeline[(groups, ctx.replicas)](
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
            rows,
            triton.next_power_of_2(width),
            groups,
            stages,
            prefetch,
            cache,
            num_warps=warps,
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


def pipeline_norm(branch, residual, weight, eps=1e-5, *, policy=(4, 256, 4, 2, False, "")):
    """Forward/backward pair with an explicit bounded memory-scheduling policy."""
    rows, groups, warps, stages, prefetch, cache = policy
    _validate_inputs(branch, residual, weight, rows)
    if groups not in (128, 256, 512, 1024) or warps not in (4, 8):
        raise ValueError("unsupported group or warp count")
    if stages not in (1, 2, 3) or not isinstance(prefetch, bool) or cache not in ("", ".ca", ".cg"):
        raise ValueError("unsupported pipeline or cache policy")
    # Keep the FP32 embedding gradient's reduction order unchanged. Its one
    # normalization per model is not the repeated BF16 projection hot path.
    if branch.dtype == torch.float32:
        return residual_norm(branch, residual, weight, eps, rows=rows)
    return _PipelineNorm.apply(branch, residual, weight, eps, policy)


def selected_norm(branch, residual, weight, eps=1e-5, *, rows=4, a100=False):
    """Fixed, measured dispatch; small tensors retain the established kernel."""
    width = branch.shape[-1]
    replicas = weight.shape[0] if weight.ndim == 2 else 1
    tokens = branch.numel() // width // replicas
    # The one-GPU A100 90M screen measured this shape; retain the existing
    # schedules for other shapes and for GH200 (docs/a100-performance.md).
    if a100 and width == 640 and tokens == 32768:
        return pipeline_norm(branch, residual, weight, eps, policy=(4, 1024, 4, 1, True, ""))
    if width == 512:
        return residual_norm(branch, residual, weight, eps, rows=rows)
    if (width == 320 and tokens < 8192) or (width == 640 and tokens < 4096):
        return residual_norm(branch, residual, weight, eps, rows=rows)
    policy = (8, 256, 4, 3, False, "") if width == 320 else (4, 256, 4, 1, True, "")
    return pipeline_norm(branch, residual, weight, eps, policy=policy)
