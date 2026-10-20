# pyright: reportMissingParameterType=false
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportPossiblyUnboundVariable=false, reportIncompatibleMethodOverride=false
"""Packed-QKV half-split RoPE with a fused backward scatter, head dimension 64.

BF16 multiply/add boundaries are explicit. V remains a view in forward, while
backward accepts SDPA's gradient strides and writes packed Q/K/V directly.
"""

import torch

from .kernels import jit, tl, triton

if triton is not None:

    @jit
    def _bf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @jit
    def _rope_fwd(
        X,
        COS,
        SIN,
        Q,
        K,
        V,
        N: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        BLOCK: tl.constexpr,
        COPY_V: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = i % 32
        head = (i // 32) % H
        row = i // (H * 32)
        seq, batch = row % S, row // S
        packed = row * (3 * H * 64) + head * 64 + col
        output = ((batch * H + head) * S + seq) * 64 + col
        co = _bf16(tl.load(COS + seq * 32 + col, i < N, other=0))
        si = _bf16(tl.load(SIN + seq * 32 + col, i < N, other=0))
        ql = tl.load(X + packed, i < N, other=0).to(tl.float32)
        qr = tl.load(X + packed + 32, i < N, other=0).to(tl.float32)
        kl = tl.load(X + packed + H * 64, i < N, other=0).to(tl.float32)
        kr = tl.load(X + packed + H * 64 + 32, i < N, other=0).to(tl.float32)
        if COPY_V:
            vl = tl.load(X + packed + 2 * H * 64, i < N, other=0)
            vr = tl.load(X + packed + 2 * H * 64 + 32, i < N, other=0)
        tl.store(Q + output, _bf16(ql * co) - _bf16(qr * si), i < N)
        tl.store(Q + output + 32, _bf16(ql * si) + _bf16(qr * co), i < N)
        tl.store(K + output, _bf16(kl * co) - _bf16(kr * si), i < N)
        tl.store(K + output + 32, _bf16(kl * si) + _bf16(kr * co), i < N)
        if COPY_V:
            tl.store(V + output, vl, i < N)
            tl.store(V + output + 32, vr, i < N)

    @jit
    def _rope_bwd(
        DQ,
        DK,
        DV,
        COS,
        SIN,
        DX,
        N: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        QS: tl.constexpr,
        KS: tl.constexpr,
        VS: tl.constexpr,
        HAS_Q: tl.constexpr,
        HAS_K: tl.constexpr,
        HAS_V: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = i % 32
        head = (i // 32) % H
        row = i // (H * 32)
        seq, batch = row % S, row // S
        packed = row * (3 * H * 64) + head * 64 + col
        co = _bf16(tl.load(COS + seq * 32 + col, i < N, other=0))
        si = _bf16(tl.load(SIN + seq * 32 + col, i < N, other=0))
        if HAS_Q:
            qi = batch * QS[0] + head * QS[1] + seq * QS[2] + col * QS[3]
            ql = tl.load(DQ + qi, i < N, other=0).to(tl.float32)
            qr = tl.load(DQ + qi + 32 * QS[3], i < N, other=0).to(tl.float32)
            qdxl = _bf16(ql * co) + _bf16(qr * si)
            qdxr = _bf16(qr * co) - _bf16(ql * si)
        else:
            qdxl, qdxr = 0.0, 0.0
        if HAS_K:
            ki = batch * KS[0] + head * KS[1] + seq * KS[2] + col * KS[3]
            kl = tl.load(DK + ki, i < N, other=0).to(tl.float32)
            kr = tl.load(DK + ki + 32 * KS[3], i < N, other=0).to(tl.float32)
            kdxl = _bf16(kl * co) + _bf16(kr * si)
            kdxr = _bf16(kr * co) - _bf16(kl * si)
        else:
            kdxl, kdxr = 0.0, 0.0
        if HAS_V:
            vi = batch * VS[0] + head * VS[1] + seq * VS[2] + col * VS[3]
            vl = tl.load(DV + vi, i < N, other=0)
            vr = tl.load(DV + vi + 32 * VS[3], i < N, other=0)
        else:
            vl, vr = 0.0, 0.0
        tl.store(DX + packed, qdxl, i < N)
        tl.store(DX + packed + 32, qdxr, i < N)
        tl.store(DX + packed + H * 64, kdxl, i < N)
        tl.store(DX + packed + H * 64 + 32, kdxr, i < N)
        tl.store(DX + packed + 2 * H * 64, vl, i < N)
        tl.store(DX + packed + 2 * H * 64 + 32, vr, i < N)


class _PackedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, cos, sin, heads, copy_value):
        batch, length, _ = packed.shape
        shape = (batch, heads, length, 64)
        q, k = (torch.empty(shape, device=packed.device, dtype=packed.dtype) for _ in range(2))
        v = torch.empty_like(q) if copy_value else None
        n = batch * length * heads * 32
        _rope_fwd[(triton.cdiv(n, 256),)](
            packed,
            cos,
            sin,
            q,
            k,
            v,
            n,
            heads,
            length,
            256,
            copy_value,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if not copy_value:
            v = packed.view(batch, length, 3, heads, 64)[:, :, 2].transpose(1, 2)
        ctx.save_for_backward(cos, sin)
        ctx.shape, ctx.heads = packed.shape, heads
        ctx.set_materialize_grads(False)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin = ctx.saved_tensors
        batch, length, _ = ctx.shape
        gradient = torch.empty(ctx.shape, device=cos.device, dtype=torch.bfloat16)
        n = batch * length * ctx.heads * 32
        _rope_bwd[(triton.cdiv(n, 256),)](
            dq,
            dk,
            dv,
            cos,
            sin,
            gradient,
            n,
            ctx.heads,
            length,
            dq.stride() if dq is not None else (0, 0, 0, 0),
            dk.stride() if dk is not None else (0, 0, 0, 0),
            dv.stride() if dv is not None else (0, 0, 0, 0),
            dq is not None,
            dk is not None,
            dv is not None,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return gradient, None, None, None, None


def packed_rope(packed, cos, sin, heads, *, copy_value=False):
    if triton is None:
        raise RuntimeError("packed RoPE requires Triton")
    if packed.dtype != torch.bfloat16 or not packed.is_cuda or not packed.is_contiguous():
        raise ValueError("packed RoPE requires contiguous CUDA BF16 QKV")
    if packed.ndim != 3 or packed.numel() == 0 or heads not in (5, 10) or packed.shape[-1] != 3 * heads * 64:
        raise ValueError("packed RoPE requires 5/10 heads of dimension 64")
    if any(
        t.device != packed.device or not t.is_contiguous() or t.dtype != torch.float32 or t.shape != cos.shape
        for t in (cos, sin)
    ):
        raise ValueError("RoPE tables must be matching contiguous FP32 on the QKV device")
    if cos.ndim != 2 or cos.shape[1] != 32 or cos.shape[0] < packed.shape[1]:
        raise ValueError("RoPE tables do not cover the sequence")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("RoPE tables must be constants")
    return _PackedRoPE.apply(packed, cos, sin, heads, copy_value)
