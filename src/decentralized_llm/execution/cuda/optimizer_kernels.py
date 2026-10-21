# pyright: reportMissingParameterType=false, reportArgumentType=false
"""Grouped FP32 AdamW kernels, independent of the model execution backend.

Only pointer metadata crosses from the host; gradients and moments stay in their
original allocations. Explicit FMA matches PyTorch's fused moment arithmetic.
"""

from typing import Any, cast

import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

jit = cast(Any, triton.jit)
BLOCK = 2048
WARPS = 4
CACHE = ""


@jit
def _prepare(META, COEFFICIENTS, lr, beta1, beta2, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # An empty gradient can have address zero; its scalar step still exists.
    active = (i < N) & (tl.load(META + i * 6 + 5, i < N, other=0) != 0)
    step_ptr = tl.load(META + i * 6 + 5, i < N, other=0).to(tl.pointer_type(tl.float32))
    step = tl.load(step_ptr, active, other=0) + 1.0
    tl.store(step_ptr, step, active)
    tl.store(COEFFICIENTS + i * 2, tl.div_rn(lr, 1.0 - libdevice.pow(beta1, step)), i < N)
    tl.store(COEFFICIENTS + i * 2 + 1, libdevice.sqrt_rn(1.0 - libdevice.pow(beta2, step)), i < N)


@jit
def _update(
    WEIGHTS,
    BASE,
    HISTORY,
    META,
    TILES,
    COEFFICIENTS,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    base_scale,
    output_scale,
    FEEDBACK_GAMMA: tl.constexpr,
    ALIGNED: tl.constexpr,
    CACHE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    tensor = tl.load(TILES + tile * 2)
    start = tl.multiple_of(tl.load(TILES + tile * 2 + 1), BLOCK)
    offset = tl.load(META + tensor * 6)
    length = tl.load(META + tensor * 6 + 1)
    grad_address = tl.load(META + tensor * 6 + 2)
    if ALIGNED:
        offset = tl.multiple_of(offset, 4)
        length = tl.multiple_of(length, 4)
        grad_address = tl.multiple_of(grad_address, 16)
    index = start + tl.arange(0, BLOCK)
    mask = index < length
    base = tl.load(BASE + offset + index, mask, other=0, cache_modifier=CACHE) * base_scale
    q = base
    if grad_address != 0:
        grad_ptr = grad_address.to(tl.pointer_type(tl.float32))
        m_address = tl.load(META + tensor * 6 + 3)
        v_address = tl.load(META + tensor * 6 + 4)
        if ALIGNED:
            m_address = tl.multiple_of(m_address, 16)
            v_address = tl.multiple_of(v_address, 16)
        m_ptr = m_address.to(tl.pointer_type(tl.float32))
        v_ptr = v_address.to(tl.pointer_type(tl.float32))
        if ALIGNED:
            grad_ptr = tl.multiple_of(grad_ptr, 16)
            m_ptr = tl.multiple_of(m_ptr, 16)
            v_ptr = tl.multiple_of(v_ptr, 16)
        g = tl.load(grad_ptr + index, mask, other=0, cache_modifier=CACHE)
        m = tl.load(m_ptr + index, mask, other=0, cache_modifier=CACHE)
        v = tl.load(v_ptr + index, mask, other=0, cache_modifier=CACHE)
        m = tl.fma(beta1, m, tl.fma(-beta1, g, g))
        square = g * g
        v = tl.fma(beta2, v, tl.fma(-beta2, square, square))
        step_size = tl.load(COEFFICIENTS + tensor * 2)
        bc2_sqrt = tl.load(COEFFICIENTS + tensor * 2 + 1)
        denom = libdevice.sqrt_rn(v) / bc2_sqrt + eps
        q = tl.fma(-lr * weight_decay, base, base)
        q = q - (step_size * m) / denom
        tl.store(m_ptr + index, m, mask)
        tl.store(v_ptr + index, v, mask)
    if FEEDBACK_GAMMA != 0:
        delta = q - base
        previous = tl.load(HISTORY + offset + index, mask, other=0, cache_modifier=CACHE)
        q = (q + FEEDBACK_GAMMA * delta) - FEEDBACK_GAMMA * previous
        tl.store(HISTORY + offset + index, delta, mask)
    tl.store(WEIGHTS + offset + index, q * output_scale, mask)


def update(
    weights: Tensor,
    base: Tensor,
    history: Tensor | None,
    metadata: Tensor,
    tiles: Tensor,
    coefficients: Tensor,
    group: dict[str, Any],
    base_divisor: int,
    output_divisor: int,
    aligned: bool,
    feedback_gamma: float,
) -> None:
    beta1, beta2 = map(float, group["betas"])
    _prepare[(triton.cdiv(metadata.shape[0], 128),)](
        metadata,
        coefficients,
        float(group["lr"]),
        beta1,
        beta2,
        metadata.shape[0],
        128,
        enable_fp_fusion=False,
    )
    if not tiles.shape[0]:
        return
    _update[(tiles.shape[0],)](
        weights,
        base,
        history if history is not None else weights,
        metadata,
        tiles,
        coefficients,
        float(group["lr"]),
        beta1,
        beta2,
        float(group["weight_decay"]),
        float(group["eps"]),
        1.0 / base_divisor,
        1.0 / output_divisor,
        feedback_gamma,
        aligned and base.data_ptr() % 16 == 0,
        CACHE,
        BLOCK,
        num_warps=WARPS,
        enable_fp_fusion=False,
    )
