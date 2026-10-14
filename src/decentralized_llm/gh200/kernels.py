# pyright: reportMissingParameterType=false
"""GH200 cross-entropy forward, adapted from the frozen optimized backend."""

from typing import Any, cast

import triton
import triton.language as tl

# Triton launch syntax is dynamically specialized; its Python signature is not a call ABI.
jit = cast(Any, triton.jit)


@jit
def _ce_forward(X, Y, LOSS, LSE, V: tl.constexpr, BLOCK: tl.constexpr):
    # Packed-8, local batch 16, context 1024 exceeds 2**31 logit elements.
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + row * V + j, j < V, other=-float("inf")).to(tl.float32)
    target = tl.load(Y + row)
    maximum = tl.max(x, axis=0)
    lse = maximum + tl.log(tl.sum(tl.exp(x - maximum), axis=0))
    chosen = tl.load(X + row * V + target, (target >= 0) & (target < V), other=0)
    valid = target != -100
    tl.store(LOSS + row, tl.where(valid, lse - chosen.to(tl.float32), 0.0))
    tl.store(LSE + row, lse)
