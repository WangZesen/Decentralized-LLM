# pyright: reportMissingParameterType=false
# pyright: reportIncompatibleMethodOverride=false, reportArgumentType=false
"""Masked per-token CE without materializing an FP32 vocabulary-sized tensor."""

import torch

from .kernels import _ce_forward, jit, tl, triton


@jit
def _ce_backward(X, Y, LSE, DOUT, DX, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * V + col, col < V, other=-float("inf")).to(tl.float32)
    target = tl.load(Y + row)
    gradient = (tl.exp(x - tl.load(LSE + row)) - (col == target).to(tl.float32)) * tl.load(DOUT + row)
    tl.store(DX + row * V + col, tl.where(target == -100, 0.0, gradient), col < V)


class _TokenCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        rows, vocab = targets.numel(), logits.shape[-1]
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty_like(losses)
        _ce_forward[(rows,)](logits, targets, losses, lse, vocab, triton.next_power_of_2(vocab), num_warps=32)
        ctx.save_for_backward(logits, targets, lse)
        return losses.view_as(targets)

    @staticmethod
    def backward(ctx, dout):
        logits, targets, lse = ctx.saved_tensors
        dx = torch.empty_like(logits)
        vocab = logits.shape[-1]
        _ce_backward[(targets.numel(),)](
            logits, targets, lse, dout.contiguous(), dx, vocab, triton.next_power_of_2(vocab), num_warps=32
        )
        return dx, None


def token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not logits.is_cuda or logits.dtype != torch.bfloat16:
        raise ValueError("GH200 CE requires CUDA BF16 logits")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and target shapes differ")
    return _TokenCE.apply(logits.contiguous(), targets.contiguous())
