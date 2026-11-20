"""Numerical checks for the 10M model's non-power-of-two fused kernel dimensions."""

import pytest
import torch
from torch.nn import functional as F

from decentralized_llm.config import PRESETS, ModelConfig
from decentralized_llm.models.llama import Attention

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize(
    "replicas,dtype,add", [(1, torch.float32, False), (1, torch.bfloat16, True), (4, torch.bfloat16, True)]
)
def test_10m_norm_forward_and_gradients(replicas: int, dtype: torch.dtype, add: bool) -> None:
    from decentralized_llm.execution.cuda.pipeline import selected_norm

    torch.manual_seed(42)
    # An incomplete final row tile exercises masks; packed weights exercise worker strides.
    shape = (33, 192) if replicas == 1 else (replicas, 33, 192)
    branch = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    residual = torch.randn(shape, device="cuda", requires_grad=True) if add else None
    weight = torch.randn((192,) if replicas == 1 else (replicas, 192), device="cuda", requires_grad=True)
    hidden = branch.float() if residual is None else branch.float() + residual
    scale = weight if replicas == 1 else weight[:, None, :]
    expected = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-5) * scale
    expected = expected.bfloat16()
    actual_hidden, actual = selected_norm(branch, residual, weight)
    torch.testing.assert_close(actual_hidden, hidden, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
    outputs = (torch.randn_like(hidden), torch.randn_like(expected))
    inputs = (branch, weight) if residual is None else (branch, residual, weight)
    reference_gradients = torch.autograd.grad((hidden, expected), inputs, outputs)
    fused_gradients = torch.autograd.grad((actual_hidden, actual), inputs, outputs)
    for fused, reference in zip(fused_gradients, reference_gradients, strict=True):
        relative_error = (fused.float() - reference.float()).norm() / reference.float().norm()
        assert float(relative_error) < 0.005


def test_10m_swiglu_forward_and_gradients() -> None:
    from decentralized_llm.execution.cuda.swiglu import paired_swiglu

    torch.manual_seed(42)
    # 544 does not divide the 1024-element tile, so loads cross gate/up rows.
    packed = torch.randn((2, 17, 1088), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate, up = packed.float().chunk(2, dim=-1)
    expected = (F.silu(gate) * up).bfloat16()
    actual = paired_swiglu(packed)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
    gradient = torch.randn_like(expected)
    (reference_gradient,) = torch.autograd.grad(expected, packed, gradient)
    (fused_gradient,) = torch.autograd.grad(actual, packed, gradient)
    relative_error = (fused_gradient.float() - reference_gradient.float()).norm() / reference_gradient.float().norm()
    assert float(relative_error) < 0.005


@pytest.mark.parametrize("copy_value", [False, True])
@pytest.mark.parametrize("active", [(0, 1, 2), (0,), (2,)])
def test_10m_rope_strided_and_missing_gradients(copy_value: bool, active: tuple[int, ...]) -> None:
    from decentralized_llm.execution.cuda.rope import packed_rope

    torch.manual_seed(42)
    config = ModelConfig(**PRESETS["10m"], context_length=17)
    attention = Attention(config, "sdpa").cuda()
    packed = torch.randn((2, 17, 576), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q, k, v = packed.view(2, 17, 3, 3, 64).unbind(2)
    expected = (attention._rotary(q.transpose(1, 2)), attention._rotary(k.transpose(1, 2)), v.transpose(1, 2))
    actual = packed_rope(packed, attention._rope_cos, attention._rope_sin, 3, copy_value=copy_value)
    for fused, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    gradients = tuple(torch.randn((2, 17, 3, 64), device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in active)
    (reference_gradient,) = torch.autograd.grad(tuple(expected[i] for i in active), packed, gradients)
    (fused_gradient,) = torch.autograd.grad(tuple(actual[i] for i in active), packed, gradients)
    torch.testing.assert_close(fused_gradient, reference_gradient, rtol=0, atol=0)
