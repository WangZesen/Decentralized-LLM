"""Independent reference comparisons for batching, replay, and canonical state."""

import copy
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from test_muon import test_training_resume_and_metrics as check_resume

from decentralized_llm.algorithms.adamw import make_optimizer
from decentralized_llm.algorithms.batched_muon import BatchedMuon, Mode
from decentralized_llm.algorithms.muon import MuonAdamW
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.execution.qualification import qualification_key
from decentralized_llm.models.llama import Llama
from decentralized_llm.training.checkpoints import recipe_identity
from decentralized_llm.tuning.spec import numeric_recipe


def exercise(mode: Mode, device: str, dtype: torch.dtype, momentum: float, nesterov: bool) -> None:
    torch.manual_seed(42)
    shapes = [(16, 16), (16, 16), (32, 16), (16, 32)]
    reference = [torch.nn.Parameter(torch.randn(s, device=device, dtype=dtype) * 0.02) for s in shapes]
    actual = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    kwargs: dict[str, Any] = dict(
        lr=0.001, momentum=momentum, nesterov=nesterov, weight_decay=0.1, adjust_lr_fn="match_rms_adamw"
    )
    expected_optimizer = torch.optim.Muon(reference, **kwargs)
    optimizer = BatchedMuon(actual, mode=mode, **kwargs)
    for step, lr in enumerate((0.001, 0.0003, 0.0, 0.001, 0.0002)):
        for group in (expected_optimizer.param_groups[0], optimizer.param_groups[0]):
            group["lr"] = lr
        for index, (expected, candidate) in enumerate(zip(reference, actual, strict=True)):
            # Changing gradient objects, missing gradients, zero and sub-epsilon
            # gradients detect stale capture inputs and normalization across matrices.
            gradient = torch.randn_like(expected) * (0.0 if step == 0 else 10.0 ** (-9 + index * 3))
            expected.grad = None if step == 2 and index == 1 else gradient
            candidate.grad = None if expected.grad is None else gradient.clone()
        expected_optimizer.step()
        optimizer.step()
        for expected, candidate in zip(reference, actual, strict=True):
            torch.testing.assert_close(candidate, expected, rtol=2e-4, atol=2e-5 if device == "cuda" else 5e-7)
            torch.testing.assert_close(
                optimizer.state[candidate]["momentum_buffer"],
                expected_optimizer.state[expected]["momentum_buffer"],
                rtol=1e-6,
                atol=1e-9,
            )
        if step == 3:
            saved = copy.deepcopy(optimizer.state_dict())
            optimizer.load_state_dict(saved)
            assert not optimizer._plans
    # State remains loadable by the upstream implementation, with no batch axis.
    restored = torch.optim.Muon(actual, **kwargs)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for parameter in actual:
        torch.testing.assert_close(
            restored.state[parameter]["momentum_buffer"], optimizer.state[parameter]["momentum_buffer"]
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("momentum", [0.0, 0.8, 0.95, 0.986150329547211])
@pytest.mark.parametrize("nesterov", [True, False])
def test_batched_reference(dtype: torch.dtype, momentum: float, nesterov: bool) -> None:
    torch.set_num_threads(1)
    exercise("batched", "cpu", dtype, momentum, nesterov)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["batched", "compiled", "cudagraph"])
def test_cuda_reference_and_replay(mode: Mode) -> None:
    exercise(mode, "cuda", torch.float32, 0.95, True)


def test_execution_selector_and_checkpoint_compatibility(tiny_config: Config, cache: TokenCache):
    cfg = tiny_config
    original = numeric_recipe(cfg)
    identity = recipe_identity(cfg, cache)
    qualification = qualification_key(cfg)
    cfg.runtime.muon_optimizer = "batched"
    assert numeric_recipe(cfg) == original
    assert recipe_identity(cfg, cache) == identity
    assert qualification_key(cfg) == qualification
    cfg.optimizer.name = "muon_adamw"
    optimized = make_optimizer(Llama(cfg.model), cfg, torch.device("cpu"))
    assert isinstance(optimized, MuonAdamW)
    assert optimized.implementation == "batched_muon_adamw"
    optimized_identity = recipe_identity(cfg, cache)
    optimized_recipe = numeric_recipe(cfg)
    cfg.runtime.muon_optimizer = "reference"
    assert recipe_identity(cfg, cache) == optimized_identity
    assert numeric_recipe(cfg) != optimized_recipe
    assert "muon_optimizer" not in numeric_recipe(cfg)["runtime"]


def test_default_mode_and_recipe_identity(tiny_config: Config) -> None:
    cfg = tiny_config
    cfg.optimizer.name = "muon_adamw"
    assert cfg.runtime.muon_optimizer == "auto"
    optimizer = make_optimizer(Llama(cfg.model), cfg, torch.device("cpu"))
    assert isinstance(optimizer, MuonAdamW)
    assert optimizer.implementation == "torch_muon_adamw"
    assert "muon_optimizer" not in numeric_recipe(cfg)["runtime"]

    cfg.runtime.device = "cuda"
    default_recipe, default_qualification = numeric_recipe(cfg), qualification_key(cfg)
    assert default_recipe["runtime"]["muon_optimizer"] == "cudagraph"
    cfg.runtime.muon_optimizer = "cudagraph"
    assert numeric_recipe(cfg) == default_recipe
    assert qualification_key(cfg) == default_qualification
    cfg.runtime.muon_optimizer = "reference"
    assert numeric_recipe(cfg) != default_recipe
    assert qualification_key(cfg) != default_qualification


@pytest.mark.parametrize("mode", ["compiled", "cudagraph"])
def test_accelerated_modes_require_cuda(mode: Mode) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        BatchedMuon([torch.nn.Parameter(torch.zeros(4, 4))], mode=mode)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_parameters_rejected(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="FP32 or FP64"):
        BatchedMuon([torch.nn.Parameter(torch.zeros(4, 4, dtype=dtype))], mode="batched")


@pytest.mark.parametrize("backend", ["distributed", "packed"])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
def test_batched_hybrid_resume(
    tiny_config: Config,
    cache: TokenCache,
    tmp_path: Path,
    backend: Literal["distributed", "packed"],
    mode: Literal["ddp", "decentralized"],
) -> None:
    tiny_config.runtime.muon_optimizer = "batched"
    check_resume(tiny_config, cache, tmp_path, backend, mode, "awc", 0.5)
