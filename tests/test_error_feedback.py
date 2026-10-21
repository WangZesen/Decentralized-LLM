import copy
import io
from unittest.mock import patch

import pytest
import torch
from torch import nn

from decentralized_llm.algorithms.adamw import ErrorFeedbackAdamW, make_optimizer, optimizer_report
from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.execution.cuda.adamw import FusedDecentralizedAdamW
from decentralized_llm.execution.qualification import qualification_key
from decentralized_llm.models.llama import Llama, token_losses
from decentralized_llm.models.storage import parameter_storage
from decentralized_llm.training.checkpoints import recipe_identity
from decentralized_llm.tuning.spec import apply_values, numeric_recipe


@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("gamma", [0, 0.5, 1, -1, 2])
def test_configuration_and_identity(tiny_config: Config, cache: TokenCache, mode: str, gamma: float) -> None:
    assert Config().distributed.error_feedback_gamma == 0.0
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    original = recipe_identity(tiny_config, cache)
    original_numeric = numeric_recipe(tiny_config)
    key = qualification_key(tiny_config)
    assert key["error_feedback_gamma"] == 0.0
    tiny_config.distributed.error_feedback_gamma = gamma
    assert (recipe_identity(tiny_config, cache) != original) == (gamma != 0)
    assert (numeric_recipe(tiny_config) != original_numeric) == (gamma != 0)
    assert qualification_key(tiny_config) == {**key, "error_feedback_gamma": gamma}
    path = cache.directory / "feedback.yaml"
    save_config(tiny_config, path)
    assert "error_feedback:" not in path.read_text()
    assert load_config(path).distributed.error_feedback_gamma == gamma
    assert (
        load_config(path, overrides=["distributed.error_feedback_gamma=-0.5"]).distributed.error_feedback_gamma == -0.5
    )
    assert (
        apply_values(tiny_config, {"distributed.error_feedback_gamma": gamma}).distributed.error_feedback_gamma == gamma
    )


@pytest.mark.parametrize("value", ["true", "false", "1", "0"])
def test_removed_flag_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        load_config(overrides=[f"distributed.error_feedback={value}"])


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), float("-inf")])
def test_invalid_gamma_rejected(tiny_config: Config, value: float | bool) -> None:
    with pytest.raises(ValueError, match="error_feedback_gamma"):
        Config.model_validate({"distributed": {"error_feedback_gamma": value}})
    with pytest.raises(ValueError):
        apply_values(tiny_config, {"distributed.error_feedback_gamma": value})
    with pytest.raises(ValueError, match="error_feedback_gamma"):
        tiny_config.distributed.error_feedback_gamma = value


@pytest.mark.parametrize("value", ["true", "false", ".nan", ".inf", "-.inf"])
def test_invalid_gamma_override_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="error_feedback_gamma"):
        load_config(overrides=[f"distributed.error_feedback_gamma={value}"])


@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
def test_disabled_returns_ordinary_adamw(tiny_config: Config, mode: str) -> None:
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    optimizer = make_optimizer(Llama(tiny_config.model), tiny_config, torch.device("cpu"))
    assert type(optimizer) is torch.optim.AdamW
    assert "error_feedback" not in optimizer.state_dict()
    assert optimizer_report(optimizer)["history_bytes"] == optimizer_report(optimizer)["scratch_bytes"] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
def test_full_displacement_and_buffer_reuse(tiny_config: Config, dtype: torch.dtype, mode: str, gamma: float) -> None:
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    model = Llama(tiny_config.model).to(dtype)
    reference = copy.deepcopy(model)
    reference_optimizer = make_optimizer(reference, tiny_config, torch.device("cpu"))
    tiny_config.distributed.error_feedback_gamma = gamma
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    storage = parameter_storage(model)
    parameters = tuple(model.parameters())
    pointers = set()
    if gamma != 0:
        assert isinstance(optimizer, ErrorFeedbackAdamW)
        pointers = {optimizer.previous_update.data_ptr(), optimizer.scratch.data_ptr()}
        assert len(pointers) == 2 and storage.flat.data_ptr() not in pointers
        assert optimizer.previous_update.dtype == dtype
        assert optimizer.previous_update.is_contiguous() and optimizer.scratch.is_contiguous()
        assert optimizer.previous_update.shape == storage.flat.shape
        assert torch.count_nonzero(optimizer.previous_update) == 0
    # A second registered name for tied embedding weights must not double their update.
    model.register_parameter("tied_embedding", model.get_parameter("embedding.weight"))
    assert tuple(model.parameters()) == parameters
    history = [torch.zeros_like(p) for p in reference.parameters()]
    padding = torch.ones_like(storage.flat, dtype=torch.bool)
    for p, offset in zip(parameters, storage.offsets, strict=True):
        padding[offset : offset + p.numel()] = False
    assert padding.any()
    for step, lr in enumerate((0.01, 0.003, 0.0, 0.005)):
        for opt in (optimizer, reference_optimizer):
            for group in opt.param_groups:
                group["lr"] = lr
        with torch.no_grad():
            for index, (actual, expected) in enumerate(zip(parameters, reference.parameters(), strict=True)):
                # Model a change from communication before the local optimizer step.
                actual.mul_(0.9).add_(0.01 * step)
                expected.mul_(0.9).add_(0.01 * step)
                actual.grad = None if step == 1 and index == 0 else torch.full_like(actual, 0.2 + index + step)
                expected.grad = None if actual.grad is None else actual.grad.clone()
            before = [p.clone() for p in reference.parameters()]
        optimizer.step()
        reference_optimizer.step()
        with torch.no_grad():
            for p, base, previous in zip(reference.parameters(), before, history, strict=True):
                delta = p - base
                if gamma != 0:
                    p.add_(delta, alpha=gamma).sub_(previous, alpha=gamma)
                previous.copy_(delta)
        for actual, expected, previous, offset in zip(
            parameters, reference.parameters(), history, storage.offsets, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if isinstance(optimizer, ErrorFeedbackAdamW):
                torch.testing.assert_close(
                    optimizer.previous_update[offset : offset + actual.numel()].view_as(actual),
                    previous,
                    rtol=0,
                    atol=0,
                )
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[actual][key], reference_optimizer.state[expected][key], rtol=0, atol=0
                )
        if isinstance(optimizer, ErrorFeedbackAdamW):
            assert {optimizer.previous_update.data_ptr(), optimizer.scratch.data_ptr()} == pointers
            assert torch.count_nonzero(optimizer.previous_update[padding]) == 0
        assert all(p is original for p, original in zip(model.parameters(), parameters, strict=True))
        assert torch.count_nonzero(storage.flat[padding]) == 0


def test_flat_correction_notifies_autograd(tiny_config: Config) -> None:
    tiny_config.distributed.mode = "decentralized"
    tiny_config.distributed.error_feedback_gamma = 1.0
    model = nn.Linear(3, 2)
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    assert isinstance(optimizer, ErrorFeedbackAdamW)
    optimizer.previous_update.fill_(0.1)
    loss = model(torch.ones(1, 3, requires_grad=True)).square().sum()
    # Isolate flat writes: upstream AdamW would independently increment versions.
    with patch.object(torch.optim.AdamW, "step"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("gamma", [0.5, 1.0, -1.0, 2.0])
def test_state_roundtrip_and_closure(tiny_config: Config, mode: str, gamma: float) -> None:
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    tiny_config.distributed.error_feedback_gamma = gamma
    model = nn.Linear(3, 2)
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    assert isinstance(optimizer, ErrorFeedbackAdamW)
    calls = []

    def closure() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(1, 3)).square().sum()
        loss.backward()
        calls.append(float(loss.detach()))
        return calls[-1]

    assert optimizer.step(closure) == calls[0]
    assert len(calls) == 1
    payload = optimizer.state_dict()
    assert set(payload["error_feedback"]) == {"previous_update"}
    file = io.BytesIO()
    torch.save(payload, file)
    file.seek(0)
    restored_model = copy.deepcopy(model)
    restored = make_optimizer(restored_model, tiny_config, torch.device("cpu"))
    assert isinstance(restored, ErrorFeedbackAdamW)
    pointer = restored.previous_update.data_ptr()
    restored.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
    assert restored.previous_update.data_ptr() == pointer
    for actual, expected in zip(model.parameters(), restored_model.parameters(), strict=True):
        actual.grad = torch.full_like(actual, 0.2)
        expected.grad = actual.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(restored.previous_update, optimizer.previous_update, rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["missing", "empty", "not_tensor", "shape", "dtype"])
def test_invalid_history_rejected(tiny_config: Config, invalid: str) -> None:
    tiny_config.distributed.mode = "decentralized"
    tiny_config.distributed.error_feedback_gamma = 1.0
    optimizer = make_optimizer(nn.Linear(3, 2), tiny_config, torch.device("cpu"))
    assert isinstance(optimizer, ErrorFeedbackAdamW)
    state = optimizer.state_dict()
    if invalid == "missing":
        del state["error_feedback"]
    elif invalid == "empty":
        state["error_feedback"] = {}
    else:
        value = {
            "not_tensor": 0,
            "shape": torch.zeros(1),
            "dtype": optimizer.previous_update.double(),
        }[invalid]
        state["error_feedback"]["previous_update"] = value
    with pytest.raises(ValueError, match="error_feedback previous_update"):
        optimizer.load_state_dict(state)


@pytest.mark.cuda
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("gamma", [0.5, 1.0, -1.0, 2.0])
def test_fused_cuda_and_compilation(tiny_config: Config, compiled: bool, mode: str, gamma: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    tiny_config.runtime.deterministic = False
    tiny_config.runtime.fused_optimizer = True
    torch.manual_seed(42)
    model = Llama(tiny_config.model).cuda()
    reference = copy.deepcopy(model)
    reference_config = tiny_config.model_copy(deep=True)
    reference_config.runtime.decentralized_optimizer = "reference"
    plain = make_optimizer(reference, reference_config, torch.device("cuda"))
    tiny_config.distributed.error_feedback_gamma = gamma
    optimizer = make_optimizer(model, tiny_config, torch.device("cuda"))
    assert isinstance(optimizer, FusedDecentralizedAdamW if mode == "decentralized" else ErrorFeedbackAdamW)
    assert all(group["fused"] for group in optimizer.param_groups)
    forward = torch.compile(model) if compiled else model
    history = [torch.zeros_like(p) for p in reference.parameters()]
    x = torch.tensor([[1, 2, 3, 4]], device="cuda")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        plain.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token_losses(forward(x), x).mean().backward()
        # Use the same production gradients to isolate optimizer arithmetic from
        # eager/compiled BF16 rounding and subsequent trajectory divergence.
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            assert actual.grad is not None
            expected.grad = actual.grad.clone()
        before = [p.detach().clone() for p in reference.parameters()]
        optimizer.step()
        plain.step()
        with torch.no_grad():
            for p, base, previous in zip(reference.parameters(), before, history, strict=True):
                delta = p - base
                p.add_(delta, alpha=gamma).sub_(previous, alpha=gamma)
                previous.copy_(delta)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Check CPU-loaded history is copied back to the CUDA allocation.
    state = optimizer.state_dict()
    state["error_feedback"]["previous_update"] = (
        optimizer.previous_update.cpu() if optimizer.previous_update is not None else None
    )
    optimizer.load_state_dict(state)
    assert optimizer.previous_update is not None and optimizer.previous_update.is_cuda
