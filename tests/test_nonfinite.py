import json
from typing import Literal

import pytest
import torch

from decentralized_llm.algorithms.packed_adamw import PackedAdamW
from decentralized_llm.backends.factory import create_backend
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.execution.native import TrainingExecution
from decentralized_llm.runtime import NonfiniteUpdateError, clip_grad_norm_
from decentralized_llm.training import loop


@pytest.mark.parametrize("maximum", [None, 1.0])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_gradient_norm_is_typed(maximum: float | None, bad_value: float) -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([bad_value, 1.0])
    with pytest.raises(NonfiniteUpdateError, match="gradient norm"):
        clip_grad_norm_(iter([parameter]), maximum)
    assert torch.equal(parameter, torch.ones(2))
    assert parameter.grad[1] == 1.0


@pytest.mark.parametrize("maximum", [None, 1.0])
def test_finite_clipping_preserves_pytorch_behavior(maximum: float | None) -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    reference = torch.nn.Parameter(torch.ones(2))
    parameter.grad = reference.grad = torch.tensor([3.0, 4.0])
    reference.grad = reference.grad.clone()
    norm = clip_grad_norm_([parameter], maximum)
    assert norm == 5.0
    if maximum is not None:
        torch.nn.utils.clip_grad_norm_([reference], maximum, error_if_nonfinite=True)
    torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("backend_name", ["distributed", "packed"])
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon_adamw"])
@pytest.mark.parametrize("bad_value", ["gradient", "loss", "generic"])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
def test_rejected_training_update_preserves_progress_and_records_failure(
    cache: TokenCache,
    tiny_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: Literal["distributed", "packed"],
    bad_value: str,
    mode: Literal["ddp", "decentralized"],
    optimizer_name: Literal["adamw", "muon_adamw"],
) -> None:
    tiny_config.optimizer.name = optimizer_name
    tiny_config.runtime.training_backend = backend_name
    tiny_config.distributed.mode = mode
    tiny_config.distributed.error_feedback_gamma = 0.5
    if backend_name == "packed":
        tiny_config.distributed.gpus_per_node = 2
    backend = create_backend(tiny_config, torch.device("cpu"))
    before = {key: value.clone() for key, value in backend.model.state_dict().items()}
    monkeypatch.setattr(loop, "create_backend", lambda *args: backend)
    forward = TrainingExecution.forward

    def broken_forward(self: TrainingExecution, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if bad_value == "generic":
            raise RuntimeError("nonfinite mentioned in an unrelated error")
        losses = forward(self, x, y)
        # Adding infinity leaves gradients finite, exercising the separate loss guard.
        return losses + float("inf") if bad_value == "loss" else losses * float("nan")

    monkeypatch.setattr(TrainingExecution, "forward", broken_forward)
    expected = RuntimeError if bad_value == "generic" else NonfiniteUpdateError
    with pytest.raises(expected, match="nonfinite"):
        loop.train(tiny_config)
    for key, value in backend.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    optimizer = backend.optimizer
    if isinstance(optimizer, PackedAdamW):
        for local in optimizer.optimizers:
            for state in local.state.values():
                assert all(torch.count_nonzero(value) == 0 for value in state.values())
    else:
        assert not optimizer.state
    assert isinstance(optimizer.previous_update, torch.Tensor)
    assert torch.count_nonzero(optimizer.previous_update) == 0
    output = tiny_config.runtime.output_dir
    failure = json.loads((output / "failure-rank-00000.json").read_text())
    assert "nonfinite" in failure["traceback"]
    if bad_value == "generic":
        assert "kind" not in failure
    else:
        assert failure["kind"] == "nonfinite_update" and failure["rank"] == 0
        assert failure["progress"] == {"cursor": 0, "step": 0, "completed_epochs": 0}
    assert not (output / "result.json").exists()
    assert not (output / "metrics-rank-00000.jsonl").exists()
    assert (output / "run.json").exists() and (output / "config.yaml").exists()


def test_fused_inspection_distinguishes_nonfinite_from_counter_mismatch() -> None:
    from decentralized_llm.execution.cuda.packed_optimizer import ArenaAdamW

    optimizer = ArenaAdamW.__new__(ArenaAdamW)
    optimizer.finite = torch.tensor(0)
    optimizer.completed = torch.tensor(3)
    with pytest.raises(NonfiniteUpdateError, match="3 successful updates"):
        optimizer.inspect(4)
    optimizer.finite.fill_(1)
    with pytest.raises(RuntimeError, match="counter mismatch"):
        optimizer.inspect(4)
    assert optimizer.inspect(3) == 3
