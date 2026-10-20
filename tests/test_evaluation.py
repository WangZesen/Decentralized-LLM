import copy

import numpy as np
import pytest
import torch
from torch import nn

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config
from decentralized_llm.evaluation import evaluate, parameter_distance
from decentralized_llm.model import Llama, token_losses
from decentralized_llm.optim import make_optimizer
from decentralized_llm.parameter_storage import prepare_parameter_storage
from decentralized_llm.runtime import rng_state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_parameter_distance_counts_ties_once_and_ignores_buffers_and_padding(dtype: torch.dtype) -> None:
    model = nn.Module()
    weight = nn.Parameter(torch.tensor([3, 0], dtype=dtype))
    model.register_parameter("weight", weight)
    model.register_parameter("bias", nn.Parameter(torch.tensor([4], dtype=dtype)))
    model.register_parameter("tied_weight", weight)
    model.register_buffer("buffer", torch.tensor([1000], dtype=dtype))
    reference = copy.deepcopy(model).float()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.zero_()
        reference.get_buffer("buffer").zero_()
        storage = prepare_parameter_storage(model)
        reference_storage = prepare_parameter_storage(reference)
        storage.flat[2 : storage.offsets[1]].fill_(100)
        reference_storage.flat[2 : reference_storage.offsets[1]].fill_(-100)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
    before = storage.flat.clone()
    reference_before = reference_storage.flat.clone()
    distance = parameter_distance(model, reference)
    assert distance.dtype == torch.float64 and not distance.requires_grad
    assert distance.item() == 5
    assert parameter_distance(model, model).item() == 0
    torch.testing.assert_close(storage.flat, before, rtol=0, atol=0)
    torch.testing.assert_close(reference_storage.flat, reference_before, rtol=0, atol=0)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)


def test_parameter_distance_squares_in_fp64() -> None:
    model = nn.Linear(1, 1, bias=False)
    reference = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(3e20)
        reference.weight.zero_()
    assert parameter_distance(model, reference).item() == pytest.approx(3e20)


@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_single_worker_consensus_and_training_state(
    cache: TokenCache, tiny_config: Config, mode: str, average: bool, training: bool
) -> None:
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    model = Llama(tiny_config.model, "reference")
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    tokens = torch.tensor([[1, 2, 3, 4]])
    token_losses(model(tokens), tokens).mean().backward()
    optimizer.step()
    model.train(training)
    weights = copy.deepcopy(model.state_dict())
    gradients = [parameter.grad.clone() if parameter.grad is not None else None for parameter in model.parameters()]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    random_state = rng_state()

    result = evaluate(model, cache, tiny_config, torch.device("cpu"), full=True, average=average)

    assert result["consensus_error"] == 0.0
    assert result["tokens"] == 38
    assert model.training is training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == optimizer_state["param_groups"]
    for key, state in optimizer_state["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][key][name], value, rtol=0, atol=0)
    actual_random = rng_state()
    assert actual_random["python"] == random_state["python"]
    np.testing.assert_equal(actual_random["numpy"], random_state["numpy"])
    torch.testing.assert_close(actual_random["torch"], random_state["torch"], rtol=0, atol=0)
