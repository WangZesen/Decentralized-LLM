import copy
import gc
import io
import weakref
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from decentralized_llm import distributed
from decentralized_llm.config import Config
from decentralized_llm.execution import build_execution
from decentralized_llm.model import Llama, token_losses
from decentralized_llm.optim import make_optimizer
from decentralized_llm.parameter_storage import PARAMETER_ALIGNMENT, parameter_storage, prepare_parameter_storage


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_aligned_storage_preserves_parameters_and_ties(dtype: torch.dtype) -> None:
    first, second = nn.Linear(3, 5), nn.Linear(5, 3)
    model = nn.Sequential(first, second).to(dtype=dtype)
    model.register_parameter("tied", first.get_parameter("weight"))
    second.bias.requires_grad_(False)
    parameters = tuple(model.parameters())
    values = [p.detach().clone() for p in parameters]
    names = tuple(model.state_dict())
    rng = torch.get_rng_state()
    storage = prepare_parameter_storage(model)
    assert prepare_parameter_storage(model) is storage
    assert tuple(model.state_dict()) == names
    assert model.get_parameter("tied") is first.weight
    assert not second.bias.requires_grad
    assert torch.equal(rng, torch.get_rng_state())
    assert storage.flat.data_ptr() % PARAMETER_ALIGNMENT == 0
    used = torch.zeros_like(storage.flat, dtype=torch.bool)
    for p, saved, original, offset in zip(model.parameters(), values, parameters, storage.offsets, strict=True):
        assert p is original and p.is_leaf
        assert p.data_ptr() % PARAMETER_ALIGNMENT == 0
        assert p.untyped_storage().data_ptr() == storage.flat.untyped_storage().data_ptr()
        torch.testing.assert_close(p, saved, rtol=0, atol=0)
        used[offset : offset + p.numel()] = True
    assert torch.count_nonzero(storage.flat[~used]) == 0
    storage.copy_from(torch.ones_like(storage.flat))
    for p in parameters:
        torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)
    with torch.no_grad():
        parameters[0].fill_(2)
    assert bool((storage.flat[: parameters[0].numel()] == 2).all())


def test_storage_lifetime_and_deepcopy() -> None:
    model = nn.Linear(3, 2)
    storage = prepare_parameter_storage(model)
    replica = copy.deepcopy(model)
    copied = prepare_parameter_storage(replica)
    assert copied.flat.data_ptr() != storage.flat.data_ptr()
    copied.copy_from(torch.zeros_like(copied.flat))
    assert bool(model.weight.ne(0).any())
    model_ref, storage_ref = weakref.ref(model), weakref.ref(storage)
    del model, storage
    gc.collect()
    assert model_ref() is None and storage_ref() is None


def test_storage_reprepared_after_conversion_and_replacement() -> None:
    model = nn.Linear(3, 2)
    first = prepare_parameter_storage(model)
    values = [p.detach().clone().double() for p in model.parameters()]
    model.double()
    second = prepare_parameter_storage(model)
    assert second is not first and second.flat.dtype == torch.float64
    for actual, expected in zip(model.parameters(), values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.bias = nn.Parameter(torch.ones(2, dtype=torch.float64))
    third = prepare_parameter_storage(model)
    assert third is not second
    third.copy_from(torch.zeros_like(third.flat))
    assert torch.count_nonzero(model.bias) == 0


@pytest.mark.parametrize("unsupported", ["mixed_dtype", "noncontiguous", "sparse", "overlap", "empty"])
def test_unsupported_storage_is_rejected_before_rebinding(unsupported: str) -> None:
    model = nn.Module()
    first = nn.Parameter(torch.randn(2, 3))
    model.register_parameter("first", first)
    pointer = first.data_ptr()
    if unsupported == "mixed_dtype":
        model.register_parameter("second", nn.Parameter(torch.ones(2, dtype=torch.float64)))
    elif unsupported == "noncontiguous":
        model.register_parameter("second", nn.Parameter(torch.randn(2, 3).t()))
    elif unsupported == "sparse":
        model.register_parameter("second", nn.Parameter(torch.eye(2).to_sparse()))
    elif unsupported == "overlap":
        model.register_parameter("second", nn.Parameter(first.detach()))
    else:
        model = nn.Module()
    with pytest.raises(ValueError):
        prepare_parameter_storage(model)
    if unsupported != "empty":
        assert model.get_parameter("first").data_ptr() == pointer


@pytest.mark.parametrize("first", ["optimizer", "execution"])
def test_training_entrypoints_share_storage(tiny_config: Config, first: str) -> None:
    model = Llama(tiny_config.model)
    parameters = tuple(model.parameters())
    device = torch.device("cpu")
    if first == "optimizer":
        optimizer = make_optimizer(model, tiny_config, device)
        storage = parameter_storage(model)
        build_execution(model, tiny_config, device)
    else:
        build_execution(model, tiny_config, device)
        storage = parameter_storage(model)
        optimizer = make_optimizer(model, tiny_config, device)
    assert parameter_storage(model) is storage
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == set(map(id, parameters))


def test_bulk_update_preserves_autograd_mutation_detection() -> None:
    model = nn.Linear(3, 2)
    storage = prepare_parameter_storage(model)
    loss = model(torch.ones(1, 3, requires_grad=True)).square().sum()
    storage.copy_from(torch.zeros_like(storage.flat))
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("compiled", [False, True])
def test_adamw_trajectory_matches_independent_parameters(tiny_config: Config, compiled: bool) -> None:
    model = Llama(tiny_config.model, "reference")
    reference = copy.deepcopy(model)
    # Construct the reference optimizer directly so it retains ordinary allocations.
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=False)
    storage = prepare_parameter_storage(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    forward = torch.compile(model, backend="aot_eager", fullgraph=True) if compiled else model
    x = torch.tensor([[1, 2, 3, 4]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        token_losses(forward(x), x).mean().backward()
        token_losses(reference(x), x).mean().backward()
        storage.copy_from(storage.flat * 0.9)
        with torch.no_grad():
            for p in reference.parameters():
                p.mul_(0.9)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[actual][name], reference_optimizer.state[expected][name])


def test_state_dict_roundtrip_and_averaged_copy() -> None:
    model = nn.Linear(3, 2)
    legacy = copy.deepcopy(model.state_dict())
    storage = prepare_parameter_storage(model)
    storage.copy_from(torch.zeros_like(storage.flat))
    model.load_state_dict(legacy)
    assert prepare_parameter_storage(model) is storage
    file = io.BytesIO()
    torch.save(model.state_dict(), file)
    file.seek(0)
    restored = nn.Linear(3, 2)
    restored_storage = prepare_parameter_storage(restored)
    restored.load_state_dict(torch.load(file, weights_only=True))
    assert prepare_parameter_storage(restored) is restored_storage
    averaged = distributed.averaged_model(model)
    averaged_storage = prepare_parameter_storage(averaged)
    assert averaged_storage.flat.data_ptr() != storage.flat.data_ptr()
    for key, value in legacy.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(averaged.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("reuse", [False, True])
def test_mix_failures_and_storage_rebuild(monkeypatch: pytest.MonkeyPatch, reuse: bool) -> None:
    model = nn.Linear(3, 2)
    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    reduce = Mock(side_effect=OSError("injected collective failure"))
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(OSError, match="injected"):
        distributed.begin_parameter_mix(model, "complete", 0, reuse=reuse)
    storage = parameter_storage(model)
    assert storage.active_mixes == 0
    before = storage.flat.clone()
    work = Mock(spec=torch.distributed.Work)
    reduce.side_effect = None
    reduce.return_value = work
    pending = distributed.begin_parameter_mix(model, "complete", 0, reuse=reuse)
    assert pending is not None
    assert storage.active_mixes == 1
    # A failed wait cannot release a still-owned communication buffer.
    work.wait.side_effect = OSError("injected wait failure")
    with pytest.raises(OSError, match="injected wait"):
        pending.close()
    assert storage.active_mixes == 1
    work.wait.side_effect = None
    model.double()
    with pytest.raises(RuntimeError, match="active exchange"):
        prepare_parameter_storage(model)
    pending.close()
    pending.close()
    assert storage.active_mixes == 0
    torch.testing.assert_close(storage.flat, before, rtol=0, atol=0)
    rebuilt = prepare_parameter_storage(model)
    next_pending = distributed.begin_parameter_mix(model, "complete", 1, reuse=reuse)
    assert next_pending is not None and next_pending.flat.dtype == torch.float64
    assert next_pending.storage is rebuilt
    next_pending.close()


@pytest.mark.cuda
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_storage_conversion_and_fused_optimizer(tiny_config: Config, compiled: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = Llama(tiny_config.model)
    prepare_parameter_storage(model)
    model.cuda()
    storage = prepare_parameter_storage(model)
    assert storage.flat.device.type == "cuda"
    assert all(p.data_ptr() % PARAMETER_ALIGNMENT == 0 for p in model.parameters())
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), fused=True)
    forward = torch.compile(model) if compiled else model
    x = torch.tensor([[1, 2, 3, 4]], device="cuda")
    for current, opt in ((forward, optimizer), (reference, reference_optimizer)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token_losses(current(x), x).mean().backward()
        opt.step()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3 if compiled else 0, atol=1e-5 if compiled else 0)
