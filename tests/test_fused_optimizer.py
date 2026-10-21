import copy
import io
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from decentralized_llm.algorithms.adamw import ErrorFeedbackAdamW, make_optimizer, optimizer_report
from decentralized_llm.backends.distributed.runtime import parameter_mix_report
from decentralized_llm.backends.distributed.update import optimizer_update
from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.execution.cuda.adamw import FusedDecentralizedAdamW
from decentralized_llm.execution.qualification import qualification_key
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.storage import parameter_storage
from decentralized_llm.training.checkpoints import recipe_identity


def test_selector_and_checkpoint_identity(tiny_config: Config, cache: TokenCache, tmp_path: Path) -> None:
    assert tiny_config.runtime.decentralized_optimizer == "auto"
    original = recipe_identity(tiny_config, cache)
    key = qualification_key(tiny_config)
    tiny_config.runtime.decentralized_optimizer = "reference"
    assert recipe_identity(tiny_config, cache) == original
    assert qualification_key(tiny_config) != key
    path = tmp_path / "reference.yaml"
    save_config(tiny_config, path)
    assert load_config(path).runtime.decentralized_optimizer == "reference"


def cuda_config(config: Config, gamma: float) -> Config:
    config = config.model_copy(deep=True)
    config.distributed.mode = "decentralized"
    config.distributed.error_feedback_gamma = gamma
    config.runtime.device = "cuda"
    config.runtime.deterministic = False
    config.runtime.fused_optimizer = True
    return config


def assert_states(actual: torch.optim.Optimizer, expected: torch.optim.Optimizer) -> None:
    for a, b in zip(actual.param_groups, expected.param_groups, strict=True):
        for p, q in zip(a["params"], b["params"], strict=True):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
            assert actual.state.get(p, {}).keys() == expected.state.get(q, {}).keys()
            for key in actual.state.get(p, {}):
                torch.testing.assert_close(actual.state[p][key], expected.state[q][key], rtol=1e-5, atol=1e-6)
    if getattr(actual, "previous_update", None) is not None:
        torch.testing.assert_close(
            getattr(actual, "previous_update", None), getattr(expected, "previous_update", None), rtol=1e-5, atol=1e-6
        )


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
@pytest.mark.parametrize("scheme", ["local", "awc", "atc"])
def test_grouped_updates(tiny_config: Config, gamma: float, scheme: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = cuda_config(tiny_config, gamma)
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(37, 65), nn.LayerNorm(65), nn.Linear(65, 17)).cuda()
    model.register_parameter("tied", model.get_parameter("0.weight"))
    model.register_parameter("empty", nn.Parameter(torch.empty(0, device="cuda")))
    reference = copy.deepcopy(model)
    optimizer = make_optimizer(model, config, torch.device("cuda"))
    config.runtime.decentralized_optimizer = "reference"
    control = make_optimizer(reference, config, torch.device("cuda"))
    assert isinstance(optimizer, FusedDecentralizedAdamW)
    if gamma == 0:
        assert optimizer.previous_update is None
    storage, ref_storage = parameter_storage(model), parameter_storage(reference)
    pointers = [p.data_ptr() for p in model.parameters()]
    for step in range(12):
        lr = (0.01, 0.003, 0.0, 0.005)[step % 4]
        for opt in (optimizer, control):
            for group in opt.param_groups:
                group["lr"] = lr
        for index, (p, q) in enumerate(zip(model.parameters(), reference.parameters(), strict=True)):
            grad = None if (step + index) % 4 == 1 else torch.randn_like(p) * 0.1
            p.grad = grad
            q.grad = None if grad is None else grad.clone()
        with torch.no_grad():
            base = None
            if scheme == "awc":
                base = storage.flat * 0.9 * 3
                ref_storage.copy_from(ref_storage.flat * 0.9)
            optimizer.step_fused(
                base=base, base_divisor=3 if base is not None else 1, output_divisor=3 if scheme == "atc" else 1
            )
            control.step()
            if scheme == "atc":
                ref_storage.flat.div_(3)
        assert_states(optimizer, control)
        assert [p.data_ptr() for p in model.parameters()] == pointers
        assert optimizer_report(optimizer)["scratch_bytes"] == 0
        padding = torch.ones_like(storage.flat, dtype=torch.bool)
        for p, offset in zip(storage.parameters, storage.offsets, strict=True):
            padding[offset : offset + p.numel()] = False
        assert torch.count_nonzero(storage.flat[padding]) == 0
        if gamma != 0:
            assert optimizer.previous_update is not None
            assert torch.count_nonzero(optimizer.previous_update[padding]) == 0


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
def test_fused_resume_closure_and_fallback(tiny_config: Config, gamma: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = cuda_config(tiny_config, gamma)
    model = nn.Linear(3, 2).cuda()
    opt = make_optimizer(model, config, torch.device("cuda"))
    assert isinstance(opt, FusedDecentralizedAdamW)
    calls = []

    def closure() -> float:
        opt.zero_grad(set_to_none=True)
        loss = model(torch.ones(1, 3, device="cuda")).square().sum()
        loss.backward()
        calls.append(float(loss.detach()))
        return calls[-1]

    assert opt.step(closure) == calls[0] and len(calls) == 1
    file = io.BytesIO()
    torch.save(opt.state_dict(), file)
    file.seek(0)
    state = torch.load(file, map_location="cpu", weights_only=True)
    restored_model = copy.deepcopy(model)
    restored = make_optimizer(restored_model, config, torch.device("cuda"))
    restored.load_state_dict(state)
    # Also exercise loading the same state into the historical reference path.
    reference_model = copy.deepcopy(model)
    config.runtime.decentralized_optimizer = "reference"
    reference = make_optimizer(reference_model, config, torch.device("cuda"))
    reference.load_state_dict(state)
    for m in (model, restored_model, reference_model):
        for p in m.parameters():
            p.grad = torch.full_like(p, 0.2)
    opt.step()
    restored.step()
    reference.step()
    assert_states(opt, restored)
    assert_states(opt, reference)
    for p, q in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # Existing checkpoints initialize a fresh fused optimizer correctly too.
    restored.load_state_dict(copy.deepcopy(reference.state_dict()))
    restored_model.load_state_dict(reference_model.state_dict())
    assert_states(restored, reference)
    # A strided gradient falls back before any moments are advanced.
    for m in (restored_model, reference_model):
        m.weight.grad = torch.full((3, 2), 0.2, device="cuda").t()
    for group in reference.param_groups:
        group.update(fused=False, foreach=False)
    restored.step()
    reference.step()
    assert_states(restored, reference)
    assert optimizer_report(restored)["implementation"] == "reference"
    # A raw kernel write must still invalidate a saved autograd version.
    loss = restored_model(torch.ones(1, 3, device="cuda", requires_grad=True)).square().sum()
    for p in restored_model.parameters():
        p.grad = torch.ones_like(p)
    restored.step()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.cuda
def test_reference_selection(tiny_config: Config) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = cuda_config(tiny_config, 1.0)
    for reason in ("dtype", "deterministic", "disabled", "reference"):
        cfg = config.model_copy(deep=True)
        model = nn.Linear(3, 2).cuda()
        if reason == "dtype":
            model.double()
        elif reason == "deterministic":
            cfg.runtime.deterministic = True
        elif reason == "disabled":
            cfg.runtime.fused_optimizer = False
        else:
            cfg.runtime.decentralized_optimizer = "reference"
        assert isinstance(make_optimizer(model, cfg, torch.device("cuda")), ErrorFeedbackAdamW)


def cuda_distributed_worker(worker: int, size: int, port: int, raw: dict[str, Any]) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(worker)
    device = torch.device("cuda", worker)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=worker, world_size=size, device_id=device, timeout=timedelta(seconds=120))
    try:
        topologies = ("complete",) if size == 3 else ("complete", "one_peer_ring", "one_peer_exponential")
        for topology in topologies:
            for scheme in ("awc", "atc"):
                for gamma in (0.0, 0.5, 1.0):
                    for reuse in (False, True):
                        config = cuda_config(Config.model_validate(raw), gamma)
                        config.distributed.gpus_per_node = size
                        config.distributed.backend = "nccl"
                        config.distributed.scheme = scheme
                        config.distributed.topology = topology
                        config.distributed.reuse_mix_buffers = reuse
                        config.training.batch_tokens = size * 4 * 4
                        config.training.num_accumulation = 2
                        config.runtime.compile = size == 2 and topology == "complete" and reuse
                        torch.manual_seed(worker + 42)
                        model = Llama(config.model).to(device)
                        control = copy.deepcopy(model)
                        forwards = [
                            cast(nn.Module, torch.compile(m)) if config.runtime.compile else m for m in (model, control)
                        ]
                        reference = config.model_copy(deep=True)
                        reference.runtime.decentralized_optimizer = "reference"
                        optimizers = [make_optimizer(m, c, device) for m, c in ((model, config), (control, reference))]
                        cache = TokenCache(config.data.cache_dir)
                        blocks = config.training.batch_tokens // 4
                        loaders = [TokenLoader(cache, config, worker, 0, blocks * 4) for _ in range(2)]
                        transfers = [DevicePrefetcher(loader, device) for loader in loaders]
                        try:
                            for step in range(4):
                                for m, forward, opt, cfg, transfer in zip(
                                    (model, control), forwards, optimizers, (config, reference), transfers, strict=True
                                ):
                                    optimizer_update(
                                        m,
                                        forward,
                                        opt,
                                        cfg,
                                        device,
                                        transfer.next,
                                        transfer.preload,
                                        step * blocks,
                                        (step + 1) * blocks,
                                        step,
                                        blocks * 4,
                                    )
                                assert_states(*optimizers)
                                assert parameter_storage(model).active_mixes == 0
                                assert parameter_mix_report(model)["snapshot_bytes"] == (
                                    0 if scheme == "atc" else parameter_storage(model).flat.nbytes
                                )
                        finally:
                            for transfer in transfers:
                                transfer.close()
                            for loader in loaders:
                                loader.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.parametrize("size", [2, 3, 4])
def test_cuda_fused_mixing(cache: TokenCache, tiny_config: Config, size: int) -> None:
    if torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} CUDA GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(cuda_distributed_worker, args=(size, port, tiny_config.model_dump()), nprocs=size, join=True)  # pyright: ignore[reportPrivateImportUsage]


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
def test_fused_failed_awc_drains(tiny_config: Config, monkeypatch: pytest.MonkeyPatch, gamma: float) -> None:
    from unittest.mock import Mock

    from decentralized_llm.backends.distributed import update as updates
    from decentralized_llm.backends.distributed.runtime import PendingParameterMix

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = cuda_config(tiny_config, gamma)
    model = Llama(config.model).cuda()
    opt = make_optimizer(model, config, torch.device("cuda"))
    storage = parameter_storage(model)
    before = storage.flat.clone()
    work = Mock(spec=dist.Work)
    pending = PendingParameterMix(storage, torch.zeros_like(storage.flat), [work], 2)
    monkeypatch.setattr(updates, "begin_parameter_mix", lambda *args, **kwargs: pending)

    def fail() -> Any:
        raise OSError("injected loader failure")

    with pytest.raises(OSError, match="injected loader failure"):
        optimizer_update(model, model, opt, config, torch.device("cuda"), fail, lambda: None, 0, 6, 0, 42)
    work.wait.assert_called_once()
    assert not opt.state and storage.active_mixes == 0
    torch.testing.assert_close(storage.flat, before, rtol=0, atol=0)
    history = getattr(opt, "previous_update", None)
    assert history is None or torch.count_nonzero(history) == 0


def test_live_mix_launch_failure_releases_storage(tiny_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from decentralized_llm.backends.distributed import runtime as distributed

    model = nn.Linear(3, 2)
    make_optimizer(model, tiny_config, torch.device("cpu"))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("injected collective failure")

    monkeypatch.setattr(dist, "all_reduce", fail)
    with pytest.raises(OSError, match="injected collective failure"):
        distributed.sum_updated_parameters(model, distributed.ParameterMixGroup(None, 2))
    assert parameter_storage(model).active_mixes == 0
