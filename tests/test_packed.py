"""Packed execution must simulate the same initialized workers and sample stream."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch

from decentralized_llm.algorithms.adamw import make_optimizer
from decentralized_llm.algorithms.schedules import learning_rate
from decentralized_llm.backends.factory import create_backend, runtime_for
from decentralized_llm.backends.packed.engine import PackedBackend
from decentralized_llm.config import Config, SlurmLauncher
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.data.sampling import GlobalSampler, memory_estimate
from decentralized_llm.launch.local import local_command
from decentralized_llm.launch.slurm import scheduler_arguments, slurm_scripts
from decentralized_llm.models.initialization import INITIALIZATION, initialize_model
from decentralized_llm.models.llama import token_losses
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.runtime import clip_grad_norm_
from decentralized_llm.training.checkpoints import validate_checkpoint
from decentralized_llm.training.loop import evaluate_checkpoint, train
from decentralized_llm.tuning.batching import resource_key


def packed_config(config: Config, workers: int = 4, **settings: Any) -> Config:
    raw = config.model_dump(mode="json")
    raw["runtime"]["training_backend"] = "packed"
    raw["distributed"].update(mode="decentralized", gpus_per_node=workers, **settings)
    raw["training"].update(batch_tokens=64, num_accumulation=2)
    return Config.model_validate(raw)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [42, 99])
def test_identical_initialization(tiny_config: Config, workers: int, seed: int) -> None:
    tiny_config.runtime.seed = seed
    canonical = initialize_model(tiny_config)
    state = torch.get_rng_state().clone()
    torch.rand(117)
    config = packed_config(tiny_config, workers)
    backend = create_backend(config, torch.device("cpu"))
    assert isinstance(backend.model, PackedLlama)
    for worker in range(workers):
        for name, value in backend.model.local_state_dict(worker).items():
            assert torch.equal(value, canonical.state_dict()[name])
    before = backend.model.local_state_dict(0)
    backend.prepare()
    assert all(torch.equal(value, backend.model.local_state_dict(0)[name]) for name, value in before.items())
    if workers > 1:
        with torch.no_grad():
            backend.model.embedding.weight[0].add_(1)
        assert torch.equal(backend.model.embedding.weight[1], canonical.embedding.weight)
    config.distributed.mode = "ddp"
    synchronous = create_backend(config, torch.device("cpu"))
    assert isinstance(synchronous.model, PackedLlama)
    assert all(
        torch.equal(value, canonical.state_dict()[name])
        for name, value in synchronous.model.local_state_dict(0).items()
    )
    torch.set_rng_state(state)
    initialize_model(config)
    assert torch.equal(state, torch.get_rng_state())
    config.runtime.seed += 1
    assert not torch.equal(initialize_model(config).embedding.weight, canonical.embedding.weight)


@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
def test_reference_updates(cache: TokenCache, tiny_config: Config, topology: str, scheme: str, gamma: float) -> None:
    cfg = packed_config(tiny_config, topology=topology, scheme=scheme, error_feedback_gamma=gamma)
    backend = PackedBackend(cfg, torch.device("cpu"), runtime_for(cfg))
    backend.prepare()
    references = [initialize_model(cfg) for _ in range(4)]
    opts = [make_optimizer(model, cfg, torch.device("cpu")) for model in references]
    sampler = GlobalSampler.from_config(cache, cfg)
    blocks, total = 16, 64
    from test_distributed import reference_mix

    with TokenLoader(cache, cfg, 0, 0, total) as loader:
        transfer = DevicePrefetcher(loader, torch.device("cpu"))
        for step in range(4):
            for rank, (model, opt) in enumerate(zip(references, opts, strict=True)):
                values = np.stack(
                    [
                        cache.read("train", sampler.block(p) * 4, sampler.block(p) * 4 + 5)
                        for p in range(step * blocks + rank, (step + 1) * blocks, 4)
                    ]
                )
                x, y = torch.tensor(values[:, :-1].astype("int64")), torch.tensor(values[:, 1:].astype("int64"))
                opt.zero_grad(set_to_none=True)
                token_losses(model(x), y).mean().backward()
                clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
                for group in opt.param_groups:
                    group["lr"] = learning_rate(cfg, (step + 1) * 64, total * 4)
            if scheme == "awc":
                reference_mix(references, topology, step)
            for opt in opts:
                opt.step()
            if scheme == "atc":
                reference_mix(references, topology, step)
            backend.update(transfer.next, transfer.preload, step * blocks, (step + 1) * blocks, step, total)
            loader.commit((step + 1) * blocks)
            for rank, model in enumerate(references):
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(backend.model.local_state_dict(rank)[name], value, rtol=2e-5, atol=3e-7)
                for entry, local_parameter in zip(
                    backend.model.layout, backend.optimizer.local_parameters[rank], strict=True
                ):
                    actual = backend.optimizer.optimizers[rank].state[local_parameter]
                    expected = opts[rank].state[model.get_parameter(entry.name)]
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(actual[key], expected[key], rtol=3e-5, atol=3e-7)
        transfer.close()


@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_packed_training_resume(
    cache: TokenCache,
    tiny_config: Config,
    tmp_path: Path,
    mode: Literal["ddp", "decentralized"],
    scheme: str,
    gamma: float,
) -> None:
    cfg = packed_config(tiny_config, scheme=scheme, error_feedback_gamma=gamma)
    cfg.distributed.mode = mode
    # Enough updates to exercise both an interruption and rotating peers.
    cfg.training.tokens_per_parameters = 0.8
    cfg.training.epoch_tokens_per_parameters = 0.4
    cfg.distributed.topology = "one_peer_ring"
    train(cfg)
    original = torch.load(cfg.runtime.output_dir / "final/packed.pt", weights_only=False)
    result = evaluate_checkpoint(cfg, cfg.runtime.output_dir / "final")
    assert result["tokens"] == 38
    cfg.runtime.output_dir = tmp_path / "interrupted"
    train(cfg, stop_after_steps=2)
    checkpoint = cfg.runtime.output_dir / "step-00000002"
    cfg.runtime.output_dir = tmp_path / "resumed"
    train(cfg, checkpoint)
    actual = torch.load(cfg.runtime.output_dir / "final/packed.pt", weights_only=False)
    for key in ("parameter_storage",):
        assert torch.equal(actual["model"][key], original["model"][key])
    for key in ("first_moment_storage", "second_moment_storage", "previous_update"):
        a, b = actual["optimizer"][key], original["optimizer"][key]
        assert (a is None and b is None) or (
            isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
        )
    manifest = validate_checkpoint(cfg.runtime.output_dir / "final", cfg, cache)
    assert manifest["initialization"] == INITIALIZATION
    assert manifest["world_size"] == 4
    raw = cfg.model_dump(mode="json")
    raw["runtime"]["training_backend"] = "distributed"
    with pytest.raises(ValueError, match="training backend"):
        validate_checkpoint(cfg.runtime.output_dir / "final", Config.model_validate(raw), cache)
    cfg.distributed.error_feedback_gamma = gamma + 0.25
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(cfg.runtime.output_dir / "final", cfg, cache)


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_physical_allocation(tiny_config: Config, workers: int, tmp_path: Path) -> None:
    cfg = packed_config(tiny_config, workers)
    cfg.training.batch_tokens *= 2
    cfg.distributed.nodes = 2
    assert cfg.physical_world_size == 1
    assert "--nproc-per-node=1" in local_command(cfg, tmp_path / "config.yaml")
    cfg.launcher = SlurmLauncher(partition="gpu")
    arguments = scheduler_arguments(cfg)
    assert "--nodes=1" in arguments and "--gpus-per-node=1" in arguments
    batch, worker = slurm_scripts(cfg, tmp_path / "config.yaml", tmp_path / "worker.sh")
    assert "--ntasks=1 " in batch and "--gpus-per-task=1 " in batch
    assert "--nnodes=1 --nproc-per-node=1 " in worker and "--standalone" in worker
    original = resource_key(cfg)
    cfg.distributed.nodes = 1
    assert resource_key(cfg) == original


def test_global_sampler_and_budget(cache: TokenCache, tiny_config: Config) -> None:
    cfg = packed_config(tiny_config)
    sampler = GlobalSampler.from_config(cache, cfg)
    packed = list(sampler.batches(cfg, 0, 0, 16))
    raw = cfg.model_dump(mode="json")
    raw["runtime"]["training_backend"] = "distributed"
    real = Config.model_validate(raw)
    for rank in range(4):
        local = list(sampler.batches(real, rank, 0, 16))
        for whole, part in zip(packed, local, strict=True):
            assert np.array_equal(whole.indices[rank::4], part.indices)
    assert memory_estimate(cache, cfg)["transfer_buffers"] == 4 * memory_estimate(cache, real)["transfer_buffers"]


def test_no_process_group(tiny_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = packed_config(tiny_config)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **_: pytest.fail("packed initialized a group"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert runtime_for(cfg).setup(cfg) == torch.device("cpu")
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="one physical process"):
        runtime_for(cfg).setup(cfg)


def initialization_worker(worker: int, port: int, raw: dict[str, Any]) -> None:
    import os
    from unittest.mock import patch

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(worker), LOCAL_RANK=str(worker), WORLD_SIZE="4"
    )
    cfg = Config.model_validate(raw)
    cfg.runtime.training_backend = "distributed"
    context = runtime_for(cfg)
    device = context.setup(cfg)
    original_compile = torch.compile

    def compile_eager(module: Any, **_: Any) -> Any:
        return original_compile(module, backend="eager")

    try:
        with patch.object(torch, "compile", compile_eager):
            for seed in (42, 123):
                cfg.runtime.seed = seed
                canonical = initialize_model(cfg)
                packed = PackedLlama(cfg.model, 4, canonical=canonical)
                for mode in ("ddp", "decentralized"):
                    cfg.distributed.mode = mode
                    for compiled in (False, True):
                        cfg.runtime.compile = compiled
                        torch.manual_seed(1000 + worker)
                        backend = create_backend(cfg, device, context)
                        backend.prepare()
                        x = torch.ones((cfg.micro_batch_size, cfg.model.context_length), dtype=torch.int64)
                        backend.execution(x, x).backward()
                        for name, value in backend.model.state_dict().items():
                            assert torch.equal(value, canonical.state_dict()[name]), (worker, mode, compiled, name)
                            assert torch.equal(value, packed.local_state_dict(worker)[name])
    finally:
        context.close()


@pytest.mark.distributed
def test_real_and_packed_initialization(tiny_config: Config) -> None:
    import torch.multiprocessing as mp
    from test_distributed import free_port

    cfg = packed_config(tiny_config)
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        initialization_worker, args=(free_port(), cfg.model_dump(mode="json")), nprocs=4, join=True
    )


@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
def test_legacy_initialization_is_preserved(
    cache: TokenCache, tiny_config: Config, tmp_path: Path, mode: Literal["ddp", "decentralized"]
) -> None:
    import json

    from decentralized_llm.training.checkpoints import load_checkpoint, recipe_identity

    tiny_config.distributed.mode = mode
    train(tiny_config, stop_after_steps=1)
    checkpoint = tiny_config.runtime.output_dir / "step-00000001"
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("initialization")
    manifest.pop("training_backend")
    manifest["recipe_identity"] = recipe_identity(tiny_config, cache, initialization=None)
    assert manifest["recipe_identity"] != recipe_identity(tiny_config, cache)
    manifest_path.write_text(json.dumps(manifest))
    backend = create_backend(tiny_config, torch.device("cpu"))
    load_checkpoint(checkpoint, tiny_config, cache, backend.model, backend.optimizer, backend=backend)
    expected = torch.load(checkpoint / "rank-00000.pt", weights_only=False)
    for name, value in backend.model.state_dict().items():
        assert torch.equal(value, expected["model"][name])
    tiny_config.runtime.output_dir = tmp_path / "legacy-resumed"
    train(tiny_config, checkpoint)
    legacy = "legacy_rank_offset_v1" if mode == "decentralized" else "legacy_shared_seed_v1"
    assert json.loads((tiny_config.runtime.output_dir / "run.json").read_text())["initialization"] == legacy
    final = validate_checkpoint(tiny_config.runtime.output_dir / "final", tiny_config, cache)
    assert final["initialization"] == legacy


@pytest.mark.parametrize("accumulation", [1, 2, 4])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
def test_packed_synchronous_global_batch(
    cache: TokenCache, tiny_config: Config, accumulation: int, gamma: float
) -> None:
    from test_distributed import reference_batch

    cfg = packed_config(tiny_config, error_feedback_gamma=gamma)
    cfg.distributed.mode = "ddp"
    cfg.training.num_accumulation = accumulation
    backend = PackedBackend(cfg, torch.device("cpu"), runtime_for(cfg))
    backend.prepare()
    if gamma == 0:
        assert backend.optimizer.previous_update is None and backend.optimizer.scratch is None
    reference = initialize_model(cfg)
    optimizer = make_optimizer(reference, cfg, torch.device("cpu"))
    with TokenLoader(cache, cfg, 0, 0, 64) as loader:
        transfer = DevicePrefetcher(loader, torch.device("cpu"))
        try:
            for step in range(4):
                x, y = reference_batch(cache, cfg, list(range(step * 16, (step + 1) * 16)))
                optimizer.zero_grad(set_to_none=True)
                token_losses(reference(x), y).mean().backward()
                clip_grad_norm_(reference.parameters(), cfg.optimizer.grad_clip)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(cfg, (step + 1) * 64, 256)
                optimizer.step()
                backend.update(transfer.next, transfer.preload, step * 16, (step + 1) * 16, step, 64)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(backend.model.local_state_dict(0)[name], value, rtol=2e-5, atol=3e-7)
        finally:
            transfer.close()


def test_packed_benchmarks(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    from decentralized_llm.benchmarks.data import benchmark_data
    from decentralized_llm.benchmarks.training import benchmark_training

    cfg = packed_config(tiny_config)
    data = benchmark_data(cfg, steps=2, warmup=1, cache_state="warm")
    assert data["world_size"] == 4 and data["physical_world_size"] == 1
    cfg.benchmark.warmup = 1
    cfg.benchmark.steps = 2
    cfg.benchmark.windows = 1
    cfg.benchmark.min_seconds = 0.001
    cfg.runtime.output_dir = tmp_path / "benchmark"
    result = benchmark_training(cfg)
    assert result["world_size"] == 4 and result["physical_world_size"] == 1


def test_packed_local_launch(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    import json

    from decentralized_llm.launch.commands import launch

    cfg = packed_config(tiny_config)
    cfg.runtime.output_dir = tmp_path / "launched"
    report = launch(cfg)
    assert report["preflight"]["world_size"] == 4
    assert report["preflight"]["physical_world_size"] == 1
    assert len(list(cfg.runtime.output_dir.glob("rank-*.json"))) == 1
    manifest = json.loads((cfg.runtime.output_dir / "final/manifest.json").read_text())
    assert {entry["file"] for entry in manifest["files"]} == {"packed.pt"}
