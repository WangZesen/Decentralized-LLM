import copy
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from decentralized_llm.cache import TokenCache
from decentralized_llm.checkpoints import load_checkpoint, save_checkpoint
from decentralized_llm.config import Config
from decentralized_llm.distributed import begin_parameter_mix, peer_for, prepare_parameter_groups
from decentralized_llm.evaluation import evaluate
from decentralized_llm.execution import build_execution
from decentralized_llm.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.model import Llama, token_losses
from decentralized_llm.optim import learning_rate, make_optimizer
from decentralized_llm.runtime import clip_grad_norm_
from decentralized_llm.sampling import GlobalSampler
from decentralized_llm.train import optimizer_update, train


def reference_batch(cache: TokenCache, config: Config, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    sampler = GlobalSampler.from_config(cache, config)
    tokens = [
        torch.tensor(cache.read("train", sampler.block(p) * 4, sampler.block(p) * 4 + 5).astype("int64"))
        for p in positions
    ]
    packed = torch.stack(tokens)
    return packed[:, :-1], packed[:, 1:]


def reference_mix(models: list[Llama], topology: str, step: int) -> None:
    with torch.no_grad():
        states = [{k: v.clone() for k, v in m.state_dict().items()} for m in models]
        for rank, model in enumerate(models):
            mixed = {}
            for key, value in states[rank].items():
                if topology == "complete":
                    mixed[key] = torch.stack([s[key] for s in states]).mean(0)
                else:
                    mixed[key] = (value + states[peer_for(rank, len(models), topology, step)][key]) / 2
            model.load_state_dict(mixed)


def distributed_comparisons(worker: int, size: int, port: int, raw: dict[str, Any], compiled: bool = False) -> None:
    torch.set_num_threads(1)
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(worker), WORLD_SIZE=str(size), LOCAL_RANK=str(worker)
    )
    dist.init_process_group("gloo")
    original_compile = torch.compile

    def eager_compile(model: torch.nn.Module, **_: Any) -> Any:
        return original_compile(model, backend="eager")

    if compiled:
        torch.compile = eager_compile
    try:
        for mode, topology, scheme in [("ddp", "complete", "awc")] + [
            ("decentralized", topology, scheme)
            for topology in ("complete", "one_peer_ring", "one_peer_exponential")
            for scheme in ("awc", "atc")
        ]:
            current = copy.deepcopy(raw)
            current["distributed"].update(gpus_per_node=size, mode=mode, topology=topology, scheme=scheme)
            current["training"].update(batch_tokens=40, micro_batch_size=2)
            cfg = Config.model_validate(current)
            cfg.runtime.compile = compiled and mode == "ddp"
            cache = TokenCache(cfg.data.cache_dir)
            references = []
            for r in range(size if mode == "decentralized" else 1):
                torch.manual_seed(42 + r)
                references.append(Llama(cfg.model, "reference"))
            torch.manual_seed(42 + (worker if mode == "decentralized" else 0))
            model = Llama(cfg.model, "reference")
            optimizer = make_optimizer(model, cfg, torch.device("cpu"))
            ref_opts = [make_optimizer(m, cfg, torch.device("cpu")) for m in references]
            forward = build_execution(model, cfg, torch.device("cpu"))
            with TokenLoader(cache, cfg, worker, 0, 40) as loader:
                loader.warmup()
                transfer = DevicePrefetcher(loader, torch.device("cpu"))
                for step in range(4):
                    start, stop = step * 10, (step + 1) * 10
                    for r, (reference, opt) in enumerate(zip(references, ref_opts, strict=True)):
                        positions = [p for p in range(start, stop) if mode == "ddp" or p % size == r]
                        x, y = reference_batch(cache, cfg, positions)
                        opt.zero_grad(set_to_none=True)
                        token_losses(reference(x), y).mean().backward()
                        clip_grad_norm_(reference.parameters(), cfg.optimizer.grad_clip)
                        for group in opt.param_groups:
                            group["lr"] = learning_rate(cfg, stop * 4, 160)
                    if mode == "decentralized" and scheme == "awc":
                        reference_mix(references, topology, step)
                    for opt in ref_opts:
                        opt.step()
                    if mode == "decentralized" and scheme == "atc":
                        reference_mix(references, topology, step)
                    optimizer_update(
                        model,
                        forward,
                        optimizer,
                        cfg,
                        torch.device("cpu"),
                        transfer.next,
                        transfer.preload,
                        start,
                        stop,
                        step,
                        40,
                    )
                    loader.commit(stop)
                    reference = references[worker if mode == "decentralized" else 0]
                    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-5)
                    reference_opt = ref_opts[worker if mode == "decentralized" else 0]
                    for actual, expected in zip(optimizer.state.values(), reference_opt.state.values(), strict=True):
                        for key in ("exp_avg", "exp_avg_sq"):
                            torch.testing.assert_close(actual[key], expected[key], atol=3e-7, rtol=3e-5)
                before = copy.deepcopy(model.state_dict())
                for full in (False, True):
                    result = evaluate(model, cache, cfg, torch.device("cpu"), full)
                    if full:
                        assert result["tokens"] == 38
                        averaged = copy.deepcopy(references[0])
                        averaged.load_state_dict(
                            {
                                key: torch.stack([m.state_dict()[key] for m in references]).mean(0)
                                for key in averaged.state_dict()
                            }
                        )
                        tokens = torch.tensor(cache.read("validation", 0, 39).astype("int64"))
                        loss_sum = 0.0
                        for offset in range(0, 38, 4):
                            x = tokens[offset : min(offset + 4, 38)][None]
                            y = tokens[offset + 1 : min(offset + 5, 39)][None]
                            loss_sum += float(token_losses(averaged(x), y).sum())
                        assert result["loss"] == pytest.approx(loss_sum / 38, abs=2e-6)
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                checkpoint = cfg.runtime.output_dir / f"{size}-{mode}-{topology}-{scheme}"
                save_checkpoint(
                    checkpoint,
                    cfg,
                    cache,
                    model,
                    optimizer,
                    loader.state_dict(),
                    {"cursor": 40, "step": 4, "completed_epochs": 1, "best_loss": 3.0, "best_epoch": 1},
                )
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(10)
                manifest, _ = load_checkpoint(checkpoint, cfg, cache, model, optimizer)
                assert manifest["cursor"] == 40
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                transfer.close()
    finally:
        torch.compile = original_compile
        dist.destroy_process_group()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def snapshot_worker(worker: int, size: int, ports: list[int], device_type: str) -> None:
    torch.set_num_threads(1)
    device = torch.device(device_type, worker) if device_type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Repeat initialization in the same process to catch stale cached group handles.
    for port in ports:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            rank=worker,
            world_size=size,
            timeout=timedelta(seconds=30),
            device_id=device if device.type == "cuda" else None,
        )
        try:
            with (
                patch.object(dist, "new_group", wraps=dist.new_group) as create,
                patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce,
            ):
                for topology in ("complete", "one_peer_ring", "one_peer_exponential"):
                    prepare_parameter_groups(topology, timeout=timedelta(seconds=30))
                    created = create.call_count
                    if topology == "complete" or size == 2:
                        assert created == 0
                    model = torch.nn.Linear(3, 2, bias=False, device=device)
                    pointer = None
                    for step in range(4):
                        model.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            model.weight.fill_(worker + 1 + step)
                        before = model.weight.detach().clone()
                        pending = begin_parameter_mix(model, topology, step, reuse=step > 0)
                        assert pending is not None
                        if step == 1:
                            pointer = pending.flat.data_ptr()
                        elif step > 1:
                            assert pending.flat.data_ptr() == pointer
                        assert create.call_count == created  # No communicator creation during updates.
                        assert reduce.call_args is not None
                        assert reduce.call_args.kwargs["async_op"]
                        members = dist.get_process_group_ranks(reduce.call_args.kwargs["group"])
                        expected_members = (
                            list(range(size))
                            if topology == "complete"
                            else sorted([worker, peer_for(worker, size, topology, step)])
                        )
                        assert members == expected_members
                        # Check both an in-flight exchange and one already joined before forward.
                        if step % 2 == 0:
                            pending.wait()
                        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
                        model(torch.ones(1, 3, device=device)).square().sum().backward()
                        torch.testing.assert_close(model.weight.grad, 6 * before, rtol=0, atol=0)
                        # Live storage must not alias the communication snapshot.
                        with torch.no_grad():
                            model.weight.add_(100)
                        pending.apply()
                        value = (
                            (size + 1) / 2 + step
                            if topology == "complete"
                            else (worker + peer_for(worker, size, topology, step)) / 2 + 1 + step
                        )
                        torch.testing.assert_close(model.weight, torch.full_like(before, value), rtol=0, atol=0)
                        torch.testing.assert_close(model.weight.grad, 6 * before, rtol=0, atol=0)
                        pending.close()
        finally:
            dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("size", [2, 4, 8])
def test_parameter_mix_snapshot(size: int) -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        snapshot_worker, args=(size, [free_port(), free_port()], "cpu"), nprocs=size, join=True
    )


@pytest.mark.cuda
@pytest.mark.parametrize("size", [2, 4])
def test_cuda_parameter_mix_snapshot(size: int) -> None:
    if torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} CUDA GPUs")
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        snapshot_worker, args=(size, [free_port(), free_port()], "cuda"), nprocs=size, join=True
    )


@pytest.mark.distributed
@pytest.mark.parametrize("size,factor,compiled", [(2, 1, False), (4, 1, False), (4, 3, False), (4, 1, True)])
def test_distributed_serial_references(
    cache: TokenCache, tiny_config: Config, size: int, factor: int, compiled: bool
) -> None:
    tiny_config.training.data_reuse_factor = factor
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage] # Public torch API missing from its export stubs.
        distributed_comparisons,
        args=(size, free_port(), tiny_config.model_dump(mode="json"), compiled),
        nprocs=size,
        join=True,
    )


def resume_worker(worker: int, size: int, ports: list[int], raw: dict[str, Any]) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(ports[0]),
        RANK=str(worker),
        WORLD_SIZE=str(size),
        LOCAL_RANK=str(worker),
    )
    port_index = 0
    for mode in ("ddp", "decentralized"):
        current = copy.deepcopy(raw)
        current["distributed"].update(gpus_per_node=size, mode=mode)
        root = Path(current["runtime"]["output_dir"])
        current["runtime"]["output_dir"] = str(root / f"{mode}-full")
        cfg = Config.model_validate(current)
        os.environ["MASTER_PORT"] = str(ports[port_index])
        port_index += 1
        train(cfg)
        original = torch.load(cfg.runtime.output_dir / f"final/rank-{worker:05d}.pt", weights_only=False)
        cfg.runtime.output_dir = root / f"{mode}-split"
        cfg.data.loader_mode = "streaming"
        os.environ["MASTER_PORT"] = str(ports[port_index])
        port_index += 1
        train(cfg, stop_after_steps=2)
        checkpoint = cfg.runtime.output_dir / "step-00000002"
        cfg.runtime.output_dir = root / f"{mode}-resumed"
        cfg.data.loader_mode = "resident"
        cfg.data.prefetch_batches = 1
        cfg.data.reader_threads = 1
        os.environ["MASTER_PORT"] = str(ports[port_index])
        port_index += 1
        train(cfg, checkpoint)
        actual = torch.load(cfg.runtime.output_dir / f"final/rank-{worker:05d}.pt", weights_only=False)
        if "model" in original:
            for key, value in original["model"].items():
                # Recreated DDP buckets may change FP32 collective summation order.
                torch.testing.assert_close(actual["model"][key], value, rtol=1e-6, atol=1e-7)
        assert original["loader"] == actual["loader"]
        assert torch.equal(original["rng"]["torch"], actual["rng"]["torch"])


@pytest.mark.distributed
@pytest.mark.parametrize("size,factor", [(2, 1), (4, 1), (2, 4), (4, 7)])
def test_multiprocess_training_resume(cache: TokenCache, tiny_config: Config, size: int, factor: int) -> None:
    tiny_config.training.data_reuse_factor = factor
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage] # Public torch API missing from its export stubs.
        resume_worker,
        args=(size, [free_port() for _ in range(6)], tiny_config.model_dump(mode="json")),
        nprocs=size,
        join=True,
    )
