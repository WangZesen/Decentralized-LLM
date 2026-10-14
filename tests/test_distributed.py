import copy
import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

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
from decentralized_llm.optim import ErrorFeedbackAdamW, learning_rate, make_optimizer
from decentralized_llm.parameter_storage import parameter_storage
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
            current["distributed"]["error_feedback"] = raw["distributed"]["error_feedback"] and mode == "decentralized"
            current["training"]["batch_tokens"] = 64
            cfg = Config.model_validate(current)
            batch_blocks = cfg.training.batch_tokens // cfg.model.context_length
            total_blocks = 4 * batch_blocks
            cfg.runtime.compile = compiled
            cache = TokenCache(cfg.data.cache_dir)
            references = []
            for r in range(size if mode == "decentralized" else 1):
                torch.manual_seed(42 + r)
                references.append(Llama(cfg.model, "reference"))
            torch.manual_seed(42 + (worker if mode == "decentralized" else 0))
            model = Llama(cfg.model, "reference")
            optimizer = make_optimizer(model, cfg, torch.device("cpu"))
            reference_config = cfg.model_copy(deep=True)
            reference_config.distributed.error_feedback = False
            ref_opts = [make_optimizer(m, reference_config, torch.device("cpu")) for m in references]
            histories = [[torch.zeros_like(p) for p in m.parameters()] for m in references]
            forward = build_execution(model, cfg, torch.device("cpu"))
            with TokenLoader(cache, cfg, worker, 0, total_blocks) as loader:
                loader.warmup()
                transfer = DevicePrefetcher(loader, torch.device("cpu"))
                for step in range(4):
                    start, stop = step * batch_blocks, (step + 1) * batch_blocks
                    for r, (reference, opt) in enumerate(zip(references, ref_opts, strict=True)):
                        positions = [p for p in range(start, stop) if mode == "ddp" or p % size == r]
                        x, y = reference_batch(cache, cfg, positions)
                        opt.zero_grad(set_to_none=True)
                        token_losses(reference(x), y).mean().backward()
                        clip_grad_norm_(reference.parameters(), cfg.optimizer.grad_clip)
                        for group in opt.param_groups:
                            group["lr"] = learning_rate(
                                cfg, stop * cfg.model.context_length, total_blocks * cfg.model.context_length
                            )
                    if mode == "decentralized" and scheme == "awc":
                        reference_mix(references, topology, step)
                    for reference, opt, history in zip(references, ref_opts, histories, strict=True):
                        before_step = [p.detach().clone() for p in reference.parameters()]
                        opt.step()
                        if cfg.distributed.error_feedback:
                            with torch.no_grad():
                                for p, base, previous in zip(reference.parameters(), before_step, history, strict=True):
                                    delta = p - base
                                    p.add_(delta).sub_(previous)
                                    previous.copy_(delta)
                    if mode == "decentralized" and scheme == "atc":
                        reference_mix(references, topology, step)
                    next_batch = Mock(wraps=transfer.next)
                    optimizer_update(
                        model,
                        forward,
                        optimizer,
                        cfg,
                        torch.device("cpu"),
                        next_batch,
                        transfer.preload,
                        start,
                        stop,
                        step,
                        total_blocks,
                    )
                    assert next_batch.call_count == cfg.training.num_accumulation
                    loader.commit(stop)
                    reference = references[worker if mode == "decentralized" else 0]
                    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-5)
                    reference_opt = ref_opts[worker if mode == "decentralized" else 0]
                    for actual, expected in zip(optimizer.state.values(), reference_opt.state.values(), strict=True):
                        for key in ("step", "exp_avg", "exp_avg_sq"):
                            torch.testing.assert_close(actual[key], expected[key], atol=3e-7, rtol=3e-5)
                    if isinstance(optimizer, ErrorFeedbackAdamW):
                        storage = parameter_storage(model)
                        for p, offset, previous in zip(
                            storage.parameters, storage.offsets, histories[worker], strict=True
                        ):
                            actual = optimizer.previous_update[offset : offset + p.numel()].view_as(p)
                            torch.testing.assert_close(actual, previous, atol=3e-7, rtol=3e-5)
                before = copy.deepcopy(model.state_dict())
                history_before = (
                    optimizer.previous_update.clone() if isinstance(optimizer, ErrorFeedbackAdamW) else None
                )
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
                if isinstance(optimizer, ErrorFeedbackAdamW):
                    torch.testing.assert_close(optimizer.previous_update, history_before, rtol=0, atol=0)
                checkpoint = cfg.runtime.output_dir / f"{size}-{mode}-{topology}-{scheme}"
                save_checkpoint(
                    checkpoint,
                    cfg,
                    cache,
                    model,
                    optimizer,
                    loader.state_dict(),
                    {"cursor": total_blocks, "step": 4, "completed_epochs": 1, "best_loss": 3.0, "best_epoch": 1},
                )
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(10)
                    if isinstance(optimizer, ErrorFeedbackAdamW):
                        optimizer.previous_update.fill_(123)
                manifest, _ = load_checkpoint(checkpoint, cfg, cache, model, optimizer)
                assert manifest["cursor"] == total_blocks
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                if isinstance(optimizer, ErrorFeedbackAdamW):
                    torch.testing.assert_close(optimizer.previous_update, history_before, rtol=0, atol=0)
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
@pytest.mark.parametrize(
    "size,factor,compiled,reuse,feedback,accumulation",
    [
        (2, 1, False, True, False, 1),
        (2, 1, False, True, False, 2),
        (4, 1, False, True, False, 1),
        (4, 3, False, True, False, 2),
        (4, 1, True, True, False, 2),
        (2, 1, True, False, False, 1),
        (2, 1, False, True, True, 2),
        (4, 1, True, False, True, 1),
    ],
)
def test_distributed_serial_references(
    cache: TokenCache,
    tiny_config: Config,
    size: int,
    factor: int,
    compiled: bool,
    reuse: bool,
    feedback: bool,
    accumulation: int,
) -> None:
    tiny_config.training.data_reuse_factor = factor
    tiny_config.training.num_accumulation = accumulation
    tiny_config.distributed.reuse_mix_buffers = reuse
    tiny_config.distributed.mode = "decentralized"
    tiny_config.distributed.error_feedback = feedback
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
        current["training"].update(batch_tokens=32, num_accumulation=2)
        current["distributed"]["error_feedback"] = raw["distributed"]["error_feedback"] and mode == "decentralized"
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
        if cfg.distributed.error_feedback:
            torch.testing.assert_close(
                actual["optimizer"]["error_feedback"]["previous_update"],
                original["optimizer"]["error_feedback"]["previous_update"],
                rtol=0,
                atol=0,
            )
            for key, state in original["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(actual["optimizer"]["state"][key][name], value, rtol=0, atol=0)
        assert original["loader"] == actual["loader"]
        assert torch.equal(original["rng"]["torch"], actual["rng"]["torch"])


@pytest.mark.distributed
@pytest.mark.parametrize(
    "size,factor,feedback,scheme",
    [
        (2, 1, False, "awc"),
        (4, 1, False, "awc"),
        (2, 4, False, "awc"),
        (4, 7, False, "awc"),
        (2, 1, True, "awc"),
        (2, 1, True, "atc"),
    ],
)
def test_multiprocess_training_resume(
    cache: TokenCache, tiny_config: Config, size: int, factor: int, feedback: bool, scheme: str
) -> None:
    tiny_config.training.log_every = 4
    tiny_config.training.data_reuse_factor = factor
    tiny_config.distributed.mode = "decentralized"
    tiny_config.distributed.error_feedback = feedback
    tiny_config.distributed.scheme = scheme  # type: ignore[assignment]
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage] # Public torch API missing from its export stubs.
        resume_worker,
        args=(size, [free_port() for _ in range(6)], tiny_config.model_dump(mode="json")),
        nprocs=size,
        join=True,
    )
    for output in tiny_config.runtime.output_dir.iterdir():
        logged = [
            [json.loads(line) for line in (output / f"metrics-rank-{worker:05d}.jsonl").read_text().splitlines()]
            for worker in range(size)
        ]
        reference = [(row["step"], row["cursor"], row["global_tokens_per_second"]) for row in logged[0]]
        assert reference and all(row[2] > 0 for row in reference)
        for rows in logged[1:]:
            assert [(row["step"], row["cursor"], row["global_tokens_per_second"]) for row in rows] == reference
