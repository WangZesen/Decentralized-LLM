"""One process per GPU, global token budgets and optimizer-committed recovery."""

import fcntl
import os
import socket
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from decentralized_llm.cache import TokenCache, training_boundaries
from decentralized_llm.checkpoints import load_checkpoint, save_checkpoint
from decentralized_llm.config import Config, save_config
from decentralized_llm.distributed import barrier, mix_parameters, rank, setup, world_size
from decentralized_llm.evaluation import evaluate
from decentralized_llm.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.model import Llama, token_losses
from decentralized_llm.optim import learning_rate, make_optimizer
from decentralized_llm.runtime import append_metric, atomic_json, clip_grad_norm_, environment, restore_rng
from decentralized_llm.sampling import BatchSpec, check_memory, rank_count
from decentralized_llm.state import Batch


def optimizer_update(
    model: nn.Module,
    forward: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    next_batch: Callable[[], tuple[Batch, BatchSpec]],
    preload: Callable[[], None],
    start: int,
    stop: int,
    step: int,
    total: int,
) -> dict[str, float]:
    length = config.model.context_length
    local_count = rank_count(start, stop, rank(), world_size())
    denominator = (stop - start) / world_size() if config.distributed.mode == "ddp" else local_count
    lr = learning_rate(config, stop * length, total * length)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    cursor = start
    loss_sum = torch.zeros((), device=device)
    while cursor < stop:
        (x, y), spec = next_batch()
        if spec.start != cursor or spec.stop > stop:
            raise ValueError("loader/optimizer global cursor mismatch")
        final_microbatch = spec.stop == stop
        sync = forward.no_sync() if isinstance(forward, DDP) and not final_microbatch else nullcontext()
        with sync:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.runtime.amp):
                losses = token_losses(forward(x), y).sum()
                loss = losses / (denominator * length)
            loss.backward()
        loss_sum += losses.detach()
        cursor = spec.stop
        # Compute is already queued on CUDA; enqueue the upcoming transfer alongside it.
        preload()
    norm = clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
    if config.distributed.mode == "decentralized" and config.distributed.scheme == "awc":
        mix_parameters(model, config.distributed.topology, step)
    optimizer.step()
    if config.distributed.mode == "decentralized" and config.distributed.scheme == "atc":
        mix_parameters(model, config.distributed.topology, step)
    return {"lr": lr, "gradient_norm": float(norm), "local_loss": float(loss_sum) / (local_count * length)}


def train(config: Config, resume: Path | None = None, *, stop_after_steps: int | None = None) -> dict:
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        device = setup(config)
        if rank() == 0:
            lock = (output / ".run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume is None and (output / "run.json").exists():
                raise ValueError("output already contains a training run; use resume or a new output directory")
        barrier()
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        check_memory(cache, config)
        boundaries = training_boundaries(config, config.model.parameter_count)
        total = boundaries[-1]
        if rank() == 0:
            name = "config.yaml" if resume is None else f"resume-{time.time_ns()}.yaml"
            save_config(config, output / name)
            atomic_json(
                output / "run.json",
                {
                    "environment": environment(),
                    "world_size": world_size(),
                    "cache": cache.manifest["identity"],
                    "boundaries": boundaries,
                    "resume": str(resume) if resume else None,
                },
            )
        atomic_json(
            output / f"rank-{rank():05d}.json",
            {
                "rank": rank(),
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
        )
        model = Llama(config.model, config.runtime.attention_backend).to(device)
        optimizer = make_optimizer(model, config, device)
        progress: dict[str, Any] = {
            "cursor": 0,
            "step": 0,
            "completed_epochs": 0,
            "best_loss": None,
            "best_epoch": None,
        }
        if resume is not None:
            manifest, rng = load_checkpoint(resume, config, cache, model, optimizer)
            progress = {key: manifest[key] for key in progress}
            restore_rng(rng)
        compiled = torch.compile(model, mode=config.runtime.compile_mode) if config.runtime.compile else model
        # torch.compile preserves Module behavior; its public overload returns a Callable.
        forward: nn.Module = cast(nn.Module, compiled)
        if config.distributed.mode == "ddp" and world_size() > 1:
            forward = DDP(
                compiled, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False
            )
        selected = config.training.saved_epochs()
        batch_blocks = config.training.batch_tokens // config.model.context_length
        with TokenLoader(cache, config, rank(), progress["cursor"], total) as loader:
            transfer = DevicePrefetcher(loader, device)
            try:
                loader.warmup()
                transfer.preload()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                initial_cursor = progress["cursor"]
                for epoch, boundary in enumerate(boundaries, 1):
                    if epoch <= progress["completed_epochs"]:
                        continue
                    while progress["cursor"] < boundary:
                        start = progress["cursor"]
                        stop = min(boundary, start + batch_blocks)
                        metrics = optimizer_update(
                            model,
                            forward,
                            optimizer,
                            config,
                            device,
                            transfer.next,
                            transfer.preload,
                            start,
                            stop,
                            progress["step"],
                            total,
                        )
                        # Read-ahead and transferred data are deliberately absent from durable state.
                        progress["cursor"] = stop
                        progress["step"] += 1
                        loader.commit(stop)
                        if progress["step"] % config.training.log_every == 0:
                            append_metric(
                                output / f"metrics-rank-{rank():05d}.jsonl",
                                {
                                    "event": "train",
                                    "step": progress["step"],
                                    "cursor": stop,
                                    **metrics,
                                    "global_tokens_per_second": (stop - initial_cursor)
                                    * config.model.context_length
                                    / (time.perf_counter() - started),
                                    "loader": loader.metrics(),
                                },
                            )
                        if stop_after_steps is not None and progress["step"] >= stop_after_steps and stop < boundary:
                            save_checkpoint(
                                output / f"step-{progress['step']:08d}",
                                config,
                                cache,
                                model,
                                optimizer,
                                loader.state_dict(),
                                progress,
                            )
                            return progress
                    result = evaluate(model, cache, config, device, full=False)
                    progress["completed_epochs"] = epoch
                    if progress["best_loss"] is None or result["loss"] < progress["best_loss"]:
                        progress["best_loss"], progress["best_epoch"] = result["loss"], epoch
                        if rank() == 0:
                            atomic_json(output / "best.json", {"epoch": epoch, "loss": result["loss"]})
                    if rank() == 0:
                        append_metric(output / "metrics.jsonl", {"event": "validation", "epoch": epoch, **result})
                    if epoch in selected:
                        save_checkpoint(
                            output / f"epoch-{epoch:03d}",
                            config,
                            cache,
                            model,
                            optimizer,
                            loader.state_dict(),
                            progress,
                            config.training.save_epoch_training_state,
                        )
                    if stop_after_steps is not None and progress["step"] >= stop_after_steps and boundary < total:
                        save_checkpoint(
                            output / f"step-{progress['step']:08d}",
                            config,
                            cache,
                            model,
                            optimizer,
                            loader.state_dict(),
                            progress,
                        )
                        return progress
                result = evaluate(model, cache, config, device, full=True)
                if config.training.checkpoint_policy != "none":
                    save_checkpoint(output / "final", config, cache, model, optimizer, loader.state_dict(), progress)
                if rank() == 0:
                    atomic_json(
                        output / "result.json",
                        {**progress, "validation": result, "seconds": time.perf_counter() - started},
                    )
                return {**progress, "validation": result}
            finally:
                transfer.close()
                atomic_json(output / f"loader-rank-{rank():05d}.json", loader.metrics())
    except BaseException:
        failed_rank = int(os.environ.get("RANK", "0"))
        atomic_json(output / f"failure-rank-{failed_rank:05d}.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if lock is not None:
            lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate_checkpoint(config: Config, checkpoint: Path, full: bool = True) -> dict:
    device = setup(config)
    try:
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        model = Llama(config.model, config.runtime.attention_backend).to(device)
        load_checkpoint(checkpoint, config, cache, model, optimizer=None)
        result = dict(evaluate(model, cache, config, device, full))
        if rank() == 0:
            print(result, flush=True)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
