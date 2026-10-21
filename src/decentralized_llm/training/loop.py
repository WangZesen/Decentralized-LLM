"""Shared training lifecycle, global token budgets and committed recovery."""

import fcntl
import os
import socket
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from decentralized_llm.backends.api import Runtime
from decentralized_llm.backends.factory import create_backend, runtime_for
from decentralized_llm.config import Config, save_config
from decentralized_llm.data.cache import TokenCache, training_boundaries
from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.data.sampling import check_memory, sampling_plan
from decentralized_llm.execution.compilation import compiler_counters
from decentralized_llm.runtime import NonfiniteUpdateError, append_metric, atomic_json, environment, restore_rng
from decentralized_llm.training.checkpoints import load_checkpoint, save_checkpoint
from decentralized_llm.training.evaluation import evaluate


class ThroughputWindow:
    """Count committed tokens against active training time between log records."""

    def __init__(self, cursor: int, device: torch.device, runtime: Runtime | None = None):
        self.runtime = runtime or runtime_for(Config())
        self.cursor = cursor
        self.device = device
        self.seconds = 0.0
        self.started: float | None = None

    def start(self) -> None:
        self.started = time.perf_counter()

    def pause(self) -> None:
        if self.started is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.seconds += time.perf_counter() - self.started
            self.started = None

    def measure(self, cursor: int, context_length: int) -> float:
        self.pause()
        elapsed = torch.tensor(self.seconds, device=self.device, dtype=torch.float64)
        self.runtime.reduce(elapsed, maximum=True)
        throughput = (cursor - self.cursor) * context_length / max(float(elapsed), 1e-12)
        self.cursor, self.seconds = cursor, 0.0
        return throughput


def train(config: Config, resume: Path | None = None, *, stop_after_steps: int | None = None) -> dict:
    session_started = time.perf_counter()
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    progress: dict[str, Any] = {}
    runtime = runtime_for(config)
    try:
        device = runtime.setup(config)
        if runtime.rank == 0:
            lock = (output / ".run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume is None and (output / "run.json").exists():
                raise ValueError("output already contains a training run; use resume or a new output directory")
        runtime.barrier()
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        memory = check_memory(cache, config)
        boundaries = training_boundaries(config, config.model.parameter_count)
        total = boundaries[-1]
        atomic_json(
            output / f"rank-{runtime.rank:05d}.json",
            {
                "rank": runtime.rank,
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            },
        )
        backend = create_backend(config, device, runtime)
        model, optimizer = backend.model, backend.optimizer
        progress = {
            "cursor": 0,
            "step": 0,
            "completed_epochs": 0,
            "best_loss": None,
            "best_epoch": None,
        }
        if resume is not None:
            manifest, rng = load_checkpoint(resume, config, cache, model, optimizer, backend=backend)
            progress = {key: manifest[key] for key in progress}
            restore_rng(rng)
        if runtime.rank == 0:
            name = "config.yaml" if resume is None else f"resume-{time.time_ns()}.yaml"
            save_config(config, output / name)
            atomic_json(
                output / "run.json",
                {
                    "environment": environment(),
                    "world_size": config.distributed.world_size,
                    "training_backend": config.runtime.training_backend,
                    "physical_world_size": config.physical_world_size,
                    "physical_gpu_count": config.physical_gpu_count,
                    "initialization": model.initialization,
                    "num_accumulation": config.training.num_accumulation,
                    "micro_batch_size": config.micro_batch_size,
                    "cache": cache.manifest["identity"],
                    "boundaries": boundaries,
                    "sampling": sampling_plan(cache, config).summary(),
                    "loader_mode": "resident" if memory["resident_tokens"] else "streaming",
                    "memory_bytes_per_rank": memory,
                    "resume": str(resume) if resume else None,
                },
            )
        backend.prepare()
        forward = backend.execution
        optimizer = backend.optimizer
        atomic_json(
            output / f"execution-rank-{runtime.rank:05d}.json", {"policy": forward.policy, "cache": forward.cache}
        )
        selected = config.training.saved_epochs()
        batch_blocks = config.training.batch_tokens // config.model.context_length
        with TokenLoader(cache, config, runtime.rank, progress["cursor"], total) as loader:
            transfer = DevicePrefetcher(loader, device)
            try:
                loader.warmup()
                transfer.preload()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                first_update_seconds = None
                training_seconds = 0.0
                window = ThroughputWindow(progress["cursor"], device, runtime)
                for epoch, boundary in enumerate(boundaries, 1):
                    if epoch <= progress["completed_epochs"]:
                        continue
                    segment_started = time.perf_counter()
                    window.start()
                    while progress["cursor"] < boundary:
                        start = progress["cursor"]
                        stop = min(boundary, start + batch_blocks)
                        log_step = (
                            (progress["step"] + 1) % config.training.log_every == 0
                            or stop == total
                            or stop_after_steps is not None
                            and progress["step"] + 1 >= stop_after_steps
                        )
                        metrics = backend.update(
                            transfer.next,
                            transfer.preload,
                            start,
                            stop,
                            progress["step"],
                            total,
                            collect_metrics=log_step,
                        )
                        if first_update_seconds is None:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            first_update_seconds = time.perf_counter() - session_started
                        # Read-ahead and transferred data are deliberately absent from durable state.
                        progress["cursor"] = stop
                        progress["step"] += 1
                        loader.commit(stop)
                        if log_step:
                            throughput = window.measure(stop, config.model.context_length)
                            append_metric(
                                output / f"metrics-rank-{runtime.rank:05d}.jsonl",
                                {
                                    "event": "train",
                                    "step": progress["step"],
                                    "cursor": stop,
                                    **metrics,
                                    "global_tokens_per_second": throughput,
                                },
                            )
                            if stop < boundary:
                                window.start()
                        if stop_after_steps is not None and progress["step"] >= stop_after_steps and stop < boundary:
                            save_checkpoint(
                                output / f"step-{progress['step']:08d}",
                                config,
                                cache,
                                model,
                                optimizer,
                                loader.state_dict(),
                                progress,
                                backend=backend,
                            )
                            return progress
                    window.pause()
                    epoch_training_seconds = time.perf_counter() - segment_started
                    training_seconds += epoch_training_seconds
                    result = evaluate(model, cache, config, device, full=False, backend=backend)
                    progress["completed_epochs"] = epoch
                    if progress["best_loss"] is None or result["loss"] < progress["best_loss"]:
                        progress["best_loss"], progress["best_epoch"] = result["loss"], epoch
                        if runtime.rank == 0:
                            atomic_json(output / "best.json", {"epoch": epoch, "loss": result["loss"]})
                    if runtime.rank == 0:
                        append_metric(
                            output / "metrics.jsonl",
                            {
                                "event": "validation",
                                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                                "epoch": epoch,
                                **{key: value for key, value in result.items() if key != "seconds"},
                                "val_seconds": round(result["seconds"], 2),
                                "train_seconds": round(epoch_training_seconds, 2),
                            },
                        )
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
                            backend=backend,
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
                            backend=backend,
                        )
                        return progress
                result = evaluate(model, cache, config, device, full=True, backend=backend)
                if config.training.checkpoint_policy != "none":
                    save_checkpoint(
                        output / "final",
                        config,
                        cache,
                        model,
                        optimizer,
                        loader.state_dict(),
                        progress,
                        backend=backend,
                    )
                if runtime.rank == 0:
                    atomic_json(
                        output / "result.json",
                        {
                            **progress,
                            "validation": result,
                            "seconds": time.perf_counter() - started,
                            "session_seconds": time.perf_counter() - session_started,
                            "first_update_seconds": first_update_seconds,
                            "training_seconds": training_seconds,
                            "execution": forward.policy,
                            "compiler_counters": compiler_counters() if forward.cache else {},
                        },
                    )
                return {**progress, "validation": result}
            finally:
                transfer.close()
                atomic_json(output / f"loader-rank-{runtime.rank:05d}.json", loader.metrics())
    except BaseException as error:
        failed_rank = int(os.environ.get("RANK", "0"))
        failure: dict[str, Any] = {"traceback": traceback.format_exc()}
        if isinstance(error, NonfiniteUpdateError):
            failure.update(
                kind="nonfinite_update",
                rank=failed_rank,
                message=str(error),
                progress={key: progress[key] for key in ("cursor", "step", "completed_epochs") if key in progress},
            )
        atomic_json(output / f"failure-rank-{failed_rank:05d}.json", failure)
        raise
    finally:
        if lock is not None:
            lock.close()
        runtime.close()


def evaluate_checkpoint(config: Config, checkpoint: Path, full: bool = True) -> dict:
    runtime = runtime_for(config)
    device = runtime.setup(config)
    try:
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        backend = create_backend(config, device, runtime)
        model = backend.model
        load_checkpoint(checkpoint, config, cache, model, optimizer=None, backend=backend)
        result = dict(evaluate(model, cache, config, device, full, backend=backend))
        if runtime.rank == 0:
            print(result, flush=True)
        return result
    finally:
        runtime.close()
