"""Bounded production-pipeline measurements on a real distributed allocation."""

import math
import os
import socket
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.nccl import version as nccl_version

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config
from decentralized_llm.distributed import barrier, parameter_mix_report, rank, setup, world_size
from decentralized_llm.execution import build_execution, compiler_counters, optimizer_revision
from decentralized_llm.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.model import Llama
from decentralized_llm.optim import make_optimizer, optimizer_report
from decentralized_llm.parameter_storage import parameter_storage
from decentralized_llm.runtime import append_metric, atomic_json, environment
from decentralized_llm.train import optimizer_update


def thread_cpu_affinities() -> list[dict]:
    """Sample all live threads; a main-thread mask can hide unbound helper threads."""
    counts: Counter[tuple[int, ...]] = Counter()
    for task in Path("/proc/self/task").iterdir():
        try:
            counts[tuple(sorted(os.sched_getaffinity(int(task.name))))] += 1
        except ProcessLookupError:
            continue  # A helper thread can exit between enumeration and sampling.
    return [{"cpus": list(cpus), "threads": count} for cpus, count in sorted(counts.items())]


def rotation_period(config: Config) -> int:
    if config.distributed.mode == "ddp" or config.distributed.topology == "complete":
        return 1
    return (
        2 if config.distributed.topology == "one_peer_ring" else max(1, config.distributed.world_size.bit_length() - 1)
    )


def synchronized_seconds(start: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    value = torch.tensor(time.perf_counter() - start, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def benchmark_training(config: Config, *, device: torch.device | None = None) -> dict:
    from torch._dynamo.utils import calculate_time_spent

    compiler_before = calculate_time_spent()
    session_start = time.perf_counter()
    source_revision = optimizer_revision()
    import_seconds = time.time() - float(os.environ.get("DECENTRALIZED_WORKER_STARTED", time.time()))
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    owns_group = device is None
    if device is None:
        device = setup(config)
    try:
        setup_seconds = synchronized_seconds(session_start, device)
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        model = Llama(config.model, config.runtime.attention_backend).to(device)
        optimizer = make_optimizer(model, config, device)
        execution = build_execution(model, config, device)
        bench = config.benchmark
        cycle = rotation_period(config)
        chunk = math.ceil(bench.steps / cycle) * cycle
        warmup = math.ceil(bench.warmup / cycle) * cycle
        # A bounded upper limit permits duration-based windows without unbounded read-ahead.
        maximum_steps = warmup + chunk * (bench.windows * 8 + 1)
        blocks = config.training.batch_tokens // config.model.context_length
        total = maximum_steps * blocks
        windows = []
        step = 0
        with TokenLoader(cache, config, rank(), 0, total) as loader:
            # Finish preload before warming compute so later variants do not idle
            # between their warmup and timing windows.
            loader.warmup(resident=bench.wait_for_resident)
            transfer = DevicePrefetcher(loader, device)
            try:
                transfer.preload()

                def update() -> None:
                    nonlocal step
                    metrics = optimizer_update(
                        model,
                        execution,
                        optimizer,
                        config,
                        device,
                        transfer.next,
                        transfer.preload,
                        step * blocks,
                        (step + 1) * blocks,
                        step,
                        total,
                        collect_metrics=(step + 1) % config.training.log_every == 0,
                    )
                    step += 1
                    loader.commit(step * blocks)
                    if metrics:
                        append_metric(output / f"metrics-rank-{rank():05d}.jsonl", {"step": step, **metrics})

                first_started = time.perf_counter()
                update()
                first_update_seconds = synchronized_seconds(first_started, device)
                for _ in range(warmup - 1):
                    update()
                warmup_seconds = synchronized_seconds(first_started, device)
                # Per-rank compiler phases overlap; they are not additive wall time.
                compilation_seconds = {
                    key: value - compiler_before.get(key, 0.0)
                    for key, value in calculate_time_spent().items()
                    if value > compiler_before.get(key, 0.0)
                }
                resident_at_measurement = loader.metrics()["resident_complete"]
                startup_seconds = synchronized_seconds(session_start, device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(bench.windows):
                    barrier()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    begin_step = step
                    started = time.perf_counter()
                    elapsed = 0.0
                    for _attempt in range(8):
                        for _ in range(chunk):
                            update()
                        elapsed = synchronized_seconds(started, device)
                        if elapsed >= bench.min_seconds:
                            break
                    if elapsed < bench.min_seconds:
                        raise RuntimeError("bounded benchmark window did not reach its minimum duration")
                    tokens = (step - begin_step) * config.training.batch_tokens
                    windows.append(
                        {"updates": step - begin_step, "seconds": elapsed, "global_tokens_per_second": tokens / elapsed}
                    )
                if bench.profile:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
                        for _ in range(cycle):
                            update()
                    profile.export_chrome_trace(str(output / f"trace-rank-{rank():05d}.json"))
                    (output / f"operators-rank-{rank():05d}.txt").write_text(
                        profile.key_averages().table(
                            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
                            row_limit=40,
                        )
                    )
                median = statistics.median(window["global_tokens_per_second"] for window in windows)
                storage = parameter_storage(model)
                parameter_bytes = sum(p.numel() * p.element_size() for p in storage.parameters)
                result = {
                    "rank": rank(),
                    "world_size": world_size(),
                    "hostname": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "nccl": nccl_version() if device.type == "cuda" else None,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "thread_cpu_affinities": thread_cpu_affinities(),
                    "setup_seconds": setup_seconds,
                    "worker_import_seconds": import_seconds,
                    "first_update_seconds": first_update_seconds,
                    "warmup_seconds": warmup_seconds,
                    "compilation_seconds_by_phase": compilation_seconds,
                    "startup_seconds": startup_seconds,
                    "windows": windows,
                    "global_tokens_per_second": median,
                    "per_gpu_tokens_per_second": median / world_size(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                    "parameter_storage": {
                        "parameter_bytes": parameter_bytes,
                        "communication_bytes": storage.flat.nbytes,
                        "padding_bytes": storage.flat.nbytes - parameter_bytes,
                        "allocation_bytes": storage.flat.untyped_storage().nbytes(),
                    },
                    "optimizer": {**optimizer_report(optimizer), "source_revision": source_revision},
                    "error_feedback": {
                        key: optimizer_report(optimizer)[key] for key in ("history_bytes", "scratch_bytes")
                    },
                    "parameter_mixing": parameter_mix_report(model),
                    "loader": loader.metrics(),
                    "resident_at_measurement": resident_at_measurement,
                    "execution": execution.policy,
                    "compile_cache": execution.cache,
                    "compiler_counters": compiler_counters() if execution.cache else {},
                    "cache_identity": cache.manifest["identity"],
                    "config": config.model_dump(mode="json"),
                    "environment": environment(),
                    "session_seconds": synchronized_seconds(session_start, device),
                }
                if optimizer_revision() != source_revision:
                    raise RuntimeError("optimizer sources changed during measurement; discard this run")
                atomic_json(output / f"benchmark-rank-{rank():05d}.json", result)
                if rank() == 0:
                    atomic_json(output / "benchmark.json", result)
                return result
            finally:
                transfer.close()
    finally:
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()
