"""Bounded production-pipeline measurements on a real distributed allocation."""

import math
import os
import socket
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from torch.cuda.nccl import version as nccl_version

from decentralized_llm.backends.api import Runtime
from decentralized_llm.backends.factory import create_backend, runtime_for
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.execution.compilation import compiler_counters
from decentralized_llm.execution.qualification import optimizer_revision
from decentralized_llm.runtime import append_metric, atomic_json, environment


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


def synchronized_seconds(start: float, device: torch.device, runtime: Runtime) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    value = torch.tensor(time.perf_counter() - start, device=device, dtype=torch.float64)
    runtime.reduce(value, maximum=True)
    return float(value)


def benchmark_training(config: Config, *, device: torch.device | None = None) -> dict:
    from torch._dynamo.utils import calculate_time_spent

    compiler_before = calculate_time_spent()
    session_start = time.perf_counter()
    source_revision = optimizer_revision()
    import_seconds = time.time() - float(os.environ.get("DECENTRALIZED_WORKER_STARTED", time.time()))
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_for(config)
    owns_group = device is None
    if device is None:
        device = runtime.setup(config)
    try:
        setup_seconds = synchronized_seconds(session_start, device, runtime)
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        backend = create_backend(config, device, runtime)
        backend.prepare()
        execution = backend.execution
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
        with TokenLoader(cache, config, runtime.rank, 0, total) as loader:
            # Finish preload before warming compute so later variants do not idle
            # between their warmup and timing windows.
            loader.warmup(resident=bench.wait_for_resident)
            transfer = DevicePrefetcher(loader, device)
            try:
                transfer.preload()

                def update() -> None:
                    nonlocal step
                    metrics = backend.update(
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
                        append_metric(output / f"metrics-rank-{runtime.rank:05d}.jsonl", {"step": step, **metrics})

                first_started = time.perf_counter()
                update()
                first_update_seconds = synchronized_seconds(first_started, device, runtime)
                for _ in range(warmup - 1):
                    update()
                warmup_seconds = synchronized_seconds(first_started, device, runtime)
                # Per-rank compiler phases overlap; they are not additive wall time.
                compilation_seconds = {
                    key: value - compiler_before.get(key, 0.0)
                    for key, value in calculate_time_spent().items()
                    if value > compiler_before.get(key, 0.0)
                }
                resident_at_measurement = loader.metrics()["resident_complete"]
                startup_seconds = synchronized_seconds(session_start, device, runtime)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(bench.windows):
                    runtime.barrier()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    begin_step = step
                    started = time.perf_counter()
                    elapsed = 0.0
                    for _attempt in range(8):
                        for _ in range(chunk):
                            update()
                        elapsed = synchronized_seconds(started, device, runtime)
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
                    profile.export_chrome_trace(str(output / f"trace-rank-{runtime.rank:05d}.json"))
                    (output / f"operators-rank-{runtime.rank:05d}.txt").write_text(
                        profile.key_averages().table(
                            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
                            row_limit=40,
                        )
                    )
                median = statistics.median(window["global_tokens_per_second"] for window in windows)
                memory = backend.memory_report()
                parameter_bytes = memory["parameter_bytes"]
                result = {
                    "rank": runtime.rank,
                    "world_size": config.distributed.world_size,
                    "training_backend": config.runtime.training_backend,
                    "physical_world_size": runtime.size,
                    "physical_gpu_count": config.physical_gpu_count,
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
                    "per_gpu_tokens_per_second": median / runtime.size,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                    "parameter_storage": {
                        "parameter_bytes": parameter_bytes,
                        "communication_bytes": memory["parameter_storage_bytes"],
                        "padding_bytes": memory["parameter_storage_bytes"] - parameter_bytes,
                        "allocation_bytes": memory.get("allocation_bytes", memory["parameter_storage_bytes"]),
                    },
                    "optimizer": {**memory["optimizer"], "source_revision": source_revision},
                    "error_feedback": {key: memory["optimizer"][key] for key in ("history_bytes", "scratch_bytes")},
                    "parameter_mixing": memory["parameter_mixing"],
                    "loader": loader.metrics(),
                    "resident_at_measurement": resident_at_measurement,
                    "execution": execution.policy,
                    "compile_cache": execution.cache,
                    "compiler_counters": compiler_counters() if execution.cache else {},
                    "cache_identity": cache.manifest["identity"],
                    "config": config.model_dump(mode="json"),
                    "environment": environment(),
                    "session_seconds": synchronized_seconds(session_start, device, runtime),
                }
                if optimizer_revision() != source_revision:
                    raise RuntimeError("optimizer sources changed during measurement; discard this run")
                atomic_json(output / f"benchmark-rank-{runtime.rank:05d}.json", result)
                if runtime.rank == 0:
                    atomic_json(output / "benchmark.json", result)
                return result
            finally:
                transfer.close()
    finally:
        if owns_group:
            runtime.close()
