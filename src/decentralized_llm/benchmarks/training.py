"""Bounded production-pipeline measurements on a real distributed allocation."""

import os
import socket
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.cuda.nccl import version as nccl_version

from decentralized_llm.backends.api import Runtime
from decentralized_llm.backends.factory import create_backend, runtime_for
from decentralized_llm.benchmarks.protocol import benchmark_bounds
from decentralized_llm.benchmarks.protocol import rotation_period as rotation_period
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.execution.compilation import compiler_counters
from decentralized_llm.execution.qualification import optimizer_revision
from decentralized_llm.runtime import atomic_json, environment


def thread_cpu_affinities() -> list[dict]:
    """Sample all live threads; a main-thread mask can hide unbound helper threads."""
    counts: Counter[tuple[int, ...]] = Counter()
    for task in Path("/proc/self/task").iterdir():
        try:
            counts[tuple(sorted(os.sched_getaffinity(int(task.name))))] += 1
        except ProcessLookupError:
            continue  # A helper thread can exit between enumeration and sampling.
    return [{"cpus": list(cpus), "threads": count} for cpus, count in sorted(counts.items())]


def compiler_delta(before: dict, after: dict) -> dict:
    """Retain positive compiler activity, including cache-backed retracing."""
    result = {}
    for group, counters in after.items():
        delta = {key: count - before.get(group, {}).get(key, 0) for key, count in counters.items()}
        changed = {key: count for key, count in delta.items() if count > 0}
        if changed:
            result[group] = changed
    return result


def phase_delta(before: dict, after: dict) -> dict:
    return {key: value - before.get(key, 0.0) for key, value in after.items() if value > before.get(key, 0.0)}


def peak_memory(device: torch.device) -> dict[str, int | None]:
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def placement(device: torch.device, runtime: Runtime) -> dict:
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    uuid = getattr(properties, "uuid", None)
    return {
        "rank": runtime.rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "hostname": socket.gethostname(),
        "device": str(device),
        "device_index": torch.cuda.current_device() if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_uuid": str(uuid) if uuid is not None else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def transport_metadata() -> dict:
    names = ("libnccl", "libfabric", "libcxi", "libhwloc", "libstdc++", "libgdrapi")
    libraries = sorted(
        {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if any(n in line for n in names)}
    )
    return {
        "network_environment": {
            key: os.environ[key]
            for key in (
                "NCCL_NET",
                "NCCL_NET_PLUGIN",
                "NCCL_SOCKET_IFNAME",
                "NCCL_NET_GDR_LEVEL",
                "NCCL_CROSS_NIC",
                "NCCL_LAUNCH_ORDER_IMPLICIT",
                "FI_PROVIDER",
                "FI_MR_CACHE_MONITOR",
                "OFI_NCCL_PROTOCOL",
                "FI_HMEM_CUDA_USE_DMABUF",
                "FI_CXI_ENABLE_WRITEDATA",
            )
            if key in os.environ
        },
        "loaded_libraries": libraries,
    }


def aggregate_measurements(ranks: list[dict]) -> dict:
    reasons = [f"rank {rank['rank']}: {reason}" for rank in ranks for reason in rank["invalid_reasons"]]
    result: dict = {"rank_measurements": ranks, "measurement": {"valid": not reasons, "invalid_reasons": reasons}}
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [rank[field] for rank in ranks if rank[field] is not None]
        result[f"max_{field}"] = max(values) if values else None
        result[f"sum_{field}"] = sum(values) if values else None
    return result


def throughput_statistics(windows: list[dict]) -> dict:
    speeds = [window["global_tokens_per_second"] for window in windows]
    mean = statistics.mean(speeds)
    stdev = statistics.pstdev(speeds)
    return {
        "min": min(speeds),
        "max": max(speeds),
        "mean": mean,
        "median": statistics.median(speeds),
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else 0.0,
    }


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
        bounds = benchmark_bounds(config)
        cycle, chunk, warmup = bounds.cycle, bounds.chunk, bounds.warmup
        blocks, total = bounds.blocks, bounds.total_blocks
        windows = []
        step = 0
        with TokenLoader(cache, config, runtime.rank, 0, total, bound_resident_to_stop=True) as loader:
            # Finish preload before warming compute so later variants do not idle
            # between their warmup and timing windows.
            loader.warmup(resident=bench.wait_for_resident)
            transfer = DevicePrefetcher(loader, device)
            try:
                transfer.preload()

                def update() -> None:
                    nonlocal step
                    backend.update(
                        transfer.next,
                        transfer.preload,
                        step * blocks,
                        (step + 1) * blocks,
                        step,
                        total,
                        collect_metrics=False,
                    )
                    step += 1
                    loader.commit(step * blocks)

                first_started = time.perf_counter()
                update()
                first_update_seconds = synchronized_seconds(first_started, device, runtime)
                for _ in range(warmup - 1):
                    update()
                warmup_seconds = synchronized_seconds(first_started, device, runtime)
                # Per-rank compiler phases overlap; they are not additive wall time.
                compilation_seconds = phase_delta(compiler_before, calculate_time_spent())
                loader_before = loader.metrics()
                resident_at_measurement = loader_before["resident_complete"]
                startup_seconds = synchronized_seconds(session_start, device, runtime)
                measured_counters_before = compiler_counters()
                measured_phases_before = calculate_time_spent()
                measured_peaks = None
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(bench.windows):
                    window_counters_before = compiler_counters()
                    window_phases_before = calculate_time_spent()
                    runtime.barrier()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    begin_step = step
                    started = time.perf_counter()
                    elapsed = 0.0
                    for _attempt in range(bench.max_chunks):
                        for _ in range(chunk):
                            update()
                        elapsed = synchronized_seconds(started, device, runtime)
                        if elapsed >= bench.min_seconds:
                            break
                    if elapsed < bench.min_seconds:
                        raise RuntimeError(
                            f"bounded benchmark window did not reach its minimum duration "
                            f"({elapsed:.3f}s < {bench.min_seconds}s after {bench.max_chunks} chunks)"
                        )
                    # Do this before compiler instrumentation, profiling or object collectives.
                    measured_peaks = peak_memory(device)
                    window_counters_delta = compiler_delta(window_counters_before, compiler_counters())
                    window_phases_delta = phase_delta(window_phases_before, calculate_time_spent())
                    tokens = (step - begin_step) * config.training.batch_tokens
                    windows.append(
                        {
                            "updates": step - begin_step,
                            "global_tokens": tokens,
                            "global_sequences": tokens // config.model.context_length,
                            "seconds": elapsed,
                            "seconds_per_update": elapsed / (step - begin_step),
                            "global_tokens_per_second": tokens / elapsed,
                            "global_sequences_per_second": tokens / config.model.context_length / elapsed,
                            "per_gpu_tokens_per_second": tokens / runtime.size / elapsed,
                            "compiler_counters_delta": window_counters_delta,
                            "compilation_seconds_by_phase": window_phases_delta,
                        }
                    )
                # The last window snapshot is the cumulative peak since the post-warmup reset.
                assert measured_peaks is not None
                measured_counters_after = compiler_counters()
                measured_phases_after = calculate_time_spent()
                loader_after = loader.metrics()
                counters_delta = compiler_delta(measured_counters_before, measured_counters_after)
                compilation_delta = phase_delta(measured_phases_before, measured_phases_after)
                read_bytes = loader_after["read_bytes"] - loader_before["read_bytes"]
                read_calls = loader_after["read_calls"] - loader_before["read_calls"]
                invalid_reasons = []
                if counters_delta or compilation_delta:
                    invalid_reasons.append("compiler_activity_during_measurement")
                if loader_before["loader_mode"] == "resident":
                    if not resident_at_measurement:
                        invalid_reasons.append("resident_preload_incomplete")
                    if read_bytes or read_calls:
                        invalid_reasons.append("resident_reads_during_measurement")
                rank_measurement = {
                    **placement(device, runtime),
                    **measured_peaks,
                    **transport_metadata(),
                    "resident_at_measurement": resident_at_measurement,
                    "timed_compiler_counters_delta": counters_delta,
                    "timed_compilation_seconds_by_phase": compilation_delta,
                    "timed_read_bytes": read_bytes,
                    "timed_read_calls": read_calls,
                    "valid": not invalid_reasons,
                    "invalid_reasons": invalid_reasons,
                }
                aggregate = aggregate_measurements(runtime.gather(rank_measurement))
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
                    **placement(device, runtime),
                    "world_size": config.distributed.world_size,
                    "training_backend": config.runtime.training_backend,
                    "physical_world_size": runtime.size,
                    "physical_gpu_count": config.physical_gpu_count,
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
                    "bounds": asdict(bounds),
                    "throughput_statistics": throughput_statistics(windows),
                    "global_tokens_per_second": median,
                    "global_sequences_per_second": median / config.model.context_length,
                    "seconds_per_update": statistics.median(window["seconds_per_update"] for window in windows),
                    "per_gpu_tokens_per_second": median / runtime.size,
                    **measured_peaks,
                    **aggregate,
                    "memory_scope": {
                        "measurement": "training updates after warmup, before profiling and result collection",
                        "allocated": "PyTorch allocator peak including persistent training state",
                        "reserved": "PyTorch allocator peak including retained startup cache",
                        "external_allocations_included": False,
                    },
                    "parameter_storage": {
                        "parameter_bytes": parameter_bytes,
                        "communication_bytes": memory["parameter_storage_bytes"],
                        "padding_bytes": memory["parameter_storage_bytes"] - parameter_bytes,
                        "allocation_bytes": memory.get("allocation_bytes", memory["parameter_storage_bytes"]),
                    },
                    "optimizer": {**memory["optimizer"], "source_revision": source_revision},
                    "error_feedback": {key: memory["optimizer"][key] for key in ("history_bytes", "scratch_bytes")},
                    "parameter_mixing": memory["parameter_mixing"],
                    "loader": loader_after,
                    "resident_at_measurement": resident_at_measurement,
                    "execution": execution.policy,
                    "compile_cache": execution.cache,
                    "compiler_counters": measured_counters_after if execution.cache else {},
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
