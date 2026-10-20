"""Measure snapshot and installation overhead; the legacy path is a benchmark-only reference."""

import argparse
import resource
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from decentralized_llm.config import load_config
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.storage import prepare_parameter_storage
from decentralized_llm.runtime import atomic_json, environment


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(operation: Callable[[], object], device: torch.device, warmup: int, steps: int, windows: int) -> dict:
    for _ in range(warmup):
        operation()
    timings = []
    for _ in range(windows):
        synchronize(device)
        started = time.perf_counter()
        for _ in range(steps):
            operation()
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000 / steps)
    return {"median_ms": statistics.median(timings), "windows_ms": timings}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--layout", choices=("legacy", "contiguous"), default="contiguous")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.windows) < 1:
        parser.error("warmup, steps and windows must be positive")
    config = load_config(args.config, args.overrides)
    device = torch.device(config.runtime.device)
    torch.set_num_threads(config.runtime.cpu_threads)
    torch.manual_seed(config.runtime.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = Llama(config.model, config.runtime.attention_backend).to(device)
    synchronize(device)
    model_init_seconds = time.perf_counter() - started
    parameters = tuple(model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
    started = time.perf_counter()
    storage = prepare_parameter_storage(model) if args.layout == "contiguous" else None
    synchronize(device)
    preparation_seconds = time.perf_counter() - started
    flat = torch.empty(sum(p.numel() for p in parameters) if storage is None else storage.flat.numel(), device=device)

    def snapshot() -> None:
        if storage is None:
            sources = [p.detach().reshape(-1) for p in model.parameters()]
            torch.cat(sources, out=flat)
        else:
            flat.copy_(storage.flat)

    def apply() -> None:
        # Divisor one keeps values stable across repetitions without a fake
        # collective. Both paths still execute the same normalization kernel.
        flat.div_(1)
        if storage is None:
            chunks = flat.split([p.numel() for p in parameters])
            sources = [chunk.view_as(p) for chunk, p in zip(chunks, parameters, strict=True)]
            torch._foreach_copy_(parameters, sources)  # pyright: ignore[reportPrivateImportUsage]
        else:
            storage.copy_from(flat)

    def cycle() -> None:
        snapshot()
        apply()

    snapshot()
    startup_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measurements = {
        name: measure(operation, device, args.warmup, args.steps, args.windows)
        for name, operation in (("snapshot", snapshot), ("apply", apply), ("cycle", cycle))
    }
    result = {
        "layout": args.layout,
        "config": config.model_dump(mode="json"),
        "warmup": args.warmup,
        "steps": args.steps,
        "windows": args.windows,
        "model_init_seconds": model_init_seconds,
        "preparation_seconds": preparation_seconds,
        "parameter_bytes": parameter_bytes,
        "communication_bytes": flat.nbytes,
        "padding_bytes": flat.nbytes - parameter_bytes,
        "live_allocation_bytes": storage.flat.untyped_storage().nbytes() if storage else parameter_bytes,
        "startup_peak_allocated_bytes": startup_peak,
        "steady_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "measurements": measurements,
        "environment": environment(),
        "scope": "Local reusable snapshot and installation only; no collectives or forward/backward.",
    }
    atomic_json(args.output, result)
    print({"layout": args.layout, "measurements": measurements, "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
