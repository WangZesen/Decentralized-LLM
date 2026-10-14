"""Compare local AdamW step time with and without the contiguous update correction."""

import argparse
import statistics
import time
from pathlib import Path

import torch

from decentralized_llm.config import Config, load_config
from decentralized_llm.model import Llama
from decentralized_llm.optim import ErrorFeedbackAdamW, make_optimizer
from decentralized_llm.parameter_storage import parameter_storage
from decentralized_llm.runtime import atomic_json, environment


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(config: Config, feedback: bool, warmup: int, steps: int, windows: int) -> dict:
    config = config.model_copy(deep=True)
    config.distributed.mode = "decentralized"
    config.distributed.error_feedback = feedback
    device = torch.device(config.runtime.device)
    torch.manual_seed(config.runtime.seed)
    model = Llama(config.model, config.runtime.attention_backend).to(device)
    optimizer = make_optimizer(model, config, device)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    for _ in range(warmup):
        optimizer.step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(windows):
        synchronize(device)
        started = time.perf_counter()
        for _ in range(steps):
            optimizer.step()
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000 / steps)
    return {
        "enabled": feedback,
        "median_step_ms": statistics.median(timings),
        "windows_ms": timings,
        "parameter_count": model.parameter_count,
        "padded_parameter_bytes": parameter_storage(model).flat.nbytes,
        "history_bytes": optimizer.previous_update.nbytes if isinstance(optimizer, ErrorFeedbackAdamW) else 0,
        "scratch_bytes": optimizer.scratch.nbytes if isinstance(optimizer, ErrorFeedbackAdamW) else 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.windows) < 1:
        parser.error("warmup, steps and windows must be positive")
    config = load_config(args.config, args.overrides)
    config.distributed.mode = "decentralized"
    torch.set_num_threads(config.runtime.cpu_threads)
    torch.use_deterministic_algorithms(config.runtime.deterministic)
    baseline = measure(config, False, args.warmup, args.steps, args.windows)
    corrected = measure(config, True, args.warmup, args.steps, args.windows)
    result = {
        "config": config.model_dump(mode="json"),
        "environment": environment(),
        "warmup": args.warmup,
        "steps": args.steps,
        "windows": args.windows,
        "baseline": baseline,
        "error_feedback": corrected,
        "step_overhead_fraction": corrected["median_step_ms"] / baseline["median_step_ms"] - 1,
        "scope": "Local optimizer steps with fixed synthetic gradients; excludes forward/backward and communication.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(
        {
            "baseline_step_ms": baseline["median_step_ms"],
            "corrected_step_ms": corrected["median_step_ms"],
            "step_overhead_fraction": result["step_overhead_fraction"],
            "additional_buffer_bytes": corrected["history_bytes"] + corrected["scratch_bytes"],
            "output": str(args.output),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
