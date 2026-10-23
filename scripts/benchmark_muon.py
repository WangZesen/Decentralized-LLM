"""Paired Muon ablations and numerical checks on one CUDA GPU; no tuning submissions."""

import argparse
import copy
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Literal

import torch

from decentralized_llm.algorithms.adamw import make_optimizer
from decentralized_llm.algorithms.muon import MuonAdamW
from decentralized_llm.benchmarks.training import benchmark_training
from decentralized_llm.config import Config, load_config
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.runtime import atomic_json, environment, source_digest

MODES = ("adamw", "reference", "batched", "compiled", "cudagraph")
type Variant = Literal["adamw", "reference", "batched", "compiled", "cudagraph"]


def configure(config: Config, mode: Variant) -> Config:
    cfg = config.model_copy(deep=True)
    cfg.optimizer.name = "adamw" if mode == "adamw" else "muon_adamw"
    cfg.runtime.muon_optimizer = "reference" if mode == "adamw" else mode
    return cfg


def make_case(config: Config, mode: Variant):
    cfg = configure(config, mode)
    torch.manual_seed(cfg.runtime.seed)
    model = initialize_model(cfg).cuda()
    optimizer = make_optimizer(model, cfg, torch.device("cuda"))
    return model, optimizer


def comparison(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    ref, value = reference.double(), actual.double()
    return {
        "max_abs_error": float((value - ref).abs().max()),
        "relative_l2_error": float((value - ref).norm() / ref.norm().clamp_min(1e-30)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(ref.reshape(1, -1), value.reshape(1, -1))),
    }


@torch.no_grad()
def numerical_check(config: Config, mode: Variant) -> dict:
    reference, reference_opt = make_case(config, "reference")
    candidate, candidate_opt = make_case(config, mode)
    assert isinstance(reference_opt, MuonAdamW) and isinstance(candidate_opt, MuonAdamW)
    parameter_errors, update_errors, momentum_errors = [], [], []
    for step in range(8):
        rate = config.optimizer.lr * (1 - step / 8)
        for opt in (reference_opt, candidate_opt):
            for group in opt.param_groups:
                group["lr"] = rate * (config.optimizer.muon_lr_ratio if group["optimizer_kind"] == "muon" else 1)
        before_reference, before_candidate = [], []
        for p, q in zip(
            reference_opt.muon.param_groups[0]["params"], candidate_opt.muon.param_groups[0]["params"], strict=True
        ):
            before_reference.append(p.detach().clone())
            before_candidate.append(q.detach().clone())
        for p, q in zip(reference.parameters(), candidate.parameters(), strict=True):
            p.grad = torch.randn_like(p) * 0.01
            q.grad = p.grad.clone()
        reference_opt.step()
        candidate_opt.step()
        reference_updates, candidate_updates = [], []
        for index, (p, q) in enumerate(
            zip(reference_opt.muon.param_groups[0]["params"], candidate_opt.muon.param_groups[0]["params"], strict=True)
        ):
            reference_updates.append((p - before_reference[index]).reshape(-1))
            candidate_updates.append((q - before_candidate[index]).reshape(-1))
        update_errors.append(comparison(torch.cat(reference_updates), torch.cat(candidate_updates)))
        parameter_errors.append(
            comparison(
                torch.cat([p.detach().reshape(-1) for p in reference.parameters()]),
                torch.cat([p.detach().reshape(-1) for p in candidate.parameters()]),
            )
        )
        momentum_errors.append(
            comparison(
                torch.cat(
                    [
                        reference_opt.muon.state[p]["momentum_buffer"].reshape(-1)
                        for p in reference_opt.muon.param_groups[0]["params"]
                    ]
                ),
                torch.cat(
                    [
                        candidate_opt.muon.state[p]["momentum_buffer"].reshape(-1)
                        for p in candidate_opt.muon.param_groups[0]["params"]
                    ]
                ),
            )
        )
        if step == 3:
            # Loading canonical state must invalidate captured addresses.
            candidate_opt.load_state_dict(copy.deepcopy(candidate_opt.state_dict()))
    result = {
        "mode": mode,
        "steps": 8,
        "parameter_errors": parameter_errors,
        "update_errors": update_errors,
        "momentum_errors": momentum_errors,
        "passed": all(r["relative_l2_error"] < 0.01 and r["cosine_similarity"] > 0.9999 for r in update_errors)
        and all(r["relative_l2_error"] < 1e-6 for r in momentum_errors)
        and all(r["relative_l2_error"] < 0.001 for r in parameter_errors),
    }
    del reference, reference_opt, candidate, candidate_opt
    gc.collect()
    torch.cuda.empty_cache()
    return result


def optimizer_case(config: Config, mode: Variant, output: Path, profile: bool) -> dict:
    model, optimizer = make_case(config, mode)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter) * 0.01
    started = time.perf_counter()
    for _ in range(20):
        optimizer.step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    from torch._dynamo.utils import counters

    compiled_before = counters["stats"]["unique_graphs"]
    windows = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            optimizer.step()
        torch.cuda.synchronize()
        windows.append((time.perf_counter() - started) * 1000 / 100)
    row = {
        "mode": mode,
        "milliseconds_per_step": statistics.median(windows),
        "windows_ms": windows,
        "warmup_seconds": warmup_seconds,
        "timed_new_graphs": counters["stats"]["unique_graphs"] - compiled_before,
    }
    if profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            optimizer.step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / f"{mode}-trace.json"))
        cuda_type = torch.autograd.DeviceType.CUDA  # pyright: ignore[reportPrivateImportUsage]
        cuda_events = [e for e in (prof.events() or []) if e.device_type == cuda_type]
        row["profile"] = {
            "cuda_events": len(cuda_events),
            "cuda_time_us": sum(e.device_time_total for e in cuda_events),
            "operators": [
                {"name": e.key, "calls": e.count, "self_cpu_us": e.self_cpu_time_total}
                for e in prof.key_averages()
                if "mm" in e.key or "foreach" in e.key
            ],
        }
    if row["timed_new_graphs"]:
        raise RuntimeError(f"Compilation occurred inside timed windows: {row}")
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("optimizer", "training"))
    parser.add_argument("--config", "-c", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if not torch.cuda.is_available():
        parser.error("A CUDA allocation is required")
    if args.trials < 1:
        parser.error("trials must be positive")
    torch.set_num_threads(config.runtime.cpu_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "src/decentralized_llm"
    revision = source_digest(source)
    result = {
        "kind": args.kind,
        "config": config.model_dump(mode="json"),
        "environment": environment(),
        "source_sha256": revision,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "checks": [],
        "rows": [],
        "complete": False,
    }
    if args.kind == "optimizer":
        for mode in args.modes:
            if mode in ("adamw", "reference"):
                continue
            check = numerical_check(config, mode)
            result["checks"].append(check)
            atomic_json(args.output / "results.json", result)
            print("CHECK", mode, check["passed"], flush=True)
            if not check["passed"]:
                raise RuntimeError(f"Numerical gate failed for {mode}; see results.json")
    for trial in range(args.trials):
        order = args.modes[trial % len(args.modes) :] + args.modes[: trial % len(args.modes)]
        if trial % 2:
            order = list(reversed(order))
        for mode in order:
            if source_digest(source) != revision:
                raise RuntimeError("Sources changed during benchmark")
            if args.kind == "optimizer":
                row = optimizer_case(config, mode, args.output, profile=trial == 0)
            else:
                cfg = configure(config, mode)
                cfg.runtime.output_dir = args.output / f"trial-{trial}" / mode
                gc.collect()
                torch.cuda.empty_cache()
                measurement = benchmark_training(cfg)
                if not measurement["measurement"]["valid"]:
                    raise RuntimeError(f"Invalid training measurement: {measurement['measurement']}")
                row = {
                    "mode": mode,
                    "tokens_per_second": measurement["global_tokens_per_second"],
                    "milliseconds_per_step": 1000 * cfg.training.batch_tokens / measurement["global_tokens_per_second"],
                    "report": str(cfg.runtime.output_dir / "benchmark.json"),
                }
            row["trial"] = trial
            result["rows"].append(row)
            atomic_json(args.output / "results.json", result)
            print(json.dumps(row), flush=True)
    result["complete"] = True
    medians = {
        mode: statistics.median(r["milliseconds_per_step"] for r in result["rows"] if r["mode"] == mode)
        for mode in args.modes
    }
    result["summary"] = {
        mode: {"milliseconds_per_step": value, "speedup_vs_reference": medians.get("reference", value) / value}
        for mode, value in medians.items()
    }
    atomic_json(args.output / "results.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
