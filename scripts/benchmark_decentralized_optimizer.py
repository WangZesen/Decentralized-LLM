"""Paired optimizer throughput matrix; run under torchrun in a GH200 allocation."""

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from decentralized_llm.backends.distributed.runtime import prepare_parameter_groups, rank, setup
from decentralized_llm.benchmarks.training import benchmark_training
from decentralized_llm.config import Config, load_config
from decentralized_llm.execution.qualification import optimizer_revision
from decentralized_llm.runtime import atomic_json, environment


def summarize(rows: list[dict]) -> dict:
    cases = {}
    for row in rows:
        key = (row["model"], row["topology"], row["scheme"], row["error_feedback_gamma"])
        cases.setdefault(key, {}).setdefault(row["selector"], {})[row["trial"]] = row["tokens_per_second"]
    comparisons = []
    for (model, topology, scheme, gamma), variants in cases.items():
        if set(variants) != {"auto", "reference"}:
            continue
        trials = sorted(variants["reference"].keys() & variants["auto"].keys())
        if not trials:
            continue
        reference, candidate = (
            statistics.median(variants[key][trial] for trial in trials) for key in ("reference", "auto")
        )
        comparisons.append(
            {
                "model": model,
                "topology": topology,
                "scheme": scheme,
                "error_feedback_gamma": gamma,
                "reference_tokens_per_second": reference,
                "fused_tokens_per_second": candidate,
                "speedup": statistics.median(
                    variants["auto"][trial] / variants["reference"][trial] for trial in trials
                ),
            }
        )
    models = sorted({row["model"] for row in comparisons})
    gains = {
        model: math.exp(statistics.mean(math.log(row["speedup"]) for row in comparisons if row["model"] == model))
        for model in models
    }
    medians = {
        model: statistics.median(row["speedup"] for row in comparisons if row["model"] == model) for model in models
    }
    return {
        "comparisons": comparisons,
        "model_geomean_speedup": gains,
        "model_median_speedup": medians,
        "passed": bool(gains)
        and all(gain > 1 for gain in gains.values())
        and all(gain > 1 for gain in medians.values())
        and all(row["speedup"] >= 0.98 for row in comparisons),
    }


def cached_pair(config: Config, parent: Path, revision: str, worker: int) -> dict[str, dict] | None:
    """Reuse only complete, matching pairs measured in the same SLURM allocation."""
    reports = {}
    for selector in ("reference", "auto"):
        expected = config.model_dump(mode="json")
        expected["runtime"].update(decentralized_optimizer=selector, output_dir=str(parent / selector))
        try:
            report = json.loads((parent / selector / f"benchmark-rank-{worker:05d}.json").read_text())
        except FileNotFoundError, ValueError:
            return None
        if report.get("config") != expected or report.get("optimizer", {}).get("source_revision") != revision:
            return None
        reports[selector] = report
    job = reports["reference"].get("slurm_job_id")
    if job is None or reports["auto"].get("slurm_job_id") != job:
        return None
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--models", nargs="+", choices=("20m", "90m"), default=["20m", "90m"])
    parser.add_argument("--schemes", nargs="+", choices=("awc", "atc"), default=["awc", "atc"])
    parser.add_argument("--resume", action="store_true", help="reuse complete pairs from matching source/configuration")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    config = load_config(args.config, args.overrides)
    config.distributed.mode = "decentralized"
    device = setup(config)
    rows = []
    source_revision = optimizer_revision()
    try:
        prepare_parameter_groups("one_peer_ring")
        for trial in range(args.trials):
            for model in args.models:
                current = load_config([*args.config, Path(f"configs/models/{model}.yaml")], args.overrides)
                current.distributed.mode = "decentralized"
                for topology in ("complete", "one_peer_ring"):
                    current.distributed.topology = topology
                    for scheme in args.schemes:
                        current.distributed.scheme = scheme
                        for gamma in (0.0, 1.0):
                            current.distributed.error_feedback_gamma = gamma
                            current.benchmark.profile = config.benchmark.profile and trial == 0
                            parent = args.output / model / topology / scheme / str(gamma) / str(trial)
                            cached = cached_pair(current, parent, source_revision, rank()) if args.resume else None
                            ready = torch.tensor(int(cached is not None), device=device)
                            if dist.is_initialized():
                                dist.all_reduce(ready, op=dist.ReduceOp.MIN)
                            if not bool(ready):
                                cached = None
                            selectors = ("reference", "auto") if trial % 2 == 0 else ("auto", "reference")
                            forward_policy = None
                            for selector in selectors:
                                current.runtime.decentralized_optimizer = selector
                                current.benchmark.profile = config.benchmark.profile and trial == 0
                                directory = args.output / model / topology / scheme / str(gamma) / str(trial) / selector
                                current.runtime.output_dir = directory
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.manual_seed(current.runtime.seed)
                                if optimizer_revision() != source_revision:
                                    raise RuntimeError("optimizer sources changed during the matrix; discard this run")
                                result = (
                                    cached[selector]
                                    if cached is not None
                                    else benchmark_training(current, device=device)
                                )
                                policy = {key: value for key, value in result["execution"].items() if key != "reason"}
                                if forward_policy is not None and policy != forward_policy:
                                    raise RuntimeError(
                                        "forward policies differ; set execution_backend and compile explicitly"
                                    )
                                forward_policy = policy
                                if selector == "auto" and result["optimizer"]["implementation"] != "triton":
                                    raise RuntimeError("benchmark did not select the fused optimizer")
                                row = {
                                    "model": model,
                                    "topology": topology,
                                    "scheme": scheme,
                                    "error_feedback_gamma": gamma,
                                    "selector": selector,
                                    "trial": trial,
                                    "tokens_per_second": result["global_tokens_per_second"],
                                    "report": str(directory / "benchmark.json"),
                                }
                                rows.append(row)
                                if rank() == 0:
                                    print(row, flush=True)
                                    summary = summarize(rows)
                                    summary["complete"] = (
                                        len(rows) == args.trials * len(args.models) * len(args.schemes) * 8
                                    )
                                    summary["passed"] = summary["passed"] and summary["complete"]
                                    atomic_json(
                                        args.output / "matrix.json",
                                        {
                                            "rows": rows,
                                            "summary": summary,
                                            "trials": args.trials,
                                            "optimizer_revision": source_revision,
                                            "environment": environment(),
                                        },
                                    )
        summary = summarize(rows)
        if rank() == 0:
            print(summary, flush=True)
        if not summary["passed"]:
            raise RuntimeError("optimizer throughput gates failed; see matrix.json")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
