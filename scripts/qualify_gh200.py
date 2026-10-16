"""Matched distributed trajectories; invoke through torchrun/srun inside an allocation."""

import argparse
import copy
import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config, load_config
from decentralized_llm.distributed import rank, setup, world_size
from decentralized_llm.evaluation import evaluate
from decentralized_llm.execution import build_execution, compiler_counters
from decentralized_llm.loader import DevicePrefetcher, LoadedBatch, TokenLoader
from decentralized_llm.model import Llama
from decentralized_llm.optim import make_optimizer
from decentralized_llm.runtime import atomic_json, environment
from decentralized_llm.train import optimizer_update


class Replay:
    def __init__(self, config: Config, batches: list[LoadedBatch]):
        self.config, self.batches = config, iter(batches)

    def next(self) -> LoadedBatch:
        return next(self.batches)


def relative_error(actual: Sequence[Tensor], reference: Sequence[Tensor]) -> float:
    numerator = sum(
        (a.detach().float() - b.detach().float()).square().sum() for a, b in zip(actual, reference, strict=True)
    )
    denominator = sum(b.detach().float().square().sum() for b in reference)
    assert isinstance(numerator, Tensor) and isinstance(denominator, Tensor)
    return float((numerator / denominator.clamp_min(1e-30)).sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--reference-compile", action="store_true", help="compare against compiled native instead of eager native"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = setup(config)
    try:
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        blocks = config.training.batch_tokens // config.model.context_length
        with TokenLoader(cache, config, rank(), 0, args.steps * blocks) as loader:
            batches = []
            while True:
                try:
                    batch = loader.next()
                    batches.append(LoadedBatch(batch.spec, batch.tokens.copy()))
                except StopIteration:
                    break
        native_cfg = config.model_copy(deep=True)
        native_cfg.runtime.execution_backend, native_cfg.runtime.compile = "native", args.reference_compile
        native_cfg.runtime.fused_rope = False
        native = Llama(config.model).to(device)
        fast = copy.deepcopy(native)
        native_execution = build_execution(native, native_cfg, device)
        fast_execution = build_execution(fast, config, device)
        optimizers = [make_optimizer(native, native_cfg, device), make_optimizer(fast, config, device)]
        transfers = [DevicePrefetcher(Replay(config, batches), device) for _ in range(2)]
        errors = []
        try:
            for step in range(args.steps):
                before = (
                    [[p.detach().clone() for p in model.parameters()] for model in (native, fast)] if step < 5 else []
                )
                metrics = []
                for model, execution, optimizer, cfg, transfer in zip(
                    (native, fast),
                    (native_execution, fast_execution),
                    optimizers,
                    (native_cfg, config),
                    transfers,
                    strict=True,
                ):
                    metrics.append(
                        optimizer_update(
                            model,
                            execution,
                            optimizer,
                            cfg,
                            device,
                            transfer.next,
                            transfer.preload,
                            step * blocks,
                            (step + 1) * blocks,
                            step,
                            args.steps * blocks,
                        )
                    )
                if step < 5:
                    native_parameters, fast_parameters = list(native.parameters()), list(fast.parameters())
                    entry = {
                        "loss": abs(metrics[1]["local_loss"] - metrics[0]["local_loss"]),
                        "loss_limit": 0.002 + 0.002 * abs(metrics[0]["local_loss"]),
                        "parameters": relative_error(fast_parameters, native_parameters),
                        "gradients": relative_error(
                            [p.grad for p in fast_parameters if p.grad is not None],
                            [p.grad for p in native_parameters if p.grad is not None],
                        ),
                        "updates": relative_error(
                            [p - b for p, b in zip(fast_parameters, before[1], strict=True)],
                            [p - b for p, b in zip(native_parameters, before[0], strict=True)],
                        ),
                    }
                    # Normalize on each rank before MAX; reference losses differ in AWC.
                    entry["loss_tolerance_ratio"] = entry["loss"] / entry["loss_limit"]
                    for moment in ("exp_avg", "exp_avg_sq"):
                        entry[moment] = relative_error(
                            [s[moment] for s in optimizers[1].state.values()],
                            [s[moment] for s in optimizers[0].state.values()],
                        )
                    values = torch.tensor(list(entry.values()), device=device, dtype=torch.float64)
                    if dist.is_initialized():
                        dist.all_reduce(values, op=dist.ReduceOp.MAX)
                    errors.append(dict(zip(entry, values.tolist(), strict=True)))
        finally:
            for transfer in transfers:
                transfer.close()
        # Validation uses native arithmetic for both states, isolating trajectory drift.
        for model in (native, fast):
            model.execution_policy = {}
        native_result = evaluate(native, cache, native_cfg, device, full=False)
        fast_result = evaluate(fast, cache, native_cfg, device, full=False)
        delta = fast_result["loss"] - native_result["loss"]
        passed = (
            all(
                e["loss_tolerance_ratio"] <= 1
                and e["parameters"] < 0.02
                and e["gradients"] < 0.05
                and e["updates"] < 0.10
                and e["exp_avg"] < 0.05
                and e["exp_avg_sq"] < 0.05
                for e in errors
            )
            and abs(delta) <= 0.02
        )
        from torch._inductor import config as inductor_config

        result = {
            "rank": rank(),
            "world_size": world_size(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "passed": passed,
            "steps": args.steps,
            "errors": errors,
            "native_validation": native_result,
            "candidate_validation": fast_result,
            "validation_delta": delta,
            "execution": fast_execution.policy,
            "reference_execution": native_execution.policy,
            "compile_cache": fast_execution.cache,
            "compiler_options": {"emulate_precision_casts": inductor_config.emulate_precision_casts},
            "compiler_counters": compiler_counters(),
            "config": config.model_dump(mode="json"),
            "cache_identity": cache.manifest["identity"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": environment(),
            "seconds": time.perf_counter() - started,
        }
        atomic_json(args.output / f"numerical-rank-{rank():05d}.json", result)
        if rank() == 0:
            print(result, flush=True)
        if not passed:
            raise RuntimeError("numerical qualification failed; see per-rank report")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
