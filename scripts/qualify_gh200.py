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
from qualification_gates import assess_numerical
from torch import Tensor

from decentralized_llm.algorithms.adamw import make_optimizer, optimizer_report
from decentralized_llm.backends.distributed.engine import build_distributed_execution
from decentralized_llm.backends.distributed.runtime import rank, setup, world_size
from decentralized_llm.backends.distributed.update import optimizer_update
from decentralized_llm.config import Config, load_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, LoadedBatch, TokenLoader
from decentralized_llm.execution.compilation import compiler_counters
from decentralized_llm.execution.qualification import optimizer_revision
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.runtime import atomic_json, environment
from decentralized_llm.training.evaluation import evaluate


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


def qualify(
    config: Config,
    output: Path,
    *,
    steps: int = 100,
    optimizer_only: bool = False,
    reference_compile: bool = False,
    device: torch.device | None = None,
    raise_on_failure: bool = True,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_revision = optimizer_revision()
    owns_group = device is None
    if device is None:
        device = setup(config)
    try:
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        blocks = config.training.batch_tokens // config.model.context_length
        with TokenLoader(cache, config, rank(), 0, steps * blocks) as loader:
            batches = []
            while True:
                try:
                    batch = loader.next()
                    batches.append(LoadedBatch(batch.spec, batch.tokens.copy()))
                except StopIteration:
                    break
        native_cfg = config.model_copy(deep=True)
        if optimizer_only:
            native_cfg.runtime.decentralized_optimizer = "reference"
            config.runtime.decentralized_optimizer = "auto"
        else:
            native_cfg.runtime.execution_backend, native_cfg.runtime.compile = "native", reference_compile
            native_cfg.runtime.fused_rope = False
        native = initialize_model(config).to(device)
        fast = copy.deepcopy(native)
        native_execution = build_distributed_execution(native, native_cfg, device)
        fast_execution = build_distributed_execution(fast, config, device)
        if optimizer_only and (
            {key: value for key, value in native_execution.policy.items() if key != "reason"}
            != {key: value for key, value in fast_execution.policy.items() if key != "reason"}
        ):
            raise RuntimeError("forward policies differ; set execution_backend and compile explicitly")
        optimizers = [make_optimizer(native, native_cfg, device), make_optimizer(fast, config, device)]
        transfers = [DevicePrefetcher(Replay(config, batches), device) for _ in range(2)]
        errors = []
        try:
            for step in range(steps):
                check = optimizer_only or step < 5
                before = [[p.detach().clone() for p in model.parameters()] for model in (native, fast)] if check else []
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
                            steps * blocks,
                        )
                    )
                if check:
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
                    if optimizer_only:
                        histories = [getattr(opt, "previous_update", None) for opt in optimizers]
                        if config.distributed.error_feedback_gamma:
                            original_history, candidate_history = histories
                            assert isinstance(original_history, Tensor) and isinstance(candidate_history, Tensor)
                            entry["history"] = relative_error([candidate_history], [original_history])
                        entry["steps"] = relative_error(
                            [s["step"] for s in optimizers[1].state.values()],
                            [s["step"] for s in optimizers[0].state.values()],
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
        gates = assess_numerical(
            errors,
            delta,
            steps=steps,
            optimizer_only=optimizer_only,
            feedback=config.distributed.error_feedback_gamma != 0,
            fused_candidate=optimizer_report(optimizers[1])["implementation"] == "triton",
        )
        passed = gates["passed"]
        from torch._inductor import config as inductor_config

        result = {
            "rank": rank(),
            "world_size": world_size(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "passed": passed,
            "gates": gates,
            "optimizer_only": optimizer_only,
            "optimizer_revision": source_revision,
            "optimizers": [optimizer_report(opt) for opt in optimizers],
            "steps": steps,
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
        if optimizer_revision() != source_revision:
            raise RuntimeError("optimizer sources changed during qualification; discard this run")
        atomic_json(output / f"numerical-rank-{rank():05d}.json", result)
        if rank() == 0:
            print({key: result[key] for key in ("passed", "validation_delta", "seconds")}, flush=True)
        if not passed and raise_on_failure:
            raise RuntimeError("numerical qualification failed; see per-rank report")
        return result
    finally:
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--optimizer-only", action="store_true", help="keep the forward backend fixed; compare optimizers"
    )
    parser.add_argument(
        "--reference-compile", action="store_true", help="compare against compiled native instead of eager native"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    qualify(
        config,
        args.output,
        steps=args.steps,
        optimizer_only=args.optimizer_only,
        reference_compile=args.reference_compile,
    )


if __name__ == "__main__":
    main()
