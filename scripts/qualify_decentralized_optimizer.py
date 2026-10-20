"""Matched optimizer trajectories over the full matrix, sharing process groups and compile caches."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from qualification_gates import GATE_PROTOCOL, GATE_REVISION
from qualify_gh200 import qualify

from decentralized_llm.backends.distributed.runtime import prepare_parameter_groups, rank, setup
from decentralized_llm.config import load_config
from decentralized_llm.execution.qualification import optimizer_revision
from decentralized_llm.runtime import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--models", nargs="+", choices=("20m", "90m"), default=["20m", "90m"])
    parser.add_argument("--schemes", nargs="+", choices=("awc", "atc"), default=["awc", "atc"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    config.distributed.mode = "decentralized"
    device = setup(config)
    revision = optimizer_revision()
    rows = []
    try:
        prepare_parameter_groups("one_peer_ring")
        for model in args.models:
            current = load_config([*args.config, Path(f"configs/models/{model}.yaml")], args.overrides)
            current.distributed.mode = "decentralized"
            current.runtime.decentralized_optimizer = "auto"
            for scheme in args.schemes:
                current.distributed.scheme = scheme
                for topology in ("complete", "one_peer_ring"):
                    current.distributed.topology = topology
                    for feedback in (False, True):
                        current.distributed.error_feedback = feedback
                        for seed in (42, 43):
                            current.runtime.seed = seed
                            output = args.output / model / scheme / topology / str(feedback) / str(seed)
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.manual_seed(seed)
                            if optimizer_revision() != revision:
                                raise RuntimeError("optimizer sources changed during qualification")
                            result = None
                            if args.resume:
                                try:
                                    candidate = json.loads((output / f"numerical-rank-{rank():05d}.json").read_text())
                                    if (
                                        candidate.get("optimizer_revision") == revision
                                        and candidate.get("passed")
                                        and candidate.get("steps") == args.steps
                                        and candidate.get("optimizer_only")
                                        and candidate.get("gates", {}).get("revision") == GATE_REVISION
                                        and candidate.get("config") == current.model_dump(mode="json")
                                    ):
                                        result = candidate
                                except FileNotFoundError, ValueError:
                                    pass
                            ready = torch.tensor(int(result is not None), device=device)
                            if dist.is_initialized():
                                dist.all_reduce(ready, op=dist.ReduceOp.MIN)
                            if not bool(ready):
                                result = qualify(
                                    current,
                                    output,
                                    steps=args.steps,
                                    optimizer_only=True,
                                    device=device,
                                    raise_on_failure=False,
                                )
                            assert result is not None
                            rows.append(
                                {
                                    "model": model,
                                    "scheme": scheme,
                                    "topology": topology,
                                    "feedback": feedback,
                                    "seed": seed,
                                    "passed": result["passed"],
                                    "all_recorded_updates_passed": result["gates"]["all_recorded_updates_passed"],
                                    "validation_delta": result["validation_delta"],
                                    "report": str(output / "numerical-rank-00000.json"),
                                }
                            )
                            if rank() == 0:
                                complete = len(rows) == len(args.models) * len(args.schemes) * 8
                                atomic_json(
                                    args.output / "matrix.json",
                                    {
                                        "rows": rows,
                                        "optimizer_revision": revision,
                                        "gate_protocol": GATE_PROTOCOL,
                                        "gate_revision": GATE_REVISION,
                                        "complete": complete,
                                        "passed": complete and all(row["passed"] for row in rows),
                                        "all_recorded_updates_passed": complete
                                        and all(row["all_recorded_updates_passed"] for row in rows),
                                    },
                                )
        if not all(row["passed"] for row in rows):
            raise RuntimeError("numerical qualification failed; see matrix.json for all cases")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
