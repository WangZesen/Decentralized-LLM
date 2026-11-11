"""CLI entry points. Worker commands are internal torchrun targets."""

import argparse
import json
from pathlib import Path

from decentralized_llm.config import initialize_local_defaults, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="decentralized-llm")
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser("benchmark-quality", help="zero-shot candidate likelihood benchmarks")
    quality.add_argument("--config", "-c", type=Path, required=True, help="saved training configuration")
    quality.add_argument("--checkpoint", type=Path, required=True)
    quality.add_argument(
        "--tasks", nargs="+", choices=("blimp", "piqa", "hellaswag"), default=["blimp", "piqa", "hellaswag"]
    )
    quality.add_argument("--output", type=Path, required=True)
    quality.add_argument("--suite", type=Path, help="prepared, checksummed benchmark suite")
    quality.add_argument("--device", default="cuda")
    quality.add_argument("--batch-size", type=int, default=32)
    quality.add_argument("--limit", type=int, help="smoke only: examples per BLiMP subset or other task")
    tuning = commands.add_parser("tune", help="run, resume, or inspect a phased hyperparameter search")
    actions = tuning.add_subparsers(dest="tune_action", required=True)
    for name in ("run", "resume", "extend", "status", "report"):
        action = actions.add_parser(name)
        if name == "run":
            action.add_argument("--output", dest="directory", type=Path, required=True)
        else:
            action.add_argument("directory", type=Path)
        if name in ("run", "extend"):
            action.add_argument("--spec", type=Path, required=True)
            action.add_argument(
                "--config",
                "-c",
                type=Path,
                action="append",
                default=[],
                help="append a base config before spec overrides (repeatable; relative to the working directory)",
            )
            action.add_argument(
                "--set",
                dest="overrides",
                action="append",
                default=[],
                help="override a base setting with dotted.key=value after spec overrides (repeatable)",
            )
            action.add_argument(
                "--no-local-config",
                action="store_true",
                help="skip initialization and loading of local execution defaults",
            )
        if name in ("run", "resume", "extend"):
            action.add_argument("--dry-run", action="store_true")
        if name == "resume":
            action.add_argument(
                "--retry-unresolved",
                action="append",
                default=[],
                metavar="ATTEMPT",
                help="confirm this unresolved attempt has no live job and permit a bounded retry",
            )
    for name in (
        "prepare",
        "validate",
        "train",
        "evaluate",
        "submit",
        "benchmark-data",
        "benchmark-training",
        "worker",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", "-c", type=Path, action="append", default=[])
        command.add_argument("--set", dest="overrides", action="append", default=[])
        if name != "worker":
            command.add_argument(
                "--no-local-config",
                action="store_true",
                help="skip initialization and loading of local execution defaults",
            )
        if name in ("validate", "train", "submit"):
            command.add_argument("--resume", type=Path)
        if name in ("train", "evaluate", "submit", "benchmark-training"):
            command.add_argument("--dry-run", action="store_true")
        if name == "submit":
            command.add_argument("--action", choices=("train", "evaluate", "benchmark-training"), default="train")
            command.add_argument("--subset", action="store_true")
        if name in ("evaluate", "worker"):
            command.add_argument("--checkpoint", type=Path, required=name == "evaluate")
            command.add_argument("--subset", action="store_true")
        if name == "worker":
            command.add_argument("--action", choices=("train", "evaluate", "benchmark-training"), default="train")
        if name == "benchmark-training":
            command.add_argument("--steps", type=int)
            command.add_argument("--warmup", type=int)
            command.add_argument("--windows", type=int)
            command.add_argument("--min-seconds", type=float)
            command.add_argument("--max-chunks", type=int)
            command.add_argument("--profile", action="store_true")
        if name == "benchmark-data":
            command.add_argument("--steps", type=int, default=100)
            command.add_argument("--warmup", type=int, default=20)
            command.add_argument("--cache-state", choices=("cold", "warm"), default="warm")
            command.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "benchmark-quality":
        from decentralized_llm.benchmarks.quality_tasks import benchmark_quality

        result = benchmark_quality(
            load_config(args.config),
            args.checkpoint,
            args.output,
            suite=args.suite,
            tasks=args.tasks,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "tune":
        from decentralized_llm.tuning.controller import execute

        result = execute(
            args.tune_action,
            args.directory,
            getattr(args, "spec", None),
            configs=getattr(args, "config", ()),
            overrides=getattr(args, "overrides", ()),
            dry_run=getattr(args, "dry_run", False),
            retry_unresolved=getattr(args, "retry_unresolved", None),
            use_local_defaults=not getattr(args, "no_local_config", False),
        )
        print(json.dumps(result, indent=2))
        if result.get("status") in {"failed", "interrupted"}:
            raise SystemExit(1 if result["status"] == "failed" else 130)
        return
    use_local_defaults = args.command != "worker" and not args.no_local_config
    if use_local_defaults:
        initialize_local_defaults(dry_run=getattr(args, "dry_run", False))
    config = load_config(args.config, args.overrides, use_local_defaults=use_local_defaults)
    if args.command == "benchmark-training":
        for key in ("steps", "warmup", "windows", "min_seconds", "max_chunks"):
            if getattr(args, key) is not None:
                setattr(config.benchmark, key, getattr(args, key))
        if args.profile:
            config.benchmark.profile = True
    if args.command == "prepare":
        from decentralized_llm.data.cache import prepare

        result = prepare(config)
    elif args.command == "validate":
        from decentralized_llm.launch.preflight import validate

        result = validate(config, args.resume)
    elif args.command in ("train", "evaluate", "submit", "benchmark-training"):
        from decentralized_llm.launch.commands import launch

        if args.command not in ("submit", "benchmark-training") and config.launcher.kind != "local":
            parser.error("use submit for a SLURM launcher")
        if args.command == "submit" and config.launcher.kind != "slurm":
            parser.error("submit requires launcher.kind=slurm")
        action = args.action if args.command == "submit" else args.command
        checkpoint = args.checkpoint if args.command == "evaluate" else getattr(args, "resume", None)
        if action == "evaluate" and checkpoint is None:
            parser.error("evaluation requires --checkpoint (or submit --resume)")
        result = launch(
            config, checkpoint, dry_run=args.dry_run, action=action, full=not getattr(args, "subset", False)
        )
    elif args.command == "worker":
        from decentralized_llm.training.loop import evaluate_checkpoint, train

        if args.action == "benchmark-training":
            from decentralized_llm.benchmarks.training import benchmark_training

            result = benchmark_training(config)
        elif args.action == "evaluate":
            if args.checkpoint is None:
                parser.error("worker evaluation requires a checkpoint")
            result = evaluate_checkpoint(config, args.checkpoint, full=not args.subset)
        else:
            result = train(config, args.checkpoint)
    else:
        from decentralized_llm.benchmarks.data import benchmark_data

        result = benchmark_data(config, args.steps, args.warmup, args.cache_state, args.output)
    print(json.dumps(result, indent=2))
