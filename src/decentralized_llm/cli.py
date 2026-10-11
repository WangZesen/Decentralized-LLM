"""CLI entry points. Worker commands are internal torchrun targets."""

import argparse
import json
from pathlib import Path

from decentralized_llm.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="decentralized-llm")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "train", "evaluate", "submit", "benchmark-data", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--config", "-c", type=Path, action="append", default=[])
        command.add_argument("--set", dest="overrides", action="append", default=[])
        if name in ("validate", "train", "submit"):
            command.add_argument("--resume", type=Path)
        if name in ("train", "evaluate", "submit"):
            command.add_argument("--dry-run", action="store_true")
        if name == "submit":
            command.add_argument("--action", choices=("train", "evaluate"), default="train")
            command.add_argument("--subset", action="store_true")
        if name in ("evaluate", "worker"):
            command.add_argument("--checkpoint", type=Path, required=name == "evaluate")
            command.add_argument("--subset", action="store_true")
        if name == "worker":
            command.add_argument("--action", choices=("train", "evaluate"), default="train")
        if name == "benchmark-data":
            command.add_argument("--steps", type=int, default=100)
            command.add_argument("--warmup", type=int, default=20)
            command.add_argument("--cache-state", choices=("cold", "warm"), default="warm")
            command.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.command == "prepare":
        from decentralized_llm.cache import prepare

        result = prepare(config)
    elif args.command == "validate":
        from decentralized_llm.preflight import validate

        result = validate(config, args.resume)
    elif args.command in ("train", "evaluate", "submit"):
        from decentralized_llm.launch import launch

        if args.command != "submit" and config.launcher.kind != "local":
            parser.error("use submit for a SLURM launcher")
        if args.command == "submit" and config.launcher.kind != "slurm":
            parser.error("submit requires launcher.kind=slurm")
        action = args.action if args.command == "submit" else args.command
        checkpoint = args.checkpoint if args.command == "evaluate" else args.resume
        if action == "evaluate" and checkpoint is None:
            parser.error("evaluation requires --checkpoint (or submit --resume)")
        result = launch(
            config, checkpoint, dry_run=args.dry_run, action=action, full=not getattr(args, "subset", False)
        )
    elif args.command == "worker":
        from decentralized_llm.train import evaluate_checkpoint, train

        if args.action == "evaluate":
            if args.checkpoint is None:
                parser.error("worker evaluation requires a checkpoint")
            result = evaluate_checkpoint(config, args.checkpoint, full=not args.subset)
        else:
            result = train(config, args.checkpoint)
    else:
        from decentralized_llm.benchmark import benchmark_data

        result = benchmark_data(config, args.steps, args.warmup, args.cache_state, args.output)
    print(json.dumps(result, indent=2))
