"""Manifest-last checkpoints with rank-specific RNG, sampler and optimizer state."""

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from decentralized_llm.cache import TokenCache, file_digest, fingerprint
from decentralized_llm.config import Config
from decentralized_llm.distributed import barrier, rank, world_size
from decentralized_llm.runtime import atomic_checkpoint, atomic_json, rng_state
from decentralized_llm.sampling import GlobalSampler
from decentralized_llm.state import RNGState

CHECKPOINT_VERSION = 1


def recipe_identity(config: Config, cache: TokenCache) -> str:
    training = config.training.model_dump(
        mode="json",
        exclude={
            "checkpoint_policy",
            "checkpoint_epochs",
            "save_epoch_training_state",
            "log_every",
        },
    )
    return fingerprint(
        {
            "model": config.model.model_dump(),
            "optimizer": config.optimizer.model_dump(),
            "schedule": config.lr_schedule.model_dump(),
            "training": training,
            "evaluation": config.evaluation.model_dump(),
            "runtime": config.runtime.model_dump(exclude={"output_dir", "cpu_threads"}),
            "distributed": config.distributed.model_dump(
                exclude={"nodes", "gpus_per_node", "timeout_seconds", "rendezvous_port"}
            ),
            "world_size": config.distributed.world_size,
            "cache": cache.manifest["identity"],
        }
    )


def validate_checkpoint(
    path: Path, config: Config, cache: TokenCache, verify: bool = True, require_training: bool = True
) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version; historical tiny-llm checkpoints are not supported")
    if manifest["world_size"] != config.distributed.world_size:
        raise ValueError("checkpoint world size mismatch")
    if manifest["recipe_identity"] != recipe_identity(config, cache):
        raise ValueError("checkpoint training recipe mismatch")
    if require_training and not manifest["training_state"]:
        raise ValueError("weights-only epoch cannot resume training")
    if manifest["sampler"] != GlobalSampler(cache, config.model.context_length, config.runtime.seed).identity:
        raise ValueError("checkpoint sampler identity mismatch")
    expected = {f"rank-{r:05d}.pt" for r in range(config.distributed.world_size)}
    if {f["file"] for f in manifest["files"]} != expected or len(manifest["files"]) != len(expected):
        raise ValueError("checkpoint rank files are incomplete")
    for item in manifest["files"]:
        filename = path / item["file"]
        if not filename.is_file() or (verify and file_digest(filename) != item["sha256"]):
            raise ValueError(f"checkpoint file missing or corrupt: {filename}")
    cursor = manifest["cursor"]
    blocks = config.training.batch_tokens // config.model.context_length
    if not isinstance(cursor, int) or cursor < 0 or cursor % blocks or manifest["step"] != cursor // blocks:
        raise ValueError("checkpoint does not describe a committed optimizer update")
    return manifest


def save_checkpoint(
    path: Path,
    config: Config,
    cache: TokenCache,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: dict,
    progress: dict,
    training_state: bool = True,
) -> None:
    if rank() == 0:
        path.mkdir(parents=True, exist_ok=False)
    barrier()
    file = path / f"rank-{rank():05d}.pt"
    owns_weights = config.distributed.mode == "decentralized" or rank() == 0
    payload: dict[str, Any] = {"rng": rng_state(), "loader": sampler, "rank": rank()}
    if owns_weights:
        payload["model"] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if training_state:
            payload["optimizer"] = optimizer.state_dict()
    result: dict[str, Any]
    try:
        atomic_checkpoint(file, payload)
        result = {"file": file.name, "sha256": file_digest(file)}
    except Exception as error:
        result = {"error": f"rank {rank()}: {type(error).__name__}: {error}"}
    results: list[Any] = [None] * world_size()
    if dist.is_initialized():
        dist.all_gather_object(results, result)
    else:
        results[0] = result
    errors = [item["error"] for item in results if "error" in item]
    if errors:
        raise OSError(f"checkpoint was not published: {errors}")
    if rank() == 0:
        atomic_json(
            path / "manifest.json",
            {
                "version": CHECKPOINT_VERSION,
                "world_size": world_size(),
                "recipe_identity": recipe_identity(config, cache),
                "sampler": sampler["sampler"],
                "training_state": training_state,
                "files": results,
                **progress,
            },
        )
    barrier()


def load_checkpoint(
    path: Path, config: Config, cache: TokenCache, model: nn.Module, optimizer: torch.optim.Optimizer | None
) -> tuple[dict, RNGState]:
    manifest = validate_checkpoint(path, config, cache, require_training=optimizer is not None)
    # Only load trusted local training checkpoints: RNG/NumPy state needs pickle.
    local = torch.load(path / f"rank-{rank():05d}.pt", map_location="cpu", weights_only=False)
    if local["rank"] != rank() or local["loader"] != {"sampler": manifest["sampler"], "cursor": manifest["cursor"]}:
        raise ValueError("rank data state does not match checkpoint manifest")
    weights = local
    if config.distributed.mode == "ddp" and rank() != 0:
        weights = torch.load(path / "rank-00000.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(weights["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(weights["optimizer"])
    return manifest, local["rng"]
