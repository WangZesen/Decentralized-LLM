"""Manifest-last checkpoints with rank-specific RNG, sampler and optimizer state."""

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from decentralized_llm.backends.api import TrainingBackend
from decentralized_llm.backends.factory import runtime_for
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache, file_digest, fingerprint
from decentralized_llm.data.sampling import GlobalSampler
from decentralized_llm.models.initialization import INITIALIZATION
from decentralized_llm.models.llama import Llama
from decentralized_llm.runtime import atomic_checkpoint, atomic_json, rng_state
from decentralized_llm.state import RNGState

CHECKPOINT_VERSION = 1


def recipe_identity(config: Config, cache: TokenCache, *, initialization: str | None = INITIALIZATION) -> str:
    runtime = config.runtime.model_dump(
        exclude={"output_dir", "cpu_threads", "compile_cache_dir", "cpu_affinity", "decentralized_optimizer"}
    )
    # Default-valued execution additions do not invalidate version-1 recipes.
    defaults = {
        "training_backend": "distributed",
        "execution_backend": "auto",
        "compile_scope": "model",
        "projection_fusion": True,
        "fused_rope": False,
        "sdpa_backend": "auto",
    }
    for key, value in defaults.items():
        if runtime[key] == value:
            runtime.pop(key)
    distributed = config.distributed.model_dump(
        exclude={"nodes", "gpus_per_node", "timeout_seconds", "rendezvous_port", "reuse_mix_buffers"}
    )
    for key, value in {
        "ddp_bucket_cap_mb": 25,
        "ddp_gradient_as_bucket_view": False,
        "ddp_batched_grad_copy": False,
    }.items():
        if distributed[key] == value:
            distributed.pop(key)
    training = config.training.model_dump(
        mode="json",
        exclude={
            "checkpoint_policy",
            "checkpoint_epochs",
            "save_epoch_training_state",
            "log_every",
            "num_accumulation",
        },
    )
    # Preserve historical recipe identities when the effective batching is identical.
    training["micro_batch_size"] = config.micro_batch_size
    # Keep recipes written before data reuse was introduced resumable at K=1.
    if config.training.data_reuse_factor == 1:
        training.pop("data_reuse_factor")
    return fingerprint(
        {
            **({"initialization": initialization} if initialization is not None else {}),
            "model": config.model.model_dump(),
            "optimizer": config.optimizer.model_dump(),
            "schedule": config.lr_schedule.model_dump(),
            "training": training,
            "evaluation": config.evaluation.model_dump(),
            "runtime": runtime,
            "distributed": distributed,
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
    if manifest.get("training_backend", "distributed") != config.runtime.training_backend:
        raise ValueError("checkpoint training backend mismatch")
    initialization = manifest.get("initialization")
    identity = recipe_identity(config, cache, initialization=initialization)
    if manifest["recipe_identity"] != identity and not manifest.get("execution"):
        legacy = config.model_copy(deep=True)
        if legacy.runtime.compile == "auto":
            legacy.runtime.compile = False
        if legacy.runtime.execution_backend == "native":
            legacy.runtime.execution_backend = "auto"
        identity = recipe_identity(legacy, cache, initialization=initialization)
    if manifest["recipe_identity"] != identity:
        raise ValueError("checkpoint training recipe mismatch")
    if require_training and not manifest["training_state"]:
        raise ValueError("weights-only epoch cannot resume training")
    if manifest["sampler"] != GlobalSampler.from_config(cache, config).identity:
        raise ValueError("checkpoint sampler identity mismatch")
    expected = (
        {"packed.pt"}
        if config.runtime.training_backend == "packed"
        else {f"rank-{r:05d}.pt" for r in range(config.distributed.world_size)}
    )
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
    *,
    backend: TrainingBackend | None = None,
) -> None:
    runtime = runtime_for(config)
    if backend is not None:
        backend.inspect(progress["step"])
    if runtime.rank == 0:
        path.mkdir(parents=True, exist_ok=False)
    runtime.barrier()
    file = path / ("packed.pt" if config.runtime.training_backend == "packed" else f"rank-{runtime.rank:05d}.pt")
    owns_weights = config.distributed.mode == "decentralized" or runtime.rank == 0
    payload: dict[str, Any] = {"rng": rng_state(), "loader": sampler, "rank": runtime.rank}
    if backend is not None:
        payload.update(backend.state_dict(training_state))
    elif owns_weights:
        payload["model"] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if training_state:
            payload["optimizer"] = optimizer.state_dict()
    result: dict[str, Any]
    try:
        atomic_checkpoint(file, payload)
        result = {"file": file.name, "sha256": file_digest(file)}
    except Exception as error:
        result = {"error": f"rank {runtime.rank}: {type(error).__name__}: {error}"}
    results = runtime.gather(result)
    errors = [item["error"] for item in results if "error" in item]
    if errors:
        raise OSError(f"checkpoint was not published: {errors}")
    if runtime.rank == 0:
        atomic_json(
            path / "manifest.json",
            {
                "version": CHECKPOINT_VERSION,
                "world_size": config.distributed.world_size,
                "training_backend": config.runtime.training_backend,
                "initialization": getattr(model, "initialization", None) or INITIALIZATION,
                "recipe_identity": recipe_identity(
                    config, cache, initialization=getattr(model, "initialization", None) or INITIALIZATION
                ),
                "sampler": sampler["sampler"],
                "training_state": training_state,
                "execution": getattr(model, "execution_policy", {}),
                "files": results,
                **progress,
            },
        )
    runtime.barrier()


def load_checkpoint(
    path: Path,
    config: Config,
    cache: TokenCache,
    model: nn.Module,
    optimizer: Any,
    *,
    backend: TrainingBackend | None = None,
) -> tuple[dict, RNGState]:
    runtime = runtime_for(config)
    manifest = validate_checkpoint(path, config, cache, require_training=optimizer is not None)
    # Only load trusted local training checkpoints: RNG/NumPy state needs pickle.
    local = torch.load(
        path / ("packed.pt" if config.runtime.training_backend == "packed" else f"rank-{runtime.rank:05d}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    if local["rank"] != runtime.rank or local["loader"] != {
        "sampler": manifest["sampler"],
        "cursor": manifest["cursor"],
    }:
        raise ValueError("rank data state does not match checkpoint manifest")
    weights = local
    if config.distributed.mode == "ddp" and runtime.rank != 0:
        weights = torch.load(path / "rank-00000.pt", map_location="cpu", weights_only=False)
    if backend is not None:
        backend.load_state_dict(weights, optimizer is not None)
    else:
        model.load_state_dict(weights["model"], strict=True)
    cast(Llama, model).initialization = manifest.get("initialization") or (
        "legacy_rank_offset_v1" if config.distributed.mode == "decentralized" else "legacy_shared_seed_v1"
    )
    # Pin automatic dispatch to the checkpoint's effective policy on resume.
    policy = manifest.get("execution") or {
        "backend": "native",
        "compile": config.runtime.compile is True,
        "compile_mode": config.runtime.compile_mode,
        "compile_scope": "model",
        "projection_fusion": False,
        "fused_rope": False,
        "sdpa_backend": "auto",
        "reason": "legacy native checkpoint",
    }
    if config.runtime.execution_backend not in ("auto", policy["backend"]):
        raise ValueError("checkpoint execution backend mismatch")
    cast(Llama, model).execution_policy = policy
    if optimizer is not None and backend is None:
        optimizer.load_state_dict(weights["optimizer"])
    return manifest, local["rng"]
