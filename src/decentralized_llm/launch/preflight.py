"""Validation before launch or scheduler submission, without creating a run."""

import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from decentralized_llm.config import Config, SlurmLauncher
from decentralized_llm.data.cache import TokenCache, training_boundaries
from decentralized_llm.data.sampling import check_memory, sampling_plan
from decentralized_llm.training.checkpoints import validate_checkpoint


def validate(
    config: Config,
    resume: Path | None = None,
    *,
    launch: bool = False,
    verify: bool = True,
    evaluation: bool = False,
    benchmark: bool = False,
) -> dict:
    if benchmark and (evaluation or resume is not None):
        raise ValueError("training benchmarks cannot evaluate or resume a checkpoint")
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    if verify:
        cache.verify()
    if cache.blocks("train", config.model.context_length) < 1:
        raise ValueError("training cache has no complete blocks")
    bounds = None
    if benchmark:
        from decentralized_llm.benchmarks.protocol import benchmark_bounds

        bounds = benchmark_bounds(config)
    memory = check_memory(cache, config, stop=bounds.total_blocks if bounds else None, bound_resident_to_stop=benchmark)
    boundaries = training_boundaries(config, config.model.parameter_count)
    if resume is not None:
        checkpoint = validate_checkpoint(resume, config, cache, verify=verify, require_training=not evaluation)
        if checkpoint["cursor"] > boundaries[-1] or not 0 <= checkpoint["completed_epochs"] <= len(boundaries):
            raise ValueError("checkpoint progress exceeds training budget")
        if not evaluation and checkpoint["cursor"] == boundaries[-1]:
            raise ValueError("checkpoint already completed its training budget; use evaluate")
        epochs = sum(boundary <= checkpoint["cursor"] for boundary in boundaries)
        if checkpoint["completed_epochs"] != epochs:
            raise ValueError("checkpoint epoch/cursor mismatch")
    output = config.runtime.output_dir.resolve()
    if output == cache.directory.resolve() or cache.directory.resolve() in output.parents:
        raise ValueError("output must not be inside the immutable token cache")
    if output.exists() and not output.is_dir():
        raise ValueError("output destination is not a directory")
    if not evaluation and resume is None and (output / "run.json").exists():
        raise ValueError("output already contains a run; use resume or a new destination")
    parent = output
    while not parent.exists():
        parent = parent.parent
    with tempfile.TemporaryFile(dir=parent):
        pass
    if isinstance(config.launcher, SlurmLauncher):
        script = config.launcher.environment_script or Path.cwd() / "scripts" / "environment.sh"
        for setup_script in [script, *config.launcher.environment_overlays]:
            if not setup_script.is_file():
                raise ValueError(f"environment setup script does not exist: {setup_script}")
        if launch:
            if not os.environ.get("account", "").strip():
                raise ValueError("SLURM submission requires a nonempty lowercase environment variable account")
            for executable in ("sbatch", "srun", "scontrol"):
                if shutil.which(executable) is None:
                    raise ValueError(f"SLURM prerequisite missing: {executable}")
    elif launch and config.runtime.device == "cuda":
        if torch.cuda.device_count() < config.processes_per_node:
            raise ValueError(f"requested {config.processes_per_node} GPUs, found {torch.cuda.device_count()}")
    return {
        "cache_identity": cache.manifest["identity"],
        "world_size": config.distributed.world_size,
        "training_backend": config.runtime.training_backend,
        "physical_world_size": config.physical_world_size,
        "physical_gpu_count": config.physical_gpu_count,
        "num_accumulation": config.training.num_accumulation,
        "micro_batch_size": config.micro_batch_size,
        "parameters": config.model.parameter_count,
        "eligible_blocks": cache.blocks("train", config.model.context_length),
        "global_tokens": boundaries[-1] * config.model.context_length,
        "virtual_epochs": len(boundaries),
        "sampling": sampling_plan(cache, config).summary(),
        "memory_bytes_per_rank": memory,
        "loader_mode": "resident" if memory["resident_tokens"] else "streaming",
        **({"benchmark_bounds": asdict(bounds)} if bounds else {}),
    }
