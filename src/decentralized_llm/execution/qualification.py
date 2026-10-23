"""Workload qualification and dependency-aware source fingerprints."""

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

from torch.cuda.nccl import version as nccl_version

from decentralized_llm.config import Config
from decentralized_llm.runtime import source_digest


def _revision(*directories: str) -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in directories:
        digest.update(source_digest(package / directory, include_metadata=False).encode())
    return digest.hexdigest()


def kernel_revision() -> str:
    return _revision("models", "execution/cuda")


def optimizer_revision() -> str:
    return _revision("algorithms", "backends", "execution/cuda", "training")


def compile_revision() -> str:
    return _revision("models", "algorithms", "execution", "backends", "training")


def qualification_key(config: Config) -> dict:
    muon_mode = config.runtime.resolved_muon_optimizer()
    return {
        **({"optimizer": config.optimizer.name} if config.optimizer.name != "adamw" else {}),
        **({"muon_optimizer": muon_mode} if config.optimizer.name == "muon_adamw" and muon_mode != "reference" else {}),
        "training_backend": config.runtime.training_backend,
        "model": config.model.model_dump(),
        "mode": config.distributed.mode,
        "nodes": config.distributed.nodes,
        "gpus_per_node": config.distributed.gpus_per_node,
        "batch_tokens": config.training.batch_tokens,
        "micro_batch_size": config.micro_batch_size,
        "topology": config.distributed.topology,
        "scheme": config.distributed.scheme,
        "error_feedback_gamma": config.distributed.error_feedback_gamma,
        "execution_options": config.runtime.model_dump(
            include={
                "compile_mode",
                "compile_scope",
                "projection_fusion",
                "fused_rope",
                "sdpa_backend",
                "fused_optimizer",
                "decentralized_optimizer",
                "cpu_threads",
                "cpu_affinity",
            }
        ),
        "communication_options": config.distributed.model_dump(
            include={
                "backend",
                "ddp_bucket_cap_mb",
                "ddp_gradient_as_bucket_view",
                "ddp_batched_grad_copy",
                "reuse_mix_buffers",
            }
        ),
    }


def qualification_environment() -> dict:
    from torch._inductor import config as inductor_config

    return {
        "architecture": platform.machine(),
        "python": list(sys.version_info[:3]),
        "triton": importlib.metadata.version("triton"),
        "nccl": list(nccl_version()),
        "emulate_precision_casts": inductor_config.emulate_precision_casts,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allowed_cpus": len(os.sched_getaffinity(0)),
    }


def profile_key(entry: dict) -> dict:
    """Historical measurements used the reference optimizer before this selector existed."""
    return {
        "training_backend": "distributed",
        **entry["key"],
        "execution_options": {
            "decentralized_optimizer": "reference",
            **entry["key"]["execution_options"],
        },
    }
