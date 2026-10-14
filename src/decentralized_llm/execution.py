"""One execution policy for training, evaluation and distributed benchmarks."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.cuda.nccl import version as nccl_version
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from decentralized_llm.config import PRESETS, Config
from decentralized_llm.model import Llama, token_losses
from decentralized_llm.parameter_storage import prepare_parameter_storage
from decentralized_llm.runtime import source_digest


def kernel_revision() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256((package / "model.py").read_bytes())
    digest.update(source_digest(package / "gh200", include_metadata=False).encode())
    return digest.hexdigest()


def compile_revision() -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(kernel_revision().encode())
    digest.update(Path(__file__).with_name("parameter_storage.py").read_bytes())
    return digest.hexdigest()


def qualification_key(config: Config) -> dict:
    return {
        "model": config.model.model_dump(),
        "mode": config.distributed.mode,
        "nodes": config.distributed.nodes,
        "gpus_per_node": config.distributed.gpus_per_node,
        "batch_tokens": config.training.batch_tokens,
        "micro_batch_size": config.micro_batch_size,
        "topology": config.distributed.topology,
        "scheme": config.distributed.scheme,
        "error_feedback": config.distributed.error_feedback,
        "execution_options": config.runtime.model_dump(
            include={
                "compile_mode",
                "compile_scope",
                "projection_fusion",
                "fused_rope",
                "sdpa_backend",
                "fused_optimizer",
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


def resolve_execution(config: Config, device: torch.device) -> dict[str, Any]:
    runtime = config.runtime
    supported = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (9, 0)
        and "GH200" in torch.cuda.get_device_name(device)
        and runtime.amp
        and not runtime.deterministic
        and runtime.attention_backend == "sdpa"
        and any(all(getattr(config.model, k) == v for k, v in preset.items()) for preset in PRESETS.values())
    )
    registry = json.loads(Path(__file__).with_name("qualification.json").read_text())
    qualified = (
        next(
            (
                entry
                for entry in registry["profiles"]
                if entry["key"] == qualification_key(config)
                and entry["torch"] == torch.__version__
                and entry["cuda"] == torch.version.cuda
                and entry["kernel_revision"] == kernel_revision()
                and entry["compile_revision"] == compile_revision()
                and entry["environment"] == qualification_environment()
            ),
            None,
        )
        if supported
        else None
    )
    backend = runtime.execution_backend
    reason = "explicit selection"
    if backend == "auto":
        backend = qualified["backend"] if qualified else "native"
        reason = "measured GH200 profile" if qualified else "no qualified profile for this workload"
    if backend == "gh200" and not supported:
        raise ValueError("GH200 execution requires a supported preset on GH200 with nondeterministic BF16 SDPA")
    compile_enabled = runtime.compile
    if compile_enabled == "auto":
        compile_enabled = backend == "gh200" or bool(
            runtime.execution_backend == "auto" and qualified and qualified["compile"]
        )
    if backend == "gh200" and not compile_enabled:
        if runtime.execution_backend == "gh200":
            raise ValueError("forced GH200 execution requires compilation")
        backend, reason = "native", "compilation explicitly disabled"
    if runtime.fused_rope and (backend != "gh200" or config.model.heads not in (5, 10)):
        raise ValueError("fused_rope requires the GH200 20M/90M path")
    if runtime.fused_rope and not runtime.projection_fusion:
        raise ValueError("fused_rope requires projection_fusion")
    if backend == "gh200" and runtime.compile_scope == "blocks":
        raise ValueError("regional compilation is a native-backend experiment")
    return {
        "backend": backend,
        "compile": compile_enabled,
        "compile_mode": runtime.compile_mode,
        "compile_scope": runtime.compile_scope,
        "projection_fusion": runtime.projection_fusion if backend == "gh200" else False,
        "fused_rope": runtime.fused_rope if backend == "gh200" else False,
        "sdpa_backend": runtime.sdpa_backend,
        "reason": reason,
    }


def configure_compile_cache(config: Config, device: torch.device) -> dict:
    """Compatible processes share artifacts; RNG, optimizer and run paths are absent."""
    identity = {
        "architecture": platform.machine(),
        "python": list(sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": importlib.metadata.version("triton") if device.type == "cuda" else None,
        "capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "source": compile_revision(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = config.runtime.compile_cache_dir.resolve() / digest
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    # Custom autograd implementations are covered by the executable-source namespace.
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    # Optimizer construction can import these modules before the environment is set.
    # Set the live configuration as well; explicit environment overrides still prevail.
    from torch._functorch import config as autograd_config
    from torch._inductor import config as inductor_config

    inductor_config.fx_graph_cache = os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    autograd_config.enable_autograd_cache = os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"
    autograd_config.autograd_cache_allow_custom_autograd_functions = (
        os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD"] == "1"
    )
    inductor_config.compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    return {
        "directory": str(root),
        "identity": identity,
        "settings": {
            "fx_graph_cache": inductor_config.fx_graph_cache,
            "autograd_cache": autograd_config.enable_autograd_cache,
            "custom_autograd_cache": autograd_config.autograd_cache_allow_custom_autograd_functions,
            "compile_threads": inductor_config.compile_threads,
        },
    }


def compiler_counters() -> dict:
    from torch._dynamo.utils import counters

    return {key: dict(value) for key, value in counters.items() if key in ("inductor", "aot_autograd", "stats")}


class LossModule(nn.Module):
    def __init__(self, model: Llama, sdpa_backend: str = "auto", *, evaluation: bool = False):
        super().__init__()
        self.model = model
        self.sdpa_backend = sdpa_backend
        self.evaluation = evaluation

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        context = (
            nullcontext()
            if self.sdpa_backend == "auto"
            else sdpa_kernel(SDPBackend.FLASH_ATTENTION if self.sdpa_backend == "flash" else SDPBackend.CUDNN_ATTENTION)
        )
        with context:
            logits = self.model(x)
            if self.model.execution_backend == "gh200":
                from decentralized_llm.gh200.loss import token_losses as gh200_losses

                losses = gh200_losses(logits, y)
            else:
                losses = token_losses(logits, y)
            return losses.sum(dtype=torch.float64 if self.evaluation else None)


class TrainingExecution(nn.Module):
    """Keep no_sync on the real DDP object, independent of compiler wrappers."""

    def __init__(self, model: Llama, config: Config, device: torch.device, *, distributed: bool = True):
        super().__init__()
        prepare_parameter_storage(model)
        self.policy = model.execution_policy or resolve_execution(config, device)
        model.execution_backend = self.policy["backend"]
        model.projection_fusion = self.policy["projection_fusion"]
        model.fused_rope = self.policy["fused_rope"]
        model.execution_policy = self.policy
        self.cache = configure_compile_cache(config, device) if self.policy["compile"] else None
        self.eager: nn.Module = LossModule(model, config.runtime.sdpa_backend, evaluation=not distributed)
        self.ddp = None
        if distributed and config.distributed.mode == "ddp" and dist.is_initialized() and dist.get_world_size() > 1:
            cfg = config.distributed
            self.ddp = DDP(
                self.eager,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                bucket_cap_mb=cfg.ddp_bucket_cap_mb,
                gradient_as_bucket_view=cfg.ddp_gradient_as_bucket_view,
                batched_grad_copy=cfg.ddp_batched_grad_copy,
            )
            self.eager = self.ddp
        self.compiled = self.eager
        if self.policy["compile"]:
            from torch._inductor import list_mode_options

            options = dict(list_mode_options(config.runtime.compile_mode))
            # NCCL and parameter mixing stay outside CUDA graph capture.
            options["triton.cudagraphs"] = False
            if config.runtime.compile_scope == "blocks":
                for block in model.blocks:
                    block.compile(dynamic=False, options=options)
            else:
                self.compiled = cast(nn.Module, torch.compile(self.eager, dynamic=False, options=options))
        self.shape = (
            (config.micro_batch_size, config.model.context_length)
            if distributed
            else (config.evaluation.batch_size, config.model.context_length)
        )

    def no_sync(self):
        return self.ddp.no_sync() if self.ddp is not None else nullcontext()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if tuple(x.shape) == self.shape:
            return self.compiled(x, y)
        if not self.policy["compile"]:
            return self.eager(x, y)
        # Regional block wrappers live inside the canonical model; bypass them too.
        with torch.compiler.set_stance("force_eager"):
            return self.eager(x, y)


def build_execution(
    model: Llama, config: Config, device: torch.device, *, distributed: bool = True
) -> TrainingExecution:
    return TrainingExecution(model, config, device, distributed=distributed)
