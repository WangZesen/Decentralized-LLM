"""Compilation caches and instrumentation shared by all execution engines."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import torch

from decentralized_llm.config import Config
from decentralized_llm.execution.qualification import compile_revision


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
        "training_backend": config.runtime.training_backend,
        "logical_world_size": config.distributed.world_size,
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
