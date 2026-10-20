"""Explicit hardware policy, independent of training placement."""

import json
from importlib.resources import files
from typing import Any

import torch

from decentralized_llm.config import PRESETS, Config
from decentralized_llm.execution.qualification import (
    compile_revision,
    kernel_revision,
    profile_key,
    qualification_environment,
    qualification_key,
)


def resolve_execution(config: Config, device: torch.device) -> dict[str, Any]:
    runtime = config.runtime
    fused_inputs = (
        device.type == "cuda"
        and runtime.amp
        and not runtime.deterministic
        and runtime.attention_backend == "sdpa"
        and any(all(getattr(config.model, k) == v for k, v in preset.items()) for preset in PRESETS.values())
    )
    supported = (
        fused_inputs
        and torch.cuda.get_device_capability(device) == (9, 0)
        and "GH200" in torch.cuda.get_device_name(device)
    )
    a100_supported = (
        fused_inputs
        and torch.cuda.get_device_capability(device) == (8, 0)
        and "A100" in torch.cuda.get_device_name(device)
    )
    registry = json.loads(files("decentralized_llm.execution").joinpath("qualification.json").read_text())
    qualified = (
        next(
            (
                entry
                for entry in registry["profiles"]
                if profile_key(entry) == qualification_key(config)
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
    if backend == "a100" and not a100_supported:
        raise ValueError("A100 execution requires a supported preset on A100 with nondeterministic BF16 SDPA")
    fused_backend = backend in ("gh200", "a100")
    compile_enabled = runtime.compile
    if compile_enabled == "auto":
        compile_enabled = fused_backend or bool(
            runtime.execution_backend == "auto" and qualified and qualified["compile"]
        )
    if fused_backend and not compile_enabled:
        if runtime.execution_backend in ("gh200", "a100"):
            raise ValueError("forced fused execution requires compilation")
        backend, reason = "native", "compilation explicitly disabled"
        fused_backend = False
    fused_rope = runtime.fused_rope
    if fused_rope == "auto":
        fused_rope = fused_backend and config.model.heads in (5, 10) and runtime.projection_fusion
    if fused_rope and (not fused_backend or config.model.heads not in (5, 10)):
        raise ValueError("fused_rope requires the GH200 or A100 20M/90M path")
    if fused_rope and not runtime.projection_fusion:
        raise ValueError("fused_rope requires projection_fusion")
    if fused_backend and runtime.compile_scope == "blocks":
        raise ValueError("regional compilation is a native-backend experiment")
    return {
        "backend": backend,
        "compile": compile_enabled,
        "compile_mode": runtime.compile_mode,
        "compile_scope": runtime.compile_scope,
        "projection_fusion": runtime.projection_fusion if fused_backend else False,
        "fused_rope": fused_rope,
        "sdpa_backend": runtime.sdpa_backend,
        "reason": reason,
    }
