"""Explicit FP32 arithmetic policy, shared by training and evaluation."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


def _backends() -> dict[str, Any]:
    return {
        "global": torch.backends,
        "cuda_matmul": torch.backends.cuda.matmul,
        "cudnn": torch.backends.cudnn,
        # PyTorch exposes this dynamically; its shipped stubs omit it.
        "cudnn_conv": torch.backends.cudnn.conv,  # pyright: ignore[reportAttributeAccessIssue]
        "cudnn_rnn": torch.backends.cudnn.rnn,
    }


def precision_state() -> dict[str, str]:
    return {name: backend.fp32_precision for name, backend in _backends().items()}


def precision_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}


def set_precision(precision: str | None) -> None:
    if precision is None:
        return
    if precision not in ("ieee", "tf32"):
        raise ValueError(f"unsupported FP32 precision: {precision}")
    overrides = precision_environment()
    if overrides["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "1" or overrides["NVIDIA_TF32_OVERRIDE"] is not None:
        raise ValueError("explicit FP32 precision requires unset TF32 environment overrides")
    # Keep other operations IEEE; permit TF32 only for CUDA matrix multiplications.
    for name, backend in _backends().items():
        backend.fp32_precision = precision if name == "cuda_matmul" else "ieee"


@contextmanager
def precision_context(precision: str | None) -> Iterator[None]:
    if precision is None:
        yield
        return
    previous = precision_state()
    try:
        set_precision(precision)
        yield
    finally:
        for name, backend in _backends().items():
            backend.fp32_precision = previous[name]


def precision_metadata(training: str | None, evaluation: str | None, model: torch.nn.Module) -> dict:
    return {
        "training_requested": training,
        "evaluation_requested": evaluation,
        "effective_training": precision_state(),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "environment_overrides": precision_environment(),
    }
