"""Explicit factories; importing launch/configuration code loads no CUDA kernels."""

import torch

from decentralized_llm.backends.api import Runtime, TrainingBackend
from decentralized_llm.config import Config


def runtime_for(config: Config) -> Runtime:
    if config.runtime.training_backend == "packed":
        from decentralized_llm.backends.packed.runtime import PackedRuntime

        return PackedRuntime()
    from decentralized_llm.backends.distributed.runtime import DistributedRuntime

    return DistributedRuntime()


def create_backend(config: Config, device: torch.device, runtime: Runtime | None = None) -> TrainingBackend:
    context = runtime or runtime_for(config)
    if config.runtime.training_backend == "packed":
        from decentralized_llm.backends.packed.engine import PackedBackend

        return PackedBackend(config, device, context)
    from decentralized_llm.backends.distributed.engine import DistributedBackend

    return DistributedBackend(config, device, context)
