"""One execution policy for training, evaluation and distributed benchmarks."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from decentralized_llm.config import Config
from decentralized_llm.execution.compilation import configure_compile_cache
from decentralized_llm.execution.policy import resolve_execution
from decentralized_llm.models.llama import Llama, token_losses
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.models.storage import prepare_parameter_storage


class LossModule(nn.Module):
    def __init__(self, model: Llama | PackedLlama, sdpa_backend: str = "auto", *, evaluation: bool = False):
        super().__init__()
        self.model = model
        self.sdpa_backend = sdpa_backend
        self.evaluation = evaluation

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        context = (
            nullcontext()
            if self.sdpa_backend == "auto"
            else sdpa_kernel(
                {
                    "flash": SDPBackend.FLASH_ATTENTION,
                    "cudnn": SDPBackend.CUDNN_ATTENTION,
                    "math": SDPBackend.MATH,
                }[self.sdpa_backend]
            )
        )
        with context:
            if self.model.execution_backend in ("gh200", "a100"):
                if isinstance(self.model, PackedLlama):
                    from decentralized_llm.execution.cuda.packed_model import forward as packed_forward

                    logits = packed_forward(self.model, x)
                else:
                    from decentralized_llm.execution.cuda.model import forward

                    logits = forward(self.model, x)
            else:
                logits = self.model(x)
            if self.model.execution_backend in ("gh200", "a100"):
                from decentralized_llm.execution.cuda.loss import token_losses as gh200_losses

                losses = gh200_losses(logits, y)
            else:
                losses = token_losses(logits, y)
            if isinstance(self.model, PackedLlama):
                return losses.sum(dim=(1, 2))
            return losses.sum(dtype=torch.float64 if self.evaluation else None)


class TrainingExecution(nn.Module):
    """Keep no_sync on the real DDP object, independent of compiler wrappers."""

    def __init__(
        self,
        model: Llama | PackedLlama,
        config: Config,
        device: torch.device,
        *,
        distributed: bool = True,
        wrap: Callable[[nn.Module], nn.Module] | None = None,
    ):
        super().__init__()
        if not isinstance(model, PackedLlama):
            prepare_parameter_storage(model)
        self.policy = model.execution_policy or resolve_execution(config, device)
        model.execution_backend = self.policy["backend"]
        model.projection_fusion = self.policy["projection_fusion"]
        model.fused_rope = self.policy["fused_rope"]
        model.execution_policy = self.policy
        self.cache = configure_compile_cache(config, device) if self.policy["compile"] else None
        self.eager: nn.Module = LossModule(model, config.runtime.sdpa_backend, evaluation=not distributed)
        if wrap is not None:
            self.eager = wrap(self.eager)
        self._no_sync: Callable[[], AbstractContextManager] = getattr(self.eager, "no_sync", nullcontext)
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

        if isinstance(model, PackedLlama):
            self.shape = (
                model.num_models,
                config.process_micro_batch_size // model.num_models,
                config.model.context_length,
            )

    def no_sync(self):
        return self._no_sync()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if tuple(x.shape) == self.shape:
            return self.compiled(x, y)
        if not self.policy["compile"]:
            return self.eager(x, y)
        # Regional block wrappers live inside the canonical model; bypass them too.
        with torch.compiler.set_stance("force_eager"):
            return self.eager(x, y)


def build_execution(
    model: Llama | PackedLlama,
    config: Config,
    device: torch.device,
    *,
    distributed: bool = True,
    wrap: Callable[[nn.Module], nn.Module] | None = None,
) -> TrainingExecution:
    return TrainingExecution(model, config, device, distributed=distributed, wrap=wrap)
