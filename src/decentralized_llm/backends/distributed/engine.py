"""Real distributed model state and update execution."""

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from decentralized_llm.algorithms.adamw import make_optimizer, optimizer_report
from decentralized_llm.backends.api import Runtime
from decentralized_llm.backends.distributed import update as updates
from decentralized_llm.backends.distributed.runtime import averaged_model, parameter_mix_report
from decentralized_llm.config import Config
from decentralized_llm.data.sampling import NextBatch
from decentralized_llm.execution.native import TrainingExecution, build_execution
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.storage import parameter_storage


class DistributedBackend:
    def __init__(self, config: Config, device: torch.device, runtime: Runtime):
        self.config, self.device, self.runtime = config, device, runtime
        self.model = initialize_model(config).to(device)
        if dist.is_initialized():
            with torch.no_grad():
                for parameter in self.model.parameters():
                    dist.broadcast(parameter, src=0)
        self.optimizer = make_optimizer(self.model, config, device)
        self.execution: TrainingExecution

    def prepare(self) -> None:
        self.execution = build_distributed_execution(self.model, self.config, self.device)

    def update(
        self,
        next_batch: NextBatch,
        preload: Callable[[], None],
        start: int,
        stop: int,
        step: int,
        total: int,
        *,
        collect_metrics: bool = True,
    ) -> dict[str, Any]:
        return updates.optimizer_update(
            self.model,
            self.execution,
            self.optimizer,
            self.config,
            self.device,
            next_batch,
            preload,
            start,
            stop,
            step,
            total,
            collect_metrics=collect_metrics,
        )

    def inspect(self, expected_step: int | None = None) -> None:
        pass

    @torch.no_grad()
    def evaluation_model(self, average: bool = True) -> tuple[Llama, torch.Tensor]:
        model = self.model
        reference = (
            averaged_model(model)
            if average and self.config.distributed.mode == "decentralized"
            else copy.deepcopy(model)
        )
        squared = torch.zeros((), device=self.device, dtype=torch.float64)
        if average and self.config.distributed.mode == "decentralized":
            for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
                squared.add_((parameter.double() - other.double()).square().sum())
        return cast(Llama, reference), squared.sqrt()

    def state_dict(self, training_state: bool) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.config.distributed.mode == "decentralized" or self.runtime.rank == 0:
            state["model"] = self.model.state_dict()
            if training_state:
                state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any], training_state: bool) -> None:
        self.model.load_state_dict(state["model"], strict=True)
        if training_state:
            self.optimizer.load_state_dict(state["optimizer"])

    def memory_report(self) -> dict[str, Any]:
        storage = parameter_storage(self.model)
        return {
            "parameter_bytes": sum(p.numel() * p.element_size() for p in storage.parameters),
            "parameter_storage_bytes": storage.flat.nbytes,
            "allocation_bytes": storage.flat.untyped_storage().nbytes(),
            "optimizer": optimizer_report(self.optimizer),
            "parameter_mixing": parameter_mix_report(self.model),
        }


def build_distributed_execution(model: Llama, config: Config, device: torch.device) -> TrainingExecution:
    """Wrap the loss for gradient collectives before optional compilation."""

    def wrap(loss: nn.Module) -> nn.Module:
        if config.distributed.mode != "ddp" or not dist.is_initialized() or dist.get_world_size() == 1:
            return loss
        cfg = config.distributed
        return DDP(
            loss,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            bucket_cap_mb=cfg.ddp_bucket_cap_mb,
            gradient_as_bucket_view=cfg.ddp_gradient_as_bucket_view,
            batched_grad_copy=cfg.ddp_batched_grad_copy,
        )

    return build_execution(model, config, device, wrap=wrap)
