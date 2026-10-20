"""Batched independent workers, or one shared synchronous model."""

from collections.abc import Callable
from typing import Any

import torch

from decentralized_llm.algorithms.packed_adamw import PackedAdamW
from decentralized_llm.algorithms.schedules import learning_rate
from decentralized_llm.backends.api import Runtime
from decentralized_llm.config import Config
from decentralized_llm.data.sampling import NextBatch
from decentralized_llm.execution.native import TrainingExecution, build_execution
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.packed import PackedLlama


class PackedBackend:
    def __init__(self, config: Config, device: torch.device, runtime: Runtime):
        self.config, self.device, self.runtime = config, device, runtime
        self.workers = config.distributed.world_size if config.distributed.mode == "decentralized" else 1
        self.model = PackedLlama(
            config.model, self.workers, config.runtime.attention_backend, canonical=initialize_model(config)
        ).to(device)
        self.optimizer: Any = PackedAdamW(self.model, config)
        self.execution: TrainingExecution
        self.capture: Any = None

    def prepare(self) -> None:
        self.execution = build_execution(self.model, self.config, self.device)
        if (
            self.execution.policy["backend"] in ("gh200", "a100")
            and self.config.runtime.fused_optimizer
            and self.config.runtime.decentralized_optimizer != "reference"
        ):
            from decentralized_llm.execution.cuda.capture import CapturedUpdate
            from decentralized_llm.execution.cuda.packed_optimizer import ArenaAdamW

            state = self.optimizer.state_dict()
            self.optimizer = ArenaAdamW(self.model, self.config)
            self.optimizer.load_state_dict(state)
            self.capture = CapturedUpdate(self.model, self.optimizer, self.execution, self.config, self.device)
            self.capture.prepare()
        self.execution.policy["update_capture"] = self.capture is not None

    def arrange(self, tensor: torch.Tensor) -> torch.Tensor:
        # The global sampler orders position r, r+N, ... onto worker r.
        return tensor.view(-1, self.workers, tensor.shape[-1]).transpose(0, 1).contiguous()

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
        cfg = self.config
        length = cfg.model.context_length
        lr = learning_rate(cfg, stop * length, total * length)
        if self.capture is not None:
            self.capture.execute(next_batch, preload, start, stop, lr)
            # Do not commit host progress or loader cursors for a rejected update.
            self.inspect(step + 1)
            if not collect_metrics:
                return {}
            norms, means = self.optimizer.norms, self.optimizer.losses
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad()
            cursor = start
            loss_sum = torch.zeros(self.workers, device=self.device)
            denominator = (stop - start) * length / self.workers
            while cursor < stop:
                (x, y), spec = next_batch()
                if spec.start != cursor or spec.stop > stop:
                    raise ValueError("loader/optimizer global cursor mismatch")
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cfg.runtime.amp):
                    losses = self.execution(self.arrange(x), self.arrange(y))
                    loss = losses.sum() / denominator
                loss.backward()
                loss_sum += losses.detach()
                cursor = spec.stop
                preload()
            norms = self.optimizer.clip_grad_norm_(cfg.optimizer.grad_clip)
            if not torch.isfinite(loss_sum).all():
                raise FloatingPointError("nonfinite packed loss")
            if cfg.distributed.mode == "decentralized" and cfg.distributed.scheme == "awc":
                self.model.mix_(cfg.distributed.topology, step)
            self.optimizer.step()
            if cfg.distributed.mode == "decentralized" and cfg.distributed.scheme == "atc":
                self.model.mix_(cfg.distributed.topology, step)
            if not collect_metrics:
                return {}
            means = loss_sum / denominator
        return {
            "lr": lr,
            "gradient_norm": float(norms.mean()),
            "local_loss": float(means.mean()),
            "local_gradient_norms": norms.tolist(),
            "local_losses": means.tolist(),
        }

    def inspect(self, expected_step: int | None = None) -> None:
        if self.capture is not None:
            self.optimizer.inspect(expected_step)

    @torch.no_grad()
    def evaluation_model(self, average: bool = True) -> tuple[Llama, torch.Tensor]:
        if not average and self.workers > 1:
            raise ValueError("packed decentralized evaluation requires averaged parameters")
        model = initialize_model(self.config).to(self.device)
        model.execution_policy = self.model.execution_policy.copy()
        self.model.copy_average_to(model)
        squared = torch.zeros(self.workers, device=self.device, dtype=torch.float64)
        for entry in self.model.layout:
            local = self.model.get_parameter(entry.name).double()
            squared += (local - model.get_parameter(entry.name).double()).flatten(1).square().sum(1)
        return model, squared.sqrt().mean()

    def state_dict(self, training_state: bool) -> dict[str, Any]:
        self.inspect()
        state = {"model": self.model.packed_state_dict()}
        if training_state:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any], training_state: bool) -> None:
        self.model.load_packed_state_dict(state["model"])
        if training_state:
            self.optimizer.load_state_dict(state["optimizer"])

    def memory_report(self) -> dict[str, Any]:
        previous = self.optimizer.previous_update
        mixing = self.optimizer.scratch if self.capture is not None else self.model.mixing_scratch_bytes
        mixing_bytes = mixing.nbytes if isinstance(mixing, torch.Tensor) else mixing or 0
        return {
            "parameter_bytes": self.model.parameter_count * self.model.parameter_storage.element_size(),
            "parameter_storage_bytes": self.model.parameter_storage.nbytes,
            "optimizer": {
                "implementation": "triton" if self.capture is not None else "reference",
                "history_bytes": previous.nbytes if previous is not None else 0,
                "scratch_bytes": (
                    self.optimizer.scratch.nbytes if self.capture is None and self.optimizer.scratch is not None else 0
                ),
            },
            "parameter_mixing": {"snapshot_bytes": mixing_bytes, "retained_snapshot_bytes": mixing_bytes},
        }
