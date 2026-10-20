"""Contracts between shared orchestration and concrete training backends."""

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn

from decentralized_llm.config import Config
from decentralized_llm.data.sampling import NextBatch
from decentralized_llm.models.llama import Llama


class Runtime(Protocol):
    @property
    def rank(self) -> int: ...
    @property
    def size(self) -> int: ...
    def setup(self, config: Config) -> torch.device: ...
    def barrier(self) -> None: ...
    def reduce(self, value: torch.Tensor, *, maximum: bool = False) -> None: ...
    def gather(self, value: Any) -> list[Any]: ...
    def close(self) -> None: ...


class TrainingBackend(Protocol):
    config: Config
    device: torch.device
    runtime: Runtime

    @property
    def model(self) -> nn.Module: ...

    optimizer: Any
    execution: Any

    def prepare(self) -> None: ...
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
    ) -> dict[str, Any]: ...
    def inspect(self, expected_step: int | None = None) -> None: ...
    def evaluation_model(self, average: bool = True) -> tuple[Llama, torch.Tensor]: ...
    def state_dict(self, training_state: bool) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any], training_state: bool) -> None: ...
    def memory_report(self) -> dict[str, Any]: ...
