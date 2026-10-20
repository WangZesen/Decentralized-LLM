"""One physical process with an arbitrary logical worker count."""

import os
from typing import Any

import torch

from decentralized_llm.config import Config
from decentralized_llm.execution.device import setup_device


class PackedRuntime:
    rank = 0
    size = 1

    def setup(self, config: Config) -> torch.device:
        if (
            int(os.environ.get("WORLD_SIZE", "1")) != 1
            or int(os.environ.get("RANK", "0")) != 0
            or int(os.environ.get("LOCAL_RANK", "0")) != 0
        ):
            raise ValueError("packed execution requires exactly one physical process")
        return setup_device(config)

    def barrier(self) -> None:
        pass

    def reduce(self, value: torch.Tensor, *, maximum: bool = False) -> None:
        pass

    def gather(self, value: Any) -> list[Any]:
        return [value]

    def close(self) -> None:
        pass
