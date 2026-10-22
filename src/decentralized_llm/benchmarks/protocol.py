"""Shared finite update and loader bounds for training measurements."""

import math
from dataclasses import dataclass

from decentralized_llm.config import Config


@dataclass(frozen=True)
class BenchmarkBounds:
    cycle: int
    warmup: int
    chunk: int
    maximum_steps: int
    blocks: int
    total_blocks: int


def rotation_period(config: Config) -> int:
    if config.distributed.mode == "ddp" or config.distributed.topology == "complete":
        return 1
    return (
        2 if config.distributed.topology == "one_peer_ring" else max(1, config.distributed.world_size.bit_length() - 1)
    )


def benchmark_bounds(config: Config) -> BenchmarkBounds:
    """Round to complete communication cycles and reserve one final read-ahead chunk."""
    cycle = rotation_period(config)
    warmup = math.ceil(config.benchmark.warmup / cycle) * cycle
    chunk = math.ceil(config.benchmark.steps / cycle) * cycle
    maximum_steps = warmup + chunk * (config.benchmark.windows * config.benchmark.max_chunks + 1)
    blocks = config.training.batch_tokens // config.model.context_length
    return BenchmarkBounds(cycle, warmup, chunk, maximum_steps, blocks, maximum_steps * blocks)
