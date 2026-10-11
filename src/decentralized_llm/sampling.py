"""One global permutation stream; only optimizer-committed positions are durable."""

from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from decentralized_llm.cache import TokenCache, fingerprint
from decentralized_llm.config import Config
from decentralized_llm.state import IndexArray

SAMPLER_VERSION = 1


@dataclass(frozen=True)
class BatchSpec:
    start: int
    stop: int
    indices: IndexArray


def rank_count(start: int, stop: int, rank: int, world_size: int) -> int:
    first = start + (rank - start) % world_size
    return max(0, (stop - first + world_size - 1) // world_size)


class GlobalSampler:
    def __init__(self, cache: TokenCache, length: int, seed: int):
        self.blocks = cache.blocks("train", length)
        if self.blocks < 1:
            raise ValueError("training cache has no complete context block plus lookahead")
        self.identity = {
            "version": SAMPLER_VERSION,
            "cache_identity": cache.manifest["identity"],
            "context_length": length,
            "seed": seed,
            "blocks": self.blocks,
            "algorithm": "numpy-PCG64-shuffle",
            "numpy": np.__version__,
        }
        self.digest = fingerprint(self.identity)
        self.permutation = np.empty(self.blocks, dtype=np.int64)
        self.pass_index = -1

    def block(self, position: int) -> int:
        if position < 0:
            raise ValueError("negative global position")
        pass_index, offset = divmod(position, self.blocks)
        if pass_index != self.pass_index:
            # Refill in bounded chunks: never retain two full permutations at a pass boundary.
            for start in range(0, self.blocks, 65536):
                stop = min(start + 65536, self.blocks)
                self.permutation[start:stop] = np.arange(start, stop, dtype=np.int64)
            seed = int(fingerprint([self.digest, pass_index]), 16)
            np.random.Generator(np.random.PCG64(seed)).shuffle(self.permutation)
            self.pass_index = pass_index
        return int(self.permutation[offset])

    def batches(self, config: Config, rank: int, cursor: int, stop: int) -> Iterator[BatchSpec]:
        world = config.distributed.world_size
        if not 0 <= rank < world or cursor < 0 or cursor > stop:
            raise ValueError("invalid rank or sampling cursor")
        batch_blocks = config.training.batch_tokens // config.model.context_length
        while cursor < stop:
            update_stop = min(stop, (cursor // batch_blocks + 1) * batch_blocks)
            end = min(update_stop, cursor + config.training.micro_batch_size * world)
            first = cursor + (rank - cursor) % world
            indices = np.fromiter((self.block(p) for p in range(first, end, world)), dtype=np.int64)
            yield BatchSpec(cursor, end, indices)
            cursor = end


def memory_estimate(cache: TokenCache, config: Config) -> dict[str, int]:
    data, train = config.data, config.training
    length, depth, batch = config.model.context_length, data.prefetch_batches, train.micro_batch_size
    offsets = cache.offsets["train"]
    # Bound crossings using actual shard geometry, including a possible one-token final shard.
    shard_crossings = max(
        (
            bisect_left(offsets, min(offsets[-1], boundary + length)) - index + 1
            for index, boundary in enumerate(offsets[1:-1], 1)
        ),
        default=1,
    )
    read_chunks = (length + data.read_range_bytes // 2) // (data.read_range_bytes // 2)
    segments = shard_crossings + read_chunks

    parts = {
        "permutation": cache.blocks("train", length) * 8 + 65536 * 8,
        "compact_windows": (depth + 2) * batch * (length + 1) * 2,
        "read_scratch": data.reader_threads * data.read_range_bytes * 3,
        "request_metadata": 2 * depth * batch * segments * 2048,
        "transfer_buffers": 2 * batch * length * 2 * 8,
        "validation_buffers": config.evaluation.batch_size * (length + 1) * 32
        + min(config.evaluation.subset_blocks, cache.blocks("validation", length, partial=True)) * 64
        + cache.blocks("validation", length, partial=True) * 8,
        "file_descriptors": 64 * data.reader_threads * 1024,
        "telemetry_and_runtime": 2 * 1024 * 1024,
    }
    parts["total"] = sum(parts.values())
    return parts


def check_memory(cache: TokenCache, config: Config) -> dict[str, int]:
    estimate = memory_estimate(cache, config)
    budget = config.data.host_memory_mib * 1024**2
    if estimate["total"] > budget:
        raise ValueError(
            f"loader requires approximately {estimate['total'] / 1024**2:.1f} MiB per rank; "
            f"host_memory_mib={config.data.host_memory_mib}. Increase budget or reduce buffers."
        )
    return estimate
