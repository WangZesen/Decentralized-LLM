"""One global permutation stream; only optimizer-committed positions are durable."""

from bisect import bisect_left
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache, fingerprint, training_boundaries
from decentralized_llm.state import Batch, IndexArray

SAMPLER_VERSION = 1
REUSE_SAMPLER_VERSION = 2
PRELOAD_BLOCKS = 1024


@dataclass(frozen=True)
class SamplingPlan:
    eligible_blocks: int
    pool_blocks: int
    total_blocks: int
    reuse_factor: int
    context_length: int
    reuse_order: Literal["reshuffle", "repeat"] = "reshuffle"

    def summary(self) -> dict[str, int | str]:
        full_passes, partial_pass_blocks = divmod(self.total_blocks, self.pool_blocks)
        return {
            "data_reuse_factor": self.reuse_factor,
            "pool_blocks": self.pool_blocks,
            "unique_training_tokens": min(self.total_blocks, self.pool_blocks) * self.context_length,
            "full_passes": full_passes,
            "partial_pass_blocks": partial_pass_blocks,
            **({"data_reuse_order": "repeat"} if self.reuse_order == "repeat" else {}),
        }


def sampling_plan(cache: TokenCache, config: Config) -> SamplingPlan:
    """Derive the pool from the full recipe, never from a loader's cursor or stop."""
    length = config.model.context_length
    eligible = cache.blocks("train", length)
    if eligible < 1:
        raise ValueError("training cache has no complete context block plus lookahead")
    total = training_boundaries(config, config.model.parameter_count)[-1]
    factor = config.training.data_reuse_factor
    pool = eligible if factor == 1 else (total + factor - 1) // factor
    if pool > eligible:
        raise ValueError(
            f"data_reuse_factor={factor} requires {pool} unique training blocks "
            f"({pool * length + 1} cached tokens including lookahead); cache has {eligible} blocks. "
            "Prepare a larger cache at a new path."
        )
    return SamplingPlan(eligible, pool, total, factor, length, config.training.data_reuse_order)


def _fill_indices(indices: IndexArray) -> None:
    # Bounded scratch space even for very large caches.
    for start in range(0, len(indices), 65536):
        stop = min(start + 65536, len(indices))
        indices[start:stop] = np.arange(start, stop, dtype=np.int64)


def _shuffle(indices: IndexArray, digest: str, pass_index: int) -> None:
    seed = int(fingerprint([digest, pass_index]), 16)
    np.random.Generator(np.random.PCG64(seed)).shuffle(indices)


@dataclass(frozen=True)
class BatchSpec:
    start: int
    stop: int
    indices: IndexArray


type NextBatch = Callable[[], tuple[Batch, BatchSpec]]


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
        self._order_digest = self.digest
        self._repeat_order = False
        self._eligible_blocks = self.blocks
        self._selection_digest: str | None = None
        # Identity checks need no index allocation or shuffling.
        self.pool = np.empty(0, dtype=np.int64)
        self.permutation = np.empty(0, dtype=np.int64)
        self.pass_index = -1

    @classmethod
    def from_config(cls, cache: TokenCache, config: Config) -> GlobalSampler:
        plan = sampling_plan(cache, config)
        sampler = cls(cache, config.model.context_length, config.runtime.seed)
        if plan.reuse_factor > 1:
            sampler._selection_digest = sampler.digest
            sampler.blocks = plan.pool_blocks
            sampler.identity = {
                **sampler.identity,
                "version": REUSE_SAMPLER_VERSION,
                "blocks": plan.pool_blocks,
                "eligible_blocks": plan.eligible_blocks,
                "reuse_factor": plan.reuse_factor,
                "total_blocks": plan.total_blocks,
                "selection_algorithm": "sampler-v1-first-pass-prefix",
            }
            sampler.digest = fingerprint(sampler.identity)
        # Keep selection and the first pass identical to the existing sampler.
        # The extra identity field distinguishes resume/trial identity, not RNG order.
        sampler._order_digest = sampler.digest
        if plan.reuse_order == "repeat":
            sampler._repeat_order = True
            sampler.identity = {**sampler.identity, "data_reuse_order": "repeat"}
            sampler.digest = fingerprint(sampler.identity)
        return sampler

    def _initialize(self) -> None:
        if self._selection_digest is not None:
            candidates = np.empty(self._eligible_blocks, dtype=np.int64)
            _fill_indices(candidates)
            _shuffle(candidates, self._selection_digest, 0)
            self.pool = candidates[: self.blocks].copy()
            self.pool.flags.writeable = False
            del candidates
        self.permutation = np.empty(self.blocks, dtype=np.int64)

    @property
    def index_bytes(self) -> int:
        return self.pool.nbytes + self.permutation.nbytes

    def close(self) -> None:
        self.pool = np.empty(0, dtype=np.int64)
        self.permutation = np.empty(0, dtype=np.int64)
        self.pass_index = -1

    def block(self, position: int) -> int:
        if position < 0:
            raise ValueError("negative global position")
        if not self.permutation.size:
            self._initialize()
        pass_index, offset = divmod(position, self.blocks)
        if pass_index != self.pass_index:
            if not self._repeat_order or self.pass_index < 0:
                if self._selection_digest is None:
                    _fill_indices(self.permutation)
                else:
                    np.copyto(self.permutation, self.pool)
                _shuffle(self.permutation, self._order_digest, 0 if self._repeat_order else pass_index)
            self.pass_index = pass_index
        return int(self.permutation[offset])

    def batches(self, config: Config, rank: int, cursor: int, stop: int) -> Iterator[BatchSpec]:
        world = config.distributed.world_size
        if not 0 <= rank < world or cursor < 0 or cursor > stop:
            raise ValueError("invalid rank or sampling cursor")
        batch_blocks = config.training.batch_tokens // config.model.context_length
        micro_batch_blocks = config.micro_batch_size * world
        while cursor < stop:
            update_stop = min(stop, (cursor // batch_blocks + 1) * batch_blocks)
            end = min(update_stop, cursor + micro_batch_blocks)
            first = cursor + (rank - cursor) % world
            positions = range(cursor, end) if config.runtime.training_backend == "packed" else range(first, end, world)
            indices = np.fromiter((self.block(p) for p in positions), dtype=np.int64)
            yield BatchSpec(cursor, end, indices)
            cursor = end


def memory_estimate(
    cache: TokenCache, config: Config, *, stop: int | None = None, bound_resident_to_stop: bool = False
) -> dict[str, int]:
    if bound_resident_to_stop and (stop is None or stop < 0):
        raise ValueError("bounded resident loading requires a nonnegative stop")
    plan = sampling_plan(cache, config)
    data = config.data
    length, depth, batch = config.model.context_length, data.prefetch_batches, config.process_micro_batch_size
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

    # Selection briefly holds the full cache permutation plus the copied pool.
    # After selection only the pool and its current permutation remain (2 * pool).
    index_blocks = plan.eligible_blocks + (plan.pool_blocks if plan.reuse_factor > 1 else 0)
    parts = {
        "permutation": index_blocks * 8 + 65536 * 8,
        "compact_windows": (depth + 2) * batch * (length + 1) * 2,
        "read_scratch": data.reader_threads * data.read_range_bytes * 3,
        "request_metadata": 2 * depth * batch * segments * 2048,
        "transfer_buffers": 2 * batch * length * 2 * 8,
        "validation_buffers": config.evaluation.batch_size * (length + 1) * 32
        + min(config.evaluation.subset_blocks, cache.blocks("validation", length, partial=True)) * 64
        + cache.blocks("validation", length, partial=True) * 8,
        "file_descriptors": 64 * data.reader_threads * 1024,
        "telemetry_and_runtime": 2 * 1024 * 1024,
        "resident_tokens": 0,
        "resident_indices": 0,
        "resident_read_state": 0,
        "preload_metadata": 0,
    }
    # K=1 may use only a prefix of its first pass through a much larger cache.
    # Cache exactly that working set, while preserving the full sampler permutation.
    # Loader-only callers may explicitly read beyond the configured training budget.
    # Benchmarks may explicitly bound this working set to their finite horizon.
    # Reused data always retains its complete fixed pool, including future passes.
    resident_horizon = max(plan.total_blocks, stop or 0)
    if bound_resident_to_stop and plan.reuse_factor == 1:
        resident_horizon = stop or 0
    resident_blocks = min(plan.pool_blocks, resident_horizon)
    resident = {
        **parts,
        "resident_tokens": resident_blocks * (length + 1) * 2,
        "resident_indices": resident_blocks * 8,
        "resident_read_state": resident_blocks * 8,
        "preload_metadata": 2 * min(PRELOAD_BLOCKS, resident_blocks) * segments * 2048,
    }
    if data.loader_mode == "resident" or (
        data.loader_mode == "auto" and sum(resident.values()) <= data.host_memory_mib * 1024**2
    ):
        parts = resident
    parts["total"] = sum(parts.values())
    return parts


def check_memory(
    cache: TokenCache, config: Config, *, stop: int | None = None, bound_resident_to_stop: bool = False
) -> dict[str, int]:
    estimate = memory_estimate(cache, config, stop=stop, bound_resident_to_stop=bound_resident_to_stop)
    budget = config.data.host_memory_mib * 1024**2
    if estimate["total"] > budget:
        raise ValueError(
            f"{config.data.loader_mode} loader requires approximately {estimate['total'] / 1024**2:.1f} MiB per rank; "
            f"host_memory_mib={config.data.host_memory_mib}. Increase budget, reduce buffers, "
            "or use data.loader_mode=auto/streaming."
        )
    return estimate
