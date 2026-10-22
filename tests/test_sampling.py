from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from conftest import make_cache

from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import TokenLoader
from decentralized_llm.data.sampling import GlobalSampler, check_memory, sampling_plan
from decentralized_llm.launch.preflight import validate


def test_sampler_sequence(cache: TokenCache, tiny_config: Config) -> None:
    # Captured from the original sampler with the pinned NumPy 2.5.3.
    expected = [
        int(value)
        for value in (
            "12 5 0 11 4 8 16 20 24 9 22 27 14 28 36 6 17 2 32 "
            "1 33 25 21 29 19 35 18 30 34 26 15 7 13 3 23 10 31 "
            "16 23 36 0 26 27 14 17 21 11 10 30 29 9 31 35 32 22 25 "
            "18 13 1 33 8 3 24 4 7 2 28 19 15 5 6 34 20 12"
        ).split()
    ]
    sampler = GlobalSampler.from_config(cache, tiny_config)
    assert sampler.identity == {
        "version": 1,
        "cache_identity": "9c70e169288cdb407a8682fea3497c38dd3b698ed1864ae3b8709cfb5c97aa1e",
        "context_length": 4,
        "seed": 42,
        "blocks": 37,
        "algorithm": "numpy-PCG64-shuffle",
        "numpy": "2.5.3",
    }
    assert [sampler.block(p) for p in range(74)] == expected


@pytest.mark.parametrize("factor,pool_size", [(2, 21), (3, 14), (5, 9), (7, 6), (42, 1), (100, 1)])
def test_reuse_pool_and_passes(cache: TokenCache, tiny_config: Config, factor: int, pool_size: int) -> None:
    tiny_config.training.data_reuse_factor = factor
    sampler = GlobalSampler.from_config(cache, tiny_config)
    assert sampler.index_bytes == 0  # Identity validation must not build a permutation.
    assert sampler.blocks == pool_size
    legacy = GlobalSampler(cache, 4, 42)
    expected_pool = {legacy.block(p) for p in range(pool_size)}
    stream = [sampler.block(p) for p in range(42)]
    assert len(stream) == 42
    assert set(stream) == expected_pool
    for start in range(0, 42, pool_size):
        one_pass = stream[start : start + pool_size]
        assert len(one_pass) == len(set(one_pass))
        if len(one_pass) == pool_size:
            assert set(one_pass) == expected_pool
    full_passes, remainder = divmod(42, pool_size)
    multiplicities = Counter(stream)
    assert sorted(multiplicities.values()) == [full_passes] * (pool_size - remainder) + [full_passes + 1] * remainder
    if pool_size > 1:
        assert stream[:pool_size] != stream[pool_size : 2 * pool_size]
    np.testing.assert_array_equal(sampler.pool, [legacy.block(p) for p in range(pool_size)])
    assert not sampler.pool.flags.writeable
    # Fresh samplers can seek directly into any pass, including backwards.
    replay = GlobalSampler.from_config(cache, tiny_config)
    for position in (41, 0, pool_size, pool_size - 1, 17):
        assert replay.block(position) == stream[position]
    assert sampler.index_bytes == 2 * pool_size * 8
    sampler.close()
    assert sampler.index_bytes == sampler.pool.size == sampler.permutation.size == 0


def test_pool_seed_and_nested_factors(cache: TokenCache, tiny_config: Config) -> None:
    tiny_config.training.data_reuse_factor = 2
    first = GlobalSampler.from_config(cache, tiny_config)
    first.block(0)
    tiny_config.training.data_reuse_factor = 5
    smaller = GlobalSampler.from_config(cache, tiny_config)
    smaller.block(0)
    assert set(smaller.pool) < set(first.pool)
    tiny_config.runtime.seed += 1
    changed = GlobalSampler.from_config(cache, tiny_config)
    changed.block(0)
    assert set(changed.pool) != set(smaller.pool)


def test_sampling_memory_and_preflight(cache: TokenCache, tiny_config: Config) -> None:
    baseline = check_memory(cache, tiny_config)
    tiny_config.training.data_reuse_factor = 5
    plan = sampling_plan(cache, tiny_config)
    assert plan.total_blocks == 42
    assert plan.summary() == {
        "data_reuse_factor": 5,
        "pool_blocks": 9,
        "unique_training_tokens": 36,
        "full_passes": 4,
        "partial_pass_blocks": 6,
    }
    memory = check_memory(cache, tiny_config)
    assert memory["permutation"] == baseline["permutation"] + 9 * 8
    assert memory["permutation"] >= (37 + 9) * 8  # Peak during selection.
    assert memory["permutation"] >= 2 * 9 * 8  # Retained pool plus pass permutation.
    report = validate(tiny_config)
    assert report["sampling"] == plan.summary()
    assert report["global_tokens"] == 168
    assert report["virtual_epochs"] == 2
    assert not tiny_config.runtime.output_dir.exists()
    with TokenLoader(cache, tiny_config, 0, 0, 6) as loader:
        loader.warmup()
        assert loader.metrics()["sampler_index_bytes"] == 2 * 9 * 8
    assert loader.sampler.index_bytes == 0


def test_reuse_rejects_small_cache(tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_config.data.cache_dir = tmp_path / "small-cache"
    cache = make_cache(tiny_config.data.cache_dir, tiny_config, tokens=33)
    # The old K=1 path continues to cycle even an undersized cache.
    assert sampling_plan(cache, tiny_config).pool_blocks == 8
    tiny_config.training.data_reuse_factor = 5  # Requires 9 blocks.

    def fail_start(*_: object) -> None:
        pytest.fail("loader threads started before cache size validation")

    monkeypatch.setattr("decentralized_llm.data.loader.threading.Thread.start", fail_start)
    for action in (
        lambda: validate(tiny_config),
        lambda: GlobalSampler.from_config(cache, tiny_config),
        lambda: TokenLoader(cache, tiny_config, 0, 0, 6),
    ):
        with pytest.raises(ValueError, match=r"data_reuse_factor=5 requires 9.*37 cached tokens.*cache has 8"):
            action()
    tiny_config.training.data_reuse_factor = 6  # Now the pool fits.
    assert sampling_plan(cache, tiny_config).pool_blocks == 7


@pytest.mark.parametrize("factor", [1, 5])
@pytest.mark.parametrize("stop", [0, 6, 120])
def test_benchmark_resident_memory_horizon(cache: TokenCache, tiny_config: Config, factor: int, stop: int) -> None:
    tiny_config.training.data_reuse_factor = factor
    tiny_config.data.loader_mode = "resident"
    plan = sampling_plan(cache, tiny_config)
    ordinary = check_memory(cache, tiny_config, stop=stop)
    bounded = check_memory(cache, tiny_config, stop=stop, bound_resident_to_stop=True)
    blocks = min(plan.pool_blocks, stop) if factor == 1 else plan.pool_blocks
    assert bounded["resident_tokens"] == blocks * 5 * 2
    assert bounded["resident_indices"] == bounded["resident_read_state"] == blocks * 8
    assert bounded["permutation"] == ordinary["permutation"]
    assert ordinary["resident_tokens"] == min(plan.pool_blocks, max(plan.total_blocks, stop)) * 5 * 2
    if factor > 1:
        assert bounded == ordinary


@pytest.mark.parametrize("stop", [None, -1])
def test_benchmark_resident_memory_requires_horizon(cache: TokenCache, tiny_config: Config, stop: int | None) -> None:
    with pytest.raises(ValueError, match="nonnegative stop"):
        check_memory(cache, tiny_config, stop=stop, bound_resident_to_stop=True)
