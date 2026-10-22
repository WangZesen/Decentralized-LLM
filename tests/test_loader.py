import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_cache

from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, PositionalReader, ReadCancelled, Segment, TokenLoader
from decentralized_llm.data.sampling import GlobalSampler, check_memory


def collect(
    cache: TokenCache, config: Config, cursor: int = 0, stop: int = 94, rank: int = 0
) -> tuple[list[int], list[int]]:
    samples, tokens = [], []
    with TokenLoader(cache, config, rank, cursor, stop) as loader:
        loader.warmup()
        while True:
            try:
                batch = loader.next()
            except StopIteration:
                break
            samples.extend(batch.spec.indices.tolist())
            tokens.extend(batch.tokens.flatten().tolist())
            for index, row in zip(batch.spec.indices, batch.tokens, strict=True):
                expected = (np.arange(int(index) * 4, int(index) * 4 + 5) % 17).astype("<u2")
                np.testing.assert_array_equal(row, expected)
    return samples, tokens


def test_global_coverage(cache: TokenCache, tiny_config: Config) -> None:
    sampler = GlobalSampler(cache, 4, 42)
    count = sampler.blocks
    one = [sampler.block(i) for i in range(count)]
    two = [sampler.block(i) for i in range(count, 2 * count)]
    assert sorted(one) == sorted(two) == list(range(count))
    assert one != two
    raw = tiny_config.model_dump()
    raw["distributed"]["gpus_per_node"] = 4
    raw["training"]["batch_tokens"] = 48
    cfg = Config.model_validate(raw)
    streams = [collect(cache, cfg, stop=2 * count, rank=r)[0] for r in range(4)]
    for rank, stream in enumerate(streams):
        assert stream == (one + two)[rank::4]


@pytest.mark.parametrize("factor", [1, 5])
@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_prefetch_threads_delays_and_resume(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, factor: int, mode: str
) -> None:
    tiny_config.data = tiny_config.data.model_copy(update={"loader_mode": mode})
    tiny_config.training.data_reuse_factor = factor
    expected = collect(cache, tiny_config)
    original = PositionalReader.read_group

    def slow_read(self: PositionalReader, segments: list[Segment]) -> None:
        if segments[0].sample % 3 == 0:
            time.sleep(0.002)
        original(self, segments)

    monkeypatch.setattr(PositionalReader, "read_group", slow_read)
    for workers, depth in ((1, 1), (4, 8), (2, 2)):
        cfg = tiny_config.model_copy(deep=True)
        cfg.data.reader_threads, cfg.data.prefetch_batches = workers, depth
        assert collect(cache, cfg) == expected
    with TokenLoader(cache, tiny_config, 0, 0, 94) as loader:
        loader.warmup()
        loader.next()
        loader.commit(2)
        loader.next()
        state = loader.state_dict()
        assert state["cursor"] == 2
    assert collect(cache, tiny_config, cursor=2)[0] == expected[0][2:]


@pytest.mark.parametrize("factor", [2, 5, 100])
@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_reuse_global_stream_and_stop(cache: TokenCache, tiny_config: Config, factor: int, mode: str) -> None:
    tiny_config.data = tiny_config.data.model_copy(update={"loader_mode": mode})
    tiny_config.training.data_reuse_factor = factor
    expected = collect(cache, tiny_config, stop=42)[0]
    pool_size = (42 + factor - 1) // factor
    assert len(set(expected)) == pool_size
    assert collect(cache, tiny_config, stop=6)[0] == expected[:6]
    for cursor in sorted({pool_size - 1, pool_size, pool_size + 1, 12, 42}):
        assert collect(cache, tiny_config, cursor=cursor, stop=42)[0] == expected[cursor:]
    tiny_config.training.num_accumulation = 1
    for world in (2, 3):
        tiny_config.distributed.gpus_per_node = world
        for rank in range(world):
            assert collect(cache, tiny_config, stop=42, rank=rank)[0] == expected[rank::world]
            first = 12 + (rank - 12) % world
            assert collect(cache, tiny_config, cursor=12, stop=42, rank=rank)[0] == expected[first::world]


@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_short_reads_and_io_errors(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    tiny_config.data = tiny_config.data.model_copy(update={"loader_mode": mode})
    original = os.pread

    def short(fd: int, size: int, offset: int) -> bytes:
        return original(fd, min(3, size), offset)

    monkeypatch.setattr(os, "pread", short)
    collect(cache, tiny_config, stop=8)

    def fail(fd: int, size: int, offset: int) -> bytes:
        raise OSError("injected I/O failure")

    monkeypatch.setattr(os, "pread", fail)
    with TokenLoader(cache, tiny_config, 0, 0, 8) as loader:
        loader.warmup()
        with pytest.raises(OSError, match="shard=.*samples=.*injected"):
            loader.next()
    monkeypatch.setattr(os, "pread", lambda fd, size, offset: b"")
    with TokenLoader(cache, tiny_config, 0, 0, 8) as loader:
        with pytest.raises(OSError, match="EOF"):
            loader.next()


def test_budget_corruption_cancel(cache: TokenCache, tiny_config: Config) -> None:
    cfg = tiny_config.model_copy(deep=True)
    cfg.data.host_memory_mib = 1
    with pytest.raises(ValueError, match="requires approximately"):
        check_memory(cache, cfg)
    loader = TokenLoader(cache, tiny_config, 0, 0, 100000)
    loader.warmup()
    loader.close()
    assert not loader.thread.is_alive()
    assert all(not handles for handles in loader.reader.handles)
    path = cache.paths["train"][0]
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError, match="checksum"):
        cache.verify()
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="truncated"):
        TokenCache(cache.directory)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_pinned_reuse(cache: TokenCache, tiny_config: Config, mode: str) -> None:
    tiny_config.data = tiny_config.data.model_copy(update={"loader_mode": mode})
    with TokenLoader(cache, tiny_config, 0, 0, 80) as loader:
        transfer = DevicePrefetcher(loader, torch.device("cuda:0"))
        observed = []
        try:
            while True:
                (x, y), spec = transfer.next()
                # Retain device tensors across many host-buffer reuses.
                observed.append((x, y, spec.indices.copy()))
                transfer.preload()
        except StopIteration:
            pass
        finally:
            transfer.close()
        torch.cuda.synchronize()
        for x, y, indices in observed:
            for row, sample in enumerate(indices):
                assert x[row].tolist() == ((np.arange(sample * 4, sample * 4 + 4) % 17).tolist())
                assert y[row].tolist() == ((np.arange(sample * 4 + 1, sample * 4 + 5) % 17).tolist())


def test_coalescing_bound(cache: TokenCache, tiny_config: Config) -> None:
    tiny_config.data.loader_mode = "streaming"
    with TokenLoader(cache, tiny_config, 0, 0, 48) as loader:
        loader.warmup()
        while True:
            try:
                loader.next()
            except StopIteration:
                break
        metrics = loader.metrics()
        assert metrics["read_bytes"] <= 2 * metrics["requested_bytes"]
        assert metrics["queue_occupancy_mean"] <= tiny_config.data.prefetch_batches


@pytest.mark.parametrize("factor", [1, 5])
def test_resident_reads_stop_after_background_fill(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, factor: int
) -> None:
    tiny_config.training.data_reuse_factor = factor
    tiny_config.data.loader_mode = "resident"
    tiny_config.data.read_range_bytes = 6  # Force lookahead and shard/range crossings.
    with TokenLoader(cache, tiny_config, 0, 0, 120) as loader:
        loader.warmup()
        assert loader.resident_ready.wait(5)
        before = loader.metrics()
        assert before["loader_mode"] == "resident"
        assert before["preload_read_bytes"] > 0
        assert before["resident_complete"]
        assert before["resident_loaded_blocks"] == before["resident_total_blocks"]
        assert before["resident_token_bytes"] == loader.sampler.blocks * 5 * 2
        assert not loader.resident_tokens.flags.writeable
        assert all(not handles for handles in loader.reader.handles)

        def fail(*_: object) -> bytes:
            pytest.fail("training read the filesystem after the cache finished filling")

        monkeypatch.setattr(os, "pread", fail)
        samples = []
        while True:
            try:
                batch = loader.next()
            except StopIteration:
                break
            samples.extend(batch.spec.indices.tolist())
            for sample, row in zip(batch.spec.indices, batch.tokens, strict=True):
                np.testing.assert_array_equal(row, np.arange(sample * 4, sample * 4 + 5) % 17)
        assert len(samples) == 120
        assert len(set(samples)) == loader.sampler.blocks
        after = loader.metrics()
        assert after["read_bytes"] == before["preload_read_bytes"]
        assert after["training_read_bytes"] == before["training_read_bytes"]
        assert after["training_read_calls"] == before["training_read_calls"]
        assert after["requested_bytes"] == loader.sampler.blocks * 5 * 2
    assert loader.resident_tokens.nbytes == loader.resident_indices.nbytes == 0


@pytest.mark.parametrize("workers", [1, 4])
def test_resident_startup_overlaps_fill(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    tiny_config.data.loader_mode = "resident"
    tiny_config.data.reader_threads = workers
    tiny_config.data.read_range_bytes = 6
    sampler = GlobalSampler.from_config(cache, tiny_config)
    specs = list(sampler.batches(tiny_config, 0, 0, 120))
    next_samples = set(specs[tiny_config.data.prefetch_batches].indices.tolist())
    blocked, release = threading.Event(), threading.Event()
    original = PositionalReader.read_group

    def delayed(self: PositionalReader, segments: list[Segment]) -> None:
        if segments[0].batch is None and not any(s.sample in next_samples for s in segments):
            blocked.set()
            assert release.wait(10), "test did not release background reads"
        original(self, segments)

    monkeypatch.setattr(PositionalReader, "read_group", delayed)
    loader = TokenLoader(cache, tiny_config, 0, 0, 120)
    consumer = ThreadPoolExecutor(1)
    try:
        assert blocked.wait(5)
        consumer.submit(loader.warmup).result(timeout=2)
        assert not loader.resident_ready.is_set()
        assert 0 < loader.resident_loaded_blocks < len(loader.resident_indices)
        # The entire initial queue can be consumed while background I/O is blocked.
        samples = []
        for _ in range(tiny_config.data.prefetch_batches):
            batch = consumer.submit(loader.next).result(timeout=2)
            samples.extend(batch.spec.indices.tolist())
            for sample, row in zip(batch.spec.indices, batch.tokens, strict=True):
                np.testing.assert_array_equal(row, np.arange(sample * 4, sample * 4 + 5) % 17)
        if workers > 1:
            # Queued cache-fill reads needed next must be promoted and use the
            # reserved reader even while unrelated background reads are blocked.
            batch = consumer.submit(loader.next).result(timeout=2)
            samples.extend(batch.spec.indices.tolist())
            np.testing.assert_array_equal(batch.spec.indices, specs[tiny_config.data.prefetch_batches].indices)
            for sample, row in zip(batch.spec.indices, batch.tokens, strict=True):
                np.testing.assert_array_equal(row, np.arange(sample * 4, sample * 4 + 5) % 17)
        release.set()
        while len(samples) < 120:
            batch = consumer.submit(loader.next).result(timeout=2)
            samples.extend(batch.spec.indices.tolist())
            for sample, row in zip(batch.spec.indices, batch.tokens, strict=True):
                np.testing.assert_array_equal(row, np.arange(sample * 4, sample * 4 + 5) % 17)
        expected = GlobalSampler.from_config(cache, tiny_config)
        assert samples == [expected.block(i) for i in range(120)]
        assert loader.resident_ready.is_set()
        # Includes foreground reads, promoted background reads, and split rows.
        assert loader.metrics()["requested_bytes"] == loader.sampler.blocks * 5 * 2
    finally:
        release.set()
        loader.close()
        consumer.shutdown(wait=True)


def test_resident_background_error(cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_config.data.loader_mode = "resident"
    original = PositionalReader.read_group

    def fail_background(self: PositionalReader, segments: list[Segment]) -> None:
        if segments[0].batch is None:
            raise OSError("injected background failure")
        original(self, segments)

    monkeypatch.setattr(PositionalReader, "read_group", fail_background)
    with TokenLoader(cache, tiny_config, 0, 0, 120) as loader:
        loader.warmup()
        for _ in range(tiny_config.data.prefetch_batches):
            loader.next()
        with pytest.raises(OSError, match="injected background failure"):
            loader.next()
    assert not loader.thread.is_alive()
    assert all(not handles for handles in loader.reader.handles)


def test_resident_background_cancel(cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_config.data.loader_mode = "resident"
    blocked = threading.Event()
    original = PositionalReader.read_group

    def delayed(self: PositionalReader, segments: list[Segment]) -> None:
        if segments[0].batch is None:
            blocked.set()
            assert self.cancel.wait(5), "loader did not cancel background work"
            raise ReadCancelled()
        original(self, segments)

    monkeypatch.setattr(PositionalReader, "read_group", delayed)
    loader = TokenLoader(cache, tiny_config, 0, 0, 120)
    try:
        assert blocked.wait(5)
    finally:
        loader.close()
    assert not loader.thread.is_alive()
    assert loader.resident_remaining.size == 0
    assert all(not handles for handles in loader.reader.handles)


def test_continuous_prefetch_releases_and_refills_early(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_config.data.loader_mode = "streaming"
    sampler = GlobalSampler.from_config(cache, tiny_config)
    specs = list(sampler.batches(tiny_config, 0, 0, 12))
    delayed_sample = int(specs[1].indices[0])
    fourth_sample = int(specs[3].indices[0])
    blocked, release, refilled = threading.Event(), threading.Event(), threading.Event()
    original = PositionalReader.read_group

    def delayed(self: PositionalReader, segments: list[Segment]) -> None:
        if any(s.sample == delayed_sample for s in segments):
            blocked.set()
            assert release.wait(10), "test did not release delayed read"
        if any(s.sample == fourth_sample for s in segments):
            refilled.set()
        original(self, segments)

    monkeypatch.setattr(PositionalReader, "read_group", delayed)
    loader = TokenLoader(cache, tiny_config, 0, 0, 12)
    consumer = ThreadPoolExecutor(1)
    try:
        assert blocked.wait(5)
        # No warmup: intentionally consume before the initial queue fills.
        first = consumer.submit(loader.next).result(timeout=2)
        np.testing.assert_array_equal(first.spec.indices, specs[0].indices)
        # Batch 4 must be submitted while batch 2 is still blocked.
        assert refilled.wait(2)
        release.set()
        for expected in specs[1:]:
            actual = consumer.submit(loader.next).result(timeout=2)
            np.testing.assert_array_equal(actual.spec.indices, expected.indices)
        with pytest.raises(StopIteration):
            loader.next()
    finally:
        release.set()
        loader.close()
        consumer.shutdown(wait=True)


def test_auto_residency_budget(tiny_config: Config, tmp_path: Path) -> None:
    cache = make_cache(tmp_path / "large-cache", tiny_config, tokens=2_000_001, shard_tokens=65536)
    tiny_config.training.tokens_per_parameters = 4000.0
    tiny_config.training.epoch_tokens_per_parameters = 2000.0
    tiny_config.data.loader_mode = "streaming"
    streaming = check_memory(cache, tiny_config)
    tiny_config.data.host_memory_mib = (streaming["total"] + 1024**2 - 1) // 1024**2
    tiny_config.data.loader_mode = "auto"
    assert check_memory(cache, tiny_config)["resident_tokens"] == 0
    with TokenLoader(cache, tiny_config, 0, 0, 6) as loader:
        loader.warmup()
        assert loader.mode == "streaming"
        assert loader.resident_tokens.nbytes == 0
    tiny_config.data.loader_mode = "resident"
    with pytest.raises(ValueError, match="resident loader requires approximately"):
        TokenLoader(cache, tiny_config, 0, 0, 6)
    tiny_config.data.host_memory_mib = 2048
    tiny_config.data.loader_mode = "auto"
    resident = check_memory(cache, tiny_config)
    assert resident["resident_tokens"] == 500_000 * 5 * 2
    assert resident["resident_indices"] == 500_000 * 8
    assert resident["resident_read_state"] == 500_000 * 8
    assert resident["total"] <= tiny_config.data.host_memory_mib * 1024**2


@pytest.mark.parametrize("world", [1, 2, 3])
def test_resident_partial_first_pass(tiny_config: Config, tmp_path: Path, world: int) -> None:
    cache = make_cache(tmp_path / "larger-cache", tiny_config, tokens=401)
    tiny_config.distributed.gpus_per_node = world
    tiny_config.training.num_accumulation = 1
    sampler = GlobalSampler.from_config(cache, tiny_config)
    expected = [sampler.block(p) for p in range(42)]
    assert sampler.blocks == 100
    memory = check_memory(cache, tiny_config)
    assert memory["resident_tokens"] == 42 * 5 * 2
    assert memory["resident_indices"] == 42 * 8
    for rank in range(world):
        # Loading a short prefix or resuming must retain the full recipe's working set.
        for cursor, stop in ((0, 6), (0, 42), (12, 42)):
            with TokenLoader(cache, tiny_config, rank, cursor, stop) as loader:
                loader.warmup()
                assert loader.mode == "resident"
                np.testing.assert_array_equal(loader.resident_indices, sorted(expected))
                assert loader.sampler.identity == sampler.identity
                assert loader.resident_tokens.nbytes == memory["resident_tokens"]
            first = cursor + (rank - cursor) % world
            assert collect(cache, tiny_config, cursor=cursor, stop=stop, rank=rank)[0] == expected[first:stop:world]
        # Explicit loader-only ranges beyond the recipe still need complete coverage.
        extended = [sampler.block(p) for p in range(120)]
        assert collect(cache, tiny_config, stop=120, rank=rank)[0] == extended[rank::world]


@pytest.mark.parametrize("factor", [1, 5])
@pytest.mark.parametrize("world", [1, 3])
@pytest.mark.parametrize("cursor,stop", [(0, 2), (0, 12), (6, 12), (24, 120)])
def test_benchmark_resident_horizon_reads_only_during_preload(
    cache: TokenCache,
    tiny_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    factor: int,
    world: int,
    cursor: int,
    stop: int,
) -> None:
    tiny_config.training.data_reuse_factor = factor
    tiny_config.training.num_accumulation = 1
    tiny_config.distributed.gpus_per_node = world
    tiny_config.data.loader_mode = "resident"
    tiny_config.data.read_range_bytes = 6
    sampler = GlobalSampler.from_config(cache, tiny_config)
    expected = [sampler.block(p) for p in range(stop)]
    resident_blocks = min(stop, sampler.blocks) if factor == 1 else sampler.blocks
    original = os.pread
    for rank in range(world):
        monkeypatch.setattr(os, "pread", original)
        with TokenLoader(cache, tiny_config, rank, cursor, stop, bound_resident_to_stop=True) as loader:
            # Run in a future so a premature producer exit cannot hang the test.
            with ThreadPoolExecutor(1) as consumer:
                try:
                    consumer.submit(loader.warmup, resident=True).result(timeout=5)
                except BaseException:
                    loader.close()
                    raise
            before = loader.metrics()
            assert before["resident_complete"]
            assert before["resident_loaded_blocks"] == before["resident_total_blocks"] == resident_blocks
            assert loader.sampler.identity == sampler.identity
            assert loader.memory == check_memory(cache, tiny_config, stop=stop, bound_resident_to_stop=True)

            def fail(*_: object) -> bytes:
                pytest.fail("bounded resident training read after preload")

            monkeypatch.setattr(os, "pread", fail)
            samples = []
            while True:
                try:
                    batch = loader.next()
                except StopIteration:
                    break
                samples.extend(batch.spec.indices.tolist())
                for sample, row in zip(batch.spec.indices, batch.tokens, strict=True):
                    np.testing.assert_array_equal(row, np.arange(sample * 4, sample * 4 + 5) % 17)
            first = cursor + (rank - cursor) % world
            assert samples == expected[first:stop:world]
            after = loader.metrics()
            assert after["read_bytes"] == before["read_bytes"]
            assert after["read_calls"] == before["read_calls"]
            assert after["requested_bytes"] == resident_blocks * 5 * 2


@pytest.mark.parametrize("phase", ["open", "read"])
def test_wait_diagnostics(cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    tiny_config.data.loader_mode = "streaming"
    tiny_config.data.reader_threads = 1
    blocked, release, consuming = threading.Event(), threading.Event(), threading.Event()

    def delay_once() -> None:
        if not blocked.is_set():
            blocked.set()
            assert release.wait(5)

    original_open, original_read = PositionalReader.descriptor, os.pread

    def delayed_open(self: PositionalReader, path: Path) -> int:
        delay_once()
        return original_open(self, path)

    def delayed_read(fd: int, size: int, offset: int) -> bytes:
        delay_once()
        return original_read(fd, size, offset)

    if phase == "open":
        monkeypatch.setattr(PositionalReader, "descriptor", delayed_open)
    else:
        monkeypatch.setattr(os, "pread", delayed_read)
    loader = TokenLoader(cache, tiny_config, 0, 0, 42)
    consumer = ThreadPoolExecutor(1)

    def consume() -> None:
        consuming.set()
        loader.next()

    try:
        assert blocked.wait(2)
        future = consumer.submit(consume)
        assert consuming.wait(2)
        time.sleep(0.05)
        release.set()
        future.result(timeout=2)
        first = loader.metrics()
        assert first["wait_lifetime_max_ms"] >= 20
        event = first["largest_waits"][0]
        assert event["global_start"] == 0 and event["queue_occupancy"] == 0
        assert event["slowest_read"][f"{phase}_ms"] >= 20
        assert first["slowest_read"][f"{phase}_ms"] >= 20
        assert event["slowest_read"]["shard"] in {str(path) for path in cache.paths["train"]}
        # Expiring the rolling window must not erase lifetime outliers.
        loader.waits.clear()
        loader.warmup()
        while True:
            try:
                loader.next()
            except StopIteration:
                break
        final = loader.metrics()
        assert len(final["largest_waits"]) == 8
        assert final["wait_lifetime_max_ms"] >= first["wait_lifetime_max_ms"]
        assert final["wait_lifetime_total_ms"] >= final["wait_lifetime_max_ms"]
        assert any(event["global_start"] == 0 for event in final["largest_waits"])
    finally:
        release.set()
        loader.close()
        consumer.shutdown(wait=True)
