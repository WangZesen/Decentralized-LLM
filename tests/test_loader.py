import os
import time

import numpy as np
import pytest
import torch

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config
from decentralized_llm.loader import DevicePrefetcher, PositionalReader, Segment, TokenLoader
from decentralized_llm.sampling import GlobalSampler, check_memory


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
    cfg = Config.model_validate(raw)
    streams = [collect(cache, cfg, stop=2 * count, rank=r)[0] for r in range(4)]
    for rank, stream in enumerate(streams):
        assert stream == (one + two)[rank::4]


def test_prefetch_threads_delays_and_resume(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_short_reads_and_io_errors(cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
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
def test_pinned_reuse(cache: TokenCache, tiny_config: Config) -> None:
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
