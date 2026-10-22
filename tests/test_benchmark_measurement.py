from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from decentralized_llm.benchmarks import training
from decentralized_llm.benchmarks.protocol import benchmark_bounds
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache


@pytest.mark.parametrize(
    "mode,topology,cycle,warmup,chunk",
    [
        ("ddp", "complete", 1, 5, 7),
        ("decentralized", "complete", 1, 5, 7),
        ("decentralized", "one_peer_ring", 2, 6, 8),
        ("decentralized", "one_peer_exponential", 3, 6, 9),
    ],
)
def test_bounds_cover_all_rotations(
    tiny_config: Config, mode: str, topology: str, cycle: int, warmup: int, chunk: int
) -> None:
    raw = tiny_config.model_dump()
    raw["distributed"].update(mode=mode, topology=topology, gpus_per_node=8)
    raw["training"]["batch_tokens"] = 96
    raw["benchmark"].update(warmup=5, steps=7, windows=2, max_chunks=3)
    bounds = benchmark_bounds(Config.model_validate(raw))
    assert (bounds.cycle, bounds.warmup, bounds.chunk) == (cycle, warmup, chunk)
    assert bounds.maximum_steps == warmup + chunk * 7
    assert bounds.blocks == 24
    assert bounds.total_blocks == bounds.maximum_steps * 24


def short_benchmark(config: Config) -> None:
    config.benchmark.warmup = 1
    config.benchmark.steps = 1
    config.benchmark.windows = 1
    config.benchmark.max_chunks = 1
    config.benchmark.min_seconds = 0.000001


def test_benchmark_disables_metric_io_and_preserves_token_accounting(
    tiny_config: Config, cache: TokenCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    short_benchmark(tiny_config)
    original = training.create_backend
    updates = []

    def backend(*args: Any, **kwargs: Any):
        instance = original(*args, **kwargs)
        update = instance.update

        def measured(*args: Any, **kwargs: Any):
            updates.append(kwargs["collect_metrics"])
            return update(*args, **kwargs)

        monkeypatch.setattr(instance, "update", measured)
        return instance

    monkeypatch.setattr(training, "create_backend", backend)
    result = training.benchmark_training(tiny_config)
    assert updates == [False, False]
    assert not list(tiny_config.runtime.output_dir.glob("metrics*.jsonl"))
    assert result["measurement"] == {"valid": True, "invalid_reasons": []}
    rank = result["rank_measurements"][0]
    assert rank["timed_read_bytes"] == rank["timed_read_calls"] == 0
    window = result["windows"][0]
    assert window["global_tokens"] == tiny_config.training.batch_tokens
    assert window["global_tokens_per_second"] * window["seconds"] == pytest.approx(window["global_tokens"])
    assert result["bounds"]["total_blocks"] == 18
    assert result["throughput_statistics"]["coefficient_of_variation"] == 0
    assert result["max_peak_allocated_bytes"] is None


def test_compiler_activity_invalidates_measurement(
    tiny_config: Config, cache: TokenCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    short_benchmark(tiny_config)
    snapshots = iter([{}, {}, {"stats": {"unique_graphs": 1}}, {"stats": {"unique_graphs": 1}}])
    monkeypatch.setattr(training, "compiler_counters", lambda: next(snapshots))
    result = training.benchmark_training(tiny_config)
    assert not result["measurement"]["valid"]
    assert result["measurement"]["invalid_reasons"] == ["rank 0: compiler_activity_during_measurement"]
    assert result["rank_measurements"][0]["timed_compiler_counters_delta"] == {"stats": {"unique_graphs": 1}}
    assert (tiny_config.runtime.output_dir / "benchmark.json").is_file()


def test_benchmark_honors_max_chunks(tiny_config: Config, cache: TokenCache, monkeypatch: pytest.MonkeyPatch) -> None:
    short_benchmark(tiny_config)
    tiny_config.benchmark.max_chunks = 3
    monkeypatch.setattr(training, "synchronized_seconds", lambda *args: 0.0)
    with pytest.raises(RuntimeError, match="after 3 chunks"):
        training.benchmark_training(tiny_config)
    assert not (tiny_config.runtime.output_dir / "benchmark.json").exists()


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("resident_reads", [False, True])
def test_gpu_peaks_precede_profiling_and_collectives(
    tiny_config: Config,
    cache: TokenCache,
    monkeypatch: pytest.MonkeyPatch,
    profile: bool,
    resident_reads: bool,
) -> None:
    short_benchmark(tiny_config)
    tiny_config.benchmark.profile = profile
    state = {"updates": 0, "peak": 0, "metrics": 0}

    class Loader:
        def __init__(self, *args: object, **kwargs: object):
            assert kwargs["bound_resident_to_stop"] is True

        def __enter__(self):
            return self

        def __exit__(self, *args: object):
            pass

        def warmup(self, **kwargs: object):
            pass

        def commit(self, cursor: int):
            pass

        def metrics(self):
            state["metrics"] += 1
            return {
                "resident_complete": True,
                "loader_mode": "resident",
                "read_bytes": 50 + int(resident_reads and state["metrics"] > 1),
                "read_calls": 2 + int(resident_reads and state["metrics"] > 1),
            }

    class Transfer:
        def __init__(self, *args: object):
            pass

        def next(self):
            pass

        def preload(self):
            pass

        def close(self):
            pass

    def update(*args: object, **kwargs: object):
        assert kwargs["collect_metrics"] is False
        state["updates"] += 1
        state["peak"] = max(state["peak"], state["updates"] * 100)

    def report():
        state["peak"] = 999000
        return {
            "parameter_bytes": 100,
            "parameter_storage_bytes": 100,
            "optimizer": {"history_bytes": 0, "scratch_bytes": 0},
            "parameter_mixing": {},
        }

    def gather(record: dict):
        state["peak"] = 9000
        remote = deepcopy(record)
        remote.update(rank=1, local_rank=1, gpu_uuid="GPU-second", peak_allocated_bytes=600, peak_reserved_bytes=1200)
        return [record, remote]

    class Profile:
        def __enter__(self):
            state["peak"] = 99000
            return self

        def __exit__(self, *args: object):
            pass

        def export_chrome_trace(self, path: str):
            pass

        def key_averages(self):
            return SimpleNamespace(table=lambda **kwargs: "operators")

    runtime = SimpleNamespace(rank=0, size=2, barrier=lambda: None, gather=gather)
    backend = SimpleNamespace(
        prepare=lambda: None,
        update=update,
        memory_report=report,
        execution=SimpleNamespace(policy={}, cache=None),
    )
    monkeypatch.setattr(training, "runtime_for", lambda config: runtime)
    monkeypatch.setattr(training, "create_backend", lambda *args: backend)
    monkeypatch.setattr(training, "TokenLoader", Loader)
    monkeypatch.setattr(training, "DevicePrefetcher", Transfer)
    monkeypatch.setattr(training, "synchronized_seconds", lambda *args: 1.0)
    monkeypatch.setattr(training, "compiler_counters", lambda: {})
    monkeypatch.setattr(training.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(training.torch.cuda, "reset_peak_memory_stats", lambda device: state.update(peak=10))
    monkeypatch.setattr(training.torch.cuda, "max_memory_allocated", lambda device: state["peak"])
    monkeypatch.setattr(training.torch.cuda, "max_memory_reserved", lambda device: state["peak"] * 2)
    monkeypatch.setattr(training.torch.cuda, "get_device_name", lambda device: "fake GPU")
    monkeypatch.setattr(training.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-first"))
    monkeypatch.setattr(training.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(training, "nccl_version", lambda: (2, 30, 7))
    monkeypatch.setattr(training.torch.profiler, "profile", lambda **kwargs: Profile())
    result = training.benchmark_training(tiny_config, device=torch.device("cuda", 0))
    assert result["peak_allocated_bytes"] == 200
    assert result["peak_reserved_bytes"] == 400
    assert result["max_peak_allocated_bytes"] == 600
    assert result["sum_peak_allocated_bytes"] == 800
    assert result["max_peak_reserved_bytes"] == 1200
    assert result["sum_peak_reserved_bytes"] == 1600
    assert result["gpu_uuid"] == "GPU-first"
    assert result["rank_measurements"][1]["gpu_uuid"] == "GPU-second"
    assert result["measurement"]["valid"] is not resident_reads
    assert state["peak"] == 999000


def test_aggregate_rejects_invalid_nonzero_rank() -> None:
    good = {"rank": 0, "invalid_reasons": [], "peak_allocated_bytes": 12, "peak_reserved_bytes": 30}
    bad = {**good, "rank": 1, "invalid_reasons": ["compiler_activity_during_measurement"]}
    aggregate = training.aggregate_measurements([good, bad])
    assert aggregate["measurement"] == {
        "valid": False,
        "invalid_reasons": ["rank 1: compiler_activity_during_measurement"],
    }
