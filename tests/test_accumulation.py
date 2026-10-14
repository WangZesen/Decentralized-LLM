from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from decentralized_llm.cache import TokenCache
from decentralized_llm.checkpoints import recipe_identity
from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.execution import build_execution
from decentralized_llm.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.model import Llama
from decentralized_llm.preflight import validate
from decentralized_llm.sampling import GlobalSampler, memory_estimate


@pytest.mark.parametrize("workers,batch", [(1, 32), (4, 8), (8, 4)])
def test_default_accumulation(workers: int, batch: int) -> None:
    config = load_config(overrides=[f"distributed.gpus_per_node={workers}"])
    assert config.training.num_accumulation == 1
    assert config.micro_batch_size == batch
    assert "micro_batch_size" not in config.model_dump_json()
    assert "micro_batch_size" not in Config.model_json_schema()["properties"]
    with pytest.raises(AttributeError, match="micro_batch_size"):
        config.micro_batch_size = 2  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("value", ["0", "-1", "true", "false", "2.0", "1.5", '"2"', "null"])
def test_invalid_accumulation(value: str) -> None:
    with pytest.raises(ValidationError, match="num_accumulation"):
        load_config(overrides=[f"training.num_accumulation={value}"])


@pytest.mark.parametrize(
    "overrides",
    [
        ["training.batch_tokens=32769"],
        ["distributed.gpus_per_node=3"],
        ["training.num_accumulation=3"],
        ["training.batch_tokens=1024", "distributed.gpus_per_node=2"],
        ["training.batch_tokens=4096", "distributed.gpus_per_node=4", "training.num_accumulation=2"],
    ],
)
def test_batch_divisibility(overrides: list[str]) -> None:
    with pytest.raises(ValidationError, match=r"divisible by context_length \* workers \* num_accumulation"):
        load_config(overrides=overrides)


def test_accumulation_layers_and_roundtrip(tmp_path: Path) -> None:
    base, workers, saved = (tmp_path / name for name in ("base.yaml", "workers.yaml", "saved.yaml"))
    base.write_text("training: {batch_tokens: 32768, num_accumulation: 2}\n")
    workers.write_text("distributed: {gpus_per_node: 4}\n")
    config = load_config([base, workers])
    assert config.micro_batch_size == 4
    save_config(config, saved)
    raw = yaml.safe_load(saved.read_text())
    assert raw["training"]["num_accumulation"] == 2
    assert "micro_batch_size" not in raw and "micro_batch_size" not in raw["training"]
    assert load_config(saved) == config
    changed = load_config(saved, ["distributed.gpus_per_node=8", "training.num_accumulation=4"])
    assert changed.micro_batch_size == 1


def test_micro_batch_size_tracks_current_settings() -> None:
    config = load_config(overrides=["launcher.kind=slurm", "launcher.partition=gpu"])
    config.distributed.gpus_per_node = 4
    assert config.micro_batch_size == 8
    config.training.num_accumulation = 2
    assert config.micro_batch_size == 4
    config.model.context_length = 512
    assert config.micro_batch_size == 8
    config.training.batch_tokens = 65536
    assert config.micro_batch_size == 16
    config.distributed.nodes = 2
    assert config.micro_batch_size == 8
    with pytest.raises(ValidationError, match="num_accumulation"):
        config.training.num_accumulation = True
    config.training.num_accumulation = 3
    with pytest.raises(ValueError, match="divisible"):
        _ = config.micro_batch_size


@pytest.mark.parametrize("explicit_accumulation", [False, True])
def test_old_micro_batch_size_rejected(tmp_path: Path, explicit_accumulation: bool) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("training: {micro_batch_size: 8}\n")
    overrides = ["training.num_accumulation=1"] if explicit_accumulation else []
    for paths, changes in ((config, overrides), (None, [*overrides, "training.micro_batch_size=8"])):
        with pytest.raises(ValidationError, match=r"training.micro_batch_size.*training.num_accumulation"):
            load_config(paths, changes)


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_sampler_accumulation_and_truncated_ranges(
    cache: TokenCache, tiny_config: Config, workers: int, accumulation: int
) -> None:
    raw = tiny_config.model_dump()
    raw["training"].update(batch_tokens=64, num_accumulation=accumulation)
    raw["distributed"]["gpus_per_node"] = workers
    config = Config.model_validate(raw)
    sampler = GlobalSampler.from_config(cache, config)
    expected = [sampler.block(p) for p in range(32)]
    for rank in range(workers):
        batches = list(sampler.batches(config, rank, 0, 32))
        assert len(batches) == 2 * accumulation
        assert all(len(batch.indices) == config.micro_batch_size for batch in batches)
        assert [index for batch in batches for index in batch.indices] == expected[rank::workers]
        for start, stop in ((0, 19), (3, 19), (16, 17)):
            partial = list(sampler.batches(config, rank, start, stop))
            assert partial[0].start == start and partial[-1].stop == stop
            assert all(left.stop == right.start for left, right in zip(partial, partial[1:], strict=False))
            first = start + (rank - start) % workers
            assert [index for batch in partial for index in batch.indices] == expected[first:stop:workers]


@pytest.mark.parametrize("accumulation", [1, 3])
@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_inferred_buffers_shapes_and_preflight(
    cache: TokenCache, tiny_config: Config, accumulation: int, mode: str
) -> None:
    raw = tiny_config.model_dump()
    raw["training"]["num_accumulation"] = accumulation
    raw["data"]["loader_mode"] = mode
    config = Config.model_validate(raw)
    batch = 6 // accumulation
    memory = memory_estimate(cache, config)
    assert memory["transfer_buffers"] == 2 * batch * 4 * 2 * 8
    report = validate(config)
    assert report["num_accumulation"] == accumulation and report["micro_batch_size"] == batch
    execution = build_execution(Llama(config.model), config, torch.device("cpu"))
    assert execution.shape == (batch, 4)
    with TokenLoader(cache, config, 0, 0, 6) as loader:
        loader.warmup()
        transfer = DevicePrefetcher(loader, torch.device("cpu"))
        try:
            assert all(buffer.shape == (batch, 4) for pair in transfer.buffers for buffer in pair)
            for _ in range(accumulation):
                (x, y), spec = transfer.next()
                assert x.shape == y.shape == (batch, 4)
                assert spec.stop - spec.start == batch
            with pytest.raises(StopIteration):
                transfer.next()
        finally:
            transfer.close()


@pytest.mark.parametrize(
    "workers,accumulation,identity",
    [
        (1, 1, "2fa738112f4c97ee26a13b700e9401bf67f540c47e1c215d32d8fb4a3a3c490f"),
        (4, 1, "d2d3239d3dfd7c7f6a7a54d597754647a717cee8d4bf54c47b42d53de0f9219b"),
        (4, 2, "a209d274419b54337641bbbc931181f93733050881687e7c6250f1c63e554049"),
    ],
)
def test_historical_batching_recipe_identity(cache: TokenCache, workers: int, accumulation: int, identity: str) -> None:
    # Captured before migration with per-rank micro_batch_size 32, 8, and 4 respectively.
    config = load_config(
        overrides=[
            f"distributed.gpus_per_node={workers}",
            f"training.num_accumulation={accumulation}",
            "runtime.fused_rope=false",
        ]
    )
    assert recipe_identity(config, cache) == identity
