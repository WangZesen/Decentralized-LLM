"""Benchmark launch estimates describe the same bounded loader used by workers."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from decentralized_llm.cli import main
from decentralized_llm.config import BenchmarkConfig, Config, save_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import TokenLoader
from decentralized_llm.launch.preflight import validate
from tests.conftest import make_cache


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "2"])
def test_max_chunks_requires_positive_integer(value: Any) -> None:
    with pytest.raises(ValidationError, match="max_chunks"):
        BenchmarkConfig(max_chunks=value)
    assert BenchmarkConfig().max_chunks == 8


@pytest.mark.parametrize("max_chunks", [1, 10])
def test_preflight_matches_bounded_loader(tiny_config: Config, tmp_path: Path, max_chunks: int) -> None:
    cache = make_cache(tiny_config.data.cache_dir, tiny_config, tokens=801)
    tiny_config.benchmark = BenchmarkConfig(warmup=1, steps=1, windows=1, max_chunks=max_chunks)
    tiny_config.data.loader_mode = "resident"
    ordinary = validate(tiny_config, verify=False)
    report = validate(tiny_config, verify=False, benchmark=True)
    horizon = (max_chunks + 2) * 6
    assert report["benchmark_bounds"]["total_blocks"] == horizon
    assert report["memory_bytes_per_rank"]["resident_tokens"] == horizon * 5 * 2
    assert ordinary["memory_bytes_per_rank"]["resident_tokens"] == 42 * 5 * 2
    with TokenLoader(cache, tiny_config, 0, 0, horizon, bound_resident_to_stop=True) as loader:
        loader.warmup(resident=True)
        assert loader.metrics()["memory_estimate_bytes"] == report["memory_bytes_per_rank"]
    assert not tiny_config.runtime.output_dir.exists()


def test_benchmark_cannot_resume_or_evaluate(tiny_config: Config, cache: TokenCache) -> None:
    with pytest.raises(ValueError, match="cannot evaluate or resume"):
        validate(tiny_config, Path("checkpoint"), benchmark=True, verify=False)
    with pytest.raises(ValueError, match="cannot evaluate or resume"):
        validate(tiny_config, benchmark=True, evaluation=True, verify=False)


def test_cli_max_chunks_override(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    filename = tmp_path / "config.yaml"
    save_config(tiny_config, filename)
    calls = []

    def fake_launch(config: Config, *_: Any, **kwargs: Any) -> dict:
        calls.append((config, kwargs))
        return {"max_chunks": config.benchmark.max_chunks}

    monkeypatch.setattr("decentralized_llm.launch.commands.launch", fake_launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decentralized-llm",
            "benchmark-training",
            "--no-local-config",
            "-c",
            str(filename),
            "--max-chunks",
            "137",
            "--dry-run",
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out) == {"max_chunks": 137}
    assert calls[0][1]["action"] == "benchmark-training"
    assert calls[0][1]["dry_run"]
