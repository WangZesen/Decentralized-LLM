import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from decentralized_llm.algorithms.adamw import make_optimizer
from decentralized_llm.config import Config, save_config
from decentralized_llm.data.cache import TokenCache, fingerprint
from decentralized_llm.data.loader import TokenLoader
from decentralized_llm.data.sampling import GlobalSampler
from decentralized_llm.models.llama import Llama
from decentralized_llm.training import checkpoints
from decentralized_llm.training.checkpoints import save_checkpoint, validate_checkpoint


def test_manifest_last_failure(
    cache: TokenCache, tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Llama(tiny_config.model)
    opt = make_optimizer(model, tiny_config, torch.device("cpu"))
    state = {"sampler": GlobalSampler(cache, 4, 42).identity, "cursor": 0}

    def fail(path: Path, value: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(checkpoints, "atomic_checkpoint", fail)
    path = tmp_path / "failed-checkpoint"
    with pytest.raises(OSError, match="not published"):
        save_checkpoint(path, tiny_config, cache, model, opt, state, {"cursor": 0, "step": 0})
    assert not (path / "manifest.json").exists()


def test_weights_only_and_world_size(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    model = Llama(tiny_config.model)
    opt = make_optimizer(model, tiny_config, torch.device("cpu"))
    state = {"sampler": GlobalSampler(cache, 4, 42).identity, "cursor": 0}
    path = tmp_path / "weights"
    save_checkpoint(path, tiny_config, cache, model, opt, state, {"cursor": 0, "step": 0}, training_state=False)
    with pytest.raises(ValueError, match="weights-only"):
        validate_checkpoint(path, tiny_config, cache)
    validate_checkpoint(path, tiny_config, cache, require_training=False)
    different = tiny_config.model_copy(deep=True)
    different.distributed.gpus_per_node = 2
    with pytest.raises(ValueError, match="world size"):
        validate_checkpoint(path, different, cache)


def test_cache_identity_and_dtype(cache: TokenCache) -> None:
    file = cache.directory / "manifest.json"
    content = json.loads(file.read_text())
    content["dtype"] = "<u4"
    content["identity"] = fingerprint({k: v for k, v in content.items() if k != "identity"})
    file.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="uint16"):
        TokenCache(cache.directory)


@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_cancel_pending_reads(cache: TokenCache, tiny_config: Config, mode: str) -> None:
    tiny_config.data = tiny_config.data.model_copy(update={"loader_mode": mode})
    loader = TokenLoader(cache, tiny_config, 0, 0, 1000000)
    loader.close()
    assert not loader.thread.is_alive()
    assert loader.sampler.permutation.size == 0
    assert loader.resident_tokens.size == 0
    assert all(not handles for handles in loader.reader.handles)


def test_torchrun_rank_failure_propagates(tmp_path: Path) -> None:
    script = tmp_path / "rank_failure.py"
    script.write_text(
        "import os, time\n"
        "if os.environ['RANK'] == '1':\n"
        "    raise RuntimeError('injected rank failure')\n"
        "time.sleep(60)\n"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--max-restarts=0",
        str(script),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=25,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
    )
    assert result.returncode != 0
    assert "injected rank failure" in result.stderr


@pytest.mark.parametrize("factor", [1, 5])
def test_compilation_wrapper_and_benchmark(
    cache: TokenCache, tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factor: int
) -> None:
    from decentralized_llm.benchmarks.data import benchmark_data

    original = torch.compile

    def eager_compile(model: torch.nn.Module, **_: Any) -> Any:
        return original(model, backend="eager")

    monkeypatch.setattr(torch, "compile", eager_compile)
    tiny_config.runtime.compile = True
    tiny_config.training.data_reuse_factor = factor
    result = benchmark_data(tiny_config, steps=2, warmup=1, cache_state="warm", output=tmp_path / "benchmark.json")
    assert result["measurements"]["network"]["loader"]["requested_bytes"] > 0
    assert result["throughput_ratio"] > 0
    assert (tmp_path / "benchmark.json").is_file()
    if factor == 5:
        # The benchmark loads 18 blocks; the configured training budget is 42.
        assert result["sampling"]["pool_blocks"] == 9
        assert result["measurements"]["network"]["loader"]["sampler_index_bytes"] == 9 * 16


def test_validate_cli_prevents_launch(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    tiny_config.data.host_memory_mib = 1
    save_config(tiny_config, config_path)
    result = subprocess.run(
        [sys.executable, "-m", "decentralized_llm", "train", "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        cwd=tmp_path,
    )
    assert result.returncode != 0 and "requires approximately" in result.stderr
    assert not tiny_config.runtime.output_dir.exists()
