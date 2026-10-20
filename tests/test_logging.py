import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from decentralized_llm.backends.distributed import update as updates
from decentralized_llm.config import Config, load_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.training import loop as training


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    clock = SimpleNamespace(now=0.0, updates=0)
    monkeypatch.setattr(training, "time", SimpleNamespace(perf_counter=lambda: clock.now, time_ns=time.time_ns))
    update = updates.optimizer_update

    def timed_update(*args: Any, **kwargs: Any) -> dict[str, float]:
        result = update(*args, **kwargs)
        clock.now += 100 if clock.updates == 0 else 2
        clock.updates += 1
        return result

    monkeypatch.setattr(updates, "optimizer_update", timed_update)
    for name in ("evaluate", "save_checkpoint", "append_metric"):
        original = getattr(training, name)

        def delayed(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            clock.now += 1000
            return _original(*args, **kwargs)

        monkeypatch.setattr(training, name, delayed)
    return clock


def records(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "metrics-rank-00000.jsonl").read_text().splitlines()]


def test_compact_logs_measure_recent_training(
    cache: TokenCache,
    tiny_config: Config,
    clock: SimpleNamespace,
) -> None:
    assert Config().training.log_every == 50
    assert load_config(overrides=["training.log_every=3"]).training.log_every == 3
    tiny_config.training.log_every = 4
    tiny_config.training.checkpoint_policy = "interval"
    training.train(tiny_config)
    logged = records(tiny_config.runtime.output_dir)
    # The first window spans validation/checkpointing; the final interval has only three updates.
    assert [row["step"] for row in logged] == [4, 7]
    assert [row["cursor"] for row in logged] == [24, 42]
    assert logged[0]["global_tokens_per_second"] == pytest.approx(96 / 106)
    assert logged[1]["global_tokens_per_second"] == pytest.approx(72 / 6)
    assert all(
        set(row)
        == {
            "event",
            "step",
            "cursor",
            "lr",
            "gradient_norm",
            "local_loss",
            "global_tokens_per_second",
        }
        for row in logged
    )
    diagnostics = json.loads((tiny_config.runtime.output_dir / "loader-rank-00000.json").read_text())
    assert "largest_waits" in diagnostics and "memory_estimate_bytes" in diagnostics
    assert clock.updates == 7
    validation = [
        json.loads(line) for line in (tiny_config.runtime.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(validation) == 2
    assert all(row["event"] == "validation" and row["consensus_error"] == 0.0 for row in validation)
    result = json.loads((tiny_config.runtime.output_dir / "result.json").read_text())
    assert result["validation"]["consensus_error"] == 0.0
    checkpoint_result = training.evaluate_checkpoint(tiny_config, tiny_config.runtime.output_dir / "final")
    assert checkpoint_result["consensus_error"] == 0.0


def test_logging_resets_on_resume(
    cache: TokenCache,
    tiny_config: Config,
    tmp_path: Path,
    clock: SimpleNamespace,
) -> None:
    tiny_config.training.log_every = 4
    training.train(tiny_config, stop_after_steps=2)
    assert [row["step"] for row in records(tiny_config.runtime.output_dir)] == [2]
    checkpoint = tiny_config.runtime.output_dir / "step-00000002"
    clock.now += 10000
    tiny_config.runtime.output_dir = tmp_path / "resumed"
    training.train(tiny_config, checkpoint)
    logged = records(tiny_config.runtime.output_dir)
    assert [row["step"] for row in logged] == [4, 7]
    assert [row["global_tokens_per_second"] for row in logged] == pytest.approx([12, 12])


def test_short_run_logs_once_and_keeps_failure_diagnostics(
    cache: TokenCache,
    tiny_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny_config.training.log_every = 50
    training.train(tiny_config)
    assert [row["step"] for row in records(tiny_config.runtime.output_dir)] == [7]
    tiny_config.runtime.output_dir = tmp_path / "failed"
    monkeypatch.setattr(updates, "optimizer_update", Mock(side_effect=RuntimeError("injected failure")))
    with pytest.raises(RuntimeError, match="injected failure"):
        training.train(tiny_config)
    assert (tiny_config.runtime.output_dir / "loader-rank-00000.json").is_file()


def test_throughput_uses_slowest_rank_and_completed_cuda_work(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(training, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    window = training.ThroughputWindow(10, torch.device("cpu"))
    window.start()
    now[0] = 2

    def slowest(elapsed: torch.Tensor, op: Any) -> None:
        assert op == torch.distributed.ReduceOp.MAX
        assert float(elapsed) == 2
        elapsed.fill_(5)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", slowest)
    assert window.measure(20, 4) == 8
    assert window.seconds == 0 and window.started is None

    def synchronize(device: torch.device) -> None:
        assert device.type == "cuda"
        now[0] += 3

    sync = Mock(side_effect=synchronize)
    monkeypatch.setattr(training.torch.cuda, "synchronize", sync)
    window = training.ThroughputWindow(0, torch.device("cuda"))
    window.start()
    now[0] += 2
    sync.assert_not_called()
    window.pause()
    assert window.seconds == 5
    window.pause()
    sync.assert_called_once()
