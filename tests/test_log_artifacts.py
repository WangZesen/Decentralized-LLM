"""Concurrent durable logs, compatibility, and resume evidence isolation."""

import fcntl
import json
import multiprocessing
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.launch.preflight import validate
from decentralized_llm.runtime import append_metric, atomic_json
from decentralized_llm.training import loop
from decentralized_llm.training.logs import (
    current_session,
    provenance_paths,
    read_config,
    read_events,
    read_failures,
    read_metadata,
    read_records,
    read_result,
    read_training_metrics,
    read_validation_metrics,
    timestamp,
)

FILES = {".run.lock", "config.yaml", "events.jsonl", "metrics.jsonl", "result.json"}


def start(directory: Path, session: int = 1, config: dict | None = None) -> None:
    directory.mkdir(exist_ok=True)
    append_metric(
        directory / "events.jsonl",
        {"event": "session_start", "session": session, "schema_version": 1, "rank": 0, "config": config or {}},
    )


def write_worker(path: Path, rank: int) -> None:
    for step in range(25):
        append_metric(path, {"rank": rank, "step": step, "payload": "λ" * 20000}, durable=step == 24)


def test_concurrent_large_records_and_interrupted_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"killed":"\xce')  # Partial UTF-8 must not be decoded.
    assert read_records(path) == []
    processes = [multiprocessing.get_context("spawn").Process(target=write_worker, args=(path, i)) for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    rows = read_records(path)
    assert len(rows) == 100
    assert all(row["payload"] == "λ" * 20000 for row in rows)
    for rank in range(4):
        assert [row["step"] for row in rows if row["rank"] == rank] == list(range(25))
    with path.open("ab") as handle:
        handle.write(b'{"other":"partial')
    assert read_records(path) == rows
    append_metric(path, {"recovered": True})
    assert read_records(path) == [*rows, {"recovered": True}]


def test_completed_corruption_is_not_hidden(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{broken}\n{"partial"')
    with pytest.raises(ValueError):
        read_records(path)
    append_metric(path, {"next": True})
    with pytest.raises(ValueError):
        read_records(path)
    assert path.read_bytes().startswith(b"{broken}\n")


@pytest.mark.parametrize("backend,workers", [("distributed", 1), ("packed", 2)])
def test_four_artifacts_and_outcome_only_result(
    cache: TokenCache, tiny_config: Config, backend: str, workers: int
) -> None:
    config = tiny_config.model_dump(mode="json")
    config["runtime"]["training_backend"] = backend
    config["distributed"]["gpus_per_node"] = workers
    config["distributed"]["mode"] = "decentralized"
    cfg = Config.model_validate(config)
    cfg.training.checkpoint_policy = "none"
    loop.train(cfg)
    output = cfg.runtime.output_dir
    assert {path.name for path in output.iterdir()} == FILES
    result = read_result(output)
    assert result["validation"]["split"] == "full" and result["validation"]["validation_complete"]
    assert not {"execution", "compiler_counters"} & result.keys()
    assert read_events(output, "execution") and read_events(output, "compiler")
    assert read_events(output, "loader") and read_events(output, "worker")
    for row in read_records(output / "events.jsonl") + read_records(output / "metrics.jsonl"):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}", row["timestamp"])
    assert all("event" not in row and "split" not in row for row in read_records(output / "metrics.jsonl"))
    if backend == "packed":
        assert all(len(row["local_losses"]) == workers for row in read_training_metrics(output))


def test_repeated_resume_keeps_initial_config_and_constant_file_count(cache: TokenCache, tiny_config: Config) -> None:
    output = tiny_config.runtime.output_dir
    loop.train(tiny_config, stop_after_steps=2)
    initial = (output / "config.yaml").read_bytes()
    tiny_config.data.reader_threads = 1
    loop.train(tiny_config, output / "step-00000002", stop_after_steps=4)
    loop.train(tiny_config, output / "step-00000004")
    assert (output / "config.yaml").read_bytes() == initial
    assert read_config(output).data.reader_threads == 1
    assert {path.name for path in output.iterdir() if path.is_file()} == FILES
    assert read_result(output)["session"] == 3
    assert [row["session"] for row in read_records(output / "events.jsonl") if row["event"] == "session_start"] == [
        1,
        2,
        3,
    ]
    assert all(row["session"] == 3 for row in read_training_metrics(output) + read_validation_metrics(output))


def test_legacy_and_compact_readers_agree(tmp_path: Path, tiny_config: Config) -> None:
    legacy, compact = tmp_path / "legacy", tmp_path / "compact"
    legacy.mkdir()
    start(compact, config=tiny_config.model_dump(mode="json"))
    metadata = {"cache": "cache", "environment": {"source_sha256": "source"}}
    train = {"event": "train", "step": 1, "local_loss": 4.0, "gradient_norm": 2.0}
    validation = {"event": "validation", "split": "subset", "epoch": 1, "loss": 3.0}
    failure = {"rank": 0, "kind": "nonfinite_update", "traceback": "traceback"}
    atomic_json(legacy / "run.json", metadata)
    append_metric(compact / "events.jsonl", {**metadata, "event": "run", "rank": 0, "session": 1})
    append_metric(legacy / "metrics-rank-00000.jsonl", train)
    append_metric(compact / "events.jsonl", {**train, "rank": 0, "session": 1})
    append_metric(legacy / "metrics.jsonl", validation)
    append_metric(compact / "metrics.jsonl", {"epoch": 1, "loss": 3.0, "session": 1})
    atomic_json(legacy / "failure-rank-00000.json", failure)
    append_metric(compact / "events.jsonl", {**failure, "event": "failure", "session": 1})
    for key, value in metadata.items():
        assert read_metadata(compact)[key] == read_metadata(legacy)[key] == value
    for read, record in [(read_training_metrics, train), (read_validation_metrics, validation)]:
        assert {key: read(compact)[0][key] for key in record} == read(legacy)[0]
    assert read_failures(compact)[0]["traceback"] == read_failures(legacy)[0]["traceback"]
    assert provenance_paths(legacy)[1].name == "run.json"
    assert provenance_paths(compact)[1].name == "events.jsonl"


def test_stale_evidence_and_late_worker_events_are_excluded(tmp_path: Path) -> None:
    start(tmp_path)
    atomic_json(tmp_path / "result.json", {"session": 1, "validation": {"loss": 3.0}})
    start(tmp_path, session=2)
    append_metric(tmp_path / "events.jsonl", {"event": "failure", "session": 1, "rank": 0})
    append_metric(tmp_path / "events.jsonl", {"event": "train", "session": 1, "rank": 0})
    assert read_failures(tmp_path) == [] and read_training_metrics(tmp_path) == []
    with pytest.raises(ValueError, match="earlier training session"):
        read_result(tmp_path)
    atomic_json(tmp_path / "run.json", {"cache": "legacy"})
    with pytest.raises(ValueError, match="no run provenance"):
        read_metadata(tmp_path)
    (tmp_path / "events.jsonl").write_bytes(b'{"partial"')
    with pytest.raises(ValueError, match="no complete session_start"):
        read_metadata(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        current_session([{"event": "session_start", "schema_version": 99, "session": 1}])


def test_rejected_launches_do_not_change_existing_logs(cache: TokenCache, tiny_config: Config) -> None:
    loop.train(tiny_config)
    output = tiny_config.runtime.output_dir
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="already contains"):
        loop.train(tiny_config)
    with (output / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError):
            loop.train(tiny_config)
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before


def test_legacy_resume_requires_new_directory(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    loop.train(tiny_config, stop_after_steps=2)
    checkpoint = tiny_config.runtime.output_dir / "step-00000002"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    atomic_json(legacy / "run.json", {"legacy": True})
    tiny_config.runtime.output_dir = legacy
    for run in (loop.train, validate):
        with pytest.raises(ValueError, match="legacy runs must resume into a new output directory"):
            run(tiny_config, checkpoint)
    assert json.loads((legacy / "run.json").read_text()) == {"legacy": True}
    tiny_config.runtime.output_dir = tmp_path / "resumed"
    loop.train(tiny_config, checkpoint)
    assert read_result(tiny_config.runtime.output_dir)["completed_epochs"] == 2


def test_logging_failure_does_not_mask_training_error(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    original = loop.append_metric

    def append(path: Path, value: dict, *, durable: bool = False) -> None:
        if value.get("event") in {"loader", "failure"}:
            raise OSError("logging unavailable")
        original(path, value, durable=durable)

    monkeypatch.setattr(loop, "append_metric", append)
    monkeypatch.setattr(
        "decentralized_llm.backends.distributed.update.optimizer_update",
        Mock(side_effect=RuntimeError("original failure")),
    )
    with pytest.raises(RuntimeError, match="original failure"):
        loop.train(tiny_config)
    stderr = capsys.readouterr().err
    assert "Could not record loader cleanup: logging unavailable" in stderr
    assert "Could not record training failure: logging unavailable" in stderr
    assert len(timestamp()) == 23
