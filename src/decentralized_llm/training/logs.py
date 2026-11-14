"""Compact training artifacts and read-only compatibility with legacy runs.

All compact timestamps are UTC, formatted to milliseconds without an offset.
The events inode is never replaced: readers and writers lock that same file.
"""

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from decentralized_llm.config import Config

SCHEMA_VERSION = 1
type Record = dict[str, Any]


def timestamp() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="milliseconds")


def read_records(path: Path, *, compact: bool = True) -> list[Record]:
    """Snapshot complete records, tolerating only an unfinished compact tail."""
    with path.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        data = handle.read()
    return decode_records(data, compact=compact)


def decode_records(data: bytes, *, compact: bool = True) -> list[Record]:
    """Decode a snapshot; also used by audit readers hashing the original bytes."""
    if compact:
        data = data[: data.rfind(b"\n") + 1]
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("invalid log record")
    return records


def current_session(rows: list[Record]) -> Record:
    starts = [row for row in rows if row.get("event") == "session_start"]
    if not starts:
        raise ValueError("compact run has no complete session_start record")
    session = starts[-1]
    if session.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training log schema version")
    if type(session.get("session")) is not int or session["session"] < 1:
        raise ValueError("invalid training session identifier")
    return session


def session_start(directory: Path) -> Record:
    return current_session(read_records(directory / "events.jsonl"))


def read_events(directory: Path, event: str | None = None, *, rank: int | None = None) -> list[Record]:
    rows = read_records(directory / "events.jsonl")
    session = current_session(rows)["session"]
    return [
        row
        for row in rows
        if row.get("session") == session
        and (event is None or row.get("event") == event)
        and (rank is None or row.get("rank") == rank)
    ]


def read_config_data(directory: Path) -> Record:
    if (directory / "events.jsonl").exists():
        return session_start(directory)["config"]
    return yaml.safe_load((directory / "config.yaml").read_text())


def read_config(directory: Path) -> Config:
    return Config.model_validate(read_config_data(directory))


def read_metadata(directory: Path) -> Record:
    if (directory / "events.jsonl").exists():
        rows = read_events(directory, "run")
        if not rows:
            raise ValueError("current session has no run provenance")
        return rows[-1]
    return json.loads((directory / "run.json").read_text())


def read_result(directory: Path) -> Record:
    result = json.loads((directory / "result.json").read_text())
    if (directory / "events.jsonl").exists() and result.get("session") != session_start(directory)["session"]:
        raise ValueError("result belongs to an earlier training session")
    return result


def read_training_metrics(directory: Path, rank: int = 0) -> list[Record]:
    if (directory / "events.jsonl").exists():
        return read_events(directory, "train", rank=rank)
    return [
        row
        for row in read_records(directory / f"metrics-rank-{rank:05d}.jsonl", compact=False)
        if row.get("event") == "train"
    ]


def read_validation_metrics(directory: Path) -> list[Record]:
    compact = (directory / "events.jsonl").exists()
    session = session_start(directory)["session"] if compact else None
    rows = read_records(directory / "metrics.jsonl", compact=compact)
    if compact:
        # Restore stream semantics in memory for consumers of either layout.
        return [{**row, "event": "validation", "split": "subset"} for row in rows if row.get("session") == session]
    return [row for row in rows if row.get("event") == "validation" and row.get("split", "subset") == "subset"]


def read_failures(directory: Path) -> list[Record]:
    if (directory / "events.jsonl").exists():
        return [{**row, "file": str(directory / "events.jsonl")} for row in read_events(directory, "failure")]
    records = []
    for path in sorted(directory.glob("failure-rank-*.json")):
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError(f"invalid worker failure record: {path}")
        records.append({**record, "file": str(path)})
    return records


def provenance_paths(directory: Path) -> list[Path]:
    metadata = "events.jsonl" if (directory / "events.jsonl").exists() else "run.json"
    return [directory / name for name in ("config.yaml", metadata, "result.json")]


def has_run(directory: Path) -> bool:
    return any((directory / name).exists() for name in ("run.json", "events.jsonl", "result.json"))


def validate_destination(directory: Path, resume: Path | None) -> None:
    if resume is None and has_run(directory):
        raise ValueError("output already contains a training run; use resume or a new output directory")
    if resume is not None and (directory / "run.json").exists():
        raise ValueError("legacy runs must resume into a new output directory; set runtime.output_dir")
