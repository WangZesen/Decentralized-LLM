"""Atomic artifacts, RNG preservation, and bounded provenance."""

import fcntl
import hashlib
import json
import os
import platform
import random
import socket
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from decentralized_llm.state import RNGState


class NonfiniteUpdateError(FloatingPointError):
    """Training diverged numerically; the rejected update must not be committed."""


def clip_grad_norm_(parameters: Iterable[Tensor], maximum: float | None) -> Tensor:
    """Return the gradient norm and reject nonfinite values; null disables scaling."""
    parameters = list(parameters)
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
    if not torch.isfinite(norm):
        raise NonfiniteUpdateError("nonfinite gradient norm")
    if maximum is not None:
        torch.nn.utils.clip_grads_with_norm_(parameters, maximum, norm)
    return norm


def rng_state() -> RNGState:
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(legacy=True),
        torch=torch.get_rng_state(),
        cuda=[torch.cuda.get_rng_state()] if torch.cuda.is_initialized() else None,
    )


def restore_rng(state: RNGState) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        # Each rank owns one GPU; do not initialize contexts on other visible GPUs.
        torch.cuda.set_rng_state(state["cuda"][0].cpu())


@contextmanager
def preserve_rng() -> Iterator[None]:
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_metric(path: Path, value: object, *, durable: bool = False) -> None:
    """Append one complete record across processes, repairing an interrupted tail."""
    encoded = (json.dumps(value, allow_nan=False) + "\n").encode()
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        # A killed writer can leave an unterminated record. Never turn it into
        # corrupt interior data by appending the next record to that fragment.
        end = handle.seek(0, os.SEEK_END)
        while end:
            start = max(0, end - 8192)
            handle.seek(start)
            block = handle.read(end - start)
            newline = block.rfind(b"\n")
            if newline >= 0:
                end = start + newline + 1
                break
            end = start
        handle.truncate(end)
        handle.write(encoded)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())


def source_digest(package: Path, *, include_metadata: bool = True) -> str:
    """Hash all Python sources, including subpackages, with stable relative paths."""
    digest = hashlib.sha256()
    paths = list(package.rglob("*.py"))
    if include_metadata:
        paths.extend(package.rglob("*.json"))
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_digest() -> str | None:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    return hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.is_file() else None


def environment() -> dict:
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": source_digest(Path(__file__).parent),
        "lock_sha256": dependency_digest(),
        "python_version": list(sys.version_info[:2]),
        "upstream": "dd03e5037b46961b1bbcbd014c445b6efebb4841",
    }
