"""Atomic artifacts, RNG preservation, and bounded provenance."""

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


def clip_grad_norm_(parameters: Iterable[Tensor], maximum: float | None) -> Tensor:
    """Return the gradient norm and reject nonfinite values; null disables scaling."""
    if maximum is None:
        return torch.nn.utils.get_total_norm(
            [p.grad for p in parameters if p.grad is not None], error_if_nonfinite=True
        )
    return torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True)


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


def append_metric(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def source_digest(package: Path) -> str:
    """Hash all Python sources, including subpackages, with stable relative paths."""
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        "upstream": "dd03e5037b46961b1bbcbd014c445b6efebb4841",
    }
