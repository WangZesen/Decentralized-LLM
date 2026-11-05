"""Stable, study-local CUDA device pools and persisted trial reservations."""

import os
import re
import socket
from collections.abc import Sequence

import torch

from decentralized_llm.tuning.spec import Record

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_DEVICE_ID = re.compile(rf"(?:GPU-{_UUID}|MIG-{_UUID}|MIG-GPU-{_UUID}/\d+/\d+)")


def _gpu_ids(value: object, description: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or _DEVICE_ID.fullmatch(item) is None for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{description} must contain unique, complete GPU or MIG UUIDs")
    return value


def _pool_ids(pool: Record) -> list[str]:
    if not isinstance(pool.get("host"), str) or not pool["host"]:
        raise ValueError("local GPU pool must identify its host")
    return _gpu_ids(pool.get("gpu_ids"), "local GPU pool")


def discover_gpus(indices: list[int] | None = None) -> Record:
    """Resolve controller-visible CUDA ordinals to identities safe for child masks."""
    count = torch.cuda.device_count()
    if count == 0:
        raise ValueError("local GPU tuning requires at least one visible CUDA device; check CUDA_VISIBLE_DEVICES")
    selected = list(range(count)) if indices is None else indices
    if (
        not selected
        or any(type(index) is not int or not 0 <= index < count for index in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError(f"local_gpus must contain unique visible CUDA indices in [0, {count - 1}]")
    identifiers = []
    selectors = {item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")}
    for index in selected:
        identifier = str(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
        # PyTorch's _CUuuid string is the UUID payload, whereas CUDA masks use GPU-UUID.
        if re.fullmatch(_UUID, identifier):
            gpu, mig = f"GPU-{identifier}", f"MIG-{identifier}"
            if mig in selectors:
                identifier = mig
            elif any(item.startswith("MIG-") for item in selectors) and gpu not in selectors:
                raise ValueError(
                    "cannot resolve bare MIG identity; use complete matching GPU/MIG UUIDs in CUDA_VISIBLE_DEVICES"
                )
            else:
                identifier = gpu
        if _DEVICE_ID.fullmatch(identifier) is None:
            raise ValueError(f"cannot resolve a complete GPU UUID for visible CUDA device {index}")
        identifiers.append(identifier)
    return {"host": socket.gethostname(), "gpu_ids": _gpu_ids(identifiers, "visible CUDA devices")}


def restore_pool(saved: Record, visible: Record) -> Record:
    """Keep a saved UUID pool when visible CUDA ordinals change across restarts."""
    saved_ids, visible_ids = _pool_ids(saved), _pool_ids(visible)
    if saved["host"] != visible["host"]:
        raise ValueError(
            f"local GPU tuning belongs to host {saved['host']}; resume it on that host, not {visible['host']}"
        )
    missing = set(saved_ids) - set(visible_ids)
    if missing:
        raise ValueError(
            "saved local GPU pool is not visible; restore CUDA_VISIBLE_DEVICES to include " + ", ".join(sorted(missing))
        )
    return {**saved, "gpu_ids": list(saved_ids)}


def available_gpus(pool: Record, active_attempts: Sequence[Record]) -> list[str]:
    """Reserve live assignments, including work retained from an earlier revision."""
    pool_ids = _pool_ids(pool)
    reserved: set[str] = set()
    unknown_assignment = False
    for attempt in active_attempts:
        if attempt.get("host", pool["host"]) != pool["host"]:
            raise ValueError("active local attempt belongs to a different host than the local GPU pool")
        assigned = attempt.get("local_gpu_ids")
        if assigned is None or assigned == []:
            # Legacy GPU attempts and CPU trials retain the previous sequential behavior.
            unknown_assignment = True
            continue
        identifiers = _gpu_ids(assigned, "active local GPU assignment")
        if reserved.intersection(identifiers):
            raise ValueError("active local GPU assignments overlap; reconcile the existing attempts before dispatch")
        reserved.update(identifiers)
    return [] if unknown_assignment else [identifier for identifier in pool_ids if identifier not in reserved]
