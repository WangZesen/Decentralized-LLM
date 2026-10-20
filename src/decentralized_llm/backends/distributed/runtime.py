"""Fixed process groups and reciprocal parameter mixing."""

import os
import weakref
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from decentralized_llm.algorithms.topology import peer_for
from decentralized_llm.config import Config
from decentralized_llm.execution.device import setup_device
from decentralized_llm.models.storage import ParameterStorage, parameter_storage, prepare_parameter_storage

_peer_group_world: dist.ProcessGroup | None = None
_peer_groups: dict[tuple[int, int], dist.ProcessGroup | None] = {}
_prepared_topologies: set[str] = set()
_mix_buffers: weakref.WeakKeyDictionary[nn.Module, MixBuffer] = weakref.WeakKeyDictionary()
_snapshot_bytes: weakref.WeakKeyDictionary[nn.Module, int] = weakref.WeakKeyDictionary()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def setup(config: Config) -> torch.device:
    expected = config.distributed.world_size
    actual = int(os.environ.get("WORLD_SIZE", "1"))
    if expected != actual:
        raise ValueError(f"configured world_size={expected}, launched WORLD_SIZE={actual}")
    device = setup_device(config)
    if "RANK" in os.environ:
        dist.init_process_group(
            backend=config.distributed.backend,
            timeout=timedelta(seconds=config.distributed.timeout_seconds),
            device_id=device if device.type == "cuda" else None,
        )
    if config.distributed.mode == "decentralized":
        prepare_parameter_groups(
            config.distributed.topology, timeout=timedelta(seconds=config.distributed.timeout_seconds)
        )
    return device


def prepare_parameter_groups(topology: str, *, timeout: timedelta | None = None) -> None:
    """Collectively create all rotating peer groups once, in the same order on every rank."""
    global _peer_group_world
    world = dist.group.WORLD
    if world is not _peer_group_world:
        # A new training run in this process must not reuse destroyed communicators.
        _peer_groups.clear()
        _prepared_topologies.clear()
        _peer_group_world = world
    size = world_size()
    if size == 1 or topology == "complete" or topology in _prepared_topologies:
        return
    peer_for(0, size, topology, 0)  # Validate before entering any group creation collectives.
    rounds = 2 if topology == "one_peer_ring" else (size - 1).bit_length()
    pairs = set()
    for step in range(rounds):
        for worker in range(size):
            peer = peer_for(worker, size, topology, step)
            if worker < peer:
                pairs.add((worker, peer))
    for pair in sorted(pairs):
        if pair not in _peer_groups:
            # Even nonmembers must enter new_group, in a globally consistent order.
            # At world size two the existing world group is already the desired pair.
            group = world if size == 2 else dist.new_group(ranks=list(pair), timeout=timeout)
            _peer_groups[pair] = group if isinstance(group, dist.ProcessGroup) else None
    _prepared_topologies.add(topology)


@dataclass
class MixBuffer:
    storage: ParameterStorage
    flat: torch.Tensor
    busy: bool = False


@dataclass
class PendingParameterMix:
    """Own communication buffers without modifying parameters used by autograd."""

    storage: ParameterStorage
    flat: torch.Tensor
    requests: list[dist.Work]
    size: int
    owner: MixBuffer | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        self.storage.active_mixes += 1

    def wait(self) -> None:
        """Join communication on the caller's stream, including on update failure."""
        for request in self.requests:
            request.wait()
        self.requests.clear()

    @torch.no_grad()
    def apply(self) -> None:
        """Install mixed weights only after all local backwards have finished."""
        self.wait()
        self.flat.div_(self.size)
        self.storage.copy_from(self.flat)

    def close(self) -> None:
        if self.closed:
            return
        self.wait()
        if self.owner is not None:
            self.owner.busy = False
            self.owner = None
        self.storage.active_mixes -= 1
        self.closed = True


@dataclass(frozen=True)
class ParameterMixGroup:
    group: dist.ProcessGroup | None
    size: int


def parameter_mix_group(topology: str, step: int) -> ParameterMixGroup:
    """Resolve the mixing divisor and communicator without touching weights."""
    size = world_size()
    if size == 1:
        return ParameterMixGroup(None, 1)
    group = dist.group.WORLD
    if topology != "complete":
        prepare_parameter_groups(topology)
        worker = rank()
        peer = peer_for(worker, size, topology, step)
        group = _peer_groups[(min(worker, peer), max(worker, peer))]
        assert group is not None
        size = 2
    return ParameterMixGroup(group, size)


@torch.no_grad()
def sum_updated_parameters(model: nn.Module, mix: ParameterMixGroup) -> None:
    """Sum pre-normalized ATC weights in place, after all backwards have finished."""
    _snapshot_bytes[model] = 0
    if mix.size == 1:
        return
    storage = parameter_storage(model)
    if storage.active_mixes:
        raise RuntimeError("cannot mix live weights during an active exchange")
    pending = PendingParameterMix(storage, storage.flat, [], 1)
    try:
        pending.requests.append(dist.all_reduce(storage.flat, group=mix.group, async_op=True))
    finally:
        pending.close()
        torch.autograd.graph.increment_version(storage.parameters)


def parameter_mix_report(model: nn.Module) -> dict[str, int]:
    owner = _mix_buffers.get(model)
    return {
        "snapshot_bytes": _snapshot_bytes.get(model, 0),
        "retained_snapshot_bytes": owner.flat.nbytes if owner is not None else 0,
    }


@torch.no_grad()
def begin_parameter_mix(
    model: nn.Module, topology: str, step: int, *, reuse: bool = False
) -> PendingParameterMix | None:
    """Exchange a snapshot asynchronously; the live weights remain unchanged."""
    mix = parameter_mix_group(topology, step)
    if mix.size == 1:
        _snapshot_bytes[model] = 0
        return None
    group, size = mix.group, mix.size
    storage = parameter_storage(model)
    owner = None
    if reuse:
        owner = _mix_buffers.get(model)
        if owner is None or owner.storage is not storage:
            owner = MixBuffer(storage, torch.empty_like(storage.flat))
            _mix_buffers[model] = owner
        if owner.busy:
            raise RuntimeError("previous parameter exchange still owns the reusable snapshot")
        owner.busy = True
        flat = owner.flat
    else:
        flat = torch.empty_like(storage.flat)
    try:
        flat.copy_(storage.flat)
        requests = [dist.all_reduce(flat, group=group, async_op=True)]
    except BaseException:
        if owner is not None:
            owner.busy = False
        raise
    _snapshot_bytes[model] = flat.nbytes
    return PendingParameterMix(storage, flat, requests, size, owner)


def mix_parameters(model: nn.Module, topology: str, step: int, *, reuse: bool = False) -> None:
    pending = begin_parameter_mix(model, topology, step, reuse=reuse)
    if pending is not None:
        try:
            pending.apply()
        finally:
            pending.close()


@torch.no_grad()
def averaged_model(model: nn.Module) -> nn.Module:
    import copy

    averaged = copy.deepcopy(model).float()
    prepare_parameter_storage(averaged)
    if world_size() > 1:
        mix_parameters(averaged, "complete", 0)
    return averaged


class DistributedRuntime:
    @property
    def rank(self) -> int:
        return rank()

    @property
    def size(self) -> int:
        return world_size()

    def setup(self, config: Config) -> torch.device:
        return setup(config)

    def barrier(self) -> None:
        barrier()

    def reduce(self, value: torch.Tensor, *, maximum: bool = False) -> None:
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)

    def gather(self, value: Any) -> list[Any]:
        values = [None] * world_size()
        if dist.is_initialized():
            dist.all_gather_object(values, value)
        else:
            values[0] = value
        return values

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
