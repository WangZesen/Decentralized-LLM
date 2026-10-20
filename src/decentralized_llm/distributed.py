"""Fixed process groups and reciprocal parameter mixing."""

import os
import random
import subprocess
import weakref
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from decentralized_llm.config import Config
from decentralized_llm.parameter_storage import ParameterStorage, parameter_storage, prepare_parameter_storage

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
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if config.runtime.device == "cuda":
        if config.runtime.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        visible = torch.cuda.device_count()
        if visible < config.distributed.gpus_per_node or local >= visible:
            raise ValueError(f"need {config.distributed.gpus_per_node} visible GPUs per node, found {visible}")
        if config.launcher.kind == "slurm":
            if int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != config.distributed.nodes:
                raise ValueError("SLURM node allocation does not match configuration")
            if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
                raise ValueError("SLURM GPU allocation is missing")
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
        if config.runtime.cpu_affinity == "local":
            bind_local_cpus(local)
        if config.runtime.amp and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 autocast requires BF16 GPU support")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(config.runtime.cpu_threads)
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
    seed = config.runtime.seed + (rank() if config.distributed.mode == "decentralized" else 0)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(config.runtime.deterministic)
    return device


def bind_local_cpus(local: int) -> None:
    """Intersect GPU-local CPUs with the scheduler cpuset; never escape the allocation."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    identifier = visible[local] if visible != [""] and local < len(visible) else str(local)
    pci = (
        subprocess.check_output(
            ["nvidia-smi", f"--id={identifier}", "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True
        )
        .strip()
        .lower()
    )
    domain, bus, function = pci.split(":")
    pci = f"{int(domain, 16):04x}:{bus}:{function}"
    ranges = (Path("/sys/bus/pci/devices") / pci / "local_cpulist").read_text().strip()
    cpus = set()
    for part in ranges.split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    allowed = cpus & os.sched_getaffinity(0)
    if not allowed:
        raise ValueError("allocation has no CPUs local to this GPU")
    os.sched_setaffinity(0, allowed)


def peer_for(worker: int, size: int, topology: str, step: int) -> int:
    if size == 1:
        return worker
    if topology == "one_peer_ring":
        if size % 2:
            raise ValueError("ring requires even world size")
        return (worker + (1 if (worker + step % 2) % 2 == 0 else -1)) % size
    if topology == "one_peer_exponential":
        if size & (size - 1):
            raise ValueError("exponential requires power-of-two world size")
        return worker ^ (1 << (step % (size - 1).bit_length()))
    raise ValueError(f"unknown reciprocal topology: {topology}")


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
