"""Fixed process groups and reciprocal parameter mixing."""

import os
import random
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from decentralized_llm.config import Config

_peer_group_world: dist.ProcessGroup | None = None
_peer_groups: dict[tuple[int, int], dist.ProcessGroup | None] = {}
_prepared_topologies: set[str] = set()


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
class PendingParameterMix:
    """Own communication buffers without modifying parameters used by autograd."""

    parameters: list[nn.Parameter]
    flat: torch.Tensor
    requests: list[dist.Work]
    size: int

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
        chunks = self.flat.split([p.numel() for p in self.parameters])
        sources = [chunk.view_as(p) for chunk, p in zip(chunks, self.parameters, strict=True)]
        # One batched copy avoids exposing a separate CPU/CUDA launch per weight
        # tensor once communication has already finished behind computation.
        torch._foreach_copy_(tuple(self.parameters), sources)  # pyright: ignore[reportPrivateImportUsage]


@torch.no_grad()
def begin_parameter_mix(model: nn.Module, topology: str, step: int) -> PendingParameterMix | None:
    """Exchange a snapshot asynchronously; the live weights remain unchanged."""
    size = world_size()
    if size == 1:
        return None
    group = dist.group.WORLD
    if topology != "complete":
        # setup() prepares these before training; direct callers may prepare lazily.
        prepare_parameter_groups(topology)
        worker = rank()
        peer = peer_for(worker, size, topology, step)
        group = _peer_groups[(min(worker, peer), max(worker, peer))]
        assert group is not None
        size = 2
    parameters = list(model.parameters())
    flat = torch.cat([p.detach().reshape(-1) for p in parameters])
    requests = [dist.all_reduce(flat, group=group, async_op=True)]
    return PendingParameterMix(parameters, flat, requests, size)


def mix_parameters(model: nn.Module, topology: str, step: int) -> None:
    pending = begin_parameter_mix(model, topology, step)
    if pending is not None:
        pending.apply()


@torch.no_grad()
def averaged_model(model: nn.Module) -> nn.Module:
    import copy

    averaged = copy.deepcopy(model).float()
    if world_size() > 1:
        mix_parameters(averaged, "complete", 0)
    return averaged
