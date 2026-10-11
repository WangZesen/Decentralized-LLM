"""Fixed process groups and reciprocal parameter mixing."""

import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from decentralized_llm.config import Config


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


@torch.no_grad()
def mix_parameters(model: nn.Module, topology: str, step: int) -> None:
    size = world_size()
    if size == 1:
        return
    parameters = list(model.parameters())
    flat = torch.cat([p.detach().reshape(-1) for p in parameters])
    if topology == "complete":
        dist.all_reduce(flat)
        flat.div_(size)
    else:
        peer = peer_for(rank(), size, topology, step)
        received = torch.empty_like(flat)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, flat, peer),
                dist.P2POp(dist.irecv, received, peer),
            ]
        )
        for request in requests:
            request.wait()
        flat.add_(received).mul_(0.5)
    offset = 0
    for parameter in parameters:
        parameter.copy_(flat[offset : offset + parameter.numel()].view_as(parameter))
        offset += parameter.numel()


@torch.no_grad()
def averaged_model(model: nn.Module) -> nn.Module:
    import copy

    averaged = copy.deepcopy(model).float()
    if world_size() > 1:
        mix_parameters(averaged, "complete", 0)
    return averaged
