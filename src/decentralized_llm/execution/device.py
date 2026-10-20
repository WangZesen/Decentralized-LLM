"""Physical device placement and process RNG setup; no communication."""

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from decentralized_llm.config import Config


def setup_device(config: Config) -> torch.device:
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if config.runtime.device == "cuda":
        if config.runtime.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        visible = torch.cuda.device_count()
        if visible < config.processes_per_node or local >= visible:
            raise ValueError(f"need {config.processes_per_node} visible GPUs per node, found {visible}")
        if config.launcher.kind == "slurm":
            if int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != config.physical_nodes:
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
    seed = config.runtime.seed
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
