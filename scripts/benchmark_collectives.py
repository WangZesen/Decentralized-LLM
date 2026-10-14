"""Bounded all-reduce probe; run with torchrun inside the requested GPU allocation."""

import argparse
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.nccl import version as nccl_version

from decentralized_llm.runtime import atomic_json, environment


def cxi_tx_counters() -> dict[str, int | None]:
    """Host-wide counters; only local rank zero samples each node."""
    if int(os.environ["LOCAL_RANK"]) != 0:
        return {}
    result = {}
    for path in Path("/sys/class/cxi").glob("cxi*/device/telemetry/hni_sts_tx_octets_opt"):
        try:
            result[path.parents[2].name] = int(path.read_text().partition("@")[0])
        except OSError, ValueError:
            result[path.parents[2].name] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != args.world_size:
        raise ValueError("unexpected allocation size")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        measurements = []
        cxi_before = cxi_tx_counters()
        for mib in (4, 25, 80, 360, 2000):
            value = torch.zeros(mib * 1024**2 // 4, device=device, dtype=torch.float32)
            for _ in range(5):
                dist.all_reduce(value)
            dist.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            iterations, elapsed = 0, 0.0
            while elapsed < 0.5 and iterations < 200:
                for _ in range(10):
                    dist.all_reduce(value)
                iterations += 10
                torch.cuda.synchronize(device)
                seconds = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
                dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
                elapsed = float(seconds)
            value.fill_(rank + 1)
            dist.all_reduce(value)
            if not bool((value == size * (size + 1) // 2).all()):
                raise RuntimeError("all-reduce correctness failure")
            bandwidth = value.numel() * value.element_size() * iterations / elapsed / 1e9
            measurements.append(
                {
                    "mib": mib,
                    "iterations": iterations,
                    "seconds": elapsed,
                    "mean_ms": elapsed * 1000 / iterations,
                    "algorithm_GB_per_second": bandwidth,
                    "bus_GB_per_second": bandwidth * 2 * (size - 1) / size,
                }
            )
            del value
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(
                    name in line for name in ("libnccl", "libfabric", "libcxi", "libhwloc", "libstdc++", "libgdrapi")
                )
            }
        )
        result = {
            "rank": rank,
            "world_size": size,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(device),
            "nccl": nccl_version(),
            "network_environment": {
                key: os.environ[key]
                for key in (
                    "NCCL_NET",
                    "NCCL_NET_PLUGIN",
                    "NCCL_SOCKET_IFNAME",
                    "NCCL_NET_GDR_LEVEL",
                    "NCCL_CROSS_NIC",
                    "NCCL_LAUNCH_ORDER_IMPLICIT",
                    "FI_PROVIDER",
                    "FI_MR_CACHE_MONITOR",
                    "OFI_NCCL_PROTOCOL",
                    "FI_HMEM_CUDA_USE_DMABUF",
                    "FI_CXI_ENABLE_WRITEDATA",
                )
                if key in os.environ
            },
            "loaded_libraries": libraries,
            "cxi_tx_bytes_before": cxi_before,
            "cxi_tx_bytes_after": cxi_tx_counters(),
            "measurements": measurements,
            "environment": environment(),
        }
        atomic_json(args.output / f"collectives-rank-{rank:05d}.json", result)
        if rank == 0:
            print(result, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
