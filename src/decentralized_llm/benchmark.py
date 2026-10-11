"""Compare the complete update pipeline with host-resident data after warmup."""

import copy
import os
import resource
import time
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config
from decentralized_llm.distributed import barrier, rank, setup, world_size
from decentralized_llm.loader import DevicePrefetcher, LoadedBatch, TokenLoader
from decentralized_llm.model import Llama
from decentralized_llm.optim import make_optimizer
from decentralized_llm.runtime import atomic_json, environment, preserve_rng
from decentralized_llm.sampling import BatchSpec, check_memory
from decentralized_llm.train import optimizer_update


class MemorySource:
    def __init__(self, config: Config, batches: list[LoadedBatch], updates: int):
        self.config, self.batches, self.updates = config, batches, updates
        self.index = 0

    def next(self) -> LoadedBatch:
        update, micro = divmod(self.index, len(self.batches))
        if update >= self.updates:
            raise StopIteration
        batch = self.batches[micro]
        offset = update * (self.config.training.batch_tokens // self.config.model.context_length)
        self.index += 1
        return LoadedBatch(
            BatchSpec(batch.spec.start + offset, batch.spec.stop + offset, batch.spec.indices), batch.tokens
        )


def advise_cold(cache: TokenCache) -> None:
    """Best-effort client cache eviction, not a claim to flush a shared storage server."""
    for path in cache.paths["train"]:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def benchmark_data(config: Config, steps: int, warmup: int, cache_state: str, output: Path | None = None) -> dict:
    if steps < 1 or warmup < 1 or cache_state not in ("cold", "warm"):
        raise ValueError("benchmark requires positive steps/warmup and cache-state cold or warm")
    device = setup(config)
    try:
        cache = TokenCache(config.data.cache_dir)
        cache.validate_config(config)
        blocks = config.training.batch_tokens // config.model.context_length
        if blocks % world_size():
            raise ValueError("pipeline benchmark requires global sequence batch divisible by world size")
        memory = check_memory(cache, config)
        resident_bytes = (blocks // world_size()) * (config.model.context_length + 1) * 2
        if memory["total"] + resident_bytes > config.data.host_memory_mib * 1024**2:
            raise ValueError("host memory budget cannot also hold the benchmark resident batch")
        total = (steps + warmup) * blocks
        initial = Llama(config.model, config.runtime.attention_backend).to(device)
        measurements: dict = {}
        resident: list[LoadedBatch] = []
        with TokenLoader(cache, config, rank(), 0, blocks) as resident_loader:
            resident_loader.warmup()
            try:
                while True:
                    resident.append(resident_loader.next())
            except StopIteration:
                pass
        for source in ("memory", "network"):
            if source == "network":
                if cache_state == "cold":
                    advise_cold(cache)
                else:
                    # Prime exactly the measured stream; no unverified claim that pages were already resident.
                    with TokenLoader(cache, config, rank(), 0, total) as primer:
                        try:
                            while True:
                                primer.next()
                        except StopIteration:
                            pass
            with preserve_rng():
                model = copy.deepcopy(initial)
                optimizer = make_optimizer(model, config, device)
                compiled = torch.compile(model, mode=config.runtime.compile_mode) if config.runtime.compile else model
                # torch.compile preserves Module behavior; its public overload returns a Callable.
                forward = cast(nn.Module, compiled)
                if config.distributed.mode == "ddp" and world_size() > 1:
                    forward = DDP(
                        compiled, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False
                    )
                context = (
                    TokenLoader(cache, config, rank(), 0, total)
                    if source == "network"
                    else nullcontext(MemorySource(config, resident, steps + warmup))
                )
                with context as loader:
                    if isinstance(loader, TokenLoader):
                        loader.warmup()
                    transfer = DevicePrefetcher(loader, device)
                    try:
                        transfer.preload()
                        started = 0.0
                        for step in range(warmup + steps):
                            if step == warmup:
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                barrier()
                                if isinstance(loader, TokenLoader):
                                    loader.waits.clear()
                                    loader.occupancies.clear()
                                    loader.starved = loader.batches = 0
                                started = time.perf_counter()
                            optimizer_update(
                                model,
                                forward,
                                optimizer,
                                config,
                                device,
                                transfer.next,
                                transfer.preload,
                                step * blocks,
                                (step + 1) * blocks,
                                step,
                                total,
                            )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        elapsed = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
                        if dist.is_initialized():
                            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                        measurements[source] = {
                            "seconds": float(elapsed),
                            "global_tokens_per_second": steps * config.training.batch_tokens / float(elapsed),
                            "loader": loader.metrics() if isinstance(loader, TokenLoader) else None,
                        }
                    finally:
                        transfer.close()
        ratio = measurements["network"]["global_tokens_per_second"] / measurements["memory"]["global_tokens_per_second"]
        result = {
            "rank": rank(),
            "world_size": world_size(),
            "cache_state_requested": cache_state,
            "cold_method": "POSIX_FADV_DONTNEED (best effort, client only)" if cache_state == "cold" else None,
            "warmup_updates": warmup,
            "measured_updates": steps,
            "measurements": measurements,
            "throughput_ratio": ratio,
            "target_ratio": 0.95,
            "meets_target": ratio >= 0.95,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "environment": environment(),
            "cache_identity": cache.manifest["identity"],
            "config": config.model_dump(mode="json"),
        }
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            target = output if world_size() == 1 else output.with_stem(f"{output.stem}-rank-{rank()}")
            atomic_json(target, result)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
