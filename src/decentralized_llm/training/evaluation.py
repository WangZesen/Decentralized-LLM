"""Partition validation without padding or duplicate samples; full scans are sequential."""

import math
import time
from collections.abc import Iterator
from typing import cast

import numpy as np
import torch
from torch import nn

from decentralized_llm.backends.api import TrainingBackend
from decentralized_llm.backends.factory import runtime_for
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.execution.native import build_execution
from decentralized_llm.models.llama import Llama
from decentralized_llm.runtime import preserve_rng
from decentralized_llm.state import Batch, EvaluationResult, IndexArray


def validation_indices(cache: TokenCache, config: Config, full: bool) -> IndexArray:
    count = cache.blocks("validation", config.model.context_length, partial=True)
    if full:
        # This helper is for callers/tests; full evaluation iterates ranges instead.
        return np.arange(count, dtype=np.int64)
    take = min(config.evaluation.subset_blocks, count)
    return np.sort(np.random.default_rng(config.evaluation.seed).choice(count, take, replace=False))


def validation_batches(cache: TokenCache, config: Config, device: torch.device, full: bool) -> Iterator[Batch]:
    runtime = runtime_for(config)
    length = config.model.context_length
    count = cache.blocks("validation", length, partial=True)
    # Contiguous disjoint rank partitions preserve sequential full-split reads.
    first, end = count * runtime.rank // runtime.size, count * (runtime.rank + 1) // runtime.size
    indices = (
        range(first, end) if full else [int(i) for i in validation_indices(cache, config, False) if first <= i < end]
    )
    for offset in range(0, len(indices), config.evaluation.batch_size):
        selected = indices[offset : offset + config.evaluation.batch_size]
        x = np.zeros((len(selected), length), dtype=np.int64)
        y = np.full_like(x, -100)
        tokens = np.empty(0, dtype="<u2")
        if full:
            # One contiguous read per batch, even across shard boundaries.
            start = selected[0] * length
            tokens = cache.read(
                "validation", start, min((selected[-1] + 1) * length + 1, cache.offsets["validation"][-1])
            )
        for row, block in enumerate(selected):
            if full:
                values = tokens[row * length : row * length + length + 1]
            else:
                start = block * length
                values = cache.read("validation", start, min(start + length + 1, cache.offsets["validation"][-1]))
            valid = len(values) - 1
            x[row, :valid], y[row, :valid] = values[:-1], values[1:]
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def parameter_distance(model: nn.Module, reference: nn.Module) -> torch.Tensor:
    """L2 distance over unique parameters, excluding buffers and storage padding."""
    squared_distance = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for parameter, averaged in zip(model.parameters(), reference.parameters(), strict=True):
        # One parameter-sized scratch tensor; copy even if the source is already FP64.
        difference = parameter.to(dtype=torch.float64, copy=True)
        difference.sub_(averaged).square_()
        squared_distance.add_(difference.sum())
    return squared_distance.sqrt()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    cache: TokenCache,
    config: Config,
    device: torch.device,
    full: bool,
    average: bool = True,
    *,
    backend: TrainingBackend | None = None,
) -> EvaluationResult:
    start = time.perf_counter()
    was_training = model.training
    runtime = runtime_for(config)
    size = runtime.size
    decentralized = config.distributed.mode == "decentralized"
    consensus_available = size == 1 or not decentralized or average
    with preserve_rng():
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        if backend is not None:
            backend.inspect()
            eval_model, totals[2] = backend.evaluation_model(average)
        else:
            # Standalone canonical-model callers use the same distributed averaging helper.
            from decentralized_llm.backends.distributed.runtime import averaged_model

            eval_model = averaged_model(model) if average and decentralized else model
            if size > 1 and decentralized and average:
                totals[2] = parameter_distance(model, eval_model)
        eval_model.eval()
        try:
            execution = build_execution(cast(Llama, eval_model), config, device, distributed=False)
            for x, y in prefetched_validation_batches(cache, config, device, full):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.runtime.amp):
                    loss = execution(x, y)
                totals[0] += loss.double()
                totals[1] += (y != -100).sum()
            runtime.reduce(totals)
        finally:
            model.train(was_training)
    loss_sum, valid, distance_sum = totals.tolist()
    if valid == 0:
        raise ValueError("validation has no targets")
    loss = loss_sum / valid
    return EvaluationResult(
        loss=loss,
        perplexity=math.exp(loss) if loss < 700 else None,
        consensus_error=distance_sum / size if consensus_available else None,
        tokens=int(valid),
        seconds=time.perf_counter() - start,
        split="full" if full else "subset",
        validation_complete=cache.manifest["validation_complete"],
    )


def prefetched_validation_batches(
    cache: TokenCache, config: Config, device: torch.device, full: bool
) -> Iterator[Batch]:
    if device.type != "cuda":
        yield from validation_batches(cache, config, device, full)
        return
    stream = torch.cuda.Stream(device=device)
    source = iter(validation_batches(cache, config, torch.device("cpu"), full))

    def preload() -> Batch | None:
        host = next(source, None)
        if host is None:
            return None
        with torch.cuda.stream(stream):
            x, y = (tensor.pin_memory().to(device, non_blocking=True) for tensor in host)
        return x, y

    pending = preload()
    try:
        while pending is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
            batch = pending
            for tensor in batch:
                tensor.record_stream(torch.cuda.current_stream(device))
            yield batch
            pending = preload()
    finally:
        stream.synchronize()
