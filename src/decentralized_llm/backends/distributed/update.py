"""One distributed optimizer update, shared by training and benchmarks."""

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from decentralized_llm.algorithms.schedules import learning_rate
from decentralized_llm.backends.distributed.runtime import (
    begin_parameter_mix,
    mix_parameters,
    parameter_mix_group,
    rank,
    sum_updated_parameters,
    world_size,
)
from decentralized_llm.config import Config
from decentralized_llm.data.sampling import BatchSpec, rank_count
from decentralized_llm.execution.cuda.adamw import FusedDecentralizedAdamW
from decentralized_llm.execution.native import TrainingExecution
from decentralized_llm.models.llama import token_losses
from decentralized_llm.runtime import clip_grad_norm_
from decentralized_llm.state import Batch


def optimizer_update(
    model: nn.Module,
    forward: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    next_batch: Callable[[], tuple[Batch, BatchSpec]],
    preload: Callable[[], None],
    start: int,
    stop: int,
    step: int,
    total: int,
    *,
    collect_metrics: bool = True,
) -> dict[str, float]:
    length = config.model.context_length
    local_count = rank_count(start, stop, rank(), world_size())
    denominator = (stop - start) / world_size() if config.distributed.mode == "ddp" else local_count
    lr = learning_rate(config, stop * length, total * length)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    cursor = start
    loss_sum = torch.zeros((), device=device)
    pending = (
        begin_parameter_mix(model, config.distributed.topology, step, reuse=config.distributed.reuse_mix_buffers)
        if config.distributed.mode == "decentralized" and config.distributed.scheme == "awc"
        else None
    )
    try:
        while cursor < stop:
            (x, y), spec = next_batch()
            if spec.start != cursor or spec.stop > stop:
                raise ValueError("loader/optimizer global cursor mismatch")
            final_microbatch = spec.stop == stop
            sync = (
                forward.no_sync()
                if isinstance(forward, (DDP, TrainingExecution)) and not final_microbatch
                else nullcontext()
            )
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.runtime.amp):
                    losses = (
                        forward(x, y) if isinstance(forward, TrainingExecution) else token_losses(forward(x), y).sum()
                    )
                    loss = losses / (denominator * length)
                loss.backward()
            loss_sum += losses.detach()
            cursor = spec.stop
            # Compute is already queued on CUDA; enqueue the upcoming transfer alongside it.
            preload()
        norm = clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
        if isinstance(optimizer, FusedDecentralizedAdamW):
            if pending is not None:
                pending.wait()
                optimizer.step_fused(base=pending.flat, base_divisor=pending.size)
            elif config.distributed.scheme == "atc":
                mix = parameter_mix_group(config.distributed.topology, step)
                optimizer.step_fused(output_divisor=mix.size)
                sum_updated_parameters(model, mix)
            else:
                optimizer.step()
        else:
            if pending is not None:
                pending.apply()
            optimizer.step()
            if config.distributed.mode == "decentralized" and config.distributed.scheme == "atc":
                mix_parameters(model, config.distributed.topology, step, reuse=config.distributed.reuse_mix_buffers)
        if not collect_metrics:
            return {}
        return {"lr": lr, "gradient_norm": float(norm), "local_loss": float(loss_sum) / (local_count * length)}
    finally:
        if pending is not None:
            # A failed forward, backward or loader call must not leave a live exchange.
            pending.close()
