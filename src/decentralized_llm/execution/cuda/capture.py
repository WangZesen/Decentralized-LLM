"""Complete packed updates, adapted from tiny-llm's captured GH200 engine."""

import time
from collections.abc import Callable

import torch

from decentralized_llm.config import Config
from decentralized_llm.data.sampling import NextBatch
from decentralized_llm.execution.native import TrainingExecution
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.runtime import preserve_rng

from .packed_optimizer import ArenaAdamW


class CapturedUpdate:
    def __init__(
        self,
        model: PackedLlama,
        optimizer: ArenaAdamW,
        execution: TrainingExecution,
        config: Config,
        device: torch.device,
    ):
        self.model, self.optimizer, self.execution = model, optimizer, execution
        self.config, self.device = config, device
        self.blocks = config.training.batch_tokens // config.model.context_length
        self.inputs = torch.zeros((self.blocks, config.model.context_length), dtype=torch.int64, device=device)
        self.targets = torch.zeros_like(self.inputs)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.preparation_seconds = 0.0

    def _update(self) -> None:
        self.optimizer.zero_grad()
        workers = self.model.num_models
        micro = self.config.process_micro_batch_size
        means = torch.zeros(workers, device=self.device)
        denominator = self.config.training.batch_tokens // workers
        for offset in range(0, self.blocks, micro):
            x, y = (
                tensor[offset : offset + micro].view(-1, workers, tensor.shape[-1]).transpose(0, 1).contiguous()
                for tensor in (self.inputs, self.targets)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = self.execution(x, y) / denominator
            losses.sum().backward()
            means.add_(losses.detach())
        self.optimizer.step(means)

    def _persistent(self) -> dict[str, torch.Tensor]:
        optimizer = self.optimizer
        state = {
            "parameters": self.model.parameter_storage,
            "first_moment": optimizer.first_moment_storage,
            "second_moment": optimizer.second_moment_storage,
            "lr": optimizer.lr,
            "gamma": optimizer.gamma,
            "finite": optimizer.finite,
            "completed": optimizer.completed,
            "clips": optimizer.clip_counts,
            "loss_sum": optimizer.loss_sum,
            "losses": optimizer.losses,
            "norms": optimizer.norms,
        }
        if optimizer.previous_update is not None:
            state["previous_update"] = optimizer.previous_update
        return state

    def prepare(self) -> None:
        if self.graph is not None:
            return
        started = time.perf_counter()
        saved = {name: tensor.detach().cpu().clone() for name, tensor in self._persistent().items()}
        with preserve_rng():
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            try:
                with torch.cuda.stream(stream):
                    self.optimizer.lr.zero_()
                    for _ in range(3):
                        self._update()
                    self.optimizer.inspect()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self._update()
                    self.optimizer.finalize_capture()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                self.graph = graph
            finally:
                stream.synchronize()
                for name, tensor in self._persistent().items():
                    tensor.copy_(saved[name])
                torch.cuda.synchronize(self.device)
        self.preparation_seconds = time.perf_counter() - started

    def execute(
        self,
        next_batch: NextBatch,
        preload: Callable[[], None],
        start: int,
        stop: int,
        lr: float,
    ) -> None:
        if self.graph is None:
            raise RuntimeError("prepare the captured update before execution")
        if stop - start != self.blocks:
            raise ValueError("captured updates require one complete global batch")
        cursor = start
        while cursor < stop:
            (x, y), spec = next_batch()
            if spec.start != cursor or spec.stop > stop:
                raise ValueError("loader/optimizer global cursor mismatch")
            offset, end = cursor - start, spec.stop - start
            self.inputs[offset:end].copy_(x)
            self.targets[offset:end].copy_(y)
            cursor = spec.stop
            preload()
        self.optimizer.lr.fill_(lr)
        self.graph.replay()
