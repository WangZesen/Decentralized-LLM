"""Real Gloo collectives with independent PyTorch hybrid reference optimizers."""

import os
from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_distributed import free_port, reference_batch, reference_mix
from test_muon import reference_optimizers

from decentralized_llm.algorithms.muon import MuonAdamW
from decentralized_llm.algorithms.schedules import learning_rates
from decentralized_llm.backends.factory import create_backend
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.models.llama import token_losses
from decentralized_llm.runtime import clip_grad_norm_


def hybrid_worker(worker: int, port: int, raw: dict[str, Any]) -> None:
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=worker, world_size=2, timeout=timedelta(seconds=60))
    try:
        for mode, scheme in (("ddp", "awc"), ("decentralized", "awc"), ("decentralized", "atc")):
            for gamma in (0.0, 0.5):
                current = {**raw, "distributed": {**raw["distributed"], "mode": mode, "scheme": scheme}}
                cfg = Config.model_validate(current)
                cfg.distributed.error_feedback_gamma = gamma
                cache = TokenCache(cfg.data.cache_dir)
                backend = create_backend(cfg, torch.device("cpu"))
                backend.prepare()
                references = [initialize_model(cfg) for _ in range(2)]
                opts = [reference_optimizers(model, cfg) for model in references]
                histories = [[torch.zeros_like(p) for p in model.parameters()] for model in references]
                blocks, total = 8, 32
                with TokenLoader(cache, cfg, worker, 0, total) as loader:
                    transfer = DevicePrefetcher(loader, torch.device("cpu"))
                    try:
                        for step in range(4):
                            start, stop = step * blocks, (step + 1) * blocks
                            rates = learning_rates(cfg, stop * 4, total * 4)
                            for rank, (model, (muon, adamw)) in enumerate(zip(references, opts, strict=True)):
                                model.zero_grad(set_to_none=True)
                                for begin in range(start, stop, blocks // cfg.training.num_accumulation):
                                    x, y = reference_batch(cache, cfg, list(range(begin + rank, begin + 4, 2)))
                                    (token_losses(model(x), y).sum() / 16).backward()
                                for group in muon.param_groups:
                                    group["lr"] = rates["muon_lr"]
                                for group in adamw.param_groups:
                                    group["lr"] = rates["adamw_lr"]
                            if mode == "ddp":
                                for left, right in zip(
                                    references[0].parameters(), references[1].parameters(), strict=True
                                ):
                                    assert left.grad is not None and right.grad is not None
                                    mean = (left.grad + right.grad) / 2
                                    left.grad, right.grad = mean.clone(), mean.clone()
                            for model in references:
                                clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
                            if mode == "decentralized" and scheme == "awc":
                                reference_mix(references, cfg.distributed.topology, step)
                            for model, (muon, adamw), history in zip(references, opts, histories, strict=True):
                                before = [p.detach().clone() for p in model.parameters()]
                                muon.step()
                                adamw.step()
                                with torch.no_grad():
                                    for parameter, base, previous in zip(
                                        model.parameters(), before, history, strict=True
                                    ):
                                        delta = parameter - base
                                        if gamma:
                                            parameter.add_(delta, alpha=gamma).sub_(previous, alpha=gamma)
                                        previous.copy_(delta)
                            if mode == "decentralized" and scheme == "atc":
                                reference_mix(references, cfg.distributed.topology, step)
                            metrics = backend.update(transfer.next, transfer.preload, start, stop, step, total)
                            loader.commit(stop)
                            assert metrics["muon_lr"] == rates["muon_lr"]
                            assert metrics["adamw_lr"] == rates["adamw_lr"]
                            optimizer = backend.optimizer
                            assert isinstance(optimizer, MuonAdamW)
                            muon, adamw = opts[worker]
                            for name, actual in backend.model.named_parameters():
                                expected = references[worker].get_parameter(name)
                                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-7)
                                expected_state = (
                                    muon.state[expected] if expected in muon.state else adamw.state[expected]
                                )
                                for key, value in expected_state.items():
                                    torch.testing.assert_close(
                                        optimizer.state[actual][key], value, rtol=3e-5, atol=3e-7
                                    )
                    finally:
                        transfer.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_hybrid_gloo_updates(tiny_config: Config, cache: TokenCache) -> None:
    raw = tiny_config.model_dump(mode="json")
    raw["optimizer"].update(name="muon_adamw", muon_lr_ratio=2.0, lr=0.0003)
    raw["distributed"].update(gpus_per_node=2, topology="one_peer_ring")
    raw["training"].update(batch_tokens=32, num_accumulation=2)
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage] # Public torch API omitted from export stubs.
        hybrid_worker, args=(free_port(), raw), nprocs=2, join=True
    )
