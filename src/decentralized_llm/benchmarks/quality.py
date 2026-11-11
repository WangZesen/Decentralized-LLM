"""Portable checkpoint loading and candidate-only, FP32 likelihood scoring."""

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache, fingerprint
from decentralized_llm.execution.precision import precision_context
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.training.checkpoints import validate_checkpoint

TokenPair = tuple[list[int], list[int]]


def load_tokenizer(cache: TokenCache):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cache.directory / "tokenizer", local_files_only=True, use_fast=True)
    identity = fingerprint(
        {
            "vocab": tokenizer.get_vocab(),
            "eos": tokenizer.eos_token_id,
            "backend": tokenizer.backend_tokenizer.to_str(),
        }
    )
    if identity != cache.manifest["tokenizer_fingerprint"]:
        raise ValueError("saved tokenizer fingerprint mismatch")
    if len(tokenizer) != cache.manifest["vocab_size"] or tokenizer.eos_token_id != cache.manifest["eos_token_id"]:
        raise ValueError("saved tokenizer vocabulary/EOS mismatch")
    return tokenizer


@torch.no_grad()
def load_quality_model(config: Config, checkpoint: Path, device: str = "cpu") -> tuple[Llama, dict[str, Any]]:
    """Load trusted local checkpoints without an optimizer or a process group."""
    cache = TokenCache(config.data.cache_dir)
    cache.validate_config(config)
    manifest = validate_checkpoint(checkpoint, config, cache, require_training=False)
    model = Llama(config.model).float()

    def payload(filename: str, rank: int) -> dict:
        value = torch.load(checkpoint / filename, map_location="cpu", weights_only=False)
        if value["rank"] != rank or value["loader"] != {
            "sampler": manifest["sampler"],
            "cursor": manifest["cursor"],
        }:
            raise ValueError("checkpoint rank/cursor mismatch")
        return value

    if config.runtime.training_backend == "packed":
        workers = config.distributed.world_size if config.distributed.mode == "decentralized" else 1
        packed = PackedLlama(config.model, workers, canonical=model)
        packed.load_packed_state_dict(payload("packed.pt", 0)["model"])
        packed.copy_average_to(model)
    else:
        workers = config.distributed.world_size if config.distributed.mode == "decentralized" else 1
        total = {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
        # Validate every rank payload, including DDP ranks that carry only RNG/data state.
        for rank in range(config.distributed.world_size):
            value = payload(f"rank-{rank:05d}.pt", rank)
            if rank < workers:
                model.load_state_dict(value["model"], strict=True)
                for name, weight in model.state_dict().items():
                    total[name].add_(weight)
        model.load_state_dict({name: weight / workers for name, weight in total.items()}, strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint has nonfinite model parameters")
    return model.to(device).eval(), manifest


def validate_pair(pair: TokenPair, max_length: int) -> int:
    context, continuation = pair
    if not context or not continuation:
        raise ValueError("likelihood requires nonempty context tokens and candidate tokens")
    if len(continuation) > max_length:
        raise ValueError(f"candidate has {len(continuation)} tokens, exceeding context length {max_length}")
    return max(0, len(context) + len(continuation) - max_length - 1)


@torch.inference_mode()
def score_token_pairs(model: Llama, pairs: list[TokenPair], batch_size: int = 32) -> list[tuple[float, bool]]:
    """Harness-compatible left context truncation; never discard a candidate token."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    maximum = model.config.context_length
    for pair in pairs:
        validate_pair(pair, maximum)
    order = sorted(range(len(pairs)), key=lambda i: min(sum(map(len, pairs[i])) - 1, maximum), reverse=True)
    results = [(0.0, False)] * len(pairs)
    device = next(model.parameters()).device
    model.eval()
    with precision_context("ieee"), torch.autocast(device.type, enabled=False):
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            sequences = [(pairs[i][0] + pairs[i][1])[-(maximum + 1) :] for i in indices]
            length = max(len(sequence) - 1 for sequence in sequences)
            inputs = torch.zeros((len(indices), length), dtype=torch.long, device=device)
            targets = torch.full_like(inputs, -100)
            for row, (index, sequence) in enumerate(zip(indices, sequences, strict=True)):
                n = len(sequence) - 1
                count = len(pairs[index][1])
                inputs[row, :n] = torch.tensor(sequence[:-1], device=device)
                targets[row, n - count : n] = torch.tensor(pairs[index][1], device=device)
            logits = model(inputs).float()
            losses = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="none",
                ignore_index=-100,
            ).view_as(targets)
            sums = -losses.double().sum(dim=1)
            greedy = ((logits.argmax(dim=-1) == targets) | (targets == -100)).all(dim=1)
            if not torch.isfinite(sums).all():
                raise ValueError("nonfinite candidate likelihood")
            for index, value, exact in zip(indices, sums.tolist(), greedy.tolist(), strict=True):
                results[index] = (value, exact)
    return results
