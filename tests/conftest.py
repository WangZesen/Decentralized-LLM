from pathlib import Path

import numpy as np
import pytest
import torch

from decentralized_llm.cache import TokenCache, TokenWriter, fingerprint
from decentralized_llm.config import Config
from decentralized_llm.runtime import atomic_json
from decentralized_llm.state import CacheContent, CacheManifest


def make_cache(path: Path, config: Config, tokens: int = 151, shard_tokens: int = 11) -> TokenCache:
    path.mkdir(parents=True)
    splits = {}
    for split, count in (("train", tokens), ("validation", 39)):
        writer = TokenWriter(path, split, shard_tokens)
        writer.write(np.arange(count) % config.model.vocab_size)
        splits[split] = writer.finish()
    content = CacheContent(
        version=1,
        dataset=config.data.dataset,
        dataset_revision="a" * 40,
        tokenizer=config.data.tokenizer,
        tokenizer_revision="b" * 40,
        tokenizer_fingerprint="test",
        vocab_size=config.model.vocab_size,
        eos_token_id=2,
        dtype="<u2",
        preprocessing={
            "shuffle_seed": config.data.shuffle_seed,
            "shuffle_buffer": config.data.shuffle_buffer,
            "append_eos": True,
            "add_special_tokens": False,
        },
        validation_complete=True,
        splits=splits,
    )
    atomic_json(path / "manifest.json", CacheManifest(**content, identity=fingerprint(content)))
    return TokenCache(path)


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    torch.set_num_threads(1)
    return Config.model_validate(
        {
            "model": {"vocab_size": 17, "layers": 1, "width": 8, "heads": 2, "ffn_width": 16, "context_length": 4},
            "training": {
                "batch_tokens": 24,
                "micro_batch_size": 2,
                "tokens_per_parameters": 0.2,
                "epoch_tokens_per_parameters": 0.1,
                "log_every": 1,
            },
            "data": {"cache_dir": str(tmp_path / "cache"), "reader_threads": 2, "prefetch_batches": 3},
            "runtime": {
                "device": "cpu",
                "amp": False,
                "compile": False,
                "fused_optimizer": False,
                "cpu_threads": 1,
                "output_dir": str(tmp_path / "run"),
                "deterministic": True,
            },
            "distributed": {"backend": "gloo", "timeout_seconds": 30},
            "lr_schedule": {"name": "cosine", "warmup_tokens_per_parameters": 0.03},
            "evaluation": {"subset_blocks": 5, "batch_size": 3},
        }
    )


@pytest.fixture
def cache(tiny_config: Config) -> TokenCache:
    return make_cache(tiny_config.data.cache_dir, tiny_config)
