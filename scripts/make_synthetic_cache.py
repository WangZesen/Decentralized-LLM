"""Create deterministic uint16 fixtures on the target filesystem for pipeline benchmarks."""

import argparse
from pathlib import Path

import numpy as np

from decentralized_llm.cache import TokenWriter, fingerprint
from decentralized_llm.config import load_config
from decentralized_llm.runtime import atomic_json
from decentralized_llm.state import CacheContent, CacheManifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, action="append", required=True)
    parser.add_argument("--tokens", type=int, default=32 * 1024**2 + 1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.data.dataset != "synthetic" or cfg.data.tokenizer != "synthetic":
        parser.error("synthetic fixtures require data.dataset=synthetic and data.tokenizer=synthetic")
    if args.tokens < cfg.model.context_length + 1:
        parser.error("fixture must contain at least one complete block plus lookahead")
    destination = cfg.data.cache_dir
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        parser.error("destination exists; choose a new fixture path")
    temporary = destination.with_name(destination.name + ".partial")
    temporary.mkdir()
    splits = {}
    rng = np.random.Generator(np.random.PCG64(2026))
    for split, total in (("train", args.tokens), ("validation", 8193)):
        writer = TokenWriter(temporary, split, cfg.data.shard_tokens)
        try:
            for start in range(0, total, 1024**2):
                writer.write(rng.integers(0, cfg.model.vocab_size, min(total - start, 1024**2), dtype=np.uint16))
            splits[split] = writer.finish()
        finally:
            writer.close()
    content = CacheContent(
        version=1,
        dataset="synthetic",
        dataset_revision="synthetic-v1",
        tokenizer="synthetic",
        tokenizer_revision="synthetic-v1",
        tokenizer_fingerprint="synthetic-v1",
        vocab_size=cfg.model.vocab_size,
        eos_token_id=2,
        dtype="<u2",
        preprocessing={
            "shuffle_seed": cfg.data.shuffle_seed,
            "shuffle_buffer": cfg.data.shuffle_buffer,
            "append_eos": True,
            "add_special_tokens": False,
        },
        validation_complete=False,
        splits=splits,
    )
    manifest = CacheManifest(**content, identity=fingerprint(content))
    atomic_json(temporary / "manifest.json", manifest)
    temporary.rename(destination)
    print(f"Published synthetic benchmark fixture: {destination} ({manifest['identity']})")


if __name__ == "__main__":
    main()
