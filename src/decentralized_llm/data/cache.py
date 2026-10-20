"""Version-1 tiny-llm cache compatibility and deterministic C4 preparation."""

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Generator, Iterable, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, wait
from contextlib import closing, nullcontext
from fractions import Fraction
from itertools import islice
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike

from decentralized_llm.config import Config
from decentralized_llm.state import CacheContent, CacheManifest, Shard, SplitManifest, TokenArray


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class TokenWriter:
    def __init__(self, directory: Path, split: str, shard_tokens: int):
        self.directory, self.split, self.shard_tokens = directory, split, shard_tokens
        self.shards: list[Shard] = []
        self.count = 0
        self.handle: BinaryIO | None = None
        self.in_shard = 0

    def write(self, tokens: ArrayLike) -> None:
        values = np.asarray(tokens)
        if values.size and (values.min() < 0 or values.max() > 65535):
            raise ValueError("token ID cannot be represented as uint16")
        values = values.astype("<u2")
        while len(values):
            if self.handle is None:
                self.path = self.directory / f"{self.split}-{len(self.shards):05d}.bin"
                self.handle = self.path.open("wb")
                self.in_shard = 0
            take = min(len(values), self.shard_tokens - self.in_shard)
            self.handle.write(values[:take].tobytes())
            self.in_shard += take
            self.count += take
            values = values[take:]
            if self.in_shard == self.shard_tokens:
                self._close_shard()

    def _close_shard(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.close()
        self.shards.append(Shard(file=self.path.name, tokens=self.in_shard, sha256=file_digest(self.path)))

    def close(self) -> None:
        """Release an open shard, including when preparation fails."""
        if self.handle is not None:
            handle, self.handle = self.handle, None
            handle.close()

    def finish(self) -> SplitManifest:
        self._close_shard()
        return SplitManifest(tokens=self.count, shards=self.shards)


class _DocumentTokenizer(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, return_attention_mask: bool
    ) -> Mapping[str, Sequence[Sequence[int]]]: ...


_worker_tokenizer: _DocumentTokenizer | None = None


def _initialize_tokenizer(tokenizer_path: str) -> None:
    # Each spawned worker owns one tokenizer and uses a single tokenizer thread.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)


def _tokenize_documents(tokenizer: _DocumentTokenizer, texts: list[str]) -> TokenArray:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define EOS")
    encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    values = np.asarray([token for document in encoded for token in (*document, eos)], dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() > 65535):
        raise ValueError("token ID cannot be represented as uint16")
    return values.astype("<u2")


def _tokenize_worker(texts: list[str]) -> TokenArray:
    if _worker_tokenizer is None:
        raise RuntimeError("tokenizer worker is not initialized")
    return _tokenize_documents(_worker_tokenizer, texts)


def _tokenized_batches(
    batches: Iterable[list[str]],
    tokenizer: _DocumentTokenizer,
    pool: Executor | None,
    workers: int,
) -> Generator[TokenArray]:
    """Consume bounded parallel work in source order, independent of worker timing."""
    if pool is None:
        for batch in batches:
            yield _tokenize_documents(tokenizer, batch)
        return

    source = iter(batches)
    pending: deque[Future[TokenArray]] = deque()
    try:
        for batch in islice(source, 2 * workers):
            pending.append(pool.submit(_tokenize_worker, batch))
        while pending:
            # Keep the current future in the queue until it completes so an
            # interruption also drains this task before the next split starts.
            values = pending[0].result()
            pending.popleft()
            yield values
            for batch in islice(source, 1):
                pending.append(pool.submit(_tokenize_worker, batch))
    finally:
        for future in pending:
            future.cancel()
        wait([future for future in pending if not future.cancelled()])


def training_boundaries(config: Config, parameters: int) -> list[int]:
    """Nearest cumulative global batches, independent of the final epoch count."""
    length = config.model.context_length
    train = config.training
    epoch_batches = Fraction(str(train.epoch_tokens_per_parameters)) * parameters / train.batch_tokens
    batch_blocks = train.batch_tokens // length
    boundaries = [
        math.floor(k * epoch_batches + Fraction(1, 2)) * batch_blocks for k in range(1, train.epoch_count + 1)
    ]
    if boundaries[0] == 0 or len(set(boundaries)) != len(boundaries):
        raise ValueError("virtual epochs must each contain at least one global batch")
    return boundaries


def prepare(config: Config) -> CacheManifest:
    """Prepare privately, then atomically publish both splits and their manifest."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    cfg = config.data
    required = (
        cfg.prepare_train_tokens
        or training_boundaries(config, config.model.parameter_count)[-1] * config.model.context_length
    )
    length = config.model.context_length
    required_blocks = (required + length - 1) // length
    train_limit = required_blocks * length + 1
    destination = cfg.cache_dir
    if destination.exists():
        cache = TokenCache(destination)
        cache.validate_config(config)
        if cache.blocks("train", length) < required_blocks:
            raise ValueError("existing cache is too small; prepare a larger cache at a new path")
        if cfg.prepare_validation_tokens is None and not cache.manifest["validation_complete"]:
            raise ValueError("existing cache has truncated validation; use a new cache path")
        cache.verify()
        logger.info("Verified and reused cache {}", destination)
        return cache.manifest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    # An exclusive lock protects preparation shared by multiple GPU processes.
    lock_path = destination.with_name(destination.name + ".lock")
    import fcntl

    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if temporary.exists():
            logger.warning("Discarding incomplete preparation at {}; completed caches are immutable", temporary)
            shutil.rmtree(temporary)
        temporary.mkdir()
        api = HfApi()
        dataset_sha = api.dataset_info(cfg.dataset, revision=cfg.revision).sha
        tokenizer_sha = api.model_info(cfg.tokenizer, revision=cfg.tokenizer_revision).sha
        if dataset_sha is None or tokenizer_sha is None:
            raise ValueError("could not resolve dataset/tokenizer commit SHA")
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, revision=tokenizer_sha, use_fast=True)
        if len(tokenizer) != config.model.vocab_size or tokenizer.eos_token_id is None:
            raise ValueError("model vocabulary must match the tokenizer, which must define EOS")
        tokenizer.save_pretrained(temporary / "tokenizer")
        tokenizer_hash = fingerprint(
            {
                "vocab": tokenizer.get_vocab(),
                "eos": tokenizer.eos_token_id,
                "backend": tokenizer.backend_tokenizer.to_str(),
            }
        )
        paths = sorted(api.list_repo_files(cfg.dataset, repo_type="dataset", revision=dataset_sha))
        manifest: CacheContent = {
            "version": 1,
            "dataset": cfg.dataset,
            "dataset_revision": dataset_sha,
            "tokenizer": cfg.tokenizer,
            "tokenizer_revision": tokenizer_sha,
            "tokenizer_fingerprint": tokenizer_hash,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "dtype": "<u2",
            "preprocessing": {
                "shuffle_seed": cfg.shuffle_seed,
                "shuffle_buffer": cfg.shuffle_buffer,
                "append_eos": True,
                "add_special_tokens": False,
            },
            "validation_complete": cfg.prepare_validation_tokens is None,
            "splits": {},
        }
        pool_context = (
            ProcessPoolExecutor(
                max_workers=cfg.prepare_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_tokenizer,
                initargs=(str((temporary / "tokenizer").resolve()),),
            )
            if cfg.prepare_workers > 1
            else nullcontext(None)
        )
        with pool_context as pool:
            for split, limit in (
                ("train", train_limit),
                ("validation", cfg.prepare_validation_tokens),
            ):
                selected = [p for p in paths if p.startswith(f"en/c4-{split}.") and p.endswith(".json.gz")]
                if not selected:
                    raise ValueError(f"no English C4 {split} shards at {dataset_sha}")
                urls = [f"hf://datasets/{cfg.dataset}@{dataset_sha}/{p}" for p in selected]
                stream = load_dataset("json", data_files={split: urls}, split=split, streaming=True)
                if split == "train":
                    stream = stream.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.shuffle_buffer)
                writer = TokenWriter(temporary, split, cfg.shard_tokens)
                start, last_log = time.monotonic(), 0.0
                with (
                    closing(stream.iter(batch_size=cfg.tokenize_batch_size)) as source,
                    closing(
                        _tokenized_batches(
                            (batch["text"] for batch in source),
                            tokenizer,
                            pool,
                            cfg.prepare_workers,
                        )
                    ) as batches,
                    closing(writer),
                ):
                    for flattened in batches:
                        if limit is not None:
                            flattened = flattened[: max(0, limit - writer.count)]
                        writer.write(flattened)
                        now = time.monotonic()
                        if now - last_log > 15:
                            logger.info(
                                "Preparing {}: {:,} tokens ({:,.0f} tokens/s)",
                                split,
                                writer.count,
                                writer.count / (now - start),
                            )
                            last_log = now
                        if limit is not None and writer.count >= limit:
                            break
                    manifest["splits"][split] = writer.finish()
                if writer.count < 2 or (limit is not None and writer.count < limit):
                    raise ValueError(f"insufficient {split} tokens: {writer.count}")
        completed_manifest = CacheManifest(**manifest, identity=fingerprint(manifest))
        from decentralized_llm.runtime import atomic_json

        atomic_json(temporary / "manifest.json", completed_manifest)
        temporary.rename(destination)
        logger.success("Published cache {} ({})", destination, completed_manifest["identity"][:12])
        return completed_manifest


class TokenCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest: CacheManifest = json.loads((self.directory / "manifest.json").read_text())
        content = {k: v for k, v in self.manifest.items() if k != "identity"}
        if self.manifest.get("version") != 1 or fingerprint(content) != self.manifest.get("identity"):
            raise ValueError("invalid cache manifest version or identity")
        if self.manifest["dtype"] != "<u2" or set(self.manifest["splits"]) != {"train", "validation"}:
            raise ValueError("cache must contain uint16 train and validation splits")
        if not self.manifest["preprocessing"].get("append_eos"):
            raise ValueError("cache must append EOS")
        self.paths: dict[str, list[Path]] = {}
        self.offsets: dict[str, list[int]] = {}
        for split, info in self.manifest["splits"].items():
            if info["tokens"] < 2:
                raise ValueError(f"insufficient {split} tokens")
            offsets, paths = [0], []
            for shard in info["shards"]:
                if Path(shard["file"]).name != shard["file"] or shard["tokens"] <= 0:
                    raise ValueError("invalid shard path or token count")
                path = self.directory / shard["file"]
                if path.stat().st_size != shard["tokens"] * 2:
                    raise ValueError(f"truncated token shard: {path}")
                paths.append(path)
                offsets.append(offsets[-1] + shard["tokens"])
            if offsets[-1] != info["tokens"]:
                raise ValueError("manifest token count does not match shards")
            self.paths[split], self.offsets[split] = paths, offsets

    def validate_config(self, config: Config) -> None:
        data, manifest = config.data, self.manifest
        if manifest["vocab_size"] != config.model.vocab_size:
            raise ValueError("cache/model vocabulary mismatch")
        for key, value in (("dataset", data.dataset), ("tokenizer", data.tokenizer)):
            if manifest[key] != value:
                raise ValueError(f"cache {key} mismatch")
        for expected, key in (
            (data.revision, "dataset_revision"),
            (data.tokenizer_revision, "tokenizer_revision"),
        ):
            if expected != "main" and expected != manifest[key]:
                raise ValueError(f"cache {key} mismatch; specify the resolved commit SHA")
        if (
            manifest["preprocessing"]["shuffle_seed"] != data.shuffle_seed
            or manifest["preprocessing"]["shuffle_buffer"] != data.shuffle_buffer
        ):
            raise ValueError("cache preprocessing mismatch")

    def verify(self) -> None:
        for info in self.manifest["splits"].values():
            for shard in info["shards"]:
                if file_digest(self.directory / shard["file"]) != shard["sha256"]:
                    raise ValueError(f"checksum mismatch: {shard['file']}")

    def blocks(self, split: str, length: int, partial: bool = False) -> int:
        targets = self.manifest["splits"][split]["tokens"] - 1
        return math.ceil(targets / length) if partial else targets // length

    def read(self, split: str, start: int, stop: int) -> TokenArray:
        """Read a contiguous interval into a compact owning array, without mmap."""
        if not 0 <= start < stop <= self.offsets[split][-1]:
            raise IndexError("token range outside cache")
        values = np.empty(stop - start, dtype="<u2")
        self.read_into(split, start, stop, values)
        return values

    def read_into(self, split: str, start: int, stop: int, destination: TokenArray) -> None:
        """Fill a preallocated uint16 array using sequential reads within each file."""
        if not 0 <= start < stop <= self.offsets[split][-1]:
            raise IndexError("token range outside cache")
        if destination.dtype != np.dtype("<u2") or not destination.flags.c_contiguous:
            raise ValueError("destination must be a contiguous uint16 array")
        if destination.size < stop - start:
            raise ValueError("destination is too small")
        output = memoryview(destination).cast("B")
        written = 0
        while start < stop:
            index = bisect_right(self.offsets[split], start) - 1
            end = min(stop, self.offsets[split][index + 1])
            remaining = 2 * (end - start)
            with self.paths[split][index].open("rb", buffering=0) as handle:
                handle.seek(2 * (start - self.offsets[split][index]))
                while remaining:
                    size = handle.readinto(output[written : written + remaining])
                    if not size:
                        raise OSError(f"unexpected EOF in token shard: {self.paths[split][index]}")
                    written += size
                    remaining -= size
            start = end
