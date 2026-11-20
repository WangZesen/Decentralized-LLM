"""Reproducible C4 BPE training and cache preparation on an allocated compute node."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from decentralized_llm.config import Config, load_config
from decentralized_llm.data.c4_source import local_stream, read_inventory
from decentralized_llm.data.cache import TokenCache, file_digest, fingerprint, prepare, tokenizer_fingerprint
from decentralized_llm.runtime import atomic_json, dependency_digest, source_digest

DATASET = "allenai/c4"
REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
ROOT = Path("data/c4-bpe")
VOCAB_SIZES = (8192, 32768)
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
RAW_BYTES = 1_000_000_000
TRAIN_TOKENS = 10_336_272_384
SHUFFLE = {"seed": 42, "buffer_size": 100_000, "max_buffer_input_shards": 1}


def versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("datasets", "tokenizers", "transformers", "numpy")}


def provenance() -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    return {
        "script_sha256": file_digest(Path(__file__)),
        "source_sha256": source_digest(project / "src/decentralized_llm"),
        "lock_sha256": dependency_digest(),
    }


def publish_manifest(directory: Path, content: dict[str, Any]) -> dict[str, Any]:
    manifest = {**content, "identity": fingerprint(content)}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def verify_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "identity"}
    if fingerprint(content) != manifest["identity"]:
        raise ValueError(f"manifest identity mismatch: {directory}")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or file_digest(directory / name) != digest:
            raise ValueError(f"artifact checksum mismatch: {directory / name}")
    return manifest


def inventory(root: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, RepoFile

    directory = root / "source"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "source.json").exists():
        return read_inventory(directory, DATASET, REVISION)
    entries = []
    for entry in HfApi().list_repo_tree(DATASET, path_in_repo="en", repo_type="dataset", revision=REVISION):
        if isinstance(entry, RepoFile) and entry.path.endswith(".json.gz"):
            if entry.lfs is None:
                raise ValueError(f"missing source checksum: {entry.path}")
            entries.append({"path": entry.path, "size": entry.size, "sha256": entry.lfs.sha256})
    content = {
        "version": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "files": sorted(entries, key=lambda e: e["path"]),
    }
    if len(entries) != 1032:
        raise ValueError(f"expected 1024 train + 8 validation shards, found {len(entries)}")
    atomic_json(directory / "source.json", {**content, "identity": fingerprint(content)})
    return read_inventory(directory, DATASET, REVISION)


def staged_entries(source: dict[str, Any], train_shards: int) -> list[dict[str, Any]]:
    train = [entry for entry in source["files"] if entry["path"].startswith("en/c4-train.")]
    indices = list(range(len(train)))
    np.random.default_rng(SHUFFLE["seed"]).shuffle(indices)
    return [train[index] for index in indices[:train_shards]] + [
        entry for entry in source["files"] if entry["path"].startswith("en/c4-validation.")
    ]


def stage(root: Path, train_shards: int) -> None:
    """Download compressed shards; no text processing or tokenizer training here."""
    from huggingface_hub import hf_hub_download

    entries = staged_entries(inventory(root), train_shards)

    def download(entry: dict[str, Any]) -> None:
        path = root / "source" / entry["path"]
        if path.exists() and path.stat().st_size == entry["size"] and file_digest(path) == entry["sha256"]:
            return
        for attempt in range(4):
            try:
                hf_hub_download(
                    DATASET,
                    entry["path"],
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=root / "source",
                    force_download=path.exists(),
                )
                if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
                    raise ValueError(f"download checksum mismatch: {path}")
                logger.info("Staged {}", entry["path"])
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, entries))
    atomic_json(
        root / "staged.json",
        {
            "dataset": DATASET,
            "revision": REVISION,
            "train_shards": train_shards,
            "shuffle_seed": SHUFFLE["seed"],
            "files": entries,
        },
    )


def sample(root: Path, target_bytes: int = RAW_BYTES) -> dict[str, Any]:
    source = read_inventory(root / "source", DATASET, REVISION)
    destination = root / "corpus"
    settings = {"source_identity": source["identity"], "target_text_bytes": target_bytes, "shuffle": SHUFFLE}
    if destination.exists():
        manifest = verify_manifest(destination)
        if manifest["settings"] != settings:
            raise ValueError("existing corpus settings differ; choose a new root")
        return manifest
    temporary = root / "corpus.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    stream = local_stream(root / "source", "train", source).shuffle(
        seed=SHUFFLE["seed"],
        buffer_size=SHUFFLE["buffer_size"],
        max_buffer_input_shards=SHUFFLE["max_buffer_input_shards"],
    )
    count = text_bytes = 0
    started = last_log = time.monotonic()
    with (temporary / "text.jsonl").open("w", encoding="utf-8") as handle, closing(iter(stream)) as documents:
        for document in documents:
            text = document["text"]
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            text_bytes += len(text.encode("utf-8"))
            if time.monotonic() - last_log > 15:
                logger.info("Corpus: {:,} text bytes / {:,}, {:,} documents", text_bytes, target_bytes, count)
                last_log = time.monotonic()
            if text_bytes >= target_bytes:
                break
        handle.flush()
        os.fsync(handle.fileno())
    if text_bytes < target_bytes:
        raise ValueError("insufficient C4 training text")
    manifest = publish_manifest(
        temporary,
        {
            "version": 1,
            "dataset": DATASET,
            "revision": REVISION,
            "split": "train",
            "settings": settings,
            "documents": count,
            "text_bytes": text_bytes,
            "seconds": time.monotonic() - started,
            "versions": versions(),
            "provenance": provenance(),
            "files": {"text.jsonl": file_digest(temporary / "text.jsonl")},
        },
    )
    temporary.rename(destination)
    return manifest


def corpus_batches(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8") as handle:
        batch = []
        for line in handle:
            batch.append(json.loads(line)["text"])
            if len(batch) == 1024:
                yield batch
                batch = []
        if batch:
            yield batch


def check_tokenizer(tokenizer: Any, size: int) -> None:
    if (
        len(tokenizer) != size
        or tokenizer.vocab_size != size
        or set(tokenizer.get_vocab().values()) != set(range(size))
    ):
        raise ValueError(f"expected exactly {size} contiguous vocabulary IDs including special tokens")
    if [tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS] != list(range(4)):
        raise ValueError("unexpected special-token IDs")
    if [tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] != list(
        range(4)
    ):
        raise ValueError("unexpected special-token roles")
    for text in ("", "Hello world!", " leading  spaces\t\n\n", "中文 العربية café 🙂", "<unk><s></s><pad>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids != tokenizer.encode(text, add_special_tokens=True):
            raise ValueError("tokenizer inserts automatic special tokens")
        if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
            raise ValueError("tokenizer failed exact Unicode/whitespace round trip")
        if any(not 0 <= token < size for token in ids):
            raise ValueError("token ID outside vocabulary")


def train_tokenizer(root: Path, size: int) -> dict[str, Any]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if not 260 <= size <= 65536:
        raise ValueError("byte-level BPE vocabulary must contain 260..65536 tokens")
    corpus = verify_manifest(root / "corpus")
    destination = root / f"tokenizer-{size}"
    settings = {
        "vocab_size": size,
        "min_frequency": 2,
        "special_tokens": SPECIAL_TOKENS,
        "initial_alphabet": sorted(pre_tokenizers.ByteLevel.alphabet()),
        "add_prefix_space": False,
        "use_regex": True,
        "normalizer": None,
        "automatic_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    if destination.exists():
        manifest = verify_manifest(destination)
        if manifest["corpus_identity"] != corpus["identity"] or manifest["settings"] != settings:
            raise ValueError("existing tokenizer settings/corpus differ")
        check_tokenizer(AutoTokenizer.from_pretrained(destination, local_files_only=True), size)
        return manifest
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=settings["initial_alphabet"],
        show_progress=True,
    )
    started = time.monotonic()
    logger.info("Training {}-token BPE on {:,} bytes", size, corpus["text_bytes"])
    backend.train_from_iterator(corpus_batches(root / "corpus/text.jsonl"), trainer, length=corpus["documents"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False,
    )
    check_tokenizer(tokenizer, size)
    identity = tokenizer_fingerprint(tokenizer)
    tokenizer.save_pretrained(temporary)
    loaded = AutoTokenizer.from_pretrained(temporary, local_files_only=True, use_fast=True)
    check_tokenizer(loaded, size)
    if tokenizer_fingerprint(loaded) != identity:
        raise ValueError("tokenizer fingerprint changed after save/reload")
    manifest = publish_manifest(
        temporary,
        {
            "version": 1,
            "corpus_identity": corpus["identity"],
            "settings": settings,
            "tokenizer_fingerprint": identity,
            "tokenizer_revision": "sha256:" + identity,
            "seconds": time.monotonic() - started,
            "versions": versions(),
            "provenance": provenance(),
            "files": {p.name: file_digest(p) for p in sorted(temporary.iterdir()) if p.is_file()},
        },
    )
    temporary.rename(destination)
    return manifest


def overlay_content(root: Path, size: int) -> dict[str, Any]:
    manifest = verify_manifest(root / f"tokenizer-{size}")
    try:
        data_root = root.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        data_root = root.resolve()
    return {
        "model": {"vocab_size": size},
        "data": {
            "dataset": DATASET,
            "revision": REVISION,
            "tokenizer": str(data_root / f"tokenizer-{size}"),
            "tokenizer_revision": manifest["tokenizer_revision"],
            "cache_dir": str(data_root / f"cache-{size}"),
            "shuffle_seed": 42,
            "shuffle_buffer": 100_000,
            "shuffle_max_input_shards": 1,
        },
    }


def overlay(root: Path, size: int, directory: Path = Path("configs/tokenizers")) -> Path:
    path = directory / f"c4-bpe-{size}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(overlay_content(root, size), sort_keys=False))
    return path


def smoke(root: Path, size: int, workers: int) -> dict[str, Any]:
    import torch

    from decentralized_llm.benchmarks.quality import load_quality_model, load_tokenizer, score_token_pairs
    from decentralized_llm.benchmarks.quality_tasks import NativeLM
    from decentralized_llm.training.loop import evaluate_checkpoint, train

    raw = Config.model_validate(overlay_content(root, size)).model_dump(mode="json")
    raw["model"].update(layers=1, width=32, heads=4, ffn_width=64, context_length=64)
    raw["data"].update(
        cache_dir=str(root / f"smoke-cache-{size}"),
        source_dir=str(root / "source"),
        prepare_train_tokens=4096,
        prepare_validation_tokens=1025,
        prepare_workers=min(workers, 2),
    )
    raw["runtime"].update(
        device="cpu",
        amp=False,
        compile=False,
        execution_backend="native",
        fused_optimizer=False,
        cpu_threads=2,
        output_dir=str(root / f"smoke-{size}"),
    )
    raw["distributed"]["backend"] = "gloo"
    parameters = Config.model_validate(raw).model.parameter_count
    budget = 256 / parameters
    raw["training"].update(batch_tokens=128, tokens_per_parameters=budget, epoch_tokens_per_parameters=budget)
    raw["lr_schedule"]["warmup_tokens_per_parameters"] = 0
    raw["evaluation"].update(subset_blocks=4, batch_size=2)
    cfg = Config.model_validate(raw)
    prepare(cfg)
    final = cfg.runtime.output_dir / "final"
    if not (final / "manifest.json").exists():
        train(cfg)
    validation = evaluate_checkpoint(cfg, final)
    model, _ = load_quality_model(cfg, final)
    cache = TokenCache(cfg.data.cache_dir)
    tokenizer = load_tokenizer(cache)
    adapter = NativeLM(model, tokenizer, batch_size=2)
    pairs = [
        adapter.encode_pair("", "A short sentence."),
        adapter.encode_pair("Question: ", "yes"),
        adapter.encode_pair("A longer context ", "with several answer tokens."),
    ]
    batched = score_token_pairs(model, pairs, 3)
    single = score_token_pairs(model, pairs, 1)
    if not math.isfinite(validation["loss"]):
        raise ValueError("nonfinite smoke validation loss")
    for actual, expected in zip(batched, single, strict=True):
        torch.testing.assert_close(actual[0], expected[0], atol=1e-4, rtol=1e-5)
    result = {
        "validation": validation,
        "candidate_scores": batched,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
    }
    atomic_json(root / f"smoke-{size}.json", result)
    return result


def compute_limits(local_cpus: int | None = None) -> int:
    if local_cpus is not None:
        if local_cpus < 1:
            raise ValueError("local CPU limit must be positive")
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:local_cpus])
    elif not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("text processing and tokenizer training require a Slurm compute allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES='' for this CPU-only workload")
    cpus = len(os.sched_getaffinity(0))
    return min(cpus, int(os.environ.get("SLURM_CPUS_ON_NODE", str(cpus))))


def run(
    root: Path,
    action: str,
    size: int | None,
    skip_stage: bool,
    train_shards: int,
    local_cpus: int | None = None,
) -> None:
    workers = compute_limits(local_cpus)
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    atomic_json(
        root / "execution.json",
        {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "placement": "local" if local_cpus is not None else "slurm",
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "host": platform.node(),
            "architecture": platform.machine(),
            "cpus": workers,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "versions": versions(),
            "provenance": provenance(),
        },
    )
    if action in ("sample", "run"):
        if not skip_stage:
            stage(root, 4)
        sample(root)
    if action in ("train", "run"):
        for vocabulary in [size] if size else VOCAB_SIZES:
            train_tokenizer(root, vocabulary)
            overlay(root, vocabulary)
            smoke(root, vocabulary, workers)
    if action in ("prepare", "run"):
        if not skip_stage:
            stage(root, train_shards)
        for vocabulary in [size] if size else VOCAB_SIZES:
            cfg = load_config(overlay(root, vocabulary))
            cfg.data.source_dir = root / "source"
            cfg.model.context_length = 1024
            cfg.data.prepare_train_tokens = TRAIN_TOKENS
            cfg.data.prepare_workers = workers
            prepare(cfg)
            cache = TokenCache(cfg.data.cache_dir)
            cache.verify()
            if cache.manifest["splits"]["train"]["tokens"] != TRAIN_TOKENS + 1:
                raise ValueError("full cache token count mismatch")
    if action == "run":
        atomic_json(
            root / "result.json",
            {
                "status": "complete",
                "execution": json.loads((root / "execution.json").read_text()),
                "corpus": verify_manifest(root / "corpus"),
                "tokenizers": {str(v): verify_manifest(root / f"tokenizer-{v}") for v in VOCAB_SIZES},
                "caches": {str(v): TokenCache(root / f"cache-{v}").manifest for v in VOCAB_SIZES},
                "smoke": {str(v): json.loads((root / f"smoke-{v}.json").read_text()) for v in VOCAB_SIZES},
            },
        )


def submit(root: Path, skip_stage: bool, train_shards: int) -> None:
    from decentralized_llm.launch.slurm import submit_batch

    project = Path.cwd().resolve()
    command = ["-m", "scripts.train_c4_tokenizers", "run", "--root", str(root), "--train-shards", str(train_shards)]
    if skip_stage:
        command.append("--skip-stage")
    script = root / "job.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(project)) + "\n"
        "source scripts/environment.sh\n"
        "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n"
        "srun --ntasks=1 --cpu-bind=cores env CUDA_VISIBLE_DEVICES='' "
        '"${DECENTRALIZED_PYTHON}" -m pytest -q -m "not cuda" '
        + "> "
        + shlex.quote(str(root))
        + "/checks-${SLURM_JOB_ID}.log 2>&1\n"
        + "exec srun --ntasks=1 --cpu-bind=cores env CUDA_VISIBLE_DEVICES='' "
        + '"${DECENTRALIZED_PYTHON}" '
        + shlex.join(command)
        + "\n"
    )
    previous = root / "submission.json"
    if previous.exists():
        job = json.loads(previous.read_text())["job_id"]
        result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%i"], text=True, capture_output=True, check=True)
        if result.stdout.strip():
            raise RuntimeError(f"tokenizer job {job} is already active")
    receipt = submit_batch(
        [
            "sbatch",
            "--parsable",
            "--partition=gpu",
            "--nodes=1",
            "--ntasks=1",
            "--gpus-per-node=1",
            "--cpus-per-task=72",
            "--mem=100G",
            "--time=24:00:00",
            "--job-name=c4-bpe",
            f"--output={root / 'job-%j.log'}",
        ],
        script,
    )
    atomic_json(previous, receipt)
    print(json.dumps(receipt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "sample", "train", "prepare", "run", "submit"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--vocab-size", type=int, choices=VOCAB_SIZES)
    parser.add_argument("--train-shards", type=int, default=128)
    parser.add_argument("--local-cpus", type=int, help="explicitly allow local processing with this CPU affinity limit")
    parser.add_argument("--skip-stage", action="store_true", help="read already-staged source files without downloads")
    args = parser.parse_args()
    if args.vocab_size is not None and args.action not in ("train", "prepare"):
        parser.error("--vocab-size is supported only by train and prepare")
    if args.local_cpus is not None and (args.local_cpus < 1 or args.action in ("stage", "submit")):
        parser.error("--local-cpus must be positive and is supported only by compute stages")
    if not 1 <= args.train_shards <= 1024:
        parser.error("--train-shards must be between 1 and 1024")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "submit":
        submit(root, args.skip_stage, args.train_shards)
        return
    lock_name = ".stage.lock" if args.action == "stage" else ".run.lock"
    with (root / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "stage":
            stage(root, args.train_shards)
        else:
            run(root, args.action, args.vocab_size, args.skip_stage, args.train_shards, args.local_cpus)


if __name__ == "__main__":
    main()
