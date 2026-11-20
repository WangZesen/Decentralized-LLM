import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from decentralized_llm.benchmarks.quality import load_tokenizer
from decentralized_llm.config import Config, load_config
from decentralized_llm.data.c4_source import local_stream, read_inventory
from decentralized_llm.data.cache import TokenCache, file_digest, fingerprint, local_tokenizer, prepare
from decentralized_llm.runtime import atomic_json
from scripts import train_c4_tokenizers as workflow


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "en").mkdir(parents=True)
    files = []
    for split, count in (("train", 4), ("validation", 2)):
        for shard in range(count):
            path = source / "en" / f"c4-{split}.{shard:05d}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for index in range(100):
                    text = (
                        f"{split} shard {shard} row {index}: Hello world. More words, café 中文 🙂\n  Whitespace\tkept."
                    )
                    handle.write(json.dumps({"text": text}) + "\n")
            files.append(
                {"path": str(path.relative_to(source)), "size": path.stat().st_size, "sha256": file_digest(path)}
            )
    content = {
        "version": 1,
        "dataset": workflow.DATASET,
        "revision": workflow.REVISION,
        "files": sorted(files, key=lambda entry: entry["path"]),
    }
    atomic_json(source / "source.json", {**content, "identity": fingerprint(content)})
    return tmp_path


def test_sample_preserves_documents_counts_text_bytes_and_excludes_validation(source_root: Path) -> None:
    manifest = workflow.sample(source_root, target_bytes=2000)
    texts = [json.loads(line)["text"] for line in (source_root / "corpus/text.jsonl").read_text().splitlines()]
    assert len(texts) == manifest["documents"]
    sizes = [len(text.encode("utf-8")) for text in texts]
    assert sum(sizes) == manifest["text_bytes"] >= 2000
    assert sum(sizes[:-1]) < 2000
    assert all(text.startswith("train ") and "\n  Whitespace\tkept." in text for text in texts)
    assert workflow.sample(source_root, target_bytes=2000) == manifest
    with pytest.raises(ValueError, match="settings differ"):
        workflow.sample(source_root, target_bytes=1000)
    with (source_root / "corpus/text.jsonl").open("a") as handle:
        handle.write("corruption\n")
    with pytest.raises(ValueError, match="checksum"):
        workflow.verify_manifest(source_root / "corpus")


def test_staging_permutation_matches_stream_and_missing_unused_shards_are_lazy(source_root: Path) -> None:
    source = read_inventory(source_root / "source", workflow.DATASET, workflow.REVISION)
    first = workflow.staged_entries(source, 1)[0]
    for entry in source["files"]:
        if entry["path"].startswith("en/c4-train.") and entry != first:
            (source_root / "source" / entry["path"]).unlink()
    stream = local_stream(source_root / "source", "train", source).shuffle(
        seed=42,
        buffer_size=1,
        max_buffer_input_shards=1,
    )
    with gzip.open(source_root / "source" / first["path"], "rt") as handle:
        expected = json.loads(next(handle))["text"]
    iterator = iter(stream)
    assert next(iterator)["text"] == expected
    iterator.close()


def test_tokenizer_training_exact_size_and_local_preparation(
    source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow.sample(source_root, target_bytes=10000)
    manifest = workflow.train_tokenizer(source_root, 300)
    assert workflow.train_tokenizer(source_root, 300) == manifest
    directory = source_root / "tokenizer-300"
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    workflow.check_tokenizer(tokenizer, 300)
    assert tokenizer.unk_token_id not in tokenizer.encode("असामान्य 🌎", add_special_tokens=False)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline preparation must not query the Hub")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", forbidden)
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", forbidden)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", forbidden)
    config = Config.model_validate(
        {
            "model": {"vocab_size": 300, "context_length": 16},
            "data": {
                "dataset": workflow.DATASET,
                "revision": workflow.REVISION,
                "source_dir": str(source_root / "source"),
                "tokenizer": str(directory),
                "tokenizer_revision": manifest["tokenizer_revision"],
                "cache_dir": str(source_root / "cache"),
                "prepare_train_tokens": 256,
                "prepare_validation_tokens": 65,
                "prepare_workers": 1,
                "shuffle_max_input_shards": 1,
            },
        }
    )
    first = prepare(config)
    assert first["splits"]["train"]["tokens"] == 257
    assert first["tokenizer_fingerprint"] == manifest["tokenizer_fingerprint"]
    assert prepare(config) == first
    cache = TokenCache(config.data.cache_dir)
    values = cache.read("train", 0, 257)
    assert values.max() < 300 and 2 in values
    assert len(load_tokenizer(cache)) == 300
    config.data.cache_dir = source_root / "parallel"
    config.data.prepare_workers = 2
    assert prepare(config) == first
    np.testing.assert_array_equal(values, TokenCache(config.data.cache_dir).read("train", 0, 257))

    with monkeypatch.context() as context:
        context.chdir(source_root)
        relative = config.model_copy(deep=True)
        relative.data.tokenizer = "tokenizer-300"
        TokenCache(relative.data.cache_dir).validate_config(relative)
        assert prepare(relative) == first
        relative.data.tokenizer = "different-tokenizer"
        with pytest.raises(ValueError, match="cache tokenizer mismatch"):
            TokenCache(relative.data.cache_dir).validate_config(relative)

    tokenizer.add_tokens(["new-token"])
    tokenizer.save_pretrained(directory)
    with pytest.raises(ValueError, match="revision mismatch"):
        prepare(config)
    config.data.tokenizer_revision = "main"
    with pytest.raises(ValueError, match="local tokenizer changed"):
        prepare(config)
    directory.rename(source_root / "moved-tokenizer")
    assert len(load_tokenizer(cache)) == 300
    with pytest.raises(FileNotFoundError, match="local tokenizer"):
        local_tokenizer(str(directory), manifest["tokenizer_revision"])


def test_source_checksums_revision_and_shuffle_settings(source_root: Path) -> None:
    with pytest.raises(ValueError, match="revision mismatch"):
        read_inventory(source_root / "source", workflow.DATASET, "wrong-revision")
    source = read_inventory(source_root / "source", workflow.DATASET, workflow.REVISION)
    entry = next(entry for entry in source["files"] if entry["path"].startswith("en/c4-validation."))
    with (source_root / "source" / entry["path"]).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        next(iter(local_stream(source_root / "source", "validation", source)))


def test_workflow_training_checkpoint_and_quality_smoke(source_root: Path) -> None:
    pytest.importorskip("lm_eval")
    workflow.sample(source_root, target_bytes=10000)
    workflow.train_tokenizer(source_root, 300)
    result = workflow.smoke(source_root, 300, workers=1)
    assert result["validation"]["tokens"] == 1024
    assert len(result["candidate_scores"]) == 3
    assert (source_root / "smoke-300/final/manifest.json").is_file()
    assert not (source_root / "configs").exists()


def test_overlay_uses_repository_configs_and_relative_data_paths(
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow.sample(source_root, target_bytes=10000)
    manifest = workflow.train_tokenizer(source_root, 300)
    monkeypatch.chdir(source_root.parent)
    path = workflow.overlay(source_root, 300)
    assert path == Path("configs/tokenizers/c4-bpe-300.yaml")
    config = load_config(path)
    assert config.data.cache_dir == Path(source_root.name) / "cache-300"
    assert config.data.tokenizer == f"{source_root.name}/tokenizer-300"
    assert config.data.tokenizer_revision == manifest["tokenizer_revision"]
    assert config.data.source_dir is None


def test_workflow_requires_cpu_compute_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm compute allocation"):
        workflow.compute_limits()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CPU-only"):
        workflow.compute_limits()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "2")
    assert 1 <= workflow.compute_limits() <= 2


def test_explicit_local_limit_sets_inherited_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    affinity = set(range(16))

    def set_affinity(pid: int, cpus: list[int]) -> None:
        affinity.intersection_update(cpus)

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(workflow.os, "sched_getaffinity", lambda pid: affinity)
    monkeypatch.setattr(workflow.os, "sched_setaffinity", set_affinity)
    assert workflow.compute_limits(12) == 12
    assert affinity == set(range(12))


def test_staging_location_preserves_tuning_identity(tmp_path: Path) -> None:
    from decentralized_llm.tuning.spec import numeric_recipe

    config = Config()
    before = numeric_recipe(config)
    config.data.source_dir = tmp_path
    assert numeric_recipe(config) == before
    config.data.shuffle_max_input_shards = 1
    assert numeric_recipe(config) != before
