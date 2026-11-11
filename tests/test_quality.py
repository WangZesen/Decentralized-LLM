import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from decentralized_llm.benchmarks.quality import TokenPair, load_quality_model, score_token_pairs
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache, file_digest, fingerprint
from decentralized_llm.data.sampling import GlobalSampler
from decentralized_llm.models.initialization import INITIALIZATION
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.runtime import atomic_json
from decentralized_llm.training.checkpoints import recipe_identity


def test_likelihood_masks_prompts_padding_and_preserves_all_targets(tiny_config: Config) -> None:
    model = Llama(tiny_config.model).eval()
    pairs = [([2], [3, 4]), ([2, 5, 6], [7]), ([2, 3, 4, 5, 6], [7, 8]), ([2], [3, 4, 5, 6])]
    batched = score_token_pairs(model, pairs, 4)
    single = score_token_pairs(model, pairs, 1)
    for pair, actual, individual in zip(pairs, batched, single, strict=True):
        sequence = (pair[0] + pair[1])[-5:]
        logits = model(torch.tensor([sequence[:-1]]))[0, -len(pair[1]) :]
        expected = -F.cross_entropy(logits, torch.tensor(pair[1]), reduction="none").double().sum().item()
        assert actual[0] == pytest.approx(expected, abs=2e-6)
        assert actual[0] == pytest.approx(individual[0], abs=2e-6)
        assert actual[1] == bool((logits.argmax(-1) == torch.tensor(pair[1])).all())


@pytest.mark.parametrize("pair", [([], [1]), ([1], []), ([1], [1, 2, 3, 4, 5])])
def test_likelihood_rejects_missing_context_or_overlong_candidate(tiny_config: Config, pair: TokenPair) -> None:
    with pytest.raises(ValueError):
        score_token_pairs(Llama(tiny_config.model), [pair])


def checkpoint_fixture(tmp_path: Path, config: Config, cache: TokenCache) -> Path:
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    sampler = GlobalSampler.from_config(cache, config)
    identity = sampler.identity
    sampler.close()
    models = [Llama(config.model) for _ in range(config.distributed.world_size)]
    for rank, model in enumerate(models):
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.01 * (rank + 1))
    state = {"sampler": identity, "cursor": 0}
    if config.runtime.training_backend == "packed":
        workers = len(models) if config.distributed.mode == "decentralized" else 1
        packed = PackedLlama(config.model, workers, canonical=models[0])
        for rank in range(workers):
            for entry in packed.layout:
                with torch.no_grad():
                    packed.get_parameter(entry.name)[rank].copy_(models[rank].get_parameter(entry.name))
        torch.save({"rank": 0, "loader": state, "model": packed.packed_state_dict()}, folder / "packed.pt")
    else:
        for rank, model in enumerate(models):
            payload = {"rank": rank, "loader": state}
            if rank == 0 or config.distributed.mode == "decentralized":
                payload["model"] = model.state_dict()
            torch.save(payload, folder / f"rank-{rank:05d}.pt")
    atomic_json(
        folder / "manifest.json",
        {
            "version": 1,
            "world_size": config.distributed.world_size,
            "training_backend": config.runtime.training_backend,
            "initialization": INITIALIZATION,
            "recipe_identity": recipe_identity(config, cache),
            "sampler": identity,
            "training_state": False,
            "cursor": 0,
            "step": 0,
            "execution": {"backend": "native"},
            "files": [{"file": p.name, "sha256": file_digest(p)} for p in sorted(folder.glob("*.pt"))],
        },
    )
    return folder


@pytest.mark.parametrize("backend", ["packed", "distributed"])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
def test_checkpoint_load_averages_workers_without_training_runtime(
    tmp_path: Path,
    tiny_config: Config,
    cache: TokenCache,
    backend: Literal["packed", "distributed"],
    mode: Literal["ddp", "decentralized"],
) -> None:
    tiny_config.runtime.training_backend = backend
    tiny_config.distributed.mode = mode
    tiny_config.distributed.gpus_per_node = 2
    checkpoint = checkpoint_fixture(tmp_path, tiny_config, cache)
    model, _ = load_quality_model(tiny_config, checkpoint)
    expected = 0.015 if mode == "decentralized" else 0.01
    for p in model.parameters():
        torch.testing.assert_close(p, torch.full_like(p, expected))
    first_file = next(checkpoint.glob("*.pt"))
    with first_file.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        load_quality_model(tiny_config, checkpoint)


def test_checkpoint_rejects_recipe_and_rank_state_changes(
    tmp_path: Path, tiny_config: Config, cache: TokenCache
) -> None:
    checkpoint = checkpoint_fixture(tmp_path, tiny_config, cache)
    changed = tiny_config.model_copy(deep=True)
    changed.runtime.seed += 1
    with pytest.raises(ValueError, match="recipe"):
        load_quality_model(changed, checkpoint)
    path = checkpoint / "rank-00000.pt"
    payload = torch.load(path, weights_only=False)
    payload["rank"] = 9
    torch.save(payload, path)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = file_digest(path)
    atomic_json(checkpoint / "manifest.json", manifest)
    with pytest.raises(ValueError, match="rank/cursor"):
        load_quality_model(tiny_config, checkpoint)


def test_harness_prompts_metrics_and_empty_context() -> None:
    pytest.importorskip("lm_eval")
    from decentralized_llm.benchmarks.quality_tasks import NativeLM, PreparedTask

    class Tokenizer:
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(c) for c in text]

    adapter = NativeLM(None, Tokenizer())
    assert adapter.encode_pair("", "ab") == ([2], [97, 98])
    assert adapter.encode_pair("x ", "ab") == ([120], [32, 97, 98])
    with pytest.raises(NotImplementedError):
        adapter.generate_until([])
    config = {
        "task": "fixture",
        "output_type": "multiple_choice",
        "validation_split": "evaluation",
        "doc_to_text": "Question: {{goal}}\nAnswer:",
        "doc_to_target": "label",
        "doc_to_choice": "{{[sol1, sol2]}}",
        "num_fewshot": 0,
        "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True} for m in ("acc", "acc_norm")],
    }
    doc = {"goal": "Test?", "sol1": "a", "sol2": "bb", "label": 1}
    task = PreparedTask(config, [doc])
    context = task.doc_to_text(doc)
    assert isinstance(context, str)
    requests = task.construct_requests(doc, context)
    assert isinstance(requests, list)
    assert [r.args for r in requests] == [("Question: Test?\nAnswer:", " a"), ("Question: Test?\nAnswer:", " bb")]
    assert task.process_results(doc, [(-3.0, False), (-4.0, False)]) == {"acc": 0.0, "acc_norm": 1.0}
    doc["label"] = 0
    assert task.process_results(doc, [(-3.0, False), (-3.0, False)])["acc"] == 1.0


def test_offline_evaluation_resume_and_sample_integrity(
    tmp_path: Path, tiny_config: Config, cache: TokenCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lm_eval")
    from decentralized_llm.benchmarks import quality_tasks as quality

    checkpoint = checkpoint_fixture(tmp_path, tiny_config, cache)
    monkeypatch.setattr(quality, "load_tokenizer", lambda cache: None)
    suite = tmp_path / "suite"
    suite.mkdir()
    doc = {"goal": "Test", "sol1": "a", "sol2": "bb", "label": 0}
    path = suite / "piqa.jsonl"
    path.write_text(
        json.dumps(
            {"id": 0, "doc": doc, "requests": [["Test", " a"], ["Test", " bb"]], "tokens": [[[2], [3]], [[2], [4, 5]]]}
        )
        + "\n"
    )
    config = {
        "task": "piqa",
        "output_type": "multiple_choice",
        "validation_split": "evaluation",
        "doc_to_text": "{{goal}}",
        "doc_to_target": "label",
        "doc_to_choice": "{{[sol1, sol2]}}",
        "num_fewshot": 0,
        "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True} for m in ("acc", "acc_norm")],
    }
    manifest = {
        "tokenizer_fingerprint": "test",
        "context_length": 4,
        "harness_version": quality.HARNESS_VERSION,
        "tasks": [
            {
                "family": "piqa",
                "name": "piqa",
                "file": path.name,
                "sha256": file_digest(path),
                "config": config,
                "count": 1,
            }
        ],
    }
    manifest["identity"] = fingerprint(manifest)
    atomic_json(suite / "manifest.json", manifest)
    output = tmp_path / "quality"
    first = quality.benchmark_quality(
        tiny_config, checkpoint, output, suite=suite, tasks=["piqa"], device="cpu", batch_size=1, limit=1
    )
    assert first["complete"] is False and first["metrics"]["piqa"]["count"] == 1
    monkeypatch.setattr(quality, "score_token_pairs", lambda *a: pytest.fail("must reuse completed task"))
    second = quality.benchmark_quality(
        tiny_config, checkpoint, output, suite=suite, tasks=["piqa"], device="cpu", batch_size=2, limit=1
    )
    assert first["identity"] == second["identity"]
    (output / "piqa.samples.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="inconsistent"):
        quality.benchmark_quality(tiny_config, checkpoint, output, suite=suite, tasks=["piqa"], device="cpu", limit=1)
