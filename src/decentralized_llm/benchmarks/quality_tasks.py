"""Pinned harness tasks, immutable tokenized suites, and resumable quality reports."""

import fcntl
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import datasets
import lm_eval
import numpy as np
from huggingface_hub import HfApi
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml

from decentralized_llm.benchmarks.quality import (
    TokenPair,
    load_quality_model,
    load_tokenizer,
    score_token_pairs,
    validate_pair,
)
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache, file_digest, fingerprint
from decentralized_llm.models.llama import Llama
from decentralized_llm.runtime import atomic_json

HARNESS_VERSION = "0.4.13"
TASKS = ("blimp", "piqa", "hellaswag")
DATASETS = {"blimp": "nyu-mll/blimp", "piqa": "baber/piqa", "hellaswag": "Rowan/hellaswag"}


def check_harness() -> None:
    if importlib.metadata.version("lm-eval") != HARNESS_VERSION:
        raise ValueError(f"quality evaluation requires lm-eval=={HARNESS_VERSION}")


class NativeLM(TemplateLM):
    """Native model adapter; generation is deliberately unsupported."""

    def __init__(self, model: Llama | None, tokenizer: Any, batch_size: int = 32):
        super().__init__()
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs: Any) -> list[int]:
        if add_special_tokens:
            raise ValueError("quality evaluation never inserts automatic special tokens")
        return self.tokenizer.encode(string, add_special_tokens=False)

    def encode_pair(self, context: str, continuation: str) -> TokenPair:
        if context:
            return self._encode_pair(context, continuation)
        encoded = self.tok_encode(continuation)
        if encoded and encoded[0] == self.eot_token_id:
            return encoded[:1], encoded[1:]
        return [self.eot_token_id], encoded

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        if self.model is None:
            raise ValueError("this adapter is configured only for tokenization")
        return score_token_pairs(self.model, [(c, t) for _, c, t in requests], self.batch_size)

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("quality benchmarks use explicit candidate likelihoods")

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("quality benchmarks do not generate answers")


class PreparedTask(ConfigurableTask):
    """Use the harness's metric implementation with already processed local documents."""

    def __init__(self, config: dict, docs: list[dict]):
        self.prepared_docs = docs
        super().__init__(config=config)

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs: Any):
        self.dataset = datasets.DatasetDict({"evaluation": datasets.Dataset.from_list(self.prepared_docs)})


def verify_suite(path: Path, config: Config, tasks: list[str] | tuple[str, ...] = TASKS) -> dict:
    check_harness()
    manifest = json.loads((path / "manifest.json").read_text())
    identity = manifest.pop("identity")
    if identity != fingerprint(manifest):
        raise ValueError("benchmark suite identity mismatch")
    manifest["identity"] = identity
    cache = TokenCache(config.data.cache_dir)
    if (
        manifest["tokenizer_fingerprint"] != cache.manifest["tokenizer_fingerprint"]
        or manifest["context_length"] != config.model.context_length
        or manifest["harness_version"] != HARNESS_VERSION
    ):
        raise ValueError("benchmark suite tokenizer/context/harness mismatch")
    if not set(tasks) <= {t["family"] for t in manifest["tasks"]}:
        raise ValueError("benchmark suite does not contain the requested tasks")
    for task in manifest["tasks"]:
        if file_digest(path / task["file"]) != task["sha256"]:
            raise ValueError(f"benchmark task changed: {task['name']}")
    return manifest


def prepare_suite(path: Path, config: Config, tasks: list[str] | tuple[str, ...] = TASKS) -> dict:
    """Resolve dataset revisions before reading; publish a checksummed manifest last."""
    check_harness()
    if not tasks or not set(tasks) <= set(TASKS):
        raise ValueError("select at least one supported quality benchmark")
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (path / "manifest.json").exists():
            return verify_suite(path, config, tasks)
        cache = TokenCache(config.data.cache_dir)
        adapter = NativeLM(None, load_tokenizer(cache))
        assert lm_eval.__file__ is not None
        task_root = Path(lm_eval.__file__).parent / "tasks"
        revisions = {family: HfApi().dataset_info(DATASETS[family]).sha for family in tasks}
        if not all(revisions.values()):
            raise ValueError("could not resolve benchmark dataset revisions")
        manifest: dict[str, Any] = {
            "version": 1,
            "harness_version": HARNESS_VERSION,
            "num_fewshot": 0,
            "tokenizer_fingerprint": cache.manifest["tokenizer_fingerprint"],
            "tokenizer_revision": cache.manifest["tokenizer_revision"],
            "context_length": config.model.context_length,
            "datasets": {k: {"repository": DATASETS[k], "revision": revisions[k]} for k in tasks},
            "tasks": [],
        }
        for family in tasks:
            paths = (
                sorted(p for p in (task_root / "blimp").glob("*.yaml") if not p.name.startswith("_"))
                if family == "blimp"
                else [task_root / family / f"{family}.yaml"]
            )
            for source in paths:
                task_config = load_yaml(source)
                task_config["dataset_kwargs"] = {"revision": revisions[family]}
                task_config["num_fewshot"] = 0
                task = ConfigurableTask(config=task_config)
                name = task_config["task"]
                filename = f"{name}.jsonl"
                temporary = path / f"{filename}.partial"
                count = truncations = max_tokens = max_candidate = 0
                with temporary.open("w") as handle:
                    for index, raw_doc in enumerate(task.eval_docs):
                        doc = cast(dict[str, Any], raw_doc)
                        context = task.doc_to_text(doc)
                        if not isinstance(context, str):
                            raise ValueError("quality benchmark prompts must be strings")
                        requests = task.construct_requests(doc, context, metadata=(name, index, 1))
                        if not isinstance(requests, list):
                            raise ValueError("quality benchmark must provide multiple choice requests")
                        pairs = [adapter.encode_pair(*request.args) for request in requests]
                        truncations += sum(validate_pair(pair, config.model.context_length) > 0 for pair in pairs)
                        max_tokens = max(max_tokens, *(len(c) + len(t) for c, t in pairs))
                        max_candidate = max(max_candidate, *(len(t) for _, t in pairs))
                        handle.write(
                            json.dumps(
                                {
                                    "id": index,
                                    "doc": doc,
                                    "requests": [r.args for r in requests],
                                    "tokens": pairs,
                                }
                            )
                            + "\n"
                        )
                        count += 1
                expected = 1000 if family == "blimp" else {"piqa": 1838, "hellaswag": 10042}[family]
                if count != expected:
                    raise ValueError(f"unexpected {name} evaluation size: {count} != {expected}")
                os.replace(temporary, path / filename)
                # Documents already include the harness's preprocessing. Keep its metric/prompt definitions.
                metric_config = {
                    **task_config,
                    "process_docs": None,
                    "dataset_kwargs": {},
                    "training_split": None,
                    "validation_split": "evaluation",
                    "test_split": None,
                }
                manifest["tasks"].append(
                    {
                        "name": name,
                        "family": family,
                        "file": filename,
                        "sha256": file_digest(path / filename),
                        "count": count,
                        "config": metric_config,
                        "prompt_truncations": truncations,
                        "max_total_tokens": max_tokens,
                        "max_candidate_tokens": max_candidate,
                        "definition_sha256": file_digest(source),
                    }
                )
                print(f"Prepared {name}: {count} examples", flush=True)
        if "blimp" in tasks and sum(t["family"] == "blimp" for t in manifest["tasks"]) != 67:
            raise ValueError("BLiMP must contain all 67 subsets")
        manifest["identity"] = fingerprint(manifest)
        atomic_json(path / "manifest.json", manifest)
        return manifest


def benchmark_quality(
    config: Config,
    checkpoint: Path,
    output: Path,
    *,
    suite: Path | None = None,
    tasks: list[str] | tuple[str, ...] = TASKS,
    device: str = "cuda",
    batch_size: int = 32,
    limit: int | None = None,
) -> dict:
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch size and optional sample limit must be positive")
    check_harness()
    output.mkdir(parents=True, exist_ok=True)
    suite = suite or output / "suite"
    manifest = prepare_suite(suite, config, tasks)
    load_tokenizer(TokenCache(config.data.cache_dir))
    model, checkpoint_manifest = load_quality_model(config, checkpoint, device)
    identity = fingerprint(
        {
            "checkpoint": checkpoint_manifest,
            "suite": manifest["identity"],
            "tasks": sorted(set(tasks)),
            "limit": limit,
            "precision": "fp32-ieee",
            "scorer_sha256": file_digest(Path(__file__).with_name("quality.py")),
            "driver_sha256": file_digest(Path(__file__)),
        }
    )
    started = time.perf_counter()
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = output / "identity.json"
        if existing.exists() and json.loads(existing.read_text())["identity"] != identity:
            raise ValueError("output contains a different evaluation; use a new directory")
        atomic_json(existing, {"identity": identity})
        summaries = {}
        for specification in manifest["tasks"]:
            if specification["family"] not in tasks:
                continue
            name = specification["name"]
            result_path = output / f"{name}.json"
            samples_path = output / f"{name}.samples.jsonl"
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous["identity"] == identity and file_digest(samples_path) == previous["samples_sha256"]:
                    summaries[name] = previous
                    continue
                raise ValueError(f"previous task output is inconsistent: {name}")
            with (suite / specification["file"]).open() as handle:
                rows = [json.loads(line) for line in handle]
            if limit is not None:
                rows = rows[:limit]
            task = PreparedTask(specification["config"], [r["doc"] for r in rows])
            pairs = [(c, t) for row in rows for c, t in row["tokens"]]
            responses = score_token_pairs(model, pairs, batch_size)
            metrics: dict[str, list[float]] = {}
            cursor = ties = truncations = 0
            temporary = samples_path.with_suffix(".partial")
            with temporary.open("w") as handle:
                for row in rows:
                    count = len(row["tokens"])
                    values = responses[cursor : cursor + count]
                    cursor += count
                    scores = [value[0] for value in values]
                    result = task.process_results(row["doc"], values)
                    for key, value in result.items():
                        metrics.setdefault(key, []).append(float(value))
                    truncated = [validate_pair(pair, config.model.context_length) for pair in row["tokens"]]
                    truncations += sum(t > 0 for t in truncated)
                    ties += sum(s == max(scores) for s in scores) > 1
                    choices = task.doc_to_choice(row["doc"])
                    handle.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "requests": row["requests"],
                                "target": task.doc_to_target(row["doc"]),
                                "loglikelihoods": scores,
                                "character_normalized": [s / len(c) for s, c in zip(scores, choices, strict=True)],
                                "candidate_tokens": [len(t) for _, t in row["tokens"]],
                                "truncated_prompt_tokens": truncated,
                                "metrics": result,
                            }
                        )
                        + "\n"
                    )
            os.replace(temporary, samples_path)
            summary = {
                "identity": identity,
                "family": specification["family"],
                "count": len(rows),
                "metrics": {key: float(task.aggregation()[key](values)) for key, values in metrics.items()},
                "ties": ties,
                "prompt_truncations": truncations,
                "samples_sha256": file_digest(samples_path),
            }
            atomic_json(result_path, summary)
            summaries[name] = summary
            print(f"Evaluated {name}: {summary['metrics']}", flush=True)
        aggregate = {}
        if "blimp" in tasks:
            aggregate["blimp"] = {
                "acc": float(
                    np.mean([value["metrics"]["acc"] for value in summaries.values() if value["family"] == "blimp"])
                ),
                "count": sum(v["count"] for v in summaries.values() if v["family"] == "blimp"),
            }
        for family in ("piqa", "hellaswag"):
            if family in tasks:
                aggregate[family] = {**summaries[family]["metrics"], "count": summaries[family]["count"]}
        result = {
            "version": 1,
            "identity": identity,
            "complete": limit is None,
            "limit_per_task": limit,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest": checkpoint_manifest,
            "suite_identity": manifest["identity"],
            "harness_version": HARNESS_VERSION,
            "device": device,
            "batch_size": batch_size,
            "precision": "fp32-ieee",
            "num_fewshot": 0,
            "metrics": aggregate,
            "tasks": summaries,
            "seconds": time.perf_counter() - started,
        }
        atomic_json(output / "results.json", result)
        return result
