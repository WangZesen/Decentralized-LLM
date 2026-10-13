import json
import os
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from decentralized_llm import cache as cache_module
from decentralized_llm.cache import TokenCache, _tokenized_batches, prepare
from decentralized_llm.config import Config, save_config
from decentralized_llm.launch import launch, local_command, slurm_scripts
from decentralized_llm.preflight import validate


class FakeTokenizer:
    eos_token_id = 2

    def __len__(self) -> int:
        return 17

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool = False, return_attention_mask: bool = False
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [[int(text), 3] for text in texts]}

    @classmethod
    def from_pretrained(cls, *_: Any, **__: Any) -> FakeTokenizer:
        return cls()

    def save_pretrained(self, path: Path) -> None:
        path.mkdir()

    def get_vocab(self) -> dict[str, int]:
        return {str(i): i for i in range(17)}

    @property
    def backend_tokenizer(self) -> FakeTokenizer:
        return self

    def to_str(self) -> str:
        return "fake-tokenizer"


class FakeApi:
    def dataset_info(self, *_: Any, **__: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(sha="a" * 40)

    model_info = dataset_info

    def list_repo_files(self, *_: Any, **__: Any) -> list[str]:
        return ["en/c4-train.0.json.gz", "en/c4-validation.0.json.gz"]


class FakeStream:
    def shuffle(self, **_: Any) -> FakeStream:
        return self

    def iter(self, batch_size: int) -> Iterator[dict[str, list[str]]]:
        for i in range(40):
            yield {"text": [str(i % 17)]}


def test_preparation_is_deterministic_and_atomic(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined] # Test double for lazy optional imports.
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeApi  # type: ignore[attr-defined]
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *a, **k: FakeStream()  # type: ignore[attr-defined]
    for name, module in (("transformers", transformers), ("huggingface_hub", hub), ("datasets", datasets)):
        monkeypatch.setitem(sys.modules, name, module)
    cfg = tiny_config.model_copy(deep=True)
    cfg.data.prepare_workers = 1
    cfg.data.prepare_train_tokens = 31
    cfg.data.prepare_validation_tokens = 14
    cfg.data.shard_tokens = 7
    first = prepare(cfg)
    assert first["splits"]["train"]["tokens"] == 33
    assert prepare(cfg) == first
    second_cfg = cfg.model_copy(deep=True)
    second_cfg.data.cache_dir = tmp_path / "second"
    second = prepare(second_cfg)
    assert first == second
    for path in cfg.data.cache_dir.glob("*.bin"):
        assert path.read_bytes() == (second_cfg.data.cache_dir / path.name).read_bytes()
    assert not cfg.data.cache_dir.with_name("cache.partial").exists()
    values = TokenCache(cfg.data.cache_dir).read("train", 0, 9)
    assert values.tolist() == [0, 3, 2, 1, 3, 2, 2, 3, 2]
    # Reuse affects sampling only, including when preparation uses its default budget.
    cfg.data.cache_dir = tmp_path / "default-budget"
    cfg.data.prepare_train_tokens = None
    cfg.training.tokens_per_parameters = 0.1
    cfg.training.data_reuse_factor = 5
    assert prepare(cfg)["splits"]["train"]["tokens"] == 73
    cfg.training.data_reuse_factor = 1
    assert prepare(cfg)["splits"]["train"]["tokens"] == 73


def test_parallel_tokenizer_order(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer()

    def delayed(texts: list[str]) -> np.ndarray:
        time.sleep((5 - int(texts[0])) * 0.002)
        return cache_module._tokenize_documents(tokenizer, texts)

    monkeypatch.setattr(cache_module, "_tokenize_worker", delayed)
    source = [[str(i)] for i in range(6)]
    serial = list(_tokenized_batches(source, tokenizer, None, 1))
    with ThreadPoolExecutor(4) as pool:
        parallel = list(_tokenized_batches(source, tokenizer, pool, 4))
    np.testing.assert_array_equal(np.concatenate(serial), np.concatenate(parallel))


def slurm_config(config: Config, tmp_path: Path) -> Config:
    env = tmp_path / "environment ' quote.sh"
    env.write_text("true\n")
    raw = config.model_dump(mode="json")
    raw["launcher"] = {
        "kind": "slurm",
        "partition": "gpu",
        "environment_script": str(env),
        "cpus_per_task": 8,
        "memory": "4G",
        "gpu_type": "testgpu",
        "qos": "test",
    }
    raw["distributed"].update(nodes=2, gpus_per_node=2)
    return Config.model_validate(raw)


def test_launcher_validation_account_dryrun(
    cache: TokenCache, tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = slurm_config(tiny_config, tmp_path)
    monkeypatch.setattr("decentralized_llm.preflight.shutil.which", lambda exe: f"/bin/{exe}")
    monkeypatch.delenv("account", raising=False)
    with pytest.raises(ValueError, match="lowercase"):
        launch(cfg, dry_run=True)
    monkeypatch.setenv("ACCOUNT", "wrong-case")
    with pytest.raises(ValueError, match="lowercase"):
        launch(cfg, dry_run=True)
    monkeypatch.setenv("account", "")
    with pytest.raises(ValueError):
        launch(cfg, dry_run=True)
    secret = "test-account-do-not-record"
    monkeypatch.setenv("account", secret)
    result = launch(cfg, dry_run=True)
    assert secret not in json.dumps(result)
    assert not cfg.runtime.output_dir.exists()
    batch, worker = slurm_scripts(cfg, tmp_path / "config a.yaml", tmp_path / "worker a.sh")
    assert "--ntasks-per-node=1" in batch and "--gpus-per-task=testgpu:2" in batch
    assert '--node-rank="$SLURM_PROCID"' in worker
    assert "--max-restarts=0" in worker and "--master-port=29500" in worker
    assert "--kill-on-bad-exit=1" in batch
    for script in (batch, worker):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    calls = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "12345;cluster\n", "")

    monkeypatch.setattr("decentralized_llm.launch.subprocess.run", fake_run)
    receipt = launch(cfg)
    assert receipt["job_id"] == "12345"
    assert calls[0][calls[0].index("--account") + 1] == secret
    assert all(secret not in path.read_text() for path in cfg.runtime.output_dir.rglob("*") if path.is_file())


def test_single_node_slurm_uses_an_available_rendezvous_port(tiny_config: Config, tmp_path: Path) -> None:
    cfg = slurm_config(tiny_config, tmp_path)
    cfg.distributed.nodes = 1
    _, worker = slurm_scripts(cfg, tmp_path / "config.yaml", tmp_path / "worker.sh")
    assert "--standalone" in worker and "--master-port" not in worker
    subprocess.run(["bash", "-n"], input=worker, text=True, check=True)


def test_local_command_failure_and_gpu_preflight(
    cache: TokenCache, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = local_command(tiny_config, Path("configuration with spaces.yaml"))
    assert command[0] == sys.executable and "--standalone" in command and "--max-restarts=0" in command
    gpu = tiny_config.model_copy(deep=True)
    gpu.runtime.device = "cuda"
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    with pytest.raises(ValueError, match="GPUs"):
        validate(gpu, launch=True)

    def fail(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(7, args)

    monkeypatch.setattr("decentralized_llm.launch.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        launch(tiny_config)
    assert error.value.returncode == 7


def test_cli_local_end_to_end(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    save_config(tiny_config, config_path)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, "-m", "decentralized_llm", "train", "-c", str(config_path)],
        check=True,
        timeout=90,
        env=env,
        capture_output=True,
    )
    assert (tiny_config.runtime.output_dir / "final/manifest.json").is_file()


def test_real_spawn_tokenizer_workers(tmp_path: Path) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "one": 2, "two": 3, "three": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(tmp_path)
    batches = [["one two", "three"], ["two"], ["one three two"]] * 4
    # Transformers exposes EOS dynamically and its BatchEncoding return stub is broader than this protocol.
    document_tokenizer = cast(cache_module._DocumentTokenizer, tokenizer)
    serial = list(_tokenized_batches(batches, document_tokenizer, None, 1))
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=cache_module._initialize_tokenizer,
        initargs=(str(tmp_path),),
    ) as pool:
        parallel = list(_tokenized_batches(batches, document_tokenizer, pool, 2))
    np.testing.assert_array_equal(np.concatenate(serial), np.concatenate(parallel))
    assert parallel[0].tolist() == [2, 3, 1, 4, 1]
