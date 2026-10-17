import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from decentralized_llm import cli
from decentralized_llm import config as config_module
from decentralized_llm.config import Config, initialize_local_defaults, load_config, save_config
from decentralized_llm.execution import resolve_execution
from decentralized_llm.tuning import local_runner
from decentralized_llm.tuning.spec import read_spec


def write_local(text: str) -> Path:
    path = config_module.LOCAL_DEFAULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_default_creation_and_preservation(capsys: pytest.CaptureFixture[str]) -> None:
    path = config_module.LOCAL_DEFAULTS_PATH
    assert load_config().runtime.execution_backend == "auto"
    config = load_config(use_local_defaults=True)
    assert not path.parent.exists()  # The low-level reader never initializes files.
    initialize_local_defaults()
    assert load_config(use_local_defaults=True) == config
    assert config.runtime.execution_backend == "native" and config.runtime.cpu_affinity == "none"
    assert resolve_execution(config, torch.device("cpu"))["compile"] is False
    assert yaml.safe_load(path.read_text()) == yaml.safe_load(Path("configs/local-defaults.example.yaml").read_text())
    custom = "# Preserve my settings and comments.\nruntime: {execution_backend: a100}\n"
    path.write_text(custom)
    initialize_local_defaults()
    assert path.read_text() == custom
    assert capsys.readouterr().err.count("Created local execution defaults:") == 1


def test_concurrent_initialization(capsys: pytest.CaptureFixture[str]) -> None:
    path = config_module.LOCAL_DEFAULTS_PATH

    def start(_: int) -> Config:
        initialize_local_defaults()
        return load_config(use_local_defaults=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        configs = list(pool.map(start, range(32)))
    assert all(config == configs[0] for config in configs)
    assert configs[0].runtime.execution_backend == "native"
    assert list(path.parent.iterdir()) == [path]
    assert capsys.readouterr().err.count("Created local execution defaults:") == 1


def test_initialization_failure_has_path_and_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("read-only directory")

    monkeypatch.setattr(os, "link", denied)
    with pytest.raises(ValueError, match=r"defaults.yaml.*read-only directory.*--no-local-config"):
        initialize_local_defaults()
    assert not list(config_module.LOCAL_DEFAULTS_PATH.parent.iterdir())


@pytest.mark.parametrize(
    "text",
    [
        "[native]",
        "runtime: [native]",
        "runtime: [",
        "training: {batch_tokens: 1024}",
        "runtime: {seed: 7}",
        "runtime: {output_dir: runs/local}",
        "runtime: {unknown: true}",
        "runtime: {execution_backend: unknown}",
        "runtime: {compile: sometimes}",
        "runtime: {cpu_threads: 0}",
    ],
)
def test_invalid_local_defaults_are_not_replaced_or_masked(text: str) -> None:
    path = write_local(text)
    initialize_local_defaults()
    assert path.read_text() == text
    with pytest.raises(ValueError, match="Invalid local execution defaults at .*defaults.yaml"):
        load_config(overrides=["runtime.execution_backend=native"], use_local_defaults=True)
    assert load_config() == Config()


@pytest.mark.parametrize("text", ["", "# no defaults\n", "{}", "runtime: {}"])
def test_empty_existing_file_provides_no_defaults(text: str) -> None:
    path = write_local(text)
    initialize_local_defaults()
    assert load_config(use_local_defaults=True) == Config()
    assert path.read_text() == text


def test_precedence_and_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "LOCAL_DEFAULTS_PATH", Path("configs/local/defaults.yaml"))
    write_local("runtime: {execution_backend: gh200, compile: true, cpu_threads: 2, compile_cache_dir: custom-cache}")
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("runtime: {execution_backend: a100, compile: auto}\n")
    second.write_text("runtime: {cpu_threads: 4}\n")
    config = load_config([first, second], use_local_defaults=True)
    assert config.runtime.execution_backend == "a100" and config.runtime.compile == "auto"
    assert config.runtime.cpu_threads == 4 and config.runtime.compile_cache_dir == Path("custom-cache")
    config = load_config([first, second], ["runtime.execution_backend=native"], use_local_defaults=True)
    assert config.runtime.execution_backend == "native"


@pytest.mark.parametrize("backend", ["gh200", "a100"])
def test_local_backend_enables_fused_policy_and_stays_strict(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    write_local(f"runtime: {{execution_backend: {backend}, compile: auto}}")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "GH200" if backend == "gh200" else "A100")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0) if backend == "gh200" else (8, 0))
    config = load_config(use_local_defaults=True)
    policy = resolve_execution(config, torch.device("cuda", 0))
    assert policy["backend"] == backend and policy["compile"] is True
    config.runtime.compile = False
    with pytest.raises(ValueError, match="requires compilation"):
        resolve_execution(config, torch.device("cuda", 0))
    with pytest.raises(ValueError, match="execution requires"):
        resolve_execution(config, torch.device("cpu"))


@pytest.mark.parametrize("command", ["train", "submit", "benchmark-training"])
@pytest.mark.parametrize("mode", ["normal", "dry-run", "no-local-config"])
def test_cli_initialization_and_opt_out(
    command: str, mode: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def launched(config: Config, *_args: Any, **_kwargs: Any) -> dict:
        return config.model_dump(mode="json")

    monkeypatch.setattr("decentralized_llm.launch.launch", launched)
    arguments = ["decentralized-llm", command]
    if command == "submit":
        arguments += ["--set", "launcher.kind=slurm", "--set", "launcher.partition=gpu"]
    if mode != "normal":
        arguments.append(f"--{mode}")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    output = capsys.readouterr()
    config = json.loads(output.out)
    assert config["runtime"]["execution_backend"] == ("auto" if mode == "no-local-config" else "native")
    assert config_module.LOCAL_DEFAULTS_PATH.exists() is (mode == "normal")
    if mode != "normal":
        assert not config_module.LOCAL_DEFAULTS_PATH.parent.exists()
    if mode == "dry-run":
        assert "no file was created" in output.err and "native" in output.err


@pytest.mark.parametrize("backend", ["gh200", "a100"])
def test_cli_and_tuning_use_local_backend_without_overlay(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_local(f"runtime: {{execution_backend: {backend}}}")
    base = tmp_path / "base.yaml"
    base.write_text("runtime: {compile: auto}\n")
    monkeypatch.setattr(
        "decentralized_llm.launch.launch", lambda config, *args, **kwargs: config.model_dump(mode="json")
    )
    monkeypatch.setattr(sys, "argv", ["decentralized-llm", "train", "-c", str(base), "--dry-run"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["runtime"]["execution_backend"] == backend
    spec: dict[str, Any] = {
        "base": {"configs": ["base.yaml"]},
        "screening_seeds": 1,
        "phases": [{"name": "lr", "search": {"optimizer.lr": [0.001]}, "top_k": 1}],
        "confirmation": {"seeds": 1},
    }
    filename = tmp_path / "tuning.yaml"
    filename.write_text(yaml.safe_dump(spec))
    assert read_spec(filename, use_local_defaults=True)[1].runtime.execution_backend == backend
    spec["base"]["overrides"] = {"runtime.execution_backend": "native"}
    filename.write_text(yaml.safe_dump(spec))
    assert read_spec(filename, use_local_defaults=True)[1].runtime.execution_backend == "native"


@pytest.mark.parametrize("backend", [None, "gh200", "a100"])
@pytest.mark.parametrize(
    "filename",
    [
        "base.yaml",
        "local-ddp.yaml",
        "local-decentralized.yaml",
        "slurm-ddp.yaml",
        "slurm-decentralized.yaml",
        "tuning-example.yaml",
        "tuning-sync-512c-32bs-20tpp.yaml",
        "tuning-sync-512c-64bs-20tpp.yaml",
        "tuning-awc-8w-ring-512c-64bs-20tpp.yaml",
        "tuning-awc-ef-8w-ring-512c-64bs-20tpp.yaml",
    ],
)
def test_shared_recipes_use_local_backend_without_site_dependencies(
    filename: str, backend: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    # No checkout-local files or compiler settings are available in this working directory.
    monkeypatch.chdir(tmp_path)
    if backend is not None:
        write_local(f"runtime: {{execution_backend: {backend}}}")
    if filename.startswith("tuning-"):
        spec, config = read_spec(configs / filename, use_local_defaults=True)
        assert all(path.parent == configs for path in spec.base.configs)
        assert {path.name for path in spec.base.configs} <= {
            "base.yaml",
            "20m.yaml",
            "slurm-ddp.yaml",
            "slurm-decentralized.yaml",
        }
    else:
        paths = [configs / "base.yaml", configs / "20m.yaml"]
        if filename != "base.yaml":
            paths.append(configs / filename)
        config = load_config(paths, use_local_defaults=True)
    assert config.runtime.execution_backend == (backend or "native")
    assert config.runtime.compile == "auto"
    assert config.runtime.cpu_affinity == "none"
    if config.launcher.kind == "slurm":
        assert config.launcher.network is None
        assert config.launcher.environment_overlays == []
    assert config_module.LOCAL_DEFAULTS_PATH.exists() is (backend is not None)


@pytest.mark.parametrize("backend", ["gh200", "a100"])
@pytest.mark.parametrize("filename", ["smoke.yaml", "benchmark-cpu.yaml"])
def test_portable_smoke_recipes_override_fused_local_defaults(filename: str, backend: str) -> None:
    write_local(f"runtime: {{execution_backend: {backend}, compile: true, cpu_affinity: exclusive}}")
    config = load_config(["configs/base.yaml", f"configs/{filename}"], use_local_defaults=True)
    policy = resolve_execution(config, torch.device(config.runtime.device))
    assert policy["backend"] == "native" and policy["compile"] is False
    assert config.runtime.cpu_affinity == "none"


def test_cli_opt_out_ignores_invalid_existing_defaults(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_local("runtime: [broken")
    monkeypatch.setattr("decentralized_llm.launch.launch", lambda config, *args, **kwargs: {})
    monkeypatch.setattr(sys, "argv", ["decentralized-llm", "train", "--no-local-config"])
    cli.main()
    assert path.read_text() == "runtime: [broken"
    assert not capsys.readouterr().err


def test_saved_worker_and_local_runner_ignore_corrupt_defaults(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "attempt"
    directory.mkdir()
    filename = directory / "config.yaml"
    save_config(tiny_config, filename)
    path = write_local("runtime: [broken")
    original = path.read_bytes()
    monkeypatch.setattr("decentralized_llm.train.train", lambda config, checkpoint: config.model_dump(mode="json"))
    monkeypatch.setattr(sys, "argv", ["decentralized-llm", "worker", "-c", str(filename)])
    cli.main()
    assert Config.model_validate(json.loads(capsys.readouterr().out)) == tiny_config
    captured: list[Config] = []
    monkeypatch.setattr(local_runner, "launch", lambda config, **kwargs: captured.append(config))
    local_runner.run(directory)
    assert captured == [tiny_config]
    assert json.loads((directory / "exit.json").read_text())["exit_code"] == 0
    assert path.read_bytes() == original
