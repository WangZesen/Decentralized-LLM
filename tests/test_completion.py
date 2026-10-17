"""Exercise completion without reading configurations or starting workloads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from decentralized_llm.cli import build_parser
from decentralized_llm.completion import config_completions

ROOT = Path(__file__).resolve().parents[1]
WORDBREAKS = " \t\n\"'><=;|&(:"


def complete(
    line: str,
    cwd: Path,
    *,
    root: Path | None = None,
    offset: int = 1,
    point: int | None = None,
) -> list[str]:
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("_ARGCOMPLETE") or name == "_ARC_DEBUG":
            del env[name]
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        DECENTRALIZED_COMPLETION_ROOT=str(root or cwd),
        COMP_LINE=line,
        COMP_POINT=str(len(line) if point is None else point),
        COMP_TYPE="9",
        _ARGCOMPLETE=str(offset),
        _ARGCOMPLETE_COMP_WORDBREAKS=WORDBREAKS,
        _ARGCOMPLETE_SHELL="bash",
        _ARGCOMPLETE_SUPPRESS_SPACE="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy; os.dup2(1, 8); runpy.run_module('decentralized_llm.completion', run_name='__main__')",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    return result.stdout.split("\v") if result.stdout else []


def test_parser_keeps_cli_syntax() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["train", "-c", "base.yaml", "--config=other.yaml", "--set", "runtime.compile=true", "--set=model.layers=4"]
    )
    assert args.config == [Path("base.yaml"), Path("other.yaml")]
    assert args.overrides == ["runtime.compile=true", "model.layers=4"]
    args = parser.parse_args(["tune", "extend", "study", "--spec", "spec.yaml", "--dry-run"])
    assert args.directory == Path("study") and args.spec == Path("spec.yaml") and args.dry_run


def test_schema_covers_union_variants_and_excludes_properties() -> None:
    keys = set(config_completions(""))
    assert {
        "model.layers=",
        "training.checkpoint_epochs=",
        "lr_schedule.min_lr_ratio=",
        "lr_schedule.decay_fraction=",
        "launcher.kind=",
        "launcher.partition=",
        "launcher.environment_overlays=",
    } <= keys
    assert (
        not {"model.parameter_count=", "distributed.world_size=", "training.epoch_count=", "micro_batch_size="} & keys
    )
    assert config_completions("runtime.comp") == [
        "runtime.compile=",
        "runtime.compile_cache_dir=",
        "runtime.compile_mode=",
        "runtime.compile_scope=",
    ]


@pytest.mark.parametrize(
    "key, values",
    [
        ("runtime.compile", ["auto", "false", "true"]),
        ("runtime.fused_rope", ["auto", "false", "true"]),
        ("runtime.deterministic", ["false", "true"]),
        ("runtime.execution_backend", ["a100", "auto", "gh200", "native"]),
        ("lr_schedule.name", ["cosine", "wsd"]),
        ("launcher.kind", ["local", "slurm"]),
        ("optimizer.grad_clip", ["null"]),
        ("launcher.memory", ["null"]),
    ],
)
def test_schema_values(key: str, values: list[str]) -> None:
    assert config_completions(f"{key}=") == [f"{key}={value}" for value in values]


@pytest.mark.parametrize("key", ["model.layers", "data.dataset", "training.checkpoint_epochs", "unknown.field"])
def test_schema_does_not_invent_unbounded_values(key: str) -> None:
    assert config_completions(f"{key}=") == []


def test_schema_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work directory").mkdir()
    (tmp_path / "work script.sh").touch()
    assert config_completions("runtime.output_dir=work") == ["runtime.output_dir=work directory/"]
    assert config_completions("launcher.environment_script=work") == [
        "launcher.environment_script=work directory/",
        "launcher.environment_script=work script.sh",
    ]
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_completions("runtime.output_dir=~/work") == ["runtime.output_dir=~/work directory/"]
    assert complete("decentralized-llm tune resume ~/work", tmp_path) == [r"~/work\ directory/"]


@pytest.mark.parametrize("prefix, offset", [("decentralized-llm", 1), ("uv run decentralized-llm", 3)])
def test_commands_options_and_nested_actions(tmp_path: Path, prefix: str, offset: int) -> None:
    assert {"prepare", "train", "submit", "tune"} <= set(complete(f"{prefix} ", tmp_path, offset=offset))
    assert {"run", "resume", "extend", "status", "report"} <= set(complete(f"{prefix} tune ", tmp_path, offset=offset))
    assert complete(f"{prefix} tune resume --retry-", tmp_path, offset=offset) == ["--retry-unresolved"]
    assert complete(f"{prefix} submit --action=e", tmp_path, offset=offset) == ["evaluate"]
    assert complete(f"{prefix} benchmark-data --cache-state c", tmp_path, offset=offset) == ["cold"]
    assert complete(f"{prefix} prepare --res", tmp_path, offset=offset) == []


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("--set runtime.execution_backend=g", ["gh200"]),
        ("--set=runtime.execution_backend=a", ["a100", "auto"]),
        ("--set runtime.compile=", ["auto", "false", "true"]),
        ("--set runtime.compile=false --set distributed.mode=d", ["ddp", "decentralized"]),
        ("--set 'runtime.execution_backend=g", ["runtime.execution_backend=gh200"]),
        ('--set="runtime.execution_backend=g', ["runtime.execution_backend=gh200"]),
        ("--set optimizer.lr=", []),
        ("--steps ", []),
    ],
)
def test_override_insertion_with_bash_wordbreaks(tmp_path: Path, arguments: str, expected: list[str]) -> None:
    assert complete(f"decentralized-llm benchmark-training {arguments}", tmp_path) == expected


def test_path_insertion_handles_spaces_and_quotes(tmp_path: Path) -> None:
    (tmp_path / "experiment one.yaml").touch()
    (tmp_path / "study one").mkdir()
    assert complete("decentralized-llm train -c exp", tmp_path) == [r"experiment\ one.yaml"]
    assert complete("decentralized-llm train --config=exp", tmp_path) == [r"experiment\ one.yaml"]
    assert complete('decentralized-llm train --config "exp', tmp_path) == ["experiment one.yaml"]
    assert complete('decentralized-llm train --config="exp', tmp_path) == ["experiment one.yaml"]
    assert complete("decentralized-llm tune resume stu", tmp_path) == [r"study\ one/"]
    assert complete("decentralized-llm train --set runtime.output_dir=stu", tmp_path) == [r"study\ one/"]
    assert complete('decentralized-llm train --set runtime.output_dir="stu', tmp_path) == ["study one/"]


def test_completion_is_scoped_to_physical_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    child = checkout / "child"
    sibling = tmp_path / "repo-other"
    child.mkdir(parents=True)
    sibling.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(checkout, target_is_directory=True)
    line = "decentralized-llm tr"
    for cwd in (checkout, child, alias):
        assert complete(line, cwd, root=checkout) == ["train"]
    for cwd in (tmp_path, sibling):
        assert complete(line, cwd, root=checkout) == []
    assert complete(line, checkout, root=alias) == ["train"]


def test_completion_ignores_configs_and_never_creates_defaults(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not: [valid yaml")
    config_dir = tmp_path / "configs/local"
    config_dir.mkdir(parents=True)
    before = set(tmp_path.rglob("*"))
    assert complete("decentralized-llm train -c invalid.yaml --set runtime.compile=t", tmp_path) == ["true"]
    assert set(tmp_path.rglob("*")) == before
    defaults = config_dir / "defaults.yaml"
    defaults.write_text("also: [invalid")
    assert complete("decentralized-llm train -c invalid.yaml --set runtime.compile=t", tmp_path) == ["true"]
    assert defaults.read_text() == "also: [invalid"


def test_path_completion_does_not_run_shell_startup_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = tmp_path / "startup.sh"
    marker = tmp_path / "startup-ran"
    startup.write_text('printf "startup hook ran\\n"; : > "$DECENTRALIZED_TEST_SENTINEL"\n')
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("ENV", str(startup))
    monkeypatch.setenv("DECENTRALIZED_TEST_SENTINEL", str(marker))
    (tmp_path / "example.yaml").touch()
    assert complete("decentralized-llm train --config ex", tmp_path) == ["example.yaml"]
    assert not marker.exists()
    assert os.environ["BASH_ENV"] == str(startup)


@pytest.mark.parametrize("configuration", [False, True])
def test_parser_and_completion_keep_imports_light(tmp_path: Path, configuration: bool) -> None:
    # A fresh process avoids the training imports made by the shared pytest fixtures.
    code = """
import io
import json
import sys
import argcomplete
from decentralized_llm import completion
from decentralized_llm.cli import build_parser
build_parser()
original = argcomplete.CompletionFinder.__call__
def capture(self, parser, **kwargs):
    original(self, parser, output_stream=io.StringIO(), exit_method=sys.exit, **kwargs)
argcomplete.CompletionFinder.__call__ = capture
try:
    completion.main()
except SystemExit as error:
    assert error.code == 0
print(json.dumps(sorted(name for name in sys.modules if name in {
    'torch', 'datasets', 'pydantic', 'yaml', 'decentralized_llm.config',
    'decentralized_llm.launch', 'decentralized_llm.train', 'decentralized_llm.tuning.controller'
})))
"""
    env = dict(os.environ)
    line = "decentralized-llm train --set runtime.compile=t" if configuration else "decentralized-llm tr"
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        DECENTRALIZED_COMPLETION_ROOT=str(tmp_path),
        COMP_LINE=line,
        COMP_POINT=str(len(line)),
        _ARGCOMPLETE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    expected = ["decentralized_llm.config", "pydantic", "yaml"] if configuration else []
    assert json.loads(result.stdout) == expected


def test_provider_without_completion_context_is_inert(tmp_path: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), DECENTRALIZED_COMPLETION_ROOT=str(tmp_path))
    env.pop("_ARGCOMPLETE", None)
    result = subprocess.run(
        [sys.executable, "-m", "decentralized_llm.completion"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0 and not result.stdout and not result.stderr
    assert list(tmp_path.iterdir()) == []
