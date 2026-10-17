"""Exercise Bash activation in isolated environments, without site modules."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, body: str = "exit 0") -> None:
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(0o755)


def activate(tmp_path: Path, commands: str = "", settings: str | None = "", arch: str = "x86_64"):
    if settings is not None:
        (tmp_path / ".toolchain-x86_64.sh").write_text(settings)
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            f'set -eu; source "{ROOT}/scripts/toolchain.sh"; {commands}\n'
            f'activate_toolchain "{tmp_path}" {arch}; printf "CC=%s CXX=%s\\n" "${{CC:-}}" "${{CXX:-}}"',
        ],
        env={"PATH": str(tmp_path), "HOME": str(tmp_path)},
        text=True,
        capture_output=True,
    )


def test_defaults_and_repeat(tmp_path: Path):
    for name in ("gcc", "g++"):
        executable(tmp_path / name)
    result = activate(tmp_path, commands=f'activate_toolchain "{tmp_path}" x86_64;')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CC=gcc CXX=g++"


@pytest.mark.parametrize("arch,settings", [("aarch64", "return 9"), ("x86_64", None)])
def test_absent_matching_settings_preserves_environment(tmp_path: Path, arch: str, settings: str | None):
    result = activate(tmp_path, "export CC=existing CXX=existing++;", settings, arch)
    assert result.returncode == 0
    assert "CC=existing CXX=existing++" in result.stdout


def test_precedence_and_skip_modules(tmp_path: Path):
    executable(tmp_path / "override cc")
    executable(tmp_path / "configured++")
    result = activate(
        tmp_path,
        f'export CC="{tmp_path}/override cc" CXX="";',
        "TOOLCHAIN_CC=invalid; TOOLCHAIN_CXX=configured++; TOOLCHAIN_MODULES=(unavailable)",
    )
    assert result.returncode == 0, result.stderr
    assert f"CC={tmp_path}/override cc CXX=configured++" in result.stdout


def test_clang_fallback(tmp_path: Path):
    executable(tmp_path / "gcc", "exit 1")
    for name in ("clang", "clang++"):
        executable(tmp_path / name)
    result = activate(tmp_path)
    assert result.returncode == 0
    assert "CC=clang CXX=clang++" in result.stdout


def test_module_order_and_repeat(tmp_path: Path):
    for name in ("gcc", "g++"):
        executable(tmp_path / name, "[[ ${READY:-} == yes ]]")
    init = tmp_path / "init.sh"
    init.write_text('echo init\nmodule() { echo "$*"; if [[ "$*" == "load second" ]]; then export READY=yes; fi; }\n')
    settings = (
        f'TOOLCHAIN_MODULE_INIT="{init}"; TOOLCHAIN_MODULE_PATHS=(one "two paths"); '
        "TOOLCHAIN_MODULES=(first second); TOOLCHAIN_CC=gcc; TOOLCHAIN_CXX=g++"
    )
    (tmp_path / ".toolchain-x86_64.sh").write_text(settings)
    result = activate(tmp_path, f'activate_toolchain "{tmp_path}" x86_64;', settings)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "init",
        "use one",
        "use two paths",
        "load first",
        "load second",
        "CC=gcc CXX=g++",
    ]


@pytest.mark.parametrize(
    "settings,commands,diagnostic",
    [
        ("TOOLCHAIN_CC=missing", "", "Host compilers unusable"),
        ("TOOLCHAIN_MODULES=(compiler)", "", "Module command unavailable"),
        ("TOOLCHAIN_MODULES=(compiler)", "module() { return 7; };", "Could not load"),
        ("TOOLCHAIN_MODULE_PATHS=(missing)", "module() { return 7; };", "Could not add"),
        ("TOOLCHAIN_MODULE_INIT=/missing; TOOLCHAIN_MODULES=(compiler)", "", "Could not initialize"),
    ],
)
def test_failures(tmp_path: Path, settings: str, commands: str, diagnostic: str):
    result = activate(tmp_path, commands, settings)
    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.fixture
def environment_script(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("environment.sh", "toolchain.sh"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, tmp_path / name)
    arch = os.uname().machine
    python = tmp_path / f".venv-{arch}" / "bin" / "python"
    python.parent.mkdir(parents=True)
    executable(python)
    fingerprint = subprocess.check_output(
        [
            "/bin/bash",
            "-c",
            f'sha256sum "{tmp_path}/uv.lock" "{tmp_path}/pyproject.toml" | sha256sum | cut -d \' \' -f 1',
        ],
        text=True,
    )
    (tmp_path / f".uv-sync-{arch}.fingerprint").write_text(fingerprint)
    settings = tmp_path / f".toolchain-{arch}.sh"
    compiler = tmp_path / "compiler"
    executable(compiler)
    settings.write_text(f'TOOLCHAIN_CC="{compiler}"; TOOLCHAIN_CXX="{compiler}"')
    return scripts / "environment.sh"


def test_environment_cached_activation_and_failure(tmp_path: Path, environment_script: Path):
    compiler = tmp_path / "compiler"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    command = f'source "{environment_script}"; source "{environment_script}"; echo "$CC"'
    result = subprocess.run(["/bin/bash", "-c", command], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(compiler)
    executable(compiler, "exit 1")
    result = subprocess.run(
        ["/bin/bash", "-c", f'if source "{environment_script}"; then exit 99; else exit 42; fi'],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 42


@pytest.mark.parametrize("errexit", [False, True])
@pytest.mark.parametrize("nounset", [False, True])
@pytest.mark.parametrize("pipefail", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_environment_preserves_shell_options(
    tmp_path: Path, environment_script: Path, errexit: bool, nounset: bool, pipefail: bool, cached: bool
):
    arch = os.uname().machine
    if not cached:
        (tmp_path / f".uv-sync-{arch}.fingerprint").unlink()
        executable(tmp_path / "uv")
    options = (
        f"set {'-' if errexit else '+'}e; set {'-' if nounset else '+'}u; set {'-' if pipefail else '+'}o pipefail; "
    )
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            options + 'before=$SHELLOPTS; source "$1"; source "$1"; [[ $SHELLOPTS == "$before" ]]',
            "bash",
            str(environment_script),
        ],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_environment_allows_completion_and_nonzero_commands(tmp_path: Path, environment_script: Path):
    # Completion functions legitimately read unset locals and return nonzero (e.g. 124 to retry).
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'source "$1"; '
            '_completion() { local optional; COMPREPLY=("$optional"); return 124; }; '
            '_completion; false; printf "shell survived\\n"',
            "bash",
            str(environment_script),
        ],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "shell survived\n"


@pytest.mark.parametrize("failure", ["compiler", "fingerprint", "lock", "sync"])
@pytest.mark.parametrize("strict", [False, True])
def test_environment_failures_return_without_changing_shell_options(
    tmp_path: Path, environment_script: Path, failure: str, strict: bool
):
    arch = os.uname().machine
    stamp = tmp_path / f".uv-sync-{arch}.fingerprint"
    stamp.unlink()
    executable(tmp_path / "uv")
    if failure == "compiler":
        executable(tmp_path / "compiler", "exit 7")
    elif failure == "fingerprint":
        (tmp_path / "uv.lock").unlink()
    elif failure == "lock":
        executable(tmp_path / "flock", "exit 7")
    else:
        executable(tmp_path / "uv", '[[ "$1" == --version ]]')
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            ("set -euo pipefail; " if strict else "") + 'before=$SHELLOPTS; if source "$1"; then exit 99; fi; '
            '[[ $SHELLOPTS == "$before" ]] || exit 98; printf "failure caught\\n"',
            "bash",
            str(environment_script),
        ],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "failure caught\n"
    assert not stamp.exists()


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_template_and_architecture(tmp_path: Path, arch: str):
    for name in ("gcc", "g++"):
        executable(tmp_path / name)
    (tmp_path / f".toolchain-{arch}.sh").write_text((ROOT / "configs/toolchain.example.sh").read_text())
    other = "aarch64" if arch == "x86_64" else "x86_64"
    (tmp_path / f".toolchain-{other}.sh").write_text("return 9")
    result = activate(tmp_path, settings=None, arch=arch)
    assert result.returncode == 0, result.stderr
    assert "CC=gcc CXX=g++" in result.stdout
