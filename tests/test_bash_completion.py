"""Check repository boundaries, Bash interoperability, and real Readline insertion."""

import os
import pty
import select
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required for shell completion")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/completion.bash", root / "scripts/completion.bash")
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    return root


def shell_environment(root: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root.parent),
        "TERM": "dumb",
        "INPUTRC": "/dev/null",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DECENTRALIZED_PYTHON": sys.executable,
    }


def run_bash(root: Path, body: str, *, cwd: Path | None = None, setup: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            BASH or "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            f'{setup}\nsource "$1"\n{body}',
            "bash",
            str(root / "scripts/completion.bash"),
        ],
        env=shell_environment(root),
        cwd=cwd or root,
        text=True,
        capture_output=True,
        timeout=15,
    )


def completion_context(line: str) -> str:
    # '=' is in Bash's default COMP_WORDBREAKS, unlike ordinary shlex.split().
    lexer = shlex.shlex(line, posix=True, punctuation_chars="=")
    lexer.whitespace_split = True
    words = list(lexer)
    if line.endswith(" "):
        words.append("")
    return (
        f"COMP_LINE={shlex.quote(line)}\nCOMP_POINT={len(line)}\n"
        f"COMP_WORDS=({shlex.join(words)})\nCOMP_CWORD={len(words) - 1}\nCOMP_TYPE=9\nCOMP_KEY=9\n"
    )


def candidates(root: Path, line: str, *, cwd: Path | None = None, setup: str = "") -> list[str]:
    function = "_decentralized_complete_uv" if line.startswith("uv ") else "_decentralized_complete_direct"
    body = completion_context(line) + (
        f'{function} "${{COMP_WORDS[0]}}" "${{COMP_WORDS[COMP_CWORD]}}" '
        '"${COMP_WORDS[COMP_CWORD-1]}"\n'
        'if ((${#COMPREPLY[@]})); then printf "%s\\0" "${COMPREPLY[@]}"; fi'
    )
    result = run_bash(root, body, cwd=cwd, setup=setup)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return result.stdout.rstrip("\0").split("\0") if result.stdout else []


@pytest.mark.parametrize(
    "line,expected",
    [
        ("decentralized-llm tr", {"train"}),
        ("decentralized-llm tune re", {"resume", "report"}),
        ("decentralized-llm submit --action ", {"train", "evaluate", "benchmark-training"}),
        ("decentralized-llm train --set runtime.compile=", {"auto", "true", "false"}),
        ("decentralized-llm train --set=runtime.compile=tr", {"true"}),
        ("decentralized-llm train --set runtime.amp=true --set distributed.mode=", {"ddp", "decentralized"}),
        ("decentralized-llm train --set launcher.cp", {"launcher.cpus_per_task="}),
        ("decentralized-llm train --set lr_schedule.dec", {"lr_schedule.decay_fraction="}),
    ],
)
def test_direct_candidates(checkout: Path, line: str, expected: set[str]) -> None:
    assert set(candidates(checkout, line)) == expected


def test_repeated_options_remain_available(checkout: Path) -> None:
    result = candidates(checkout, "decentralized-llm train --set runtime.amp=true --config first.yaml --")
    assert {"--set", "--config", "--resume", "--dry-run"} <= set(result)
    assert "--checkpoint" not in result


@pytest.mark.parametrize("wrapper", ["uv run", "uv run --locked", "uv run --frozen --offline --no-sync", "uv run --"])
def test_supported_uv_wrappers(checkout: Path, wrapper: str) -> None:
    assert candidates(checkout, f"{wrapper} decentralized-llm train --set runtime.compile=tr") == ["true"]


def test_physical_checkout_boundaries(checkout: Path) -> None:
    descendant = checkout / "nested"
    descendant.mkdir()
    sibling = checkout.with_name(checkout.name + "-sibling")
    sibling.mkdir()
    alias = checkout.parent / "alias"
    alias.symlink_to(checkout, target_is_directory=True)
    outside_alias = checkout / "outside"
    outside_alias.symlink_to(sibling, target_is_directory=True)
    for cwd in (checkout, descendant, alias, alias / "nested"):
        assert candidates(checkout, "decentralized-llm tr", cwd=cwd) == ["train"]
    for cwd in (checkout.parent, sibling, outside_alias):
        assert candidates(checkout, "decentralized-llm tr", cwd=cwd) == []
    # Sourcing through a symlink must bind the same physical checkout.
    assert candidates(alias, "decentralized-llm tr", cwd=checkout) == ["train"]


@pytest.mark.parametrize(
    "line",
    [
        "uv sync --",
        "uv run another-tool --",
        "uv run --project elsewhere decentralized-llm tr",
        "uv --offline run decentralized-llm tr",
    ],
)
def test_existing_uv_completion_receives_unhandled_invocations(checkout: Path, line: str) -> None:
    setup = '_old_uv() { COMPREPLY=("original:$1:$2:$3"); }; complete -o nospace -o default -F _old_uv uv'
    expected_current = shlex.split(line)[-1]
    result = candidates(checkout, line, setup=setup)
    assert len(result) == 1
    assert result[0].startswith(f"original:uv:{expected_current}:")


def test_existing_uv_completion_outside_checkout(checkout: Path) -> None:
    setup = "_old_uv() { COMPREPLY=(outside); }; complete -F _old_uv uv"
    assert candidates(checkout, "uv run decentralized-llm tr", cwd=checkout.parent, setup=setup) == ["outside"]


def test_uv_delegation_preserves_retry_status_and_arguments(checkout: Path) -> None:
    result = run_bash(
        checkout,
        completion_context("uv sync --")
        + '_decentralized_complete_uv uv -- sync; status=$?; printf "%s:%s\\n" "$status" "${COMPREPLY[0]}"',
        setup='_old_uv() { COMPREPLY=("$1|$2|$3"); return 124; }; complete -F _old_uv uv',
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "124:uv|--|sync\n"


def test_uv_function_options_and_repeat_sourcing(checkout: Path) -> None:
    setup = "_old_uv() { COMPREPLY=(original); }; complete -o nospace -o default -o bashdefault -F _old_uv uv"
    result = run_bash(
        checkout,
        'first=$(complete -p uv); source "$1"; [[ $(complete -p uv) == "$first" ]] || exit 7\n'
        "complete -p uv\n"
        + completion_context("uv sync --")
        + '_decentralized_complete_uv uv -- sync\nprintf "%s\\n" "${COMPREPLY[@]}"',
        setup=setup,
    )
    assert result.returncode == 0, result.stderr
    specification, reply = result.stdout.splitlines()
    for option in ("-o nospace", "-o default", "-o bashdefault"):
        assert option in specification
    assert reply == "original"


def test_lazy_uv_completion_is_loaded_before_wrapping(checkout: Path) -> None:
    setup = (
        "_lazy_uv() { COMPREPLY=(lazy); }; "
        "_completion_loader() { [[ $1 == uv ]] || return 1; "
        "complete -o nospace -F _lazy_uv uv; return 124; }"
    )
    assert candidates(checkout, "uv sync --", setup=setup) == ["lazy"]
    assert candidates(checkout, "uv run decentralized-llm tr", setup=setup) == ["train"]


@pytest.mark.parametrize(
    "specification",
    ["complete -W 'one two' uv", "complete -C /bin/true uv", "complete -W 'one two' -F _old_uv uv"],
)
def test_unusual_uv_completion_specs_are_untouched(checkout: Path, specification: str) -> None:
    result = run_bash(
        checkout,
        '[[ $(complete -p uv) == "$before" ]] && complete -p decentralized-llm',
        setup=specification + "; before=$(complete -p uv)",
    )
    assert result.returncode == 0, result.stderr
    assert "_decentralized_complete_direct" in result.stdout


@pytest.mark.parametrize("errexit", [False, True])
@pytest.mark.parametrize("nounset", [False, True])
@pytest.mark.parametrize("pipefail", [False, True])
def test_sourcing_preserves_shell_settings(checkout: Path, errexit: bool, nounset: bool, pipefail: bool) -> None:
    setup = (
        f"set {'-' if errexit else '+'}e; set {'-' if nounset else '+'}u; "
        f"set {'-' if pipefail else '+'}o pipefail; "
        "shopt -s nullglob; before=$SHELLOPTS; shopts=$BASHOPTS; words=$COMP_WORDBREAKS"
    )
    result = run_bash(
        checkout,
        'source "$1"; [[ $SHELLOPTS == "$before" && $BASHOPTS == "$shopts" && $COMP_WORDBREAKS == "$words" ]]',
        setup=setup,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


def test_missing_environment_is_silent_and_never_runs_uv(checkout: Path) -> None:
    setup = 'unset DECENTRALIZED_PYTHON; uv() { printf "unexpected uv call\\n" >&2; return 99; }'
    assert candidates(checkout, "decentralized-llm tr", setup=setup) == []
    assert candidates(checkout, "uv run decentralized-llm tr", setup=setup) == []
    assert not (checkout / "configs").exists()
    assert not list(checkout.glob(".uv-sync*"))


def test_missing_provider_dependencies_are_silent(checkout: Path) -> None:
    python = checkout / "unavailable-python"
    python.write_text('#!/bin/bash\nprintf "missing argcomplete\\n" >&2\nexit 1\n')
    python.chmod(0o755)
    result = run_bash(
        checkout,
        completion_context("decentralized-llm tr")
        + "_decentralized_complete_direct decentralized-llm tr decentralized-llm\n"
        "_decentralized_complete_direct decentralized-llm tr decentralized-llm\n"
        "[[ ${#COMPREPLY[@]} == 0 ]]",
        setup=f"export DECENTRALIZED_PYTHON={shlex.quote(str(python))}",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


def readline_line(root: Path, keystrokes: str) -> str:
    """Complete with an actual interactive Bash, then capture without executing the line."""
    master, slave = pty.openpty()
    output = root.parent / "readline-result"
    environment = shell_environment(root) | {"PS1": "completion-test> ", "CAPTURE_FILE": str(output)}
    process = subprocess.Popen(
        [BASH or "/bin/bash", "--noprofile", "--norc", "-i"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=environment,
        cwd=root,
        start_new_session=True,
    )
    os.close(slave)
    transcript = bytearray()

    def wait_for(marker: bytes) -> None:
        deadline = time.monotonic() + 15
        received = bytearray()
        while marker not in received:
            remaining = deadline - time.monotonic()
            assert remaining > 0, transcript.decode(errors="replace")
            if select.select([master], [], [], remaining)[0]:
                chunk = os.read(master, 65536)
                assert chunk, transcript.decode(errors="replace")
                transcript.extend(chunk)
                received.extend(chunk)

    try:
        wait_for(b"completion-test> ")
        initialization = (
            f"source {shlex.quote(str(root / 'scripts/completion.bash'))}; "
            '_capture() { printf "%s" "$READLINE_LINE" > "$CAPTURE_FILE"; '
            'printf "\\nCAPTURE%s\\n" DONE; READLINE_LINE=; READLINE_POINT=0; }; '
            'bind -x \'"\\C-x\\C-g":_capture\'; printf "\\nREADY%s\\n" NOW\n'
        )
        os.write(master, initialization.encode())
        wait_for(b"\r\nREADYNOW\r\n")
        os.write(master, keystrokes.encode() + b"\x18\x07")
        wait_for(b"\r\nCAPTUREDONE\r\n")
        return output.read_text()
    finally:
        try:
            # Interactive Bash ignores SIGTERM; exit through Readline to avoid
            # a three-second termination timeout for every completed example.
            os.write(master, b"\x15exit\n")
            process.wait(timeout=3)
        except OSError, subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        os.close(master)


@pytest.mark.parametrize(
    "keystrokes,expected",
    [
        ("decentralized-llm tr\t", "decentralized-llm train "),
        ("decentralized-llm train --set runtime.determin\t", "decentralized-llm train --set runtime.deterministic="),
        (
            "decentralized-llm train --set runtime.determin\ttr\t",
            "decentralized-llm train --set runtime.deterministic=true ",
        ),
        ("decentralized-llm train --set=runtime.compile=tr\t", "decentralized-llm train --set=runtime.compile=true "),
        (
            'decentralized-llm train --set="runtime.execution_backend=g\t',
            'decentralized-llm train --set="runtime.execution_backend=gh200" ',
        ),
        (
            'decentralized-llm train --set "runtime.execution_backend=g\t',
            'decentralized-llm train --set "runtime.execution_backend=gh200" ',
        ),
        (
            "uv run --locked -- decentralized-llm train --set runtime.compile=tr\t",
            "uv run --locked -- decentralized-llm train --set runtime.compile=true ",
        ),
    ],
)
def test_readline_insertion(checkout: Path, keystrokes: str, expected: str) -> None:
    assert readline_line(checkout, keystrokes) == expected
    assert not (checkout / "configs").exists()


@pytest.mark.parametrize("quote", ["", "'", '"'])
@pytest.mark.parametrize("attached", [False, True])
def test_readline_paths_with_spaces_and_unfinished_quotes(checkout: Path, quote: str, attached: bool) -> None:
    (checkout / "experiment settings.yaml").touch()
    separator = "=" if attached else " "
    completed = readline_line(checkout, f"decentralized-llm train --config{separator}{quote}exper\t")
    expected = ["--config=experiment settings.yaml"] if attached else ["--config", "experiment settings.yaml"]
    assert shlex.split(completed) == ["decentralized-llm", "train", *expected]
    assert completed.endswith(" ")


def test_readline_config_path_value(checkout: Path) -> None:
    (checkout / "cache directory").mkdir()
    completed = readline_line(checkout, "decentralized-llm train --set data.cache_dir=cach\t")
    assert shlex.split(completed) == ["decentralized-llm", "train", "--set", "data.cache_dir=cache directory/"]
    assert not completed.endswith(" ")
