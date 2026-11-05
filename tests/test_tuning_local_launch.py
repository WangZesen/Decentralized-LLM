"""Local GPU isolation and durable supervisor records without starting CUDA jobs."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from decentralized_llm.config import Config, save_config
from decentralized_llm.launch.slurm import SubmissionRejected
from decentralized_llm.tuning import jobs, local_runner


def cuda_config(config: Config, *, packed: bool = False, workers: int = 2) -> Config:
    raw = config.model_dump(mode="json")
    raw["runtime"].update(device="cuda", training_backend="packed" if packed else "distributed")
    raw["distributed"]["gpus_per_node"] = workers
    raw["training"]["batch_tokens"] = config.model.context_length * workers * config.training.num_accumulation
    return Config.model_validate(raw)


@pytest.mark.parametrize("packed,workers,gpu_ids", [(False, 2, ["GPU-second", "GPU-first"]), (True, 8, ["GPU-second"])])
def test_local_submit_isolates_assigned_gpus(
    tiny_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    packed: bool,
    workers: int,
    gpu_ids: list[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.setenv("LOCAL_LAUNCH_TEST", "preserved")
    captured: dict[str, Any] = {}
    process = SimpleNamespace(pid=12345)

    def spawn(command: list[str], **kwargs: Any) -> Any:
        captured.update(command=command, **kwargs)
        return process

    monkeypatch.setattr(jobs.subprocess, "Popen", spawn)
    monkeypatch.setattr(jobs, "process_identity", lambda _: "test-birth")
    config = cuda_config(tiny_config, packed=packed, workers=workers)
    backend = jobs.Backend()
    receipt = backend.submit(config, {"name": "trial", "directory": str(tmp_path), "local_gpu_ids": gpu_ids})
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(gpu_ids)
    assert captured["env"]["LOCAL_LAUNCH_TEST"] == "preserved"
    assert captured["env"] is not os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,0"
    assert captured["start_new_session"] is True
    assert captured["stderr"] == subprocess.STDOUT
    assert captured["command"][-2:] == ["decentralized_llm.tuning.local_runner", str(tmp_path)]
    assert backend.children["trial"] is process
    assert receipt["pid"] == 12345 and receipt["birth"] == "test-birth"


@pytest.mark.parametrize("device,assignment", [("cuda", None), ("cpu", None), ("cpu", [])])
@pytest.mark.parametrize("mask", [None, "GPU-restricted"])
def test_legacy_and_cpu_submissions_preserve_visibility(
    tiny_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    assignment: list[str] | None,
    mask: str | None,
) -> None:
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    captured: dict[str, Any] = {}

    def spawn(command: list[str], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(jobs.subprocess, "Popen", spawn)
    monkeypatch.setattr(jobs, "process_identity", lambda _: "test-birth")
    config = cuda_config(tiny_config) if device == "cuda" else tiny_config
    attempt: dict[str, Any] = {"name": "trial", "directory": str(tmp_path)}
    if assignment is not None:
        attempt["local_gpu_ids"] = assignment
    jobs.Backend().submit(config, attempt)
    assert captured["env"].get("CUDA_VISIBLE_DEVICES") == mask


@pytest.mark.parametrize(
    "gpu_ids",
    [None, "GPU-a,GPU-b", [], ["GPU-a"], ["GPU-a", "GPU-a"], ["GPU-a", ""], ["GPU-a", 1], ["GPU-a", "GPU-b,GPU-c"]],
)
def test_invalid_assignments_fail_before_spawn(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu_ids: Any
) -> None:
    def unexpected_spawn(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid GPU assignments must not launch a process")

    monkeypatch.setattr(jobs.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(ValueError, match="unique UUID per physical GPU"):
        jobs.Backend().submit(
            cuda_config(tiny_config), {"name": "trial", "directory": str(tmp_path), "local_gpu_ids": gpu_ids}
        )
    assert not (tmp_path / "config.yaml").exists()


def test_definitive_spawn_failure_is_rejected(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_spawn(*args: Any, **kwargs: Any) -> Any:
        raise OSError("cannot execute runner")

    monkeypatch.setattr(jobs.subprocess, "Popen", fail_spawn)
    backend = jobs.Backend()
    with pytest.raises(SubmissionRejected, match="local runner could not start"):
        backend.submit(tiny_config, {"name": "trial", "directory": str(tmp_path)})
    assert not backend.children


def test_interrupted_spawn_is_not_reported_as_rejection(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt_spawn(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(jobs.subprocess, "Popen", interrupt_spawn)
    with pytest.raises(KeyboardInterrupt):
        jobs.Backend().submit(tiny_config, {"name": "trial", "directory": str(tmp_path)})


@pytest.mark.parametrize("fails", [False, True])
def test_runner_persists_visibility_and_exit_status(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    save_config(tiny_config, tmp_path / "config.yaml")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")

    def launch(config: Config, **kwargs: Any) -> None:
        metadata = json.loads((tmp_path / "process.json").read_text())
        assert metadata["pid"] == os.getpid()
        assert metadata["cuda_visible_devices"] == "GPU-selected"
        assert metadata["birth"] == jobs.process_identity(os.getpid())
        assert kwargs == {"verify": False, "submission_dir": tmp_path / "submission"}
        if fails:
            raise RuntimeError("training failed")

    monkeypatch.setattr(local_runner, "launch", launch)
    if fails:
        with pytest.raises(RuntimeError, match="training failed"):
            local_runner.run(tmp_path)
    else:
        local_runner.run(tmp_path)
    assert json.loads((tmp_path / "exit.json").read_text()) == {"exit_code": int(fails)}


@pytest.mark.parametrize(
    "birth,group_active,exit_code,host,expected",
    [
        ("original", False, 0, "local", {"state": "RUNNING", "exit_code": None}),
        ("replacement", False, 0, "local", {"state": "UNKNOWN", "exit_code": None}),
        (None, True, 1, "local", {"state": "UNKNOWN", "exit_code": None}),
        (None, None, 1, "local", {"state": "UNKNOWN", "exit_code": None}),
        (None, False, 0, "local", {"state": "COMPLETED", "exit_code": "0:0"}),
        (None, False, 1, "local", {"state": "FAILED", "exit_code": "1:0"}),
        (None, False, None, "local", {"state": "FAILED", "exit_code": None}),
        (None, True, None, "local", {"state": "UNKNOWN", "exit_code": None}),
        (None, False, 0, "other", {"state": "UNKNOWN", "exit_code": None}),
    ],
)
@pytest.mark.parametrize("metadata", [False, True])
def test_poll_releases_gpu_only_after_supervisor_and_workers_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    birth: str | None,
    group_active: bool | None,
    exit_code: int | None,
    host: str,
    expected: dict[str, Any],
    metadata: bool,
) -> None:
    attempt = {
        "name": "trial",
        "directory": str(tmp_path),
        "backend": "local",
        "pid": 12345,
        "host": host,
        "birth": "original",
    }
    if metadata:
        (tmp_path / "process.json").write_text(json.dumps(attempt))
    if exit_code is not None:
        (tmp_path / "exit.json").write_text(json.dumps({"exit_code": exit_code}))
    monkeypatch.setattr(jobs.socket, "gethostname", lambda: "local")
    monkeypatch.setattr(jobs, "process_identity", lambda _: birth)
    monkeypatch.setattr(jobs, "process_group_active", lambda _: group_active)
    assert jobs.Backend().poll([attempt])["trial"] == expected


@pytest.mark.parametrize("states,expected", [(["S"], True), (["Z", "S"], True), (["Z", "X"], False), ([], None)])
def test_process_group_probe_ignores_zombies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, states: list[str], expected: bool | None
) -> None:
    process_root = tmp_path / "proc"
    process_root.mkdir()
    for index, state in enumerate(states, start=10):
        process = process_root / str(index)
        process.mkdir()
        (process / "stat").write_text(f"{index} (worker with ) parentheses) {state} 1 12345 12345 0 0")
    unrelated = process_root / "999"
    unrelated.mkdir()
    (unrelated / "stat").write_text("999 (unrelated) S 1 999 999 0 0")
    monkeypatch.setattr(jobs.os, "killpg", lambda *_: None)
    monkeypatch.setattr(jobs, "Path", lambda _: process_root)
    assert jobs.process_group_active(12345) is expected


@pytest.mark.parametrize("error,expected", [(ProcessLookupError, False), (PermissionError, None)])
def test_process_group_probe_handles_absent_or_inaccessible_group(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError], expected: bool | None
) -> None:
    def probe(*args: Any) -> None:
        raise error

    monkeypatch.setattr(jobs.os, "killpg", probe)
    assert jobs.process_group_active(12345) is expected


def test_process_group_probe_retains_unreadable_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = tmp_path / "12345"
    process.mkdir()
    (process / "stat").write_text("unreadable or malformed process state")
    monkeypatch.setattr(jobs.os, "killpg", lambda *_: None)
    monkeypatch.setattr(jobs, "Path", lambda _: tmp_path)
    assert jobs.process_group_active(12345) is None


def test_real_orphan_worker_retains_process_group_until_stopped() -> None:
    # The child imports only the standard library. Never touch a GPU or any
    # process group other than the new session owned by this regression test.
    script = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(child.pid, flush=True)\n"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", script],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = supervisor.communicate(timeout=5)
        assert supervisor.returncode == 0, stderr
        assert int(stdout.strip()) > 0
        assert jobs.process_identity(supervisor.pid) is None
        assert jobs.process_group_active(supervisor.pid) is True
    finally:
        try:
            os.killpg(supervisor.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        supervisor.wait(timeout=5)
    deadline = time.monotonic() + 2
    while jobs.process_group_active(supervisor.pid) is not False and time.monotonic() < deadline:
        time.sleep(0.01)
    assert jobs.process_group_active(supervisor.pid) is False
