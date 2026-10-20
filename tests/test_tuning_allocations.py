"""Allocation arithmetic and execution without a live scheduler or GPUs."""

import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

from decentralized_llm.config import Config, SlurmLauncher
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning import allocations, backends
from decentralized_llm.tuning.backends import Backend
from decentralized_llm.tuning.batching import (
    allocation_failures,
    balanced_sizes,
    check_walltime,
    format_time,
    group_trials,
    time_minutes,
    training_attempts,
    trial_minutes,
)
from decentralized_llm.tuning.spec import Controller, Record


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_batch_size_requires_positive_integer(value: object) -> None:
    with pytest.raises(ValueError):
        Controller.model_validate({"jobs_per_allocation": value})
    assert Controller().jobs_per_allocation == 1


@pytest.mark.parametrize(
    ("count", "maximum", "expected"), [(0, 4, []), (1, 4, [1]), (10, 4, [4, 3, 3]), (12, 4, [4, 4, 4]), (7, 5, [4, 3])]
)
def test_balanced_group_sizes(count: int, maximum: int, expected: list[int]) -> None:
    assert balanced_sizes(count, maximum) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("15", 15),
        ("1:00", 1),
        ("1:01", 2),
        ("01:00:01", 61),
        ("1-02", 1560),
        ("1-02:03", 1563),
        ("1-02:03:01", 1564),
        ("0:00", None),
        ("UNLIMITED", None),
    ],
)
def test_slurm_time_arithmetic(value: str, expected: int | None) -> None:
    assert time_minutes(value) == expected
    if expected is not None:
        assert SlurmLauncher(partition="gpu", time=value).time == value
        assert time_minutes(format_time(expected)) == expected


def test_walltime_limits_and_partition_query(monkeypatch: pytest.MonkeyPatch) -> None:
    check_walltime(4320, 4320, "gpu", 15)
    check_walltime(9999, None, "gpu", 15)
    with pytest.raises(ValueError, match="jobs_per_allocation <= 288"):
        check_walltime(4335, 4320, "gpu", 15)
    with pytest.raises(ValueError, match="even one trial"):
        check_walltime(15, 10, "gpu", 15)
    monkeypatch.setattr(backends, "query", lambda _: "PartitionName=gpu State=UP MaxTime=3-00:00:00")
    assert Backend().partition_limit("gpu") == 4320
    monkeypatch.setattr(backends, "query", lambda _: "PartitionName=wrong MaxTime=UNLIMITED")
    with pytest.raises(RuntimeError, match="cannot determine"):
        Backend().partition_limit("gpu")


@pytest.mark.parametrize("failure", ["exit", "missing", "timeout"])
def test_partition_query_failure_explains_configuration_without_scheduler_stderr(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = "private-scheduler-account"

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert arguments == ["scontrol", "show", "partition", "gpu", "--oneliner"]
        if failure == "missing":
            raise FileNotFoundError(private)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(arguments, 30, stderr=private)
        return subprocess.CompletedProcess(arguments, 1, stdout="", stderr=private)

    monkeypatch.setattr(backends.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="cannot query MaxTime for SLURM partition 'gpu'") as raised:
        Backend().partition_limit("gpu")
    message = str(raised.value)
    assert "launcher.partition" in message and "base.overrides" in message
    assert "sinfo --noheader --format=%P" in message
    assert "scontrol show partition gpu --oneliner" in message
    diagnostic = "".join(traceback.format_exception(raised.value))
    assert private not in diagnostic
    assert "retaining active jobs" not in diagnostic


@pytest.mark.parametrize("value", ["0", "0:00", "00:00:00", "0-00"])
def test_tuning_requires_finite_positive_trial_limits(tiny_config: Config, value: str) -> None:
    tiny_config.launcher = SlurmLauncher(partition="gpu", time=value)
    with pytest.raises(ValueError, match="finite, positive"):
        trial_minutes(tiny_config)


def test_retry_groups_are_balanced_separately(tiny_config: Config) -> None:
    tiny_config.launcher = SlurmLauncher(partition="gpu")
    trials = [
        {"id": index, "config": tiny_config.model_dump(mode="json"), "attempts": [{}] if index < 10 else []}
        for index in range(12)
    ]
    groups = group_trials(trials, tiny_config, 4)
    assert [[trial["id"] for trial in group] for group in groups] == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11]]


def test_scheduler_refuses_job_identity_conflicting_with_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = {
        "name": "batch",
        "directory": str(tmp_path),
        "receipt_path": str(tmp_path / "receipt.json"),
        "created": time.time(),
    }
    atomic_json(tmp_path / "receipt.json", {"job_id": "123", "cluster": None})
    monkeypatch.setattr(backends, "query", lambda _: "456|batch|RUNNING\n")
    with pytest.raises(RuntimeError, match="disagrees with its receipt"):
        backends.scheduler_status([parent])


def test_resource_compatibility_and_retry_groups(tiny_config: Config) -> None:
    tiny_config.launcher = SlurmLauncher(partition="gpu")
    tiny_config.training.batch_tokens = 48
    trials = []
    for index, (nodes, gpus) in enumerate([(1, 1), (2, 1), (1, 1), (2, 2), (1, 1)]):
        config = tiny_config.model_copy(deep=True)
        config.distributed.nodes, config.distributed.gpus_per_node = nodes, gpus
        trials.append({"id": index, "config": config.model_dump(mode="json"), "attempts": [{}] if index == 4 else []})
    groups = group_trials(trials, tiny_config, 4)
    assert [[trial["id"] for trial in group] for group in groups] == [[0, 2], [1], [3], [4]]
    trial = {
        "attempts": [
            {"status": "failed"},
            {"status": "failed", "allocation_id": "a", "started": False},
            {"status": "failed", "allocation_id": "b", "started": True},
        ]
    }
    assert training_attempts(trial) == 2
    assert allocation_failures(trial) == 1


@pytest.fixture
def prepared_allocation(tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Record:
    environment = tmp_path / "environment.sh"
    environment.write_text(f"export DECENTRALIZED_PYTHON={shlex.quote(sys.executable)}\n")
    tiny_config.launcher = SlurmLauncher(partition="gpu", time="00:05:00", environment_script=environment)
    configs, attempts = [], []
    root = tmp_path / "slurm" / "phase" / "allocation"
    allocation = {
        "id": "allocation",
        "directory": str(root),
        "name": "allocation-name",
        "time_minutes": 15,
        "stdout": str(root / "slurm-%j.out"),
        "stderr": str(root / "slurm-%j.err"),
        "receipt_path": str(root / "receipt.json"),
    }
    for index in range(3):
        destination = tmp_path / "trials" / f"trial-{index}" / "attempt-001"
        config = tiny_config.model_copy(deep=True)
        config.runtime.seed = 42 + index
        config.runtime.output_dir = destination / "run"
        configs.append(config)
        attempts.append(
            {
                "directory": str(destination),
                "number": 1,
                "trial_id": f"trial-{index}",
                "stdout": str(root / "steps" / f"{index}.out"),
                "stderr": str(root / "steps" / f"{index}.err"),
                "execution_record": str(root / "steps" / f"{index}.json"),
            }
        )
    calls = []

    def submit(arguments: list[str], path: Path) -> Record:
        calls.append(arguments)
        assert path.is_file()
        return {"job_id": "123", "cluster": None}

    monkeypatch.setattr(allocations, "submit_batch", submit)
    receipt = allocations.submit_allocation(configs, allocation, attempts)
    for script in [root / "submit.sh", *tmp_path.glob("trials/*/attempt-001/submission/*.sh")]:
        subprocess.run(["bash", "-n", str(script)], check=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "scontrol").write_text("#!/bin/sh\nprintf 'node1\\n'\n")
    (commands / "scontrol").chmod(0o755)
    (commands / "srun").write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys\nfrom pathlib import Path\nimport yaml\n"
        "config = yaml.safe_load(Path(sys.argv[-1]).with_name('config.yaml').read_text())\n"
        "seed = config['runtime']['seed']\n"
        "lock = Path(os.environ['EVENTS'] + '.lock')\n"
        "with lock.open('x'):\n"
        "    with open(os.environ['EVENTS'], 'a') as stream:\n"
        "        stream.write(json.dumps({'seed': seed, 'args': sys.argv[1:]}) + '\\n')\n"
        "    print('stdout', seed, flush=True)\n"
        "    print('stderr', seed, file=sys.stderr, flush=True)\n"
        "lock.unlink()\n"
        "if os.environ.get('STOP_AFTER_FIRST'):\n"
        "    os.kill(os.getppid(), signal.SIGTERM)\n"
        "sys.exit(7 if seed == 43 else 0)\n"
    )
    (commands / "srun").chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{commands}:{os.environ['PATH']}",
        SLURM_JOB_NODELIST="node1",
        SLURM_JOB_END_TIME=str(int(time.time() + 1800)),
        EVENTS=str(tmp_path / "events.jsonl"),
    )
    return {"root": root, "attempts": attempts, "calls": calls, "receipt": receipt, "env": env}


def test_generated_allocation_runs_sequential_steps_and_preserves_logs(prepared_allocation: Record) -> None:
    root, env = prepared_allocation["root"], prepared_allocation["env"]
    result = subprocess.run(["bash", str(root / "submit.sh")], env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode == 1, result.stderr
    arguments = prepared_allocation["calls"][0]
    assert "--time=00:15:00" in arguments
    assert f"--output={root}/slurm-%j.out" in arguments
    assert f"--error={root}/slurm-%j.err" in arguments
    events = [json.loads(line) for line in Path(env["EVENTS"]).read_text().splitlines()]
    assert [event["seed"] for event in events] == [42, 43, 44]
    assert all("--time=00:05:00" in event["args"] for event in events)
    assert [json.loads((root / "steps" / f"{i}.json").read_text())["exit_code"] for i in range(3)] == [0, 7, 0]
    for index in range(3):
        assert f"stdout {42 + index}" in (root / "steps" / f"{index}.out").read_text()
        assert f"stderr {42 + index}" in (root / "steps" / f"{index}.err").read_text()
    before = Path(env["EVENTS"]).read_text()
    # Requeue/restart of the same allocation never runs an already-started trial twice.
    subprocess.run(["bash", str(root / "submit.sh")], env=env, capture_output=True, timeout=40)
    assert Path(env["EVENTS"]).read_text() == before


@pytest.mark.parametrize("terminate", [False, True])
def test_allocation_skips_unstarted_steps_on_deadline_or_signal(prepared_allocation: Record, terminate: bool) -> None:
    root, env = prepared_allocation["root"], prepared_allocation["env"]
    if terminate:
        env["STOP_AFTER_FIRST"] = "1"
    else:
        env["SLURM_JOB_END_TIME"] = str(time.time() + 60)
    result = subprocess.run(["bash", str(root / "submit.sh")], env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode == 1, result.stderr
    records = [json.loads((root / "steps" / f"{index}.json").read_text()) for index in range(3)]
    assert all(not record["started"] and record["state"] == "SKIPPED" for record in records[1 if terminate else 0 :])
    if terminate:
        assert len(Path(env["EVENTS"]).read_text().splitlines()) == 1
    else:
        assert not Path(env["EVENTS"]).exists()


def test_single_step_can_start_with_subminute_setup_overhead(prepared_allocation: Record) -> None:
    root, env = prepared_allocation["root"], prepared_allocation["env"]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["steps"] = manifest["steps"][:1]
    atomic_json(manifest_path, manifest)
    env["SLURM_JOB_END_TIME"] = str(time.time() + 290)
    result = subprocess.run(["bash", str(root / "submit.sh")], env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    assert json.loads((root / "steps/0.json").read_text())["started"]
