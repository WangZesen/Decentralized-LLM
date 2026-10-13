"""Batch scheduler reconciliation and restartable local process supervision."""

import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from decentralized_llm.config import Config, save_config
from decentralized_llm.launch import launch
from decentralized_llm.tuning.spec import Record

ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESV_DEL_HOLD",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
}
FAILED = {
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "OUT_OF_MEMORY",
    "REVOKED",
}


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot}:{fields[19]}"
    except FileNotFoundError, ProcessLookupError:
        return None


def query(arguments: list[str]) -> str:
    process = subprocess.run(arguments, capture_output=True, text=True, check=False, timeout=30)
    if process.returncode:
        # Deliberately omit scheduler stderr, which may contain account names.
        raise RuntimeError(f"{arguments[0]} status query failed ({process.returncode}); retaining active jobs")
    return process.stdout


def scheduler_status(attempts: list[Record]) -> dict[str, Record]:
    groups: dict[str, list[Record]] = {}
    for attempt in attempts:
        receipt = Path(attempt["directory"]) / "submission" / "receipt.json"
        if receipt.is_file():
            recovered = json.loads(receipt.read_text())
            attempt.update({key: recovered.get(key) for key in ("job_id", "cluster")})
        groups.setdefault(attempt.get("cluster") or "", []).append(attempt)
    result: dict[str, Record] = {}
    for cluster, group in groups.items():
        cluster_args = [f"--clusters={cluster}"] if cluster else []
        names = {attempt["name"] for attempt in group}
        queued = query(
            [
                "squeue",
                "--noheader",
                "--all",
                "--states=all",
                f"--user={os.getuid()}",
                "--format=%i|%j|%T",
                *cluster_args,
            ]
        )
        for line in queued.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) == 3 and parts[1] in names:
                job, name, state = parts
                if name in result and result[name]["job_id"] != job:
                    raise RuntimeError(
                        f"multiple scheduler jobs have attempt name {name}; manual reconciliation required"
                    )
                result[name] = {"job_id": job, "cluster": cluster or None, "state": state, "exit_code": None}
        missing = [attempt for attempt in group if result.get(attempt["name"], {}).get("state") not in ACTIVE]
        if not missing:
            continue
        since = datetime.fromtimestamp(min(attempt["created"] for attempt in missing) - 86400, UTC)
        accounting = query(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                f"--user={os.getuid()}",
                f"--starttime={since.strftime('%Y-%m-%dT%H:%M:%S')}",
                "--format=JobIDRaw,JobName%100,State%40,ExitCode",
                "--name=" + ",".join(attempt["name"] for attempt in missing),
                *cluster_args,
            ]
        )
        for line in accounting.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4 or parts[1] not in names or not parts[0].isdigit():
                continue
            job, name, state, code = parts[:4]
            previous = result.get(name)
            if previous and previous["job_id"] != job:
                raise RuntimeError(f"multiple scheduler jobs have attempt name {name}; manual reconciliation required")
            if previous and previous["state"] in ACTIVE:
                continue
            result[name] = {
                "job_id": job,
                "cluster": cluster or None,
                "state": state.split()[0].rstrip("+"),
                "exit_code": code,
            }
    return result


class Backend:
    def __init__(self) -> None:
        self.children: dict[str, subprocess.Popen] = {}

    def submit(self, config: Config, attempt: Record) -> Record:
        directory = Path(attempt["directory"])
        if config.launcher.kind == "slurm":
            return launch(config, verify=False, job_name=attempt["name"], submission_dir=directory / "submission")
        save_config(config, directory / "config.yaml")
        with (directory / "runner.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "decentralized_llm.tuning.local_runner", str(directory)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children[attempt["name"]] = process
        return {"pid": process.pid, "host": socket.gethostname(), "birth": process_identity(process.pid)}

    def poll(self, attempts: list[Record]) -> dict[str, Record]:
        result = scheduler_status([attempt for attempt in attempts if attempt["backend"] == "slurm"])
        for attempt in attempts:
            if attempt["backend"] != "local":
                continue
            child = self.children.get(attempt["name"])
            if child is not None:
                child.poll()  # Reap completed children, including zombies.
            directory = Path(attempt["directory"])
            exit_path = directory / "exit.json"
            if exit_path.is_file():
                code = json.loads(exit_path.read_text())["exit_code"]
                result[attempt["name"]] = {"state": "COMPLETED" if code == 0 else "FAILED", "exit_code": f"{code}:0"}
                continue
            metadata = directory / "process.json"
            info = json.loads(metadata.read_text()) if metadata.is_file() else attempt
            state = "UNKNOWN"
            if info.get("host") == socket.gethostname() and info.get("pid"):
                birth = process_identity(info["pid"])
                if birth is not None and birth == info.get("birth"):
                    state = "RUNNING"
                elif birth is None:
                    # A killed supervisor may still have live torchrun descendants in its group.
                    try:
                        os.killpg(info["pid"], 0)
                    except ProcessLookupError:
                        state = "FAILED"
            result[attempt["name"]] = {"state": state, "exit_code": None}
        return result
