"""Detached supervisor whose durable records survive controller restarts."""

import fcntl
import os
import socket
import sys
from pathlib import Path

from decentralized_llm.config import load_config
from decentralized_llm.launch import launch
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.backends import process_identity


def run(directory: Path) -> None:
    with (directory / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(
            directory / "process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "birth": process_identity(os.getpid()),
            },
        )
        code = 1
        try:
            launch(load_config(directory / "config.yaml"), verify=False, submission_dir=directory / "submission")
            code = 0
        finally:
            atomic_json(directory / "exit.json", {"exit_code": code})


if __name__ == "__main__":
    run(Path(sys.argv[1]))
