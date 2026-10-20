"""Local process launch command construction."""

import sys
from pathlib import Path

from decentralized_llm.config import Config


def local_command(
    config: Config, filename: Path, resume: Path | None = None, action: str = "train", full: bool = True
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={config.processes_per_node}",
        *(["--numa-binding=exclusive"] if config.runtime.cpu_affinity == "exclusive" else []),
        "--max-restarts=0",
        f"--rdzv-conf=timeout={config.distributed.timeout_seconds},join_timeout={config.distributed.timeout_seconds}",
        "--module",
        "decentralized_llm",
        "worker",
        "--action",
        action,
        "--config",
        str(filename),
    ]
    if resume is not None:
        command += ["--checkpoint", str(resume.resolve())]
    if not full:
        command += ["--subset"]
    return command
