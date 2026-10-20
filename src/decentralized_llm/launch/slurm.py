"""Slurm allocation and job-step scripts."""

import os
import shlex
import subprocess
from pathlib import Path

from decentralized_llm.config import Config, SlurmLauncher


class SubmissionRejected(RuntimeError):
    """The scheduler rejected a submission without returning a job ID."""


def slurm_scripts(
    config: Config,
    filename: Path,
    worker_path: Path,
    resume: Path | None = None,
    action: str = "train",
    full: bool = True,
    *,
    step_time: str | None = None,
) -> tuple[str, str]:
    launcher = config.launcher
    if not isinstance(launcher, SlurmLauncher):
        raise ValueError("SLURM launcher required")
    project = Path.cwd().resolve()
    environment = launcher.environment_script or project / "scripts" / "environment.sh"
    arguments = ["--module", "decentralized_llm", "worker", "--action", action, "--config", str(filename)]
    if resume is not None:
        arguments += ["--checkpoint", str(resume.resolve())]
    if not full:
        arguments += ["--subset"]
    # Independent single-node allocations can share a host; let torchrun reserve
    # an available port instead of making concurrent jobs bind the same one.
    rendezvous = (
        "--standalone "
        if config.physical_nodes == 1
        else '--node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" '
        f"--master-port={config.distributed.rendezvous_port} --rdzv-backend=static "
    )
    affinity = "--numa-binding=exclusive " if config.runtime.cpu_affinity == "exclusive" else ""
    overlays = "".join(f"source {shlex.quote(str(path.resolve()))}\n" for path in launcher.environment_overlays)
    worker = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(project))}\n"
        f"source {shlex.quote(str(environment.resolve()))}\n"
        f"{overlays}"
        'exec "${DECENTRALIZED_PYTHON:-python3}" -m torch.distributed.run '
        f"--nnodes={config.physical_nodes} --nproc-per-node={config.processes_per_node} "
        f"{rendezvous}{affinity}--max-restarts=0 "
        f"--rdzv-conf=timeout={config.distributed.timeout_seconds} " + shlex.join(arguments) + "\n"
    )
    gpu_request = f"{launcher.gpu_type}:" if launcher.gpu_type else ""
    batch = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'mapfile -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")\n'
        'export MASTER_ADDR="${allocated_hosts[0]}"\n'
        "export OMP_NUM_THREADS=" + str(config.runtime.cpu_threads) + "\n"
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1\n"
        + (f"exec srun --time={shlex.quote(step_time)} " if step_time is not None else "srun ")
        + f"--ntasks={config.physical_nodes} --ntasks-per-node=1 "
        f"--gpus-per-task={gpu_request}{config.processes_per_node} --gpu-bind=none "
        "--cpu-bind=none --kill-on-bad-exit=1 bash " + shlex.quote(str(worker_path)) + "\n"
    )
    return batch, worker


def scheduler_arguments(config: Config) -> list[str]:
    launcher = config.launcher
    if not isinstance(launcher, SlurmLauncher):
        raise ValueError("SLURM launcher required")
    gpu = f"{launcher.gpu_type}:" if launcher.gpu_type else ""
    arguments = [
        "sbatch",
        "--parsable",
        f"--partition={launcher.partition}",
        f"--time={launcher.time}",
        f"--nodes={config.physical_nodes}",
        "--ntasks-per-node=1",
        f"--gpus-per-node={gpu}{config.processes_per_node}",
    ]
    # An explicit CPU request overrides Slurm's GPU-proportional CPU default.
    if launcher.cpus_per_task is not None:
        arguments.append(f"--cpus-per-task={launcher.cpus_per_task}")
    # Leave memory allocation to the site's GPU-based policy unless overridden.
    if launcher.memory is not None:
        arguments.append(f"--mem={launcher.memory}")
    for key in ("constraint", "qos", "network"):
        value = getattr(launcher, key)
        if value:
            arguments.append(f"--{key}={value}")
    return arguments


def submit_batch(arguments: list[str], batch_path: Path) -> dict:
    """Submit prepared artifacts; keep the account out of receipts and exceptions."""
    account = os.environ["account"].strip()
    process = subprocess.run(
        [*arguments, "--account", account, str(batch_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        reason = process.stderr.replace(account, "<redacted>")
        raise SubmissionRejected(f"sbatch failed ({process.returncode}): {reason.strip()}")
    job_id, _, cluster = process.stdout.strip().partition(";")
    if not job_id.isdigit():
        raise RuntimeError("sbatch returned an invalid job identifier")
    return {"job_id": job_id, "cluster": cluster or None}
