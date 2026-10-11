"""Local torchrun and fixed-size SLURM allocations; account never enters artifacts."""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from decentralized_llm.config import Config, SlurmLauncher, save_config
from decentralized_llm.preflight import validate
from decentralized_llm.runtime import atomic_json


def absolute_config(config: Config) -> Config:
    raw = config.model_dump(mode="json")
    raw["data"]["cache_dir"] = str(config.data.cache_dir.resolve())
    raw["runtime"]["output_dir"] = str(config.runtime.output_dir.resolve())
    if isinstance(config.launcher, SlurmLauncher) and config.launcher.environment_script is not None:
        raw["launcher"]["environment_script"] = str(config.launcher.environment_script.resolve())
    return Config.model_validate(raw)


def local_command(
    config: Config, filename: Path, resume: Path | None = None, action: str = "train", full: bool = True
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={config.distributed.gpus_per_node}",
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


def slurm_scripts(
    config: Config,
    filename: Path,
    worker_path: Path,
    resume: Path | None = None,
    action: str = "train",
    full: bool = True,
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
    worker = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(project))}\n"
        f"source {shlex.quote(str(environment.resolve()))}\n"
        'exec "${DECENTRALIZED_PYTHON:-python3}" -m torch.distributed.run '
        f"--nnodes={config.distributed.nodes} --nproc-per-node={config.distributed.gpus_per_node} "
        '--node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" '
        f"--master-port={config.distributed.rendezvous_port} --rdzv-backend=static --max-restarts=0 "
        f"--rdzv-conf=timeout={config.distributed.timeout_seconds} " + shlex.join(arguments) + "\n"
    )
    gpu_request = f"{launcher.gpu_type}:" if launcher.gpu_type else ""
    batch = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'mapfile -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")\n'
        'export MASTER_ADDR="${allocated_hosts[0]}"\n'
        "export OMP_NUM_THREADS=" + str(config.runtime.cpu_threads) + "\n"
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1\n"
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        f"srun --ntasks={config.distributed.nodes} --ntasks-per-node=1 "
        f"--gpus-per-task={gpu_request}{config.distributed.gpus_per_node} --gpu-bind=none "
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
        f"--nodes={config.distributed.nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={launcher.cpus_per_task}",
        f"--mem={launcher.memory}",
        f"--gpus-per-node={gpu}{config.distributed.gpus_per_node}",
    ]
    for key in ("constraint", "qos"):
        value = getattr(launcher, key)
        if value:
            arguments.append(f"--{key}={value}")
    return arguments


def launch(
    config: Config, resume: Path | None = None, *, dry_run: bool = False, action: str = "train", full: bool = True
) -> dict:
    config = absolute_config(config)
    report = validate(config, resume, launch=True, evaluation=action == "evaluate")
    destination = config.runtime.output_dir / f"submission-{time.time_ns()}"
    filename = destination / "config.yaml"
    if config.launcher.kind == "local":
        command = local_command(config, filename, resume, action, full)
        result = {"launcher": "local", "command": command, "preflight": report}
        if dry_run:
            return result
        destination.mkdir(parents=True)
        save_config(config, filename)
        atomic_json(destination / "receipt.json", result)
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(config.runtime.cpu_threads))
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        subprocess.run(command, check=True, env=env)
        return result
    batch_path, worker_path = destination / "submit.sh", destination / "worker.sh"
    batch, worker = slurm_scripts(config, filename, worker_path, resume, action, full)
    arguments = scheduler_arguments(config) + [f"--output={destination / 'slurm-%j.out'}"]
    result = {
        "launcher": "slurm",
        "arguments": arguments,
        "batch_script": batch,
        "worker_script": worker,
        "preflight": report,
    }
    if dry_run:
        return result
    destination.mkdir(parents=True)
    save_config(config, filename)
    batch_path.write_text(batch)
    worker_path.write_text(worker)
    # The account is passed only to sbatch, never to shell text, config, receipts or exception command strings.
    process = subprocess.run(
        [*arguments, "--account", os.environ["account"].strip(), str(batch_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        # Scheduler error messages can echo the account; redact before reporting.
        reason = process.stderr.replace(os.environ["account"].strip(), "<redacted>")
        raise RuntimeError(f"sbatch failed ({process.returncode}): {reason.strip()}")
    job_id = process.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError("sbatch returned an invalid job identifier")
    result["job_id"] = job_id
    atomic_json(destination / "receipt.json", result)
    return result


def print_submission(result: dict) -> None:
    print(json.dumps(result, indent=2))
