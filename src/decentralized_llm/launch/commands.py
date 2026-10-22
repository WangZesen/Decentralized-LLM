"""Local torchrun and fixed-size SLURM allocations; account never enters artifacts."""

import json
import os
import subprocess
import time
from pathlib import Path

from decentralized_llm.config import Config, SlurmLauncher, save_config
from decentralized_llm.launch.local import local_command
from decentralized_llm.launch.preflight import validate
from decentralized_llm.launch.slurm import scheduler_arguments, slurm_scripts, submit_batch
from decentralized_llm.runtime import atomic_json


def absolute_config(config: Config) -> Config:
    raw = config.model_dump(mode="json")
    raw["data"]["cache_dir"] = str(config.data.cache_dir.resolve())
    raw["runtime"]["compile_cache_dir"] = str(config.runtime.compile_cache_dir.resolve())
    raw["runtime"]["output_dir"] = str(config.runtime.output_dir.resolve())
    if isinstance(config.launcher, SlurmLauncher):
        if config.launcher.environment_script is not None:
            raw["launcher"]["environment_script"] = str(config.launcher.environment_script.resolve())
        raw["launcher"]["environment_overlays"] = [str(path.resolve()) for path in config.launcher.environment_overlays]
    return Config.model_validate(raw)


def launch(
    config: Config,
    resume: Path | None = None,
    *,
    dry_run: bool = False,
    action: str = "train",
    full: bool = True,
    verify: bool = True,
    job_name: str | None = None,
    submission_dir: Path | None = None,
) -> dict:
    config = absolute_config(config)
    report = validate(
        config,
        resume,
        launch=True,
        evaluation=action == "evaluate",
        benchmark=action == "benchmark-training",
        verify=verify,
    )
    destination = submission_dir or config.runtime.output_dir / f"submission-{time.time_ns()}"
    filename = destination / "config.yaml"
    if config.launcher.kind == "local":
        command = local_command(config, filename, resume, action, full)
        result = {"launcher": "local", "command": command, "preflight": report, "submission_dir": str(destination)}
        if dry_run:
            return result
        destination.mkdir(parents=True)
        save_config(config, filename)
        atomic_json(destination / "receipt.json", result)
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(config.runtime.cpu_threads))
        if config.runtime.deterministic:
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        subprocess.run(command, check=True, env=env)
        return result
    batch_path, worker_path = destination / "submit.sh", destination / "worker.sh"
    batch, worker = slurm_scripts(config, filename, worker_path, resume, action, full)
    arguments = scheduler_arguments(config) + [f"--output={destination / 'slurm-%j.out'}"]
    if job_name is not None:
        arguments.append(f"--job-name={job_name}")
    result = {
        "launcher": "slurm",
        "arguments": arguments,
        "batch_script": batch,
        "worker_script": worker,
        "preflight": report,
        "submission_dir": str(destination),
        "job_name": job_name,
    }
    if dry_run:
        return result
    destination.mkdir(parents=True)
    save_config(config, filename)
    batch_path.write_text(batch)
    worker_path.write_text(worker)
    result.update(submit_batch(arguments, batch_path))
    atomic_json(destination / "receipt.json", result)
    return result


def print_submission(result: dict) -> None:
    print(json.dumps(result, indent=2))
