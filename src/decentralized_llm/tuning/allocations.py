"""Prepare a single batch allocation containing sequential training steps."""

import shlex
from pathlib import Path

from decentralized_llm.config import Config, SlurmLauncher, save_config
from decentralized_llm.launch.slurm import scheduler_arguments, slurm_scripts, submit_batch
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.batching import format_time, trial_minutes
from decentralized_llm.tuning.spec import Record


def submit_allocation(configs: list[Config], allocation: Record, attempts: list[Record]) -> Record:
    directory = Path(allocation["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "steps").mkdir(exist_ok=True)
    steps = []
    for config, attempt in zip(configs, attempts, strict=True):
        submission = Path(attempt["directory"]) / "submission"
        submission.mkdir(parents=True)
        filename, worker = submission / "config.yaml", submission / "worker.sh"
        save_config(config, filename)
        step, script = slurm_scripts(config, filename, worker, step_time=format_time(trial_minutes(config)))
        worker.write_text(script)
        step_path = submission / "step.sh"
        step_path.write_text(step)
        steps.append(
            {
                "trial_id": attempt["trial_id"],
                "attempt": attempt["number"],
                "script": str(step_path),
                "time_minutes": trial_minutes(config),
                **{key: attempt[key] for key in ("stdout", "stderr", "execution_record")},
            }
        )
    manifest = directory / "manifest.json"
    atomic_json(manifest, {"allocation_id": allocation["id"], "steps": steps})
    config = configs[0].model_copy(deep=True)
    assert isinstance(config.launcher, SlurmLauncher)
    config.launcher.time = format_time(allocation["time_minutes"])
    project = Path.cwd().resolve()
    environment = config.launcher.environment_script or project / "scripts/environment.sh"
    batch = directory / "submit.sh"
    batch.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(project))}\n"
        f"source {shlex.quote(str(environment.resolve()))}\n"
        'exec "${DECENTRALIZED_PYTHON:-python3}" -m decentralized_llm.tuning.slurm_runner '
        + shlex.quote(str(manifest))
        + "\n"
    )
    arguments = scheduler_arguments(config) + [
        f"--job-name={allocation['name']}",
        f"--output={allocation['stdout']}",
        f"--error={allocation['stderr']}",
    ]
    receipt = {"arguments": arguments, "batch_script": str(batch), **submit_batch(arguments, batch)}
    atomic_json(Path(allocation["receipt_path"]), receipt)
    return receipt
