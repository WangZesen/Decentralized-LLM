"""Run sequential srun steps, publishing independent results for reconciliation."""

import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from decentralized_llm.runtime import atomic_json


def run(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    stopping = False
    child: subprocess.Popen | None = None

    def terminate(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    failed = False
    try:
        with (manifest_path.parent / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            end = os.environ.get("SLURM_JOB_END_TIME")
            if end is None:
                raise RuntimeError("SLURM_JOB_END_TIME is required to respect the allocation deadline")
            deadline = float(end)
            for step in manifest["steps"]:
                path = Path(step["execution_record"])
                identity = {
                    "allocation_id": manifest["allocation_id"],
                    "trial_id": step["trial_id"],
                    "attempt": step["attempt"],
                }
                if path.exists():
                    record = json.loads(path.read_text())
                    if any(record.get(key) != value for key, value in identity.items()):
                        raise ValueError("step execution record identity mismatch")
                    # A requeued allocation must not repeat a trial that already started.
                    if record["started"]:
                        if record["state"] == "RUNNING":
                            record.update(
                                state="FAILED", exit_code=None, finished=time.time(), reason="Allocation restarted"
                            )
                            atomic_json(path, record)
                        failed |= record["state"] != "COMPLETED"
                        continue
                # Slurm budgets have minute resolution. Allow sub-minute launcher overhead,
                # including the first step of a one-trial allocation; the parent deadline is final.
                remaining = math.ceil(max(0, deadline - time.time()) / 60)
                if stopping or remaining < step["time_minutes"]:
                    failed = True
                    atomic_json(
                        path,
                        {
                            **identity,
                            "started": False,
                            "state": "SKIPPED",
                            "exit_code": None,
                            "finished": time.time(),
                            "reason": "Allocation terminating"
                            if stopping
                            else "Insufficient allocation time remaining",
                        },
                    )
                    continue
                record = {**identity, "started": True, "state": "RUNNING", "started_at": time.time()}
                atomic_json(path, record)
                print(f"Starting trial {step['trial_id']} attempt {step['attempt']}", flush=True)
                try:
                    with Path(step["stdout"]).open("a") as stdout, Path(step["stderr"]).open("a") as stderr:
                        child = subprocess.Popen(
                            ["bash", step["script"]], stdout=stdout, stderr=stderr, start_new_session=True
                        )
                        # A termination signal can arrive between publishing intent and Popen.
                        if stopping:
                            terminate(signal.SIGTERM, None)
                        code = child.wait()
                except OSError as error:
                    code = 1
                    record.update(started=False, reason=str(error))
                finally:
                    child = None
                record.update(state="COMPLETED" if code == 0 else "FAILED", exit_code=code, finished=time.time())
                atomic_json(path, record)
                failed |= code != 0
                print(f"Finished trial {step['trial_id']} attempt {step['attempt']}: exit={code}", flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return int(failed)


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])))
