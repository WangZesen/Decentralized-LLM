"""Freeze, retrain, and evaluate tuning winners in resumable Slurm jobs."""

import argparse
import copy
import csv
import fcntl
import importlib.metadata
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml
from packaging.requirements import Requirement

from decentralized_llm.config import load_config
from decentralized_llm.data.cache import TokenCache, file_digest, fingerprint
from decentralized_llm.launch.slurm import SubmissionRejected, scheduler_arguments, submit_batch
from decentralized_llm.runtime import atomic_json, source_digest
from scripts.report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER

DEFAULT_SOURCE = Path("runs/p-tuning/20m-512c-128bs-20-40tpp-lr-wd-betas-warmup3-20261106")
DEFAULT_ROOT = Path("runs/quality/20m-512c-128bs-20-40tpp-winners-20261106")
SEEDS = (42, 43, 44)
METRICS = ("validation_loss", "blimp_acc", "piqa_acc", "piqa_acc_norm", "hellaswag_acc", "hellaswag_acc_norm")
TRAIN_CODE = """
import sys
from pathlib import Path
from decentralized_llm.config import load_config
from decentralized_llm.training.loop import train, evaluate_checkpoint
from decentralized_llm.runtime import atomic_json
c = load_config(sys.argv[1])
checkpoint = Path(sys.argv[2]) if sys.argv[2] != '-' else None
stop = int(sys.argv[3]) if sys.argv[3] != '-' else None
if (c.runtime.output_dir / 'final/manifest.json').exists():
    import json
    final = c.runtime.output_dir / 'final'
    result = json.loads((final / 'manifest.json').read_text())
    result['validation'] = evaluate_checkpoint(c, final)
    atomic_json(c.runtime.output_dir / 'result.json', result)
else:
    train(c, checkpoint, stop_after_steps=stop)
"""


def derive_runs(source: Path, root: Path) -> list[dict]:
    """Read the saved winners; never select again using benchmark outcomes."""
    historical = json.loads((source / "campaign.json").read_text())
    studies = {(s["tokens_per_parameters"], s["method"]): s for s in historical["studies"]}
    if len(studies) != len(historical["studies"]):
        raise ValueError("duplicate budget/method studies")
    runs = []
    for budget in sorted({int(budget) for budget, _ in studies}):
        for method in METHOD_ORDER:
            if (budget, method) not in studies:
                raise ValueError(f"missing winner study: {budget} TPP {method}")
            study = source / Path(studies[budget, method]["directory"]).name
            winner = study / "winner.yaml"
            original = yaml.safe_load(winner.read_text())
            report = json.loads((study / "report.json").read_text())["winner"]
            for key, value in report["values"].items():
                section, field = key.split(".")
                if original[section][field] != value:
                    raise ValueError(f"winner/report mismatch: {winner}: {key}")
            if original["training"]["tokens_per_parameters"] != budget:
                raise ValueError("winner token budget mismatch")
            if sorted(s["seed"] for s in report["seeds"]) != list(SEEDS):
                raise ValueError("winner must have exactly seeds 42, 43, 44")
            for seed in SEEDS:
                folder = root / f"{budget}tpp" / method / f"seed-{seed}"
                raw = copy.deepcopy(original)
                raw["runtime"].update(seed=seed, output_dir=str(folder / "run"))
                raw["training"]["checkpoint_policy"] = "final"
                raw["launcher"].update(environment_script=str(root / "environment.sh"), time="06:00:00")
                seed_result = next(s for s in report["seeds"] if s["seed"] == seed)
                runs.append(
                    {
                        "key": f"{budget}tpp/{method}/seed-{seed}",
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "winner": str(winner),
                        "winner_sha256": file_digest(winner),
                        "winner_id": report["id"],
                        "trial_id": seed_result["trial_id"],
                        "historical_loss": seed_result["loss"],
                        "raw": raw,
                        "config": str(folder / "resolved.yaml"),
                        "run": str(folder / "run"),
                        "quality": str(folder / "quality"),
                    }
                )
    return runs


def allocation_groups(manifest: dict) -> list[dict]:
    """Keep each seed triplet together; retain budget lanes for older campaigns."""
    grouping = manifest.get("job_grouping", "budget")
    if grouping not in ("budget", "configuration"):
        raise ValueError(f"unknown job grouping: {grouping}")
    groups = []
    for budget in sorted({r["budget"] for r in manifest["runs"]}):
        methods = METHOD_ORDER if grouping == "configuration" else (None,)
        for method in methods:
            rows = [r for r in manifest["runs"] if r["budget"] == budget and (method is None or r["method"] == method)]
            if method is not None and sorted(r["seed"] for r in rows) != list(SEEDS):
                raise ValueError(f"job requires exactly seeds 42, 43, 44: {budget} TPP {method}")
            groups.append(
                {
                    "key": f"lane-{budget}" + (f"-{method}" if method else ""),
                    "budget": budget,
                    "method": method,
                    "runs": [r["key"] for r in rows],
                }
            )
    return groups


def dependency_pins() -> list[str]:
    pending, found = ["lm-eval"], {}
    while pending:
        name = pending.pop()
        distribution = importlib.metadata.distribution(name)
        name = distribution.metadata["Name"]
        if name in found:
            continue
        found[name] = distribution.version
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return [f"{name}=={version}" for name, version in sorted(found.items())]


def verify_snapshot(root: Path, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        if file_digest(root / name) != expected:
            raise ValueError(f"frozen campaign file changed: {name}")
    if fingerprint(manifest["files"]) != manifest["snapshot_identity"]:
        raise ValueError("campaign snapshot identity mismatch")
    if "job-plan.json" in manifest["files"]:
        if json.loads((root / "job-plan.json").read_text()) != allocation_groups(manifest):
            raise ValueError("campaign allocation plan changed")


def prepare(root: Path, source: Path, *, grouping: str = "budget", suite_source: Path | None = None) -> dict:
    from decentralized_llm.benchmarks.quality_tasks import prepare_suite

    if (root / "campaign.json").exists():
        manifest = json.loads((root / "campaign.json").read_text())
        verify_snapshot(root, manifest)
        if manifest["source_campaign"] != str(source) or manifest.get("job_grouping", "budget") != grouping:
            raise ValueError("existing campaign has a different source or job grouping")
        return manifest
    project = Path(__file__).resolve().parents[1]
    historical = json.loads((source / "campaign.json").read_text())
    if source_digest(source / "source/src/decentralized_llm") != historical["provenance"]["source_sha256"]:
        raise ValueError("historical training source no longer matches its recorded provenance")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required to install pinned evaluation dependencies")
    root.mkdir(parents=True, exist_ok=True)
    runs = derive_runs(source, root)
    training_source, evaluation_source = root / "training-source", root / "evaluation-source"
    shutil.copytree(source / "source", training_source, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(project / "src", evaluation_source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (evaluation_source / "scripts").mkdir()
    for name in ("quality_campaign.py", "report_style.py"):
        shutil.copy2(project / "scripts" / name, evaluation_source / "scripts" / name)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(project / name, evaluation_source / name)
    (root / "evaluation-requirements.txt").write_text("\n".join(dependency_pins()) + "\n")
    environment = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"source {shlex.quote(str(training_source / 'scripts/toolchain.sh'))}\n"
        f'activate_toolchain {shlex.quote(str(training_source))} "$(uname -m)"\n'
        f"export DECENTRALIZED_PYTHON={shlex.quote(str(project))}/.venv-$(uname -m)/bin/python\n"
        f"export PYTHONPATH={shlex.quote(str(evaluation_source / 'src') + ':' + str(evaluation_source))}\n"
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8\n"
        'export PATH="$(dirname "$DECENTRALIZED_PYTHON"):$PATH"\n'
    )
    (root / "environment.sh").write_text(environment)
    for row in runs:
        config_path = Path(row["config"])
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(row.pop("raw"), sort_keys=False))
    config = load_config(runs[0]["config"])
    cache = TokenCache(config.data.cache_dir)
    cache.verify()
    if suite_source is not None:
        shutil.copytree(suite_source, root / "suite", ignore=shutil.ignore_patterns("*.lock"))
    suite = prepare_suite(root / "suite", config)
    historical = json.loads((source / "campaign.json").read_text())
    atomic_json(root / "account-routing.json", historical["accounts"])
    manifest = {
        "version": 1,
        "project": str(project),
        "source_campaign": str(source),
        "historical_source_provenance": historical["provenance"],
        "cache_identity": cache.manifest["identity"],
        "suite_identity": suite["identity"],
        "runs": runs,
        "job_grouping": grouping,
        "submissions": {},
        "uv_aarch64": str(Path(uv).parents[2] / "aarch64/bin/uv"),
    }
    atomic_json(root / "job-plan.json", allocation_groups(manifest))
    paths = [p for directory in (training_source, evaluation_source) for p in directory.rglob("*") if p.is_file()]
    paths += [Path(r["config"]) for r in runs]
    paths += [
        root / n
        for n in (
            "environment.sh",
            "evaluation-requirements.txt",
            "suite/manifest.json",
            "account-routing.json",
            "job-plan.json",
        )
    ]
    manifest["files"] = {str(p.relative_to(root)): file_digest(p) for p in sorted(paths)}
    manifest["snapshot_identity"] = fingerprint(manifest["files"])
    atomic_json(root / "campaign.json", manifest)
    return manifest


def active_job(job_id: str) -> bool:
    # Query the queue rather than a single ID: Slurm rejects IDs once completed
    # jobs have aged out of its live records, even though they remain in sacct.
    response = subprocess.run(
        ["squeue", "--noheader", "--format=%i"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return job_id in response.stdout.split()


def require_smoke(root: Path, manifest: dict) -> None:
    evidence = json.loads((root / "smoke/result.json").read_text())
    if not evidence["passed"] or evidence["snapshot_identity"] != manifest["snapshot_identity"]:
        raise ValueError("matching successful GPU smoke evidence is required")


def submit(
    root: Path,
    manifest: dict,
    *,
    smoke: bool = False,
    after_jobs: tuple[str, ...] = (),
    wait_for_smoke: bool = False,
    account_override: str | None = None,
    concurrent: bool | None = None,
) -> list[dict]:
    verify_snapshot(root, manifest)
    smoke_job = None
    if not smoke:
        try:
            require_smoke(root, manifest)
        except FileNotFoundError:
            receipts = manifest["submissions"].get("smoke", [])
            if not wait_for_smoke or not receipts:
                raise ValueError("successful or explicitly awaited GPU smoke test is required") from None
            last = receipts[-1]
            if last["status"] != "submitted" or not last.get("job_id") or not active_job(last["job_id"]):
                raise ValueError("awaited smoke test must have an active scheduler receipt") from None
            smoke_job = last["job_id"]
    results = []
    accounts = json.loads((root / "account-routing.json").read_text())
    policy_path = root / "submission-policy.json"
    previous_policy = (
        json.loads(policy_path.read_text()) if policy_path.exists() else {"account": None, "concurrent": False}
    )
    policy = dict(previous_policy)
    if account_override is not None:
        if not account_override.strip():
            raise ValueError("account override must not be empty")
        policy["account"] = account_override.strip()
    if concurrent is not None:
        policy["concurrent"] = concurrent
    concurrent = policy["concurrent"]
    if policy["account"]:
        accounts = {budget: policy["account"] for budget in accounts}
    if any(not job.isdigit() for job in after_jobs):
        raise ValueError("dependencies must be numeric Slurm job IDs")
    groups = allocation_groups(manifest)
    if smoke:
        groups = [{**groups[0], "key": "smoke"}]
    active = {}
    dependencies = {account: list(after_jobs) for account in accounts.values()}
    # Reconcile every group first. Serial retries wait for all existing jobs on
    # the same account; concurrent retries reuse active groups independently.
    for group in groups:
        submissions = manifest["submissions"].get(group["key"], [])
        if submissions:
            last = submissions[-1]
            if last.get("job_id") and active_job(last["job_id"]):
                active[group["key"]] = last
                if not concurrent:
                    dependencies[accounts[str(group["budget"])]].append(last["job_id"])
            elif last["status"] == "submitting":
                raise RuntimeError(f"reconcile uncertain Slurm submission before retrying: {group['key']}")
    if active and policy != previous_policy:
        raise ValueError("cancel active jobs before changing the submission account or concurrency policy")
    if account_override is not None or concurrent != previous_policy["concurrent"]:
        atomic_json(policy_path, policy)
    for group in groups:
        key, budget, method = group["key"], group["budget"], group["method"]
        submissions = manifest["submissions"].setdefault(key, [])
        if key in active:
            results.append(active[key])
            continue
        if smoke and (root / "smoke/result.json").exists():
            results.append({"status": "complete", "key": key})
            continue
        rows = [r for r in manifest["runs"] if r["key"] in group["runs"]]
        if not smoke and all(run_result(r)["status"] == "complete" for r in rows):
            results.append({"status": "complete", "key": key})
            continue
        config = load_config(rows[0]["config"])
        if config.physical_world_size != 1:
            raise ValueError("quality campaign jobs require exactly one physical GPU")
        folder = root / "submissions" / key / f"attempt-{len(submissions) + 1:03d}"
        folder.mkdir(parents=True)
        script = folder / "submit.sh"
        action = ["smoke"] if smoke else ["lane", "--budget", str(budget)]
        if not smoke and method is not None:
            action += ["--method", method]
        install = (
            (
                shlex.join([manifest["uv_aarch64"], "pip", "install", "--python"])
                + ' "$DECENTRALIZED_PYTHON" --requirement '
                + shlex.quote(str(root / "evaluation-requirements.txt"))
                + "\n"
            )
            if smoke
            else ""
        )
        # Keep queued-smoke submission compatible with older frozen workers:
        # the batch script checks the evidence even if that worker predates the gate.
        smoke_check = ""
        if not smoke:
            check_code = (
                "import json,sys; evidence=json.load(open(sys.argv[1])); "
                "sys.exit(0 if evidence.get('passed') is True and "
                "evidence.get('snapshot_identity') == sys.argv[2] else "
                "'matching successful GPU smoke evidence is required')"
            )
            smoke_check = (
                '"$DECENTRALIZED_PYTHON" '
                + shlex.join(["-c", check_code, str(root / "smoke/result.json"), manifest["snapshot_identity"]])
                + "\n"
            )
        shutil.copy2(Path(__file__), folder / "submission-controller.py")
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"source {shlex.quote(str(root / 'environment.sh'))}\n{install}"
            f"cd {shlex.quote(str(root / 'evaluation-source'))}\n"
            + smoke_check
            + "exec srun --ntasks=1 --gpus-per-task=nvidia_gh200_120gb:1 --cpu-bind=none "
            '"$DECENTRALIZED_PYTHON" scripts/quality_campaign.py ' + shlex.join([*action, "--root", str(root)]) + "\n"
        )
        account = accounts[str(budget)]
        depends_on = list(dict.fromkeys(dependencies[account]))
        atomic_json(folder / "submission-policy.json", {"account": account, "concurrent": concurrent})
        record = {
            "status": "submitting",
            "key": key,
            "script": str(script),
            "time": time.time(),
            "depends_on": depends_on,
            "runs": group["runs"],
            "requires_smoke_job": smoke_job,
            "submission_controller_sha256": file_digest(folder / "submission-controller.py"),
            "script_sha256": file_digest(script),
            "submission_policy_sha256": file_digest(folder / "submission-policy.json"),
        }
        submissions.append(record)
        atomic_json(root / "campaign.json", manifest)
        previous_account = os.environ.get("account")
        try:
            os.environ["account"] = account
            conditions = ["afterany:" + ":".join(depends_on)] if depends_on else []
            if smoke_job:
                conditions.append("afterok:" + smoke_job)
            dependency_args = (
                ["--dependency=" + ",".join(conditions), "--kill-on-invalid-dep=yes"] if conditions else []
            )
            arguments = scheduler_arguments(config)
            if smoke:
                arguments = ["--time=00:15:00" if arg.startswith("--time=") else arg for arg in arguments]
            receipt = submit_batch(
                arguments + dependency_args + [f"--job-name=quality-{key}", f"--output={folder}/slurm-%j.out"],
                script,
            )
        except SubmissionRejected as error:
            record.update(status="rejected", error=str(error))
            atomic_json(root / "campaign.json", manifest)
            raise
        finally:
            if previous_account is None:
                os.environ.pop("account", None)
            else:
                os.environ["account"] = previous_account
        record.update(receipt, status="submitted")
        if not concurrent:
            dependencies[account] = [receipt["job_id"]]
        atomic_json(folder / "receipt.json", record)
        atomic_json(root / "campaign.json", manifest)
        results.append(record)
    return results


def verify_environment(root: Path) -> None:
    import platform
    import tomllib

    import torch

    if platform.machine() != "aarch64" or "GH200" not in torch.cuda.get_device_name():
        raise ValueError("this frozen campaign requires a GH200")
    project = tomllib.loads((root / "training-source/pyproject.toml").read_text())
    for requirement in project["project"]["dependencies"]:
        name, version = requirement.split("==")
        if importlib.metadata.version(name).split("+")[0] != version:
            raise ValueError(f"historical dependency changed: {name}")
    for requirement in (root / "evaluation-requirements.txt").read_text().splitlines():
        name, version = requirement.split("==")
        if importlib.metadata.version(name) != version:
            raise ValueError(f"evaluation dependency changed: {name}")


def train_run(root: Path, row: dict, stop: int | None = None) -> Path:
    run = Path(row["run"])
    final = run / "final"
    if final.joinpath("manifest.json").exists() and (run / "result.json").exists():
        return final
    checkpoints = sorted(
        (p.parent for p in run.glob("*/manifest.json")),
        key=lambda p: json.loads((p / "manifest.json").read_text())["step"],
    )
    resume = checkpoints[-1] if checkpoints else None
    if (run / "run.json").exists() and resume is None:
        run.rename(run.with_name(f"run-failed-{time.time_ns()}"))
    env = dict(os.environ, PYTHONPATH=str(root / "training-source/src"))
    log = Path(row["config"]).parent / f"training-{time.time_ns()}.log"
    with log.open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "-c",
                TRAIN_CODE,
                row["config"],
                str(resume) if resume else "-",
                str(stop) if stop else "-",
            ],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return run / f"step-{stop:08d}" if stop else final


def smoke_run(root: Path, manifest: dict) -> dict:
    from decentralized_llm.benchmarks.quality_tasks import benchmark_quality

    verify_snapshot(root, manifest)
    verify_environment(root)
    results = {}
    for method in ("sync", "ring"):
        original = next(r for r in manifest["runs"] if r["method"] == method)
        folder = root / "smoke" / method
        folder.mkdir(parents=True, exist_ok=True)
        raw = yaml.safe_load(Path(original["config"]).read_text())
        raw["runtime"]["output_dir"] = str(folder / "run")
        config_file = folder / "resolved.yaml"
        config_file.write_text(yaml.safe_dump(raw, sort_keys=False))
        row = {"config": str(config_file), "run": str(folder / "run")}
        checkpoint = train_run(root, row, stop=2)
        result = benchmark_quality(
            load_config(config_file), checkpoint, folder / "quality", suite=root / "suite", limit=2, batch_size=8
        )
        results[method] = {"identity": result["identity"], "metrics": result["metrics"]}
    result = {"passed": True, "snapshot_identity": manifest["snapshot_identity"], "results": results}
    atomic_json(root / "smoke/result.json", result)
    return result


def lane(root: Path, manifest: dict, budget: int, method: str | None = None) -> dict:
    verify_snapshot(root, manifest)
    require_smoke(root, manifest)
    verify_environment(root)
    group = next((g for g in allocation_groups(manifest) if g["budget"] == budget and g["method"] == method), None)
    if group is None:
        raise ValueError("worker selection does not match the frozen allocation plan")
    key = group["key"]
    with (root / f".{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for row in manifest["runs"]:
            if row["key"] not in group["runs"] or run_result(row)["status"] == "complete":
                continue
            print(f"Training/evaluating {row['key']}", flush=True)
            checkpoint = train_run(root, row)
            # Release each evaluation process and CUDA allocator before the next training process.
            with (Path(row["config"]).parent / f"evaluation-{time.time_ns()}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "decentralized_llm",
                        "benchmark-quality",
                        "-c",
                        row["config"],
                        "--checkpoint",
                        str(checkpoint),
                        "--suite",
                        str(root / "suite"),
                        "--output",
                        row["quality"],
                        "--batch-size",
                        "32",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(json.dumps(run_result(row)), flush=True)
        atomic_json(root / f"{key}-complete.json", {"complete": True, "time": time.time()})
    return {"budget": budget, "method": method, "complete": True}


def run_result(row: dict) -> dict:
    result = {key: row[key] for key in ("key", "budget", "method", "seed", "historical_loss")}
    result["status"] = "pending"
    training, quality = Path(row["run"]) / "result.json", Path(row["quality"]) / "results.json"
    if training.exists():
        result["status"] = "trained"
        result["validation_loss"] = json.loads(training.read_text())["validation"]["loss"]
        result["validation_delta"] = result["validation_loss"] - row["historical_loss"]
        result["review_validation"] = abs(result["validation_delta"]) > 0.005
    if quality.exists() and result["status"] == "trained":
        evaluation = json.loads(quality.read_text())
        if evaluation["complete"] and set(evaluation["metrics"]) == {"blimp", "piqa", "hellaswag"}:
            result["status"] = "complete"
            for task, values in evaluation["metrics"].items():
                result.update({f"{task}_{metric}": value for metric, value in values.items() if metric != "count"})
    return result


def summarize_results(rows: list[dict]) -> list[dict]:
    budgets = sorted({r["budget"] for r in rows})
    groups = []
    for budget in budgets:
        for method in METHOD_ORDER:
            selected = [
                r for r in rows if r["budget"] == budget and r["method"] == method and r["status"] == "complete"
            ]
            metrics = (
                {
                    metric: {
                        "mean": statistics.mean(r[metric] for r in selected),
                        "sample_std": statistics.stdev(r[metric] for r in selected) if len(selected) > 1 else None,
                    }
                    for metric in METRICS
                }
                if selected
                else {}
            )
            groups.append(
                {
                    "budget": budget,
                    "method": method,
                    "completed_seeds": len(selected),
                    "complete": len(selected) == len(SEEDS),
                    "metrics": metrics,
                }
            )
    return groups


def report(root: Path, manifest: dict, *, rows: list[dict] | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [run_result(row) for row in manifest["runs"]] if rows is None else rows
    budgets = sorted({r["budget"] for r in rows})
    groups = summarize_results(rows)
    result = {
        "complete": all(r["status"] == "complete" for r in rows),
        "runs": rows,
        "groups": groups,
        "snapshot_identity": manifest["snapshot_identity"],
        "suite_identity": manifest["suite_identity"],
    }
    atomic_json(root / "report.json", result)
    with (root / "results.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "budget",
                "method",
                "seed",
                "status",
                "historical_loss",
                "validation_delta",
                "review_validation",
                *METRICS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Winner benchmark evaluation",
        "",
        "Zero-shot candidate likelihood; final parameter-averaged models.",
        "FP32 IEEE inference; pinned lm-eval 0.4.13. Target seeds: 42–44.",
        "Means and sample standard deviations use the completed seeds shown below; incomplete groups are interim.",
        "Hatched bars show interim groups. Standard deviation is unavailable for a single completed seed.",
        "",
        f"Completed evaluations: {sum(r['status'] == 'complete' for r in rows)}/{len(rows)}.",
        "Per-seed metrics: [CSV](results.csv). BLiMP subset scores: [CSV](blimp-subsets.csv).",
        "",
        "| TPP | Method | Seeds | C4 loss | BLiMP (%) | PIQA (%) | HellaSwag normalized (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    fig, axes = plt.subplots(len(budgets), 3, figsize=(13, 3.5 * len(budgets)), squeeze=False)
    for group in groups:
        values = group["metrics"]
        formatted = []
        for metric in ("validation_loss", "blimp_acc", "piqa_acc", "hellaswag_acc_norm"):
            if not values:
                formatted.append("pending")
                continue
            value = values[metric]
            scale, digits = (1, 4) if metric == "validation_loss" else (100, 2)
            mean = f"{value['mean'] * scale:.{digits}f}"
            deviation = f"± {value['sample_std'] * scale:.{digits}f}" if value["sample_std"] is not None else "(SD n/a)"
            formatted.append(f"{mean} {deviation}")
        seed_label = f"{group['completed_seeds']}/3" + (" (interim)" if 0 < group["completed_seeds"] < 3 else "")
        text.append(
            f"| {group['budget']} | {METHOD_LABELS[group['method']]} | {seed_label} | " + " | ".join(formatted) + " |"
        )
    for row_index, budget in enumerate(budgets):
        for column, metric in enumerate(("blimp_acc", "piqa_acc", "hellaswag_acc_norm")):
            ax = axes[row_index, column]
            for index, method in enumerate(METHOD_ORDER):
                group = next(g for g in groups if g["budget"] == budget and g["method"] == method)
                if group["metrics"]:
                    value = group["metrics"][metric]
                    ax.bar(
                        index,
                        value["mean"] * 100,
                        yerr=value["sample_std"] * 100 if value["sample_std"] is not None else None,
                        color=METHOD_COLORS[method],
                        hatch=None if group["complete"] else "//",
                        alpha=1.0 if group["complete"] else 0.55,
                        capsize=4,
                    )
            ax.set_xticks(range(4), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15)
            ax.set_title(f"{budget} TPP · {metric}")
            ax.set_ylabel("Accuracy (%)")
            if not any(g["metrics"] for g in groups if g["budget"] == budget):
                ax.text(0.5, 0.5, "No completed evaluations", transform=ax.transAxes, ha="center")
    if any(g["metrics"] and not g["complete"] for g in groups):
        fig.suptitle("Hatched bars: incomplete seed groups (interim results)")
    fig.tight_layout()
    fig.savefig(root / "benchmark-quality.png", dpi=180)
    plt.close(fig)
    # Preserve all BLiMP subset scores for detailed comparison.
    with (root / "blimp-subsets.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["budget", "method", "seed", "subset", "accuracy"])
        completed = {r["key"] for r in rows if r["status"] == "complete"}
        for row in manifest["runs"]:
            if row["key"] not in completed:
                continue
            path = Path(row["quality"]) / "results.json"
            if path.exists():
                for name, value in json.loads(path.read_text())["tasks"].items():
                    if value["family"] == "blimp":
                        writer.writerow([row["budget"], row["method"], row["seed"], name, value["metrics"]["acc"]])
    flagged = [r["key"] for r in rows if r.get("review_validation")]
    text += [
        "",
        "Validation differences exceeding 0.005 nats: " + (", ".join(flagged) if flagged else "none."),
        "",
        "![Benchmark comparison](benchmark-quality.png)",
    ]
    (root / "report.md").write_text("\n".join(text) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke-submit", "smoke", "submit", "lane", "status", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--method", choices=METHOD_ORDER)
    parser.add_argument("--group-by", choices=("budget", "configuration"), default="budget")
    parser.add_argument("--suite-source", type=Path, help="Reuse and verify an already frozen benchmark suite")
    parser.add_argument(
        "--after-job", action="append", default=[], help="Wait for this Slurm job before new allocations"
    )
    parser.add_argument(
        "--wait-for-smoke", action="store_true", help="Queue full jobs behind the active smoke job's success"
    )
    parser.add_argument("--account", help="Persist a Slurm account override without changing the training snapshot")
    parser.add_argument(
        "--concurrent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Persist independent job scheduling instead of one job at a time per account",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".controller.lock").open("a") as lock:
        if args.action in ("prepare", "smoke-submit", "submit", "report"):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            result = prepare(root, args.source.resolve(), grouping=args.group_by, suite_source=args.suite_source)
        else:
            manifest = json.loads((root / "campaign.json").read_text())
            if args.action in ("smoke-submit", "submit"):
                result = submit(
                    root,
                    manifest,
                    smoke=args.action == "smoke-submit",
                    after_jobs=tuple(args.after_job),
                    wait_for_smoke=args.wait_for_smoke,
                    account_override=args.account,
                    concurrent=args.concurrent,
                )
            elif args.action == "smoke":
                result = smoke_run(root, manifest)
            elif args.action == "lane":
                if args.budget is None:
                    parser.error("lane requires --budget")
                result = lane(root, manifest, args.budget, args.method)
            elif args.action == "report":
                result = report(root, manifest)
            else:
                result = {"runs": [run_result(row) for row in manifest["runs"]], "submissions": manifest["submissions"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
