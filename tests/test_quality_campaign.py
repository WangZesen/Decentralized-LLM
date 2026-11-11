import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from decentralized_llm.config import Config
from decentralized_llm.runtime import atomic_json
from scripts import quality_campaign as campaign


@pytest.mark.parametrize("budgets", [(20, 40), (60,), (80, 100), (120,)])
def test_derivation_retains_winners_and_changes_only_authorized_fields(
    tmp_path: Path, tiny_config: Config, budgets: tuple[int, ...]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    studies = []
    for budget in budgets:
        for method in campaign.METHOD_ORDER:
            folder = source / f"{budget}-{method}"
            folder.mkdir()
            raw = tiny_config.model_dump(mode="json")
            raw["training"].update(tokens_per_parameters=budget, checkpoint_policy="none")
            (folder / "winner.yaml").write_text(yaml.safe_dump(raw))
            atomic_json(
                folder / "report.json",
                {
                    "winner": {
                        "id": "winner",
                        "values": {"optimizer.lr": raw["optimizer"]["lr"]},
                        "seeds": [{"seed": seed, "loss": 3.5, "trial_id": str(seed)} for seed in campaign.SEEDS],
                    }
                },
            )
            studies.append({"tokens_per_parameters": budget, "method": method, "directory": str(folder)})
    atomic_json(source / "campaign.json", {"studies": studies})
    rows = campaign.derive_runs(source, tmp_path / "destination")
    assert len(rows) == 12 * len(budgets)
    assert [r["method"] for r in rows[::3]] == list(campaign.METHOD_ORDER) * len(budgets)
    groups = campaign.allocation_groups({"runs": rows, "job_grouping": "configuration"})
    assert len(groups) == 4 * len(budgets)
    assert all(len(group["runs"]) == 3 for group in groups)
    assert [key for group in groups for key in group["runs"]] == [row["key"] for row in rows]
    for row in rows:
        original = yaml.safe_load(Path(row["winner"]).read_text())
        raw = row["raw"]
        assert raw["runtime"]["seed"] == row["seed"]
        assert raw["training"]["checkpoint_policy"] == "final"
        for section in ("model", "data", "optimizer", "lr_schedule", "distributed", "evaluation"):
            assert raw[section] == original[section]
    winner = Path(rows[0]["winner"])
    data = yaml.safe_load(winner.read_text())
    data["optimizer"]["lr"] *= 2
    winner.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="winner/report mismatch"):
        campaign.derive_runs(source, tmp_path / "destination")


@pytest.mark.parametrize("queued", ["101\n103\n", ""])
def test_active_job_handles_ids_aged_out_of_slurm(monkeypatch: pytest.MonkeyPatch, queued: str) -> None:
    def slurm(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--jobs" in args:
            raise subprocess.CalledProcessError(1, args, stderr="Invalid job id specified")
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(args, 0, stdout=queued)

    monkeypatch.setattr(campaign.subprocess, "run", slurm)
    assert not campaign.active_job("102")
    assert campaign.active_job("101") == bool(queued)


def test_active_job_does_not_treat_scheduler_errors_as_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(args: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, args, stderr="Unable to contact slurm controller")

    monkeypatch.setattr(campaign.subprocess, "run", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.active_job("102")


def test_submission_reuses_active_receipt_and_rejects_uncertain_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "verify_snapshot", lambda *a: None)
    atomic_json(tmp_path / "account-routing.json", {"20": "test"})
    manifest = {
        "runs": [{"budget": 20, "key": "test"}],
        "submissions": {"smoke": [{"status": "submitted", "job_id": "123"}]},
    }
    monkeypatch.setattr(campaign, "active_job", lambda job: True)
    assert campaign.submit(tmp_path, manifest, smoke=True)[0]["job_id"] == "123"
    manifest["submissions"]["smoke"] = [{"status": "submitting"}]
    with pytest.raises(RuntimeError, match="uncertain"):
        campaign.submit(tmp_path, manifest, smoke=True)


def grouped_manifest(tmp_path: Path) -> dict:
    return {
        "snapshot_identity": "snapshot",
        "job_grouping": "configuration",
        "submissions": {},
        "runs": [
            {
                "key": f"{budget}tpp/{method}/seed-{seed}",
                "budget": budget,
                "method": method,
                "seed": seed,
                "config": str(tmp_path / f"{budget}-{method}-{seed}.yaml"),
                "quality": str(tmp_path / f"quality-{budget}-{method}-{seed}"),
            }
            for budget in (80, 100)
            for method in campaign.METHOD_ORDER
            for seed in campaign.SEEDS
        ],
    }


def test_configuration_submission_and_retry_preserve_account_limit(
    tmp_path: Path, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = grouped_manifest(tmp_path)
    (tmp_path / "smoke").mkdir()
    atomic_json(tmp_path / "smoke/result.json", {"passed": True, "snapshot_identity": "snapshot"})
    atomic_json(tmp_path / "account-routing.json", {"80": "same-account", "100": "same-account"})
    monkeypatch.setattr(campaign, "verify_snapshot", lambda *a: None)
    monkeypatch.setattr(campaign, "load_config", lambda *a: tiny_config)
    monkeypatch.setattr(campaign, "scheduler_arguments", lambda *a: ["sbatch"])
    monkeypatch.setattr(campaign, "run_result", lambda *a: {"status": "pending"})
    requests = []

    def submit(arguments: list[str], script: Path) -> dict:
        assert os.environ["account"] == "same-account"
        requests.append((arguments, script.read_text()))
        return {"job_id": str(100 + len(requests))}

    monkeypatch.setattr(campaign, "submit_batch", submit)
    receipts = campaign.submit(tmp_path, manifest, after_jobs=("99",))
    assert len(receipts) == 8
    for index, (receipt, (arguments, script)) in enumerate(zip(receipts, requests, strict=True)):
        assert receipt["depends_on"] == (["99"] if index == 0 else [str(100 + index)])
        assert "--dependency=afterany:" + receipt["depends_on"][0] in arguments
        assert len(receipt["runs"]) == 3
        assert f"--budget {80 if index < 4 else 100}" in script
        assert f"--method {campaign.METHOD_ORDER[index % 4]}" in script
    monkeypatch.setattr(campaign, "active_job", lambda job: True)
    campaign.submit(tmp_path, manifest)
    assert len(requests) == 8  # Do not duplicate any active job.
    monkeypatch.setattr(campaign, "active_job", lambda job: job != "101")
    retried = campaign.submit(tmp_path, manifest)
    assert len(requests) == 9
    assert retried[0]["depends_on"] == [str(job) for job in range(102, 109)]


def test_configuration_worker_runs_only_its_three_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = grouped_manifest(tmp_path)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda *a: None)
    monkeypatch.setattr(campaign, "require_smoke", lambda *a: None)
    monkeypatch.setattr(campaign, "verify_environment", lambda *a: None)
    monkeypatch.setattr(campaign, "run_result", lambda *a: {"status": "pending"})
    trained, evaluated = [], []

    def train(root: Path, row: dict) -> Path:
        trained.append(row["key"])
        return root / "checkpoint"

    monkeypatch.setattr(campaign, "train_run", train)
    monkeypatch.setattr(campaign.subprocess, "run", lambda args, **kw: evaluated.append(args))
    campaign.lane(tmp_path, manifest, 100, "exp")
    assert trained == [f"100tpp/exp/seed-{seed}" for seed in campaign.SEEDS]
    assert len(evaluated) == 3
    assert (tmp_path / "lane-100-exp-complete.json").exists()
    with pytest.raises(ValueError, match="selection"):
        campaign.lane(tmp_path, manifest, 100)  # Cannot accidentally run the full budget.


def test_account_override_and_concurrent_retries_are_persistent(
    tmp_path: Path, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = grouped_manifest(tmp_path)
    (tmp_path / "smoke").mkdir()
    atomic_json(tmp_path / "smoke/result.json", {"passed": True, "snapshot_identity": "snapshot"})
    original_accounts = {"80": "historical-account", "100": "historical-account"}
    atomic_json(tmp_path / "account-routing.json", original_accounts)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda *a: None)
    monkeypatch.setattr(campaign, "load_config", lambda *a: tiny_config)
    monkeypatch.setattr(campaign, "scheduler_arguments", lambda *a: ["sbatch"])
    monkeypatch.setattr(campaign, "run_result", lambda *a: {"status": "pending"})
    requests = []

    def submit(arguments: list[str], script: Path) -> dict:
        assert os.environ["account"] == "new-account"
        assert not any(arg.startswith("--dependency=") for arg in arguments)
        requests.append(arguments)
        return {"job_id": str(100 + len(requests))}

    monkeypatch.setattr(campaign, "submit_batch", submit)
    receipts = campaign.submit(tmp_path, manifest, account_override="new-account", concurrent=True)
    assert len(receipts) == 8 and all(r["depends_on"] == [] for r in receipts)
    assert json.loads((tmp_path / "account-routing.json").read_text()) == original_accounts
    assert json.loads((tmp_path / "submission-policy.json").read_text()) == {
        "account": "new-account",
        "concurrent": True,
    }
    monkeypatch.setattr(campaign, "active_job", lambda job: True)
    campaign.submit(tmp_path, manifest)
    assert len(requests) == 8
    with pytest.raises(ValueError, match="cancel active jobs"):
        campaign.submit(tmp_path, manifest, account_override="another-account")
    monkeypatch.setattr(campaign, "active_job", lambda job: job != "101")
    campaign.submit(tmp_path, manifest)  # No flags: retries retain the new policy.
    assert len(requests) == 9


def test_queued_jobs_require_smoke_success_in_scheduler_and_worker(
    tmp_path: Path, tiny_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = grouped_manifest(tmp_path)
    manifest["submissions"]["smoke"] = [{"status": "submitted", "job_id": "99"}]
    atomic_json(tmp_path / "account-routing.json", {"80": "same-account", "100": "same-account"})
    monkeypatch.setattr(campaign, "verify_snapshot", lambda *a: None)
    monkeypatch.setattr(campaign, "load_config", lambda *a: tiny_config)
    monkeypatch.setattr(campaign, "scheduler_arguments", lambda *a: ["sbatch"])
    monkeypatch.setattr(campaign, "run_result", lambda *a: {"status": "pending"})
    monkeypatch.setattr(campaign, "active_job", lambda job: True)
    requests = []

    def submit(arguments: list[str], script: Path) -> dict:
        requests.append(arguments)
        return {"job_id": str(100 + len(requests))}

    monkeypatch.setattr(campaign, "submit_batch", submit)
    with pytest.raises(ValueError, match="smoke"):
        campaign.submit(tmp_path, manifest)
    receipts = campaign.submit(tmp_path, manifest, wait_for_smoke=True)
    assert len(receipts) == 8
    assert all(r["requires_smoke_job"] == "99" for r in receipts)
    assert all(any(arg.startswith("--dependency=") and "afterok:99" in arg for arg in args) for args in requests)
    assert all("--kill-on-invalid-dep=yes" in args for args in requests)
    with pytest.raises(FileNotFoundError):
        campaign.lane(tmp_path, manifest, 100, "sync")
    (tmp_path / "smoke").mkdir()
    atomic_json(tmp_path / "smoke/result.json", {"passed": True, "snapshot_identity": "wrong-snapshot"})
    (tmp_path / "environment.sh").write_text(f"export DECENTRALIZED_PYTHON={shlex.quote(sys.executable)}\n")
    (tmp_path / "evaluation-source").mkdir()
    batch = subprocess.run(["bash", receipts[0]["script"]], capture_output=True, text=True)
    assert batch.returncode != 0
    assert "matching successful GPU smoke evidence is required" in batch.stderr
    with pytest.raises(ValueError, match="matching successful"):
        campaign.submit(tmp_path, manifest, wait_for_smoke=True)
    with pytest.raises(ValueError, match="matching successful"):
        campaign.lane(tmp_path, manifest, 100, "sync")


def test_training_retry_reuses_complete_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    atomic_json(run / "final/manifest.json", {"step": 2})
    atomic_json(run / "result.json", {"validation": {"loss": 3.0}})
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw: pytest.fail("must not retrain"))
    assert campaign.train_run(tmp_path, {"run": str(run)}) == run / "final"


def test_partial_results_are_not_complete(tmp_path: Path) -> None:
    row = {
        "key": "x",
        "budget": 20,
        "method": "sync",
        "seed": 42,
        "historical_loss": 3.0,
        "run": str(tmp_path / "run"),
        "quality": str(tmp_path / "quality"),
    }
    Path(row["run"]).mkdir()
    Path(row["quality"]).mkdir()
    atomic_json(Path(row["run"]) / "result.json", {"validation": {"loss": 3.01}})
    atomic_json(Path(row["quality"]) / "results.json", {"complete": False, "metrics": {}})
    result = campaign.run_result(row)
    assert result["status"] == "trained" and result["review_validation"]
    assert json.loads((Path(row["run"]) / "result.json").read_text())["validation"]["loss"] == 3.01


def test_report_aggregates_available_seeds_without_claiming_complete_groups() -> None:
    rows = []
    for method, values in [("sync", [1.0, 2.0, 3.0]), ("cmp", [1.0, 3.0]), ("exp", [2.0]), ("ring", [])]:
        for seed in campaign.SEEDS:
            index = seed - 42
            row = {
                "budget": 100,
                "method": method,
                "seed": seed,
                "status": "complete" if index < len(values) else "pending",
            }
            if row["status"] == "complete":
                row.update({metric: values[index] for metric in campaign.METRICS})
            rows.append(row)
    groups = campaign.summarize_results(rows)
    assert [g["method"] for g in groups] == list(campaign.METHOD_ORDER)
    assert [g["completed_seeds"] for g in groups] == [3, 2, 1, 0]
    assert [g["complete"] for g in groups] == [True, False, False, False]
    assert groups[0]["metrics"]["blimp_acc"] == {"mean": 2.0, "sample_std": 1.0}
    assert groups[1]["metrics"]["blimp_acc"]["mean"] == 2.0
    assert groups[1]["metrics"]["blimp_acc"]["sample_std"] == pytest.approx(2**0.5)
    assert groups[2]["metrics"]["blimp_acc"] == {"mean": 2.0, "sample_std": None}
    assert groups[3]["metrics"] == {}
