import copy
import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from decentralized_llm.config import SlurmLauncher, load_config, save_config
from decentralized_llm.data.cache import file_digest
from decentralized_llm.launch.slurm import SubmissionRejected
from decentralized_llm.runtime import atomic_json
from scripts import benchmark_training_campaign as campaign


def single_case(tmp_path: Path) -> tuple[dict, dict, dict]:
    configs, allocations = campaign.matrix(tmp_path, tmp_path / "cache")
    allocation = allocations[0]
    allocation.update(state="COMPLETED", job_id="123")
    key = allocation["cases"][0]
    config = configs[key]
    directory = config.runtime.output_dir
    directory.mkdir(parents=True)
    filename = directory / "config.yaml"
    save_config(config, filename)
    case = {
        "key": key,
        "allocation_id": allocation["id"],
        "model": "20m",
        "context_length": 512,
        "global_batch_size": 32,
        "error_feedback_gamma": 0.0,
        "variant": "sync",
        "directory": str(directory),
        "config": str(filename),
        "config_sha256": file_digest(filename),
        "execution_record": str(directory / "execution.json"),
    }
    manifest = {
        "source_sha256": "source",
        "lock_sha256": "lock",
        "cache_identity": "cache",
        "cases": [case],
        "allocations": [allocation],
    }
    return case, allocation, manifest


def completed_reports(case: dict, allocation: dict, manifest: dict) -> list[dict]:
    config = load_config(case["config"])
    ranks = []
    for rank in range(8):
        ranks.append(
            {
                "rank": rank,
                "hostname": f"node{rank // 4}",
                "local_rank": rank % 4,
                "gpu_uuid": f"gpu-{rank}",
                "gpu": "NVIDIA GH200 120GB",
                "peak_allocated_bytes": 1000 + rank,
                "peak_reserved_bytes": 2000 + rank,
                "network_environment": {"NCCL_NET": "AWS Libfabric", "FI_PROVIDER": "cxi", "OFI_NCCL_PROTOCOL": "RDMA"},
                "loaded_libraries": ["/lib/libnccl-net-ofi.so", "/lib/libfabric.so.1", "/lib/libcxi.so.1"],
            }
        )
    windows = []
    for seconds in (15.0, 16.0, 17.0):
        tokens = 600 * config.training.batch_tokens
        windows.append(
            {
                "updates": 600,
                "seconds": seconds,
                "global_tokens": tokens,
                "global_sequences": 600 * 32,
                "seconds_per_update": seconds / 600,
                "global_sequences_per_second": 600 * 32 / seconds,
                "per_gpu_tokens_per_second": tokens / seconds / 8,
                "global_tokens_per_second": tokens / seconds,
            }
        )
    reports = [
        {
            **rank,
            "world_size": 8,
            "physical_world_size": 8,
            "physical_gpu_count": 8,
            "training_backend": "distributed",
            "config": config.model_dump(mode="json"),
            "environment": {"source_sha256": "source", "lock_sha256": "lock"},
            "cache_identity": "cache",
            "slurm_job_id": allocation["job_id"],
            "execution": {"backend": "gh200", "compile": True},
            "resident_at_measurement": True,
            "measurement": {"valid": True, "invalid_reasons": []},
            "windows": windows,
            "rank_measurements": ranks,
        }
        for rank in ranks
    ]
    save_reports(case, reports)
    atomic_json(
        Path(case["execution_record"]),
        {
            "state": "COMPLETED",
            "exit_code": 0,
            "started": True,
            "allocation_id": allocation["id"],
            "trial_id": case["key"],
            "attempt": 1,
        },
    )
    return copy.deepcopy(reports)


def save_reports(case: dict, reports: list[dict]) -> None:
    for rank, report in enumerate(reports):
        atomic_json(Path(case["directory"]) / f"benchmark-rank-{rank:05d}.json", report)


def test_matrix_geometry_algorithms_and_measurement_budget(tmp_path: Path) -> None:
    from decentralized_llm.benchmarks.protocol import benchmark_bounds

    configs, allocations = campaign.matrix(tmp_path, tmp_path / "cache")
    assert len(configs) == 216 and len(allocations) == 54
    assert len({a["name"] for a in allocations}) == 54
    assert sum(a["pilot"] for a in allocations) == 4
    assert [a["id"] for a in allocations if a["pilot"]] == [
        "20m-c512-b32-ef0",
        "20m-c512-b32-ef1",
        "90m-c2048-b128-ef0",
        "90m-c2048-b128-ef1",
    ]
    for allocation in allocations:
        assert len(allocation["cases"]) == 4
        variants = [configs[key] for key in allocation["cases"]]
        assert [c.distributed.topology for c in variants] == [
            "complete",
            "complete",
            "one_peer_ring",
            "one_peer_exponential",
        ]
        assert variants[0].distributed.mode == "ddp"
        for config in variants:
            assert config.physical_nodes == 2 and config.processes_per_node == 4 and config.physical_gpu_count == 8
            assert config.runtime.training_backend == "distributed" and config.runtime.execution_backend == "gh200"
            assert config.runtime.compile is True and config.runtime.amp
            assert config.data.loader_mode == "resident" and config.benchmark.wait_for_resident
            assert config.training.num_accumulation == 1 and config.micro_batch_size in (4, 8, 16)
            assert config.training.tokens_per_parameters == 20 and config.training.checkpoint_policy == "none"
            assert config.benchmark.warmup == 30 and config.benchmark.steps == 60
            assert config.benchmark.windows == 3 and config.benchmark.min_seconds == 15
            assert isinstance(config.launcher, SlurmLauncher)
            assert not config.benchmark.profile and config.launcher.time == "00:07:00"
            per_window_tokens = config.benchmark.max_chunks * 60 * config.training.batch_tokens
            assert per_window_tokens >= 128 * 1024**2
            assert per_window_tokens < 128 * 1024**2 + 60 * config.training.batch_tokens
            assert benchmark_bounds(config).total_blocks > 0
    summary = campaign.plan_summary(configs, allocations)
    assert summary["minimum_measurement_gpu_hours"] == 21.6 and summary["max_jobs"] == 8


def test_prepare_is_nonmutating_when_dry_and_freezes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, cache = tmp_path / "campaign", tmp_path / "cache"
    cache.mkdir()
    atomic_json(cache / "manifest.json", {"identity": "cache"})
    token_cache = Mock(manifest={"identity": "cache"})
    monkeypatch.setattr(campaign, "TokenCache", lambda _: token_cache)
    checks = []
    monkeypatch.setattr(campaign, "validate", lambda config, **kw: checks.append((config, kw)))
    plan = campaign.prepare(root, cache, dry_run=True)
    assert plan["cases"] == 216 and not root.exists()
    token_cache.verify.assert_not_called()
    assert len(checks) == 216 and all(kw == {"verify": False, "benchmark": True} for _, kw in checks)
    manifest = campaign.prepare(root, cache)
    token_cache.verify.assert_called_once()
    assert len(manifest["cases"]) == 216 and len(manifest["allocations"]) == 54
    campaign.verify_snapshot(manifest)
    repeated = campaign.prepare(root, cache)
    assert repeated == manifest
    token_cache.verify.assert_called_once()
    allocation = manifest["allocations"][0]
    assert "--nodes=2" in allocation["arguments"] and "--gpus-per-node=nvidia_gh200_120gb:4" in allocation["arguments"]
    assert "--time=00:30:00" in allocation["arguments"] and "--network=job_vni" in allocation["arguments"]
    steps = json.loads((Path(allocation["directory"]) / "steps.json").read_text())["steps"]
    assert len(steps) == 4 and all(s["time_minutes"] == 7 for s in steps)
    first = manifest["cases"][0]
    assert "--time=00:07:00" in (Path(first["directory"]) / "step.sh").read_text()
    worker = (Path(first["directory"]) / "worker.sh").read_text()
    assert "--action benchmark-training" in worker and "--nnodes=2 --nproc-per-node=4" in worker
    assert "--numa-binding=exclusive" in worker
    assert "PYTHONPATH=" in (root / "source/environment.sh").read_text()
    Path(first["config"]).write_text("changed: true\n")
    with pytest.raises(ValueError, match="frozen configuration"):
        campaign.prepare(root, cache)


def test_complete_result_aggregates_all_ranks_and_preserves_provenance(tmp_path: Path) -> None:
    case, allocation, manifest = single_case(tmp_path)
    completed_reports(case, allocation, manifest)
    result = campaign.case_result(case, allocation, manifest)
    assert result["status"] == "complete", result
    assert result["max_peak_allocated_bytes"] == 1007 and result["max_peak_reserved_bytes"] == 2007
    assert result["sum_peak_allocated_bytes"] == sum(range(1000, 1008))
    assert result["global_tokens_per_second"] == 600 * 16384 / 16
    assert result["global_sequences_per_second"] == 600 * 32 / 16
    assert result["seconds_per_update"] == pytest.approx(16 / 600)
    assert result["per_gpu_tokens_per_second"] == result["global_tokens_per_second"] / 8
    assert result["source_sha256"] == "source" and result["config_sha256"] == case["config_sha256"]
    assert len(result["rank_memory"]) == 8 and len(result["transport"]) == 8
    assert result["gpu_names"] == ["NVIDIA GH200 120GB"]
    assert campaign.summarize(manifest)["pilot_passed"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ("compile", "not resident and valid"),
        ("source", "source fingerprint"),
        ("placement", "eight distinct GPU"),
        ("host", "two hosts"),
        ("transport", "RDMA environment"),
        ("libraries", "libraries were not loaded"),
        ("timing", "rank timing windows disagree"),
        ("tokens", "token accounting"),
        ("config", "configuration mismatch"),
    ],
)
def test_report_validation_fails_closed(tmp_path: Path, change: str, reason: str) -> None:
    case, allocation, manifest = single_case(tmp_path)
    reports = completed_reports(case, allocation, manifest)
    if change == "compile":
        reports[1]["measurement"] = {"valid": False, "invalid_reasons": ["compilation during timing"]}
    elif change == "source":
        reports[1]["environment"]["source_sha256"] = "different"
    elif change == "placement":
        reports[0]["rank_measurements"][7]["gpu_uuid"] = "gpu-0"
    elif change == "host":
        reports[0]["rank_measurements"][7]["hostname"] = "node2"
    elif change == "transport":
        reports[0]["rank_measurements"][3]["network_environment"]["FI_PROVIDER"] = "tcp"
    elif change == "libraries":
        reports[0]["rank_measurements"][3]["loaded_libraries"] = []
    elif change == "timing":
        reports[1]["windows"] = copy.deepcopy(reports[1]["windows"])
        reports[1]["windows"][0]["seconds"] = 20
    elif change == "tokens":
        reports[0]["windows"][0]["global_tokens"] = 0
    elif change == "config":
        reports[1]["config"]["training"]["num_accumulation"] = 2
    save_reports(case, reports)
    result = campaign.case_result(case, allocation, manifest)
    assert result["status"] == "invalid" and reason in result["reason"]
    assert campaign.summarize(manifest)["pilot_failed"]


def test_partial_reports_clean_exit_and_failure_classification(tmp_path: Path) -> None:
    case, allocation, manifest = single_case(tmp_path)
    completed_reports(case, allocation, manifest)
    path = Path(case["directory"]) / "benchmark-rank-00007.json"
    path.unlink()
    assert campaign.case_result(case, allocation, manifest)["status"] == "failed"
    for text, status in (
        ("torch.OutOfMemoryError: CUDA out of memory", "oom"),
        ("nonfinite gradient norm", "nonfinite"),
        ("DUE TO TIME LIMIT", "timeout"),
    ):
        (Path(case["directory"]) / "stderr.log").write_text(text)
        atomic_json(
            Path(case["execution_record"]),
            {
                "state": "FAILED",
                "exit_code": 1,
                "allocation_id": allocation["id"],
                "trial_id": case["key"],
                "attempt": 1,
            },
        )
        assert campaign.case_result(case, allocation, manifest)["status"] == status
    Path(case["execution_record"]).unlink()
    allocation["state"] = "TIMEOUT"
    assert campaign.case_result(case, allocation, manifest)["status"] == "timeout"


def test_duration_bound_failure_is_invalid_and_exports_effective_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, allocation, manifest = single_case(tmp_path)
    config = load_config(case["config"])
    config.benchmark.max_chunks = 18
    save_config(config, Path(case["config"]))
    case["config_sha256"] = file_digest(Path(case["config"]))
    message = "bounded benchmark window did not reach its minimum duration (13.514s < 15.0s after 18 chunks)"
    (Path(case["directory"]) / "stderr.log").write_text(
        '    f"bounded benchmark window did not reach its minimum duration "\n'
        f"[rank5]: RuntimeError: {message}\n"
        "srun: Terminating StepId=123.1\n"
    )
    atomic_json(
        Path(case["execution_record"]),
        {"state": "FAILED", "exit_code": 1, "allocation_id": allocation["id"], "trial_id": case["key"], "attempt": 1},
    )
    result = campaign.case_result(case, allocation, manifest)
    assert result["status"] == "invalid" and result["reason"] == message
    assert result["global_tokens_per_second"] is None and result["per_gpu_tokens_per_second"] is None
    assert result["benchmark_warmup_updates"] == 30 and result["benchmark_updates_per_chunk"] == 60
    assert result["benchmark_max_chunks"] == 18 and result["benchmark_min_seconds"] == 15
    assert result["benchmark_windows"] == 3 and result["benchmark_max_updates_per_window"] == 1080
    assert result["benchmark_max_global_tokens_per_window"] == 1080 * config.training.batch_tokens
    assert result["benchmark_maximum_updates"] == 30 + 60 * (3 * 18 + 1)
    assert (
        result["benchmark_maximum_global_tokens"] == result["benchmark_maximum_updates"] * config.training.batch_tokens
    )
    assert campaign.summarize(manifest)["pilot_failed"]
    monkeypatch.setattr(campaign, "plots", lambda *_: None)
    campaign.write_report(tmp_path, manifest)
    saved = json.loads((tmp_path / "report.json").read_text())["rows"][0]
    assert saved["reason"] == message and saved["benchmark_max_chunks"] == 18
    with (tmp_path / "report.csv").open(newline="") as handle:
        exported = next(csv.DictReader(handle))
    assert exported["status"] == "invalid" and exported["reason"] == message
    assert exported["benchmark_min_seconds"] == "15.0" and exported["global_tokens_per_second"] == ""


def test_pending_cases_also_report_configured_measurement_bounds(tmp_path: Path) -> None:
    case, allocation, manifest = single_case(tmp_path)
    allocation["state"] = "PLANNED"
    result = campaign.case_result(case, allocation, manifest)
    assert result["status"] == "pending"
    assert result["benchmark_min_seconds"] == 15 and result["benchmark_windows"] == 3
    assert result["benchmark_max_global_tokens_per_window"] >= 128 * 1024**2


def submission_fixture(tmp_path: Path) -> dict:
    _, allocations = campaign.matrix(tmp_path, tmp_path / "cache")
    for allocation in allocations:
        Path(allocation["directory"]).mkdir(parents=True)
        allocation.update(arguments=["sbatch"], script=str(Path(allocation["directory"]) / "submit.sh"))
    return {"allocations": allocations}


def test_pilots_gate_and_submission_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = submission_fixture(tmp_path)
    calls = []

    def submit(arguments: list[str], path: Path) -> dict:
        saved = json.loads((tmp_path / "campaign.json").read_text())
        assert any(a["state"] == "SUBMITTING" and a["script"] == str(path) for a in saved["allocations"])
        calls.append(path)
        return {"job_id": str(len(calls)), "cluster": None}

    monkeypatch.setattr(campaign, "submit_batch", submit)
    campaign.submit_ready(tmp_path, manifest, {"pilot_passed": False, "pilot_failed": False})
    assert len(calls) == 4 and all(a["pilot"] for a in manifest["allocations"] if a["state"] == "SUBMITTED")
    campaign.submit_ready(tmp_path, manifest, {"pilot_passed": False, "pilot_failed": False})
    assert len(calls) == 4
    for allocation in manifest["allocations"]:
        if allocation["pilot"]:
            allocation["state"] = "COMPLETED"
    campaign.submit_ready(tmp_path, manifest, {"pilot_passed": True, "pilot_failed": False})
    assert len(calls) == 12
    assert sum(a["state"] == "SUBMITTED" for a in manifest["allocations"]) == 8


def test_unknown_submission_is_not_duplicated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = submission_fixture(tmp_path)
    submit = Mock(side_effect=OSError("submission transport interrupted"))
    monkeypatch.setattr(campaign, "submit_batch", submit)
    with pytest.raises(OSError):
        campaign.submit_ready(tmp_path, manifest, {"pilot_passed": False, "pilot_failed": False})
    assert manifest["allocations"][0]["state"] == "UNKNOWN"
    monkeypatch.setattr(campaign, "scheduler_status", lambda _: {})
    campaign.reconcile(manifest)
    submit.side_effect = None
    submit.return_value = {"job_id": "other", "cluster": None}
    campaign.submit_ready(tmp_path, manifest, {"pilot_passed": False, "pilot_failed": False})
    assert submit.call_count == 4  # Three other pilots can be submitted; the unresolved first one cannot.
    assert manifest["allocations"][0]["state"] == "UNKNOWN"


def test_reconcile_recovers_scheduler_receipt_and_readonly_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = submission_fixture(tmp_path)
    allocation = manifest["allocations"][0]
    allocation["state"] = "SUBMITTING"
    monkeypatch.setattr(
        campaign,
        "scheduler_status",
        lambda _: {allocation["name"]: {"state": "RUNNING", "job_id": "456", "cluster": None, "exit_code": None}},
    )
    snapshot = copy.deepcopy(manifest)
    campaign.reconcile(snapshot, persist_receipts=False)
    assert not Path(allocation["receipt_path"]).exists()
    campaign.reconcile(manifest)
    assert allocation["state"] == "RUNNING" and allocation["job_id"] == "456"
    assert json.loads(Path(allocation["receipt_path"]).read_text())["recovered"]


def test_scheduler_rejection_persists_and_pilot_failure_stops_new_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = submission_fixture(tmp_path)
    submit = Mock(side_effect=SubmissionRejected("scheduler rejected allocation"))
    monkeypatch.setattr(campaign, "submit_batch", submit)
    campaign.submit_ready(tmp_path, manifest, {"pilot_passed": False, "pilot_failed": False})
    assert manifest["allocations"][0]["state"] == "REJECTED"
    campaign.submit_ready(tmp_path, manifest, {"pilot_passed": False, "pilot_failed": True})
    assert submit.call_count == 1


def test_report_keeps_failure_rows_and_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, allocation, manifest = single_case(tmp_path)
    completed_reports(case, allocation, manifest)
    plot = Mock()
    monkeypatch.setattr(campaign, "plots", plot)
    result = campaign.write_report(tmp_path, manifest)
    assert len(result["rows"]) == 1
    assert "config_sha256" in (tmp_path / "report.csv").read_text()
    assert "not total device usage" in json.loads((tmp_path / "report.json").read_text())["measurement_scope"]
    plot.assert_called_once()


def retry_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict, dict]:
    failed, allocation, manifest = single_case(tmp_path)
    completed_reports(failed, allocation, manifest)
    failure = json.loads(Path(failed["execution_record"]).read_text())
    failure.update(state="FAILED", exit_code=1)
    atomic_json(Path(failed["execution_record"]), failure)
    (Path(failed["directory"]) / "stderr.log").write_text("DUE TO TIME LIMIT\n")
    success = copy.deepcopy(failed)
    success["key"] = failed["key"].removesuffix("sync") + "complete"
    success["variant"] = "complete"
    directory = tmp_path / "cases" / success["key"]
    directory.mkdir()
    config = load_config(failed["config"])
    config.distributed.mode = "decentralized"
    config.runtime.output_dir = directory
    filename = directory / "config.yaml"
    save_config(config, filename)
    success.update(
        directory=str(directory),
        config=str(filename),
        config_sha256=file_digest(filename),
        execution_record=str(directory / "execution.json"),
    )
    completed_reports(success, allocation, manifest)
    allocation["state"] = "FAILED"
    Path(allocation["directory"]).mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    (source / "untouched.txt").write_text("original training sources")
    manifest.update(cases=[failed, success], source=str(source), launch_files={}, controller_revisions=[])
    atomic_json(tmp_path / "campaign.json", manifest)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda _: None)
    return failed, success, allocation, manifest


def test_retry_preserves_successful_evidence_and_recovers_pilot_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed, success, allocation, manifest = retry_fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(manifest)
    hashes = {
        str(path): file_digest(path) for path in tmp_path.rglob("*") if path.is_file() and path.name != "campaign.json"
    }
    assert campaign.summarize(manifest)["pilot_failed"]
    files_before = {str(path) for path in tmp_path.rglob("*")}
    preview = campaign.retry_cases(tmp_path, [failed["key"]], dry_run=True)
    assert preview["retry_allocations"][0]["attempt"] == 2
    assert preview["training_source_sha256"] == "source"
    assert files_before == {str(path) for path in tmp_path.rglob("*")}
    campaign.retry_cases(tmp_path, [failed["key"]])
    updated = json.loads((tmp_path / "campaign.json").read_text())
    retry, preserved = updated["cases"]
    current = updated["allocations"][0]
    assert len(updated["allocations"]) == 1 and current["attempt"] == 2
    assert current["history"][0]["job_id"] == "123" and current["history"][0]["state"] == "FAILED"
    assert current["active_cases"] == [failed["key"]]
    assert preserved == original["cases"][1]
    assert retry["history"][0]["config"] == failed["config"] and retry["history"][0]["job_id"] == "123"
    assert retry["history"][0]["status"] == "timeout" and "time limit" in retry["history"][0]["reason"]
    assert all(file_digest(Path(path)) == expected for path, expected in hashes.items())
    old_config = load_config(failed["config"]).model_dump(mode="json")
    new_config = load_config(retry["config"]).model_dump(mode="json")
    old_config["runtime"]["output_dir"] = new_config["runtime"]["output_dir"]
    assert old_config == new_config  # Output path is the only recipe difference.
    step_manifest = json.loads((Path(current["directory"]) / "steps.json").read_text())
    assert len(step_manifest["steps"]) == 1 and step_manifest["steps"][0]["attempt"] == 2
    current.update(job_id="456", state="COMPLETED")
    completed_reports(retry, current, updated)
    execution = json.loads(Path(retry["execution_record"]).read_text())
    execution["attempt"] = 2
    atomic_json(Path(retry["execution_record"]), execution)
    assert campaign.case_result(preserved, current, updated)["job_id"] == "123"
    assert campaign.case_result(preserved, current, updated)["status"] == "complete"
    assert campaign.case_result(retry, current, updated)["job_id"] == "456"
    summary = campaign.summarize(updated)
    assert summary["pilot_passed"] and not summary["pilot_failed"]
    assert len(summary["allocation_attempts"]) == 2
    assert [row["job_id"] for row in summary["allocation_attempts"]] == ["123", "456"]


@pytest.mark.parametrize("state", ["UNKNOWN", "SUBMITTING", "RUNNING", "PLANNED"])
def test_retry_rejects_unresolved_or_active_allocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    failed, _, allocation, manifest = retry_fixture(tmp_path, monkeypatch)
    allocation["state"] = state
    atomic_json(tmp_path / "campaign.json", manifest)
    with pytest.raises(ValueError, match="known terminal"):
        campaign.retry_cases(tmp_path, [failed["key"]], dry_run=True)


def test_retry_extends_measurement_capacity_after_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed, success, _, _ = retry_fixture(tmp_path, monkeypatch)
    memory = {"total": 1800 * 1024**2, "resident_tokens": 1600 * 1024**2}
    preflight = Mock(return_value={"memory_bytes_per_rank": memory})
    monkeypatch.setattr(campaign, "validate", preflight)
    old_config = load_config(failed["config"]).model_dump(mode="json")
    original = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    preview = campaign.retry_cases(tmp_path, [failed["key"]], dry_run=True, max_chunks_factor=2)
    adjustment = preview["measurement_changes"][failed["key"]]
    assert adjustment["previous_max_chunks"] == old_config["benchmark"]["max_chunks"]
    assert adjustment["max_chunks"] == 2 * adjustment["previous_max_chunks"]
    assert adjustment["loader_memory_bytes_per_rank"] == memory
    assert not preview["training_settings_changed"]
    assert original == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert preflight.call_args.kwargs == {"verify": False, "benchmark": True}
    campaign.retry_cases(tmp_path, [failed["key"]], max_chunks_factor=2)
    updated = json.loads((tmp_path / "campaign.json").read_text())
    retry, preserved = updated["cases"]
    assert preserved == success and retry["measurement_adjustment"] == adjustment
    new_config = load_config(retry["config"]).model_dump(mode="json")
    old_config["runtime"]["output_dir"] = new_config["runtime"]["output_dir"]
    old_config["benchmark"]["max_chunks"] *= 2
    assert old_config == new_config  # Only the iteration ceiling and output path changed.
    assert retry["history"][0]["config"] == failed["config"]
    result = campaign.case_result(retry, updated["allocations"][0], updated)
    assert result["measurement_adjustment"] == adjustment
    assert result["benchmark_max_chunks"] == adjustment["max_chunks"]


def test_retry_expansion_memory_failure_leaves_campaign_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed, _, _, _ = retry_fixture(tmp_path, monkeypatch)
    (tmp_path / ".controller.lock").touch()
    original = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(campaign, "validate", Mock(side_effect=ValueError("resident loader exceeds budget")))
    with pytest.raises(ValueError, match="exceeds budget"):
        campaign.retry_cases(tmp_path, [failed["key"]], max_chunks_factor=2)
    assert original == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_retry_rejects_invalid_chunk_factor(tmp_path: Path, factor: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        campaign.retry_cases(tmp_path, ["case"], max_chunks_factor=factor)  # type: ignore[arg-type]


def test_retry_refuses_successful_cases_and_controller_revision_preserves_training_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, success, _, manifest = retry_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not a terminal failure"):
        campaign.retry_cases(tmp_path, [success["key"]], dry_run=True)
    script = tmp_path / "revised-controller.py"
    script.write_text("# revised supervision only\n")
    process = Mock(pid=98765)
    launch = Mock(return_value=process)
    monkeypatch.setattr(campaign.subprocess, "Popen", launch)
    monkeypatch.setattr(campaign, "process_identity", lambda _: "process-birth")
    result = campaign.detach(tmp_path, 30, controller_script=script)
    updated = json.loads((tmp_path / "campaign.json").read_text())
    assert updated["source_sha256"] == manifest["source_sha256"] == "source"
    assert updated["source"] == manifest["source"]
    assert len(updated["controller_revisions"]) == 1
    revision = updated["controller_revisions"][0]
    assert revision["sha256"] == file_digest(script) and file_digest(Path(revision["path"])) == file_digest(script)
    assert result["script"] == revision["path"] and launch.call_args.args[0][1] == revision["path"]
    assert launch.call_args.kwargs["env"]["PYTHONPATH"] == str(Path(manifest["source"]) / "src")
    (tmp_path / "controller-process.json").unlink()
    restarted = campaign.detach(tmp_path, 30)
    assert restarted["script"] == revision["path"] and launch.call_args.args[0][1] == revision["path"]
