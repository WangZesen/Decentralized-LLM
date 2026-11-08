import copy
import itertools
import json
import math
from pathlib import Path

import pytest
from test_tuning import FakeBackend

from decentralized_llm.config import Config, SlurmLauncher, load_config, save_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning import controller, report
from decentralized_llm.tuning.spec import Record, TuningSpec, expand, fingerprint, numeric_recipe
from scripts import tune_data_reuse as campaign


class CampaignBackend(FakeBackend):
    target_beta1 = 0.9

    def publish_outcome(self, attempt: Record) -> bool:
        failed = super().publish_outcome(attempt)
        if failed:
            return True
        cfg = load_config(Path(attempt["directory"]) / "config.yaml")
        path = cfg.runtime.output_dir / "result.json"
        result = json.loads(path.read_text())
        result["validation"]["loss"] = (
            3
            + math.log2(cfg.optimizer.lr / 0.001) ** 2
            + 100 * (cfg.optimizer.beta1 - self.target_beta1) ** 2
            + 100 * (cfg.optimizer.beta2 - 0.95) ** 2
            + cfg.runtime.seed / 100000
        )
        result["validation"]["consensus_error"] = 0.0
        result["session_seconds"] = 60.0
        atomic_json(path, result)
        return False


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch, tiny_config: Config, cache: TokenCache) -> CampaignBackend:
    backend = CampaignBackend()
    monkeypatch.setattr(controller, "Backend", lambda: backend)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    monkeypatch.setattr(report, "write_heatmaps", lambda *_: [])
    monkeypatch.setenv("account", "test")
    monkeypatch.setattr(controller.shutil, "which", lambda _: "/bin/true")
    setup = tiny_config.data.cache_dir.parent / "environment.sh"
    setup.write_text("#!/bin/bash\n")
    tiny_config.launcher = SlurmLauncher(partition="gpu", time="00:30:00", environment_script=setup)
    tiny_config.training.data_reuse_factor = 2
    return backend


def condition(tmp_path: Path, cfg: Config) -> tuple[Record, Record]:
    directory = tmp_path / "condition"
    directory.mkdir()
    save_config(cfg, directory / "base.yaml")
    row = {
        "key": "condition",
        "directory": str(directory),
        "config": cfg.model_dump(mode="json"),
        "batch": 64,
        "method": "sync",
        "gamma": 0,
        "reuse": cfg.training.data_reuse_factor,
        "grid": campaign.local_grid(cfg),
    }
    manifest = {"provenance": controller.provenance(TokenCache(cfg.data.cache_dir)), "conditions": [row]}
    return manifest, row


def synthetic_references() -> list[Record]:
    references = []
    for batch, method, gamma in itertools.product((64, 128), campaign.METHODS, (0, 1)):
        cfg = Config()
        cfg.model.context_length = 512
        cfg.training.batch_tokens = batch * 512
        cfg.optimizer.lr = 0.0078125
        cfg.distributed.error_feedback_gamma = gamma
        cfg.distributed.mode = "ddp" if method == "sync" else "decentralized"
        cfg.distributed.gpus_per_node = 1 if method == "sync" else 8
        references.append(
            {
                "batch": batch,
                "method": method,
                "gamma": gamma,
                "config": cfg.model_dump(mode="json"),
                "candidate_id": f"{batch}-{method}-{gamma}",
                "files": {},
                "trials": [],
            }
        )
    return references


@pytest.fixture
def matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Record:
    index = tmp_path / "references.csv"
    index.write_text("reference fixture")
    monkeypatch.setattr(campaign, "reference_candidates", lambda *_: synthetic_references())
    return campaign.plan_campaign(tmp_path / "campaign", index, tmp_path)


def test_matrix_and_global_batch_contract(matrix: Record) -> None:
    assert len(matrix["references"]) == 16
    assert len(matrix["conditions"]) == 48
    assert sum(row["reuse"] > 1 for row in matrix["conditions"]) == 32
    for row in matrix["conditions"]:
        cfg = Config.model_validate(row["config"])
        assert cfg.training.batch_tokens == 512 * row["batch"]
        assert cfg.physical_gpu_count == cfg.physical_world_size == 1
        assert cfg.distributed.world_size == (1 if row["method"] == "sync" else 8)
        assert cfg.training.tokens_per_parameters == 20
        assert cfg.training.num_accumulation == 1
        assert cfg.runtime.training_backend == "packed" and cfg.runtime.execution_backend == "gh200"
        assert cfg.runtime.amp and cfg.optimizer.grad_clip == 1 and cfg.optimizer.weight_decay == 0.1
        assert cfg.distributed.error_feedback_gamma == row["gamma"]
        assert cfg.training.data_reuse_factor == row["reuse"]
        spec = campaign.phase_spec(Path(row["directory"]) / "base.yaml", cfg, joint=row["reuse"] > 1)
        assert spec["controller"]["max_jobs"] == 25
        assert spec["controller"]["max_retries"] == 1
        assert len(expand(cfg, TuningSpec.model_validate(spec).phases[0])) == (27 if row["reuse"] > 1 else 1)


def test_grid_preserves_center_exactly(tiny_config: Config, tmp_path: Path) -> None:
    tiny_config.optimizer.beta1 = 0.974003746425297
    tiny_config.optimizer.beta2 = 0.999949910146738
    grid = campaign.local_grid(tiny_config)
    for field in campaign.FIELDS:
        assert grid[f"optimizer.{field}"][1] == getattr(tiny_config.optimizer, field)
    spec = TuningSpec.model_validate(campaign.phase_spec(tmp_path / "base.yaml", tiny_config, joint=True))
    candidates = expand(tiny_config, spec.phases[0])
    center = fingerprint(numeric_recipe(tiny_config))
    assert len(candidates) == 27 and sum(c["id"] == center for c in candidates) == 1


def test_dry_prepare_does_not_create_files(matrix: Record, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(matrix["root"])
    monkeypatch.setattr(campaign, "plan_campaign", lambda *_: matrix)
    monkeypatch.setattr(campaign, "validate_plan", lambda *_, **__: [])
    monkeypatch.setattr(campaign, "snapshot", lambda *_: pytest.fail("dry run attempted a snapshot"))
    result = campaign.prepare(root, Path(matrix["index"]), root.parent, dry_run=True)
    assert result["dry_run"] and not root.exists()


def test_frozen_manifest_and_files_are_checked(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_text("original")
    manifest = {
        "source": str(tmp_path),
        "source_files": {"input": campaign.file_digest(path)},
        "frozen_files": {},
    }
    manifest["contract_sha256"] = fingerprint(manifest)
    campaign.verify_manifest(manifest)
    changed = {**manifest, "max_studies": 3}
    with pytest.raises(ValueError, match="manifest changed"):
        campaign.verify_manifest(changed)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        campaign.verify_manifest(manifest)


def test_current_tree_snapshot_includes_uncommitted_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project, source = tmp_path / "project", tmp_path / "snapshot"
    project.mkdir()
    (project / "new.py").write_text("uncommitted = True\n")
    (project / "old.py").write_text("edited = True\n")
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *_, **__: b"old.py\x00new.py\x00")
    hashes = campaign.snapshot(project, source)
    assert (source / "new.py").read_text() == "uncommitted = True\n"
    assert (source / "old.py").read_text() == "edited = True\n"
    assert set(hashes) == {"old.py", "new.py", "environment.sh"}


def test_historical_schema_and_relocation_are_read_only(tmp_path: Path, tiny_config: Config) -> None:
    original = tiny_config.model_dump(mode="json")
    original["distributed"].pop("error_feedback_gamma")
    original["distributed"]["error_feedback"] = True
    saved = copy.deepcopy(original)
    assert campaign.historical_config(original).distributed.error_feedback_gamma == 1
    assert original == saved
    original["distributed"]["error_feedback_gamma"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        campaign.historical_config(original)
    study = tmp_path / "batch" / "study"
    run = study / "trials" / "phase" / "trial" / "attempt-001" / "run"
    run.mkdir(parents=True)
    (run / "result.json").write_text("{}")
    trial = {"attempts": [{"directory": str(tmp_path / "old" / "study/trials/phase/trial/attempt-001")}]}
    relocated = campaign.relocated_trial(trial, study)
    assert Path(relocated["attempts"][-1]["directory"]) == run.parent
    assert trial != relocated


def test_condition_reuses_fixed_center_and_resumes_without_submitting(
    tmp_path: Path,
    tiny_config: Config,
    scheduler: CampaignBackend,
) -> None:
    manifest, row = condition(tmp_path, tiny_config)
    campaign.run_condition(manifest, row)
    states = campaign.condition_states(row)
    trials, _ = campaign.collect_trials(states, manifest["provenance"], authenticate=True)
    assert len(trials) == len(scheduler.submissions) == 33
    assert len(states[0]["trials"]) == 3
    assert states[1]["phases"][0]["reused_trials"] == 1
    assert states[1]["phases"][1]["reused_trials"] == 5
    assert scheduler.max_active <= 25
    assert campaign.read_json(Path(row["directory"]) / "boundary-plan.json")["probes"] == []
    campaign.run_condition(manifest, row)
    assert len(scheduler.submissions) == 33
    directory = Path(row["directory"]) / "joint"
    changed = controller.load_state(directory)
    changed["spec"]["controller"]["max_jobs"] = 50
    controller.save_state(directory, changed)
    with pytest.raises(ValueError, match="frozen recipe"):
        campaign.run_condition(manifest, row)


def test_retry_and_numerical_divergence_are_separate(
    tmp_path: Path,
    tiny_config: Config,
    scheduler: CampaignBackend,
) -> None:
    scheduler.failures = 1
    scheduler.diverges = lambda cfg: cfg.runtime.seed == 44
    tiny_config.training.data_reuse_factor = 1
    manifest, row = condition(tmp_path, tiny_config)
    campaign.run_condition(manifest, row)
    trials, _ = campaign.collect_trials(campaign.condition_states(row), manifest["provenance"], authenticate=True)
    assert len(trials) == 3 and len(scheduler.submissions) == 6
    assert sorted(t["status"] for t in trials.values()) == ["diverged", "succeeded", "succeeded"]
    assert all(len(t["attempts"]) == 2 for t in trials.values())
    assert campaign.confirmed_candidates(trials, manifest["provenance"]) == []


def test_boundary_round_is_frozen_and_not_recursive(
    tmp_path: Path,
    tiny_config: Config,
    scheduler: CampaignBackend,
) -> None:
    scheduler.target_beta1 = 0.7
    manifest, row = condition(tmp_path, tiny_config)
    campaign.run_condition(manifest, row)
    directory = Path(row["directory"])
    plan = campaign.read_json(directory / "boundary-plan.json")
    assert len(plan["probes"]) == 1
    assert plan["probes"][0]["field"] == "optimizer.beta1"
    assert plan["probes"][0]["value"] == pytest.approx(0.9**4)
    trials, _ = campaign.collect_trials(campaign.condition_states(row), manifest["provenance"], authenticate=True)
    winner = campaign.confirmed_candidates(trials, manifest["provenance"])[0]
    assert winner["config"]["optimizer"]["beta1"] == pytest.approx(0.9**4)
    assert len(trials) == len(scheduler.submissions) == 38
    campaign.run_condition(manifest, row)
    assert len(scheduler.submissions) == 38
    plan["probes"][0]["value"] += 0.01
    atomic_json(directory / "boundary-plan.json", plan)
    with pytest.raises(ValueError, match="boundary plan changed"):
        campaign.run_condition(manifest, row)


def test_resume_reconciles_existing_submissions(
    tmp_path: Path,
    tiny_config: Config,
    scheduler: CampaignBackend,
) -> None:
    tiny_config.training.data_reuse_factor = 1
    manifest, row = condition(tmp_path, tiny_config)
    scheduler.interrupt = True
    with pytest.raises(RuntimeError, match="needs attention"):
        campaign.run_condition(manifest, row)
    assert len(scheduler.submissions) == 3
    scheduler.interrupt = False
    campaign.run_condition(manifest, row)
    assert len(scheduler.submissions) == 3


def test_invalid_history_cannot_be_reused(
    tmp_path: Path,
    tiny_config: Config,
    scheduler: CampaignBackend,
) -> None:
    tiny_config.training.data_reuse_factor = 1
    manifest, row = condition(tmp_path, tiny_config)
    campaign.run_condition(manifest, row)
    state = campaign.condition_states(row)[0]
    changed = copy.deepcopy(state)
    next(iter(changed["trials"].values()))["config"]["training"]["data_reuse_factor"] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        campaign.collect_trials([changed], manifest["provenance"])
    changed = copy.deepcopy(state)
    changed["provenance"]["torch"] = "different"
    with pytest.raises(ValueError, match="Mixed"):
        campaign.collect_trials([changed], manifest["provenance"])
    changed = copy.deepcopy(state)
    next(iter(changed["trials"].values()))["result"]["validation"]["validation_complete"] = False
    with pytest.raises(ValueError, match="Incomplete validation"):
        campaign.collect_trials([changed], manifest["provenance"])
    changed = copy.deepcopy(state)
    next(iter(changed["trials"].values()))["loss"] += 1
    with pytest.raises(ValueError, match="Cached result changed"):
        campaign.collect_trials([changed], manifest["provenance"], authenticate=True)
    target = tiny_config.model_copy(deep=True)
    target.training.data_reuse_factor = 2
    with pytest.raises(ValueError, match="does not belong"):
        campaign.run_phase(
            tmp_path / "wrong-reuse",
            target,
            campaign.phase_spec(tmp_path / "base.yaml", target, joint=True),
            manifest["provenance"],
            [state],
        )


@pytest.mark.parametrize("upper", [False, True])
def test_boundary_neighbors_are_axial_and_bounded(tiny_config: Config, tmp_path: Path, upper: bool) -> None:
    grid = campaign.local_grid(tiny_config)
    config = tiny_config.model_dump(mode="json")
    for field, values in grid.items():
        config["optimizer"][field.split(".")[1]] = values[-1 if upper else 0]
    state = {
        "base": tiny_config.model_dump(mode="json"),
        "winner": {"config": config},
        "spec": campaign.phase_spec(tmp_path / "base.yaml", tiny_config, joint=True),
    }
    probes = campaign.boundary_probes(state)
    assert len(probes) == 3
    for probe in probes:
        values = grid[probe["field"]]
        assert probe["value"] > max(values) if upper else probe["value"] < min(values)


def test_lane_partition_keeps_global_cap_and_failed_lane_slot(matrix: Record, monkeypatch: pytest.MonkeyPatch) -> None:
    lanes = [campaign.lane_conditions(matrix, i) for i in (0, 1)]
    assert len(lanes[0]) == len(lanes[1]) == 24
    assert not ({r["key"] for r in lanes[0]} & {r["key"] for r in lanes[1]})
    assert 2 * campaign.CONTROLLER["max_jobs"] == 50
    root = Path(matrix["root"])
    root.mkdir()
    monkeypatch.setattr(campaign, "verify_manifest", lambda _: None)
    monkeypatch.setattr(campaign, "TokenCache", lambda _: None)
    matrix["provenance"] = {}
    monkeypatch.setattr(campaign, "provenance", lambda _: {})
    called = []

    def fail(_: Record, row: Record) -> None:
        called.append(row["key"])
        raise RuntimeError("unresolved allocation")

    monkeypatch.setattr(campaign, "run_condition", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unresolved"):
            campaign.run_lane(root, matrix, 0)
    assert called == [lanes[0][0]["key"]] * 2
    with campaign.lock_file(root / ".lane-0.lock"):
        with pytest.raises(BlockingIOError):
            campaign.run_lane(root, matrix, 0)


def result(losses: list[float]) -> Record:
    return {"seeds": [{"seed": seed, "loss": loss} for seed, loss in zip(campaign.SEEDS, losses, strict=True)]}


def test_degradation_uses_paired_differences_and_matching_gamma() -> None:
    rows = []
    for method, reuse in itertools.product(("sync", "ring"), (1, 2, 4)):
        fixed = result([3 + (reuse - 1) * (0.1 if method == "sync" else 0.05) + i / 100 for i in range(3)])
        retuned = result([s["loss"] - (0.02 if reuse > 1 else 0) for s in fixed["seeds"]])
        rows.append(
            {
                "key": f"{method}-{reuse}",
                "batch": 64,
                "method": method,
                "gamma": 1,
                "reuse": reuse,
                "fixed": fixed,
                "retuned": retuned,
                "complete": True,
                "trial_statuses": {},
                "boundary_edges": [],
            }
        )
    comparisons = campaign.comparison_rows(rows)
    selected = next(r for r in comparisons if r["method"] == "ring" and r["reuse"] == 4 and r["view"] == "retuned")
    assert selected["reuse_penalty"]["mean"] == pytest.approx(0.13)
    assert selected["relative_degradation"]["mean"] == pytest.approx(-0.15)
    assert selected["retuning_benefit"]["mean"] == pytest.approx(0.02)
    assert selected["relative_degradation"]["std"] == pytest.approx(0)
    rows[0]["fixed"] = None
    assert campaign.comparison_rows(rows)[0]["reuse_penalty"] is None
    with pytest.raises(ValueError, match="three paired seeds"):
        campaign.paired_difference({"seeds": [{"seed": 42, "loss": 1}]}, result([1, 2, 3]))


def test_paired_inputs_on_small_cache(tiny_config: Config, cache: TokenCache) -> None:
    # Keep total blocks below the cache size, so reuse 1 is a genuine no-reuse reference.
    tiny_config.training.tokens_per_parameters = 0.1
    rows = []
    for batch, reuse, gamma in itertools.product((64, 128), (1, 2, 4), (0, 1)):
        cfg = tiny_config.model_copy(deep=True)
        cfg.training.data_reuse_factor = reuse
        cfg.distributed.error_feedback_gamma = gamma
        rows.append({"batch": batch, "reuse": reuse, "config": cfg.model_dump(mode="json")})
    evidence = campaign.paired_inputs({"conditions": rows})
    assert len(evidence) == 6
    assert all(len(e["sampling"]) == 3 for e in evidence)


def test_empty_report_exports_all_conditions_without_inventing_results(
    matrix: Record,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(matrix["root"])
    root.mkdir()
    matrix["provenance"] = {}
    monkeypatch.setattr(campaign, "accounting", lambda _: {"gpu_hours": 0, "complete": True})
    report = campaign.write_report(root, matrix)
    assert report["status"]["completed_conditions"] == 0
    assert len(report["comparisons"]) == 96
    assert all(row["result"] is None and row["reuse_penalty"] is None for row in report["comparisons"])
    assert (root / "results.csv").is_file() and (root / "report.json").is_file()
    assert len(list((root / "plots").glob("*.png"))) == 6
    assert len(list((root / "plots").glob("*.svg"))) == 6
