import csv
import json
from pathlib import Path

import pytest

from scripts import stability_report as report
from scripts.report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER


def measurements(losses: list[float]) -> list[report.Record]:
    return [
        {"seed": seed, "status": "succeeded", "loss": loss} for seed, loss in zip((42, 43, 44), losses, strict=True)
    ]


def test_statistics_use_sample_sd_and_paired_seed_differences() -> None:
    point = measurements([2, 4, 6])
    anchor = measurements([1, 3, 5])
    summary = report.summarize_point(point, list(reversed(anchor)), [42, 43, 44])
    assert summary["mean_loss"] == 4
    assert summary["std_loss"] == 2
    assert summary["mean_delta"] == 1
    assert summary["std_delta"] == 0
    assert summary["status"] == "complete"
    center = report.summarize_point(anchor, anchor, [42, 43, 44])
    assert center["mean_delta"] == center["std_delta"] == 0
    assert not center["better_than_anchor"]


@pytest.mark.parametrize("status", ["missing", "diverged", "invalid", "failed", "submitted"])
def test_incomplete_points_never_receive_finite_subset_aggregates(status: str) -> None:
    point = measurements([2, 4, 6])
    point[1].update(status=status, loss=None)
    result = report.summarize_point(point, measurements([1, 3, 5]), [42, 43, 44])
    assert result["completed_seeds"] == 2
    assert result["mean_loss"] is result["std_loss"] is result["mean_delta"] is result["std_delta"] is None
    assert not result["better_than_anchor"]
    assert result["status"] == (status if status in {"diverged", "invalid", "failed"} else "incomplete")


def test_missing_anchor_prevents_paired_summary_but_not_complete_point_mean() -> None:
    anchor = measurements([1, 3, 5])
    anchor[2].update(status="missing", loss=None)
    result = report.summarize_point(measurements([2, 4, 6]), anchor, [42, 43, 44])
    assert result["mean_loss"] == 4
    assert result["mean_delta"] is result["std_delta"] is None


def test_duplicate_or_wrong_point_seeds_are_rejected() -> None:
    rows = measurements([1, 2, 3])
    rows[-1]["seed"] = 42
    with pytest.raises(ValueError, match="exactly once"):
        report.summarize_point(rows, measurements([1, 2, 3]), [42, 43, 44])


def study_fixture(method: str = "sync", workers: int = 1) -> tuple[report.Record, report.Record]:
    values = {"lr": 0.01, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.99}
    anchor = {
        "id": "center",
        "config": {"optimizer": values},
        "seeds": [{"seed": seed, "trial_id": f"center-{seed}"} for seed in [42, 43, 44]],
    }
    trials, slices = {}, []
    for index, field in enumerate(report.FIELDS):
        key = field.split(".")[1]
        points = []
        for candidate, value in [("center", values[key]), (f"neighbor{index}", values[key] / 2)]:
            entries = []
            for seed in [42, 43, 44]:
                identity = f"{candidate}-{seed}"
                entries.append({"seed": seed, "trial_id": identity})
                loss = 1 + (seed - 43) * 0.1 + (0 if candidate == "center" else 0.2)
                result = {"validation": {"loss": loss, "validation_complete": True}, "session_seconds": 10}
                trials[identity] = {
                    "id": identity,
                    "candidate_id": candidate,
                    "seed": seed,
                    "status": "succeeded",
                    "loss": loss,
                    "result": result,
                    "attempts": [{"directory": f"/fake/{identity}", "status": "succeeded"}],
                }
            points.append({"value": value, "expression": str(value), "candidate_id": candidate, "seeds": entries})
        slices.append({"field": field, "points": points})
    study = {
        "name": f"{method}-{workers}w",
        "method": method,
        "workers": workers,
        "directory": "study",
        "source_directory": "original",
        "anchor": anchor,
        "slices": slices,
        "required_trial_ids": list(trials),
        "reused_trial_ids": [f"center-{seed}" for seed in [42, 43, 44]],
        "new_trial_ids": [key for key in trials if not key.startswith("center")],
        "imported_trial_ids": [f"center-{seed}" for seed in [42, 43, 44]],
        "imported_allocation_ids": ["imported"],
        "counts": {"required": 15, "reused": 3, "new": 12, "allocations": 6},
    }
    return study, {"status": "complete", "trials": trials, "allocations": {}, "provenance": {}}


@pytest.fixture
def fake_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    from decentralized_llm.tuning import controller

    monkeypatch.setattr(controller, "check_result", lambda trial, attempt: trial["result"])
    monkeypatch.setattr(report, "authenticate_identity", lambda trial, evidence: None)


def test_slices_keep_original_center_and_csv_deduplicates_it(tmp_path: Path, fake_authentication: None) -> None:
    study, state = study_fixture()
    state["trials"]["neighbor0-42"]["loss"] = 0.1
    state["trials"]["neighbor0-42"]["result"]["validation"]["loss"] = 0.1
    summary = report.summarize_study(study, state, [42, 43, 44])
    assert summary["anchor"]["id"] == "center"
    assert summary["counts"]["completed"] == 15
    assert summary["slices"][0]["better_points"] == [0.005]
    assert all(axis["boundary"] == "upper" for axis in summary["slices"])
    assert all(axis["neighbors"][0]["direction"] == "lower" for axis in summary["slices"])
    report.export_tables(tmp_path, [summary])
    seed_rows = list(csv.DictReader((tmp_path / "per-seed.csv").open()))
    assert len(seed_rows) == len({row["trial_id"] for row in seed_rows}) == 15
    center = next(row for row in seed_rows if row["trial_id"] == "center-42")
    assert center["slices"].split(";") == list(report.FIELDS)
    assert len(list(csv.DictReader((tmp_path / "slices.csv").open()))) == 8


def test_result_authentication_failure_is_visible_without_statistics(
    monkeypatch: pytest.MonkeyPatch,
    fake_authentication: None,
) -> None:
    from decentralized_llm.tuning import controller

    study, state = study_fixture()

    def check(trial: report.Record, attempt: report.Record) -> report.Record:
        if trial["id"] == "neighbor0-42":
            raise ValueError("source provenance mismatch")
        return trial["result"]

    monkeypatch.setattr(controller, "check_result", check)
    summary = report.summarize_study(study, state, [42, 43, 44])
    point = summary["slices"][0]["points"][0]
    assert point["status"] == "invalid"
    assert point["mean_loss"] is None
    assert point["seeds"][0]["reason"] == "source provenance mismatch"
    assert summary["counts"]["completed"] == 14
    assert summary["counts"]["missing"] == 1


def test_comparison_uses_shared_order_and_preserves_colors(fake_authentication: None) -> None:
    studies = [
        report.summarize_study(*study_fixture(method, 1 if method == "sync" else 4), [42, 43, 44])
        for method in reversed(METHOD_ORDER)
    ]
    figure = report.comparison_figure(studies, 4)
    for axis in figure.axes:
        handles, labels = axis.get_legend_handles_labels()
        assert labels == [METHOD_LABELS[method] for method in METHOD_ORDER]
        assert [handle.lines[0].get_color() for handle in handles] == [METHOD_COLORS[method] for method in METHOD_ORDER]
    figure.clear()


def test_comparison_aligns_different_grids_by_parameter_value(fake_authentication: None) -> None:
    sync = report.summarize_study(*study_fixture("sync", 1), [42, 43, 44])
    study, state = study_fixture("cmp", 4)
    neighbor = next(point for point in study["slices"][0]["points"] if point["candidate_id"] != "center")
    neighbor.update(value=0.02, expression="0.02")
    complete = report.summarize_study(study, state, [42, 43, 44])
    figure = report.comparison_figure([complete, sync], 4)
    for axis in (figure.axes[0], figure.axes[4]):
        handles, labels = axis.get_legend_handles_labels()
        assert labels == [METHOD_LABELS["sync"], METHOD_LABELS["cmp"]]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["0.005", "0.01", "0.02"]
        assert list(handles[0].lines[0].get_xdata()) == [0, 1]
        assert list(handles[1].lines[0].get_xdata()) == [1, 2]
    figure.clear()


@pytest.mark.parametrize("grid_spec", [None, "/reference/original-grid.yaml"])
def test_build_report_counts_new_allocations_only_and_allows_missing_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_authentication: None,
    grid_spec: str | None,
) -> None:
    from scripts import tune_50m

    study, state = study_fixture()
    study["grid_spec"] = {"path": grid_spec, "sha256": "reference-digest"} if grid_spec else None
    state["allocations"] = {"imported": {"job_id": "100"}, "new": {"job_id": "200"}}
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "state.json").write_text(json.dumps(state))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "seeds": [42, 43, 44],
                "studies": [study],
                "counts": study["counts"],
                "provenance": {},
                "grid_spec": grid_spec,
            }
        )
    )
    seen = []

    def accounting(jobs: list[str]) -> report.Record:
        seen.extend(jobs)
        return {"total_jobs": len(jobs), "gpu_hours": None, "complete": False}

    monkeypatch.setattr(tune_50m, "accounting", accounting)
    monkeypatch.setattr(
        report, "write_plots", lambda output, studies: [{"name": study["name"], "png": "x.png", "svg": "x.svg"}]
    )
    result = report.build_report(tmp_path)
    assert seen == ["200"]
    assert result["status"] == "complete"
    assert result["counts"] == {"required": 15, "reused": 3, "new": 12, "allocations": 6, "completed": 15, "missing": 0}
    assert not result["accounting"]["complete"]
    assert (tmp_path / "analysis" / "report.md").is_file()
    assert (tmp_path / "analysis" / "report.json").is_file()
    assert (tmp_path / "analysis" / "index.html").is_file()
    assert result["grid_spec"] == grid_spec
    assert result["studies"][0]["grid_spec"] == study["grid_spec"]
    if grid_spec:
        for filename in ("report.md", "index.html"):
            assert grid_spec in (tmp_path / "analysis" / filename).read_text()


def test_trial_scientific_identity_rejects_tampered_configuration_or_provenance() -> None:
    from decentralized_llm.config import Config
    from decentralized_llm.tuning.spec import fingerprint, numeric_recipe, trial_id

    config = Config()
    evidence = {"source_sha256": "original"}
    candidate = {"id": fingerprint(numeric_recipe(config))}
    trial = {
        "id": trial_id(candidate, 42, evidence),
        "candidate_id": candidate["id"],
        "seed": 42,
        "provenance": evidence,
        "config": config.model_dump(mode="json"),
    }
    report.authenticate_identity(trial, evidence)
    with pytest.raises(ValueError, match="provenance"):
        report.authenticate_identity(trial, {"source_sha256": "different"})
    trial["config"]["optimizer"]["lr"] *= 2
    with pytest.raises(ValueError, match="scientific identity"):
        report.authenticate_identity(trial, evidence)
