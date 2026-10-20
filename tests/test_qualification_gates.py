import itertools
import json
import math
from pathlib import Path

import pytest

from scripts.qualification_gates import assess_numerical


def records() -> list[dict[str, float]]:
    return [
        dict.fromkeys(
            ("loss_tolerance_ratio", "parameters", "gradients", "updates", "exp_avg", "exp_avg_sq", "history", "steps"),
            0.0,
        )
        for _ in range(100)
    ]


def assess(errors: list[dict[str, float]], delta: float = 0) -> dict:
    return assess_numerical(errors, delta, steps=100, optimizer_only=True, feedback=True, fused_candidate=True)


@pytest.mark.parametrize(
    ("field", "failure"),
    [
        ("loss_tolerance_ratio", 1.001),
        ("parameters", 0.02),
        ("gradients", 0.05),
        ("updates", 0.10),
        ("exp_avg", 0.05),
        ("exp_avg_sq", 0.05),
        ("history", 0.10),
    ],
)
def test_state_probe_preserves_original_window_and_retains_late_failures(field: str, failure: float) -> None:
    errors = records()
    errors[4][field] = failure
    assert not assess(errors)["passed"]
    errors[4][field] = 0
    errors[5][field] = failure
    result = assess(errors)
    assert result["probe_updates"] == 5
    assert result["recorded_updates"] == 100
    assert result["passed"]
    assert not result["all_recorded_updates_passed"]


@pytest.mark.parametrize("delta", [0.020001, -0.020001, math.nan, math.inf])
def test_final_validation_is_required(delta: float) -> None:
    assert not assess(records(), delta)["passed"]


def test_missing_records_history_late_counter_changes_and_nonfinite_values_fail() -> None:
    errors = records()
    assert assess(errors)["passed"]
    assert not assess(errors[:-1])["passed"]
    errors[-1]["steps"] = 1
    assert not assess(errors)["passed"]
    errors[-1]["steps"] = 0
    del errors[-1]["history"]
    assert not assess(errors)["passed"]
    errors[-1]["history"] = math.nan
    assert not assess(errors)["passed"]


def test_optimizer_selection_and_historical_nonoptimizer_records() -> None:
    assert not assess_numerical(records(), 0, steps=100, optimizer_only=True, feedback=True, fused_candidate=False)[
        "passed"
    ]
    legacy = [{k: v for k, v in e.items() if k not in {"history", "steps"}} for e in records()[:5]]
    assert assess_numerical(legacy, 0, steps=100, optimizer_only=False, feedback=True, fused_candidate=False)["passed"]


def test_assessment_preserves_raw_results_and_checks_every_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.assess_optimizer_trajectories import assess_matrix

    rows = []
    original = {}
    for index, (model, scheme, topology, feedback, seed) in enumerate(
        itertools.product(("20m", "90m"), ("awc", "atc"), ("complete", "one_peer_ring"), (False, True), (42, 43))
    ):
        directory = tmp_path / str(index)
        directory.mkdir()
        rows.append(
            dict(
                model=model,
                scheme=scheme,
                topology=topology,
                feedback=feedback,
                seed=seed,
                report=str(directory / "numerical-rank-00000.json"),
            )
        )
        for worker in (0, 1):
            errors = records()
            errors[-1]["gradients"] = 0.2
            report = {
                "rank": worker,
                "world_size": 2,
                "config": {
                    "model": {"width": 320 if model == "20m" else 640},
                    "distributed": {"scheme": scheme, "topology": topology, "error_feedback": feedback},
                    "runtime": {"seed": seed},
                },
                "optimizer_revision": "revision",
                "optimizer_only": True,
                "steps": 100,
                "execution": {},
                "reference_execution": {},
                "optimizers": [{"implementation": "reference"}, {"implementation": "triton"}],
                "errors": errors,
                "validation_delta": 0.001,
                "passed": False,
            }
            path = directory / f"numerical-rank-{worker:05d}.json"
            original[path] = json.dumps(report).encode()
            path.write_bytes(original[path])
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"rows": rows, "complete": True, "optimizer_revision": "revision"}))
    result = assess_matrix(matrix)
    assert result["passed"] and not result["all_recorded_updates_passed"]
    assert all(not report["recorded_passed"] for row in result["rows"] for report in row["reports"])
    assert all(path.read_bytes() == raw for path, raw in original.items())
    path = tmp_path / "0/numerical-rank-00001.json"
    report = json.loads(path.read_text())
    report["errors"][-1]["steps"] = 1
    path.write_text(json.dumps(report))
    assert not assess_matrix(matrix)["passed"]
    report["config"]["runtime"]["seed"] = 999
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incompatible measurement"):
        assess_matrix(matrix)
