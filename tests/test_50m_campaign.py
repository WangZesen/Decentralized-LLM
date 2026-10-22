import copy
import json
from pathlib import Path

import pytest
import yaml

from decentralized_llm.config import Config, load_config
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.estimate import estimate
from decentralized_llm.tuning.spec import Record, TuningSpec, expand, read_spec
from scripts import tune_50m

SPECS = sorted(Path("configs/tunings").glob("50m-*-1024c-64bs-20tpp.yaml"))


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_campaign_count_and_physical_resources(path: Path) -> None:
    spec, base = read_spec(path)
    assert len(SPECS) == 8
    assert base.physical_gpu_count == 1 and base.runtime.training_backend == "packed"
    assert base.distributed.world_size == (1 if "-sync-" in path.stem else 8)
    assert base.distributed.error_feedback_gamma == (1.0 if "-ef-" in path.stem else 0.0)
    assert base.model.parameter_count == 48507392 and base.model.context_length == 1024
    assert base.training.batch_tokens == 65536 and base.training.tokens_per_parameters == 20
    assert base.launcher.kind == "slurm" and base.launcher.time == "00:45:00"
    assert spec.controller.max_jobs == 30 and spec.controller.max_retries == 1
    assert spec.controller.jobs_per_allocation == 1
    assert estimate(spec, base, {})["new_jobs"] == 27
    first = expand(base, spec.phases[0])
    # Every possible retained LR pair must reuse exactly two first-phase trials.
    for i, candidate in enumerate(first):
        for other in first[i + 1 :]:
            second = expand(base, spec.phases[1], [candidate, other])
            assert len(second) == 18
            assert len({row["id"] for row in first} & {row["id"] for row in second}) == 2


def completed_state(candidate: Record, spec: TuningSpec, base: Config, loss: float) -> Record:
    winner = copy.deepcopy(candidate)
    winner.update(mean_loss=loss, std_loss=0.01, seeds=[])
    trials = {}
    for seed in [42, 43, 44]:
        key = f"{candidate['id']}-{seed}"
        winner["seeds"].append({"trial_id": key, "seed": seed, "loss": loss})
        trials[key] = {
            "status": "succeeded",
            "attempts": [{"directory": f"/test/{key}"}],
            "result": {
                "session_seconds": 1200,
                "validation": {"split": "full", "consensus_error": 0.1},
            },
        }
    return {
        "version": 3,
        "status": "complete",
        "spec": spec.model_dump(mode="json"),
        "base": base.model_dump(mode="json"),
        "provenance": {"test": "frozen"},
        "winner": winner,
        "phases": [{"rankings": [winner]}],
        "trials": trials,
        "allocations": {},
    }


@pytest.fixture
def core() -> Record:
    spec, base = read_spec(Path("configs/tunings/50m-sync-1w-1024c-64bs-20tpp.yaml"))
    first = expand(base, spec.phases[0])
    second = expand(base, spec.phases[1], first[:1])
    return completed_state(second[0], spec, base, 3.5)


def test_boundary_probes_are_axial_and_three_seed(core: Record, tmp_path: Path) -> None:
    probes = tune_50m.boundary_probes(core)
    assert len(probes) == 3
    values = {row["field"]: row["value"] for row in probes}
    assert values["optimizer.lr"] == 2**-10
    assert values["optimizer.beta1"] == pytest.approx(0.81**2)
    assert values["optimizer.beta2"] == pytest.approx(0.95**4)
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump(core["winner"]["config"]))
    for probe in probes:
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(tune_50m.boundary_spec(core, config, probe)))
        spec, base = read_spec(path)
        assert estimate(spec, base, {})["new_jobs"] == 3
        candidate = expand(base, spec.phases[0])[0]
        changed = [
            field
            for field in tune_50m.FIELDS
            if candidate["config"]["optimizer"][field.split(".")[1]]
            != core["winner"]["config"]["optimizer"][field.split(".")[1]]
        ]
        assert changed == [probe["field"]]


def test_upper_edges_and_interior(core: Record) -> None:
    for field, values in tune_50m.grids(core).items():
        core["winner"]["config"]["optimizer"][field.split(".")[1]] = values[-1]
    probes = tune_50m.boundary_probes(core)
    assert len(probes) == 3 and all(probe["side"] == "upper" for probe in probes)
    assert all(probe["value"] > max(tune_50m.grids(core)[probe["field"]]) for probe in probes)
    for field, values in tune_50m.grids(core).items():
        core["winner"]["config"]["optimizer"][field.split(".")[1]] = values[1]
    assert tune_50m.boundary_probes(core) == []
    core["winner"] = None
    assert tune_50m.boundary_probes(core) == []


def test_boundary_resume_does_not_expand_again(core: Record, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core_path = tmp_path / "state.json"
    atomic_json(core_path, core)
    launches = []

    def complete(directory: Path, spec_path: Path) -> Record:
        path = directory / "state.json"
        if path.exists():
            return json.loads(path.read_text())
        spec, base = read_spec(spec_path)
        # Every outward probe improves: a recursive implementation would never stop.
        state = completed_state(expand(base, spec.phases[0])[0], spec, base, 3.0)
        atomic_json(path, state)
        launches.append(directory)
        return state

    monkeypatch.setattr(tune_50m, "ensure_complete", complete)
    tune_50m.run_study(tmp_path, Path("unused"))
    tune_50m.run_study(tmp_path, Path("unused"))
    assert len(launches) == 3
    result, states = tune_50m.study_result(tmp_path)
    assert result["status"] == "complete" and result["winner"]["mean_loss"] == 3.0
    assert result["boundary_trials"] == 9 and len(states) == 4
    assert result["remaining_boundary_coordinates"]
    assert load_config(tmp_path / "final-winner.yaml").runtime.seed == 42


def test_incomplete_and_divergent_boundaries(core: Record, tmp_path: Path) -> None:
    atomic_json(tmp_path / "state.json", core)
    row, _ = tune_50m.study_result(tmp_path)
    assert row["status"] == "running"
    assert not (tmp_path / "final-winner.yaml").exists()
    for probe in tune_50m.boundary_probes(core):
        location = tmp_path / "boundary" / probe["field"].split(".")[1]
        location.mkdir(parents=True)
        boundary = copy.deepcopy(core)
        boundary["winner"] = None
        atomic_json(location / "state.json", boundary)
    row, _ = tune_50m.study_result(tmp_path)
    assert row["status"] == "complete" and row["winner"]["mean_loss"] == 3.5
    location = tmp_path / "boundary" / "lr"
    boundary = json.loads((location / "state.json").read_text())
    boundary["provenance"] = {"test": "changed"}
    atomic_json(location / "state.json", boundary)
    with pytest.raises(ValueError, match="Mixed provenance"):
        tune_50m.study_result(tmp_path)


def test_accounting_counts_allocations_and_retries_once() -> None:
    output = (
        "10|COMPLETED|1800|gres/gpu=1,gres/gpu:gh200=1|\n"
        "10.batch|COMPLETED|1800|gres/gpu=1|\n"
        "11|FAILED|900|gres/gpu:gh200=1|\n"
        "12|PENDING|0||\n"
    )
    usage = tune_50m.parse_accounting(output, ["10", "10", "11", "12", "13"])
    assert usage["gpu_hours"] == 0.75
    assert usage["accounted_jobs"] == 3 and usage["total_jobs"] == 4
    assert usage["complete"] is False
