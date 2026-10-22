import copy
import json
from pathlib import Path

import pytest

from decentralized_llm.config import Config, save_config
from decentralized_llm.data.cache import training_boundaries
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.controller import add_phase, load_state, save_state
from decentralized_llm.tuning.spec import TuningSpec, expand, fingerprint, numeric_recipe, read_spec, trial_id
from scripts import tune_50m as campaign


def evidence() -> dict:
    return {
        "cache_identity": "cache",
        "source_sha256": "source",
        "lock_sha256": "lock",
        "python_version": [3, 14],
        "torch": "test",
        "numpy": "test",
    }


def publish(base: dict, seed: int, loss: float, root: Path) -> dict:
    config = Config.model_validate(base)
    config.runtime.seed = seed
    candidate = {"id": fingerprint(numeric_recipe(config))}
    key = trial_id(candidate, seed, evidence())
    attempt = root / key / "attempt-001"
    config.runtime.output_dir = attempt / "run"
    config.runtime.output_dir.mkdir(parents=True)
    save_config(config, config.runtime.output_dir / "config.yaml")
    atomic_json(config.runtime.output_dir / "run.json", {"cache": "cache", "environment": evidence()})
    result = {
        "cursor": training_boundaries(config, config.model.parameter_count)[-1],
        "session_seconds": 1200,
        "validation": {"loss": loss, "split": "full", "validation_complete": True, "consensus_error": 0.0},
    }
    atomic_json(config.runtime.output_dir / "result.json", result)
    return {
        "id": key,
        "candidate_id": candidate["id"],
        "seed": seed,
        "directory": str(attempt.parent),
        "config": config.model_dump(mode="json"),
        "provenance": evidence(),
        "loss": loss,
        "status": "succeeded",
        "result": result,
        "attempts": [{"number": 1, "directory": str(attempt), "status": "succeeded"}],
    }


@pytest.fixture
def history(tmp_path: Path) -> dict:
    spec, base = read_spec(Path("configs/tunings/50m-sync-1w-1024c-64bs-20tpp.yaml"))
    base.optimizer.lr = 2**-8
    base.optimizer.beta1 = 0.974003746425297
    base.optimizer.beta2 = 0.996799302289812
    raw = base.model_dump(mode="json")
    trials = [publish(raw, seed, 3.2 + (seed - 42) / 100, tmp_path / "artifacts") for seed in (42, 43, 44)]
    state = campaign.seeded_state(spec.model_dump(mode="json"), raw, evidence(), {t["id"]: t for t in trials}, {})
    state["status"] = "complete"
    state["winner"] = campaign.confirmed_candidates(state["trials"], evidence())[0]
    directory = tmp_path / "history"
    directory.mkdir()
    save_state(directory, state)
    return state


def test_import_authenticates_and_preserves_artifact_identity(history: dict, tmp_path: Path) -> None:
    path = tmp_path / "history" / "state.json"
    before = path.read_bytes()
    trials, allocations, imports = campaign.import_history([path], history)
    assert trials == history["trials"] and allocations == {}
    assert path.read_bytes() == before
    assert imports[0]["sha256"] == campaign.sha256(path)
    trial = next(iter(trials.values()))
    result = Path(trial["attempts"][0]["directory"]) / "run" / "result.json"
    changed = json.loads(result.read_text())
    changed["validation"]["loss"] += 1
    atomic_json(result, changed)
    with pytest.raises(ValueError, match="History result changed"):
        campaign.import_history([path], history)


@pytest.mark.parametrize("change", ["provenance", "gamma", "identity", "incomplete"])
def test_import_rejects_incompatible_history(history: dict, tmp_path: Path, change: str) -> None:
    state = copy.deepcopy(history)
    if change == "provenance":
        state["provenance"]["source_sha256"] = "different"
    elif change == "gamma":
        history["base"]["distributed"]["error_feedback_gamma"] = 1
    elif change == "identity":
        next(iter(state["trials"].values()))["candidate_id"] = "wrong"
    else:
        state["status"] = "running"
    save_state(tmp_path / "history", state)
    with pytest.raises(ValueError):
        campaign.import_history([tmp_path / "history" / "state.json"], history)


def test_snap_does_not_approximate_trial_identity(history: dict, tmp_path: Path) -> None:
    nominal = 0.95 ** (1 / 16)
    center = campaign.snap_beta(nominal, "beta2", history["trials"])
    assert center == 0.996799302289812
    config = Config.model_validate(history["base"])
    identity = fingerprint(numeric_recipe(config))
    assert trial_id({"id": identity}, 42, evidence()) in history["trials"]
    config.optimizer.beta2 += 1e-15
    assert trial_id({"id": fingerprint(numeric_recipe(config))}, 42, evidence()) not in history["trials"]
    spec = campaign.extension_spec(tmp_path / "base.yaml", history, history["trials"])
    candidates = expand(config, TuningSpec.model_validate(spec).phases[0])
    assert len(candidates) == 27
    assert identity in {c["id"] for c in candidates}


def test_joint_grid_includes_both_beta_interactions_and_reuses_seed42(history: dict, tmp_path: Path) -> None:
    spec = TuningSpec.model_validate(campaign.extension_spec(tmp_path / "base.yaml", history, history["trials"]))
    base = Config.model_validate(history["base"])
    candidates = expand(base, spec.phases[0])
    for lr in [2**-9, 2**-8, 2**-7]:
        assert len([c for c in candidates if c["config"]["optimizer"]["lr"] == lr]) == 9
    state = campaign.seeded_state(spec.model_dump(mode="json"), history["base"], evidence(), history["trials"], {})
    first = add_phase(state, spec, base)
    assert first["new_trials"] == 26 and first["reused_trials"] == 1
    # Use three existing seed-42 candidates as confirmation finalists.
    first["rankings"] = candidates[:3]
    first["selected"] = [c["id"] for c in candidates[:3]]
    second = add_phase(state, spec, base)
    assert second["reused_trials"] >= 3 and second["new_trials"] <= 6


def test_confirmation_requires_all_seeds_and_retains_incumbent_on_ties(history: dict, tmp_path: Path) -> None:
    incumbent = history["winner"]
    raw = copy.deepcopy(history["base"])
    raw["optimizer"]["lr"] *= 2
    trials = copy.deepcopy(history["trials"])
    for seed in (42, 43):
        t = publish(raw, seed, 3.0, tmp_path / "other")
        trials[t["id"]] = t
    assert campaign.preferred_winner(trials, evidence(), incumbent)["id"] == incumbent["id"]
    t = publish(raw, 44, 3.0, tmp_path / "other")
    trials[t["id"]] = t
    assert campaign.preferred_winner(trials, evidence(), incumbent)["mean_loss"] == 3.0
    trials[t["id"]].update(status="diverged", loss="inf")
    assert campaign.preferred_winner(trials, evidence(), incumbent)["id"] == incumbent["id"]


def test_boundary_budget_reserves_complete_seed_sets_and_is_axial(history: dict, tmp_path: Path) -> None:
    spec = campaign.extension_spec(tmp_path / "base.yaml", history, history["trials"])
    core = {**history, "spec": spec}
    # A three-coordinate upper corner, already evaluated on all three seeds.
    candidate = expand(Config.model_validate(history["base"]), TuningSpec.model_validate(spec).phases[0])[-1]
    corner = [publish(candidate["config"], s, 3.0, tmp_path / "corner") for s in (42, 43, 44)]
    trials = {**history["trials"], **{t["id"]: t for t in corner}}
    budget = {"imported_trial_ids": list(trials), "limit": 4}
    plan = campaign.plan_extension_boundaries(core, trials, history["winner"], budget)
    assert [p["admitted"] for p in plan["probes"]] == [True, False, False]
    assert len(plan["reserved_new_trial_ids"]) == 3
    for p in plan["probes"]:
        changed = [
            k
            for k in campaign.FIELDS
            if p["candidate"]["config"]["optimizer"][k.split(".")[1]]
            != candidate["config"]["optimizer"][k.split(".")[1]]
        ]
        assert changed == [p["field"]]
    # A completed seed makes a second whole probe affordable; no partial launches.
    beta_probe = plan["probes"][1]
    t = publish(beta_probe["candidate"]["config"], 42, 3.1, tmp_path / "known")
    trials[t["id"]] = t
    budget.update(limit=5, imported_trial_ids=list(trials))
    plan = campaign.plan_extension_boundaries(core, trials, history["winner"], budget)
    assert [p["admitted"] for p in plan["probes"]] == [True, True, False]
    assert len(plan["reserved_new_trial_ids"]) == 5


def test_extension_restart_does_not_repeat_or_expand_boundaries(
    history: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "extension"
    directory.mkdir()
    spec = campaign.extension_spec(directory / "base.yaml", history, history["trials"])
    core = {**copy.deepcopy(history), "spec": spec}
    candidate = expand(Config.model_validate(core["base"]), TuningSpec.model_validate(spec).phases[0])[-1]
    for seed in (42, 43, 44):
        trial = publish(candidate["config"], seed, 3.0, tmp_path / "corner")
        core["trials"][trial["id"]] = trial
    save_state(directory, core)
    (directory / "spec.yaml").write_text("frozen")
    atomic_json(
        directory / "extension.json",
        {"incumbent": history["winner"], "spec_sha256": campaign.sha256(directory / "spec.yaml")},
    )
    atomic_json(directory / "budget.json", {"imported_trial_ids": list(history["trials"]), "limit": 9})
    launches = []

    def finish(path: Path, _: Path) -> dict:
        state = load_state(path)
        if state["status"] == "complete":
            return state
        s = TuningSpec.model_validate(state["spec"])
        c = expand(Config.model_validate(state["base"]), s.phases[0])[0]
        for seed in (42, 43, 44):
            trial = publish(c["config"], seed, 2.9, path / "artifacts")
            state["trials"][trial["id"]] = trial
        state["status"] = "complete"
        save_state(path, state)
        launches.append(path)
        return state

    monkeypatch.setattr(campaign, "ensure_complete", finish)
    campaign.run_extension_study(directory)
    campaign.run_extension_study(directory)
    assert len(launches) == 2  # three core + two three-seed probes exhaust the nine-run test budget
    budget = json.loads((directory / "budget.json").read_text())
    assert len(budget["actual_new_trial_ids"]) == 9
    assert (directory / "campaign-complete.json").exists()


def test_new_budget_rejects_grid_without_enough_history(history: dict, tmp_path: Path) -> None:
    state = copy.deepcopy(history)
    state["spec"] = campaign.extension_spec(tmp_path / "base.yaml", state, state["trials"])
    with pytest.raises(ValueError, match="approved budget"):
        campaign.joint_budget(state)


def test_report_excludes_imported_jobs_and_counts_boundary_retries_once(
    history: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "report"
    directory = root / "sync"
    directory.mkdir(parents=True)
    state = copy.deepcopy(history)
    state["spec"] = campaign.extension_spec(directory / "base.yaml", state, state["trials"])
    raw = copy.deepcopy(state["base"])
    raw["optimizer"]["lr"] *= 2
    new = []
    for seed in (42, 43, 44):
        trial = publish(raw, seed, 3.0, directory / "artifacts")
        trial["attempts"][0]["job_id"] = str(seed)
        if seed == 42:
            trial["attempts"].append({**trial["attempts"][0], "job_id": "45", "number": 2})
        state["trials"][trial["id"]] = trial
        new.append(trial["id"])
    state["allocations"] = {key: {"job_id": key, "status": "finished"} for key in ["old", "42", "43", "44", "45"]}
    save_state(directory, state)
    atomic_json(directory / "extension.json", {"incumbent": history["winner"], "imported_job_ids": ["old"]})
    atomic_json(
        directory / "budget.json",
        {"imported_trial_ids": list(history["trials"]), "limit": 30, "core_new_upper_bound": 26},
    )
    atomic_json(
        directory / "boundary-plan.json",
        {"probes": [{"field": "optimizer.lr", "value": 2**-6, "admitted": True, "new_trial_ids": new}]},
    )
    atomic_json(directory / "campaign-complete.json", {})
    monkeypatch.setattr(campaign, "plot_winner", lambda *_: None)
    received = []

    def accounting(jobs: list[str]) -> dict:
        received.extend(jobs)
        return {"complete": True, "gpu_hours": len(jobs), "jobs": [{"job_id": key, "gpu_hours": 1.0} for key in jobs]}

    monkeypatch.setattr(campaign, "accounting", accounting)
    report = campaign.write_extension_report(root, {"studies": [{"directory": str(directory)}]})
    assert set(received) == {"42", "43", "44", "45"} and len(received) == 4
    row = report["studies"][0]
    assert row["unique_trials"] == row["boundary_trials"] == 3
    assert row["allocated_gpu_hours"] == row["boundary_gpu_hours"] == 4
    assert row["retry_gpu_hours"] == row["retry_attempts"] == 1
    assert row["mean_loss_change"] == pytest.approx(3.0 - history["winner"]["mean_loss"])
    assert row["status"] == "complete" and (directory / "final-winner.yaml").exists()


@pytest.mark.parametrize("path", sorted(Path("configs/tunings").glob("50m-*-1024c-64bs-20tpp.yaml")))
def test_saved_recipes_pass_extension_validation(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, base = read_spec(path)
    directory = tmp_path / path.stem
    directory.mkdir()
    core = {"base": base.model_dump(mode="json"), "spec": spec.model_dump(mode="json")}
    joint = campaign.extension_spec(directory / "base.yaml", core, {})
    state = campaign.seeded_state(joint, core["base"], evidence(), {}, {})
    save_state(directory, state)
    atomic_json(directory / "extension.json", {"imports": []})
    atomic_json(
        directory / "budget.json",
        {"limit": 30, "imported_trial_ids": [], "core_new_upper_bound": 26, "screening_reused": 7},
    )
    monkeypatch.setattr(campaign, "execute", lambda *args, **kwargs: {"new_jobs": 0, "phases": []})
    result = campaign.validate_extension(tmp_path, {"studies": [{"name": path.stem, "directory": str(directory)}]})
    assert result["passed"] and result["studies"][0]["training_tokens"] == 970129408
