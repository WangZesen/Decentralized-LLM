import copy
import json
from pathlib import Path

import pytest
import yaml
from test_50m_extension import evidence, publish

from decentralized_llm.config import Config
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.controller import add_phase, load_state, save_state
from decentralized_llm.tuning.estimate import ranking
from decentralized_llm.tuning.spec import TuningSpec, expand, fingerprint, numeric_recipe, trial_id
from scripts import tune_stability as campaign
from scripts.tune_50m import seeded_state


@pytest.fixture
def history(tmp_path: Path, tiny_config: Config) -> tuple[Path, dict]:
    base = tiny_config.model_copy(deep=True)
    base.optimizer.lr = 0.001
    base.optimizer.weight_decay = 0.1
    base.optimizer.beta1 = 0.9
    base.optimizer.beta2 = 0.99
    base.distributed.mode = "ddp"
    base.distributed.nodes = 1
    base.distributed.gpus_per_node = 1
    raw = base.model_dump(mode="json")
    spec = TuningSpec.model_validate(
        {
            "screening_seeds": 1,
            "confirmation": {"seeds": 3},
            "phases": [
                {
                    "name": "joint",
                    "search": {
                        "optimizer.lr": [0.001, 0.002],
                        "optimizer.weight_decay": [0.01, 0.1],
                        "optimizer.beta1": [0.8, 0.9],
                        "optimizer.beta2": [0.95, 0.99],
                    },
                    "top_k": 1,
                }
            ],
            "controller": {"jobs_per_allocation": 2},
        }
    )
    trials = [publish(raw, seed, 3 + seed / 10000, tmp_path / "artifacts") for seed in campaign.SEEDS]
    alternate = copy.deepcopy(raw)
    alternate["optimizer"]["lr"] = 0.002
    trials.append(publish(alternate, 42, 3.1, tmp_path / "artifacts"))
    alternate = copy.deepcopy(raw)
    alternate["optimizer"]["beta1"] = 0.8
    trials.extend(publish(alternate, seed, 3.2, tmp_path / "artifacts") for seed in campaign.SEEDS)
    # Retain off-slice artifacts, including any allocation membership they own.
    alternate["optimizer"]["lr"] = 0.002
    trials.append(publish(alternate, 42, 3.3, tmp_path / "artifacts"))
    state = seeded_state(spec.model_dump(mode="json"), raw, evidence(), {t["id"]: t for t in trials}, {})
    center = {
        "id": fingerprint(numeric_recipe(base)),
        "config": raw,
        "values": {field: raw["optimizer"][field.split(".")[1]] for field in campaign.FIELDS},
        "labels": {},
    }
    ranks = ranking([center], list(campaign.SEEDS), state["trials"], evidence())
    assert ranks is not None
    state.update(status="complete", winner=ranks[0])
    source = tmp_path / "history" / "study"
    source.mkdir(parents=True)
    save_state(source, state)
    return source, state


@pytest.fixture
def broad_grid_spec(tmp_path: Path) -> Path:
    path = tmp_path / "reference.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base": {
                    "overrides": {
                        "training.tokens_per_parameters": 20.0,
                        "training.batch_tokens": 65536,
                        "optimizer.lr": 0.004,
                        "distributed.mode": "decentralized",
                    }
                },
                "screening_seeds": 1,
                "confirmation": {"seeds": 1},
                "phases": [
                    {
                        "name": "lr-and-decay",
                        "initial_values": {"optimizer.beta1": 0.7, "optimizer.beta2": 0.9},
                        "search": {
                            "optimizer.lr": [0.0005, 0.001, 0.002, 0.004],
                            "optimizer.weight_decay": [0.005, 0.01, 0.1, 0.2],
                        },
                        "top_k": 3,
                    },
                    {
                        "name": "betas",
                        "search": {
                            "optimizer.beta1": [0.7, 0.8, 0.9, 0.95],
                            "optimizer.beta2": [0.9, 0.95, 0.99, 0.999],
                        },
                        "top_k": 6,
                    },
                ],
                "controller": {"jobs_per_allocation": 1},
            },
            sort_keys=False,
        )
    )
    return path


def test_fixed_slices_ignore_the_previous_selected_parent(history: tuple[Path, dict], tmp_path: Path) -> None:
    source, original = history
    row, state = campaign.plan_study(source, tmp_path / "slices")
    spec = TuningSpec.model_validate(row["spec"])
    base = Config.model_validate(row["base"])
    parent = None
    center = original["winner"]["id"]
    union = set()
    for field, phase in zip(campaign.FIELDS, spec.phases, strict=True):
        candidates = expand(base, phase, parent)
        assert len(candidates) == 2
        assert sum(candidate["id"] == center for candidate in candidates) == 1
        assert list(phase.search)[0] == field
        for candidate in candidates:
            union.add(candidate["id"])
            for fixed in set(campaign.FIELDS) - {field}:
                assert candidate["values"][fixed] == original["winner"]["values"][fixed]
        # Deliberately select the off-center point; the next phase must reset it.
        parent = [next(candidate for candidate in candidates if candidate["id"] != center)]
    assert len(union) == 5
    assert state["phases"] == [] and state["status"] == "prepared"
    assert spec.seeds() == spec.seeds(True) == list(campaign.SEEDS)


def test_reference_grid_expands_slices_without_changing_the_source_winner(
    history: tuple[Path, dict], broad_grid_spec: Path, tmp_path: Path
) -> None:
    source, original = history
    before = (source / "state.json").read_bytes()
    grids = campaign.winner_grids(original, grid_spec=broad_grid_spec)
    row, state = campaign.plan_study(source, tmp_path / "slices", grid_spec=broad_grid_spec)
    spec = TuningSpec.model_validate(row["spec"])
    base = Config.model_validate(row["base"])
    assert row["anchor"] == original["winner"]
    assert row["base"] == original["winner"]["config"]
    assert row["counts"] == {"required": 39, "reused": 7, "new": 32, "allocations": 17}
    assert set(row["imported_trial_ids"]) == set(original["trials"]) == set(state["trials"])
    assert set(row["reused_trial_ids"]) == set(row["required_trial_ids"]) & set(original["trials"])
    assert len(set(row["required_trial_ids"])) == 39
    assert spec.controller.jobs_per_allocation == 2
    assert spec.seeds() == spec.seeds(True) == list(campaign.SEEDS)
    parent = None
    union = set()
    for field, phase in zip(campaign.FIELDS, spec.phases, strict=True):
        candidates = expand(base, phase, parent)
        assert len(candidates) == len(grids[field]) == 4
        assert {candidate["values"][field] for candidate in candidates} == {point["value"] for point in grids[field]}
        for candidate in candidates:
            union.add(candidate["id"])
            restored = Config.model_validate(candidate["config"])
            setattr(restored.optimizer, field.split(".")[1], original["winner"]["values"][field])
            assert numeric_recipe(restored) == numeric_recipe(base)
        parent = [next(candidate for candidate in candidates if candidate["id"] != original["winner"]["id"])]
    assert len(union) == 13
    assert (source / "state.json").read_bytes() == before


@pytest.mark.parametrize("change", ["missing-field", "winner-outside-grid"])
def test_reference_grid_requires_all_fields_and_the_source_winner(
    history: tuple[Path, dict], broad_grid_spec: Path, tmp_path: Path, change: str
) -> None:
    raw = yaml.safe_load(broad_grid_spec.read_text())
    if change == "missing-field":
        del raw["phases"][1]["search"]["optimizer.beta2"]
        del raw["phases"][0]["initial_values"]["optimizer.beta2"]
        message = "Expected the four AdamW tuning fields"
    else:
        raw["phases"][0]["search"]["optimizer.lr"] = [0.002, 0.004]
        message = "Original winner is absent"
    broad_grid_spec.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        campaign.plan_study(history[0], tmp_path / "slices", grid_spec=broad_grid_spec)


def test_campaign_forwards_the_reference_grid(
    history: tuple[Path, dict], broad_grid_spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, original = history
    environment = source.parent / "environment.sh"
    monkeypatch.setattr(campaign, "source_environment", lambda _: environment)
    monkeypatch.setattr(campaign, "training_environment", lambda *_: {"environment_script": str(environment)})
    manifest, states = campaign.plan_campaign(tmp_path / "slices", source.parent, grid_spec=broad_grid_spec)
    assert manifest["grid_spec"] == str(broad_grid_spec.resolve())
    assert manifest["counts"] == {"required": 39, "reused": 7, "new": 32, "allocations": 17}
    assert states[source.name]["provenance"] == original["provenance"]
    assert manifest["studies"][0]["base"] == original["winner"]["config"]


def test_reuse_one_or_three_seeds_and_deduplicate_center(history: tuple[Path, dict], tmp_path: Path) -> None:
    source, original = history
    before = (source / "state.json").read_bytes()
    row, state = campaign.plan_study(source, tmp_path / "slices")
    assert row["counts"] == {"required": 15, "reused": 7, "new": 8, "allocations": 5}
    assert len(row["imported_trial_ids"]) == len(original["trials"]) == 8
    assert len(set(row["required_trial_ids"])) == 15
    assert set(row["new_trial_ids"]).isdisjoint(row["reused_trial_ids"])
    assert set(row["required_trial_ids"]) == set(row["new_trial_ids"]) | set(row["reused_trial_ids"])
    assert (source / "state.json").read_bytes() == before
    assert all(Path(trial["directory"]).is_absolute() for trial in state["trials"].values())
    spec = TuningSpec.model_validate(row["spec"])
    first = add_phase(state, spec, Config.model_validate(row["base"]))
    assert first["reused_trials"] == 4 and first["new_trials"] == 2
    first["rankings"] = first["candidates"]
    first["selected"] = [first["candidates"][0]["id"]]
    second = add_phase(state, spec, Config.model_validate(row["base"]))
    assert second["reused_trials"] == 3 and second["new_trials"] == 3


@pytest.mark.parametrize("change", ["loss", "provenance", "identity", "partial", "active"])
def test_reuse_rejects_untrusted_results(history: tuple[Path, dict], tmp_path: Path, change: str) -> None:
    source, state = history
    trial = next(iter(state["trials"].values()))
    if change == "loss":
        path = Path(trial["attempts"][0]["directory"]) / "run/result.json"
        result = json.loads(path.read_text())
        result["validation"]["loss"] += 1
        atomic_json(path, result)
    elif change == "provenance":
        trial["provenance"]["source_sha256"] = "different"
    elif change == "identity":
        trial["candidate_id"] = "different"
    elif change == "partial":
        trial["result"]["validation"]["validation_complete"] = False
    else:
        state["allocations"] = {"active": {"status": "submitted"}}
    save_state(source, state)
    with pytest.raises(ValueError):
        campaign.plan_study(source, tmp_path / "slices")


def fixture_plan(history: tuple[Path, dict], root: Path, *, grid_spec: Path | None = None) -> tuple[dict, dict]:
    source, _ = history
    row, state = campaign.plan_study(source, root, grid_spec=grid_spec)
    manifest = {
        "version": 1,
        "root": str(root),
        "source_root": str(source.parent),
        "grid_spec": str(grid_spec.resolve()) if grid_spec is not None else None,
        "seeds": list(campaign.SEEDS),
        "training": {"environment_script": str(source.parent / "environment.sh")},
        "provenance": evidence(),
        "studies": [row],
        "counts": row["counts"],
    }
    return manifest, {row["name"]: state}


def test_preparation_freezes_the_reference_grid_and_rejects_changes(
    history: tuple[Path, dict], broad_grid_spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "slices"
    plan = fixture_plan(history, root, grid_spec=broad_grid_spec)
    calls = []

    def planned(destination: Path, source_root: Path, **kwargs: object) -> tuple[dict, dict]:
        calls.append((destination, source_root, kwargs))
        return copy.deepcopy(plan)

    monkeypatch.setattr(campaign, "plan_campaign", planned)
    monkeypatch.setattr(campaign, "training_environment", lambda *_: plan[0]["training"])
    preview = campaign.prepare(root, history[0].parent, grid_spec=broad_grid_spec, dry_run=True)
    assert preview["dry_run"] and not root.exists()
    prepared = campaign.prepare(root, history[0].parent, grid_spec=broad_grid_spec)
    reference = str(broad_grid_spec.resolve())
    expected_hash = campaign.file_digest(broad_grid_spec)
    assert prepared["grid_spec"] == reference
    assert prepared["studies"][0]["grid_spec"] == {"path": reference, "sha256": expected_hash}
    assert prepared["studies"][0]["source_files"][reference] == expected_hash
    assert prepared["frozen_files"][reference] == expected_hash
    assert all(kwargs == {"grid_spec": broad_grid_spec.resolve()} for _, _, kwargs in calls)
    assert len(calls) == 2
    assert campaign.prepare(root, history[0].parent) == prepared
    assert campaign.prepare(root, history[0].parent, grid_spec=broad_grid_spec) == prepared
    different_reference = tmp_path / "different-reference.yaml"
    different_reference.write_bytes(broad_grid_spec.read_bytes())
    with pytest.raises(ValueError, match="different grid specification"):
        campaign.prepare(root, history[0].parent, grid_spec=different_reference)
    broad_grid_spec.write_text(broad_grid_spec.read_text() + "\n# Reference changed after preparation.\n")
    with pytest.raises(ValueError, match="Frozen campaign file changed"):
        campaign.prepare(root, history[0].parent)


def test_cli_passes_the_reference_grid_to_dry_run(
    broad_grid_spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "slices"
    source = tmp_path / "history"
    calls = []

    def preview(destination: Path, source_root: Path, **kwargs: object) -> dict:
        calls.append((destination, source_root, kwargs))
        return {"root": str(destination), "counts": {}, "remaining_runs": 0, "studies": []}

    monkeypatch.setattr(campaign, "bootstrap", lambda _: None)
    monkeypatch.setattr(campaign, "prepare", preview)
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        [
            "tune_stability.py",
            "prepare",
            "--root",
            str(root),
            "--source-root",
            str(source),
            "--grid-spec",
            str(broad_grid_spec),
            "--dry-run",
        ],
    )
    campaign.main()
    assert calls == [(root.resolve(), source.resolve(), {"dry_run": True, "grid_spec": broad_grid_spec.resolve()})]
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_cli_rejects_a_changed_grid_before_bootstrap_or_launch(
    broad_grid_spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "slices"
    root.mkdir()
    original_reference = tmp_path / "original-reference.yaml"
    original_reference.write_bytes(broad_grid_spec.read_bytes())
    atomic_json(root / "manifest.json", {"grid_spec": str(original_reference.resolve())})
    before = (root / "manifest.json").read_bytes()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A mismatched grid must be rejected before bootstrap or launch")

    monkeypatch.setattr(campaign, "bootstrap", forbidden)
    monkeypatch.setattr(campaign, "run", forbidden)
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        [
            "tune_stability.py",
            "run",
            "--root",
            str(root),
            "--grid-spec",
            str(original_reference),
            "--grid-spec",
            str(broad_grid_spec),
        ],
    )
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2
    assert "Existing campaign uses a different grid specification" in capsys.readouterr().err
    assert (root / "manifest.json").read_bytes() == before


def test_preparation_is_readonly_when_dry_and_idempotent_when_saved(
    history: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "slices"
    plan = fixture_plan(history, root)
    monkeypatch.setattr(campaign, "plan_campaign", lambda *_: copy.deepcopy(plan))
    monkeypatch.setattr(campaign, "training_environment", lambda *_: plan[0]["training"])
    preview = campaign.prepare(root, history[0].parent, dry_run=True)
    assert preview["dry_run"] and not root.exists()
    prepared = campaign.prepare(root, history[0].parent)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    again = campaign.prepare(root, history[0].parent)
    assert prepared == again
    assert all(path.read_bytes() == content for path, content in before.items())
    assert set(prepared["frozen_files"]) >= {str(root / "code" / name) for name in campaign.CODE_FILES}
    assert (root / "run.sh").stat().st_mode & 0o111
    assert (root / "manifest.json").is_file()
    (root / history[0].name / "spec.yaml").write_text("changed")
    with pytest.raises(ValueError, match="Frozen campaign file changed"):
        campaign.prepare(root, history[0].parent)


def test_saved_campaign_detects_unplanned_new_trials(history: tuple[Path, dict], tmp_path: Path) -> None:
    source, _ = history
    row, state = campaign.plan_study(source, tmp_path / "slices")
    directory = Path(row["directory"])
    directory.mkdir(parents=True)
    save_state(directory, state)
    manifest = {"provenance": state["provenance"]}
    campaign.check_study(manifest, row)
    state["trials"]["unplanned"] = {}
    save_state(directory, state)
    with pytest.raises(ValueError, match="Unexpected or missing trial"):
        campaign.check_study(manifest, row)


def test_run_study_resumes_without_changing_its_anchor(
    history: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = history
    root = tmp_path / "slices"
    manifest, states = fixture_plan(history, root)
    row = manifest["studies"][0]
    directory = Path(row["directory"])
    directory.mkdir(parents=True)
    state = states[row["name"]]
    save_state(directory, state)
    atomic_json(root / "manifest.json", manifest)
    monkeypatch.setattr(campaign, "verify_manifest", lambda _: None)
    calls = []

    def complete(action: str, destination: Path, **kwargs: object) -> dict:
        calls.append((action, destination, kwargs))
        saved = load_state(destination)
        base = Config.model_validate(saved["base"])
        for phase in TuningSpec.model_validate(saved["spec"]).phases:
            for candidate in expand(base, phase):
                for seed in campaign.SEEDS:
                    key = trial_id(candidate, seed, saved["provenance"])
                    if key not in saved["trials"]:
                        saved["trials"][key] = publish(candidate["config"], seed, 3.4, tmp_path / "new")
        saved["status"] = "complete"
        save_state(destination, saved)
        return {"status": "complete"}

    monkeypatch.setattr(campaign, "execute", complete)
    anchor = copy.deepcopy(row["anchor"])
    assert campaign.run_study(root, row["name"])["status"] == "complete"
    first = load_state(directory)
    assert campaign.run_study(root, row["name"])["status"] == "complete"
    assert load_state(directory)["trials"] == first["trials"]
    assert read_manifest(root)["studies"][0]["anchor"] == anchor
    assert all(action == "resume" and kwargs == {"use_local_defaults": False} for action, _, kwargs in calls)


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())
