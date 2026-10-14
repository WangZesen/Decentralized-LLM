import copy
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from decentralized_llm.cache import TokenCache, training_boundaries
from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.launch import SubmissionRejected
from decentralized_llm.runtime import atomic_json, environment
from decentralized_llm.tuning import backends, controller
from decentralized_llm.tuning.backends import Backend, scheduler_status
from decentralized_llm.tuning.controller import execute, load_state, reconcile, study_lock
from decentralized_llm.tuning.estimate import estimate, ranking
from decentralized_llm.tuning.spec import (
    Interval,
    Phase,
    PowerInterval,
    Record,
    TuningSpec,
    expand,
    grid,
    read_spec,
    trial_id,
)


@pytest.fixture
def tuning_file(tiny_config: Config, tmp_path: Path) -> Path:
    save_config(tiny_config, tmp_path / "base.yaml")
    raw = yaml.safe_load(Path("configs/tuning-example.yaml").read_text())
    raw["base"] = {"configs": ["base.yaml"]}
    raw["controller"].update(poll_seconds=0.001, result_grace_seconds=0)
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def publish_result(config: Config, *, invalid: bool = False) -> None:
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    cache = TokenCache(config.data.cache_dir)
    atomic_json(output / "run.json", {"cache": cache.manifest["identity"], "environment": environment()})
    loss = (
        (math.log10(config.optimizer.lr) + 3) ** 2
        + (config.optimizer.weight_decay - 0.1) ** 2
        + (config.optimizer.beta1 - 0.85) ** 2
        + config.runtime.seed / 100000
    )
    atomic_json(
        output / "result.json",
        {
            "cursor": training_boundaries(config, config.model.parameter_count)[-1],
            "validation": {"loss": loss, "split": "subset" if invalid else "full", "validation_complete": True},
        },
    )


class FakeBackend(Backend):
    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[Record] = []
        self.failures = 0
        self.invalid = False
        self.interrupt = False
        self.unknown = False
        self.max_active = 0

    def submit(self, config: Config, attempt: Record) -> Record:
        self.submissions.append(copy.deepcopy(attempt))
        save_config(config, Path(attempt["directory"]) / "config.yaml")
        return {"job_id": str(len(self.submissions))}

    def poll(self, attempts: list[Record]) -> dict[str, Record]:
        self.max_active = max(self.max_active, len(attempts))
        if attempts and self.interrupt:
            raise KeyboardInterrupt
        result = {}
        for attempt in attempts:
            if self.unknown:
                continue
            failed = attempt["number"] <= self.failures
            if not failed:
                publish_result(load_config(Path(attempt["directory"]) / "config.yaml"), invalid=self.invalid)
            result[attempt["name"]] = {
                "state": "FAILED" if failed else "COMPLETED",
                "exit_code": "1:0" if failed else "0:0",
            }
        return result


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(controller, "Backend", lambda: backend)
    return backend


def test_search_ranges_and_numeric_dedup(tiny_config: Config) -> None:
    assert [p["value"] for p in grid(Interval(scale="linear", min=0.1, max=0.3, points=3))] == [0.1, 0.2, 0.3]
    assert [p["value"] for p in grid(Interval(scale="log", min=0.0001, max=0.01, points=3))] == [0.0001, 0.001, 0.01]
    powers = grid(PowerInterval(scale="power", base=2, exponent_min=-4, exponent_max=0, points=5))
    assert [p["expression"] for p in powers] == ["2^-4", "2^-3", "2^-2", "2^-1", "2^0"]
    fractional = grid(PowerInterval(scale="power", base=2, exponent_min=-0.5, exponent_max=0.5, points=3))
    assert fractional[0]["value"] == pytest.approx(2**-0.5)
    phase = Phase(
        name="a",
        search={"optimizer.lr": PowerInterval(scale="power", base=2, exponent_min=-4, exponent_max=-4, points=1)},
        top_k=1,
    )
    powered = expand(tiny_config, phase)[0]
    phase.search = {"optimizer.lr": [0.0625, 0.0625]}
    numeric = expand(tiny_config, phase)
    assert len(numeric) == 1 and numeric[0]["id"] == powered["id"]
    assert powered["labels"]["optimizer.lr"]["expression"] == "2^-4"


@pytest.mark.parametrize(
    "search",
    [
        {"scale": "log", "min": 0, "max": 1, "points": 2},
        {"scale": "linear", "min": 2, "max": 1, "points": 2},
        {"scale": "linear", "min": 1, "max": 2, "points": 1},
        {"scale": "power", "base": -2, "exponent_min": 0, "exponent_max": 2, "points": 3},
        {"scale": "power", "base": 10, "exponent_min": 1000, "exponent_max": 1001, "points": 2},
        {"scale": "power", "base": 10, "exponent_min": -1001, "exponent_max": -1000, "points": 2},
        [float("inf")],
        [],
    ],
)
def test_invalid_searches(tuning_file: Path, search: Any) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["phases"][0]["search"]["optimizer.lr"] = search
    with pytest.raises(ValueError):
        TuningSpec.model_validate(raw)


def test_initial_values_reserved_fields_and_integer_ranges(tuning_file: Path, tiny_config: Config) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["phases"][0]["initial_values"] = {}
    with pytest.raises(ValueError, match="initial_values"):
        TuningSpec.model_validate(raw)
    raw = yaml.safe_load(tuning_file.read_text())
    raw["phases"][0]["search"]["runtime.seed"] = [1, 2]
    with pytest.raises(ValueError, match="reserved"):
        TuningSpec.model_validate(raw)
    phase = Phase(
        name="integer", search={"training.data_reuse_factor": Interval(scale="linear", min=1, max=3, points=3)}, top_k=1
    )
    assert [c["values"]["training.data_reuse_factor"] for c in expand(tiny_config, phase)] == [1, 2, 3]
    phase.search = {"training.data_reuse_factor": [1.5]}
    with pytest.raises(ValueError):
        expand(tiny_config, phase)


def test_joint_carryover_and_revisited_dimension(tuning_file: Path) -> None:
    spec, base = read_spec(tuning_file)
    candidates = expand(base, spec.phases[0])
    assert len(candidates) == 15
    parents = [candidates[0], candidates[-1]]
    children = expand(base, spec.phases[1], parents)
    assert len(children) == 6
    expected = {(p["values"]["optimizer.lr"], p["values"]["optimizer.weight_decay"]) for p in parents}
    assert {(c["values"]["optimizer.lr"], c["values"]["optimizer.weight_decay"]) for c in children} == expected
    assert all("optimizer.lr" in child["labels"] for child in children)
    revisit = Phase(name="again", search={"optimizer.lr": [0.001], "optimizer.weight_decay": [0.1]}, top_k=1)
    assert len(expand(base, revisit, parents)) == 1


def test_total_estimate_and_no_dry_run_artifacts(cache: TokenCache, tuning_file: Path, tmp_path: Path) -> None:
    destination = tmp_path / "study"
    result = execute("run", destination, tuning_file, dry_run=True)
    assert result["exact"] and result["new_jobs"] == 44
    assert result["additional_attempts_upper_bound"] == 132
    assert [row["new_jobs"] for row in result["phases"]] == [30, 8, 6]
    assert not destination.exists()
    assert result["phases"][0]["search"]["optimizer.lr"][0]["expression"] == "10^-4"


def test_unknown_winners_estimate_is_upper_bound(tuning_file: Path) -> None:
    spec, base = read_spec(tuning_file)
    spec.phases.append(Phase(name="revisit", search={"optimizer.lr": [0.001]}, top_k=1))
    result = estimate(spec, base, {})
    assert not result["exact"] and "at most" in result["summary"]


def test_estimate_normalizes_numeric_aliases(tuning_file: Path) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["phases"][0]["initial_values"]["optimizer.beta1"] = "0.9"
    raw["phases"][1]["search"]["optimizer.beta1"] = [0.85, "0.9", 0.9, 0.95]
    tuning_file.write_text(yaml.safe_dump(raw))
    spec, base = read_spec(tuning_file)
    result = estimate(spec, base, {})
    assert result["exact"] and result["new_jobs"] == 44


def test_schedule_variant_search(tiny_config: Config) -> None:
    phase = Phase(name="schedule", search={"lr_schedule.name": ["cosine", "wsd"]}, top_k=1)
    candidates = expand(tiny_config, phase)
    assert {candidate["config"]["lr_schedule"]["name"] for candidate in candidates} == {"cosine", "wsd"}
    wsd = Config.model_validate(candidates[1]["config"])
    assert wsd.lr_schedule.warmup_tokens_per_parameters == tiny_config.lr_schedule.warmup_tokens_per_parameters
    phase.search = {"lr_schedule.name": ["cosine"], "lr_schedule.min_lr_ratio": [0.1]}
    assert expand(wsd, phase)[0]["config"]["lr_schedule"]["min_lr_ratio"] == 0.1


@pytest.mark.parametrize("name", ["cosine", "wsd"])
def test_fractional_warmup_search(tiny_config: Config, name: str) -> None:
    phase = Phase(
        name="warmup",
        search={
            "lr_schedule.name": [name],
            "lr_schedule.warmup_tokens_per_parameters": Interval(scale="linear", min=0, max=0.5, points=3),
        },
        top_k=1,
    )
    candidates = expand(tiny_config, phase)
    assert [c["values"]["lr_schedule.warmup_tokens_per_parameters"] for c in candidates] == [0, 0.25, 0.5]
    assert [Config.model_validate(c["config"]).lr_schedule.warmup_tokens_per_parameters for c in candidates] == [
        0,
        0.25,
        0.5,
    ]


def test_estimate_counts_existing_attempts(tuning_file: Path) -> None:
    spec, base = read_spec(tuning_file)
    candidates = expand(base, spec.phases[0])
    keys = [trial_id(candidates[0], seed, {}) for seed in spec.seeds()]
    trials = {
        keys[0]: {"id": keys[0], "status": "submitted", "attempts": [{}], "provenance": {}},
        keys[1]: {"id": keys[1], "status": "failed", "attempts": [{}, {}], "provenance": {}},
    }
    result = estimate(spec, base, {}, trials)
    assert result["already_submitted"] == 1
    assert result["new_jobs"] == 43
    assert result["additional_attempts_upper_bound"] == 129
    trials[keys[1]]["attempts"].append({})
    result = estimate(spec, base, {}, trials)
    assert result["completion_blocked"] and result["exhausted_trials"] == [keys[1]]


def test_full_controller_reuses_trials_and_extends(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    destination = tmp_path / "study"
    assert execute("run", destination, tuning_file)["status"] == "complete"
    assert len(fake_backend.submissions) == 44
    state = load_state(destination)
    assert len(state["trials"]) == 44
    assert all(phase["finished"] >= phase["started"] for phase in state["phases"])
    assert all(
        earlier["finished"] <= later["started"]
        for earlier, later in zip(state["phases"], state["phases"][1:], strict=False)
    )
    assert fake_backend.max_active == 1
    assert state["winner"]["values"]["optimizer.lr"] == 0.001
    assert state["winner"]["values"]["optimizer.beta1"] == 0.85
    assert "10^-3" in (destination / "report.md").read_text()
    assert load_config(destination / "winner.yaml").optimizer.lr == 0.001
    before = len(fake_backend.submissions)
    assert execute("resume", destination, dry_run=True)["new_jobs"] == 0
    assert execute("resume", destination)["status"] == "complete"
    assert len(fake_backend.submissions) == before
    with pytest.raises(ValueError, match="already contains state"):
        execute("run", destination, tuning_file, dry_run=True)
    raw = yaml.safe_load(tuning_file.read_text())
    raw["confirmation"]["seeds"] = 6
    tuning_file.write_text(yaml.safe_dump(raw))
    result = execute("extend", destination, tuning_file, dry_run=True)
    assert result["exact"] and result["new_jobs"] == 2
    assert execute("extend", destination, tuning_file)["revision"] == 2
    assert len(fake_backend.submissions) == before + 2
    assert (destination / "revisions/0001/winner.yaml").exists()
    # Revising search notation preserves numeric identities but changes phase display metadata.
    raw["phases"][0]["search"]["optimizer.lr"] = [0.001]
    tuning_file.write_text(yaml.safe_dump(raw))
    execute("extend", destination, tuning_file)
    assert len(fake_backend.submissions) == before + 2
    assert "optimizer.lr" not in load_state(destination)["winner"]["labels"]


@pytest.mark.parametrize("invalid", [False, True])
def test_retry_exhaustion_and_fresh_attempts(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    invalid: bool,
) -> None:
    fake_backend.failures = 0 if invalid else 9
    fake_backend.invalid = invalid
    directory = tmp_path / "study"
    assert execute("run", directory, tuning_file)["status"] == "failed"
    assert len(fake_backend.submissions) == 3
    attempts = fake_backend.submissions
    assert len({attempt["directory"] for attempt in attempts}) == 3
    configs = [load_config(Path(attempt["directory"]) / "config.yaml") for attempt in attempts]
    assert len({config.runtime.seed for config in configs}) == 1
    assert execute("resume", directory)["status"] == "failed"
    assert len(fake_backend.submissions) == 3
    fake_backend.failures = 3
    fake_backend.invalid = False
    raw = yaml.safe_load(tuning_file.read_text())
    raw["controller"]["max_retries"] = 3
    tuning_file.write_text(yaml.safe_dump(raw))
    assert execute("extend", directory, tuning_file)["status"] == "complete"


def test_resume_after_controller_interruption(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    fake_backend.interrupt = True
    destination = tmp_path / "study"
    assert execute("run", destination, tuning_file)["status"] == "interrupted"
    assert len(fake_backend.submissions) == 1
    fake_backend.interrupt = False
    assert execute("resume", destination)["status"] == "complete"
    assert len(fake_backend.submissions) == 44


def test_provenance_change_requires_extension(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "study"
    execute("run", destination, tuning_file)
    monkeypatch.setattr(controller, "source_digest", lambda _: "different")
    with pytest.raises(ValueError, match="provenance changed"):
        execute("resume", destination)
    result = execute("extend", destination, tuning_file, dry_run=True)
    assert result["new_jobs"] == 44


def test_controller_lock(tmp_path: Path) -> None:
    with study_lock(tmp_path):
        with pytest.raises(ValueError, match="another controller"):
            with study_lock(tmp_path):
                pytest.fail("second controller acquired the lock")


def test_ranking_uses_requested_seeds_and_stable_ties(tuning_file: Path) -> None:
    spec, base = read_spec(tuning_file)
    candidates = expand(base, spec.phases[0])[:2]
    trials = {}
    for candidate in candidates:
        for seed in [42, 43, 44]:
            key = trial_id(candidate, seed, {})
            trials[key] = {"id": key, "status": "succeeded", "loss": 1.0 if seed < 44 else 1000.0}
    ranks = ranking(candidates, [42, 43], trials, {})
    assert ranks is not None
    assert [row["id"] for row in ranks] == sorted(candidate["id"] for candidate in candidates)
    assert all(row["mean_loss"] == 1.0 for row in ranks)


def test_scheduler_reconciliation_and_receipt_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempt = {"name": "trial-a", "directory": str(tmp_path), "created": time.time(), "backend": "slurm"}
    (tmp_path / "submission").mkdir()
    atomic_json(tmp_path / "submission/receipt.json", {"job_id": "123", "cluster": "cluster-a"})
    calls = []

    def fake_query(args: list[str]) -> str:
        calls.append(args)
        if args[0] == "squeue":
            return ""
        return "123|trial-a|COMPLETED|0:0\n123.batch|batch|FAILED|1:0\n"

    monkeypatch.setattr(backends, "query", fake_query)
    status = scheduler_status([attempt])["trial-a"]
    assert status["state"] == "COMPLETED" and status["job_id"] == "123"
    assert attempt["cluster"] == "cluster-a"
    assert all("--clusters=cluster-a" in args for args in calls)
    assert "--allocations" in calls[1]
    monkeypatch.setattr(backends, "query", lambda _: "123|trial-a|REQUEUED\n")
    assert scheduler_status([attempt])["trial-a"]["state"] == "REQUEUED"
    monkeypatch.setattr(backends, "query", lambda _: "")
    assert scheduler_status([attempt]) == {}


def test_missing_scheduler_records_do_not_trigger_retry(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    destination = tmp_path / "study"
    fake_backend.interrupt = True
    execute("run", destination, tuning_file)
    state = load_state(destination)
    fake_backend.interrupt = False
    fake_backend.unknown = True
    assert not reconcile(state, fake_backend, 0)
    trial = next(iter(state["trials"].values()))
    assert trial["status"] == "submitted" and len(trial["attempts"]) == 1
    assert "Unresolved attempt" in state["warning"]


def test_accounting_lag_outage_and_publication_grace(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend.interrupt = True
    destination = tmp_path / "study"
    execute("run", destination, tuning_file)
    state = load_state(destination)
    trial = next(iter(state["trials"].values()))
    attempt = trial["attempts"][-1]
    name = attempt["name"]
    monkeypatch.setattr(fake_backend, "poll", lambda _: {name: {"state": "COMPLETED", "exit_code": None}})
    assert not reconcile(state, fake_backend, 0)
    assert trial["status"] == "submitted"

    def timeout(_: list[Record]) -> dict[str, Record]:
        raise subprocess.TimeoutExpired("sacct", 30)

    monkeypatch.setattr(fake_backend, "poll", timeout)
    assert not reconcile(state, fake_backend, 0)
    assert trial["status"] == "submitted"
    monkeypatch.setattr(fake_backend, "poll", lambda _: {name: {"state": "COMPLETED", "exit_code": "0:0"}})
    assert reconcile(state, fake_backend, 120)
    assert trial["status"] == "submitted"
    attempt["terminal_seen"] -= 121
    assert reconcile(state, fake_backend, 120)
    assert trial["status"] == "failed" and len(trial["attempts"]) == 1


def test_failed_retry_uses_extended_scheduler_settings(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["controller"]["max_retries"] = 0
    tuning_file.write_text(yaml.safe_dump(raw))
    fake_backend.failures = 1
    destination = tmp_path / "study"
    execute("run", destination, tuning_file)
    raw["controller"]["max_retries"] = 1
    raw["base"]["overrides"] = {"launcher.partition": "new-partition", "launcher.kind": "slurm"}
    tuning_file.write_text(yaml.safe_dump(raw, sort_keys=False))
    monkeypatch.setenv("account", "test-account")
    monkeypatch.setattr("shutil.which", lambda _: "/bin/slurm")
    execute("extend", destination, tuning_file)
    retry = fake_backend.submissions[1]
    assert retry["backend"] == "slurm"
    config = load_config(Path(retry["directory"]) / "config.yaml")
    assert config.launcher.kind == "slurm" and config.launcher.partition == "new-partition"


def test_slurm_concurrency_cap(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["base"]["overrides"] = {"launcher.kind": "slurm", "launcher.partition": "gpu"}
    raw["controller"]["max_jobs"] = 5
    tuning_file.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("account", "test-account")
    monkeypatch.setattr("shutil.which", lambda _: "/bin/slurm")
    execute("run", tmp_path / "study", tuning_file)
    assert fake_backend.max_active == 5


def test_rejected_submission_is_bounded(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected(config: Config, attempt: Record) -> Record:
        raise SubmissionRejected("rejected")

    monkeypatch.setattr(fake_backend, "submit", rejected)
    directory = tmp_path / "study"
    assert execute("run", directory, tuning_file)["status"] == "failed"
    assert len(next(iter(load_state(directory)["trials"].values()))["attempts"]) == 3


def test_real_local_tuning_cli(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    save_config(tiny_config, tmp_path / "base.yaml")
    spec = {
        "base": {"configs": ["base.yaml"]},
        "screening_seeds": 1,
        "phases": [
            {
                "name": "joint",
                "search": {"optimizer.lr": [0.001, 0.002], "optimizer.weight_decay": [0.0, 0.1]},
                "top_k": 1,
            }
        ],
        "confirmation": {"seeds": 2},
        "controller": {"poll_seconds": 0.05, "result_grace_seconds": 0},
    }
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(spec))
    directory = tmp_path / "study"
    result = subprocess.run(
        [sys.executable, "-m", "decentralized_llm", "tune", "run", "--spec", str(path), "--output", str(directory)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-5000:]
    state = load_state(directory)
    assert state["status"] == "complete" and len(state["trials"]) == 5
    assert all(trial["status"] == "succeeded" for trial in state["trials"].values())
