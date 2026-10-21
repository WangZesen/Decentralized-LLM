import copy
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from decentralized_llm import cli
from decentralized_llm import config as config_module
from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.data.cache import TokenCache, training_boundaries
from decentralized_llm.launch.slurm import SubmissionRejected
from decentralized_llm.runtime import atomic_json, environment
from decentralized_llm.tuning import controller, report
from decentralized_llm.tuning import jobs as backends
from decentralized_llm.tuning.batching import allocation_failures, training_attempts
from decentralized_llm.tuning.controller import execute, load_state, reconcile, study_lock
from decentralized_llm.tuning.estimate import estimate, ranking
from decentralized_llm.tuning.jobs import Backend, scheduler_status
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
    raw = yaml.safe_load(Path("configs/tunings/tuning-example.yaml").read_text())
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


def publish_divergence(config: Config, *, rank: int = 0) -> Path:
    publish_result(config)
    (config.runtime.output_dir / "result.json").unlink()
    path = config.runtime.output_dir / f"failure-rank-{rank:05d}.json"
    atomic_json(
        path,
        {
            "kind": "nonfinite_update",
            "rank": rank,
            "message": "nonfinite gradient norm",
            "traceback": "NonfiniteUpdateError: nonfinite gradient norm",
            "progress": {"cursor": 0, "step": 0, "completed_epochs": 0},
        },
    )
    return path


class FakeBackend(Backend):
    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[Record] = []
        self.failures = 0
        self.invalid = False
        self.diverges: Callable[[Config], bool] = lambda _: False
        self.interrupt = False
        self.unknown = False
        self.max_active = 0
        self.allocation_submissions: list[Record] = []
        self.allocation_members: dict[str, list[Record]] = {}

    def partition_limit(self, partition: str) -> int | None:
        return 4320

    def submit_allocation(self, configs: list[Config], allocation: Record, attempts: list[Record]) -> Record:
        self.allocation_submissions.append(copy.deepcopy(allocation))
        self.allocation_members[allocation["name"]] = copy.deepcopy(attempts)
        for config, attempt in zip(configs, attempts, strict=True):
            Path(attempt["directory"]).mkdir(parents=True)
            self.submit(config, attempt)
        return {"job_id": str(len(self.allocation_submissions))}

    def submit(self, config: Config, attempt: Record) -> Record:
        self.submissions.append(copy.deepcopy(attempt))
        save_config(config, Path(attempt["directory"]) / "config.yaml")
        return {"job_id": str(len(self.submissions))}

    def publish_outcome(self, attempt: Record) -> bool:
        if attempt["number"] <= self.failures:
            return True
        config = load_config(Path(attempt["directory"]) / "config.yaml")
        if self.diverges(config):
            publish_divergence(config)
            return True
        publish_result(config, invalid=self.invalid)
        return False

    def poll(self, attempts: list[Record]) -> dict[str, Record]:
        self.max_active = max(self.max_active, len(attempts))
        if attempts and self.interrupt:
            raise KeyboardInterrupt
        result = {}
        for attempt in attempts:
            if self.unknown:
                continue
            if "members" in attempt:
                for member in self.allocation_members[attempt["name"]]:
                    failed = self.publish_outcome(member)
                    path = Path(member["execution_record"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(
                        path,
                        {
                            "allocation_id": attempt["id"],
                            "trial_id": member["trial_id"],
                            "attempt": member["number"],
                            "started": True,
                            "state": "FAILED" if failed else "COMPLETED",
                            "exit_code": int(failed),
                        },
                    )
                result[attempt["name"]] = {"state": "COMPLETED", "exit_code": "0:0"}
                continue
            failed = self.publish_outcome(attempt)
            result[attempt["name"]] = {
                "state": "FAILED" if failed else "COMPLETED",
                "exit_code": "1:0" if failed else "0:0",
            }
        return result


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(controller, "Backend", lambda: backend)

    def plot_artifacts(directory: Path, phase: Record) -> list[dict[str, str]]:
        # Real rendering is covered in test_tuning_plots and the local CLI smoke test.
        # Controller tests exercise report links and archival without rasterizing every revision.
        (directory / "plots").mkdir(exist_ok=True)
        paths = {extension: f"plots/{phase['name']}-001.{extension}" for extension in ("png", "svg")}
        for path in paths.values():
            (directory / path).write_text("plot fixture")
        return [paths]

    monkeypatch.setattr(report, "write_heatmaps", plot_artifacts)
    return backend


def test_base_layers_precedence_and_relative_paths(
    tuning_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["base"]["overrides"] = {"training.batch_tokens": 48, "runtime.cpu_threads": 4}
    tuning_file.write_text(yaml.safe_dump(raw))
    working = tmp_path / "working"
    working.mkdir()
    # This file must not replace the spec-relative base.yaml.
    (working / "base.yaml").write_text("model: {width: 32}\n")
    (working / "first.yaml").write_text(
        "model: {context_length: 8}\ntraining: {batch_tokens: 1}\nruntime: {cpu_threads: 2, output_dir: relative-run}\n"
    )
    (working / "second.yaml").write_text("model: {width: 16}\ntraining: {batch_tokens: 96}\n")
    monkeypatch.chdir(working)
    configs = [Path("first.yaml"), Path("second.yaml")]
    spec, base = read_spec(tuning_file, configs=configs)
    assert base.training.batch_tokens == 48 and base.runtime.cpu_threads == 4
    assert base.model.width == 16 and base.model.context_length == 8
    assert base.runtime.output_dir == working / "relative-run"
    assert spec.base.configs == [tmp_path / "base.yaml", *[working / path for path in configs]]
    _, base = read_spec(
        tuning_file,
        configs=configs,
        overrides=["training.batch_tokens=72", "training.batch_tokens=96", "runtime.cpu_threads=3"],
    )
    assert base.training.batch_tokens == 96 and base.runtime.cpu_threads == 3
    # Phase search values continue to override the composed base.
    spec, base = read_spec(tuning_file, overrides=["optimizer.lr=0.5"])
    assert base.optimizer.lr == 0.5
    assert all(candidate["config"]["optimizer"]["lr"] != 0.5 for candidate in expand(base, spec.phases[0]))


def test_base_layers_switch_variants(tuning_file: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "variants.yaml"
    overlay.write_text(
        "lr_schedule: {name: wsd, decay_fraction: 0.3}\nlauncher: {kind: slurm, partition: test, network: job_vni}\n"
    )
    _, base = read_spec(tuning_file, configs=[overlay])
    assert base.lr_schedule.name == "wsd" and base.lr_schedule.decay_fraction == 0.3
    assert base.launcher.kind == "slurm" and base.launcher.network == "job_vni"
    _, base = read_spec(
        tuning_file,
        configs=[overlay],
        overrides=["lr_schedule.name=cosine", "lr_schedule.min_lr_ratio=0.2", "launcher.kind=local"],
    )
    assert base.lr_schedule.name == "cosine" and base.lr_schedule.min_lr_ratio == 0.2
    assert base.launcher.kind == "local"


@pytest.mark.parametrize(
    "override",
    ["missing-equals", "runtime..compile=true", "runtime.unknown=true", "training.batch_tokens=1", "optimizer.lr=.nan"],
)
def test_invalid_base_overrides(tuning_file: Path, override: str) -> None:
    with pytest.raises(ValueError):
        read_spec(tuning_file, overrides=[override])


def test_invalid_extra_config(tuning_file: Path, tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    with pytest.raises(FileNotFoundError):
        read_spec(tuning_file, configs=[path])
    path.write_text("[not, a, mapping]")
    with pytest.raises(ValueError, match="YAML mapping"):
        read_spec(tuning_file, configs=[path])


@pytest.mark.parametrize("action", ["run", "extend"])
def test_cli_forwards_base_overrides(
    action: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[tuple, Record]] = []

    def invoked(*args: Any, **kwargs: Any) -> Record:
        calls.append((args, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(controller, "execute", invoked)
    destination = tmp_path / "study"
    arguments = ["--output", str(destination)] if action == "run" else [str(destination)]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decentralized-llm",
            "tune",
            action,
            *arguments,
            "--spec",
            "study.yaml",
            "-c",
            "model.yaml",
            "--config",
            "setup.yaml",
            "--set",
            "training.batch_tokens=8192",
            "--set",
            "training.batch_tokens=16384",
            "--dry-run",
            "--no-local-config",
        ],
    )
    cli.main()
    args, kwargs = calls[0]
    assert args == (action, destination, Path("study.yaml"))
    assert kwargs["configs"] == [Path("model.yaml"), Path("setup.yaml")]
    assert kwargs["overrides"] == ["training.batch_tokens=8192", "training.batch_tokens=16384"]
    assert kwargs["dry_run"] and not kwargs["use_local_defaults"]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize("action", ["resume", "status", "report"])
@pytest.mark.parametrize("option,value", [("--config", "model.yaml"), ("--set", "training.batch_tokens=8192")])
def test_saved_study_actions_reject_overrides(
    action: str, option: str, value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["decentralized-llm", "tune", action, str(tmp_path), option, value])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    with pytest.raises(ValueError, match="require run or extend"):
        if option == "--config":
            execute(action, tmp_path, configs=[Path(value)])
        else:
            execute(action, tmp_path, overrides=[value])


def test_override_dry_run_is_read_only(
    cache: TokenCache, tuning_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = tmp_path / "batch.yaml"
    overlay.write_text("training: {batch_tokens: 48}\n")
    original = overlay.read_bytes()
    checked: list[Config] = []
    monkeypatch.setattr(controller, "validate", lambda config, **kwargs: checked.append(config))
    destination = tmp_path / "dry-study"
    result = execute(
        "run", destination, tuning_file, configs=[overlay], overrides=["training.batch_tokens=72"], dry_run=True
    )
    assert result["new_jobs"] == 44
    assert checked and all(config.training.batch_tokens == 72 for config in checked)
    assert not destination.exists() and not config_module.LOCAL_DEFAULTS_PATH.exists()
    assert overlay.read_bytes() == original


def test_base_overrides_are_frozen_and_refreshed_on_extend(
    cache: TokenCache, tuning_file: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw.update(
        screening_seeds=1,
        phases=[{"name": "lr", "search": {"optimizer.lr": [0.001]}, "top_k": 1}],
        confirmation={"seeds": 1},
    )
    spec_text = yaml.safe_dump(raw)
    tuning_file.write_text(spec_text)
    overlay = tmp_path / "batch.yaml"
    overlay.write_text("training: {batch_tokens: 72}\n")
    directory = tmp_path / "study"
    assert (
        execute("run", directory, tuning_file, configs=[overlay], overrides=["training.batch_tokens=48"])["status"]
        == "complete"
    )
    frozen = load_state(directory)["base"]
    assert frozen["training"]["batch_tokens"] == 48
    assert str(overlay) in load_state(directory)["spec"]["base"]["configs"]
    overlay.write_text("training: {batch_tokens: 96}\n")
    assert execute("resume", directory, dry_run=True)["new_jobs"] == 0
    overlay.unlink()
    tuning_file.unlink()
    assert execute("resume", directory)["status"] == "complete"
    assert load_state(directory)["base"] == frozen and len(fake_backend.submissions) == 1
    tuning_file.write_text(spec_text)
    overlay.write_text("training: {batch_tokens: 96}\n")
    assert execute("extend", directory, tuning_file, configs=[overlay], dry_run=True)["new_jobs"] == 1
    assert execute("extend", directory, tuning_file, configs=[overlay])["revision"] == 2
    assert load_state(directory)["base"]["training"]["batch_tokens"] == 96
    assert json.loads((directory / "revisions/0001/state.json").read_text())["base"] == frozen
    # Omitted CLI inputs are not inherited from the preceding revision.
    assert execute("extend", directory, tuning_file)["revision"] == 3
    assert load_state(directory)["base"]["training"]["batch_tokens"] == 24


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


@pytest.mark.parametrize("low, high, points", [(-4, 0, 5), (-0.5, 0.5, 3), (-2, -2, 1)])
def test_power_exponent_spacing_defaults_to_linear(low: float, high: float, points: int) -> None:
    raw = {"scale": "power", "base": 2, "exponent_min": low, "exponent_max": high, "points": points}
    default = PowerInterval.model_validate(raw)
    explicit = PowerInterval.model_validate({**raw, "exponent_scale": "linear"})
    assert default.exponent_scale == "linear"
    assert grid(default) == grid(explicit)


@pytest.mark.parametrize("base", [2, 0.5])
@pytest.mark.parametrize(
    "exponents",
    [
        [2, 4, 8],
        [-8, -4, -2],
        [0.125, 0.25, 0.5],
        [-0.5, -0.25, -0.125],
        [2.5],
        [-2.5],
        [2.1234567890123457, 8.987654321098765],
        [-8.987654321098765, -2.1234567890123457],
    ],
)
def test_logarithmic_power_exponents(base: float, exponents: list[float]) -> None:
    points = grid(
        PowerInterval(
            scale="power",
            base=base,
            exponent_scale="log",
            exponent_min=exponents[0],
            exponent_max=exponents[-1],
            points=len(exponents),
        )
    )
    assert [point["exponent"] for point in points] == exponents
    assert [point["value"] for point in points] == pytest.approx([base**exponent for exponent in exponents])
    assert [point["expression"] for point in points] == [f"{base:.15g}^{exponent:.15g}" for exponent in exponents]


@pytest.mark.parametrize(
    "values",
    [[0.0001, 0.001, 0.01], [0.125, 0.25, 0.5], [0.12345678901234567], [0.12345678901234567, 0.9876543210987654]],
)
def test_logarithmic_value_intervals(values: list[float]) -> None:
    points = grid(Interval(scale="log", min=values[0], max=values[-1], points=len(values)))
    assert [point["value"] for point in points] == values


@pytest.mark.parametrize(
    "overrides",
    [
        {"exponent_min": 0, "exponent_max": 8},
        {"exponent_min": -8, "exponent_max": 0},
        {"exponent_min": -8, "exponent_max": 8},
        {"exponent_min": 0, "exponent_max": 0, "points": 1},
        {"exponent_min": 8, "exponent_max": 2},
        {"exponent_min": -2, "exponent_max": -8},
        {"exponent_min": 2, "exponent_max": 2},
        {"points": 1},
        {"points": 0},
        {"points": 2.5},
        {"exponent_scale": "invalid"},
        {"base": 10, "exponent_min": 1000, "exponent_max": 1001},
        {"base": 10, "exponent_min": -1001, "exponent_max": -1000},
    ],
)
def test_invalid_logarithmic_power_searches(tuning_file: Path, overrides: Record) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["phases"][0]["search"]["optimizer.lr"] = {
        "scale": "power",
        "base": 2,
        "exponent_scale": "log",
        "exponent_min": -8,
        "exponent_max": -2,
        "points": 3,
        **overrides,
    }
    with pytest.raises(ValueError):
        TuningSpec.model_validate(raw)


def test_logarithmic_power_spec_and_dry_run(cache: TokenCache, tuning_file: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["phases"][0]["search"]["optimizer.lr"] = {
        "scale": "power",
        "base": 2,
        "exponent_scale": "log",
        "exponent_min": -8,
        "exponent_max": -2,
        "points": 3,
    }
    tuning_file.write_text(yaml.safe_dump(raw))
    spec, base = read_spec(tuning_file)
    serialized = yaml.safe_load(yaml.safe_dump(spec.model_dump(mode="json")))
    assert serialized["phases"][0]["search"]["optimizer.lr"]["exponent_scale"] == "log"
    assert TuningSpec.model_validate(serialized) == spec

    powered = expand(base, spec.phases[0])
    spec.phases[0].search["optimizer.lr"] = [1 / 256, 1 / 16, 1 / 4]
    numeric = expand(base, spec.phases[0])
    assert [candidate["id"] for candidate in powered] == [candidate["id"] for candidate in numeric]
    assert powered[0]["labels"]["optimizer.lr"]["expression"] == "2^-8"

    destination = tmp_path / "study"
    result = execute("run", destination, tuning_file, dry_run=True)
    assert result["exact"] and result["new_jobs"] == 32
    points = result["phases"][0]["search"]["optimizer.lr"]
    assert [point["expression"] for point in points] == ["2^-8", "2^-4", "2^-2"]
    assert [point["value"] for point in points] == [1 / 256, 1 / 16, 1 / 4]
    assert not destination.exists()


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
    for trial in state["trials"].values():
        root = destination / trial["directory"]
        assert root.is_dir() and not root.is_symlink()
        assert not (destination / "trials" / trial["id"]).exists()
        assert all(Path(attempt["directory"]).parent == root for attempt in trial["attempts"])
    for phase in state["phases"]:
        for keys in phase["trials"].values():
            for key in keys:
                view = destination / "trials" / phase["name"] / key
                root = destination / state["trials"][key]["directory"]
                assert view.resolve() == root
                if view != root:
                    assert view.is_symlink() and not Path(os.readlink(view)).is_absolute()
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
    archived = json.loads((destination / "revisions/0001/report.json").read_text())
    for phase in archived["phases"]:
        assert phase["heatmaps"]
        for plot in phase["heatmaps"]:
            assert all((destination / "revisions/0001" / path).is_file() for path in plot.values())
    # Revising search notation preserves numeric identities but changes phase display metadata.
    raw["phases"][0]["search"]["optimizer.lr"] = [0.001]
    raw["phases"][0]["name"] = "renamed-first-phase"
    tuning_file.write_text(yaml.safe_dump(raw))
    execute("extend", destination, tuning_file)
    assert len(fake_backend.submissions) == before + 2
    assert "optimizer.lr" not in load_state(destination)["winner"]["labels"]
    assert all(path.is_symlink() for path in (destination / "trials/renamed-first-phase").iterdir())


def test_local_defaults_are_frozen_for_resume_and_refreshed_for_extend(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    base_path = tuning_file.parent / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["runtime"].pop("execution_backend")
    base["runtime"].pop("cpu_threads")
    base_path.write_text(yaml.safe_dump(base))
    spec = yaml.safe_load(tuning_file.read_text())
    spec.update(
        screening_seeds=1,
        phases=[{"name": "lr", "search": {"optimizer.lr": [0.001]}, "top_k": 1}],
        confirmation={"seeds": 1},
    )
    tuning_file.write_text(yaml.safe_dump(spec))
    directory = tmp_path / "local-defaults-study"
    assert execute("run", directory, tuning_file)["status"] == "complete"
    defaults = config_module.LOCAL_DEFAULTS_PATH
    assert defaults.is_file()
    state = load_state(directory)
    frozen = state["base"]
    assert frozen["runtime"]["execution_backend"] == "native"
    for changed in ("runtime: {execution_backend: gh200}", "runtime: [broken"):
        defaults.write_text(changed)
        assert execute("resume", directory, dry_run=True)["new_jobs"] == 0
        assert execute("resume", directory)["status"] == "complete"
        assert execute("status", directory)["status"] == "complete"
        execute("report", directory)
        assert load_state(directory)["base"] == frozen
        assert defaults.read_text() == changed
        for trial in state["trials"].values():
            controller.check_result(trial, trial["attempts"][-1])
    defaults.unlink()
    execute("resume", directory, dry_run=True)
    execute("status", directory)
    execute("report", directory)
    assert not defaults.exists()
    defaults.write_text("runtime: {execution_backend: native, cpu_threads: 2}")
    assert execute("extend", directory, tuning_file)["revision"] == 2
    assert load_state(directory)["base"]["runtime"]["cpu_threads"] == 2
    assert json.loads((directory / "revisions/0001/state.json").read_text())["base"] == frozen


@pytest.mark.parametrize("dry_run,use_local_defaults", [(True, True), (True, False), (False, False)])
def test_controller_dry_run_and_opt_out_do_not_initialize_defaults(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    dry_run: bool,
    use_local_defaults: bool,
) -> None:
    directory = tmp_path / "no-local-defaults-study"
    execute("run", directory, tuning_file, dry_run=dry_run, use_local_defaults=use_local_defaults)
    assert not config_module.LOCAL_DEFAULTS_PATH.parent.exists()
    if dry_run:
        assert not directory.exists()


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
    assert len({Path(attempt["directory"]).parent for attempt in attempts}) == 1
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


@pytest.mark.parametrize("slurm", [False, True])
@pytest.mark.parametrize("diverge_in", ["some", "screen", "confirmation"])
def test_divergence_continues_tuning_without_retries(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    slurm: bool,
    diverge_in: str,
) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw.update(
        screening_seeds=1,
        phases=[
            {
                "name": "screen",
                "initial_values": {"optimizer.beta1": 0.8},
                "search": {"optimizer.lr": [0.001, 0.1]},
                "top_k": 2,
            },
            {"name": "refine", "search": {"optimizer.beta1": [0.9]}, "top_k": 2},
        ],
        confirmation={"seeds": 2},
    )
    if slurm:
        raw["base"]["overrides"] = {"launcher.kind": "slurm", "launcher.partition": "gpu"}
        raw["controller"].update(jobs_per_allocation=4, max_jobs=1)
        monkeypatch.setenv("account", "test-account")
        monkeypatch.setattr("shutil.which", lambda _: "/bin/slurm")
    tuning_file.write_text(yaml.safe_dump(raw))
    fake_backend.diverges = lambda config: (
        diverge_in == "screen"
        or diverge_in == "some"
        and config.optimizer.lr == 0.1
        or diverge_in == "confirmation"
        and config.runtime.seed > raw["seed_start"]
    )
    destination = tmp_path / "study"
    assert execute("run", destination, tuning_file)["status"] == "complete"
    state = load_state(destination)
    diverged = [trial for trial in state["trials"].values() if trial["status"] == "diverged"]
    assert diverged
    assert all(trial["status"] in {"succeeded", "diverged"} for trial in state["trials"].values())
    assert all(len(trial["attempts"]) == 1 for trial in state["trials"].values())
    for trial in diverged:
        assert trial["loss"] == "inf" and "result" not in trial
        attempt = trial["attempts"][0]
        assert attempt["status"] == "diverged"
        assert not (Path(attempt["directory"]) / "run/result.json").exists()
        assert Path(trial["divergence"]["file"]).is_file()
    for phase in state["phases"]:
        for row in phase["rankings"]:
            if row["mean_loss"] == "inf":
                assert row["id"] not in phase["selected"] and row["std_loss"] is None
    if diverge_in == "some":
        assert state["winner"]["values"]["optimizer.lr"] == 0.001
        assert len(state["phases"][1]["candidates"]) == 1
        assert (destination / "winner.yaml").exists()
    else:
        assert state["winner"] is None and "No finite-loss candidates" in state["reason"]
        assert not (destination / "winner.yaml").exists()
        assert state["phases"][-1]["name"] == diverge_in
    if diverge_in == "confirmation":
        assert all(row["seeds"][0]["loss"] != "inf" for row in state["phases"][-1]["rankings"])
    for filename in ("state.json", "report.json"):
        content = (destination / filename).read_text()
        assert '"inf"' in content and "Infinity" not in content and "NaN" not in content
    markdown = (destination / "report.md").read_text()
    assert "diverged; loss=∞" in markdown
    submissions = len(fake_backend.submissions)
    assert execute("resume", destination)["status"] == "complete"
    preview = execute("extend", destination, tuning_file, dry_run=True)
    assert preview["new_jobs"] == 0 and not preview["completion_blocked"]
    assert execute("extend", destination, tuning_file)["status"] == "complete"
    assert len(fake_backend.submissions) == submissions
    # Reusing a divergent result requires the original, authenticated evidence.
    Path(diverged[0]["divergence"]["file"]).unlink()
    with pytest.raises(ValueError, match="divergence"):
        execute("resume", destination)


@pytest.mark.parametrize("backend", ["local", "slurm", "grouped"])
@pytest.mark.parametrize(
    "problem", [None, "generic", "rank", "progress", "provenance", "timeout", "oom", "cancel", "parent_timeout"]
)
def test_divergence_requires_valid_worker_and_execution_evidence(
    cache: TokenCache, tiny_config: Config, tmp_path: Path, backend: str, problem: str | None
) -> None:
    config = tiny_config.model_copy(deep=True)
    config.distributed.gpus_per_node = 2
    config.runtime.output_dir = tmp_path / "attempt/run"
    marker = publish_divergence(config, rank=1)
    # A peer's generic distributed failure must not hide the originating numerical error.
    atomic_json(config.runtime.output_dir / "failure-rank-00000.json", {"traceback": "peer exited"})
    if problem in {"generic", "rank", "progress"}:
        record = json.loads(marker.read_text())
        if problem == "generic":
            record.pop("kind")
        elif problem == "rank":
            record["rank"] = 0
        else:
            record["progress"]["cursor"] = -1
        atomic_json(marker, record)
    trial = {
        "id": "trial",
        "status": "submitted",
        "config": config.model_dump(mode="json"),
        "seed": config.runtime.seed,
        "provenance": controller.provenance(cache),
    }
    if problem == "provenance":
        trial["provenance"]["source_sha256"] = "different source"
    attempt = {
        "name": "attempt",
        "status": "submitted",
        "backend": "slurm" if backend == "grouped" else backend,
        "directory": str(tmp_path / "attempt"),
        "number": 1,
    }
    trial["attempts"] = [attempt]
    condition = {"timeout": "TIMEOUT", "parent_timeout": "TIMEOUT", "oom": "OUT_OF_MEMORY", "cancel": "CANCELLED"}.get(
        problem or "", "FAILED"
    )
    if backend == "grouped":
        record_path = tmp_path / "step.json"
        attempt["execution_record"] = str(record_path)
        atomic_json(
            record_path,
            {
                "allocation_id": "a",
                "trial_id": "trial",
                "attempt": 1,
                "started": True,
                "state": "FAILED",
                "exit_code": {"timeout": 143, "oom": 137, "cancel": -15}.get(problem or "", 1),
            },
        )
        allocation = {"id": "a", "status": "finished", "scheduler_state": condition, "terminal_seen": 0}
        controller.reconcile_step(trial, attempt, allocation, 0)
    else:

        class FinishedBackend(Backend):
            def poll(self, attempts: list[Record]) -> dict[str, Record]:
                return {"attempt": {"state": condition, "exit_code": "1:0"}}

        assert reconcile({"trials": {"trial": trial}}, FinishedBackend(), 0)
    if problem is None or backend == "grouped" and problem == "parent_timeout":
        assert trial["status"] == attempt["status"] == "diverged"
        assert trial["loss"] == "inf" and trial["divergence"]["rank"] == 1
    else:
        assert trial["status"] == attempt["status"] == "failed"
        assert "loss" not in trial


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


@pytest.fixture
def batched_spec(tuning_file: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["base"]["overrides"] = {"launcher.kind": "slurm", "launcher.partition": "gpu", "launcher.time": "00:15:00"}
    raw["phases"] = [{"name": "screen", "search": {"optimizer.lr": [0.001 * i for i in range(1, 11)]}, "top_k": 1}]
    raw["screening_seeds"] = raw["confirmation"]["seeds"] = 1
    raw["controller"].update(jobs_per_allocation=4, max_jobs=2)
    tuning_file.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("account", "test-account")
    monkeypatch.setattr("shutil.which", lambda _: "/bin/slurm")
    return tuning_file


def test_batched_controller_balances_before_concurrency_limit(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    destination = tmp_path / "study"
    preview = execute("run", destination, batched_spec, dry_run=True)
    assert not destination.exists()
    assert preview["new_jobs"] == 10 and preview["new_allocations"] == 3 and preview["allocations_exact"]
    assert [row["jobs"] for row in preview["phases"][0]["allocations"]] == [4, 3, 3]
    assert execute("run", destination, batched_spec)["status"] == "complete"
    assert [len(item["trial_ids"]) for item in fake_backend.allocation_submissions] == [4, 3, 3]
    assert [item["time_minutes"] for item in fake_backend.allocation_submissions] == [60, 45, 45]
    assert fake_backend.max_active == 2
    state = load_state(destination)
    assert all(training_attempts(trial) == 1 for trial in state["trials"].values())
    assert all(
        trial["attempts"][0]["stdout"].startswith(str(destination / "slurm")) for trial in state["trials"].values()
    )
    assert execute("status", destination)["allocations"] == {"finished": 3}
    assert json.loads((destination / "report.json").read_text())["allocation_counts"] == {"finished": 3}
    assert "Slurm allocations" in (destination / "report.md").read_text()
    assert execute("resume", destination)["status"] == "complete"
    assert len(fake_backend.submissions) == 10


def test_batched_confirmation_and_reused_seeds(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(tuning_file.read_text())
    raw["base"]["overrides"] = {"launcher.kind": "slurm", "launcher.partition": "gpu", "launcher.time": "00:15:00"}
    raw["controller"].update(jobs_per_allocation=4, max_jobs=3)
    tuning_file.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("account", "test-account")
    monkeypatch.setattr("shutil.which", lambda _: "/bin/slurm")
    destination = tmp_path / "study"
    preview = execute("run", destination, tuning_file, dry_run=True)
    assert preview["new_jobs"] == 44 and preview["new_allocations"] == 12
    assert execute("run", destination, tuning_file)["status"] == "complete"
    assert len(fake_backend.submissions) == 44
    assert len(fake_backend.allocation_submissions) == 12
    confirmation = [item for item in fake_backend.allocation_submissions if item["phase"] == "confirmation"]
    assert [len(item["trial_ids"]) for item in confirmation] == [3, 3]


def test_batched_unknown_parent_and_publication_grace(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend.interrupt = True
    destination = tmp_path / "study"
    execute("run", destination, batched_spec)
    state = load_state(destination)
    fake_backend.interrupt = False
    fake_backend.unknown = True
    assert not reconcile(state, fake_backend, 120)
    assert controller.outstanding_allocations(state) == 2
    assert all(not trial["attempts"] or trial["status"] == "submitted" for trial in state["trials"].values())
    assert len(fake_backend.submissions) == 7
    # Terminal parent state alone does not mean the shared filesystem has published every record.
    monkeypatch.setattr(
        fake_backend,
        "poll",
        lambda parents: {item["name"]: {"state": "COMPLETED", "exit_code": "0:0"} for item in parents},
    )
    assert reconcile(state, fake_backend, 120)
    assert controller.outstanding_allocations(state) == 0
    assert sum(trial["status"] == "submitted" for trial in state["trials"].values()) == 7
    for item in state["allocations"].values():
        if item["status"] == "finished":
            item["terminal_seen"] -= 121
    assert reconcile(state, fake_backend, 120)
    assert sum(trial["status"] == "failed" for trial in state["trials"].values()) == 7
    assert all(training_attempts(trial) == 0 for trial in state["trials"].values())


def test_unknown_resource_winners_have_conservative_allocation_estimates(batched_spec: Path) -> None:
    raw = yaml.safe_load(batched_spec.read_text())
    raw["base"]["overrides"]["training.batch_tokens"] = 48
    raw["phases"][0]["search"] = {"distributed.gpus_per_node": [1, 2, 4]}
    raw["confirmation"]["seeds"] = 3
    batched_spec.write_text(yaml.safe_dump(raw))
    spec, base = read_spec(batched_spec)
    preview = estimate(spec, base, {})
    assert preview["phases"][0]["new_allocations"] == 3
    assert not preview["phases"][-1]["allocations_exact"]
    assert preview["phases"][-1]["max_allocation_time_minutes_upper_bound"] == 30


def test_partition_limit_rejects_entire_phase_before_submission(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_backend, "partition_limit", lambda _: 59)
    destination = tmp_path / "study"
    with pytest.raises(ValueError, match="jobs_per_allocation <= 3"):
        execute("run", destination, batched_spec, dry_run=True)
    assert not destination.exists()
    with pytest.raises(ValueError, match="exceeding partition"):
        execute("run", destination, batched_spec)
    assert not fake_backend.submissions
    monkeypatch.setattr(fake_backend, "partition_limit", lambda _: 60)
    assert execute("resume", destination)["status"] == "complete"


def test_summed_limits_round_individual_trials_before_grouping(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(batched_spec.read_text())
    raw["base"]["overrides"]["launcher.time"] = "1-00:00:01"
    batched_spec.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(fake_backend, "partition_limit", lambda _: 4 * 1441)
    destination = tmp_path / "study"
    preview = execute("run", destination, batched_spec, dry_run=True)
    assert [item["time_minutes"] for item in preview["phases"][0]["allocations"]] == [5764, 4323, 4323]
    assert not destination.exists()


def test_allocation_failure_keeps_success_and_retries_only_unfinished_trials(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(batched_spec.read_text())
    raw["phases"][0]["search"]["optimizer.lr"] = [0.001, 0.002, 0.003]
    raw["controller"]["max_jobs"] = 1
    batched_spec.write_text(yaml.safe_dump(raw))
    original_poll = fake_backend.poll
    first = True

    def partial_failure(records: list[Record]) -> dict[str, Record]:
        nonlocal first
        statuses = original_poll(records)
        if records and first:
            first = False
            parent = records[0]
            members = fake_backend.allocation_members[parent["name"]]
            path = Path(members[1]["execution_record"])
            record = json.loads(path.read_text())
            record.update(state="FAILED", exit_code=9)
            atomic_json(path, record)
            Path(members[2]["execution_record"]).unlink()
            statuses[parent["name"]] = {"state": "NODE_FAIL", "exit_code": "1:0"}
        return statuses

    monkeypatch.setattr(fake_backend, "poll", partial_failure)
    destination = tmp_path / "study"
    assert execute("run", destination, batched_spec)["status"] == "complete"
    trials = list(load_state(destination)["trials"].values())
    assert [len(trial["attempts"]) for trial in trials] == [1, 2, 2]
    assert [training_attempts(trial) for trial in trials] == [1, 2, 1]
    assert [allocation_failures(trial) for trial in trials] == [0, 0, 1]
    assert [len(item["trial_ids"]) for item in fake_backend.allocation_submissions] == [3, 2]


@pytest.mark.parametrize("started", [False, True])
def test_batched_retry_budgets_are_independent_and_bounded(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    started: bool,
) -> None:
    raw = yaml.safe_load(batched_spec.read_text())
    raw["phases"][0]["search"]["optimizer.lr"] = [0.001]
    batched_spec.write_text(yaml.safe_dump(raw))
    if started:
        fake_backend.failures = 99
    else:

        def reject(*_: object) -> Record:
            raise SubmissionRejected("scheduler rejected allocation")

        monkeypatch.setattr(fake_backend, "submit_allocation", reject)
    destination = tmp_path / "study"
    result = execute("run", destination, batched_spec)
    assert result["status"] == "failed"
    trial = next(iter(load_state(destination)["trials"].values()))
    assert len(trial["attempts"]) == 3
    assert training_attempts(trial) == (3 if started else 0)
    assert allocation_failures(trial) == (0 if started else 3)
    assert ("retry limit" if started else "allocation-failure limit") in result["reason"]
    assert execute("resume", destination)["status"] == "failed"
    assert len(next(iter(load_state(destination)["trials"].values()))["attempts"]) == 3


def test_step_launch_failure_does_not_consume_training_retry(tmp_path: Path) -> None:
    path = tmp_path / "step.json"
    attempt = {"number": 1, "allocation_id": "a", "execution_record": str(path), "status": "submitted"}
    trial = {"id": "t", "status": "submitted", "attempts": [attempt]}
    allocation = {"id": "a", "status": "submitted"}
    record = {"allocation_id": "a", "trial_id": "t", "attempt": 1, "started": True, "state": "RUNNING"}
    atomic_json(path, record)
    controller.reconcile_step(trial, attempt, allocation, 0)
    assert training_attempts(trial) == 1
    # The runner could publish RUNNING before discovering it cannot spawn srun.
    record.update(started=False, state="FAILED", exit_code=1, reason="Could not create process")
    atomic_json(path, record)
    controller.reconcile_step(trial, attempt, allocation, 0)
    assert training_attempts(trial) == 0 and allocation_failures(trial) == 1


def test_finished_steps_retain_parent_slot_until_allocation_ends(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(batched_spec.read_text())
    raw["controller"]["max_jobs"] = 1
    batched_spec.write_text(yaml.safe_dump(raw))
    original_poll = fake_backend.poll
    seen = set()
    active_counts = []

    def delayed_parent(records: list[Record]) -> dict[str, Record]:
        statuses = original_poll(records)
        for record in records:
            if record["name"] not in seen:
                seen.add(record["name"])
                statuses[record["name"]] = {"state": "RUNNING", "exit_code": None}
                active_counts.append(len(fake_backend.allocation_submissions))
        return statuses

    monkeypatch.setattr(fake_backend, "poll", delayed_parent)
    assert execute("run", tmp_path / "study", batched_spec)["status"] == "complete"
    assert active_counts == [1, 2, 3]
    assert fake_backend.max_active == 1


@pytest.mark.parametrize("member_name", [False, True])
def test_unresolved_recovery_releases_whole_allocation(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    member_name: bool,
) -> None:
    fake_backend.interrupt = True
    destination = tmp_path / "study"
    assert execute("run", destination, batched_spec)["status"] == "interrupted"
    state = load_state(destination)
    first = next(item for item in state["allocations"].values() if item["status"] == "submitted")
    fake_backend.interrupt = False
    original_poll = fake_backend.poll
    unknown_once = True

    def missing_once(records: list[Record]) -> dict[str, Record]:
        nonlocal unknown_once
        if unknown_once:
            unknown_once = False
            return {}
        return original_poll(records)

    monkeypatch.setattr(fake_backend, "poll", missing_once)
    name = state["trials"][first["trial_ids"][0]]["attempts"][0]["name"] if member_name else first["name"]
    assert execute("resume", destination, retry_unresolved=[name])["status"] == "complete"
    state = load_state(destination)
    for key in first["trial_ids"]:
        assert len(state["trials"][key]["attempts"]) == 2
        assert allocation_failures(state["trials"][key]) == 1
        assert training_attempts(state["trials"][key]) == 1


def test_batched_extension_replans_only_unsubmitted_work(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
) -> None:
    raw = yaml.safe_load(batched_spec.read_text())
    raw["controller"]["max_jobs"] = 1
    batched_spec.write_text(yaml.safe_dump(raw))
    fake_backend.interrupt = True
    destination = tmp_path / "study"
    execute("run", destination, batched_spec)
    raw["controller"]["jobs_per_allocation"] = 2
    batched_spec.write_text(yaml.safe_dump(raw))
    fake_backend.interrupt = False
    assert execute("extend", destination, batched_spec)["status"] == "complete"
    state = load_state(destination)
    assert sum(item["status"] == "abandoned" for item in state["allocations"].values()) == 2
    assert [len(item["trial_ids"]) for item in fake_backend.allocation_submissions] == [4, 2, 2, 2]
    assert len(fake_backend.submissions) == 10


@pytest.mark.parametrize("version", [1, 2])
def test_allocation_receipt_recovery_and_legacy_state(
    cache: TokenCache,
    tuning_file: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    fake_backend.interrupt = True
    destination = tmp_path / "study"
    execute("run", destination, tuning_file)
    state = load_state(destination)
    state["version"] = version
    state.pop("allocations")
    state["spec"]["controller"].pop("jobs_per_allocation")
    atomic_json(destination / "state.json", state)
    fake_backend.interrupt = False
    assert execute("resume", destination)["status"] == "complete"
    assert load_state(destination)["version"] == controller.VERSION
    parent = {
        "name": "batch-a",
        "directory": str(tmp_path),
        "receipt_path": str(tmp_path / "receipt.json"),
        "created": time.time(),
    }
    atomic_json(tmp_path / "receipt.json", {"job_id": "900", "cluster": "cluster-a"})
    monkeypatch.setattr(backends, "query", lambda args: "900|batch-a|RUNNING\n" if args[0] == "squeue" else "")
    assert scheduler_status([parent])["batch-a"]["state"] == "RUNNING"
    assert parent["job_id"] == "900" and parent["cluster"] == "cluster-a"


def test_legacy_slurm_jobs_share_concurrency_cap_with_new_allocations(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "study"

    # Prepare the study without submitting, then populate one legacy individual intent.
    def interrupt(_: str) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_backend, "partition_limit", interrupt)
    assert execute("run", destination, batched_spec)["status"] == "interrupted"
    state = load_state(destination)
    trial = next(iter(state["trials"].values()))
    controller.submit(destination, state, trial, fake_backend)
    legacy_name = trial["attempts"][-1]["name"]
    state["version"] = 1
    state.pop("allocations")
    atomic_json(destination / "state.json", state)
    original_poll = fake_backend.poll
    seen = []

    def hold_legacy(records: list[Record]) -> dict[str, Record]:
        names = {record["name"] for record in records}
        seen.append(names)
        result = original_poll([record for record in records if record["name"] != legacy_name])
        if legacy_name in names:
            if len(seen) <= 2:
                result[legacy_name] = {"state": "RUNNING", "exit_code": None}
            else:
                result.update(original_poll([record for record in records if record["name"] == legacy_name]))
        return result

    monkeypatch.setattr(fake_backend, "partition_limit", lambda _: 4320)
    monkeypatch.setattr(fake_backend, "poll", hold_legacy)
    assert execute("resume", destination)["status"] == "complete"
    assert any(legacy_name in names and len(names) == 2 for names in seen)
    assert max(map(len, seen)) == 2
    assert len(fake_backend.submissions) == 10
    assert [len(item["trial_ids"]) for item in fake_backend.allocation_submissions] == [3, 3, 3]


def test_interrupt_after_allocation_receipt_recovers_without_duplicate_submission(
    cache: TokenCache,
    batched_spec: Path,
    tmp_path: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_submit, original_poll = fake_backend.submit_allocation, fake_backend.poll
    interrupt = True

    def submitted(configs: list[Config], allocation: Record, attempts: list[Record]) -> Record:
        nonlocal interrupt
        receipt = original_submit(configs, allocation, attempts)
        original_poll([allocation])
        atomic_json(Path(allocation["receipt_path"]), receipt)
        if interrupt:
            interrupt = False
            raise KeyboardInterrupt
        return receipt

    def query(args: list[str]) -> str:
        if args[0] == "squeue":
            return ""
        return "\n".join(
            f"{index}|{item['name']}|COMPLETED|0:0" for index, item in enumerate(fake_backend.allocation_submissions, 1)
        )

    monkeypatch.setattr(fake_backend, "submit_allocation", submitted)
    monkeypatch.setattr(fake_backend, "poll", scheduler_status)
    monkeypatch.setattr(backends, "query", query)
    destination = tmp_path / "study"
    assert execute("run", destination, batched_spec)["status"] == "interrupted"
    state = load_state(destination)
    first = next(iter(state["allocations"].values()))
    assert first["status"] == "submitting" and "job_id" not in first
    assert execute("resume", destination)["status"] == "complete"
    state = load_state(destination)
    assert state["allocations"][first["id"]]["job_id"] == "1"
    assert len(fake_backend.allocation_submissions) == 3
    assert all(len(trial["attempts"]) == 1 for trial in state["trials"].values())


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
    overlay = tmp_path / "batch.yaml"
    overlay.write_text("training: {batch_tokens: 48}\n")
    directory = tmp_path / "study"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "decentralized_llm",
            "tune",
            "run",
            "--spec",
            str(path),
            "-c",
            overlay.name,
            "--set",
            "training.batch_tokens=36",
            "--output",
            str(directory),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        # Five real jobs each import torch in the supervisor, launcher, and worker.
        # Shared-filesystem startup plus headless plotting can exceed three minutes.
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-5000:]
    state = load_state(directory)
    assert state["status"] == "complete" and len(state["trials"]) == 5
    assert state["base"]["training"]["batch_tokens"] == 36
    assert all(trial["status"] == "succeeded" for trial in state["trials"].values())
