import json
import sys
from pathlib import Path

import pytest
import torch

from decentralized_llm.config import Config
from scripts.benchmark_decentralized_optimizer import cached_pair, summarize


def test_resume_requires_complete_matching_same_allocation_pair(tiny_config: Config, tmp_path: Path) -> None:
    parent = tmp_path / "pair"
    for selector in ("reference", "auto"):
        directory = parent / selector
        directory.mkdir(parents=True)
        config = tiny_config.model_copy(deep=True)
        config.runtime.output_dir = directory
        config.runtime.decentralized_optimizer = selector
        report = {
            "config": config.model_dump(mode="json"),
            "optimizer": {"source_revision": "revision"},
            "slurm_job_id": "123",
        }
        (directory / "benchmark-rank-00000.json").write_text(json.dumps(report))
        if selector == "reference":
            assert cached_pair(tiny_config, parent, "revision", 0) is None
    assert cached_pair(tiny_config, parent, "revision", 0) is not None
    assert cached_pair(tiny_config, parent, "changed", 0) is None
    assert cached_pair(tiny_config, parent, "revision", 1) is None
    changed = tiny_config.model_copy(deep=True)
    changed.optimizer.lr *= 2
    assert cached_pair(changed, parent, "revision", 0) is None
    path = parent / "auto/benchmark-rank-00000.json"
    report = json.loads(path.read_text())
    report["slurm_job_id"] = "456"
    path.write_text(json.dumps(report))
    assert cached_pair(tiny_config, parent, "revision", 0) is None


def test_summary_uses_paired_ratios_and_ignores_incomplete_trials() -> None:
    rows = []
    for trial, (reference, candidate) in enumerate(((10, 10.5), (100, 80), (11, 22))):
        for selector, speed in (("reference", reference), ("auto", candidate)):
            rows.append(
                {
                    "model": "20m",
                    "topology": "complete",
                    "scheme": "awc",
                    "error_feedback_gamma": 1.0,
                    "selector": selector,
                    "trial": trial,
                    "tokens_per_second": speed,
                }
            )
    result = summarize(rows)
    assert result["comparisons"][0]["speedup"] == pytest.approx(1.05)
    rows.append({**rows[-1], "trial": 3, "tokens_per_second": 1e6})
    assert summarize(rows) == result


@pytest.mark.parametrize("fail_first", [False, True])
def test_trajectory_matrix_collects_all_cases_before_failing(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_first: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts import qualify_decentralized_optimizer as runner

    output = tmp_path / "qualification"
    monkeypatch.setattr(sys, "argv", ["qualify", "--models", "20m", "--schemes", "awc", "--output", str(output)])
    monkeypatch.setattr(runner, "load_config", lambda *args: tiny_config.model_copy(deep=True))
    monkeypatch.setattr(runner, "setup", lambda config: torch.device("cpu"))
    monkeypatch.setattr(runner, "prepare_parameter_groups", lambda topology: None)
    monkeypatch.setattr(runner, "optimizer_revision", lambda: "revision")
    cases = []

    def qualify(config: Config, path: Path, **kwargs: object) -> dict:
        assert kwargs["raise_on_failure"] is False
        path.mkdir(parents=True)
        cases.append((config.distributed.topology, config.distributed.error_feedback_gamma, config.runtime.seed))
        return {
            "passed": not (fail_first and len(cases) == 1),
            "validation_delta": 0.0,
            "gates": {"all_recorded_updates_passed": False},
        }

    monkeypatch.setattr(runner, "qualify", qualify)
    if fail_first:
        with pytest.raises(RuntimeError, match="see matrix.json for all cases"):
            runner.main()
    else:
        runner.main()
    result = json.loads((output / "matrix.json").read_text())
    assert len(set(cases)) == len(result["rows"]) == 8
    assert result["complete"] is True
    assert result["passed"] is not fail_first
    assert sum(not row["passed"] for row in result["rows"]) == int(fail_first)
