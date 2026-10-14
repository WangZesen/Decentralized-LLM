import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from decentralized_llm.tuning.controller import materialize_phase, trial_directory
from decentralized_llm.tuning.plots import phase_heatmaps, write_heatmaps
from decentralized_llm.tuning.report import write_report
from decentralized_llm.tuning.spec import Record


def candidate(identifier: str, values: Record, losses: list[float | None]) -> Record:
    return {
        "id": identifier,
        "values": values,
        "labels": {},
        "seeds": [
            {"seed": 42 + i, "loss": loss, "status": "succeeded" if loss is not None else "pending"}
            for i, loss in enumerate(losses)
        ],
    }


def phase(candidates: list[Record], *keys: str) -> Record:
    return {
        "name": "screening" if keys else "confirmation",
        "definition": {"search": dict.fromkeys(keys, [])} if keys else {"seeds": 2},
        "candidates": candidates,
        "selected": [],
        "rankings": [],
    }


def test_screening_facets_labels_and_missing_cells() -> None:
    a = candidate("a", {"lr": 0.1, "beta": 0.9, "momentum": 0.5}, [2, 4])
    a["labels"]["lr"] = {"expression": "10^-1"}
    b = candidate("b", {"lr": 0.2, "beta": 0.9, "momentum": 0.5}, [1, None])
    c = candidate("c", {"lr": 0.1, "beta": 0.8, "momentum": 0.5}, [None, None])
    c["seeds"][0]["status"] = "failed"
    d = candidate("d", {"lr": 0.2, "beta": 0.8, "momentum": 0.7}, [4, 6])
    current = phase([a, b, c, d], "lr", "beta")
    current["selected"] = ["a"]
    panels = phase_heatmaps(current)
    assert len(panels) == 2
    assert panels[0].losses.shape == (2, 2)
    assert panels[0].losses[0, 0] == 3
    assert panels[1].losses[1, 1] == 5
    assert panels[0].statuses.tolist() == [["complete", "pending"], ["failed", "unsearched"]]
    assert panels[0].selected.tolist() == [[True, False], [False, False]]
    assert "momentum=0.5" in panels[0].title
    # Power notation survives even when another candidate has only a numeric label for the same value.
    assert panels[0].columns[0] == "10^-1 (0.1)"
    current["definition"]["search"]["momentum"] = []
    assert len(phase_heatmaps(current)) == 2


def test_confirmation_seed_values_means_and_selection() -> None:
    a = candidate("a", {"lr": 0.1}, [4, 6])
    b = candidate("b", {"lr": 0.2}, [3, None])
    current = phase([a, b])
    current["rankings"] = [{"id": "b"}, {"id": "a"}]
    current["selected"] = ["a"]
    panel = phase_heatmaps(current)[0]
    assert panel.columns == ["42", "43", "Mean"]
    assert panel.losses[0, 0] == 3 and np.isnan(panel.losses[0, -1])
    assert panel.losses[1].tolist() == [4, 6, 5]
    assert panel.statuses[0].tolist() == ["complete", "pending", "pending"]
    assert panel.selected.tolist() == [[False, False, False], [False, False, True]]


@pytest.mark.parametrize("loss", [None, 3.5])
@pytest.mark.parametrize("confirmation", [False, True])
def test_headless_heatmaps_handle_missing_and_equal_losses(
    tmp_path: Path, loss: float | None, confirmation: bool
) -> None:
    current = phase([candidate("a", {"lr": 0.1}, [loss, loss])], *([] if confirmation else ["lr"]))
    plots = write_heatmaps(tmp_path, current)
    assert len(plots) == 1
    png = tmp_path / plots[0]["png"]
    svg = tmp_path / plots[0]["svg"]
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg"
    assert "pending" in svg.read_text() if loss is None else "3.50000" in svg.read_text()
    assert phase_heatmaps(phase([], "lr")) == []


def test_rendered_facets_share_scale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import seaborn as sns

    actual = sns.heatmap
    limits = []

    def record(*args: Any, **kwargs: Any) -> Any:
        limits.append((kwargs["vmin"], kwargs["vmax"]))
        return actual(*args, **kwargs)

    monkeypatch.setattr(sns, "heatmap", record)
    current = phase(
        [
            candidate("a", {"lr": 0.1, "beta": 0.9}, [1, 3]),
            candidate("b", {"lr": 0.1, "beta": 0.8}, [5, 7]),
        ],
        "lr",
    )
    assert len(write_heatmaps(tmp_path, current)) == 2
    assert limits == [(2, 6), (2, 6)]


def test_report_uses_legacy_phase_seed_subset_without_relocating(tmp_path: Path) -> None:
    root = tmp_path / "trials/old-trial"
    root.mkdir(parents=True)
    marker = root / "metrics.jsonl"
    marker.write_text("original log\n")
    state = {
        "revision": 1,
        "status": "complete",
        "provenance": {},
        "updated": 10,
        "trials": {
            "old-trial": {"id": "old-trial", "seed": 42, "loss": 2.5, "status": "succeeded", "attempts": []},
            "later-seed": {"id": "later-seed", "seed": 43, "loss": 9, "status": "succeeded", "attempts": []},
        },
        "phases": [
            {
                "name": "legacy",
                "complete": True,
                "definition": {"search": {"lr": [0.1]}},
                "reused_trials": 0,
                "new_trials": 1,
                "candidates": [{"id": "a", "values": {"lr": 0.1}, "labels": {}}],
                "trials": {"a": ["old-trial"]},
                "selected": ["a"],
                "rankings": [
                    {
                        "id": "a",
                        "values": {"lr": 0.1},
                        "labels": {},
                        "mean_loss": 2.5,
                        "std_loss": None,
                        "seeds": [{"seed": 42, "loss": 2.5, "trial_id": "old-trial"}],
                    }
                ],
            }
        ],
    }
    before = copy.deepcopy(state)
    report = write_report(tmp_path, state)
    assert state == before
    assert marker.read_text() == "original log\n"
    assert not (tmp_path / "trials/legacy").exists()
    panel = phase_heatmaps(report["phases"][0])[0]
    assert panel.losses.tolist() == [[2.5]]
    assert report["phases"][0]["rankings"][0]["mean_loss"] == 2.5
    assert report == json.loads((tmp_path / "report.json").read_text())
    assert report["phases"][0]["heatmaps"][0]["png"] in (tmp_path / "report.md").read_text()


def test_legacy_trial_views_are_relative_idempotent_and_preserve_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "trials/old-trial"
    attempt = root / "attempt-001"
    attempt.mkdir(parents=True)
    marker = attempt / "result.json"
    marker.write_text("original result\n")
    trial = {"id": "old-trial", "attempts": [{"directory": str(attempt)}]}
    state = {"trials": {"old-trial": trial}}
    current = {"name": "next", "trials": {"candidate": ["old-trial"]}}
    for _ in range(2):
        materialize_phase(tmp_path, state, current)
    link = tmp_path / "trials/next/old-trial"
    assert link.readlink() == Path("../old-trial")
    assert trial_directory(tmp_path, trial) == root
    assert marker.read_text() == "original result\n"
    current["name"] = "confirmation"
    materialize_phase(tmp_path, state, current)
    assert (tmp_path / "trials/confirmation/old-trial").readlink() == Path("../old-trial")
    link.unlink()
    link.mkdir()
    current["name"] = "next"
    with pytest.raises(ValueError, match="already exists"):
        materialize_phase(tmp_path, state, current)
    assert link.is_dir() and not link.is_symlink()
