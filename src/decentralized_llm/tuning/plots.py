"""Phase-aware heatmap data and headless Seaborn report figures."""

import json
import math
import statistics
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decentralized_llm.tuning.spec import Record


@dataclass
class Heatmap:
    title: str
    xlabel: str
    ylabel: str
    columns: list[str]
    rows: list[str]
    losses: np.ndarray
    statuses: np.ndarray
    selected: np.ndarray


def value_label(candidate: Record, key: str) -> str:
    value = candidate["values"][key]
    label = candidate["labels"].get(key)
    if label:
        return f"{label['expression']} ({value:.6g})"
    return str(value)


def result_cell(seeds: list[Record]) -> tuple[float, str]:
    if any(seed["status"] == "diverged" for seed in seeds):
        return float("inf"), "diverged (∞)"
    if seeds and all(
        seed["status"] == "succeeded" and isinstance(seed.get("loss"), (int, float)) and math.isfinite(seed["loss"])
        for seed in seeds
    ):
        return statistics.mean(seed["loss"] for seed in seeds), "complete"
    return float("nan"), "failed" if any(seed["status"] == "failed" for seed in seeds) else "pending"


def empty_heatmap(title: str, xlabel: str, ylabel: str, columns: list[str], rows: list[str]) -> Heatmap:
    shape = (len(rows), len(columns))
    return Heatmap(
        title,
        xlabel,
        ylabel,
        columns,
        rows,
        np.full(shape, np.nan),
        np.full(shape, "unsearched", dtype=object),
        np.zeros(shape, dtype=bool),
    )


def phase_heatmaps(phase: Record) -> list[Heatmap]:
    """Use only the seed records belonging to this phase, including in partial reports."""
    candidates = phase["candidates"]
    if not candidates:
        return []
    selected = set(phase["selected"])
    search = list(phase["definition"].get("search", {}))
    if not search:
        order = {row["id"]: i for i, row in enumerate(phase["rankings"])}
        candidates = sorted(candidates, key=lambda row: (order.get(row["id"], len(order)), row["id"]))
        seeds = sorted({seed["seed"] for candidate in candidates for seed in candidate["seeds"]})
        labels = [
            "\n".join(f"{key}={value_label(candidate, key)}" for key in sorted(candidate["values"]))
            or candidate["id"][:12]
            for candidate in candidates
        ]
        panel = empty_heatmap(phase["name"], "Seed / mean", "Candidate", [str(s) for s in seeds] + ["Mean"], labels)
        for row, candidate in enumerate(candidates):
            by_seed = {seed["seed"]: seed for seed in candidate["seeds"]}
            for column, seed in enumerate(seeds):
                panel.losses[row, column], panel.statuses[row, column] = result_cell(
                    [by_seed[seed]] if seed in by_seed else []
                )
            panel.losses[row, -1], panel.statuses[row, -1] = result_cell(candidate["seeds"])
            panel.selected[row, -1] = candidate["id"] in selected
        return [panel]

    xkey = search[0]
    ykey = search[1] if len(search) > 1 else None
    facets = sorted(set().union(*(candidate["values"] for candidate in candidates)) - {xkey, ykey})
    axes: dict[str, dict[str, str]] = {}
    for key in search[:2]:
        axes[key] = {}
        for candidate in candidates:
            value = json.dumps(candidate["values"][key])
            if value not in axes[key] or key in candidate["labels"]:
                axes[key][value] = value_label(candidate, key)
    xpositions = {value: i for i, value in enumerate(axes[xkey])}
    ypositions = {value: i for i, value in enumerate(axes[ykey])} if ykey else {}
    panels: dict[tuple[str, ...], Heatmap] = {}
    for candidate in candidates:
        group = tuple(json.dumps(candidate["values"][key]) for key in facets)
        if group not in panels:
            caption = ", ".join(f"{key}={value_label(candidate, key)}" for key in facets)
            panels[group] = empty_heatmap(
                phase["name"] + (f"\n{caption}" if caption else ""),
                xkey,
                ykey or "",
                list(axes[xkey].values()),
                list(axes[ykey].values()) if ykey else ["Mean loss"],
            )
        panel = panels[group]
        column = xpositions[json.dumps(candidate["values"][xkey])]
        row = ypositions[json.dumps(candidate["values"][ykey])] if ykey else 0
        panel.losses[row, column], panel.statuses[row, column] = result_cell(candidate["seeds"])
        panel.selected[row, column] = candidate["id"] in selected
    return list(panels.values())


def write_heatmaps(directory: Path, phase: Record) -> list[dict[str, str]]:
    panels = phase_heatmaps(phase)
    if not panels:
        return []
    # Training workers need not import plotting libraries or initialize a GUI backend.
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    finite = np.concatenate([panel.losses[np.isfinite(panel.losses)] for panel in panels])
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if low == high:
        padding = max(abs(low) * 1e-6, 1e-6)
        low, high = low - padding, high + padding
    paths = []
    (directory / "plots").mkdir(parents=True, exist_ok=True)
    confirmation = not phase["definition"].get("search")
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=0.9):
        for index, panel in enumerate(panels, 1):
            row_lines = max(label.count("\n") + 1 for label in panel.rows)
            width = max(7.0, len(panel.columns) * 1.25 + (6 if confirmation else 3))
            height = max(3.8, len(panel.rows) * max(0.55, row_lines * 0.23) + 2.6)
            figure = Figure(figsize=(width, height), layout="constrained")
            FigureCanvasAgg(figure)
            ax = figure.subplots()
            try:
                sns.heatmap(
                    panel.losses,
                    ax=ax,
                    cmap="mako",
                    vmin=low,
                    vmax=high,
                    mask=~np.isfinite(panel.losses),
                    annot=True,
                    fmt=".5f",
                    linewidths=0.5,
                    linecolor="white",
                    xticklabels=panel.columns,
                    yticklabels=panel.rows,
                    cbar=bool(finite.size),
                    cbar_kws={
                        "label": (
                            "Final validation loss — lower is better"
                            if confirmation
                            else "Mean final validation loss — lower is better"
                        ),
                        **({"orientation": "horizontal", "pad": 0.12, "aspect": 40} if len(panel.rows) == 1 else {}),
                    },
                )
                ax.set_facecolor("#eeeeee")
                for (row, column), status in np.ndenumerate(panel.statuses):
                    if status != "complete":
                        ax.text(
                            column + 0.5,
                            row + 0.5,
                            status,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="#a32828" if status == "failed" else "#555555",
                        )
                    if panel.selected[row, column]:
                        ax.add_patch(Rectangle((column, row), 1, 1, fill=False, edgecolor="#d89000", linewidth=2.5))
                title = "\n".join(textwrap.fill(line, 95) for line in panel.title.splitlines())
                ax.set_title(title, pad=14)
                ax.set_xlabel(panel.xlabel)
                ax.set_ylabel(panel.ylabel)
                ax.tick_params(axis="both", length=0)
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=0 if confirmation else 35, ha="center" if confirmation else "right"
                )
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                figure.supxlabel("Outlined cells: selected candidates", fontsize=9)
                references = {}
                for extension in ("png", "svg"):
                    relative = Path("plots") / f"{phase['name']}-{index:03d}.{extension}"
                    destination = directory / relative
                    temporary = destination.with_suffix(f".{extension}.tmp")
                    figure.savefig(temporary, format=extension, dpi=160, bbox_inches="tight", pad_inches=0.12)
                    temporary.replace(destination)
                    references[extension] = relative.as_posix()
                paths.append(references)
            finally:
                figure.clear()
    return paths
