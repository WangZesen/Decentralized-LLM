"""Authenticated three-seed, winner-centered sensitivity reports.

The manifest fixes the center and the requested seeds. A point receives summary
statistics only after every requested seed has a valid final full-validation result.
"""

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER
except ModuleNotFoundError:
    from scripts.report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER

Record = dict[str, Any]
FIELDS = ("optimizer.lr", "optimizer.weight_decay", "optimizer.beta1", "optimizer.beta2")
FIELD_LABELS = ("Learning rate", "Weight decay", "Beta1", "Beta2")


def ordered_studies(studies: list[Record]) -> list[Record]:
    return sorted(studies, key=lambda row: (METHOD_ORDER.index(row["method"]), row["workers"], row["name"]))


def summarize_point(rows: list[Record], anchor: list[Record], seeds: list[int]) -> Record:
    """Use sample SD and matched-seed differences, never a finite-only subset mean."""
    if len({row["seed"] for row in rows}) != len(rows) or {row["seed"] for row in rows} != set(seeds):
        raise ValueError("Point must contain each requested seed exactly once")
    baseline = {row["seed"]: row for row in anchor}
    valid = [row for row in rows if row["status"] == "succeeded" and math.isfinite(row["loss"])]
    complete = len(valid) == len(seeds)
    deltas = [
        row["loss"] - baseline[row["seed"]]["loss"]
        for row in valid
        if row["seed"] in baseline
        and baseline[row["seed"]]["status"] == "succeeded"
        and math.isfinite(baseline[row["seed"]]["loss"])
    ]
    paired = complete and len(deltas) == len(seeds)
    outcomes = Counter(row["status"] for row in rows)
    status = (
        "complete"
        if complete
        else next((state for state in ("invalid", "diverged", "failed") if outcomes[state]), "incomplete")
    )
    return {
        "status": status,
        "completed_seeds": len(valid),
        "required_seeds": len(seeds),
        "outcomes": dict(outcomes),
        "mean_loss": statistics.mean(row["loss"] for row in valid) if complete else None,
        "std_loss": statistics.stdev(row["loss"] for row in valid) if complete and len(seeds) > 1 else None,
        "mean_delta": statistics.mean(deltas) if paired else None,
        "std_delta": statistics.stdev(deltas) if paired and len(seeds) > 1 else None,
        "min_delta": min(deltas) if paired else None,
        "max_delta": max(deltas) if paired else None,
        "better_than_anchor": paired and statistics.mean(deltas) < 0,
    }


def authenticate_identity(trial: Record, expected_provenance: Record) -> None:
    from decentralized_llm.config import Config
    from decentralized_llm.tuning.spec import fingerprint, numeric_recipe, trial_id

    if trial["provenance"] != expected_provenance:
        raise ValueError("Trial provenance differs from frozen campaign provenance")
    candidate = {"id": fingerprint(numeric_recipe(Config.model_validate(trial["config"])))}
    if (
        candidate["id"] != trial["candidate_id"]
        or trial_id(candidate, trial["seed"], expected_provenance) != trial["id"]
    ):
        raise ValueError("Trial scientific identity does not match its configuration, seed, and provenance")


def authenticated_trial(
    trial: Record | None,
    trial_id: str,
    seed: int,
    reused: bool,
    expected_provenance: Record,
) -> Record:
    row: Record = {
        "trial_id": trial_id,
        "seed": seed,
        "status": "missing" if trial is None else trial["status"],
        "loss": None,
        "reused": reused,
        "run": None,
        "reason": None if trial is None else trial.get("reason"),
    }
    if trial is None:
        return row
    if trial.get("id") != trial_id or trial.get("seed") != seed:
        row.update(status="invalid", reason="Manifest trial identity or seed mismatch")
        return row
    try:
        authenticate_identity(trial, expected_provenance)
    except (ValueError, KeyError, TypeError) as error:
        row.update(status="invalid", reason=str(error))
        return row
    attempts = trial.get("attempts", [])
    if attempts:
        row["run"] = str(Path(attempts[-1]["directory"]) / "run")
    if trial["status"] != "succeeded":
        row["divergence"] = trial.get("divergence")
        return row
    try:
        from decentralized_llm.tuning.controller import check_result

        if not attempts or attempts[-1]["status"] != "succeeded":
            raise ValueError("Successful trial has no successful final attempt")
        result = check_result(trial, attempts[-1])
        if result != trial.get("result") or result["validation"]["loss"] != trial.get("loss"):
            raise ValueError("Cached result differs from authenticated final result")
        if result["validation"].get("validation_complete") is not True:
            raise ValueError("Final full validation is not marked complete")
        row.update(loss=result["validation"]["loss"], session_seconds=result.get("session_seconds"))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        row.update(status="invalid", reason=str(error))
    return row


def summarize_study(
    study: Record,
    state: Record,
    seeds: list[int],
    expected_provenance: Record | None = None,
) -> Record:
    evidence: Record = state.get("provenance", {}) if expected_provenance is None else expected_provenance
    if state and state.get("provenance") != evidence:
        raise ValueError("Study provenance differs from frozen campaign provenance")
    trials = state.get("trials", {})
    reused = set(study["reused_trial_ids"])
    rows = {}
    references = {}
    for axis in study["slices"]:
        for point in axis["points"]:
            for entry in point["seeds"]:
                identity = entry["trial_id"]
                expected = (point["candidate_id"], entry["seed"])
                if identity in references and references[identity] != expected:
                    raise ValueError("Manifest maps one trial to different candidates or seeds")
                references[identity] = expected
                if identity not in rows:
                    trial = trials.get(identity)
                    rows[identity] = authenticated_trial(
                        trial,
                        identity,
                        entry["seed"],
                        identity in reused,
                        evidence,
                    )
                    if trial is not None and trial.get("candidate_id") != point["candidate_id"]:
                        rows[identity].update(status="invalid", loss=None, reason="Candidate identity mismatch")
    if set(rows) != set(study["required_trial_ids"]):
        raise ValueError("Manifest slices and required trials disagree")
    anchor = study["anchor"]
    anchor_rows = [rows[entry["trial_id"]] for entry in anchor["seeds"]]
    if sorted(entry["seed"] for entry in anchor_rows) != sorted(seeds):
        raise ValueError("Anchor seeds differ from the requested seeds")
    baseline = {entry["seed"]: entry for entry in anchor_rows}
    for row in rows.values():
        center = baseline[row["seed"]]
        row["delta"] = row["loss"] - center["loss"] if row["status"] == center["status"] == "succeeded" else None
    slices = []
    for axis in study["slices"]:
        field = axis["field"]
        section, key = field.split(".")
        center_value = anchor["config"][section][key]
        points = []
        for point in sorted(axis["points"], key=lambda point: point["value"]):
            seed_rows = [rows[entry["trial_id"]] for entry in point["seeds"]]
            points.append(
                {
                    **point,
                    "expression": point.get("expression") or f"{point['value']:.6g}",
                    "is_anchor": point["candidate_id"] == anchor["id"],
                    "seeds": seed_rows,
                    **summarize_point(seed_rows, anchor_rows, seeds),
                }
            )
        indices = [index for index, point in enumerate(points) if point["is_anchor"]]
        if len(indices) != 1 or points[indices[0]]["value"] != center_value:
            raise ValueError(f"Slice does not contain exactly the original center: {field}")
        center_index = indices[0]
        neighbors = []
        for direction, index in (("lower", center_index - 1), ("upper", center_index + 1)):
            if 0 <= index < len(points):
                point = points[index]
                neighbors.append(
                    {
                        "direction": direction,
                        **{key: point[key] for key in ("value", "expression", "status", "mean_delta", "std_delta")},
                    }
                )
        slices.append(
            {
                "field": field,
                "anchor_value": center_value,
                "boundary": "lower" if center_index == 0 else "upper" if center_index == len(points) - 1 else None,
                "points": points,
                "neighbors": neighbors,
                "better_points": [point["value"] for point in points if point["better_than_anchor"]],
            }
        )
    outcomes = Counter(row["status"] for row in rows.values())
    completed = outcomes["succeeded"]
    return {
        "name": study["name"],
        "method": study["method"],
        "workers": study["workers"],
        "directory": study["directory"],
        "source_directory": study["source_directory"],
        "grid_spec": study.get("grid_spec"),
        "provenance": state.get("provenance"),
        "controller_status": state.get("status", "not_started"),
        "status": "complete" if completed == len(rows) else "incomplete",
        "counts": {**study["counts"], "completed": completed, "missing": len(rows) - completed},
        "outcomes": dict(outcomes),
        "anchor": {**anchor, "summary": summarize_point(anchor_rows, anchor_rows, seeds)},
        "slices": sorted(slices, key=lambda axis: FIELDS.index(axis["field"])),
        "trials": list(rows.values()),
    }


def draw_slice(
    axis: Any,
    study: Record,
    field: str,
    *,
    delta: bool,
    individual: bool = True,
    positions: dict[float, int] | None = None,
) -> None:
    """Plot ordered numeric-grid positions, retaining exact power-expression labels."""
    data = next(item for item in study["slices"] if item["field"] == field)
    points = data["points"]
    color = METHOD_COLORS[study["method"]]
    xs = list(range(len(points))) if positions is None else [positions[point["value"]] for point in points]
    mean_key, std_key, seed_key = ("mean_delta", "std_delta", "delta") if delta else ("mean_loss", "std_loss", "loss")
    means = [point[mean_key] if point[mean_key] is not None else float("nan") for point in points]
    errors = [point[std_key] if point[std_key] is not None else 0 for point in points]
    axis.errorbar(
        xs,
        means,
        yerr=errors,
        color=color,
        marker="o",
        capsize=3,
        linewidth=1.5,
        label=METHOD_LABELS[study["method"]],
    )
    if individual:
        for index, marker in enumerate((".", "+", "x")):
            values = [point["seeds"][index].get(seed_key) for point in points]
            axis.scatter(
                [x + (index - 1) * 0.065 for x in xs],
                [value if value is not None else float("nan") for value in values],
                color=color,
                marker=marker,
                alpha=0.5,
                s=22,
                linewidths=0.8,
            )
    for x, point in zip(xs, points, strict=True):
        if point["is_anchor"] and point[mean_key] is not None:
            axis.scatter([x], [point[mean_key]], facecolors="white", edgecolors=color, s=100, zorder=5, marker="D")
        if point["status"] != "complete":
            axis.text(
                x,
                0.02,
                f"{point['status']}\n{point['completed_seeds']}/3",
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="bottom",
                ha="center",
                color=color,
                fontsize=7,
            )
    if positions is None:
        axis.set_xticks(xs, [point["expression"] for point in points], rotation=35, ha="right", fontsize=8)
    axis.grid(alpha=0.2)
    axis.set_xlabel(FIELD_LABELS[FIELDS.index(field)])


def comparison_figure(studies: list[Record], workers: int) -> Any:
    from matplotlib.figure import Figure

    selected = ordered_studies([row for row in studies if row["method"] == "sync" or row["workers"] == workers])
    figure = Figure(figsize=(17, 8), layout="constrained")
    axes = figure.subplots(2, 4)
    for column, field in enumerate(FIELDS):
        labels = {}
        for study in selected:
            data = next(item for item in study["slices"] if item["field"] == field)
            for point in data["points"]:
                labels.setdefault(point["value"], point["expression"])
        values = sorted(labels)
        positions = {value: index for index, value in enumerate(values)}
        for row, delta in enumerate((False, True)):
            axis = axes[row, column]
            for study in selected:
                draw_slice(axis, study, field, delta=delta, individual=False, positions=positions)
            axis.set_xticks(
                list(positions.values()), [labels[value] for value in values], rotation=35, ha="right", fontsize=8
            )
            axis.set_ylabel("Loss − original winner" if delta else "Final full-validation loss")
            if delta:
                axis.axhline(0, color="gray", linewidth=0.7)
            axis.legend(fontsize=8)
    figure.suptitle(f"{workers}-worker comparison · sync reference · original winner held fixed · mean ± sample SD")
    return figure


def write_plots(output: Path, studies: list[Record]) -> list[Record]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    plots = []

    def save(figure: Any, name: str, label: str) -> None:
        for extension in ("png", "svg"):
            figure.savefig(directory / f"{name}.{extension}", dpi=150)
        plots.append({"name": label, "png": f"plots/{name}.png", "svg": f"plots/{name}.svg"})
        figure.clear()

    for study in studies:
        figure = Figure(figsize=(17, 8), layout="constrained")
        axes = figure.subplots(2, 4)
        for column, field in enumerate(FIELDS):
            data = next(axis for axis in study["slices"] if axis["field"] == field)
            for row, delta in enumerate((False, True)):
                axis = axes[row, column]
                draw_slice(axis, study, field, delta=delta)
                axis.set_ylabel("Paired loss difference" if delta else "Final full-validation loss")
                if delta:
                    axis.axhline(0, color="gray", linewidth=0.7)
                if data["boundary"]:
                    axis.set_title(f"Winner at {data['boundary']} grid boundary", fontsize=10)
        figure.suptitle(
            f"{METHOD_LABELS[study['method']]} · {study['workers']} workers · seeds 42–44 · "
            "mean ± sample SD; faint markers: seeds; diamond: original winner"
        )
        save(figure, study["name"], study["name"])
    for workers in sorted({study["workers"] for study in studies if study["method"] != "sync"}):
        save(comparison_figure(studies, workers), f"comparison-{workers}w", f"{workers}-worker comparison")
    return plots


def write_csv(path: Path, rows: list[Record]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output: Path, studies: list[Record]) -> None:
    seed_rows, point_rows = [], []
    for study in studies:
        identities = {}
        for axis in study["slices"]:
            for point in axis["points"]:
                point_rows.append(
                    {
                        "study": study["name"],
                        "method": study["method"],
                        "workers": study["workers"],
                        "field": axis["field"],
                        "value": point["value"],
                        "expression": point["expression"],
                        "candidate_id": point["candidate_id"],
                        "is_anchor": point["is_anchor"],
                        "anchor_boundary": axis["boundary"],
                        **{
                            key: point[key]
                            for key in (
                                "status",
                                "completed_seeds",
                                "required_seeds",
                                "mean_loss",
                                "std_loss",
                                "mean_delta",
                                "std_delta",
                                "min_delta",
                                "max_delta",
                                "better_than_anchor",
                            )
                        },
                    }
                )
                for seed in point["seeds"]:
                    if seed["trial_id"] in identities:
                        identities[seed["trial_id"]]["slices"] += ";" + axis["field"]
                        continue
                    values = {field: study["anchor"]["config"]["optimizer"][field.split(".")[1]] for field in FIELDS}
                    values[axis["field"]] = point["value"]
                    identities[seed["trial_id"]] = {
                        "study": study["name"],
                        "method": study["method"],
                        "workers": study["workers"],
                        "slices": axis["field"],
                        "candidate_id": point["candidate_id"],
                        **values,
                        **{
                            key: seed.get(key)
                            for key in (
                                "trial_id",
                                "seed",
                                "status",
                                "loss",
                                "delta",
                                "reused",
                                "run",
                                "reason",
                                "session_seconds",
                            )
                        },
                    }
        seed_rows.extend(identities.values())
    write_csv(output / "per-seed.csv", seed_rows)
    write_csv(output / "slices.csv", point_rows)


def format_number(value: float | None, *, signed: bool = False) -> str:
    return "—" if value is None else format(value, "+.6f" if signed else ".6f")


def write_document(output: Path, report: Record) -> None:
    counts = report["counts"]
    grid_note = (
        f"Search grids come from the explicit reference specification: {report['grid_spec']}. "
        "Centers and reused results come from the recorded winner studies."
        if report.get("grid_spec")
        else "Search grids come from each winner's original tuning specification."
    )
    lines = [
        "# Winner hyperparameter stability",
        "",
        f"Status: **{report['status']}**. "
        f"{counts['completed']}/{counts['required']} requested seed results authenticated; "
        f"{counts['reused']} reused and {counts['new']} newly planned, {counts['missing']} missing or unsuccessful.",
        "",
        grid_note,
        "",
        "Each slice varies one parameter over the recorded search grid and holds the other parameters at the original "
        "winner. Seeds are 42, 43, and 44. Loss is final full-validation loss; lower is better. Error bars are sample "
        "standard deviations across three seeds, not confidence intervals. Paired differences subtract the original "
        "winner's result for the same seed. No aggregate is reported for an incomplete or divergent point.",
        "",
        "These seeds participated in tuning: this is a descriptive sensitivity analysis, not an independent evaluation "
        "of selection bias. Grid boundaries limit the tested range. Better slice points are listed without moving the "
        "original center; joint parameter interactions and values outside the original grid are untested.",
        "",
        "[Per-seed data](per-seed.csv) · [Slice summaries](slices.csv) · [Machine-readable report](report.json)",
        "",
        "| Method | Workers | Completed | Original winner loss ± SD | Boundary parameters |",
        "|---|---:|---:|---:|---|",
    ]
    for study in report["studies"]:
        summary = study["anchor"]["summary"]
        edges = ", ".join(
            f"{axis['field'].split('.')[1]} ({axis['boundary']})" for axis in study["slices"] if axis["boundary"]
        )
        lines.append(
            f"| {study['method']} | {study['workers']} | "
            f"{study['counts']['completed']}/{study['counts']['required']} | "
            f"{format_number(summary['mean_loss'])} ± {format_number(summary['std_loss'])} | {edges or 'none'} |"
        )
    budget = report["accounting"]
    lines += [
        "",
        "## New-run compute",
        "",
        f"New allocations only: {budget.get('accounted_jobs', 0)}/{budget['total_jobs']} accounted; "
        f"GPU-hours: {format_number(budget.get('gpu_hours'))}; accounting complete: {budget['complete']}. "
        "Imported allocations are excluded. Running or unavailable accounting can change this total.",
        "",
    ]
    for plot in report["plots"]:
        if plot["name"].endswith("comparison"):
            lines += [f"## {plot['name']}", "", f"![{plot['name']}]({plot['png']})", "", f"[SVG]({plot['svg']})", ""]
    for study in report["studies"]:
        plot = next(plot for plot in report["plots"] if plot["name"] == study["name"])
        lines += [
            f"## {study['name']}",
            "",
            f"![Winner slices]({plot['png']})",
            "",
            f"[SVG]({plot['svg']})",
            "",
            "| Slice | Neighbor | Value | Paired loss change ± SD | Status |",
            "|---|---|---:|---:|---|",
        ]
        for axis in study["slices"]:
            for neighbor in axis["neighbors"]:
                lines.append(
                    f"| {axis['field']} | {neighbor['direction']} | {neighbor['expression']} | "
                    f"{format_number(neighbor['mean_delta'], signed=True)} ± "
                    f"{format_number(neighbor['std_delta'])} | {neighbor['status']} |"
                )
        better = [
            f"{axis['field']}={point['expression']} ({format_number(point['mean_delta'], signed=True)})"
            for axis in study["slices"]
            for point in axis["points"]
            if point["better_than_anchor"]
        ]
        lines += [
            "",
            "Points with lower three-seed mean loss than the original winner: "
            + ("; ".join(better) or "none among completed points")
            + ".",
            "",
        ]
        failed = [
            row
            for row in study["trials"]
            if row["status"] not in {"succeeded", "missing", "pending", "submitted", "submitting"}
        ]
        for row in failed:
            lines += [
                f"- Seed {row['seed']}, trial `{row['trial_id']}`: **{row['status']}**; "
                f"{row.get('reason') or 'see report.json for diagnostics'}."
            ]
        lines += [""]
    (output / "report.md").write_text("\n".join(lines))
    cards = "".join(
        f'<section><h2>{html.escape(plot["name"])}</h2><a href="{plot["svg"]}"><img src="{plot["png"]}" '
        f'alt="{html.escape(plot["name"])}"></a></section>'
        for plot in report["plots"]
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Winner hyperparameter stability</title><style>body{font:16px system-ui;"
        "max-width:1600px;margin:2rem auto;"
        "padding:0 1rem;color:#222}img{width:100%}section{margin:3rem 0}pre{white-space:pre-wrap}</style>"
        "<h1>Winner hyperparameter stability</h1>"
        f"<p>{html.escape(grid_note)}</p>"
        f"<p>{html.escape(report['status'])}: {counts['completed']}/{counts['required']} seed results. "
        "Original centers fixed; seeds 42–44; mean ± sample SD. Diamonds mark original winners. "
        "Lower loss is better; paired differences use matching seeds. Incomplete points have no aggregate.</p>"
        "<p>These are descriptive results on seeds used in tuning. Values beyond the original grids and joint "
        'parameter interactions are untested.</p><p><a href="report.md">Full report and neighbor summaries</a> · '
        '<a href="per-seed.csv">Per-seed CSV</a> · <a href="slices.csv">Slice CSV</a> · '
        '<a href="report.json">JSON</a></p>' + cards + "</html>"
    )


def build_report(root: Path) -> Record:
    from scripts.tune_50m import accounting

    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seeds = manifest["seeds"]
    if seeds != [42, 43, 44]:
        raise ValueError("Stability report expects seeds 42, 43, and 44")
    studies, jobs = [], set()
    for study in ordered_studies(manifest["studies"]):
        directory = Path(study["directory"])
        if not directory.is_absolute():
            directory = root / directory
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        studies.append(summarize_study(study, state, seeds, manifest["provenance"]))
        imported = set(study["imported_allocation_ids"])
        for identity, allocation in state.get("allocations", {}).items():
            if identity not in imported and allocation.get("job_id"):
                jobs.add(str(allocation["job_id"]))
    completed = sum(study["counts"]["completed"] for study in studies)
    required = sum(study["counts"]["required"] for study in studies)
    counts = {**manifest["counts"], "required": required, "completed": completed, "missing": required - completed}
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    report = {
        "status": "complete" if completed == required else "incomplete",
        "manifest": str(manifest_path),
        "grid_spec": manifest.get("grid_spec"),
        "seeds": seeds,
        "counts": counts,
        "method_order": list(METHOD_ORDER),
        "method_colors": METHOD_COLORS,
        "studies": studies,
        "accounting": accounting(sorted(jobs)),
        "plots": write_plots(output, studies),
    }
    export_tables(output, studies)
    write_document(output, report)
    from decentralized_llm.runtime import atomic_json

    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    result = build_report(arguments.root)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
