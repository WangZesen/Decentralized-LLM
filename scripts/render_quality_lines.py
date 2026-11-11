"""Render an additive presentation revision of audited quality results across TPP."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import yaml

from decentralized_llm.data.cache import fingerprint

if __package__:
    from scripts.report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER
else:
    from report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "runs/quality"
SOURCES = (
    BASE / "20m-512c-128bs-20-40tpp-winners-20261106",
    BASE / "20m-512c-128bs-60-120tpp-winners-20261106",
)
BUDGETS = (20, 40, 60, 80, 100, 120)
METRICS = (
    ("validation_loss", "C4 validation loss", "Loss (nats; lower is better)", 1),
    ("blimp_acc", "BLiMP accuracy", "Accuracy (%)", 100),
    ("piqa_acc", "PIQA accuracy", "Accuracy (%)", 100),
    ("piqa_acc_norm", "PIQA normalized accuracy", "Accuracy (%)", 100),
    ("hellaswag_acc", "HellaSwag accuracy", "Accuracy (%)", 100),
    ("hellaswag_acc_norm", "HellaSwag normalized accuracy", "Accuracy (%)", 100),
)
MARKERS = {"sync": "o", "cmp": "s", "exp": "^", "ring": "D"}
HYPERPARAMETERS = (
    ("weight_decay", "Selected weight decay", "Weight decay"),
    ("lr", "Selected learning rate", "Learning rate (configured peak)"),
    ("beta1", "Selected β₁", "β₁"),
    ("beta2", "Selected β₂", "β₂ (logit scale)"),
)
LINESTYLES = {"sync": "-", "cmp": "--", "exp": ":", "ring": "-."}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_sources(
    sources: tuple[Path, ...], budgets: tuple[int, ...] = BUDGETS
) -> tuple[list[dict], list[dict], list[dict]]:
    provenance, groups, runs = [], [], []
    for root in sources:
        name = (
            "final-report-manifest.json" if (root / "final-report-manifest.json").exists() else "report-manifest.json"
        )
        manifest = json.loads((root / name).read_text())
        for filename, expected in manifest["files"].items():
            if digest(root / filename) != expected:
                raise ValueError(f"audited report artifact changed: {root / filename}")
        report = json.loads((root / "report.json").read_text())
        provenance.append(
            {
                "root": str(root),
                "manifest": name,
                "manifest_sha256": digest(root / name),
                "report_sha256": digest(root / "report.json"),
                "files": manifest["files"],
                "snapshot_identity": report["snapshot_identity"],
                "suite_identity": report["suite_identity"],
                "training_snapshots": report.get("source_snapshots", {str(root): report["snapshot_identity"]}),
            }
        )
        groups.extend(g for g in report["groups"] if g["budget"] in budgets)
        runs.extend(r for r in report["runs"] if r["budget"] in budgets)
    if len({s["suite_identity"] for s in provenance}) != 1:
        raise ValueError("source reports use different benchmark suites")
    expected = {(budget, method) for budget in budgets for method in METHOD_ORDER}
    if len(groups) != len(expected) or {(g["budget"], g["method"]) for g in groups} != expected:
        raise ValueError("expected exactly one group per TPP and method")
    expected_runs = len(expected) * 3
    if len(runs) != expected_runs or len({r["key"] for r in runs}) != expected_runs:
        raise ValueError(f"expected {expected_runs} distinct seed results")
    for group in groups:
        selected = [r for r in runs if (r["budget"], r["method"]) == (group["budget"], group["method"])]
        if group["completed_seeds"] != 3 or sorted(r["seed"] for r in selected) != [42, 43, 44]:
            raise ValueError("all plotted groups require seeds 42, 43, and 44")
        if any(r["status"] != "complete" for r in selected):
            raise ValueError("cannot plot incomplete evaluations as final results")
        for metric, *_ in METRICS:
            values = [r[metric] for r in selected]
            saved = group["metrics"][metric]
            if not math.isclose(saved["mean"], statistics.mean(values), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"mean mismatch: {metric}")
            if not math.isclose(saved["sample_std"], statistics.stdev(values), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"sample standard deviation mismatch: {metric}")
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    groups.sort(key=lambda g: (g["budget"], order[g["method"]]))
    runs.sort(key=lambda r: (r["budget"], order[r["method"]], r["seed"]))
    return provenance, groups, runs


def load_hyperparameters(provenance: list[dict], budgets: tuple[int, ...]) -> tuple[list[dict], dict[str, str]]:
    """Cross-check each selected optimizer against its report and all three frozen retraining configs."""
    snapshots = {root: sha for source in provenance for root, sha in source["training_snapshots"].items()}
    groups: dict[tuple[int, str], dict] = {}
    files = {}
    for root_name, snapshot in snapshots.items():
        root = Path(root_name)
        campaign_file = root / "campaign.json"
        campaign = json.loads(campaign_file.read_text())
        if campaign["snapshot_identity"] != snapshot or fingerprint(campaign["files"]) != snapshot:
            raise ValueError(f"training snapshot does not match audited results: {root}")
        files[str(campaign_file)] = digest(campaign_file)
        for row in campaign["runs"]:
            if row["budget"] not in budgets:
                continue
            config_file, winner_file = Path(row["config"]), Path(row["winner"])
            config_sha = campaign["files"][str(config_file.relative_to(root))]
            if digest(config_file) != config_sha or digest(winner_file) != row["winner_sha256"]:
                raise ValueError(f"selected or frozen configuration changed: {row['key']}")
            config = yaml.safe_load(config_file.read_text())
            winner = yaml.safe_load(winner_file.read_text())
            selection_file = winner_file.parent / "report.json"
            selection = json.loads(selection_file.read_text())["winner"]
            if selection["id"] != row["winner_id"] or config["optimizer"] != winner["optimizer"]:
                raise ValueError(f"selected optimizer differs from retraining: {row['key']}")
            if config["runtime"]["seed"] != row["seed"] or config["training"]["tokens_per_parameters"] != row["budget"]:
                raise ValueError(f"seed or budget mismatch: {row['key']}")
            values = {key: winner["optimizer"][key] for key, *_ in HYPERPARAMETERS}
            for key, value in values.items():
                if selection["values"][f"optimizer.{key}"] != value or not math.isfinite(value):
                    raise ValueError(f"selected value disagrees with tuning report: {row['key']}: {key}")
            key = (row["budget"], row["method"])
            if key not in groups:
                groups[key] = {
                    "budget": row["budget"],
                    "method": row["method"],
                    **values,
                    "winner": str(winner_file),
                    "winner_sha256": row["winner_sha256"],
                    "selection_report": str(selection_file),
                    "campaign": str(campaign_file),
                    "seeds": [],
                }
            group = groups[key]
            if any(group[name] != value for name, value in values.items()) or group["winner"] != str(winner_file):
                raise ValueError(f"seeds use different selected configurations: {key}")
            group["seeds"].append({"seed": row["seed"], "config": str(config_file), "sha256": config_sha})
            files.update(
                {
                    str(config_file): config_sha,
                    str(winner_file): row["winner_sha256"],
                    str(selection_file): digest(selection_file),
                }
            )
    expected = {(budget, method) for budget in budgets for method in METHOD_ORDER}
    if set(groups) != expected or any(sorted(s["seed"] for s in g["seeds"]) != [42, 43, 44] for g in groups.values()):
        raise ValueError("selected configurations must cover every plotted group and all three seeds")
    return [groups[budget, method] for budget in budgets for method in METHOD_ORDER], files


def hyperparameter_panel(ax: Axes, rows: list[dict], parameter: tuple) -> None:
    key, title, ylabel = parameter
    for index, method in enumerate(METHOD_ORDER):
        selected = [row for row in rows if row["method"] == method]
        ax.plot(
            [row["budget"] for row in selected],
            [row[key] for row in selected],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=11 - index,
            markerfacecolor="none",
            markeredgewidth=1.5,
            linewidth=1.8,
        )
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Training tokens per parameter (TPP)")
    ax.set_ylabel(ylabel)
    budgets = sorted({row["budget"] for row in rows})
    ax.set_xticks(budgets)
    margin = max(4, (budgets[-1] - budgets[0]) * 0.04)
    ax.set_xlim(budgets[0] - margin, budgets[-1] + margin)
    if key == "beta2":
        ax.set_yscale("logit")
    values = sorted({row[key] for row in rows})
    ax.set_yticks(values, labels=[f"{value:.7g}" for value in values])
    ax.minorticks_off()
    ax.margins(y=0.2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def render_hyperparameters(output: Path, rows: list[dict], files: dict[str, str]) -> None:
    archive = output / "hyperparameter-sources"
    archive.mkdir()
    for name in files:
        target = archive / Path(name).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    write_json(output / "selected-hyperparameters.json", {"groups": rows, "source_files": files})
    with (output / "selected-hyperparameters.csv").open("w") as handle:
        fields = ["budget", "method", *[key for key, *_ in HYPERPARAMETERS]]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, parameter in zip(axes.flat, HYPERPARAMETERS, strict=True):
            hyperparameter_panel(ax, rows, parameter)
        fig.suptitle("Selected optimizer settings across training budgets", fontsize=17, fontweight="bold", y=0.98)
        fig.text(0.5, 0.935, "Historical tuning winners · same settings for seeds 42–44", ha="center")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.115, top=0.81, hspace=0.5, wspace=0.32)
        fig.text(
            0.5,
            0.025,
            "Overlapping curves share selected values. β₂ uses logit spacing to resolve values near 1.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"selected-hyperparameters.{extension}", dpi=200)
        plt.close(fig)
        for parameter in HYPERPARAMETERS:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            hyperparameter_panel(ax, rows, parameter)
            ax.legend(ncol=4, frameon=False, fontsize=8, loc="best")
            fig.text(
                0.5,
                0.015,
                "Selected tuning values · same for seeds 42–44 · overlapping curves share values",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            for extension in ("png", "svg"):
                fig.savefig(output / f"selected-{parameter[0]}.{extension}", dpi=200)
            plt.close(fig)


def panel(ax: Axes, groups: list[dict], metric: tuple) -> None:
    key, title, ylabel, scale = metric
    for method in METHOD_ORDER:
        selected = [g for g in groups if g["method"] == method]
        ax.errorbar(
            [g["budget"] for g in selected],
            [g["metrics"][key]["mean"] * scale for g in selected],
            yerr=[g["metrics"][key]["sample_std"] * scale for g in selected],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=MARKERS[method],
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=1.8,
            elinewidth=1,
            capsize=3,
            capthick=1,
        )
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Training tokens per parameter (TPP)")
    ax.set_ylabel(ylabel)
    budgets = sorted({g["budget"] for g in groups})
    ax.set_xticks(budgets)
    margin = max(4, (budgets[-1] - budgets[0]) * 0.04)
    ax.set_xlim(budgets[0] - margin, budgets[-1] + margin)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def render(destination: Path, revision: str, budgets: tuple[int, ...] = BUDGETS) -> Path:
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be distinct and increasing")
    provenance, groups, runs = load_sources(SOURCES, budgets)
    hyperparameters, config_files = load_hyperparameters(provenance, budgets)
    output = destination / "presentation-revisions" / revision
    output.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(provenance):
        target = output / "sources" / str(index)
        target.mkdir(parents=True)
        original = Path(source["root"])
        for name in ("report.json", source["manifest"]):
            shutil.copy2(original / name, target / name)
    shutil.copy2(Path(__file__), output / "render_quality_lines.py")
    shutil.copy2(PROJECT / "scripts/report_style.py", output / "report_style.py")
    render_hyperparameters(output, hyperparameters, config_files)
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        for ax, metric in zip(axes.flat, METRICS, strict=True):
            panel(ax, groups, metric)
        fig.suptitle("Model quality across training budgets", fontsize=17, fontweight="bold", y=0.987)
        fig.text(0.5, 0.952, "20M parameters · 512-token context · mean ± 1 sample SD over seeds 42–44", ha="center")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.932), ncol=4, frameon=False)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.86, hspace=0.46, wspace=0.25)
        fig.text(
            0.5,
            0.018,
            "Normalized accuracy uses candidate character length. "
            "Error bars show seed variability, not confidence intervals.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"quality-lines.{extension}", dpi=200)
        plt.close(fig)
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            panel(ax, groups, metric)
            ax.legend(ncol=4, frameon=False, fontsize=8)
            fig.text(0.5, 0.012, "Mean ± 1 sample SD · seeds 42–44", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            for extension in ("png", "svg"):
                fig.savefig(output / f"{metric}.{extension}", dpi=200)
            plt.close(fig)
    with (output / "plot-data.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tpp", "method", "metric", "unit", "seeds", "mean", "sample_std"])
        for group in groups:
            for key, _, _, scale in METRICS:
                value = group["metrics"][key]
                writer.writerow(
                    [
                        group["budget"],
                        group["method"],
                        key,
                        "nats" if scale == 1 else "%",
                        3,
                        value["mean"] * scale,
                        value["sample_std"] * scale,
                    ]
                )
    with (output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    write_json(
        output / "report.json",
        {
            "complete": True,
            "budgets": budgets,
            "runs": runs,
            "groups": groups,
            "sources": provenance,
            "selected_hyperparameters": hyperparameters,
        },
    )
    text = [
        f"# Model quality across TPP {budgets[0]}–{budgets[-1]}",
        "",
        "Presentation revision of existing audited results: "
        f"{len(runs)} models, four methods, {len(budgets)} token budgets, and seeds 42–44.",
        "Points show three-seed means; error bars show ±1 sample standard deviation. Lines connect evaluated budgets.",
        "Accuracy is shown as a percentage. Normalized PIQA and HellaSwag accuracy use candidate character length.",
        "",
        "![All quality metrics across TPP](quality-lines.png)",
        "",
        "Download: [PNG](quality-lines.png) · [SVG](quality-lines.svg) · [PDF](quality-lines.pdf).",
        "Data: [plotted means and standard deviations](plot-data.csv) · [per-seed results](results.csv).",
        "",
        "## Individual metrics",
        "",
    ]
    for key, title, *_ in METRICS:
        text.append(f"- {title}: [PNG]({key}.png) · [SVG]({key}.svg)")
    text += [
        "",
        "## Selected optimizer settings",
        "",
        "These are the historical tuning winners used for retraining at each TPP. "
        "All three seeds share the selected values, so these plots have no seed error bars.",
        "LR is the configured peak learning rate. β₂ uses a logit axis with ticks labeled in β₂ units "
        "to distinguish values close to 1. Overlapping curves indicate equal selections.",
        "",
        "![Selected weight decay, learning rate, beta1 and beta2](selected-hyperparameters.png)",
        "",
        "Download: [PNG](selected-hyperparameters.png) · [SVG](selected-hyperparameters.svg) · "
        "[PDF](selected-hyperparameters.pdf). Exact values: [CSV](selected-hyperparameters.csv) · "
        "[JSON and source checksums](selected-hyperparameters.json).",
        "",
    ]
    for key, title, *_ in HYPERPARAMETERS:
        text.append(f"- {title}: [PNG](selected-{key}.png) · [SVG](selected-{key}.svg)")
    text += ["", "| TPP | Method | Weight decay | LR | β₁ | β₂ |", "|---|---|---|---|---|---|"]
    for row in hyperparameters:
        values = [f"{row[key]:.10g}" for key, *_ in HYPERPARAMETERS]
        text.append(f"| {row['budget']} | {METHOD_LABELS[row['method']]} | " + " | ".join(values) + " |")
    text += [
        "",
        "## Results by training budget",
        "",
        "Three-seed mean ± sample standard deviation; accuracy metrics are percentages.",
        "",
        "| TPP | Method | C4 loss | BLiMP | PIQA | PIQA normalized | HellaSwag | HellaSwag normalized |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group in groups:
        cells = []
        for key, _, _, scale in METRICS:
            value = group["metrics"][key]
            digits = 4 if scale == 1 else 2
            cells.append(f"{value['mean'] * scale:.{digits}f} ± {value['sample_std'] * scale:.{digits}f}")
        text.append(f"| {group['budget']} | {METHOD_LABELS[group['method']]} | " + " | ".join(cells) + " |")
    text += [
        "",
        "## Provenance",
        "",
        "Original scientific results and audited reports are preserved. "
        "Both sources use the same frozen benchmark suite.",
        "",
    ]
    for source in provenance:
        text.append(f"- [Original report]({source['root']}/report.md) · report SHA-256 `{source['report_sha256']}`.")
    flags = [r for r in runs if r.get("review_validation")]
    for row in flags:
        text += [
            "",
            f"Existing validation review retained: **{row['key']}**, "
            f"C4 loss difference **{row['validation_delta']:+.6f} nats** from its historical seed. "
            "The recipe and execution checks passed; the cause is undetermined.",
        ]
    (output / "report.md").write_text("\n".join(text) + "\n")
    for source in provenance:
        original = Path(source["root"])
        for name, expected in source["files"].items():
            if digest(original / name) != expected:
                raise ValueError(f"source artifact changed during rendering: {original / name}")
    for name, expected in config_files.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"configuration source changed during rendering: {name}")
    files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(
        output / "manifest.json",
        {
            "kind": "presentation-revision",
            "created_at": datetime.now(UTC).isoformat(),
            "sources": provenance,
            "budgets": budgets,
            "method_order": METHOD_ORDER,
            "method_colors": METHOD_COLORS,
            "metrics": [m[0] for m in METRICS],
            "models": len(runs),
            "selected_hyperparameters": [key for key, *_ in HYPERPARAMETERS],
            "hyperparameter_source_files": config_files,
            "beta2_axis": "logit; ticks show original beta2 values",
            "error_bars": "sample standard deviation across three seeds",
            "source_artifacts_unchanged": True,
            "files": files,
        },
    )
    published = {}
    for path in output.iterdir():
        if (
            path.is_file()
            and path.name != "manifest.json"
            and path.suffix in (".md", ".json", ".csv", ".png", ".svg", ".pdf")
        ):
            shutil.copy2(path, destination / path.name)
            published[path.name] = digest(destination / path.name)
    write_json(
        destination / "manifest.json",
        {
            "kind": "presentation-revision",
            "directory": str(output),
            "presentation_manifest_sha256": digest(output / "manifest.json"),
            "files": published,
        },
    )
    write_json(
        destination / "latest-presentation.json",
        {"directory": str(output), "manifest_sha256": digest(output / "manifest.json")},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS, help="Increasing TPP budgets to include")
    parser.add_argument("--revision", default="lines-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    budgets = tuple(args.budgets)
    destination = args.output or BASE / f"20m-512c-128bs-{budgets[0]}-{budgets[-1]}tpp-winners-20261106"
    print(render(destination.resolve(), args.revision, budgets), flush=True)


if __name__ == "__main__":
    main()
