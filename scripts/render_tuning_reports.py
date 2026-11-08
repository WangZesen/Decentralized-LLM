"""Render presentation revisions from existing audited tables; never launch training.

Usage: .venv-x86_64/bin/python scripts/render_tuning_reports.py
Reproduction: add --suffix communication-order-repeat to use fresh output directories.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import itertools
import json
import re
import shutil
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

if __package__:
    from scripts.report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER
else:
    from report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

Record = dict[str, Any]
PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT / "runs/p-tuning/20m-512c-128bs-20-40tpp-lr-wd-betas-warmup3-20261106"
REPORTS = (
    ("joint", PROJECT / "runs/analysis/joint-tuning-20261105/results", "analyze.py", "plots"),
    ("warmup", PROJECT / "runs/analysis/warmup-tuning-20261106/results-v2", "analyze.py", "plots"),
    ("sensitivity", PROJECT / "runs/analysis/hyperparameter-sensitivity-20261106/results-v2", "analyze.py", "plots"),
    ("batch-budget", CAMPAIGN / "analysis", "campaign_analysis.py", "plot_results"),
    ("correlation-review", CAMPAIGN / "correlation-review", "review_correlations.py", "plots"),
)


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_json(path: Path, data: Record) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_rows(path: Path) -> list[Record]:
    rows: list[Record] = list(csv.DictReader(path.open()))
    for row in rows:
        for field, value in row.items():
            if value == "":
                row[field] = None
            elif value in ("True", "False"):
                row[field] = value == "True"
            else:
                try:
                    row[field] = json.loads(value)
                except ValueError:
                    pass
    return rows


def authenticate(source: Path) -> dict[str, str]:
    manifest = read(source / "manifest.json")
    if isinstance(manifest["outputs"], dict):
        for name, sha in manifest["outputs"].items():
            assert digest(source / name) == sha, name
        for name, sha in manifest["inputs"].items():
            assert digest(name) == sha, name
    else:
        assert manifest["run"]["status"] == "completed"
        for record in manifest["outputs"]:
            assert digest(PROJECT / record["path"]) == record["sha256"], record["path"]
    return {p.name: digest(p) for p in source.iterdir() if p.is_file()}


def symmetric_plot_rows(rows: list[Record]) -> list[Record]:
    """Expose both orientations of symmetric correlations, without changing numeric tables."""
    result = copy.deepcopy(rows)
    for row in rows:
        if row["kind"] not in ("scheme/topology", "topology/scheme"):
            continue
        reverse = {k: v for k, v in row.items() if k not in ("mean_b_minus_a", "median_b_minus_a", "b_better")}
        for field in ("method", "batch", "tpp", "warmup"):
            if field + "_a" in row:
                reverse[field + "_a"], reverse[field + "_b"] = row[field + "_b"], row[field + "_a"]
        result.append(reverse)
    return result


def load_plotter(path: Path, function: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("report_presentation_" + path.parent.name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load report plotter: {path}")
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Change presentation globals after initialization, preserving label-to-method associations.
    module.METHODS = METHOD_ORDER
    if hasattr(module, "LABELS"):
        module.LABELS = {m: module.LABELS[m] for m in METHOD_ORDER}
    module.METHOD_COLORS = METHOD_COLORS
    module.METHOD_LABELS = METHOD_LABELS
    source = textwrap.dedent(inspect.getsource(getattr(module, function)))
    source = source.replace(
        'colors = dict(zip(METHODS, ("#4477AA", "#EE6677", "#228833", "#AA3377"), strict=True))',
        "colors = METHOD_COLORS.copy()",
    )
    source = source.replace(
        '["Ring–exp", "Ring–complete", "Exp–complete"]',
        '[f"{METHOD_LABELS[a]}–{METHOD_LABELS[b]}" for a, b in pairs]',
    )
    # Only the plotting function is recompiled; frozen scientific code and receipts remain intact.
    exec(compile(source, str(path) + ":presentation", "exec"), module.__dict__)
    return getattr(module, function)


def plot_arguments(kind: str, source: Path) -> list[Any]:
    if kind == "joint":
        return [read(source / "analysis.json"), csv_rows(source / "stage1.csv"), csv_rows(source / "beta2.csv")]
    if kind == "warmup":
        return [{name: csv_rows(source / f"{name}.csv") for name in ("winners", "stage1", "curves")}]
    if kind == "sensitivity":
        html = (source / "explorer.html").read_text()
        payload = json.loads(html.split("const data=", 1)[1].split(";\nconst names=", 1)[0])
        tables = {p.stem: csv_rows(p) for p in source.glob("*.csv")}
        tables["correlations"] = symmetric_plot_rows(tables["correlations"])
        return [tables, payload["grids"]]
    if kind == "batch-budget":
        return [
            csv_rows(source / "winners.csv"),
            csv_rows(source / "profiles.csv"),
            symmetric_plot_rows(csv_rows(source / "correlations.csv")),
        ]
    surfaces = {}
    for row in csv_rows(source.parent / "analysis/screening.csv"):
        surfaces.setdefault((row["batch"], row["tpp"], row["method"], row["phase"]), []).append(row)
    return [csv_rows(source / "winners.csv"), surfaces, symmetric_plot_rows(csv_rows(source / "correlations.csv"))]


def method_in_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    value = label.lower().removeprefix("awc ").split(" · ")[0].split(":")[0]
    return {
        "synchronous": "sync",
        "complete": "cmp",
        "exponential": "exp",
        "ring": "ring",
        "sync": "sync",
        "cmp": "cmp",
        "exp": "exp",
    }.get(value)


def check_figure(fig: Figure) -> Record:
    panels, legends, matrix_axes, pair_axes = [], [], [], []
    for ax in fig.axes:
        method = method_in_label(ax.get_title())
        if method:
            panels.append(method)
        legend = ax.get_legend()
        if legend:
            methods = [m for text in legend.get_texts() if (m := method_in_label(text.get_text()))]
            if methods:
                assert methods == sorted(methods, key=METHOD_ORDER.index), methods
                legends.append(methods)
        for ticklabels in (ax.get_xticklabels(), ax.get_yticklabels()):
            methods = [method_in_label(t.get_text()) for t in ticklabels]
            if len(methods) == 4 and all(methods):
                assert tuple(methods) == METHOD_ORDER, methods
                matrix_axes.append(methods)
            labels = [t.get_text() for t in ticklabels]
            if len(labels) == 3 and any("–" in x for x in labels):
                expected = [
                    f"{METHOD_LABELS[a]}–{METHOD_LABELS[b]}" for a, b in itertools.combinations(METHOD_ORDER[1:], 2)
                ]
                assert labels == expected, labels
                pair_axes.append(labels)
        for line in ax.lines:
            method = method_in_label(line.get_label())
            if method:
                from matplotlib.colors import to_rgba

                assert to_rgba(line.get_color()) == to_rgba(METHOD_COLORS[method])
    if panels:
        assert len(panels) % 4 == 0 and panels == list(METHOD_ORDER) * (len(panels) // 4), panels
    assert panels or legends or matrix_axes or pair_axes, "No method-order layout verified"
    return {"panels": panels, "legends": legends, "matrix_axes": matrix_axes, "pair_axes": pair_axes}


def render(kind: str, source: Path, filename: str, function: str, suffix: str) -> Record:
    hashes = authenticate(source)
    output = source.with_name(source.name.split("-v2")[0] + "-" + suffix)
    assert not output.exists(), f"Output already exists: {output}; choose a fresh --suffix"
    output.mkdir()
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, output / ("scientific-evidence-manifest.json" if p.name == "manifest.json" else p.name))
    plotter = load_plotter(source.parent / filename, function)
    layouts = {}
    original_save = Figure.savefig

    def checked_save(fig: Figure, path: str | Path, *args: Any, **kwargs: Any) -> None:
        layouts[Path(path).name] = check_figure(fig)
        return original_save(fig, path, *args, **kwargs)

    Figure.savefig = checked_save  # pyright: ignore[reportAttributeAccessIssue] -- transparent plotting audit wrapper
    try:
        plotter(output, *plot_arguments(kind, source))
    finally:
        Figure.savefig = original_save
    if (output / "explorer.html").exists():
        html = (output / "explorer.html").read_text()
        replacement = (
            '<select id="method">'
            + "".join(
                f'<option value="{m}">{"AWC " if m != "sync" else ""}{METHOD_LABELS[m]}</option>' for m in METHOD_ORDER
            )
            + "</select>"
        )
        html, count = re.subn(r'<select id="method">.*?</select>', lambda _: replacement, html, count=1)
        assert count == 1
        assert re.findall(r'<option value="(sync|cmp|exp|ring)">', replacement) == list(METHOD_ORDER)
        (output / "explorer.html").write_text(html)
    report = (output / "report.md").read_text()
    title, rest = report.split("\n", 1)
    note = (
        "\n\nPlots use synchronous → complete → exponential → ring, from synchronous to sparse communication. "
        "Method colors and all numerical tables are preserved. "
        f"[Original audited report](../{source.name}/report.md). "
        "Regenerate with `python scripts/render_tuning_reports.py --suffix communication-order-repeat`.\n"
    )
    (output / "report.md").write_text(title + note + rest)
    for name, sha in hashes.items():
        assert digest(source / name) == sha, "Original report changed"
        if name.endswith(".csv"):
            assert digest(output / name) == sha, "Numerical table changed"
    record = {
        "kind": "presentation_revision",
        "version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "method_order": METHOD_ORDER,
        "method_colors": METHOD_COLORS,
        "scientific_source": str(source),
        "scientific_manifest_sha256": hashes["manifest.json"],
        "source_files": hashes,
        "renderer": {
            str(Path(__file__)): digest(__file__),
            str(Path(__file__).with_name("report_style.py")): digest(Path(__file__).with_name("report_style.py")),
            str(source.parent / filename): digest(source.parent / filename),
        },
        "verified_layouts": layouts,
        "numeric_tables_byte_identical": True,
        "outputs": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    atomic_json(output / "manifest.json", record)
    return {
        "kind": kind,
        "report": str(output / "report.md"),
        "source": str(source),
        "manifest": str(output / "manifest.json"),
        "plot_files": len(layouts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="communication-order")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9-]+", args.suffix)
    rows = []
    for spec in REPORTS:
        row = render(*spec, args.suffix)
        rows.append(row)
        print(json.dumps(row), flush=True)
    index = PROJECT / "runs/analysis" / f"reports-{args.suffix}.json"
    atomic_json(index, {"method_order": METHOD_ORDER, "reports": rows})
    for row in rows[:3]:
        parent = Path(row["source"]).parent
        path = parent / "index.json"
        existing = read(path) if path.exists() else {}
        if existing and not (parent / "index-before-communication-order.json").exists():
            shutil.copy2(path, parent / "index-before-communication-order.json")
        output = Path(row["report"]).parent
        existing.update(
            latest=output.name,
            report=f"{output.name}/report.md",
            manifest=f"{output.name}/manifest.json",
            run_status="completed",
            method_order=METHOD_ORDER,
            reproduction_command=(
                ".venv-x86_64/bin/python scripts/render_tuning_reports.py " + f"--suffix {args.suffix}-repeat"
            ),
        )
        if (output / "explorer.html").exists():
            existing["interactive_explorer"] = f"{output.name}/explorer.html"
        existing.setdefault("superseded", {})[Path(row["source"]).name] = (
            "Presentation order updated; numerical tables unchanged."
        )
        atomic_json(path, existing)
    atomic_json(CAMPAIGN / "report-index.json", {"method_order": METHOD_ORDER, "reports": rows[3:]})
    print(json.dumps({"reports": len(rows), "plot_files": sum(r["plot_files"] for r in rows), "index": str(index)}))


if __name__ == "__main__":
    main()
