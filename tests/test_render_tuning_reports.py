import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from scripts import render_tuning_reports as renderer
from scripts.report_style import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER


def test_presentation_revision_preserves_scientific_evidence(tmp_path: Path) -> None:
    source = tmp_path / "results"
    source.mkdir()
    (source / "report.md").write_text("# Audited report\n\nOriginal measurements.\n")
    for name in ("winners", "stage1", "curves"):
        (source / f"{name}.csv").write_text("method,loss\nsync,3.4\nring,3.5\n")
    outputs = {p.name: renderer.digest(p) for p in source.iterdir()}
    (source / "manifest.json").write_text(json.dumps({"inputs": {}, "outputs": outputs}))
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    (tmp_path / "analyze.py").write_text(
        "from matplotlib.figure import Figure\n"
        'METHODS = ("sync", "ring", "exp", "cmp")\n'
        "def plots(output, data):\n"
        '    colors = dict(zip(METHODS, ("#4477AA", "#EE6677", "#228833", "#AA3377"), strict=True))\n'
        "    fig = Figure(figsize=(8, 3))\n"
        "    for ax, method in zip(fig.subplots(1, 4), METHODS, strict=True):\n"
        "        ax.plot([0, 1], [3, 2], color=colors[method], label=method)\n"
        "        ax.set_title(method)\n"
        "        ax.legend()\n"
        '    fig.savefig(output / "comparison.png")\n'
    )
    result = renderer.render("warmup", source, "analyze.py", "plots", "presentation")
    output = Path(result["report"]).parent
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["verified_layouts"]["comparison.png"]["panels"] == list(METHOD_ORDER)
    assert manifest["method_colors"] == METHOD_COLORS
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert all((output / name).read_bytes() == raw for name, raw in original.items() if name.endswith(".csv"))
    assert (output / "scientific-evidence-manifest.json").read_bytes() == original["manifest.json"]
    with pytest.raises(AssertionError, match="already exists"):
        renderer.render("warmup", source, "analyze.py", "plots", "presentation")


@pytest.mark.parametrize("wrong_order", [False, True])
def test_layout_audit_rejects_changed_colors_or_method_order(wrong_order: bool) -> None:
    figure = Figure()
    axis = figure.subplots()
    methods = tuple(reversed(METHOD_ORDER)) if wrong_order else METHOD_ORDER
    for method in methods:
        axis.plot([0, 1], [0, 1], label=METHOD_LABELS[method], color=METHOD_COLORS[method] if wrong_order else "black")
    axis.legend()
    with pytest.raises(AssertionError):
        renderer.check_figure(figure)
