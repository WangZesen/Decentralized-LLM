import json
import sys
from pathlib import Path
from typing import Any

import pytest

from decentralized_llm import cli
from decentralized_llm.tuning import controller


@pytest.mark.parametrize("output", [None, Path("custom/study")])
def test_run_output_directory(
    output: Path | None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[tuple, dict]] = []

    def invoked(*args: Any, **kwargs: Any) -> dict:
        calls.append((args, kwargs))
        return {"status": "planned"}

    monkeypatch.setattr(controller, "execute", invoked)
    spec = Path("configs/tunings/20m-sync-1w-warmup4.yaml")
    arguments = ["decentralized-llm", "tune", "run", "--spec", str(spec), "--dry-run"]
    if output is not None:
        arguments.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", arguments)

    cli.main()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("run", output or Path("runs/r-tuning/20m-sync-1w-warmup4"), spec)
    assert kwargs["dry_run"] is True
    assert json.loads(capsys.readouterr().out) == {"status": "planned"}
