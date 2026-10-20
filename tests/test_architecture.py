"""Keep placement and hardware dependencies out of portable model definitions."""

import ast
import importlib
from pathlib import Path


def test_dependency_boundaries() -> None:
    filename = importlib.import_module("decentralized_llm").__file__
    assert filename is not None
    root = Path(filename).parent
    for folder, forbidden in (
        ("models", ("decentralized_llm.execution", "decentralized_llm.backends", "torch.distributed", "triton")),
        ("execution", ("decentralized_llm.backends", "torch.distributed")),
        ("training", ("torch.distributed",)),
    ):
        for path in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                assert all(not name.startswith(forbidden) for name in names), path
