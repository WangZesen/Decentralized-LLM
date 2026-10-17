"""Read-only completion provider used by the repository's Bash adapter."""

import argparse
import os
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from decentralized_llm.cli import build_parser


def _variants(annotation: Any) -> tuple[Any, ...]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _variants(get_args(annotation)[0])
    if origin in (Union, UnionType):
        return tuple(variant for member in get_args(annotation) for variant in _variants(member))
    return (annotation,)


@cache
def _config_fields() -> dict[str, list[Any]]:
    # Ordinary command/flag completion needs neither Pydantic nor the YAML loader.
    from pydantic import BaseModel

    from decentralized_llm.config import Config

    fields: dict[str, list[Any]] = {}

    def visit(annotation: Any, prefix: str) -> None:
        for variant in _variants(annotation):
            if isinstance(variant, type) and issubclass(variant, BaseModel):
                for name, field in variant.model_fields.items():
                    visit(field.annotation, f"{prefix}.{name}" if prefix else name)
            else:
                fields.setdefault(prefix, []).append(variant)

    visit(Config, "")
    return fields


def config_completions(prefix: str, **_: Any) -> list[str]:
    """Complete real writable fields and bounded values, including every union variant."""
    from argcomplete.completers import FilesCompleter

    fields = _config_fields()
    key, separator, value = prefix.partition("=")
    if not separator:
        return sorted(f"{name}=" for name in fields if name.startswith(key))

    values: set[str] = set()
    for annotation in fields.get(key, ()):
        if get_origin(annotation) is Literal:
            import yaml

            values.update(
                yaml.safe_dump(choice, default_flow_style=True).split("\n...", 1)[0].rstrip("\n")
                for choice in get_args(annotation)
            )
        elif annotation is bool:
            values.update(("true", "false"))
        elif annotation is type(None):
            values.add("null")
        elif isinstance(annotation, type) and issubclass(annotation, Path):
            completer = _directory_completions if key.endswith("_dir") else FilesCompleter()
            values.update(completer(prefix=value))
    return sorted(f"{key}={candidate}" for candidate in values if candidate.startswith(value))


def _directory_completions(prefix: str, **_: Any) -> list[str]:
    from argcomplete.completers import DirectoriesCompleter

    expanded = os.path.expanduser(prefix)
    # Argcomplete's directory completer returns expanded paths; keep the user's
    # spelling so its prefix validator and Bash's tilde handling still agree.
    return [
        prefix + candidate[len(expanded) :] if expanded != prefix and candidate.startswith(expanded) else candidate
        for candidate in DirectoriesCompleter()(prefix=prefix)
    ]


def completion_parser() -> argparse.ArgumentParser:
    """Attach value completers to the same parser used by normal commands."""
    from argcomplete.completers import FilesCompleter

    parser = build_parser()

    def attach(current: argparse.ArgumentParser) -> None:
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    attach(child)
            elif action.dest == "overrides":
                # Argcomplete deliberately extends argparse actions at runtime.
                action.completer = config_completions  # type: ignore[attr-defined]
            elif action.type is Path:
                completer = _directory_completions if action.dest == "directory" else FilesCompleter()
                action.completer = completer  # type: ignore[attr-defined]

    attach(parser)
    return parser


def main() -> None:
    """Run only inside the adapter's checkout and an explicit completion request."""
    root = os.environ.get("DECENTRALIZED_COMPLETION_ROOT")
    if not root or "_ARGCOMPLETE" not in os.environ:
        return
    try:
        checkout = Path(root).resolve(strict=True)
        if not checkout.is_dir() or not Path.cwd().resolve().is_relative_to(checkout):
            return
        if int(os.environ["_ARGCOMPLETE"]) < 1:
            return
        int(os.environ["COMP_POINT"])
        os.environ["COMP_LINE"]
    except OSError, KeyError, ValueError:
        return

    from argcomplete.completers import SuppressCompleter
    from argcomplete.finders import CompletionFinder

    # FilesCompleter runs noninteractive Bash for filename expansion. Startup
    # hooks may activate environments or run arbitrary commands on each Tab.
    os.environ.pop("BASH_ENV", None)
    os.environ.pop("ENV", None)

    class BashCompletionFinder(CompletionFinder):
        def quote_completions(
            self, completions: list[str], cword_prequote: str, last_wordbreak_pos: int | None
        ) -> list[str]:
            # Bash replaces only the quoted value in --option="value and
            # key="value. Argcomplete 3.7.2 otherwise repeats the unquoted prefix.
            if cword_prequote and last_wordbreak_pos is not None:
                completions = [candidate[last_wordbreak_pos + 1 :] for candidate in completions]
                last_wordbreak_pos = None
            return super().quote_completions(completions, cword_prequote, last_wordbreak_pos)

    BashCompletionFinder()(completion_parser(), default_completer=SuppressCompleter())


if __name__ == "__main__":
    main()
