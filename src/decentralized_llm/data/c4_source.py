"""Pinned C4 shard inventory and lazy, verified reads from shared storage."""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from decentralized_llm.data.cache import file_digest, fingerprint


def read_inventory(directory: Path, dataset: str, revision: str) -> dict[str, Any]:
    manifest = json.loads((directory / "source.json").read_text())
    content = {key: value for key, value in manifest.items() if key != "identity"}
    if manifest.get("version") != 1 or fingerprint(content) != manifest.get("identity"):
        raise ValueError("invalid C4 source inventory")
    if manifest["dataset"] != dataset or revision not in ("main", manifest["revision"]):
        raise ValueError("C4 source dataset/revision mismatch")
    paths = [entry["path"] for entry in manifest["files"]]
    if paths != sorted(set(paths)):
        raise ValueError("C4 source paths must be sorted and unique")
    for path in paths:
        if Path(path).parts[:1] != ("en",) or len(Path(path).parts) != 2 or not path.endswith(".json.gz"):
            raise ValueError("invalid C4 source path")
    return manifest


def _documents(entries: list[dict[str, Any]], directory: str) -> Iterator[dict[str, str]]:
    # Open only the next shard: an inventory can describe unstaged shards beyond
    # a bounded token budget without changing the global shard permutation.
    for entry in entries:
        path = Path(directory) / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"stage the next C4 source shard before continuing: {path}")
        if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
            raise ValueError(f"C4 source checksum mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line)["text"]
                if not isinstance(text, str):
                    raise ValueError(f"non-string C4 text in {path}")
                yield {"text": text}


def local_stream(directory: Path, split: str, inventory: dict[str, Any]):
    from datasets import IterableDataset

    entries = [entry for entry in inventory["files"] if entry["path"].startswith(f"en/c4-{split}.")]
    if not entries:
        raise ValueError(f"no English C4 {split} shards in source inventory")
    return IterableDataset.from_generator(_documents, gen_kwargs={"entries": entries, "directory": str(directory)})
