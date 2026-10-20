"""Assess existing full-matrix reports without changing the raw measurements."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from qualification_gates import GATE_PROTOCOL, GATE_REVISION, assess_numerical


def assess_matrix(path: Path) -> dict:
    matrix = json.loads(path.read_text())
    axes = ("model", "scheme", "topology", "feedback", "seed")
    expected = set(
        itertools.product(("20m", "90m"), ("awc", "atc"), ("complete", "one_peer_ring"), (False, True), (42, 43))
    )
    if (
        not matrix.get("complete")
        or len(matrix["rows"]) != len(expected)
        or {tuple(row[key] for key in axes) for row in matrix["rows"]} != expected
    ):
        raise ValueError("assessment requires the complete 32-case trajectory matrix")
    rows = []
    world_size = None
    for row in matrix["rows"]:
        first = Path(row["report"])
        rank_zero = json.loads(first.read_text())
        if world_size is None:
            world_size = rank_zero["world_size"]
        if world_size < 1 or rank_zero["world_size"] != world_size:
            raise ValueError(f"incompatible worker count: {first}")
        config = rank_zero["config"]
        if (
            config["model"]["width"] != {"20m": 320, "90m": 640}[row["model"]]
            or any(config["distributed"][key] != row[key] for key in ("scheme", "topology"))
            or config["distributed"]["error_feedback"] != row["feedback"]
            or config["runtime"]["seed"] != row["seed"]
        ):
            raise ValueError(f"measurement does not match its matrix case: {first}")
        reports = []
        for worker in range(rank_zero["world_size"]):
            source = first.with_name(f"numerical-rank-{worker:05d}.json")
            raw = source.read_bytes()
            report = json.loads(raw)
            if (
                report["rank"] != worker
                or report["world_size"] != rank_zero["world_size"]
                or report["config"] != rank_zero["config"]
                or report["optimizer_revision"] != matrix["optimizer_revision"]
                or not report["optimizer_only"]
                or report["steps"] != 100
                or report["execution"] != report["reference_execution"]
                or report["optimizers"][0]["implementation"] != "reference"
            ):
                raise ValueError(f"incompatible measurement: {source}")
            gates = assess_numerical(
                report["errors"],
                report["validation_delta"],
                steps=report["steps"],
                optimizer_only=True,
                feedback=report["config"]["distributed"]["error_feedback"],
                fused_candidate=report["optimizers"][1]["implementation"] == "triton",
            )
            reports.append(
                {
                    "source": str(source),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "recorded_passed": report["passed"],
                    "gates": gates,
                }
            )
        rows.append(
            {
                **{key: row[key] for key in axes},
                "passed": all(report["gates"]["passed"] for report in reports),
                "all_recorded_updates_passed": all(
                    report["gates"]["all_recorded_updates_passed"] for report in reports
                ),
                "validation_delta": rank_zero["validation_delta"],
                "reports": reports,
            }
        )
    return {
        "source_matrix": str(path),
        "source_matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "optimizer_revision": matrix["optimizer_revision"],
        "world_size": world_size,
        "gate_protocol": GATE_PROTOCOL,
        "gate_revision": GATE_REVISION,
        "raw_measurements_unchanged": True,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "all_recorded_updates_passed": all(row["all_recorded_updates_passed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.matrix.resolve():
        parser.error("output must not overwrite the source matrix")
    result = assess_matrix(args.matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in ("passed", "all_recorded_updates_passed")})
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
