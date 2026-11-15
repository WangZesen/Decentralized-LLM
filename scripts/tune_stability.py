"""Prepare and run fixed-winner, three-seed slices using frozen tuning code."""

import argparse
import copy
import fcntl
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.data.cache import file_digest, fingerprint
from decentralized_llm.runtime import atomic_json, source_digest
from decentralized_llm.tuning.controller import execute, load_state, outstanding_allocations, save_state
from decentralized_llm.tuning.spec import TuningSpec, expand, numeric_recipe, search_points, trial_id

if __package__:
    from scripts.report_style import METHOD_ORDER
    from scripts.tune_50m import seeded_state
    from scripts.tune_data_reuse import collect_trials
else:
    from report_style import METHOD_ORDER

    try:
        # The frozen environment includes its original scripts on PYTHONPATH.
        from scripts.tune_50m import seeded_state
        from scripts.tune_data_reuse import collect_trials
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        from tune_50m import seeded_state
        from tune_data_reuse import collect_trials

FIELDS = ("optimizer.lr", "optimizer.weight_decay", "optimizer.beta1", "optimizer.beta2")
SEEDS = (42, 43, 44)
DEFAULT_SOURCE = Path("runs/r-tuning")
DEFAULT_ROOT = Path("runs/r-stability/20m-512c-128bs-20tpp-warmup4-20261114")
CODE_FILES = ("tune_stability.py", "stability_report.py", "report_style.py")


@contextmanager
def lock_file(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another stability process owns {path.name}") from error
        yield


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def discover_studies(source_root: Path) -> list[Path]:
    result = []
    for path in sorted(source_root.glob("*/state.json")):
        state = read_json(path)
        if state.get("status") == "complete" and state.get("winner") is not None:
            result.append(path.parent.resolve())
    if not result:
        raise ValueError(f"No completed tuning winners found in {source_root}")
    return result


def source_environment(source_root: Path) -> Path:
    environments = {
        str(Path(read_json(path / "state.json")["base"]["launcher"]["environment_script"]).resolve())
        for path in discover_studies(source_root)
    }
    if len(environments) != 1:
        raise ValueError("The selected studies must share one frozen training environment")
    environment = Path(environments.pop())
    if not environment.is_file():
        raise ValueError(f"Historical training environment is unavailable: {environment}")
    return environment


def training_environment(environment: Path, evidence: dict) -> dict:
    snapshot = environment.parent / "source"
    package = snapshot / "src/decentralized_llm"
    lockfile = snapshot / "uv.lock"
    if not package.is_dir() or source_digest(package) != evidence["source_sha256"]:
        raise ValueError("Frozen training source does not match the original tuning provenance")
    if not lockfile.is_file() or file_digest(lockfile) != evidence["lock_sha256"]:
        raise ValueError("Frozen dependency lock does not match the original tuning provenance")
    return {
        "environment_script": str(environment),
        "environment_sha256": file_digest(environment),
        "source_snapshot": str(snapshot),
        "source_sha256": evidence["source_sha256"],
        "lock_sha256": evidence["lock_sha256"],
    }


def winner_grids(state: dict, grid_spec: Path | None = None) -> dict:
    raw = state["spec"] if grid_spec is None else yaml.safe_load(grid_spec.read_text())
    spec = TuningSpec.model_validate(raw)
    base = Config.model_validate(state["base"])
    grids: dict = {}
    for phase in spec.phases:
        for field, search in phase.search.items():
            points = grids.setdefault(field, [])
            for point in search_points(base, field, search):
                if not any(previous["value"] == point["value"] for previous in points):
                    points.append(point)
    if set(grids) != set(FIELDS):
        raise ValueError(f"Expected the four AdamW tuning fields; found {sorted(grids)}")
    for field, points in grids.items():
        points.sort(key=lambda point: point["value"])
        if state["winner"]["values"][field] not in [point["value"] for point in points]:
            raise ValueError(f"Original winner is absent from its {field} grid")
    return grids


def slice_spec(state: dict, base_path: Path, grids: dict) -> dict:
    anchor = state["winner"]["values"]
    phases = []
    for field in FIELDS:
        # Every phase resets all fields; selected parents cannot move the center.
        search = {field: [point["value"] for point in grids[field]]}
        search.update({fixed: [anchor[fixed]] for fixed in FIELDS if fixed != field})
        phases.append({"name": "slice-" + field.split(".")[1], "search": search, "top_k": 1})
    raw = {
        "base": {"configs": [str(base_path)]},
        "seed_start": SEEDS[0],
        "screening_seeds": len(SEEDS),
        "phases": phases,
        "confirmation": {"seeds": len(SEEDS)},
        "controller": copy.deepcopy(state["spec"]["controller"]),
    }
    return TuningSpec.model_validate(raw).model_dump(mode="json")


def plan_study(
    source: Path, root: Path, *, authenticate: bool = True, grid_spec: Path | None = None
) -> tuple[dict, dict]:
    grid_spec = grid_spec.resolve() if grid_spec is not None else None
    state = load_state(source)
    if state["status"] != "complete" or state.get("winner") is None or outstanding_allocations(state):
        raise ValueError(f"A finished study with a winner is required: {source}")
    if any(item["status"] in {"planned", "submitting", "submitted"} for item in state["allocations"].values()):
        raise ValueError(f"Historical study still has active allocations: {source}")
    anchor = copy.deepcopy(state["winner"])
    base = Config.model_validate(anchor["config"])
    if fingerprint(numeric_recipe(base)) != anchor["id"]:
        raise ValueError("Original winner identity does not match its configuration")
    if sorted(row["seed"] for row in anchor["seeds"]) != list(SEEDS):
        raise ValueError("The original winner must have seeds 42, 43, and 44")
    for field in FIELDS:
        section, name = field.split(".")
        if getattr(getattr(base, section), name) != anchor["values"][field]:
            raise ValueError(f"Original winner values/configuration mismatch: {field}")
    winner_file = source / "winner.yaml"
    if winner_file.exists() and numeric_recipe(load_config(winner_file)) != numeric_recipe(base):
        raise ValueError("Original winner.yaml disagrees with the saved winner")
    report_file = source / "report.json"
    if report_file.exists() and read_json(report_file)["winner"]["id"] != anchor["id"]:
        raise ValueError("Original report disagrees with the saved winner")
    trials, allocations = collect_trials([state], state["provenance"], authenticate=authenticate)
    for row in anchor["seeds"]:
        saved = trials.get(row["trial_id"])
        if (
            saved is None
            or saved["status"] != "succeeded"
            or saved["loss"] != row["loss"]
            or saved["candidate_id"] != anchor["id"]
            or saved["seed"] != row["seed"]
        ):
            raise ValueError("Original winner does not have its authenticated seed results")
    directory = root / source.name
    grids = winner_grids(state, grid_spec)
    spec = slice_spec(state, directory / "base.yaml", grids)
    required: set[str] = set()
    reused: set[str] = set()
    pending: set[str] = set()
    slices = []
    allocation_count = 0
    for field, phase in zip(FIELDS, TuningSpec.model_validate(spec).phases, strict=True):
        candidates = {candidate["values"][field]: candidate for candidate in expand(base, phase)}
        points = []
        phase_new: set[str] = set()
        for point in grids[field]:
            candidate = candidates[point["value"]]
            seed_rows = []
            for seed in SEEDS:
                key = trial_id(candidate, seed, state["provenance"])
                seed_rows.append({"seed": seed, "trial_id": key})
                required.add(key)
                if key in trials:
                    if trials[key]["status"] not in {"succeeded", "diverged"}:
                        raise ValueError(f"Reusable historical trial is unfinished: {key}")
                    reused.add(key)
                elif key not in pending:
                    phase_new.add(key)
            points.append(
                {
                    **point,
                    "expression": point.get("expression"),
                    "candidate_id": candidate["id"],
                    "seeds": seed_rows,
                }
            )
        pending.update(phase_new)
        allocation_count += math.ceil(len(phase_new) / spec["controller"]["jobs_per_allocation"])
        slices.append({"field": field, "points": points})
    mode = base.distributed.mode
    topology = base.distributed.topology
    method = (
        "sync"
        if mode == "ddp"
        else {"complete": "cmp", "one_peer_exponential": "exp", "one_peer_ring": "ring"}[topology]
    )
    source_files = {
        str(path): file_digest(path) for path in (source / "state.json", winner_file, report_file) if path.is_file()
    }
    if grid_spec is not None:
        source_files[str(grid_spec)] = file_digest(grid_spec)
    row = {
        "name": source.name,
        "method": method,
        "workers": base.distributed.world_size,
        "source_directory": str(source),
        "grid_spec": {"path": str(grid_spec), "sha256": source_files[str(grid_spec)]}
        if grid_spec is not None
        else None,
        "directory": str(directory),
        "anchor": anchor,
        "slices": slices,
        "required_trial_ids": sorted(required),
        "reused_trial_ids": sorted(reused),
        "new_trial_ids": sorted(pending),
        "imported_trial_ids": sorted(trials),
        "imported_allocation_ids": sorted(allocations),
        "counts": {
            "required": len(required),
            "reused": len(reused),
            "new": len(pending),
            "allocations": allocation_count,
        },
        "source_files": source_files,
        "spec": spec,
        "base": base.model_dump(mode="json"),
    }
    return row, seeded_state(spec, row["base"], state["provenance"], trials, allocations)


def plan_campaign(
    root: Path, source_root: Path, *, authenticate: bool = True, grid_spec: Path | None = None
) -> tuple[dict, dict]:
    root, source_root = root.resolve(), source_root.resolve()
    grid_spec = grid_spec.resolve() if grid_spec is not None else None
    if root == source_root or source_root in root.parents:
        raise ValueError("Use an output directory outside the original tuning tree")
    sources = discover_studies(source_root)
    planned = [plan_study(source, root, authenticate=authenticate, grid_spec=grid_spec) for source in sources]
    planned.sort(key=lambda item: (METHOD_ORDER.index(item[0]["method"]), item[0]["workers"], item[0]["name"]))
    evidence = planned[0][1]["provenance"]
    if any(state["provenance"] != evidence for _, state in planned):
        raise ValueError("Original studies have mixed training provenance")
    environment = source_environment(source_root)
    frozen_training = training_environment(environment, evidence)
    rows = [row for row, _ in planned]
    manifest = {
        "version": 1,
        "kind": "winner-stability-slices",
        "root": str(root),
        "source_root": str(source_root),
        "grid_spec": str(grid_spec) if grid_spec is not None else None,
        "seeds": list(SEEDS),
        "provenance": evidence,
        "training": frozen_training,
        "studies": rows,
        "counts": {
            key: sum(row["counts"][key] for row in rows) for key in ("required", "reused", "new", "allocations")
        },
    }
    return manifest, {row["name"]: state for row, state in planned}


def contract_digest(manifest: dict) -> str:
    return fingerprint({key: value for key, value in manifest.items() if key != "contract_sha256"})


def verify_manifest(manifest: dict) -> None:
    if contract_digest(manifest) != manifest.get("contract_sha256"):
        raise ValueError("Stability campaign manifest changed")
    for filename, expected in manifest["frozen_files"].items():
        if not Path(filename).is_file() or file_digest(Path(filename)) != expected:
            raise ValueError(f"Frozen campaign file changed: {filename}")
    training = manifest["training"]
    if training_environment(Path(training["environment_script"]), manifest["provenance"]) != training:
        raise ValueError("Historical training environment changed")


def prepare(
    root: Path, source_root: Path = DEFAULT_SOURCE, *, dry_run: bool = False, grid_spec: Path | None = None
) -> dict:
    root, source_root = root.resolve(), source_root.resolve()
    grid_spec = grid_spec.resolve() if grid_spec is not None else None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        verify_manifest(manifest)
        if manifest["root"] != str(root) or manifest["source_root"] != str(source_root):
            raise ValueError("Existing campaign uses a different source or destination")
        if grid_spec is not None and manifest.get("grid_spec") != str(grid_spec):
            raise ValueError("Existing campaign uses a different grid specification")
        return {**manifest, "dry_run": True, "remaining_runs": remaining_runs(manifest)} if dry_run else manifest
    manifest, states = (
        plan_campaign(root, source_root) if grid_spec is None else plan_campaign(root, source_root, grid_spec=grid_spec)
    )
    if dry_run:
        return {**manifest, "dry_run": True, "remaining_runs": manifest["counts"]["new"]}
    code_source = Path(__file__).resolve().parent
    for name in CODE_FILES:
        if not (code_source / name).is_file():
            raise ValueError(f"Campaign implementation is incomplete: {code_source / name}")
    root.mkdir(parents=True, exist_ok=True)
    with lock_file(root / ".prepare.lock"):
        if manifest_path.exists():
            existing = read_json(manifest_path)
            verify_manifest(existing)
            if grid_spec is not None and existing.get("grid_spec") != str(grid_spec):
                raise ValueError("Existing campaign uses a different grid specification")
            return existing
        if any(path.name != ".prepare.lock" for path in root.iterdir()):
            raise ValueError("Output directory contains incomplete preparation; use a new empty campaign root")
        code = root / "code"
        code.mkdir()
        frozen = {}
        snapshot = Path(manifest["training"].get("source_snapshot", code_source.parent))
        for name in ("tune_50m.py", "tune_data_reuse.py", "toolchain.sh"):
            path = snapshot / "scripts" / name
            frozen[str(path)] = file_digest(path)
        for name in CODE_FILES:
            destination = code / name
            shutil.copy2(code_source / name, destination)
            frozen[str(destination)] = file_digest(destination)
        wrapper = root / "run.sh"
        grid_argument = f" --grid-spec {shlex.quote(str(grid_spec))}" if grid_spec is not None else ""
        wrapper.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"source {shlex.quote(manifest['training']['environment_script'])}\n"
            f"export DECENTRALIZED_STABILITY_ENV={shlex.quote(manifest['training']['environment_script'])}\n"
            f'exec "$DECENTRALIZED_PYTHON" {shlex.quote(str(code / "tune_stability.py"))}{grid_argument} "$@" '
            f"--root {shlex.quote(str(root))} --source-root {shlex.quote(str(source_root))}\n"
        )
        wrapper.chmod(0o755)
        frozen[str(wrapper)] = file_digest(wrapper)
        for row in manifest["studies"]:
            directory = Path(row["directory"])
            directory.mkdir()
            save_config(Config.model_validate(row["base"]), directory / "base.yaml")
            (directory / "spec.yaml").write_text(yaml.safe_dump(row["spec"], sort_keys=False))
            atomic_json(directory / "anchor.json", row["anchor"])
            save_state(directory, states[row["name"]])
            for name in ("base.yaml", "spec.yaml", "anchor.json"):
                frozen[str(directory / name)] = file_digest(directory / name)
            frozen.update(row["source_files"])
        manifest.update(created=time.time(), frozen_files=frozen)
        manifest["contract_sha256"] = contract_digest(manifest)
        atomic_json(manifest_path, manifest)
        verify_manifest(manifest)
    return manifest


def check_study(manifest: dict, row: dict) -> dict:
    state = load_state(Path(row["directory"]))
    if state["provenance"] != manifest["provenance"] or state["base"] != row["base"] or state["spec"] != row["spec"]:
        raise ValueError(f"Saved study differs from its frozen recipe: {row['name']}")
    imported = set(row["imported_trial_ids"])
    new = set(state["trials"]) - imported
    if not imported <= set(state["trials"]) or not new <= set(row["new_trial_ids"]):
        raise ValueError(f"Unexpected or missing trial identities: {row['name']}")
    for key, trial in state["trials"].items():
        candidate = {"id": fingerprint(numeric_recipe(Config.model_validate(trial["config"])))}
        if (
            trial["id"] != key
            or trial["candidate_id"] != candidate["id"]
            or trial["provenance"] != manifest["provenance"]
            or trial_id(candidate, trial["seed"], manifest["provenance"]) != key
        ):
            raise ValueError(f"Saved trial identity changed: {key}")
    return state


def remaining_runs(manifest: dict) -> int:
    return sum(
        state["trials"].get(key, {}).get("status") not in {"succeeded", "diverged"}
        for row in manifest["studies"]
        for state in [check_study(manifest, row)]
        for key in row["required_trial_ids"]
    )


def run_study(root: Path, name: str) -> dict:
    manifest = read_json(root / "manifest.json")
    verify_manifest(manifest)
    matches = [row for row in manifest["studies"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown stability study: {name}")
    row = matches[0]
    state = check_study(manifest, row)
    # execute performs artifact reauthentication and owns the durable study lock.
    result = execute("resume", Path(row["directory"]), use_local_defaults=False)
    state = check_study(manifest, row)
    if state["status"] != "complete" or outstanding_allocations(state):
        raise RuntimeError(f"Stability study needs attention: {name}: {result.get('reason')}")
    missing = set(row["required_trial_ids"]) - set(state["trials"])
    unfinished = [
        key
        for key in row["required_trial_ids"]
        if key in state["trials"] and state["trials"][key]["status"] not in {"succeeded", "diverged"}
    ]
    if missing or unfinished:
        raise ValueError(f"Completed study has missing slice results: {name}")
    return {"name": name, "status": "complete", "counts": row["counts"]}


def status(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    verify_manifest(manifest)
    rows = []
    for row in manifest["studies"]:
        state = check_study(manifest, row)
        counts = Counter(state["trials"].get(key, {"status": "pending"})["status"] for key in row["required_trial_ids"])
        rows.append(
            {
                "name": row["name"],
                "method": row["method"],
                "workers": row["workers"],
                "status": state["status"],
                "trials": dict(counts),
                "outstanding_allocations": outstanding_allocations(state),
                "reason": state.get("reason"),
                "warning": state.get("warning"),
            }
        )
    return {
        "root": str(root),
        "complete": all(row["status"] == "complete" for row in rows),
        "counts": manifest["counts"],
        "counts_basis": "initial_plan",
        "remaining_runs": sum(
            count
            for row in rows
            for condition, count in row["trials"].items()
            if condition not in {"succeeded", "diverged"}
        ),
        "studies": rows,
    }


def report(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    verify_manifest(manifest)
    if __package__:
        from scripts.stability_report import build_report
    else:
        from stability_report import build_report
    return build_report(root)


def run(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    verify_manifest(manifest)
    with lock_file(root / ".run.lock"):
        children = []
        try:
            for row in manifest["studies"]:
                log = (Path(row["directory"]) / "controller.log").open("a")
                try:
                    process = subprocess.Popen(
                        [str(root / "run.sh"), "worker", "--study", row["name"]],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log.close()
                children.append((row["name"], process))
            # These durable receipts describe controllers, not permission to resubmit jobs.
            atomic_json(
                root / "controllers.json",
                {"started": time.time(), "controllers": [{"name": name, "pid": p.pid} for name, p in children]},
            )
            failures = [{"name": name, "returncode": process.wait()} for name, process in children]
        except BaseException:
            # Stop controllers only. Their submitted Slurm jobs remain resumable.
            for _, process in children:
                if process.poll() is None:
                    process.terminate()
            for _, process in children:
                process.wait()
            raise
        outcome = status(root)
        outcome["controllers"] = failures
        atomic_json(root / "run-result.json", outcome)
        report(root)
        if any(row["returncode"] for row in failures) or not outcome["complete"]:
            raise RuntimeError("Some stability studies need attention; inspect status and controller.log")
        return outcome


def bootstrap(args: argparse.Namespace) -> None:
    manifest = args.root / "manifest.json"
    environment = (
        Path(read_json(manifest)["training"]["environment_script"])
        if manifest.exists()
        else source_environment(args.source_root)
    )
    if os.environ.get("DECENTRALIZED_STABILITY_ENV") == str(environment):
        return
    command = (
        f"source {shlex.quote(str(environment))}\n"
        f"export DECENTRALIZED_STABILITY_ENV={shlex.quote(str(environment))}\n"
        f'exec "$DECENTRALIZED_PYTHON" {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}'
    )
    os.execv("/bin/bash", ["bash", "-euc", command])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "report", "worker"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--grid-spec", type=Path, help="Use search ranges from this tuning YAML while retaining the source winners"
    )
    parser.add_argument("--dry-run", action="store_true", help="Read-only preparation and exact missing-run counts")
    parser.add_argument("--study", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.source_root = args.root.resolve(), args.source_root.resolve()
    args.grid_spec = args.grid_spec.resolve() if args.grid_spec is not None else None
    if args.dry_run and args.action not in {"prepare", "run"}:
        parser.error("--dry-run is supported for prepare and run")
    if args.action == "worker" and not args.study:
        parser.error("worker requires --study")
    manifest_path = args.root / "manifest.json"
    if (
        args.grid_spec is not None
        and manifest_path.exists()
        and read_json(manifest_path).get("grid_spec") != str(args.grid_spec)
    ):
        parser.error("Existing campaign uses a different grid specification")
    bootstrap(args)
    if args.action == "prepare" or args.dry_run:
        result = prepare(args.root, args.source_root, dry_run=args.dry_run, grid_spec=args.grid_spec)
        result = {
            "root": result["root"],
            "dry_run": args.dry_run,
            "counts": result["counts"],
            "counts_basis": "initial_plan",
            "remaining_runs": result.get("remaining_runs", remaining_runs(result) if not args.dry_run else 0),
            "studies": [
                {key: row[key] for key in ("name", "method", "workers", "counts")} for row in result["studies"]
            ],
        }
    elif args.action == "worker":
        result = run_study(args.root, args.study)
    else:
        result = {"run": run, "status": status, "report": report}[args.action](args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
