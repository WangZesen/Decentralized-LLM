"""Prepare, qualify, submit, and report a paired TF32/FP32 campaign."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml

from decentralized_llm.config import Config, SlurmLauncher, load_config, save_config
from decentralized_llm.data.cache import TokenCache, file_digest, training_boundaries
from decentralized_llm.launch.commands import launch
from decentralized_llm.launch.slurm import SubmissionRejected, scheduler_arguments, submit_batch
from decentralized_llm.runtime import atomic_json, source_digest
from decentralized_llm.training.logs import read_failures, read_metadata, read_result, read_validation_metrics

DEFAULT_NAME = "20m-awc-ef-8w-ring-512c-128bs-20tpp"
SEEDS = (42, 43, 44)
MODES = {"tf32": "tf32", "fp32": "ieee"}
Record = dict[str, Any]


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def derive_configs(winner: Path, root: Path) -> dict[str, Config]:
    original = yaml.safe_load(winner.read_text())
    if "error_feedback" in original["distributed"]:
        original["distributed"]["error_feedback_gamma"] = float(original["distributed"].pop("error_feedback"))
    base = Config.model_validate(original)
    configs = {}
    for mode, precision in MODES.items():
        for seed in SEEDS:
            key = f"{mode}/seed-{seed}"
            raw = base.model_dump(mode="json")
            raw["runtime"].update(
                seed=seed,
                amp=False,
                execution_backend="native",
                compile=False,
                fp32_precision=precision,
                sdpa_backend="math",
                fused_rope=False,
                output_dir=str(root / key),
            )
            raw["evaluation"]["fp32_precision"] = "ieee"
            raw["launcher"].update(
                time="03:00:00",
                gpu_type="nvidia_gh200_120gb",
                environment_script=str(root / "source/environment.sh"),
            )
            configs[key] = Config.model_validate(raw)
    return configs


def paired_inputs(configs: dict[str, Config]) -> list[Record]:
    from decentralized_llm.data.sampling import GlobalSampler
    from decentralized_llm.models.initialization import initialize_model

    evidence = []
    for seed in SEEDS:
        pair = [configs[f"{mode}/seed-{seed}"] for mode in MODES]
        hashes = []
        samples = []
        for config in pair:
            model = initialize_model(config)
            digest = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.numpy().tobytes())
            hashes.append(digest.hexdigest())
            sampler = GlobalSampler.from_config(TokenCache(config.data.cache_dir), config)
            samples.append([int(sampler.block(i)) for i in range(32)])
            sampler.close()
        assert hashes[0] == hashes[1] and samples[0] == samples[1]
        assert pair[0].optimizer == pair[1].optimizer and pair[0].training == pair[1].training
        evidence.append({"seed": seed, "initial_weights_sha256": hashes[0], "first_blocks": samples[0]})
    return evidence


def verify_snapshot(manifest: Record) -> None:
    source = Path(manifest["source"])
    for name, expected in manifest["source_files"].items():
        if file_digest(source / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    for row in manifest["runs"]:
        if file_digest(Path(row["config"])) != row["config_sha256"]:
            raise ValueError(f"frozen configuration changed: {row['key']}")


def prepare(root: Path, winner: Path) -> Record:
    manifest_path = root / "campaign.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_snapshot(manifest)
        return manifest
    configs = derive_configs(winner, root)
    project = Path(__file__).resolve().parents[1]
    source = root / "source"
    source.mkdir()
    shutil.copytree(project / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (source / "scripts").mkdir()
    shutil.copy2(__file__, source / "scripts/precision_campaign.py")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(project / name, source / name)
    shutil.copy2(winner, root / "winner-original.yaml")
    # Activate existing architecture-specific dependencies without importing live project code.
    environment = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export DECENTRALIZED_PYTHON={shlex.quote(str(project))}/.venv-$(uname -m)/bin/python\n"
        f"export PYTHONPATH={shlex.quote(str(source / 'src'))}\n"
        "export PYTHONNOUSERSITE=1\n"
        "\"$DECENTRALIZED_PYTHON\" - <<'PY'\n"
        "import importlib.metadata, sys, tomllib\n"
        "from pathlib import Path\n"
        "assert sys.version_info[:2] == (3, 14)\n"
        f"project = tomllib.loads(Path({str(source / 'pyproject.toml')!r}).read_text())\n"
        "for requirement in project['project']['dependencies']:\n"
        "    name, expected = requirement.split('==')\n"
        "    assert importlib.metadata.version(name).split('+')[0] == expected\n"
        "PY\n"
    )
    (source / "environment.sh").write_text(environment)
    manifest: Record = {
        "version": 1,
        "winner": str(winner),
        "winner_sha256": file_digest(winner),
        "source": str(source),
        "source_sha256": source_digest(source / "src/decentralized_llm"),
        "source_files": {str(p.relative_to(source)): file_digest(p) for p in sorted(source.rglob("*")) if p.is_file()},
        "paired_inputs": paired_inputs(configs),
        "runs": [],
    }
    TokenCache(next(iter(configs.values())).data.cache_dir).verify()
    for key, config in configs.items():
        folder = config.runtime.output_dir
        folder.mkdir(parents=True)
        filename = folder / "resolved.yaml"
        save_config(config, filename)
        with working_directory(source):
            dry = launch(config, dry_run=True, verify=False, job_name=f"precision-{key.replace('/', '-')}")
        atomic_json(folder / "dry-run.json", dry)
        manifest["runs"].append(
            {
                "key": key,
                "mode": key.split("/")[0],
                "seed": config.runtime.seed,
                "run": str(folder),
                "config": str(filename),
                "config_sha256": file_digest(filename),
                "status": "prepared",
            }
        )
    atomic_json(manifest_path, manifest)
    return manifest


def smoke_submit(root: Path, manifest: Record) -> Record:
    if "smoke" in manifest and manifest["smoke"]["status"] != "rejected":
        return manifest["smoke"]
    verify_snapshot(manifest)
    config = load_config(manifest["runs"][0]["config"])
    assert isinstance(config.launcher, SlurmLauncher)
    config.launcher.time = "00:20:00"
    folder = root / "smoke"
    folder.mkdir(exist_ok=True)
    source = Path(manifest["source"])
    script = folder / "submit.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(source))}\nsource {shlex.quote(str(source / 'environment.sh'))}\n"
        "exec srun --ntasks=1 --gpus-per-task=nvidia_gh200_120gb:1 --cpu-bind=none "
        '"$DECENTRALIZED_PYTHON" scripts/precision_campaign.py smoke --root '
        f"{shlex.quote(str(root))}\n"
    )
    manifest["smoke"] = {"status": "submitting"}
    atomic_json(root / "campaign.json", manifest)
    try:
        receipt = submit_batch(
            scheduler_arguments(config) + ["--job-name=precision-smoke", f"--output={folder}/slurm-%j.out"], script
        )
    except SubmissionRejected as error:
        manifest["smoke"] = {"status": "rejected", "error": str(error)}
        atomic_json(root / "campaign.json", manifest)
        raise
    manifest["smoke"] = {**receipt, "status": "submitted"}
    atomic_json(root / "campaign.json", manifest)
    return manifest["smoke"]


def smoke(root: Path, manifest: Record) -> Record:
    from decentralized_llm.backends.factory import create_backend, runtime_for
    from decentralized_llm.backends.packed.engine import PackedBackend
    from decentralized_llm.data.loader import DevicePrefetcher, TokenLoader
    from decentralized_llm.execution.precision import precision_context, precision_state
    from decentralized_llm.training.evaluation import evaluate

    verify_snapshot(manifest)
    result: Record = {"source_sha256": manifest["source_sha256"], "modes": {}}
    for mode in MODES:
        row = next(r for r in manifest["runs"] if r["mode"] == mode and r["seed"] == 42)
        config = load_config(row["config"])
        runtime = runtime_for(config)
        device = runtime.setup(config)
        assert "GH200" in torch.cuda.get_device_name(device)
        torch.manual_seed(123)
        a, b = (torch.randn(512, 512, device=device) for _ in range(2))
        reference = a.double() @ b.double()
        with precision_context("ieee"):
            ieee = a @ b
        with precision_context("tf32"):
            tf32 = a @ b
        ieee_error = float((ieee.double() - reference).abs().mean())
        tf32_error = float((tf32.double() - reference).abs().mean())
        assert ieee.dtype == tf32.dtype == torch.float32
        assert tf32_error > ieee_error * 10 and tf32_error < 0.1
        before = precision_state()
        backend = create_backend(config, device, runtime)
        assert isinstance(backend, PackedBackend)
        backend.prepare()
        assert all(p.dtype == torch.float32 for p in backend.model.parameters())
        assert backend.execution.policy["backend"] == "native" and not backend.execution.policy["compile"]
        assert backend.capture is None
        torch.cuda.reset_peak_memory_stats(device)
        cache = TokenCache(config.data.cache_dir)
        blocks = config.training.batch_tokens // config.model.context_length
        steps = 3
        total = training_boundaries(config, config.model.parameter_count)[-1]
        metrics = []
        with TokenLoader(cache, config, 0, 0, blocks * steps) as loader:
            transfer = DevicePrefetcher(loader, device)
            try:
                loader.warmup()
                transfer.preload()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for step in range(steps):
                    metrics.append(
                        backend.update(transfer.next, transfer.preload, step * blocks, (step + 1) * blocks, step, total)
                    )
                    loader.commit((step + 1) * blocks)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            finally:
                transfer.close()
        config.evaluation.subset_blocks = 128
        validation = evaluate(backend.model, cache, config, device, full=False, backend=backend)
        assert precision_state() == before
        assert "fp32_precision" in validation
        assert all(v == "ieee" for v in validation["fp32_precision"].values())
        assert all(bool(torch.isfinite(p).all()) for p in backend.model.parameters())
        assert backend.optimizer.first_moment_storage.dtype == torch.float32
        assert backend.optimizer.second_moment_storage.dtype == torch.float32
        assert math.isfinite(validation["loss"])
        assert all(math.isfinite(m["local_loss"]) and math.isfinite(m["gradient_norm"]) for m in metrics)
        result["modes"][mode] = {
            "gpu": torch.cuda.get_device_name(device),
            "precision": before,
            "parameter_dtypes": sorted({str(p.dtype) for p in backend.model.parameters()}),
            "ieee_matmul_mean_error": ieee_error,
            "tf32_matmul_mean_error": tf32_error,
            "steps": metrics,
            "seconds_per_step": elapsed / steps,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "validation": validation,
            "evaluation_precision_restored": True,
        }
        del backend, transfer, loader
        runtime.close()
        torch.cuda.empty_cache()
    result["passed"] = True
    atomic_json(root / "smoke/result.json", result)
    return result


def submit(root: Path, manifest: Record) -> list[Record]:
    verify_snapshot(manifest)
    evidence = json.loads((root / "smoke/result.json").read_text())
    if not evidence.get("passed") or evidence["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("matching successful GH200 smoke evidence is required")
    for row in manifest["runs"]:
        if row.get("job_id"):
            continue
        destination = Path(row["run"]) / "submission"
        receipt = destination / "receipt.json"
        if receipt.exists():
            row.update(json.loads(receipt.read_text()) | {"status": "submitted"})
            atomic_json(root / "campaign.json", manifest)
            continue
        if row["status"] == "submitting":
            raise RuntimeError(f"reconcile uncertain submission before retrying: {row['key']}")
        row["status"] = "submitting"
        atomic_json(root / "campaign.json", manifest)
        try:
            with working_directory(Path(manifest["source"])):
                result = launch(
                    load_config(row["config"]),
                    verify=False,
                    job_name=f"precision-{row['key'].replace('/', '-')}",
                    submission_dir=destination,
                )
        except SubmissionRejected:
            row["status"] = "rejected"
            shutil.rmtree(destination)
            atomic_json(root / "campaign.json", manifest)
            raise
        row.update(job_id=result["job_id"], status="submitted", receipt=str(receipt))
        atomic_json(root / "campaign.json", manifest)
    return [{key: row[key] for key in ("key", "job_id", "run")} for row in manifest["runs"]]


def summarize(manifest: Record) -> Record:
    rows = []
    for run in manifest["runs"]:
        folder = Path(run["run"])
        row = {key: run.get(key) for key in ("mode", "seed", "job_id", "run")}
        path = folder / "result.json"
        failures = read_failures(folder)
        if path.exists():
            try:
                result = read_result(folder)
            except ValueError:
                row["status"] = "failed" if failures else "incomplete"
                rows.append(row)
                continue
            config = load_config(run["config"])
            expected = training_boundaries(config, config.model.parameter_count)[-1]
            validation = result["validation"]
            complete = result["cursor"] == expected and result["completed_epochs"] == config.training.epoch_count
            if complete and validation["split"] == "full" and math.isfinite(validation["loss"]):
                row.update(
                    status="complete",
                    loss=validation["loss"],
                    perplexity=validation["perplexity"],
                    consensus_error=validation["consensus_error"],
                )
            else:
                row["status"] = "incomplete"
        elif failures:
            row["status"] = "failed"
        else:
            row["status"] = "pending" if run.get("job_id") else "missing"
        rows.append(row)
    groups = {}
    for mode in MODES:
        losses = [r["loss"] for r in rows if r["mode"] == mode and r["status"] == "complete"]
        groups[mode] = {
            "completed_seeds": len(losses),
            "expected_seeds": len(SEEDS),
            "mean_loss": statistics.mean(losses) if len(losses) == len(SEEDS) else None,
            "sample_std_loss": statistics.stdev(losses) if len(losses) == len(SEEDS) else None,
        }
    pairs = []
    for seed in SEEDS:
        pair = {r["mode"]: r for r in rows if r["seed"] == seed and r["status"] == "complete"}
        if len(pair) == 2:
            pairs.append({"seed": seed, "tf32_minus_fp32_loss": pair["tf32"]["loss"] - pair["fp32"]["loss"]})
    return {"runs": rows, "groups": groups, "paired_differences": pairs}


def bf16_baselines(manifest: Record) -> list[Record]:
    """Locate the successful winner trials, including seeds reused from screening."""
    if not manifest.get("winner"):
        return []
    winner_path = Path(manifest["winner"])
    study = winner_path.parent
    if not all((study / name).exists() for name in ("report.json", "state.json")):
        return []
    if manifest.get("winner_sha256") and file_digest(winner_path) != manifest["winner_sha256"]:
        raise ValueError("the tuning winner changed since this precision campaign was prepared")
    winner_config = yaml.safe_load(winner_path.read_text())
    if not winner_config["runtime"]["amp"]:
        return []
    winner = json.loads((study / "report.json").read_text())["winner"]
    trials = json.loads((study / "state.json").read_text())["trials"]
    baselines = []
    for seed in sorted(winner["seeds"], key=lambda row: row["seed"]):
        if seed["seed"] not in SEEDS:
            continue
        trial = trials[seed["trial_id"]]
        attempt = next(attempt for attempt in reversed(trial["attempts"]) if attempt["status"] == "succeeded")
        folder = Path(attempt["directory"]) / "run"
        config = yaml.safe_load((folder / "config.yaml").read_text())
        if not config["runtime"]["amp"] or config["optimizer"] != winner_config["optimizer"]:
            raise ValueError(f"BF16 baseline does not match the tuning winner: {folder}")
        baselines.append(
            {
                "mode": "bf16",
                "seed": seed["seed"],
                "trial_id": seed["trial_id"],
                "run": str(folder),
                "config": str(folder / "config.yaml"),
                "evaluation_precision": "bf16",
                "execution_backend": config["runtime"]["execution_backend"],
            }
        )
    return baselines


def report(root: Path, manifest: Record) -> Record:
    result = summarize(manifest)
    baselines = bf16_baselines(manifest)
    result["plot_baselines"] = baselines
    if baselines:
        result["plot_note"] = (
            "TF32/FP32: native execution, FP32 evaluation. Historical BF16: fused GH200 execution, BF16 evaluation."
        )
    ids = [r["job_id"] for r in result["runs"] if r["job_id"]]
    if ids and shutil.which("sacct"):
        process = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "--format=JobIDRaw,State,ExitCode"],
            capture_output=True,
            text=True,
            check=False,
        )
        states = {parts[0]: parts[1:] for line in process.stdout.splitlines() if len(parts := line.split("|")) >= 3}
        for row in result["runs"]:
            if row["job_id"] in states:
                row["scheduler_state"], row["exit_code"] = states[row["job_id"]][:2]
                if row["status"] == "pending" and row["scheduler_state"] == "RUNNING":
                    row["status"] = "running"
                if row["status"] == "pending" and row["scheduler_state"] not in ("PENDING", "RUNNING", "CONFIGURING"):
                    row["status"] = "incomplete" if row["scheduler_state"] == "COMPLETED" else "failed"
    atomic_json(root / "report.json", result)
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "seed", "job_id", "status", "loss", "perplexity", "consensus_error", "run"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(result["runs"])
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, (ax, zoom_ax) = plt.subplots(1, 2, figsize=(14, 5.8))
    curves: list[tuple[str, list[int], list[float]]] = []
    final_tokens = 0
    for run in [*manifest["runs"], *baselines]:
        path = Path(run["run"]) / "metrics.jsonl"
        if not path.exists():
            continue
        records = read_validation_metrics(Path(run["run"]))
        if run["mode"] == "bf16":
            # Historical configs predate the error_feedback_gamma schema. Use the
            # actual saved epoch boundaries rather than reinterpreting the recipe.
            saved = read_metadata(Path(run["run"]))
            bounds = saved["boundaries"]
            length = yaml.safe_load(Path(run["config"]).read_text())["model"]["context_length"]
        else:
            config = load_config(run["config"])
            bounds = training_boundaries(config, config.model.parameter_count)
            length = config.model.context_length
        tokens = [bounds[r["epoch"] - 1] * length for r in records]
        losses = [r["loss"] for r in records]
        curves.append((run["mode"], tokens, losses))
        final_tokens = max(final_tokens, bounds[-1] * length)
        for axis in (ax, zoom_ax):
            axis.plot(
                [value / 1e6 for value in tokens],
                losses,
                color={"tf32": "tab:blue", "fp32": "tab:orange", "bf16": "tab:green"}[run["mode"]],
                linestyle={42: "-", 43: "--", 44: ":"}[run["seed"]],
                marker="o" if axis is zoom_ax else None,
                markersize=3,
                label=f"{run['mode']} seed {run['seed']}",
            )
    ax.set_title("Full training")
    zoom_ax.set_title("Final 20% of training")
    if final_tokens:
        zoom_start = 0.8 * final_tokens
        ax.axvspan(zoom_start / 1e6, final_tokens / 1e6, color="gray", alpha=0.12)
        zoom_ax.set_xlim(zoom_start / 1e6, final_tokens / 1e6)
        zoom_losses = [
            loss
            for _, tokens, losses in curves
            for token, loss in zip(tokens, losses, strict=True)
            if zoom_start <= token <= final_tokens and math.isfinite(loss)
        ]
        # Include the line segment crossing the left edge, even when epoch
        # rounding puts its previous validation point just outside the window.
        for _, tokens, losses in curves:
            for left, right, low_loss, high_loss in zip(tokens[:-1], tokens[1:], losses[:-1], losses[1:], strict=True):
                if left < zoom_start < right:
                    value = low_loss + (high_loss - low_loss) * (zoom_start - left) / (right - left)
                    if math.isfinite(value):
                        zoom_losses.append(value)
        if zoom_losses:
            low, high = min(zoom_losses), max(zoom_losses)
            padding = max((high - low) * 0.1, 1e-4)
            zoom_ax.set_ylim(low - padding, high + padding)
        else:
            zoom_ax.text(0.5, 0.5, "No validation points in this window yet", ha="center", transform=zoom_ax.transAxes)
        present = {mode for mode, tokens, _ in curves if any(token >= zoom_start for token in tokens)}
        missing = sorted({mode for mode, _, _ in curves} - present)
        if zoom_losses and missing:
            zoom_ax.text(
                0.98,
                0.98,
                f"{', '.join(mode.upper() for mode in missing)}: no data in this window yet",
                ha="right",
                va="top",
                transform=zoom_ax.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
    for axis in (ax, zoom_ax):
        axis.set(xlabel="Training tokens (millions)", ylabel="Validation cross entropy")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(alpha=0.2)
    if ax.lines:
        ax.legend(fontsize=8)
    if baselines:
        fig.text(0.5, 0.02, result["plot_note"], ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05 if baselines else 0, 1, 1))
    fig.savefig(root / "validation-loss.png", dpi=180)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke-submit", "smoke", "submit", "report"))
    parser.add_argument("--root", type=Path, default=Path("runs/precision") / DEFAULT_NAME)
    parser.add_argument("--winner", type=Path, default=Path("runs/p-tuning") / DEFAULT_NAME / "winner.yaml")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            result = prepare(root, args.winner.resolve())
        else:
            manifest = json.loads((root / "campaign.json").read_text())
            result = {"smoke-submit": smoke_submit, "smoke": smoke, "submit": submit, "report": report}[args.action](
                root, manifest
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
