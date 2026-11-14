"""Revision-aware machine-readable and human-readable tuning reports."""

import json
from collections import Counter
from pathlib import Path

from decentralized_llm.config import Config, save_config
from decentralized_llm.runtime import atomic_json
from decentralized_llm.tuning.batching import format_time
from decentralized_llm.tuning.outcomes import format_loss
from decentralized_llm.tuning.plots import write_heatmaps
from decentralized_llm.tuning.spec import Record


def describe(candidate: Record) -> str:
    parts = []
    for key, value in sorted(candidate["values"].items()):
        label = candidate["labels"].get(key)
        display = f"{label['expression']} ({value:g})" if label else str(value)
        parts.append(f"{key}={display}")
    return ", ".join(parts)


def write_report(directory: Path, state: Record) -> Record:
    trials = state["trials"]
    phase_rows = []
    for phase in state["phases"]:
        rows = []
        for candidate in phase["candidates"]:
            seeds = []
            for key in phase["trials"][candidate["id"]]:
                trial = trials[key]
                seeds.append(
                    {
                        "trial_id": key,
                        "display_id": trial.get("display_id", key),
                        "seed": trial["seed"],
                        "status": trial["status"],
                        "loss": trial.get("loss"),
                        "attempts": trial["attempts"],
                        "result": trial.get("result"),
                        "divergence": trial.get("divergence"),
                    }
                )
            rows.append(
                {
                    "id": candidate["id"],
                    "values": candidate["values"],
                    "labels": candidate["labels"],
                    "description": describe(candidate),
                    "seeds": seeds,
                }
            )
        rankings = [
            {key: row[key] for key in ("id", "values", "labels", "mean_loss", "std_loss", "seeds")}
            for row in phase.get("rankings", [])
        ]
        phase_rows.append(
            {
                "name": phase["name"],
                "complete": phase["complete"],
                "started": phase.get("started"),
                "finished": phase.get("finished"),
                "wall_seconds": None
                if phase.get("started") is None
                else phase.get("finished", state["updated"]) - phase["started"],
                "definition": phase["definition"],
                "reused_trials": phase["reused_trials"],
                "new_trials": phase["new_trials"],
                "candidates": rows,
                "rankings": rankings,
                "selected": phase.get("selected", []),
            }
        )
    for phase in phase_rows:
        phase["heatmaps"] = write_heatmaps(directory, phase)
    winner = state.get("winner")
    report = {
        "revision": state["revision"],
        "status": state["status"],
        "reason": state.get("reason"),
        "warning": state.get("warning"),
        "provenance": state["provenance"],
        "updated": state["updated"],
        "phases": phase_rows,
        "allocations": list(state.get("allocations", {}).values()),
        "allocation_counts": dict(Counter(item["status"] for item in state.get("allocations", {}).values())),
        "winner": None
        if winner is None
        else {key: winner[key] for key in ("id", "values", "labels", "mean_loss", "std_loss", "seeds")},
    }
    atomic_json(directory / "report.json", report)
    lines = [f"# Tuning report — revision {state['revision']}", "", f"Status: **{state['status']}**", ""]
    if state.get("reason"):
        lines += [str(state["reason"]), ""]
    if state.get("warning"):
        lines += [str(state["warning"]), ""]
    if report["allocations"]:
        lines += [
            "## Slurm allocations",
            "",
            "; ".join(f"{status}: {count}" for status, count in report["allocation_counts"].items()) + ".",
            "",
            "| Allocation | Phase / revision | Status | Trials | Walltime | Logs |",
            "|---|---|---|---:|---|---|",
        ]
        for allocation in report["allocations"]:
            job = str(allocation.get("job_id", allocation["name"]))
            stdout = allocation["stdout"].replace("%j", job)
            stderr = allocation["stderr"].replace("%j", job)
            lines.append(
                f"| [{job}]({allocation['directory']}/manifest.json) | "
                f"{allocation['phase']} / {allocation['revision']} | {allocation['status']} | "
                f"{len(allocation['trial_ids'])} | {format_time(allocation['time_minutes'])} | "
                f"[stdout]({stdout}), [stderr]({stderr}) |"
            )
        lines += [""]
    for phase in phase_rows:
        lines += [
            f"## {phase['name']}",
            "",
            "```json",
            json.dumps(phase["definition"], indent=2),
            "```",
            "",
            f"Reused trials: {phase['reused_trials']}; new trials: {phase['new_trials']}.",
            "",
            *(
                [f"Phase wall time (including queueing and controller downtime): {phase['wall_seconds']:.1f} s.", ""]
                if phase["wall_seconds"] is not None
                else []
            ),
        ]
        for plot in phase["heatmaps"]:
            lines += [f"![{phase['name']} tuning losses]({plot['png']})", "", f"[SVG version]({plot['svg']})", ""]
        lines += [
            "| Combination | Mean objective loss | Sample std | Selected |",
            "|---|---:|---:|---|",
        ]
        for row in phase["rankings"]:
            std = "—" if row["std_loss"] is None else f"{row['std_loss']:.6g}"
            selected = "yes" if row["id"] in phase["selected"] else ""
            lines.append(f"| {describe(row)} | {format_loss(row['mean_loss'])} | {std} | {selected} |")
        lines += [""]
        for candidate in phase["candidates"]:
            lines += [f"### {candidate['description']}", ""]
            for trial in candidate["seeds"]:
                loss = "None" if trial["loss"] is None else format_loss(trial["loss"])
                lines.append(f"- Seed {trial['seed']}: {trial['status']}; loss={loss}; trial `{trial['display_id']}`.")
                for attempt in trial["attempts"]:
                    label = attempt.get("job_id", attempt.get("pid", "unresolved"))
                    message = attempt.get("error", attempt.get("scheduler_state", ""))
                    lines.append(
                        f"  - Attempt {attempt['number']}: {attempt['status']}; job/process {label}; "
                        f"[artifacts]({attempt['directory']}); {message}"
                    )
                    if "allocation_id" in attempt:
                        allocation = state["allocations"][attempt["allocation_id"]]
                        lines.append(
                            f"    [Allocation {attempt['allocation_id']}]({allocation['directory']}/manifest.json); "
                            f"[stdout]({attempt['stdout']}), [stderr]({attempt['stderr']}), "
                            f"[execution record]({attempt['execution_record']}); "
                            f"started={attempt.get('started', False)}."
                        )
            lines += [""]
    if winner:
        lines += ["## Final winner", "", describe(winner), "", f"Mean final loss: {winner['mean_loss']:.8g}", ""]
        config = Config.model_validate(winner["config"])
        config.runtime.output_dir = directory / f"winner-run-revision-{state['revision']}"
        config.runtime.seed = state["spec"]["seed_start"]
        save_config(config, directory / "winner.yaml")
    else:
        (directory / "winner.yaml").unlink(missing_ok=True)
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(directory / "report.md")
    return report
