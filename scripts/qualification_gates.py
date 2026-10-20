"""The existing five-update state probe and end-of-trajectory validation gates."""

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

GATE_PROTOCOL = "five-update-state-probe-plus-final-validation-v1"
PROBE_UPDATES = 5
GATE_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def assess_numerical(
    errors: Sequence[dict[str, float]],
    validation_delta: float,
    *,
    steps: int,
    optimizer_only: bool,
    feedback: bool,
    fused_candidate: bool,
) -> dict:
    """Keep long-trajectory diagnostics separate from the historical state probe.

    Optimizer comparisons record every update, including history. Their counters
    must agree throughout and all recorded values must be finite. The historical
    relative-error thresholds apply to the first five updates; final validation
    measures the independently evolving trajectories.
    """
    probe_updates = min(steps, PROBE_UPDATES)
    expected = steps if optimizer_only else probe_updates
    complete = steps > 0 and len(errors) == expected
    required = {"loss_tolerance_ratio", "parameters", "gradients", "updates", "exp_avg", "exp_avg_sq"}
    if optimizer_only:
        required.add("steps")
        if feedback:
            required.add("history")
    finite = complete and all(
        required.issubset(entry) and all(math.isfinite(value) for value in entry.values()) for entry in errors
    )

    def accurate(entry: dict[str, float]) -> bool:
        return (
            entry["loss_tolerance_ratio"] <= 1
            and entry["parameters"] < 0.02
            and entry["gradients"] < 0.05
            and entry["updates"] < 0.10
            and entry["exp_avg"] < 0.05
            and entry["exp_avg_sq"] < 0.05
            and (not (optimizer_only and feedback) or entry["history"] < 0.10)
        )

    probe = finite and all(accurate(entry) for entry in errors[:probe_updates])
    counters = finite and (not optimizer_only or all(entry["steps"] == 0 for entry in errors))
    validation = math.isfinite(validation_delta) and abs(validation_delta) <= 0.02
    implementation = not optimizer_only or fused_candidate
    common = finite and counters and validation and implementation
    return {
        "protocol": GATE_PROTOCOL,
        "revision": GATE_REVISION,
        "probe_updates": probe_updates,
        "recorded_updates": len(errors),
        "records_complete": complete,
        "finite": finite,
        "state_probe": probe,
        "step_counters": counters,
        "validation": validation,
        "implementation": implementation,
        "passed": common and probe,
        "all_recorded_updates_passed": common and all(accurate(entry) for entry in errors),
    }
