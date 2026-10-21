"""Terminal trial outcomes and strict-JSON representations of objective losses."""

import math
import statistics
from collections.abc import Sequence
from typing import Literal

from decentralized_llm.tuning.spec import Record

TERMINAL_TRIALS = frozenset({"succeeded", "diverged"})
type Loss = float | Literal["inf"]


def loss_value(value: object) -> float:
    """Decode a finite loss or the sole supported nonfinite JSON sentinel."""
    if value == "inf":
        return math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("loss must be finite or the string 'inf'")
    return float(value)


def summarize_losses(losses: Sequence[object]) -> tuple[Loss, float | None]:
    values = [loss_value(value) for value in losses]
    if any(math.isinf(value) for value in values):
        return "inf", None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def select_finite(rankings: list[Record], top_k: int) -> list[Record]:
    return [row for row in rankings if math.isfinite(loss_value(row["mean_loss"]))][:top_k]


def format_loss(value: object) -> str:
    return "∞" if value == "inf" else f"{loss_value(value):.8g}"
