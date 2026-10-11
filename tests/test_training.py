from pathlib import Path

import pytest
import torch

from decentralized_llm.cache import TokenCache
from decentralized_llm.checkpoints import validate_checkpoint
from decentralized_llm.config import Config
from decentralized_llm.train import train


def test_resume_equivalence(cache: TokenCache, tiny_config: Config, tmp_path: Path) -> None:
    full = train(tiny_config)
    original = torch.load(tiny_config.runtime.output_dir / "final/rank-00000.pt", weights_only=False)
    split = tiny_config.model_copy(deep=True)
    split.runtime.output_dir = tmp_path / "split"
    interrupted = train(split, stop_after_steps=2)
    assert interrupted["cursor"] == 12
    checkpoint = split.runtime.output_dir / "step-00000002"
    resumed = tiny_config.model_copy(deep=True)
    resumed.runtime.output_dir = tmp_path / "resumed"
    resumed.data.reader_threads = 1
    resumed.data.prefetch_batches = 1
    result = train(resumed, checkpoint)
    final = torch.load(resumed.runtime.output_dir / "final/rank-00000.pt", weights_only=False)
    assert result["cursor"] == full["cursor"]
    assert result["validation"]["tokens"] == 38
    for key, value in original["model"].items():
        torch.testing.assert_close(value, final["model"][key], rtol=0, atol=0)
    assert original["loader"] == final["loader"]
    assert torch.equal(original["rng"]["torch"], final["rng"]["torch"])
    changed = resumed.model_copy(deep=True)
    changed.optimizer.lr *= 2
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, changed, cache)
    (checkpoint / "rank-00000.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        validate_checkpoint(checkpoint, resumed, cache)
