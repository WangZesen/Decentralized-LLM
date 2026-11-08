import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from conftest import make_cache

from decentralized_llm.algorithms.adamw import ErrorFeedbackAdamW, make_optimizer
from decentralized_llm.backends.distributed import update as updates
from decentralized_llm.backends.distributed.runtime import PendingParameterMix
from decentralized_llm.config import Config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.data.sampling import BatchSpec
from decentralized_llm.models.llama import Llama
from decentralized_llm.models.storage import parameter_storage
from decentralized_llm.state import Batch
from decentralized_llm.training.checkpoints import validate_checkpoint
from decentralized_llm.training.loop import train


@pytest.mark.parametrize("feedback", [False, True])
def test_awc_failed_update_drains_without_applying(
    tiny_config: Config, monkeypatch: pytest.MonkeyPatch, feedback: bool
) -> None:
    tiny_config.distributed.mode = "decentralized"
    tiny_config.distributed.error_feedback_gamma = float(feedback)
    model = Llama(tiny_config.model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    work = Mock(spec=torch.distributed.Work)
    storage = parameter_storage(model)
    pending = PendingParameterMix(storage, torch.zeros_like(storage.flat), [work], 2)
    launch = Mock(return_value=pending)
    monkeypatch.setattr(updates, "begin_parameter_mix", launch)

    def fail() -> tuple[Batch, BatchSpec]:
        launch.assert_called_once()
        work.wait.assert_not_called()
        raise OSError("injected loader failure")

    with pytest.raises(OSError, match="injected loader failure"):
        updates.optimizer_update(
            model, model, optimizer, tiny_config, torch.device("cpu"), fail, lambda: None, 0, 6, 0, 42
        )
    work.wait.assert_called_once()
    assert not pending.requests
    assert not optimizer.state
    if isinstance(optimizer, ErrorFeedbackAdamW):
        assert torch.count_nonzero(optimizer.previous_update) == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("split_step", [1, 2])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_error_feedback_resume(
    cache: TokenCache, tiny_config: Config, tmp_path: Path, scheme: str, split_step: int, mode: str, gamma: float
) -> None:
    tiny_config.distributed.mode = mode  # type: ignore[assignment]
    tiny_config.distributed.error_feedback_gamma = gamma
    tiny_config.distributed.scheme = scheme  # type: ignore[assignment]
    train(tiny_config)
    original = torch.load(tiny_config.runtime.output_dir / "final/rank-00000.pt", weights_only=False)
    tiny_config.runtime.output_dir = tmp_path / "split"
    train(tiny_config, stop_after_steps=split_step)
    checkpoint = tiny_config.runtime.output_dir / f"step-{split_step:08d}"
    tiny_config.runtime.output_dir = tmp_path / "resumed"
    train(tiny_config, checkpoint)
    actual = torch.load(tiny_config.runtime.output_dir / "final/rank-00000.pt", weights_only=False)
    for key, value in original["model"].items():
        torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
    for key, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual["optimizer"]["state"][key][name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        actual["optimizer"]["error_feedback"]["previous_update"],
        original["optimizer"]["error_feedback"]["previous_update"],
        rtol=0,
        atol=0,
    )
    assert actual["loader"] == original["loader"]
    assert torch.equal(actual["rng"]["torch"], original["rng"]["torch"])
    tiny_config.distributed.error_feedback_gamma = gamma + 0.25
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, tiny_config, cache)


@pytest.mark.parametrize("factor", [1, 2, 5, 7])
@pytest.mark.parametrize("larger_cache,accumulation", [(False, 1), (True, 3)])
@pytest.mark.parametrize("schedule", ["cosine", "wsd"])
@pytest.mark.parametrize("repeat", [False, True])
def test_resume_equivalence(
    cache: TokenCache,
    tiny_config: Config,
    tmp_path: Path,
    factor: int,
    larger_cache: bool,
    accumulation: int,
    schedule: str,
    repeat: bool,
) -> None:
    raw = tiny_config.model_dump()
    raw["training"]["num_accumulation"] = accumulation
    # Interrupt after two of four warmup updates, then resume into the decay phase.
    raw["lr_schedule"] = {"name": schedule, "warmup_tokens_per_parameters": 0.12}
    tiny_config = Config.model_validate(raw)
    if larger_cache:
        tiny_config.data.cache_dir = tmp_path / "larger-cache"
        cache = make_cache(tiny_config.data.cache_dir, tiny_config, tokens=401)
    tiny_config.training.data_reuse_factor = factor
    tiny_config.training.data_reuse_order = "repeat" if repeat else "reshuffle"
    tiny_config.data.loader_mode = "streaming" if factor == 7 else "resident"
    full = train(tiny_config)
    assert full["cursor"] == 42 and full["step"] == 7 and full["completed_epochs"] == 2
    run = json.loads((tiny_config.runtime.output_dir / "run.json").read_text())
    assert run["num_accumulation"] == accumulation
    assert run["micro_batch_size"] == 6 // accumulation
    pool = cache.blocks("train", 4) if factor == 1 else (42 + factor - 1) // factor
    assert run["sampling"] == {
        "data_reuse_factor": factor,
        "pool_blocks": pool,
        "unique_training_tokens": min(pool, 42) * 4,
        "full_passes": 42 // pool,
        "partial_pass_blocks": 42 % pool,
        **({"data_reuse_order": "repeat"} if repeat else {}),
    }
    original = torch.load(tiny_config.runtime.output_dir / "final/rank-00000.pt", weights_only=False)
    split = tiny_config.model_copy(deep=True)
    split.data.loader_mode = "resident" if factor == 7 else "streaming"
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
    for key, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, final["optimizer"]["state"][key][name], rtol=0, atol=0)
    assert original["loader"] == final["loader"]
    assert torch.equal(original["rng"]["torch"], final["rng"]["torch"])
    changed = resumed.model_copy(deep=True)
    changed.training.num_accumulation = 2
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, changed, cache)
    changed = resumed.model_copy(deep=True)
    changed.optimizer.lr *= 2
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, changed, cache)
    changed = resumed.model_copy(deep=True)
    changed.lr_schedule.warmup_tokens_per_parameters = 0.15
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, changed, cache)
    changed = resumed.model_copy(deep=True)
    changed.training.data_reuse_factor = 2 if factor == 1 else 1
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, changed, cache)
    changed = resumed.model_copy(deep=True)
    changed.training.data_reuse_order = "reshuffle" if repeat else "repeat"
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, changed, cache)
    manifest_path = checkpoint / "manifest.json"
    original_manifest = manifest_path.read_text()
    manifest = json.loads(original_manifest)
    manifest["sampler"]["blocks"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sampler identity"):
        validate_checkpoint(checkpoint, resumed, cache)
    manifest_path.write_text(original_manifest)
    (checkpoint / "rank-00000.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        validate_checkpoint(checkpoint, resumed, cache)
