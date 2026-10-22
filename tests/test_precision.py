from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.execution.precision import precision_context, precision_state, set_precision
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.training import checkpoints, evaluation


def test_precision_config_roundtrip(tiny_config: Config, tmp_path: Path) -> None:
    raw = tiny_config.model_dump()
    raw["runtime"].update(device="cuda", fp32_precision="tf32", sdpa_backend="math")
    raw["evaluation"]["fp32_precision"] = "ieee"
    config = Config.model_validate(raw)
    path = tmp_path / "precision.yaml"
    save_config(config, path)
    assert load_config(path) == config
    raw["runtime"]["amp"] = True
    with pytest.raises(ValidationError, match="amp=false"):
        Config.model_validate(raw)
    raw["runtime"].update(amp=False, device="cpu")
    with pytest.raises(ValidationError, match="requires CUDA"):
        Config.model_validate(raw)
    raw["runtime"]["fp32_precision"] = "high"
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_overrides_and_restores_precision(
    tiny_config: Config, cache: TokenCache, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    tiny_config.evaluation.fp32_precision = "ieee"
    tiny_config.runtime.sdpa_backend = "math"
    model = initialize_model(tiny_config)
    original = evaluation.prefetched_validation_batches
    observed = []

    def batches(*args: Any, **kwargs: Any):
        observed.append(precision_state())
        if fail:
            raise RuntimeError("validation reader failed")
        yield from original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "prefetched_validation_batches", batches)
    before = precision_state()
    with precision_context("tf32"):
        expected = precision_state()
        if fail:
            with pytest.raises(RuntimeError, match="reader failed"):
                evaluation.evaluate(model, cache, tiny_config, torch.device("cpu"), full=False)
        else:
            result = evaluation.evaluate(model, cache, tiny_config, torch.device("cpu"), full=False)
            assert result["loss"] > 0
            assert "fp32_precision" in result
            assert all(v == "ieee" for v in result["fp32_precision"].values())
        assert precision_state() == expected
        assert model.training
    assert precision_state() == before
    assert observed and all(v == "ieee" for v in observed[0].values())


def test_unset_precision_leaves_process_and_checkpoint_identity_unchanged(
    tiny_config: Config, cache: TokenCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    with precision_context("tf32"):
        before = precision_state()
        set_precision(None)
        with precision_context(None):
            assert precision_state() == before
    captured = []
    fingerprint = checkpoints.fingerprint

    def capture(value: Any):
        captured.append(value)
        return fingerprint(value)

    monkeypatch.setattr(checkpoints, "fingerprint", capture)
    old = checkpoints.recipe_identity(tiny_config, cache)
    assert "fp32_precision" not in captured[-1]["runtime"]
    assert "fp32_precision" not in captured[-1]["evaluation"]
    tiny_config.runtime.fp32_precision = "ieee"
    assert checkpoints.recipe_identity(tiny_config, cache) != old
    tiny_config.runtime.fp32_precision = None
    tiny_config.evaluation.fp32_precision = "ieee"
    assert checkpoints.recipe_identity(tiny_config, cache) != old


@pytest.mark.parametrize("name,value", [("NVIDIA_TF32_OVERRIDE", "0"), ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")])
def test_environment_cannot_silently_override_precision(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    before = precision_state()
    with pytest.raises(ValueError, match="environment overrides"), precision_context("ieee"):
        pass
    assert precision_state() == before
