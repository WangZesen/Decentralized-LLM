from pathlib import Path

import pytest
from pydantic import ValidationError

from decentralized_llm.config import Config, load_config, save_config
from decentralized_llm.optim import learning_rate, warmup_updates


@pytest.mark.parametrize("name", ["cosine", "wsd"])
def test_warmup_configuration(name: str, tmp_path: Path) -> None:
    overrides = [f"lr_schedule.name={name}"]
    assert load_config(overrides=overrides).lr_schedule.warmup_tokens_per_parameters == 1.0
    for value in (0, 0.125, 1):
        config = load_config(overrides=[*overrides, f"lr_schedule.warmup_tokens_per_parameters={value}"])
        assert config.lr_schedule.warmup_tokens_per_parameters == value
        path = tmp_path / "config.yaml"
        save_config(config, path)
        assert load_config(path).lr_schedule == config.lr_schedule
        overridden = load_config(path, ["lr_schedule.warmup_tokens_per_parameters=0.25"])
        assert overridden.lr_schedule.warmup_tokens_per_parameters == 0.25
        assert overridden.lr_schedule.name == name
    for value in ("-0.1", ".nan", ".inf", "-.inf", "null"):
        with pytest.raises(ValidationError, match="warmup_tokens_per_parameters"):
            load_config(overrides=[*overrides, f"lr_schedule.warmup_tokens_per_parameters={value}"])
    with pytest.raises(ValidationError, match="warmup_steps"):
        load_config(overrides=[*overrides, "lr_schedule.warmup_steps=1"])


@pytest.mark.parametrize(
    ("tpp", "updates"),
    [(0, 0), (1e-300, 1), (0.0149, 1), (0.015, 1), (0.0151, 1), (0.0749, 2), (0.075, 3), (0.0751, 3), (0.285, 10)],
)
def test_warmup_rounding(tiny_config: Config, tpp: float, updates: int) -> None:
    # This model has 800 parameters and consumes 24 tokens per update.
    # Decimal 0.285 gives exactly 9.5 updates; float arithmetic falls just below the tie.
    tiny_config.lr_schedule.warmup_tokens_per_parameters = tpp
    assert warmup_updates(tiny_config) == updates


def test_warmup_scaling(tiny_config: Config) -> None:
    tiny_config.lr_schedule.warmup_tokens_per_parameters = 0.12
    assert warmup_updates(tiny_config) == 4
    larger_batch = tiny_config.model_copy(deep=True)
    larger_batch.training.batch_tokens = 48
    assert warmup_updates(larger_batch) == 2
    larger_model = tiny_config.model_copy(deep=True)
    larger_model.model.vocab_size = 117
    assert larger_model.model.parameter_count == 2 * tiny_config.model.parameter_count
    assert warmup_updates(larger_model) == 8
    for overrides in (
        {"training": {"micro_batch_size": 1}},
        {"training": {"data_reuse_factor": 7}},
        {"distributed": {"gpus_per_node": 2}},
        {"distributed": {"mode": "decentralized", "gpus_per_node": 2}},
    ):
        raw = tiny_config.model_dump()
        for section, values in overrides.items():
            raw[section].update(values)
        assert warmup_updates(Config.model_validate(raw)) == 4


@pytest.mark.parametrize("name", ["cosine", "wsd"])
def test_warmup_and_decay(tiny_config: Config, name: str) -> None:
    raw = tiny_config.model_dump()
    options = {"min_lr_ratio": 0.2} if name == "cosine" else {"decay_fraction": 0.5}
    raw["lr_schedule"] = {"name": name, "warmup_tokens_per_parameters": 0.06, **options}
    config = Config.model_validate(raw)
    lr = config.optimizer.lr
    total = 168
    assert warmup_updates(config) == 2
    assert learning_rate(config, 0, total) == 0
    assert learning_rate(config, 24, total) == pytest.approx(lr / 2)
    assert learning_rate(config, 48, total) == lr
    if name == "cosine":
        assert learning_rate(config, 108, total) == pytest.approx(lr * 0.6)
        assert learning_rate(config, total, total) == pytest.approx(lr * 0.2)
    else:
        assert learning_rate(config, 84, total) == lr
        assert learning_rate(config, 105, total) == pytest.approx(lr / 2)
        assert learning_rate(config, total, total) == 0

    config.lr_schedule.warmup_tokens_per_parameters = 0
    assert learning_rate(config, 0, total) == lr
    assert learning_rate(config, total // 2, total) == pytest.approx(lr * 0.6 if name == "cosine" else lr)


@pytest.mark.parametrize("name", ["cosine", "wsd"])
@pytest.mark.parametrize(("tpp", "final_ratio"), [(0.18, 1), (0.36, 0.5)])
def test_warmup_reaches_or_exceeds_training(tiny_config: Config, name: str, tpp: float, final_ratio: float) -> None:
    raw = tiny_config.model_dump()
    raw["lr_schedule"] = {"name": name, "warmup_tokens_per_parameters": tpp}
    config = Config.model_validate(raw)
    assert learning_rate(config, 144, 144) == pytest.approx(config.optimizer.lr * final_ratio)


def test_wsd_without_decay(tiny_config: Config) -> None:
    raw = tiny_config.model_dump()
    raw["lr_schedule"] = {"name": "wsd", "warmup_tokens_per_parameters": 0.06, "decay_fraction": 0}
    config = Config.model_validate(raw)
    assert learning_rate(config, 24, 168) == pytest.approx(config.optimizer.lr / 2)
    for tokens in (48, 96, 168):
        assert learning_rate(config, tokens, 168) == config.optimizer.lr


def test_warmup_examples() -> None:
    assert load_config("configs/base.yaml").lr_schedule.warmup_tokens_per_parameters == 1.0
    assert load_config(["configs/base.yaml", "configs/wsd.yaml"]).lr_schedule.warmup_tokens_per_parameters == 1.0
    assert warmup_updates(load_config(["configs/base.yaml", "configs/smoke.yaml"])) == 1
    qualification = load_config(
        ["configs/base.yaml", "configs/90m.yaml", "configs/gh200-8gpu.yaml"],
        ["lr_schedule.warmup_tokens_per_parameters=0.007"],
    )
    assert warmup_updates(qualification) == 20
