from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from decentralized_llm.algorithms.adamw import make_optimizer
from decentralized_llm.algorithms.schedules import learning_rate
from decentralized_llm.config import PRESETS, Config, ModelConfig, load_config, save_config
from decentralized_llm.data.cache import training_boundaries
from decentralized_llm.models.llama import Llama, token_losses


def test_config_layers(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("lr_schedule: {name: cosine, min_lr_ratio: 0.2}\nlauncher: {kind: slurm, partition: gpu}\n")
    b.write_text("lr_schedule: {name: wsd, decay_fraction: 0.3}\nlauncher: {kind: local}\n")
    cfg = load_config([a, b], ["runtime.seed=17", "model.width=640"])
    assert cfg.runtime.seed == 17 and cfg.lr_schedule.name == "wsd" and cfg.launcher.kind == "local"
    assert cfg.model.width == 640
    with pytest.raises(ValidationError):
        load_config(overrides=["runtime.unknown=true"])
    for value in (".nan", ".inf", "-.inf"):
        with pytest.raises(ValidationError):
            load_config(overrides=[f"optimizer.lr={value}"])
    with pytest.raises(ValidationError, match="one node"):
        load_config(overrides=["distributed.nodes=2"])
    with pytest.raises(ValidationError, match="power-of-two"):
        load_config(overrides=["distributed.gpus_per_node=3", "distributed.topology=one_peer_exponential"])
    with pytest.raises(ValidationError):
        load_config(overrides=["launcher.kind=slurm", "launcher.partition=gpu", "launcher.account=secret"])
    with pytest.raises(ValueError):
        load_config(overrides=["optimizer.lr.no=1"])


def test_presets_and_parameter_count() -> None:
    assert set(PRESETS) == {"20m", "50m", "90m", "250m", "500m"}
    for values in PRESETS.values():
        cfg = ModelConfig(**values)
        with torch.device("meta"):
            model = Llama(cfg)
        assert model.parameter_count == cfg.parameter_count
        assert "embedding.weight" in model.state_dict()
        assert len([p for p in model.parameters() if p.shape == model.embedding.weight.shape]) == 1


def test_data_reuse_configuration(tmp_path: Path) -> None:
    assert Config().training.data_reuse_factor == 1
    assert Config().training.data_reuse_order == "reshuffle"
    cfg = load_config(overrides=["training.data_reuse_factor=4"])
    path = tmp_path / "reuse.yaml"
    save_config(cfg, path)
    assert load_config(path).training.data_reuse_factor == 4
    assert load_config(path, ["training.data_reuse_factor=2"]).training.data_reuse_factor == 2
    for value in ("0", "-1", "true", "false", "2.0", "1.5", '"2"', "null"):
        with pytest.raises(ValidationError, match="data_reuse_factor"):
            load_config(overrides=[f"training.data_reuse_factor={value}"])
    with pytest.raises(ValidationError, match="data_reuse_factor"):
        cfg.training.data_reuse_factor = True
    cfg.training.data_reuse_order = "repeat"
    save_config(cfg, path)
    assert load_config(path).training.data_reuse_order == "repeat"
    for value in ("unknown", "true", "1", "null"):
        with pytest.raises(ValidationError, match="data_reuse_order"):
            load_config(overrides=[f"training.data_reuse_order={value}"])


def test_loader_configuration() -> None:
    assert Config().data.loader_mode == "auto"
    assert Config().data.host_memory_mib == 2048
    for mode in ("auto", "resident", "streaming"):
        assert load_config(overrides=[f"data.loader_mode={mode}"]).data.loader_mode == mode
    for override in ("data.loader_mode=unknown", "data.preload_max_read_amplification=0.5"):
        with pytest.raises(ValidationError):
            load_config(overrides=[override])


def test_model_causality_and_backends(tiny_config: Config) -> None:
    torch.manual_seed(1)
    ref = Llama(tiny_config.model, "reference").double()
    fast = Llama(tiny_config.model, "sdpa").double()
    fast.load_state_dict(ref.state_dict())
    x, changed = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 7, 8]])
    torch.testing.assert_close(ref(x)[:, :2], ref(changed)[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(ref(x), fast(x), atol=1e-12, rtol=1e-10)
    y = torch.tensor([[2, 3, 4, -100]])
    a, b = token_losses(ref(x), y), token_losses(fast(x), y)
    assert a[0, -1] == 0
    ga = torch.autograd.grad(a.sum(), tuple(ref.parameters()))
    gb = torch.autograd.grad(b.sum(), tuple(fast.parameters()))
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-9)


def test_rounding_schedules_optimizer(tiny_config: Config) -> None:
    boundaries = training_boundaries(tiny_config, tiny_config.model.parameter_count)
    assert boundaries == [18, 42]
    total = boundaries[-1] * 4
    lr = tiny_config.optimizer.lr
    assert learning_rate(tiny_config, 24, total) == lr
    assert learning_rate(tiny_config, total, total) == 0
    raw = tiny_config.model_dump()
    raw["lr_schedule"] = {"name": "wsd", "warmup_tokens_per_parameters": 0.03, "decay_fraction": 0.5}
    wsd = Config.model_validate(raw)
    assert learning_rate(wsd, 24, total) == lr
    assert learning_rate(wsd, total, total) == 0
    assert learning_rate(wsd, total // 2, total) == lr
    for factor in (2, 5, 7, 100):
        for baseline in (tiny_config, wsd):
            reuse = baseline.model_copy(deep=True)
            reuse.training.data_reuse_factor = factor
            assert training_boundaries(reuse, reuse.model.parameter_count) == boundaries
            assert [learning_rate(reuse, tokens, total) for tokens in range(24, total + 1, 24)] == [
                learning_rate(baseline, tokens, total) for tokens in range(24, total + 1, 24)
            ]
    optimizer = make_optimizer(Llama(tiny_config.model), tiny_config, torch.device("cpu"))
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0
    assert all(p.ndim < 2 for p in optimizer.param_groups[1]["params"])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_fused_and_compiled_model() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(False)
    config = Config.model_validate(
        {
            "model": {"vocab_size": 128, "layers": 2, "width": 64, "heads": 2, "ffn_width": 176, "context_length": 32},
            "runtime": {"device": "cuda", "amp": True, "fused_optimizer": True},
        }
    )
    ref = Llama(config.model, "reference").cuda()
    fast = Llama(config.model, "sdpa").cuda()
    fast.load_state_dict(ref.state_dict())
    x = torch.randint(128, (2, 32), device="cuda")
    y = torch.randint(128, (2, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_loss = token_losses(ref(x), y).mean()
        fast_loss = token_losses(fast(x), y).mean()
    torch.testing.assert_close(reference_loss, fast_loss, atol=0.002, rtol=0.002)
    reference_grad = torch.cat([g.flatten() for g in torch.autograd.grad(reference_loss, tuple(ref.parameters()))])
    fast_grad = torch.cat([g.flatten() for g in torch.autograd.grad(fast_loss, tuple(fast.parameters()))])
    assert (reference_grad - fast_grad).norm() / reference_grad.norm() < 0.05
    optimizer = make_optimizer(fast, config, torch.device("cuda"))
    assert optimizer.param_groups[0]["fused"] is True
    compiled = torch.compile(fast)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = token_losses(compiled(x), y).mean()
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in fast.parameters())
