"""Hybrid optimizer contracts, independent reference updates, and training integration."""

import copy
import json
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
import torch
from test_distributed import reference_mix

from decentralized_llm.algorithms.adamw import make_optimizer, optimizer_report
from decentralized_llm.algorithms.muon import MuonAdamW
from decentralized_llm.algorithms.packed_muon import PackedMuonAdamW
from decentralized_llm.algorithms.schedules import learning_rate, learning_rates, set_learning_rates
from decentralized_llm.backends.factory import create_backend
from decentralized_llm.backends.packed.engine import PackedBackend
from decentralized_llm.config import Config, OptimizerConfig, load_config, save_config
from decentralized_llm.data.cache import TokenCache
from decentralized_llm.execution.qualification import profile_key, qualification_key
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.models.llama import Block, Llama, token_losses
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.runtime import clip_grad_norm_
from decentralized_llm.training.checkpoints import recipe_identity, validate_checkpoint
from decentralized_llm.training.loop import train
from decentralized_llm.tuning.spec import Phase, expand, numeric_recipe


def assert_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_state_equal(left, right)
    else:
        assert actual == expected


def reference_optimizers(model: Llama, config: Config) -> tuple[torch.optim.Muon, torch.optim.AdamW]:
    # Derive membership independently from the module tree, not the production classifier.
    matrices = [p for block in model.blocks for p in block.parameters() if p.ndim == 2]
    cfg = config.optimizer
    return (
        torch.optim.Muon(
            matrices,
            lr=cfg.lr * cfg.muon_lr_ratio,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        ),
        torch.optim.AdamW(
            [
                {"params": [model.embedding.weight], "weight_decay": cfg.weight_decay},
                {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            fused=False,
            foreach=False,
        ),
    )


def test_configuration_and_legacy_identities(tiny_config: Config, cache: TokenCache, tmp_path: Path) -> None:
    assert tiny_config.optimizer.muon_lr_ratio == 1.0
    legacy = tiny_config.optimizer.model_dump(exclude={"name", "muon_lr_ratio", "muon_momentum"})
    assert tiny_config.optimizer.recipe() == legacy
    identity = recipe_identity(tiny_config, cache)
    numeric = numeric_recipe(tiny_config)
    key = qualification_key(tiny_config)
    # Inactive settings must not invalidate an AdamW run or tuning candidate.
    tiny_config.optimizer.muon_lr_ratio = 4.0
    tiny_config.optimizer.muon_momentum = 0.8
    assert recipe_identity(tiny_config, cache) == identity
    assert numeric_recipe(tiny_config) == numeric
    assert qualification_key(tiny_config) == key
    assert profile_key({"key": key}) == key
    tiny_config.optimizer.name = "muon_adamw"
    assert recipe_identity(tiny_config, cache) != identity
    assert numeric_recipe(tiny_config) != numeric
    assert qualification_key(tiny_config) != key
    for field, value in (("lr", 0.002), ("muon_lr_ratio", 3.0), ("muon_momentum", 0.9)):
        other = tiny_config.model_copy(deep=True)
        setattr(other.optimizer, field, value)
        assert recipe_identity(other, cache) != recipe_identity(tiny_config, cache)
    path = tmp_path / "config.yaml"
    save_config(tiny_config, path)
    assert load_config(path).optimizer == tiny_config.optimizer
    assert "muon_lr:" not in path.read_text()
    assert load_config("configs/muon.yaml").optimizer.muon_lr_ratio == 1.0
    assert load_config(path, ["optimizer.muon_lr_ratio=0.5"]).optimizer.muon_lr_ratio == 0.5
    phase = Phase(name="rates", search={"optimizer.lr": [0.001, 0.002], "optimizer.muon_lr_ratio": [0.5, 3.0]}, top_k=1)
    assert len(expand(tiny_config, phase)) == 4
    for raw in ({"name": "muon"}, {"muon_lr": 0.001}, {"muon_momentum": 1}):
        with pytest.raises(ValueError):
            OptimizerConfig.model_validate(raw)
    for ratio in (0, -1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="muon_lr_ratio"):
            OptimizerConfig(muon_lr_ratio=ratio)


def test_partition_ties_and_biases(tiny_config: Config) -> None:
    tiny_config.optimizer.name = "muon_adamw"
    model = Llama(tiny_config.model)
    block = cast(Block, model.blocks[0])
    block.attention.q_proj.bias = torch.nn.Parameter(torch.zeros(tiny_config.model.width))
    model.lm_head = torch.nn.Linear(tiny_config.model.width, tiny_config.model.vocab_size, bias=False)
    model.lm_head.weight = model.embedding.weight
    model.norm.weight.requires_grad_(False)
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    assert isinstance(optimizer, MuonAdamW)
    groups = optimizer.param_groups
    assert len(groups[0]["params"]) == 7 * tiny_config.model.layers
    assert groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    assert groups[0]["ns_steps"] == 5 and groups[0]["nesterov"]
    assert groups[1]["params"] == [model.embedding.weight]
    assert groups[2]["weight_decay"] == 0
    assert any(p is block.attention.q_proj.bias for p in groups[2]["params"])
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert optimizer_report(optimizer)["implementation"] == "torch_muon_adamw"


@pytest.mark.parametrize("name", ["cosine", "wsd"])
@pytest.mark.parametrize("warmup", [0, 0.06, 0.36])
@pytest.mark.parametrize("ratio", [0.5, 1.0, 3.0])
def test_two_scheduled_rates(tiny_config: Config, name: str, warmup: float, ratio: float) -> None:
    raw = tiny_config.model_dump()
    raw["lr_schedule"] = {"name": name, "warmup_tokens_per_parameters": warmup}
    raw["optimizer"].update(name="muon_adamw", lr=0.0003, muon_lr_ratio=ratio)
    cfg = Config.model_validate(raw)
    opt = make_optimizer(Llama(cfg.model), cfg, torch.device("cpu"))
    assert [group["lr"] for group in opt.param_groups] == [cfg.optimizer.lr * ratio, cfg.optimizer.lr, cfg.optimizer.lr]
    for consumed in (0, 24, 48, 84, 108, 168):
        rates = learning_rates(cfg, consumed, 168)
        assert rates["lr"] == rates["adamw_lr"] == learning_rate(cfg, consumed, 168)
        assert rates["muon_lr"] == rates["lr"] * ratio
        set_learning_rates(opt.param_groups, rates)
        assert [group["lr"] for group in opt.param_groups] == [rates["muon_lr"], rates["lr"], rates["lr"]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gamma", [0.0, 0.5, -1.0])
def test_reference_updates_and_roundtrip(tiny_config: Config, dtype: torch.dtype, gamma: float) -> None:
    cfg = tiny_config
    cfg.optimizer.name = "muon_adamw"
    cfg.optimizer.muon_lr_ratio = 3.0
    cfg.distributed.error_feedback_gamma = gamma
    model = initialize_model(cfg).to(dtype)
    reference = copy.deepcopy(model)
    optimizer = make_optimizer(model, cfg, torch.device("cpu"))
    assert isinstance(optimizer, MuonAdamW)
    muon, adamw = reference_optimizers(reference, cfg)
    history = [torch.zeros_like(p) for p in reference.parameters()]
    for step, factor in enumerate((1.0, 0.2, 0.0, 0.8)):
        set_learning_rates(
            optimizer.param_groups,
            {"lr": cfg.optimizer.lr * factor, "muon_lr": cfg.optimizer.lr * cfg.optimizer.muon_lr_ratio * factor},
        )
        for child, base in ((muon, cfg.optimizer.lr * cfg.optimizer.muon_lr_ratio), (adamw, cfg.optimizer.lr)):
            for group in child.param_groups:
                group["lr"] = base * factor
        for index, (actual, expected) in enumerate(zip(model.parameters(), reference.parameters(), strict=True)):
            gradient = torch.arange(actual.numel(), dtype=dtype).view_as(actual).sin() * (index + step + 1)
            actual.grad = None if step == 1 and index in (0, 2) else gradient
            expected.grad = None if actual.grad is None else actual.grad.clone()
        before = [p.detach().clone() for p in reference.parameters()]
        optimizer.step()
        muon.step()
        adamw.step()
        with torch.no_grad():
            for actual, expected, base, previous in zip(
                model.parameters(), reference.parameters(), before, history, strict=True
            ):
                delta = expected - base
                if gamma:
                    expected.add_(delta, alpha=gamma).sub_(previous, alpha=gamma)
                previous.copy_(delta)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected_state = muon.state[expected] if expected in muon.state else adamw.state[expected]
                assert_state_equal(optimizer.state[actual], expected_state)
        restored_model = copy.deepcopy(model)
        restored = make_optimizer(restored_model, cfg, torch.device("cpu"))
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assert_state_equal(restored.state_dict(), optimizer.state_dict())
        # Continue with restored objects, exercising the live groups and buffers after load.
        model, optimizer = restored_model, restored


def test_closure_and_invalid_state(tiny_config: Config) -> None:
    tiny_config.optimizer.name = "muon_adamw"
    tiny_config.distributed.error_feedback_gamma = 0.5
    model = initialize_model(tiny_config)
    optimizer = make_optimizer(model, tiny_config, torch.device("cpu"))
    calls = []

    def closure() -> float:
        optimizer.zero_grad()
        loss = token_losses(model(torch.tensor([[1, 2, 3, 4]])), torch.tensor([[2, 3, 4, 5]])).mean()
        loss.backward()
        calls.append(float(loss.detach()))
        return calls[-1]

    assert optimizer.step(closure) == calls[0] and len(calls) == 1
    good = copy.deepcopy(optimizer.state_dict())
    for field, value in (("kind", "adamw"), ("layout", []), ("previous_update", None)):
        with pytest.raises(ValueError):
            optimizer.load_state_dict({**good, field: value})
        assert_state_equal(optimizer.state_dict(), good)
    bad = copy.deepcopy(good)
    next(iter(bad["adamw"]["state"].values()))["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="exp_avg"):
        optimizer.load_state_dict(bad)
    assert_state_equal(optimizer.state_dict(), good)


@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("topology", ["complete", "one_peer_ring", "one_peer_exponential"])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_packed_independent_updates(tiny_config: Config, scheme: str, topology: str, gamma: float) -> None:
    cfg = tiny_config
    cfg.optimizer.name = "muon_adamw"
    cfg.optimizer.muon_lr_ratio = 3.0
    cfg.distributed.error_feedback_gamma = gamma
    canonical = initialize_model(cfg)
    packed = PackedLlama(cfg.model, 4, canonical=canonical)
    optimizer = PackedMuonAdamW(packed, cfg)
    references = [copy.deepcopy(canonical) for _ in range(4)]
    opts = [reference_optimizers(m, cfg) for m in references]
    histories = [[torch.zeros_like(p) for p in m.parameters()] for m in references]
    parameters = tuple(packed.parameters())
    for step in range(4):
        optimizer.zero_grad()
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter).sin() + step
            for worker, model in enumerate(references):
                model.get_parameter(packed.layout[index].name).grad = parameter.grad[worker].clone()
        optimizer.clip_grad_norm_(cfg.optimizer.grad_clip)
        for model in references:
            clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        if scheme == "awc":
            packed.mix_(topology, step)
            reference_mix(references, topology, step)
        optimizer.step()
        for model, (muon, adamw), history in zip(references, opts, histories, strict=True):
            before = [p.detach().clone() for p in model.parameters()]
            muon.step()
            adamw.step()
            with torch.no_grad():
                for parameter, base, previous in zip(model.parameters(), before, history, strict=True):
                    delta = parameter - base
                    if gamma:
                        parameter.add_(delta, alpha=gamma).sub_(previous, alpha=gamma)
                    previous.copy_(delta)
        if scheme == "atc":
            packed.mix_(topology, step)
            reference_mix(references, topology, step)
        for worker, reference in enumerate(references):
            assert_state_equal(packed.local_state_dict(worker), reference.state_dict())
            local = optimizer.local_state_dict(worker)
            for name, p in reference.named_parameters():
                muon, adamw = opts[worker]
                expected = muon.state[p] if p in muon.state else adamw.state[p]
                assert_state_equal(local["state"][name], expected)
        assert all(actual is expected for actual, expected in zip(packed.parameters(), parameters, strict=True))
        if step == 1:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = PackedMuonAdamW(packed, cfg)
            optimizer.load_state_dict(state)
            assert_state_equal(optimizer.state_dict(), state)
    bad = copy.deepcopy(optimizer.state_dict())
    bad["workers"][-1]["layout"] = []
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="layout"):
        optimizer.load_state_dict(bad)
    assert_state_equal(optimizer.state_dict(), before)


@pytest.mark.parametrize("backend", ["distributed", "packed"])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_training_resume_and_metrics(
    tiny_config: Config,
    cache: TokenCache,
    tmp_path: Path,
    backend: Literal["distributed", "packed"],
    mode: Literal["ddp", "decentralized"],
    scheme: Literal["awc", "atc"],
    gamma: float,
) -> None:
    cfg = tiny_config
    cfg.optimizer.name = "muon_adamw"
    cfg.optimizer.muon_lr_ratio = 2.0
    cfg.runtime.training_backend = backend
    cfg.distributed.mode = mode
    cfg.distributed.scheme = scheme
    cfg.distributed.error_feedback_gamma = gamma
    if backend == "packed":
        cfg.distributed.gpus_per_node = 2
    cfg.distributed.topology = "one_peer_exponential"
    filename = "packed.pt" if backend == "packed" else "rank-00000.pt"
    train(cfg)
    expected = torch.load(cfg.runtime.output_dir / "final" / filename, weights_only=False)
    metrics = [
        json.loads(line) for line in (cfg.runtime.output_dir / "metrics-rank-00000.jsonl").read_text().splitlines()
    ]
    for row in metrics:
        assert row["lr"] == row["adamw_lr"]
        assert row["muon_lr"] == pytest.approx(2 * row["adamw_lr"])
    cfg.runtime.output_dir = tmp_path / "split"
    train(cfg, stop_after_steps=2)
    checkpoint = cfg.runtime.output_dir / "step-00000002"
    cfg.runtime.output_dir = tmp_path / "resume"
    train(cfg, checkpoint)
    actual = torch.load(cfg.runtime.output_dir / "final" / filename, weights_only=False)
    for key in ("model", "optimizer", "loader"):
        assert_state_equal(actual[key], expected[key])
    cfg.optimizer.muon_lr_ratio = 1.0
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, cfg, cache)
    cfg.optimizer.muon_lr_ratio = 2.0
    cfg.optimizer.name = "adamw"
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(checkpoint, cfg, cache)


@pytest.mark.parametrize("execution_backend", ["gh200", "a100"])
def test_hybrid_skips_adamw_capture(tiny_config: Config, execution_backend: str) -> None:
    tiny_config.optimizer.name = "muon_adamw"
    tiny_config.runtime.training_backend = "packed"
    tiny_config.runtime.fused_optimizer = True
    backend = create_backend(tiny_config, torch.device("cpu"))
    assert isinstance(backend, PackedBackend)
    with patch("decentralized_llm.backends.packed.engine.build_execution") as build:
        build.return_value.policy = {"backend": execution_backend}
        backend.prepare()
    assert backend.capture is None
    assert backend.execution.policy["update_capture"] is False
    assert backend.memory_report()["optimizer"]["implementation"] == "torch_muon_adamw"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend_name", ["distributed", "packed"])
def test_cuda_hybrid_update(tiny_config: Config, backend_name: Literal["distributed", "packed"]) -> None:
    cfg = tiny_config
    cfg.optimizer.name = "muon_adamw"
    cfg.runtime.training_backend = backend_name
    cfg.runtime.device = "cuda"
    cfg.runtime.fused_optimizer = True
    cfg.runtime.deterministic = False
    cfg.distributed.mode = "decentralized"
    cfg.distributed.error_feedback_gamma = 0.5
    device = torch.device("cuda")
    backend = create_backend(cfg, device)
    model, optimizer = backend.model, backend.optimizer
    x = torch.randint(cfg.model.vocab_size, (2, 4), device=device)
    if backend_name == "packed":
        x = x.unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = token_losses(model(x), x).mean()
    loss.backward()
    optimizer.step()
    children = optimizer.optimizers if isinstance(optimizer, PackedMuonAdamW) else [optimizer]
    for child in children:
        assert isinstance(child, MuonAdamW)
        assert child.implementation == "cudagraph_muon_adamw"
        assert all(group["fused"] for group in child.adamw.param_groups)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    restored = create_backend(cfg, device)
    restored.load_state_dict(copy.deepcopy(backend.state_dict(True)), True)
    assert_state_equal(restored.state_dict(True), backend.state_dict(True))
