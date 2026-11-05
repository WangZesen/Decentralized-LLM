"""CUDA checks that can run on a single GPU, independently of hardware qualification."""

import io
from typing import Literal

import pytest
import torch

from decentralized_llm.algorithms.packed_adamw import PackedAdamW
from decentralized_llm.backends.factory import runtime_for
from decentralized_llm.config import Config
from decentralized_llm.data.sampling import BatchSpec
from decentralized_llm.execution.native import TrainingExecution
from decentralized_llm.models.initialization import initialize_model
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.runtime import NonfiniteUpdateError

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("scheme", ["awc", "atc"])
@pytest.mark.parametrize("mode", ["ddp", "decentralized"])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, -1.0, 2.0])
def test_captured_update_parity(
    tiny_config: Config,
    scheme: Literal["awc", "atc"],
    mode: Literal["ddp", "decentralized"],
    gamma: float,
) -> None:
    from decentralized_llm.execution.cuda.capture import CapturedUpdate
    from decentralized_llm.execution.cuda.packed_optimizer import ArenaAdamW

    raw = tiny_config.model_dump(mode="json")
    raw["runtime"].update(
        training_backend="packed", device="cuda", amp=True, deterministic=False, fused_optimizer=False
    )
    raw["distributed"].update(
        mode=mode, scheme=scheme, gpus_per_node=4, topology="one_peer_exponential", error_feedback_gamma=gamma
    )
    raw["training"].update(batch_tokens=64, num_accumulation=2)
    cfg = Config.model_validate(raw)
    device = runtime_for(cfg).setup(cfg)
    workers = 4 if mode == "decentralized" else 1
    model = PackedLlama(cfg.model, workers, canonical=initialize_model(cfg)).to(device)
    reference = PackedLlama(cfg.model, workers, canonical=initialize_model(cfg)).to(device)
    optimizer = ArenaAdamW(model, cfg)
    ordinary = PackedAdamW(reference, cfg)
    execution = TrainingExecution(model, cfg, device)
    expected_execution = TrainingExecution(reference, cfg, device)
    engine = CapturedUpdate(model, optimizer, execution, cfg, device)
    before = model.parameter_storage.clone()
    rng = torch.cuda.get_rng_state().clone()
    engine.prepare()
    assert torch.equal(model.parameter_storage, before)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert optimizer.inspect(0) == 0
    assert not torch.count_nonzero(optimizer.first_moment_storage)
    if gamma != 0:
        assert optimizer.previous_update is not None
        assert not torch.count_nonzero(optimizer.previous_update)
    else:
        assert optimizer.previous_update is None and ordinary.previous_update is None
        assert ordinary.scratch is None
    blocks, micro = 16, cfg.process_micro_batch_size
    for step in range(4):
        if step == 2:
            # A resumed graph must preserve nonzero moments, history, and its rotation phase.
            checkpoint = io.BytesIO()
            torch.save({"model": model.packed_state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
            checkpoint.seek(0)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = PackedLlama(cfg.model, workers, canonical=initialize_model(cfg)).to(device)
            model.load_packed_state_dict(saved["model"])
            optimizer = ArenaAdamW(model, cfg)
            optimizer.load_state_dict(saved["optimizer"])
            execution = TrainingExecution(model, cfg, device)
            engine = CapturedUpdate(model, optimizer, execution, cfg, device)
            rng = torch.cuda.get_rng_state().clone()
            engine.prepare()
            assert torch.equal(model.parameter_storage.cpu(), saved["model"]["parameter_storage"])
            assert torch.equal(optimizer.first_moment_storage.cpu(), saved["optimizer"]["first_moment_storage"])
            if gamma != 0:
                assert optimizer.previous_update is not None
                assert torch.equal(optimizer.previous_update.cpu(), saved["optimizer"]["previous_update"])
            assert torch.equal(torch.cuda.get_rng_state(), rng)
            optimizer.inspect(step)
        x = torch.randint(cfg.model.vocab_size, (blocks, 4), device=device)
        y = torch.randint(cfg.model.vocab_size, x.shape, device=device)
        batches = iter(
            [
                (
                    (x[offset : offset + micro], y[offset : offset + micro]),
                    BatchSpec(
                        step * blocks + offset, step * blocks + offset + micro, torch.empty(0).numpy().astype("int64")
                    ),
                )
                for offset in range(0, blocks, micro)
            ]
        )
        lr = 0.001 / (step + 1)
        for group in ordinary.param_groups:
            group["lr"] = lr
        ordinary.zero_grad()
        for offset in range(0, blocks, micro):
            a, b = (
                tensor[offset : offset + micro].view(-1, workers, 4).transpose(0, 1).contiguous() for tensor in (x, y)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = expected_execution(a, b).sum() / (64 / workers)
            loss.backward()
        ordinary.clip_grad_norm_(cfg.optimizer.grad_clip)
        if mode == "decentralized" and scheme == "awc":
            reference.mix_(cfg.distributed.topology, step)
        ordinary.step()
        if mode == "decentralized" and scheme == "atc":
            reference.mix_(cfg.distributed.topology, step)
        engine.execute(batches.__next__, lambda: None, step * blocks, (step + 1) * blocks, lr)
        optimizer.inspect(step + 1)
        torch.testing.assert_close(model.parameter_storage, reference.parameter_storage, rtol=3e-4, atol=3e-6)
        for actual, expected in [
            (optimizer.first_moment_storage, ordinary.first_moment_storage),
            (optimizer.second_moment_storage, ordinary.second_moment_storage),
        ]:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-5)
        if gamma != 0:
            torch.testing.assert_close(optimizer.previous_update, ordinary.previous_update, rtol=3e-4, atol=3e-6)
    # A nonfinite worker must prevent every worker's parameters/moments/counter from advancing.
    snapshot = model.parameter_storage.clone()
    moments = optimizer.first_moment_storage.clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    gradient = next(iter(model.parameters())).grad
    assert gradient is not None
    gradient.fill_(float("nan"))
    optimizer._finalized = False
    optimizer.step(torch.zeros(workers, device=device))
    with pytest.raises(NonfiniteUpdateError, match="nonfinite"):
        optimizer.inspect()
    assert torch.equal(model.parameter_storage, snapshot)
    assert torch.equal(optimizer.first_moment_storage, moments)
    assert int(optimizer.completed) == 4


@pytest.mark.parametrize("hardware", ["gh200", "a100"])
def test_fused_packed_forward(tiny_config: Config, hardware: str) -> None:
    from decentralized_llm.execution.native import LossModule

    raw = tiny_config.model_dump(mode="json")
    raw["model"].update(width=320, heads=5, ffn_width=896, layers=1, context_length=16, vocab_size=128)
    raw["training"].update(batch_tokens=64, num_accumulation=1)
    raw["runtime"].update(amp=True, device="cuda", deterministic=False)
    cfg = Config.model_validate(raw)
    torch.use_deterministic_algorithms(False)
    reference = PackedLlama(cfg.model, 2, canonical=initialize_model(cfg)).cuda()
    fused = PackedLlama(cfg.model, 2, canonical=initialize_model(cfg)).cuda()
    fused.execution_backend = hardware
    fused.fused_rope = True
    x = torch.randint(128, (2, 2, 16), device="cuda")
    y = torch.randint(128, x.shape, device="cuda")
    y[0, 0].fill_(-100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = LossModule(reference)(x, y).sum() / 64
        actual = LossModule(fused)(x, y).sum() / 64
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    a = torch.cat([g.flatten() for g in torch.autograd.grad(actual, tuple(fused.parameters()))])
    b = torch.cat([g.flatten() for g in torch.autograd.grad(expected, tuple(reference.parameters()))])
    assert float((a - b).norm() / b.norm()) < 0.05


def test_production_compiled_packed_update(tiny_config: Config) -> None:
    """Exercise policy selection, Inductor, fused layers, and full-update replay together."""
    import numpy as np

    from decentralized_llm.backends.factory import create_backend
    from decentralized_llm.config import PRESETS

    name = torch.cuda.get_device_name()
    hardware = (
        "gh200"
        if "GH200" in name
        else "a100"
        if "A100" in name or torch.cuda.get_device_capability() == (8, 9)
        else None
    )
    if hardware is None:
        pytest.skip("production fused policy requires GH200, A100, or experimental Ada 20M")
    raw = tiny_config.model_dump(mode="json")
    raw["model"].update(**PRESETS["20m"], context_length=16, vocab_size=128)
    raw["runtime"].update(
        training_backend="packed",
        device="cuda",
        amp=True,
        deterministic=False,
        execution_backend=hardware,
        compile=True,
        fused_optimizer=True,
        compile_cache_dir=str(tiny_config.runtime.output_dir.parent / "compile"),
    )
    raw["distributed"].update(mode="decentralized", scheme="atc", gpus_per_node=2, error_feedback_gamma=1.0)
    raw["training"].update(batch_tokens=64, num_accumulation=2)
    cfg = Config.model_validate(raw)
    runtime = runtime_for(cfg)
    backend = create_backend(cfg, runtime.setup(cfg), runtime)
    before = {name: p.detach().clone() for name, p in backend.model.named_parameters()}
    backend.prepare()
    assert backend.execution.policy["update_capture"]
    for name, p in backend.model.named_parameters():
        assert torch.equal(p, before[name]), name
    blocks = cfg.training.batch_tokens // cfg.model.context_length
    for step in range(2):
        batches = iter(
            [
                (
                    (torch.randint(128, (2, 16), device="cuda"), torch.randint(128, (2, 16), device="cuda")),
                    BatchSpec(step * blocks + offset, step * blocks + offset + 2, np.empty(0, dtype=np.int64)),
                )
                for offset in range(0, blocks, 2)
            ]
        )
        result = backend.update(batches.__next__, lambda: None, step * blocks, (step + 1) * blocks, step, blocks * 2)
        backend.inspect(step + 1)
        assert np.isfinite(result["local_losses"]).all()
    assert any(not torch.equal(p, before[name]) for name, p in backend.model.named_parameters())
    from conftest import make_cache

    from decentralized_llm.data.sampling import GlobalSampler
    from decentralized_llm.training.checkpoints import load_checkpoint, save_checkpoint
    from decentralized_llm.training.evaluation import evaluate

    cache = make_cache(cfg.data.cache_dir, cfg)
    checkpoint = cfg.runtime.output_dir.parent / "captured-checkpoint"
    sampler = {"sampler": GlobalSampler.from_config(cache, cfg).identity, "cursor": 2 * blocks}
    progress = {"cursor": 2 * blocks, "step": 2, "completed_epochs": 0, "best_loss": None, "best_epoch": None}
    save_checkpoint(checkpoint, cfg, cache, backend.model, backend.optimizer, sampler, progress, backend=backend)
    resumed = create_backend(cfg, backend.device, runtime)
    load_checkpoint(checkpoint, cfg, cache, resumed.model, resumed.optimizer, backend=resumed)
    resumed.prepare()
    for actual, expected in zip(resumed.model.parameters(), backend.model.parameters(), strict=True):
        assert torch.equal(actual, expected)
    shared = [
        (
            (torch.randint(128, (2, 16), device="cuda"), torch.randint(128, (2, 16), device="cuda")),
            BatchSpec(2 * blocks + offset, 2 * blocks + offset + 2, np.empty(0, dtype=np.int64)),
        )
        for offset in range(0, blocks, 2)
    ]
    for current in (backend, resumed):
        batches = iter(shared)
        current.update(batches.__next__, lambda: None, 2 * blocks, 3 * blocks, 2, blocks * 4)
    for actual, expected in zip(resumed.model.parameters(), backend.model.parameters(), strict=True):
        assert torch.equal(actual, expected)
    result = evaluate(resumed.model, cache, cfg, resumed.device, full=True, backend=resumed)
    assert result["tokens"] == 38 and np.isfinite(result["loss"])
