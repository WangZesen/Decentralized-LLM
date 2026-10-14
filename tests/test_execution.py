import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from decentralized_llm.cache import TokenCache
from decentralized_llm.checkpoints import load_checkpoint, validate_checkpoint
from decentralized_llm.config import Config
from decentralized_llm.execution import (
    LossModule,
    build_execution,
    configure_compile_cache,
    qualification_key,
    resolve_execution,
)
from decentralized_llm.model import Llama, token_losses
from decentralized_llm.train import train
from decentralized_llm.training_benchmark import benchmark_training


def test_native_execution_and_canonical_state(tiny_config: Config) -> None:
    model = Llama(tiny_config.model)
    before = copy.deepcopy(model.state_dict())
    execution = build_execution(model, tiny_config, torch.device("cpu"))
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, -100]])
    expected = token_losses(model(x), y).sum()
    torch.testing.assert_close(execution(x, y), expected, rtol=0, atol=0)
    assert execution.policy["backend"] == "native" and not execution.policy["compile"]
    assert model.state_dict().keys() == before.keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    evaluation = LossModule(model, evaluation=True)
    torch.testing.assert_close(evaluation(x, y), token_losses(model(x), y).double().sum(), rtol=0, atol=0)


def test_forced_backend_rejects_cpu(tiny_config: Config) -> None:
    tiny_config.runtime.execution_backend = "gh200"
    with pytest.raises(ValueError, match="GH200 execution requires"):
        resolve_execution(tiny_config, torch.device("cpu"))


def test_regional_compilation_keeps_tail_eager(tiny_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.compile
    graphs = []

    def backend(graph: torch.fx.GraphModule, _: Any) -> Any:
        graphs.append(graph)
        return graph.forward

    def compile(model: Any, **_: Any) -> Any:
        return original(model, backend=backend, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile)
    tiny_config.runtime.compile = True
    tiny_config.runtime.compile_scope = "blocks"
    model = Llama(tiny_config.model)
    execution = build_execution(model, tiny_config, torch.device("cpu"))
    x = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    execution(x, x).backward()
    count = len(graphs)
    assert count > 0
    tail = execution(x[:1], x[:1])
    tail.backward()
    assert len(graphs) == count
    # deepcopy used by decentralized validation drops Module.compile wrappers.
    replica = copy.deepcopy(model)
    torch.testing.assert_close(tail, token_losses(replica(x[:1]), x[:1]).sum(), rtol=0, atol=0)


def test_automatic_selection_requires_matching_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config()
    config.distributed.gpus_per_node = 4
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GH200 120GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr("decentralized_llm.execution.kernel_revision", lambda: "kernel-v1")
    monkeypatch.setattr("decentralized_llm.execution.compile_revision", lambda: "compiler-v1")
    monkeypatch.setattr("decentralized_llm.execution.qualification_environment", lambda: {"triton": "measured"})
    entry = {
        "key": qualification_key(config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kernel_revision": "kernel-v1",
        "compile_revision": "compiler-v1",
        "environment": {"triton": "measured"},
        "backend": "gh200",
        "compile": True,
    }
    original = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "qualification.json":
            return json.dumps({"profiles": [entry]})
        return original(path)  # Only the registry is replaced in this test.

    monkeypatch.setattr(Path, "read_text", read)
    device = torch.device("cuda", 0)
    assert resolve_execution(config, device)["backend"] == "gh200"
    config.runtime.execution_backend = "native"
    policy = resolve_execution(config, device)
    assert policy["backend"] == "native" and policy["compile"] is False
    config.runtime.execution_backend = "auto"
    entry["environment"]["triton"] = "different"
    assert resolve_execution(config, device)["backend"] == "native"
    entry["environment"]["triton"] = "measured"
    entry["compile_revision"] = "old-compiler"
    assert resolve_execution(config, device)["backend"] == "native"
    entry["compile_revision"] = "compiler-v1"
    config.runtime.fused_rope = True
    with pytest.raises(ValueError, match="fused_rope requires"):
        resolve_execution(config, device)
    config.runtime.fused_rope = False
    config.distributed.ddp_bucket_cap_mb = 64
    assert resolve_execution(config, device)["backend"] == "native"
    config.distributed.ddp_bucket_cap_mb = 25
    config.runtime.compile = False
    assert resolve_execution(config, device)["backend"] == "native"
    config.runtime.compile = "auto"
    entry["kernel_revision"] = "old-kernel"
    assert resolve_execution(config, device)["backend"] == "native"


def test_published_flash_profile_is_scoped_to_measured_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    from decentralized_llm.config import load_config
    from decentralized_llm.execution import compile_revision, kernel_revision

    root = Path(__file__).resolve().parents[1]
    # Reconstruct the historical profile independently of current training presets.
    config = load_config(
        [root / "configs" / name for name in ("base.yaml", "20m.yaml")],
        [
            "distributed.gpus_per_node=4",
            "distributed.mode=decentralized",
            "runtime.sdpa_backend=flash",
        ],
    )
    registry = json.loads((root / "src/decentralized_llm/qualification.json").read_text())
    assert len(registry["profiles"]) == 1
    entry = registry["profiles"][0]
    assert entry["kernel_revision"] == kernel_revision()
    # Keep the measured profile intact; changed storage/compiler code is unqualified.
    assert entry["compile_revision"] != compile_revision()
    assert entry["key"] == qualification_key(config)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GH200 120GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch, "__version__", entry["torch"])
    monkeypatch.setattr(torch.version, "cuda", entry["cuda"])
    monkeypatch.setattr("decentralized_llm.execution.qualification_environment", lambda: entry["environment"])
    device = torch.device("cuda", 0)
    policy = resolve_execution(config, device)
    assert policy["backend"] == "native" and policy["compile"] is False
    # Reconstruct the historical revision to verify its original workload bounds.
    monkeypatch.setattr("decentralized_llm.execution.compile_revision", lambda: entry["compile_revision"])
    policy = resolve_execution(config, device)
    assert policy["backend"] == "gh200" and policy["compile"] is True
    monkeypatch.setattr(
        "decentralized_llm.execution.qualification_environment", lambda: {**entry["environment"], "allowed_cpus": 288}
    )
    policy = resolve_execution(config, device)
    assert policy["backend"] == "native" and policy["compile"] is False
    monkeypatch.setattr("decentralized_llm.execution.qualification_environment", lambda: entry["environment"])
    for section, field, value in [
        ("distributed", "nodes", 2),
        ("distributed", "mode", "ddp"),
        ("distributed", "backend", "gloo"),
        ("distributed", "error_feedback", True),
        ("runtime", "sdpa_backend", "auto"),
        ("runtime", "cpu_affinity", "exclusive"),
        ("runtime", "deterministic", True),
        ("training", "num_accumulation", 2),
    ]:
        changed = config.model_copy(deep=True)
        setattr(getattr(changed, section), field, value)
        policy = resolve_execution(changed, device)
        assert policy["backend"] == "native" and policy["compile"] is False


@pytest.mark.parametrize("feedback", [False, True])
def test_benchmark_uses_complete_update_windows(tiny_config: Config, cache: TokenCache, feedback: bool) -> None:
    tiny_config.distributed.mode = "decentralized"
    tiny_config.distributed.error_feedback = feedback
    tiny_config.benchmark.warmup = 2
    tiny_config.benchmark.steps = 2
    tiny_config.benchmark.windows = 2
    tiny_config.benchmark.min_seconds = 0.001
    result = benchmark_training(tiny_config)
    assert result["resident_at_measurement"]
    assert len(result["windows"]) == 2
    for window in result["windows"]:
        assert window["updates"] >= 2
        assert window["global_tokens_per_second"] * window["seconds"] == pytest.approx(
            window["updates"] * tiny_config.training.batch_tokens
        )
    assert (tiny_config.runtime.output_dir / "benchmark.json").is_file()
    expected_bytes = result["parameter_storage"]["communication_bytes"] if feedback else 0
    assert result["error_feedback"] == {"history_bytes": expected_bytes, "scratch_bytes": expected_bytes}


def test_legacy_checkpoint_pins_native_execution(tiny_config: Config, cache: TokenCache) -> None:
    train(tiny_config, stop_after_steps=2)
    checkpoint = tiny_config.runtime.output_dir / "step-00000002"
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("execution")
    path.write_text(json.dumps(manifest))
    tiny_config.runtime.compile = "auto"
    validate_checkpoint(checkpoint, tiny_config, cache)
    model = Llama(tiny_config.model)
    load_checkpoint(checkpoint, tiny_config, cache, model, None)
    assert model.execution_policy["backend"] == "native"
    assert model.execution_policy["compile"] is False


def test_compile_cache_shared_across_trials(
    tiny_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        monkeypatch.setenv(name, "2" if name == "TORCHINDUCTOR_COMPILE_THREADS" else "1")
    tiny_config.runtime.compile_cache_dir = tmp_path
    first = configure_compile_cache(tiny_config, torch.device("cpu"))
    tiny_config.runtime.seed += 1
    tiny_config.optimizer.lr *= 2
    tiny_config.optimizer.beta1 = 0.85
    tiny_config.runtime.output_dir = tmp_path / "other-trial"
    second = configure_compile_cache(tiny_config, torch.device("cpu"))
    assert first == second
    monkeypatch.setattr("decentralized_llm.execution.compile_revision", lambda: "changed-code")
    assert configure_compile_cache(tiny_config, torch.device("cpu"))["directory"] != first["directory"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fusion,rope", [(False, False), (True, False), (True, True)])
def test_gh200_kernel_gradients_and_mask(fusion: bool, rope: bool) -> None:
    cfg = Config().model
    cfg.layers, cfg.context_length, cfg.vocab_size = 2, 32, 128
    native = Llama(cfg).cuda()
    fast = copy.deepcopy(native)
    fast.execution_backend, fast.projection_fusion, fast.fused_rope = "gh200", fusion, rope
    x = torch.randint(128, (2, 32), device="cuda")
    y = torch.randint(128, (2, 32), device="cuda")
    y[0].fill_(-100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = LossModule(native)(x, y) / 32, LossModule(fast)(x, y) / 32
    torch.testing.assert_close(a, b, atol=0.002, rtol=0.002)
    ga = torch.cat([g.flatten() for g in torch.autograd.grad(a, tuple(native.parameters()))])
    gb = torch.cat([g.flatten() for g in torch.autograd.grad(b, tuple(fast.parameters()))])
    assert float((ga - gb).norm() / ga.norm()) < 0.05
    y.fill_(-100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        zero = LossModule(fast)(x, y)
    zero.backward()
    assert float(zero) == 0
    assert all(p.grad is not None and not bool(p.grad.count_nonzero()) for p in fast.parameters())
