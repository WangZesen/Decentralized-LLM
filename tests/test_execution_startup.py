import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from decentralized_llm.backends.distributed import runtime as distributed
from decentralized_llm.config import Config, load_config
from decentralized_llm.execution.policy import resolve_execution
from decentralized_llm.launch.commands import absolute_config
from decentralized_llm.launch.local import local_command
from decentralized_llm.launch.slurm import scheduler_arguments, slurm_scripts


@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("cpu_override", [None, "null", "64"])
def test_slurm_cpu_allocation_defaults_to_site_policy(
    nodes: int, cpu_override: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "configs" / name for name in ("base.yaml", "models/90m.yaml", f"setups/gh200-{nodes * 4}gpu.yaml")]
    overrides = [] if cpu_override is None else [f"launcher.cpus_per_task={cpu_override}"]
    config = load_config(paths, overrides)
    arguments = scheduler_arguments(config)
    assert f"--nodes={nodes}" in arguments and "--gpus-per-node=4" in arguments
    cpu_arguments = [arg for arg in arguments if arg.startswith(("--cpus-per-task=", "--cpus-per-gpu="))]
    assert cpu_arguments == (["--cpus-per-task=64"] if cpu_override == "64" else [])
    batch, worker = slurm_scripts(config, tmp_path / "config.yaml", tmp_path / "worker.sh")
    assert "--cpus-per-task" not in batch + worker and "--cpus-per-gpu" not in batch + worker
    assert "--gpus-per-task=4" in batch and "--nproc-per-node=4" in worker
    assert "export OMP_NUM_THREADS=8" in batch
    assert "--numa-binding=exclusive" in worker
    assert config.training.batch_tokens == 32768 and config.micro_batch_size == 8 // nodes
    if nodes == 2:
        assert "--network=job_vni" in arguments
        assert worker.index("scripts/environment.sh") < worker.index("scripts/activate_cxi.sh") < worker.index("exec ")
    else:
        assert not any(arg.startswith("--network=") for arg in arguments)
        assert "activate_cxi.sh" not in worker
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GH200 120GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    policy = resolve_execution(config, torch.device("cuda", 0))
    assert policy["backend"] == "gh200" and policy["compile"] is True
    assert policy["fused_rope"] is True
    assert policy["reason"] == "explicit selection"


@pytest.mark.parametrize("affinity", ["none", "local", "exclusive"])
def test_affinity_is_parsed_by_torchrun_for_local_and_slurm_launches(
    affinity: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torch.distributed.run import parse_args

    raw = Config().model_dump()
    raw["runtime"]["cpu_affinity"] = affinity
    raw["distributed"]["gpus_per_node"] = 4
    config = Config.model_validate(raw)
    filename = tmp_path / "config with spaces.yaml"
    local = parse_args(local_command(config, filename)[3:])
    assert local.numa_binding == ("exclusive" if affinity == "exclusive" else None)
    assert local.training_script == "decentralized_llm" and local.module
    assert local.training_script_args[-1] == str(filename)

    raw["launcher"] = {"kind": "slurm", "partition": "gpu"}
    raw["distributed"]["nodes"] = 2
    config = Config.model_validate(raw)
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("MASTER_ADDR", "node-zero")
    _, worker = slurm_scripts(config, filename, tmp_path / "worker.sh")
    args = shlex.split(os.path.expandvars(worker.split("exec ", 1)[1]))[3:]
    slurm = parse_args(args)
    assert slurm.numa_binding == local.numa_binding
    assert slurm.nnodes == "2" and slurm.node_rank == 1
    assert slurm.training_script_args == local.training_script_args


def test_exclusive_affinity_requires_accelerator() -> None:
    raw = Config().model_dump()
    raw["runtime"].update(device="cpu", cpu_affinity="exclusive")
    raw["distributed"]["backend"] = "gloo"
    with pytest.raises(ValueError, match="requires GPU placement"):
        Config.model_validate(raw)


def test_slurm_network_overlay_precedes_torch_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base environment.sh"
    overlay = tmp_path / "network overlay.sh"
    base.write_text("export OVERLAY_ORDER=base\n")
    overlay.write_text('export OVERLAY_ORDER="$OVERLAY_ORDER,network"\n')
    python = tmp_path / "fake python"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$OVERLAY_ORDER"\n')
    python.chmod(0o755)
    raw = Config().model_dump()
    raw["distributed"].update(nodes=2, gpus_per_node=4)
    raw["launcher"] = {
        "kind": "slurm",
        "partition": "gpu",
        "network": "job_vni",
        "environment_script": base.name,
        "environment_overlays": [overlay.name],
    }
    config = absolute_config(Config.model_validate(raw))
    assert "--network=job_vni" in scheduler_arguments(config)
    _, worker = slurm_scripts(config, tmp_path / "config.yaml", tmp_path / "worker.sh")
    env = dict(os.environ, DECENTRALIZED_PYTHON=str(python), SLURM_PROCID="0", MASTER_ADDR="node0")
    result = subprocess.run(["bash", "-c", worker], env=env, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "base,network"
    overlay.write_text("return 7\n")
    result = subprocess.run(["bash", "-c", worker], env=env, check=False, capture_output=True, text=True)
    assert result.returncode == 7 and not result.stdout
    raw["launcher"].pop("network")
    assert not any(arg.startswith("--network=") for arg in scheduler_arguments(Config.model_validate(raw)))


@pytest.mark.parametrize("deterministic", [False, True])
def test_cublas_workspace_follows_determinism(monkeypatch: pytest.MonkeyPatch, deterministic: bool) -> None:
    config = Config()
    config.runtime.deterministic = deterministic
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="visible GPUs"):
        distributed.setup(config)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == (":4096:8" if deterministic else None)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="visible GPUs"):
        distributed.setup(config)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_compile_cache_configuration_after_optimizer_import(tmp_path: Path) -> None:
    code = """
import sys, torch
from pathlib import Path
from decentralized_llm.config import Config
from decentralized_llm.execution.compilation import configure_compile_cache
torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
from torch._functorch import config as autograd_config
assert not autograd_config.autograd_cache_allow_custom_autograd_functions
config = Config()
config.runtime.compile_cache_dir = Path(sys.argv[1])
cache = configure_compile_cache(config, torch.device("cpu"))
assert cache["settings"]["custom_autograd_cache"]
assert cache["settings"]["fx_graph_cache"]
assert cache["settings"]["compile_threads"] == 4
"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("TORCHINDUCTOR_")}
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, check=True, capture_output=True)


def test_numa_binding_preserves_pci_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    query = Mock(return_value="00000009:00:00.0\n")
    monkeypatch.setattr(subprocess, "check_output", query)

    def read(path: Path) -> str:
        assert str(path) == "/sys/bus/pci/devices/0009:00:00.0/local_cpulist"
        return "2-5,8"

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 2, 3, 7, 8})
    setter = Mock()
    monkeypatch.setattr(os, "sched_setaffinity", setter)
    from decentralized_llm.execution.device import bind_local_cpus

    bind_local_cpus(1)
    assert "--id=3" in query.call_args.args[0]
    setter.assert_called_once_with(0, {2, 3, 8})


def test_reusable_mix_owns_snapshot_until_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    request = Mock(spec=torch.distributed.Work)

    def reduce(tensor: torch.Tensor, **kwargs: object) -> Mock:
        tensor.mul_(2)
        return request

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    before = [p.clone() for p in model.parameters()]
    pending = distributed.begin_parameter_mix(model, "complete", 0, reuse=True)
    assert pending is not None
    pointer = pending.flat.data_ptr()
    pending.wait()
    with pytest.raises(RuntimeError, match="still owns"):
        distributed.begin_parameter_mix(model, "complete", 1, reuse=True)
    for p, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(p, expected, atol=0, rtol=0)
    pending.apply()
    pending.close()
    second = distributed.begin_parameter_mix(model, "complete", 1, reuse=True)
    assert second is not None and second.flat.data_ptr() == pointer
    second.apply()
    second.close()
    for p, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(p, expected, atol=0, rtol=0)


def test_environment_fast_activation_and_invalidation(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts/environment.sh"
    for name in ("environment.sh", "toolchain.sh"):
        (root / "scripts" / name).write_text((Path(__file__).resolve().parents[1] / "scripts" / name).read_text())
    (root / "uv.lock").write_text("fixture")
    (root / "pyproject.toml").write_text("[project]\ndependencies=[]\n[dependency-groups]\ndev=[]\n")
    arch = platform.machine()
    executable = root / f".venv-{arch}/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    env_root = tmp_path / "environments"
    uv = env_root / arch / "bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text('#!/usr/bin/env bash\nif [[ "$1" == sync ]]; then echo sync >> "$UV_PROJECT_ENVIRONMENT/syncs"; fi\n')
    uv.chmod(0o755)
    env = dict(os.environ, DECENTRALIZED_ENV_ROOT=str(env_root), PATH="/usr/bin:/bin")
    command = ["bash", "-c", 'source "$1"', "bash", str(script)]
    subprocess.run(command, env=env, check=True, capture_output=True)
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert (executable.parent.parent / "syncs").read_text() == "sync\n"
    uv.unlink()
    subprocess.run(command, env=env, check=True, capture_output=True)
    (root / "uv.lock").write_text("changed")
    result = subprocess.run(command, env=env, check=False, capture_output=True, text=True)
    assert result.returncode != 0 and "Install native" in result.stderr
