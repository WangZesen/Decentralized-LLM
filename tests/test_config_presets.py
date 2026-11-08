from pathlib import Path

import pytest

from decentralized_llm.config import SlurmLauncher, load_config
from decentralized_llm.launch.slurm import scheduler_arguments, slurm_scripts
from decentralized_llm.tuning.spec import read_spec

HARDWARE = [
    ("a100", 1, 1, 1),
    ("a100", 2, 1, 2),
    ("a100", 4, 1, 4),
    ("a100", 8, 1, 8),
    ("gh200", 1, 1, 1),
    ("gh200", 2, 1, 2),
    ("gh200", 4, 1, 4),
    ("gh200", 8, 2, 4),
]


@pytest.mark.parametrize("path", sorted(Path("configs/tunings").rglob("*.yaml")), ids=lambda path: str(path))
def test_tuning_specs_resolve_active_and_archived_base_layers(path: Path) -> None:
    spec, config = read_spec(path, use_local_defaults=False)
    assert spec.base.configs
    assert all((path.parent / base).is_file() for base in spec.base.configs)
    assert config.model.parameter_count > 0


@pytest.mark.parametrize("family,gpus,nodes,per_node", HARDWARE)
@pytest.mark.parametrize("model", ["20m", "50m", "90m", "250m", "500m"])
def test_hardware_presets_and_launch_scripts(
    family: str, gpus: int, nodes: int, per_node: int, model: str, tmp_path: Path
) -> None:
    accumulation = tmp_path / "accumulation.yaml"
    accumulation.write_text("training: {num_accumulation: 2}\n")
    paths = [
        "configs/base.yaml",
        f"configs/models/{model}.yaml",
        "configs/training-512c-32bs-20tpp.yaml",
        str(accumulation),
    ]
    baseline = load_config(paths)
    config = load_config([*paths, f"configs/setups/{family}-{gpus}gpu.yaml"])
    assert config.model == baseline.model and config.training == baseline.training
    assert config.distributed.nodes == nodes and config.distributed.gpus_per_node == per_node
    assert config.distributed.world_size == gpus
    assert config.runtime.execution_backend == family and config.runtime.compile is True
    assert config.distributed.backend == "nccl" and config.distributed.timeout_seconds == 900
    assert config.runtime.cpu_affinity == ("none" if family == "a100" else "exclusive")
    assert config.runtime.cpu_threads == 8
    assert config.micro_batch_size == 32 // (gpus * 2)
    assert isinstance(config.launcher, SlurmLauncher)
    assert config.launcher.partition == "gpu" and config.launcher.time == "00:15:00"
    args = scheduler_arguments(config)
    batch, worker = slurm_scripts(config, tmp_path / "config.yaml", tmp_path / "worker.sh")
    assert f"--nodes={nodes}" in args and f"--gpus-per-node={per_node}" in args
    assert f"--ntasks={nodes} " in batch and f"--gpus-per-task={per_node} " in batch
    assert f"--nnodes={nodes} --nproc-per-node={per_node}" in worker
    assert ("--standalone" in worker) is (nodes == 1)
    assert ("--numa-binding=exclusive" in worker) is (family == "gh200")
    if nodes == 2:
        assert config.launcher.network == "job_vni"
        assert config.launcher.environment_overlays == [Path("scripts/activate_cxi.sh")]
        assert "--network=job_vni" in args and "activate_cxi.sh" in worker
    else:
        assert config.launcher.network is None and config.launcher.environment_overlays == []
        assert not any(arg.startswith("--network=") for arg in args)
        assert "activate_cxi.sh" not in worker


@pytest.mark.parametrize("family,gpus,nodes,per_node", [case for case in HARDWARE if case[2] == 1])
def test_single_node_setup_clears_cross_node_transport(family: str, gpus: int, nodes: int, per_node: int) -> None:
    config = load_config(
        ["configs/base.yaml", "configs/setups/gh200-8gpu.yaml", f"configs/setups/{family}-{gpus}gpu.yaml"]
    )
    assert isinstance(config.launcher, SlurmLauncher)
    assert config.launcher.network is None and config.launcher.environment_overlays == []
    assert config.distributed.nodes == nodes and config.distributed.gpus_per_node == per_node


@pytest.mark.parametrize("family,gpus,nodes,per_node", HARDWARE)
@pytest.mark.parametrize("recipe", ["sync", "awc-ef-8w-ring"])
def test_tuning_setup_and_training_layers(family: str, gpus: int, nodes: int, per_node: int, recipe: str) -> None:
    _, config = read_spec(
        Path(f"configs/tunings/20m-{recipe}-512c-64bs-20tpp.yaml"),
        configs=[
            Path("configs/models/90m.yaml"),
            Path("configs/training-512c-32bs-20tpp.yaml"),
            Path(f"configs/setups/{family}-{gpus}gpu.yaml"),
        ],
        overrides=["model.context_length=1024"],
    )
    assert config.model.width == 640 and config.model.context_length == 1024
    assert config.training.batch_tokens == 16384 and config.training.tokens_per_parameters == 20
    assert config.distributed.nodes == nodes and config.distributed.gpus_per_node == per_node
    assert config.runtime.execution_backend == family
    assert config.distributed.mode == ("ddp" if recipe == "sync" else "decentralized")
    assert config.distributed.error_feedback_gamma == (1.0 if recipe != "sync" else 0.0)
    if recipe != "sync":
        assert config.distributed.scheme == "awc" and config.distributed.topology == "one_peer_ring"
