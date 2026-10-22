"""Strict YAML configuration, shared by the CLI and Python API."""

import os
import sys
import tempfile
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

LOCAL_DEFAULTS_PATH = Path("configs/local/defaults.yaml")
LOCAL_DEFAULTS_TEMPLATE = """# Local execution defaults, automatically loaded from the repository working directory.
# Save as Git-ignored configs/local/defaults.yaml; experiment files and explicit overrides take precedence.
# Set execution_backend to gh200 or a100 to enable fused execution on that hardware.
# compile: auto keeps native execution eager and enables compilation for fused backends.
runtime:
  execution_backend: native
  compile: auto
  cpu_affinity: none
"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelConfig(StrictModel):
    vocab_size: int = Field(default=32000, ge=2, le=65536)
    layers: int = Field(default=8, gt=0)
    width: int = Field(default=320, gt=0)
    heads: int = Field(default=5, gt=0)
    ffn_width: int = Field(default=896, gt=0)
    context_length: int = Field(default=1024, gt=0)
    rope_theta: float = Field(default=10000.0, gt=0)
    norm_eps: float = Field(default=1e-5, gt=0)
    init_std: float = Field(default=0.02, gt=0)

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into heads with an even head dimension")
        return self

    @property
    def parameter_count(self) -> int:
        d = self.width
        return self.vocab_size * d + self.layers * (4 * d * d + 3 * d * self.ffn_width + 2 * d) + d


PRESETS = {
    "20m": dict(layers=8, width=320, heads=5, ffn_width=896),
    "50m": dict(layers=10, width=512, heads=8, ffn_width=1408),
    "90m": dict(layers=14, width=640, heads=10, ffn_width=1792),
    "250m": dict(layers=20, width=960, heads=15, ffn_width=2560),
    "500m": dict(layers=24, width=1280, heads=20, ffn_width=3456),
}


class DataConfig(StrictModel):
    cache_dir: Path = Path("data/c4")
    dataset: str = "allenai/c4"
    revision: str = "main"
    tokenizer: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer_revision: str = "main"
    shuffle_seed: int = Field(default=42, ge=0, lt=2**32)
    shuffle_buffer: int = Field(default=100_000, gt=0)
    shard_tokens: int = Field(default=33_554_432, gt=0)
    tokenize_batch_size: int = Field(default=256, gt=0)
    prepare_workers: int = Field(default=8, gt=0, strict=True)
    # Runtime buffering; separate from the document shuffle used during preparation.
    loader_mode: Literal["auto", "resident", "streaming"] = "auto"
    reader_threads: int = Field(default=4, gt=0, le=64)
    prefetch_batches: int = Field(default=8, gt=0)
    host_memory_mib: int = Field(default=2048, gt=0)
    read_range_bytes: int = Field(default=1048576, ge=2)
    coalesce_gap_bytes: int = Field(default=65536, ge=0)
    max_read_amplification: float = Field(default=2.0, ge=1)
    preload_max_read_amplification: float = Field(default=16.0, ge=1)
    # Overrides for preparation/smoke runs; None means the complete validation split.
    prepare_train_tokens: int | None = Field(default=None, gt=0)
    prepare_validation_tokens: int | None = Field(default=None, gt=0)


class OptimizerConfig(StrictModel):
    lr: float = Field(default=1e-3, gt=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.95, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    grad_clip: float | None = Field(default=1.0, gt=0)


class CosineScheduleConfig(StrictModel):
    name: Literal["cosine"] = "cosine"
    warmup_tokens_per_parameters: float = Field(default=1.0, ge=0)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1)


class WSDScheduleConfig(StrictModel):
    name: Literal["wsd"] = "wsd"
    warmup_tokens_per_parameters: float = Field(default=1.0, ge=0)
    decay_fraction: float = Field(default=0.1, ge=0, le=1)


class TrainingConfig(StrictModel):
    tokens_per_parameters: float = Field(default=20.0, gt=0)
    epoch_tokens_per_parameters: float = Field(default=0.5, gt=0)
    data_reuse_factor: int = Field(default=1, gt=0, strict=True)
    batch_tokens: int = Field(default=32768, gt=0)
    num_accumulation: int = Field(default=1, gt=0, strict=True)
    checkpoint_policy: Literal["interval", "explicit", "final", "none"] = "final"
    checkpoint_epochs: list[Annotated[int, Field(strict=True, gt=0)]] = Field(default_factory=lambda: [1], min_length=1)
    save_epoch_training_state: bool = True
    log_every: int = Field(default=50, gt=0)

    @model_validator(mode="before")
    @classmethod
    def reject_micro_batch_size(cls, value: object) -> object:
        if isinstance(value, dict) and "micro_batch_size" in value:
            raise ValueError(
                "training.micro_batch_size has been replaced by training.num_accumulation (default 1). "
                "To preserve batching, set num_accumulation = batch_tokens / "
                "(context_length * workers * old_micro_batch_size), which must be a positive integer."
            )
        return value

    @field_validator("checkpoint_epochs", mode="before")
    @classmethod
    def normalize_epochs(cls, value: object) -> object:
        return [value] if isinstance(value, int) else value

    @field_validator("checkpoint_epochs")
    @classmethod
    def unique_epochs(cls, value: list[int], info: ValidationInfo) -> list[int]:
        return sorted(set(value)) if info.data.get("checkpoint_policy") == "explicit" else value

    @property
    def epoch_count(self) -> int:
        count = Fraction(str(self.tokens_per_parameters)) / Fraction(str(self.epoch_tokens_per_parameters))
        if count.denominator != 1:
            raise ValueError("tokens_per_parameters must be divisible by epoch_tokens_per_parameters")
        return count.numerator

    @model_validator(mode="after")
    def schedule(self) -> Self:
        epochs = self.epoch_count
        if self.checkpoint_policy == "interval" and len(self.checkpoint_epochs) != 1:
            raise ValueError("interval checkpoint policy requires exactly one epoch interval")
        if self.checkpoint_policy == "explicit" and self.checkpoint_epochs[-1] > epochs:
            raise ValueError("checkpoint epoch exceeds the training epoch count")
        return self

    def saved_epochs(self) -> frozenset[int]:
        if self.checkpoint_policy == "interval":
            interval = self.checkpoint_epochs[0]
            return frozenset(range(interval, self.epoch_count + 1, interval))
        if self.checkpoint_policy == "explicit":
            return frozenset(self.checkpoint_epochs)
        return frozenset()


class EvaluationConfig(StrictModel):
    fp32_precision: Literal["ieee", "tf32"] | None = None
    subset_blocks: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=128, gt=0)
    seed: int = Field(default=12345, ge=0, lt=2**32)


class RuntimeConfig(StrictModel):
    fp32_precision: Literal["ieee", "tf32"] | None = None
    training_backend: Literal["distributed", "packed"] = "distributed"
    seed: int = Field(default=42, ge=0, lt=2**32)
    deterministic: bool = False
    device: Literal["cpu", "cuda"] = "cuda"
    amp: bool = True
    execution_backend: Literal["auto", "native", "gh200", "a100"] = "auto"
    compile: bool | Literal["auto"] = "auto"
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_cache_dir: Path = Path(".cache/compile")
    compile_scope: Literal["model", "blocks"] = "model"
    projection_fusion: bool = True
    fused_rope: bool | Literal["auto"] = "auto"
    sdpa_backend: Literal["auto", "flash", "cudnn", "math"] = "auto"
    cpu_affinity: Literal["none", "local", "exclusive"] = "none"
    fused_optimizer: bool = True
    decentralized_optimizer: Literal["auto", "reference"] = "auto"
    attention_backend: Literal["sdpa", "reference"] = "sdpa"
    output_dir: Path = Path("runs/default")
    cpu_threads: int = Field(default=8, gt=0)


class DistributedConfig(StrictModel):
    mode: Literal["ddp", "decentralized"] = "ddp"
    nodes: int = Field(default=1, gt=0)
    gpus_per_node: int = Field(default=1, gt=0)
    backend: Literal["nccl", "gloo"] = "nccl"
    timeout_seconds: int = Field(default=300, gt=0)
    rendezvous_port: int = Field(default=29500, ge=1024, le=65535)
    topology: Literal["complete", "one_peer_ring", "one_peer_exponential"] = "complete"
    scheme: Literal["awc", "atc"] = "awc"
    error_feedback_gamma: float = Field(default=0.0, strict=True, allow_inf_nan=False)
    ddp_bucket_cap_mb: int = Field(default=25, gt=0)
    ddp_gradient_as_bucket_view: bool = False
    ddp_batched_grad_copy: bool = False
    reuse_mix_buffers: bool = True

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @model_validator(mode="after")
    def topology_size(self) -> Self:
        size = self.world_size
        if self.topology == "one_peer_ring" and size > 1 and size % 2:
            raise ValueError("one_peer_ring requires an even world size (or 1)")
        if self.topology == "one_peer_exponential" and size & (size - 1):
            raise ValueError("one_peer_exponential requires a power-of-two world size")
        return self


class LocalLauncher(StrictModel):
    kind: Literal["local"] = "local"


class BenchmarkConfig(StrictModel):
    warmup: int = Field(default=20, ge=1)
    steps: int = Field(default=40, ge=1)
    windows: int = Field(default=3, ge=1)
    min_seconds: float = Field(default=2.0, gt=0)
    profile: bool = False
    wait_for_resident: bool = True


class SlurmLauncher(StrictModel):
    kind: Literal["slurm"] = "slurm"
    partition: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    time: str = Field(default="01:00:00", pattern=r"^(\d+-)?\d+(:\d+){0,2}$")
    cpus_per_task: int | None = Field(default=None, gt=0)
    memory: str | None = Field(default=None, pattern=r"^\d+[KMGT]?$", min_length=1)
    gpu_type: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    constraint: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.&|\[\]-]+$")
    qos: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    network: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_=,.-]+$")
    environment_script: Path | None = None
    environment_overlays: list[Path] = Field(default_factory=list)


class Config(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_schedule: Annotated[CosineScheduleConfig | WSDScheduleConfig, Field(discriminator="name")] = Field(
        default_factory=CosineScheduleConfig
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    launcher: Annotated[LocalLauncher | SlurmLauncher, Field(discriminator="kind")] = Field(
        default_factory=LocalLauncher
    )

    @property
    def physical_nodes(self) -> int:
        return 1 if self.runtime.training_backend == "packed" else self.distributed.nodes

    @property
    def processes_per_node(self) -> int:
        return 1 if self.runtime.training_backend == "packed" else self.distributed.gpus_per_node

    @property
    def physical_world_size(self) -> int:
        return self.physical_nodes * self.processes_per_node

    @property
    def physical_gpu_count(self) -> int:
        return self.physical_world_size if self.runtime.device == "cuda" else 0

    @property
    def process_micro_batch_size(self) -> int:
        multiplier = self.distributed.world_size if self.runtime.training_backend == "packed" else 1
        return self.micro_batch_size * multiplier

    @property
    def micro_batch_size(self) -> int:
        """Sequences per rank and backward, derived from the current global recipe."""
        divisor = self.model.context_length * self.distributed.world_size * self.training.num_accumulation
        batch, remainder = divmod(self.training.batch_tokens, divisor)
        if batch < 1 or remainder:
            raise ValueError(
                f"batch_tokens={self.training.batch_tokens} must be divisible by "
                "context_length * workers * num_accumulation "
                f"({self.model.context_length} * {self.distributed.world_size} * "
                f"{self.training.num_accumulation} = {divisor}) "
                "and provide at least one sequence per rank per accumulation step"
            )
        return batch

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if self.launcher.kind == "local" and self.physical_nodes != 1:
            raise ValueError("local launch supports exactly one node")
        if self.runtime.training_backend == "packed" and self.runtime.compile_scope == "blocks":
            raise ValueError("packed execution supports compile_scope=model only")
        _ = self.micro_batch_size
        if (
            self.runtime.training_backend == "distributed"
            and self.runtime.device == "cpu"
            and self.distributed.backend != "gloo"
        ):
            raise ValueError("CPU runs require the gloo communication backend")
        if self.runtime.device == "cpu" and self.runtime.cpu_affinity == "exclusive":
            raise ValueError("exclusive CPU affinity requires GPU placement through torchrun")
        precisions = (self.runtime.fp32_precision, self.evaluation.fp32_precision)
        if any(value is not None for value in precisions) and self.runtime.amp:
            raise ValueError("explicit FP32 precision requires runtime.amp=false")
        if "tf32" in precisions and self.runtime.device != "cuda":
            raise ValueError("TF32 precision requires CUDA")
        return self


def _merge_config(target: dict, source: dict, *, root: bool = True) -> None:
    for key, value in source.items():
        previous = target.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            if (
                root
                and key in ("lr_schedule", "launcher")
                and previous.get("name", previous.get("kind")) != value.get("name", value.get("kind"))
                and ("name" in value or "kind" in value)
            ):
                previous.clear()
            _merge_config(previous, value, root=False)
        else:
            target[key] = value


def initialize_local_defaults(*, dry_run: bool = False) -> None:
    """Publish a complete default file once; concurrent controllers never replace it."""
    path = LOCAL_DEFAULTS_PATH
    if path.exists() or path.is_symlink():
        return
    if dry_run:
        print(
            f"Local execution defaults missing: {path}; using native, compile=auto, cpu_affinity=none "
            "in memory. Experiment settings and overrides take precedence; no file was created.",
            file=sys.stderr,
        )
        return
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(LOCAL_DEFAULTS_TEMPLATE)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return
        print(f"Created local execution defaults: {path} (native backend).", file=sys.stderr)
    except OSError as error:
        raise ValueError(
            f"Could not initialize local execution defaults at {path}: {error}. "
            "Use --no-local-config to bypass local defaults."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _local_defaults() -> dict:
    path = LOCAL_DEFAULTS_PATH
    try:
        try:
            text = path.read_text()
        except FileNotFoundError:
            if path.is_symlink():
                raise
            text = LOCAL_DEFAULTS_TEMPLATE
        value = yaml.safe_load(text)
        if value is None:
            return {}
        if not isinstance(value, dict) or value.keys() - {"runtime"}:
            raise ValueError("expected a YAML mapping containing only runtime settings")
        runtime = value.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be a YAML mapping")
        if {"seed", "output_dir"} & runtime.keys():
            raise ValueError("runtime.seed and runtime.output_dir belong in experiment configurations")
        return {"runtime": RuntimeConfig.model_validate(runtime).model_dump(mode="json", exclude_unset=True)}
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"Invalid local execution defaults at {path}: {error}") from error


def load_config(
    path: str | Path | Sequence[str | Path] | None = None,
    overrides: Sequence[str] = (),
    *,
    use_local_defaults: bool = False,
) -> Config:
    """Merge optional local defaults, YAML files, then dotted overrides; never write files."""
    paths = () if path is None else (path,) if isinstance(path, (str, Path)) else path
    raw: dict = Config().model_dump(mode="json")
    if use_local_defaults:
        _merge_config(raw, _local_defaults())
    for filename in paths:
        value = yaml.safe_load(Path(filename).read_text())
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"configuration must be a YAML mapping: {filename}")
        _merge_config(raw, value)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or not key or any(not part for part in key.split(".")):
            raise ValueError(f"expected dotted.key=value, got {override!r}")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into scalar configuration: {key}")
        parsed = yaml.safe_load(value)
        if parts in (["lr_schedule", "name"], ["launcher", "kind"]) and node.get(parts[-1], parsed) != parsed:
            node.clear()
        node[parts[-1]] = parsed
    return Config.model_validate(raw)


def save_config(config: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
