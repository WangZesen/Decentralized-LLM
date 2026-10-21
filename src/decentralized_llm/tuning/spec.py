"""Validated search specifications and pure Cartesian candidate expansion."""

import copy
import itertools
import math
from collections.abc import Sequence
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import Field, TypeAdapter, model_validator

from decentralized_llm.config import Config, CosineScheduleConfig, StrictModel, WSDScheduleConfig, load_config
from decentralized_llm.data.cache import fingerprint
from decentralized_llm.launch.commands import absolute_config
from decentralized_llm.models.initialization import INITIALIZATION

type Record = dict[str, Any]
type Scalar = str | int | float | bool | None


class Interval(StrictModel):
    scale: Literal["linear", "log"]
    min: float
    max: float
    points: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        check_bounds(self.min, self.max, self.points)
        if self.scale == "log" and self.min <= 0:
            raise ValueError("log bounds must be positive")
        return self


class PowerInterval(StrictModel):
    scale: Literal["power"]
    base: float = Field(gt=0)
    exponent_scale: Literal["linear", "log"] = "linear"
    exponent_min: float
    exponent_max: float
    points: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        check_bounds(self.exponent_min, self.exponent_max, self.points)
        if self.exponent_scale == "log" and self.exponent_min <= 0 <= self.exponent_max:
            raise ValueError("log exponent bounds must be nonzero and have the same sign")
        return self


type Search = list[Scalar] | Annotated[Interval | PowerInterval, Field(discriminator="scale")]


def check_bounds(low: float, high: float, points: int) -> None:
    if (points == 1 and low != high) or (points > 1 and low >= high):
        raise ValueError("interval requires min < max, or equal bounds for one point")


def linear_points(low: float, high: float, count: int) -> list[float]:
    if count == 1:
        return [low]
    start, end = Decimal(str(low)), Decimal(str(high))
    return [float(start + (end - start) * i / (count - 1)) for i in range(count)]


def log_points(low: float, high: float, count: int) -> list[float]:
    if count == 1:
        return [low]
    exponents = linear_points(math.log10(abs(low)), math.log10(abs(high)), count)
    values = [float(format(math.copysign(10**exponent, low), ".15g")) for exponent in exponents[1:-1]]
    return [low, *values, high]


def grid(search: Search) -> list[Record]:
    if isinstance(search, list):
        if not search:
            raise ValueError("search lists must be nonempty")
        result = [{"value": value} for value in search]
    elif isinstance(search, PowerInterval):
        result = []
        spacing = log_points if search.exponent_scale == "log" else linear_points
        for exponent in spacing(search.exponent_min, search.exponent_max, search.points):
            try:
                value = search.base**exponent
            except OverflowError as error:
                raise ValueError("power range overflows") from error
            if value == 0 or not math.isfinite(value):
                raise ValueError("power range overflows or underflows")
            result.append(
                {
                    "value": float(format(value, ".15g")),
                    "expression": f"{search.base:.15g}^{exponent:.15g}",
                    "base": search.base,
                    "exponent": exponent,
                }
            )
    else:
        if search.scale == "linear":
            values = linear_points(search.min, search.max, search.points)
        else:
            values = log_points(search.min, search.max, search.points)
        result = [{"value": value} for value in values]
    unique: list[Record] = []
    for point in result:
        value = point["value"]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("search values must be finite")
        if not any(
            isinstance(item["value"], bool) == isinstance(value, bool) and item["value"] == value for item in unique
        ):
            unique.append(point)
    return unique


class Phase(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    initial_values: dict[str, Scalar] = Field(default_factory=dict)
    search: dict[str, Search] = Field(min_length=1)
    top_k: int = Field(ge=1, strict=True)


class Base(StrictModel):
    configs: list[Path] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)


class Confirmation(StrictModel):
    seeds: int = Field(ge=1, strict=True)


class Controller(StrictModel):
    max_jobs: int = Field(default=50, ge=1, strict=True)
    jobs_per_allocation: int = Field(default=1, ge=1, strict=True)
    max_retries: int = Field(default=2, ge=0, strict=True)
    poll_seconds: float = Field(default=30, gt=0)
    result_grace_seconds: float = Field(default=120, ge=0)


class TuningSpec(StrictModel):
    base: Base = Field(default_factory=Base)
    seed_start: int = Field(default=42, ge=0, lt=2**32, strict=True)
    screening_seeds: int = Field(ge=1, strict=True)
    phases: list[Phase] = Field(min_length=1)
    confirmation: Confirmation
    controller: Controller = Field(default_factory=Controller)

    @model_validator(mode="after")
    def phases_valid(self) -> Self:
        if self.confirmation.seeds < self.screening_seeds:
            raise ValueError("confirmation seeds must include all screening seeds")
        if self.seed_start + self.confirmation.seeds > 2**32:
            raise ValueError("seed sequence exceeds the training seed range")
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)) or "confirmation" in names:
            raise ValueError("phase names must be unique and cannot be confirmation")
        fields = set().union(*(phase.search for phase in self.phases))
        for field in fields:
            if (
                len(field.split(".")) != 2
                or any(not part for part in field.split("."))
                or field.startswith("launcher.")
                or field in {"runtime.seed", "runtime.output_dir", "data.cache_dir"}
            ):
                raise ValueError(f"reserved or invalid search field: {field}")
        previous: set[str] = set()
        for phase in self.phases:
            required = fields - previous - phase.search.keys()
            if set(phase.initial_values) != required:
                raise ValueError(f"{phase.name}: initial_values must specify exactly {sorted(required)}")
            for search in phase.search.values():
                grid(search)
            previous.update(phase.search)
        return self

    def seeds(self, confirmation: bool = False) -> list[int]:
        count = self.confirmation.seeds if confirmation else self.screening_seeds
        return list(range(self.seed_start, self.seed_start + count))


@cache
def field_adapter(owner: type[StrictModel], name: str) -> TypeAdapter:
    field = owner.model_fields.get(name)
    if field is None:
        raise ValueError(f"unknown configuration field: {owner.__name__}.{name}")
    return TypeAdapter(field.rebuild_annotation())


def field_value(config: Config, key: str, value: object) -> Any:
    parts = key.split(".")
    if len(parts) != 2 or parts[0] not in Config.model_fields:
        raise ValueError(f"invalid configuration field: {key}")
    section, name = parts
    if key == "lr_schedule.name":
        return TypeAdapter(Literal["cosine", "wsd"]).validate_python(value)
    if key == "launcher.kind":
        return TypeAdapter(Literal["local", "slurm"]).validate_python(value)
    owner = type(getattr(config, section))
    if section == "lr_schedule" and name not in owner.model_fields:
        owner = WSDScheduleConfig if name in WSDScheduleConfig.model_fields else CosineScheduleConfig
    field = owner.model_fields.get(name)
    if field is not None and field.annotation is int and isinstance(value, float) and value.is_integer():
        value = int(value)
    normalized = field_adapter(owner, name).validate_python(value)
    return 0.0 if isinstance(normalized, float) and normalized == 0 else normalized


def search_points(config: Config, key: str, search: Search) -> list[Record]:
    points: list[Record] = []
    for point in grid(search):
        normalized = {**point, "value": field_value(config, key, point["value"])}
        if not any(previous["value"] == normalized["value"] for previous in points):
            points.append(normalized)
    return points


def validate_fields(spec: TuningSpec, base: Config) -> None:
    for phase in spec.phases:
        for key, search in phase.search.items():
            search_points(base, key, search)
        for key, value in phase.initial_values.items():
            field_value(base, key, value)


def apply_values(config: Config, values: Record) -> Config:
    raw = config.model_dump(mode="json")
    # Change a tagged variant before assigning its other fields, independently of YAML key order.
    ordered = sorted(values, key=lambda key: (key not in {"lr_schedule.name", "launcher.kind"}, key))
    for key in ordered:
        parts = key.split(".")
        if len(parts) != 2 or parts[0] not in raw:
            raise ValueError(f"invalid configuration field: {key}")
        section, field = parts
        if key in {"lr_schedule.name", "launcher.kind"} and raw[section].get(field) != values[key]:
            if key == "lr_schedule.name":
                variant = WSDScheduleConfig if values[key] == "wsd" else CosineScheduleConfig
                # Shared base settings, such as warmup, remain fixed while incompatible variant fields disappear.
                raw[section] = {name: value for name, value in raw[section].items() if name in variant.model_fields}
            else:
                raw[section] = {}
        raw[section][field] = field_value(config, key, values[key])
    return Config.model_validate(raw)


def read_spec(
    path: Path,
    *,
    configs: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    use_local_defaults: bool = False,
) -> tuple[TuningSpec, Config]:
    """Resolve spec files, extra files, spec overrides, then explicit overrides; never write files."""
    spec = TuningSpec.model_validate(yaml.safe_load(path.read_text()))
    paths = [(path.parent / filename).resolve() for filename in spec.base.configs]
    paths.extend(Path(filename).resolve() for filename in configs)
    spec_overrides = [
        f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}"
        for key, value in sorted(
            spec.base.overrides.items(), key=lambda item: item[0] not in {"lr_schedule.name", "launcher.kind"}
        )
    ]
    base = absolute_config(load_config(paths, [*spec_overrides, *overrides], use_local_defaults=use_local_defaults))
    spec.base.configs = paths
    validate_fields(spec, base)
    return spec, base


def numeric_recipe(config: Config) -> Record:
    raw = config.model_dump(mode="json")
    raw["initialization"] = INITIALIZATION
    raw.pop("launcher")
    raw["runtime"].pop("output_dir")
    raw["runtime"].pop("seed")
    raw["data"].pop("cache_dir")
    return raw


def expand(base: Config, phase: Phase, parents: list[Record] | None = None) -> list[Record]:
    candidates: dict[str, Record] = {}
    keys = list(phase.search)
    grids = [search_points(base, key, search) for key, search in phase.search.items()]
    for parent in parents if parents is not None else [{"values": {}, "labels": {}}]:
        for points in itertools.product(*grids):
            values = {**parent["values"], **{key: point["value"] for key, point in zip(keys, points, strict=True)}}
            config = apply_values(base, {**phase.initial_values, **values})
            # Record coerced values, so integral ranges and explicit integers share identities.
            raw = config.model_dump(mode="json")
            values = {key: raw[key.split(".")[0]][key.split(".")[1]] for key in values}
            labels = copy.deepcopy(parent["labels"])
            for key, point in zip(keys, points, strict=True):
                labels.pop(key, None)
                if "expression" in point:
                    labels[key] = {**point, "value": values[key]}
            identity = fingerprint(numeric_recipe(config))
            candidates.setdefault(identity, {"id": identity, "values": values, "labels": labels, "config": raw})
    # Later phases can have fewer parents after divergent candidates are eliminated.
    if parents is None and len(candidates) < phase.top_k:
        raise ValueError(f"{phase.name}: top_k exceeds the number of distinct candidates")
    return list(candidates.values())


def trial_id(candidate: Record, seed: int, provenance: Record) -> str:
    return fingerprint({"candidate": candidate["id"], "seed": seed, "provenance": provenance})
