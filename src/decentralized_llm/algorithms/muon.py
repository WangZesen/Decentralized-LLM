"""Muon for canonical hidden matrices, with auxiliary AdamW and local error feedback."""

from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import nn

from decentralized_llm.config import Config
from decentralized_llm.models.storage import ParameterStorage

_PROJECTIONS = (
    ".attention.q_proj.weight",
    ".attention.k_proj.weight",
    ".attention.v_proj.weight",
    ".attention.out_proj.weight",
    ".ffn.gate_proj.weight",
    ".ffn.up_proj.weight",
    ".ffn.down_proj.weight",
)


def is_muon_parameter(name: str, parameter: nn.Parameter) -> bool:
    """Accept canonical, per-worker hidden projections, never a packed worker axis."""
    return parameter.ndim == 2 and name.startswith("blocks.") and name.endswith(_PROJECTIONS)


def validate_history(previous: Any, expected: torch.Tensor | None) -> None:
    if expected is None:
        if previous is not None:
            raise ValueError("unexpected error_feedback previous_update")
    elif (
        not isinstance(previous, torch.Tensor)
        or previous.layout != torch.strided
        or previous.shape != expected.shape
        or previous.dtype != expected.dtype
    ):
        raise ValueError("missing or incompatible error_feedback previous_update")


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer interface over disjoint Muon and AdamW parameter groups.

    Packed callers supply canonical worker views and apply feedback over the full
    arena themselves. Ordinary callers pass storage to enable local feedback here.
    """

    implementation = "torch_muon_adamw"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        config: Config,
        device: torch.device,
        storage: ParameterStorage | None = None,
    ) -> None:
        named = []
        seen: set[int] = set()
        for name, parameter in named_parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                named.append((name, parameter))
        self.layout = [(name, tuple(p.shape)) for name, p in named]
        partitions: list[list[tuple[str, nn.Parameter]]] = [[], [], []]
        for name, parameter in named:
            index = 0 if is_muon_parameter(name, parameter) else 1 if parameter.ndim >= 2 else 2
            partitions[index].append((name, parameter))
        cfg = config.optimizer
        groups = [
            {
                "params": [p for _, p in partition],
                "param_names": [name for name, _ in partition],
                "optimizer_kind": "muon" if index == 0 else "adamw",
                "weight_decay": cfg.weight_decay if index != 2 else 0.0,
            }
            for index, partition in enumerate(partitions)
        ]
        super().__init__(groups, {})
        self.muon = torch.optim.Muon(
            [self.param_groups[0]],
            lr=cfg.lr * cfg.muon_lr_ratio,
            momentum=cfg.muon_momentum,
            nesterov=True,
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adamw = torch.optim.AdamW(
            self.param_groups[1:],
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            fused=config.runtime.fused_optimizer and device.type == "cuda" and not config.runtime.deterministic,
            foreach=False if config.runtime.deterministic else None,
        )
        self.storage = storage
        self.feedback_gamma = config.distributed.error_feedback_gamma if storage is not None else 0.0
        self.previous_update = torch.zeros_like(storage.flat) if storage is not None and self.feedback_gamma else None
        self.scratch = torch.empty_like(self.previous_update) if self.previous_update is not None else None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def _refresh_state(self) -> None:
        # Child load_state_dict replaces dictionaries; expose the live dictionaries.
        self.param_groups = [*self.muon.param_groups, *self.adamw.param_groups]
        self.state.clear()
        self.state.update(self.muon.state)
        self.state.update(self.adamw.state)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.scratch is not None and self.storage is not None:
            self.scratch.copy_(self.storage.flat)
        self.muon.step()
        self.adamw.step()
        if self.scratch is not None and self.previous_update is not None and self.storage is not None:
            torch.sub(self.storage.flat, self.scratch, out=self.scratch)
            self.storage.flat.add_(self.scratch, alpha=self.feedback_gamma).sub_(
                self.previous_update, alpha=self.feedback_gamma
            )
            torch.autograd.graph.increment_version(self.storage.parameters)
            self.previous_update, self.scratch = self.scratch, self.previous_update
        self._refresh_state()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "muon_adamw",
            "layout": self.layout,
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "previous_update": self.previous_update,
        }

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("version") != 1 or state_dict.get("kind") != "muon_adamw":
            raise ValueError("incompatible Muon/AdamW optimizer kind or version")
        if state_dict.get("layout") != self.layout:
            raise ValueError("incompatible Muon/AdamW parameter layout")
        validate_history(state_dict.get("previous_update"), self.previous_update)
        for key, optimizer in (("muon", self.muon), ("adamw", self.adamw)):
            saved = state_dict.get(key)
            if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
                raise ValueError(f"missing {key} optimizer state")
            groups = saved.get("param_groups")
            if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
                raise ValueError(f"incompatible {key} parameter groups")
            indices = []
            for group, current in zip(groups, optimizer.param_groups, strict=True):
                if (
                    group.get("param_names") != current["param_names"]
                    or group.get("optimizer_kind") != current["optimizer_kind"]
                    or len(group.get("params", [])) != len(current["params"])
                ):
                    raise ValueError(f"incompatible {key} parameter groups")
                for index, parameter in zip(group["params"], current["params"], strict=True):
                    indices.append(index)
                    state = saved["state"].get(index, {})
                    if not state:
                        continue  # Parameters without gradients have no state yet.
                    tensors = ("momentum_buffer",) if key == "muon" else ("exp_avg", "exp_avg_sq")
                    for name in tensors:
                        value = state.get(name)
                        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                            raise ValueError(f"incompatible {key} {name}")
                    if key == "adamw" and (
                        not isinstance(state.get("step"), torch.Tensor) or state["step"].numel() != 1
                    ):
                        raise ValueError("incompatible adamw step")
            if len(set(indices)) != len(indices) or not set(saved["state"]).issubset(indices):
                raise ValueError(f"incompatible {key} state indices")

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        self.muon.load_state_dict(state_dict["muon"])
        # Keep device-dependent flags from construction, including placement of step tensors.
        adamw = {**state_dict["adamw"], "param_groups": []}
        for saved, current in zip(state_dict["adamw"]["param_groups"], self.adamw.param_groups, strict=True):
            adamw["param_groups"].append(
                {**saved, **{key: current[key] for key in ("fused", "foreach", "capturable", "differentiable")}}
            )
        self.adamw.load_state_dict(adamw)
        if self.previous_update is not None:
            self.previous_update.copy_(state_dict["previous_update"])
        self._refresh_state()
