"""Independent hybrid optimizers over canonical packed worker views."""

import torch
from torch import nn

from decentralized_llm.algorithms.muon import MuonAdamW, validate_history
from decentralized_llm.algorithms.packed_adamw import PackedAdamW
from decentralized_llm.config import Config
from decentralized_llm.models.packed import PackedLlama
from decentralized_llm.state import TorchState


class PackedMuonAdamW(PackedAdamW):
    """Reuse packed gradient binding, clipping and arena-level feedback, without Adam moments for Muon."""

    implementation = "torch_muon_adamw"

    def __init__(self, model: PackedLlama, config: Config):
        self.model = model
        self.layout = model.layout
        self._parameters = tuple(model.parameters())
        self._arena = model.parameter_storage
        self.feedback_gamma = config.distributed.error_feedback_gamma
        self.previous_update = torch.zeros_like(self._arena) if self.feedback_gamma != 0 else None
        self.scratch = torch.empty_like(self._arena) if self.previous_update is not None else None
        self.optimizers = []
        self.local_parameters = []
        for worker in range(model.num_models):
            parameters = [nn.Parameter(p[worker].detach(), requires_grad=p.requires_grad) for p in self._parameters]
            named = [(entry.name, parameter) for entry, parameter in zip(self.layout, parameters, strict=True)]
            self.optimizers.append(MuonAdamW(named, config, self._arena.device))
            self.local_parameters.append(parameters)

    def state_dict(self) -> TorchState:
        return {
            "version": 1,
            "kind": "packed_muon_adamw",
            "layout": self.model.layout_metadata(),
            "previous_update": self.previous_update,
            "workers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    @torch.no_grad()
    def load_state_dict(self, state: TorchState) -> None:
        self._check_storage()
        if state.get("version") != 1 or state.get("kind") != "packed_muon_adamw":
            raise ValueError("incompatible packed Muon/AdamW optimizer kind or version")
        if state.get("layout") != self.model.layout_metadata():
            raise ValueError("incompatible packed Muon/AdamW parameter layout")
        validate_history(state.get("previous_update"), self.previous_update)
        workers = state.get("workers")
        if not isinstance(workers, list) or len(workers) != len(self.optimizers):
            raise ValueError("incompatible packed optimizer worker count")
        # Validate every worker before changing any optimizer state.
        for optimizer, saved in zip(self.optimizers, workers, strict=True):
            assert isinstance(optimizer, MuonAdamW)
            optimizer.validate_state_dict(saved)
        for optimizer, saved in zip(self.optimizers, workers, strict=True):
            optimizer.load_state_dict(saved)
        if self.previous_update is not None:
            self.previous_update.copy_(state["previous_update"])
