"""Aligned live parameter storage, prepared before optimizers and execution wrappers."""

import weakref
from dataclasses import dataclass

import torch
from torch import nn

PARAMETER_ALIGNMENT = 256


@dataclass
class ParameterStorage:
    flat: torch.Tensor
    parameters: tuple[nn.Parameter, ...]
    offsets: tuple[int, ...]
    shapes: tuple[torch.Size, ...]
    active_mixes: int = 0

    def matches(self, parameters: tuple[nn.Parameter, ...]) -> bool:
        """Validate at setup boundaries, never in the communication hot path."""
        if len(parameters) != len(self.parameters):
            return False
        pointer = self.flat.untyped_storage().data_ptr()
        return all(
            actual is expected
            and actual.layout == torch.strided
            and actual.is_contiguous()
            and actual.shape == shape
            and actual.dtype == self.flat.dtype
            and actual.device == self.flat.device
            and actual.untyped_storage().data_ptr() == pointer
            and actual.storage_offset() == self.flat.storage_offset() + offset
            for actual, expected, offset, shape in zip(
                parameters, self.parameters, self.offsets, self.shapes, strict=True
            )
        )

    @torch.no_grad()
    def copy_from(self, source: torch.Tensor) -> None:
        self.flat.copy_(source)
        # set_ shares storage, but each Parameter retains its own version counter.
        # Notify autograd about writes through the flat alias in one batched call.
        torch.autograd.graph.increment_version(self.parameters)


_parameter_storages: weakref.WeakKeyDictionary[nn.Module, ParameterStorage] = weakref.WeakKeyDictionary()


@torch.no_grad()
def prepare_parameter_storage(model: nn.Module) -> ParameterStorage:
    """Pack once after placement/conversion, preserving Parameter objects and ties.

    Call again after replacing parameters or changing device/dtype, before creating
    an optimizer or execution wrapper. Never change placement during an exchange or
    between a forward and its backward. Runtime metadata stays outside state_dict
    and deepcopy; copied models must be prepared independently.
    """
    parameters = tuple(model.parameters())  # Deduplicates actual tied Parameters.
    previous = _parameter_storages.get(model)
    if previous is not None:
        if previous.matches(parameters):
            return previous
        if previous.active_mixes:
            raise RuntimeError("cannot rebuild parameter storage during an active exchange")
    if not parameters:
        raise ValueError("parameter storage requires a nonempty model")
    first = parameters[0]
    intervals: dict[tuple[torch.device, int], list[tuple[int, int]]] = {}
    for parameter in parameters:
        if (
            parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or not parameter.is_floating_point()
            or parameter.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("parameter storage requires dense contiguous floating-point CPU or CUDA parameters")
        if parameter.device != first.device or parameter.dtype != first.dtype:
            raise ValueError("parameter storage requires one device and dtype")
        if parameter.numel():
            key = (parameter.device, parameter.untyped_storage().data_ptr())
            start = int(parameter.storage_offset())
            intervals.setdefault(key, []).append((start, start + parameter.numel()))
    for spans in intervals.values():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:], strict=False)):
            raise ValueError("distinct Parameters must not overlap; tie weights with the same Parameter object")
    alignment = PARAMETER_ALIGNMENT // first.element_size()
    offsets = []
    length = 0
    for parameter in parameters:
        length = (length + alignment - 1) // alignment * alignment
        offsets.append(length)
        length += parameter.numel()
    # CPU allocations need not be 256-byte aligned. Reserve room to align the
    # base as well as each slice; this also works with CUDA's aligned allocator.
    allocation = torch.zeros(length + alignment - 1, dtype=first.dtype, device=first.device)
    base = (-allocation.data_ptr() % PARAMETER_ALIGNMENT) // first.element_size()
    flat = allocation.narrow(0, base, length)
    for parameter, offset in zip(parameters, offsets, strict=True):
        view = flat.narrow(0, offset, parameter.numel()).view_as(parameter)
        view.copy_(parameter)
        parameter.set_(flat.untyped_storage(), base + offset, parameter.size(), parameter.stride())
    storage = ParameterStorage(flat, parameters, tuple(offsets), tuple(p.shape for p in parameters))
    _parameter_storages[model] = storage
    return storage


def parameter_storage(model: nn.Module) -> ParameterStorage:
    """Constant-time runtime lookup; direct mixing callers prepare on first use."""
    storage = _parameter_storages.get(model)
    return storage if storage is not None else prepare_parameter_storage(model)
