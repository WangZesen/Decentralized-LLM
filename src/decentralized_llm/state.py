"""Shared types for tensor batches and serialized training state."""

from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

type Batch = tuple[Tensor, Tensor]
type LossFunction = Callable[[Tensor, Tensor], Tensor]
type TokenArray = NDArray[np.uint16]
type IndexArray = NDArray[np.int64]
# PyTorch state dictionaries and Pydantic's serialized configuration are heterogeneous.
type TorchState = dict[str, Any]


class RNGState(TypedDict):
    python: tuple[Any, ...]
    numpy: tuple[str, NDArray[np.uint32], int, int, float]
    torch: Tensor
    cuda: list[Tensor] | None


class EvaluationResult(TypedDict):
    loss: float
    perplexity: float | None
    tokens: int
    seconds: float
    split: str
    validation_complete: bool


class Shard(TypedDict):
    file: str
    tokens: int
    sha256: str


class SplitManifest(TypedDict):
    tokens: int
    shards: list[Shard]


class CacheContent(TypedDict):
    version: int
    dataset: str
    dataset_revision: str
    tokenizer: str
    tokenizer_revision: str
    tokenizer_fingerprint: str
    vocab_size: int
    eos_token_id: int
    dtype: str
    preprocessing: dict[str, int | bool]
    validation_complete: bool
    splits: dict[str, SplitManifest]


class CacheManifest(CacheContent):
    identity: str
