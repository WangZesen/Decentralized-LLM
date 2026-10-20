"""One canonical initialization, independent of placement and execution policy."""

import torch

from decentralized_llm.config import Config
from decentralized_llm.models.llama import Llama

INITIALIZATION = "shared_seed_v1"


def initialize_model(config: Config) -> Llama:
    """Build CPU FP32 weights without consuming the caller's RNG stream.

    Use the ordinary constructor, including its intermediate random draws, so
    every representation starts with the same canonical tensors. GPU placement
    and fused execution are installed only after initialization.
    """
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(config.runtime.seed).get_state())
            model = Llama(config.model, config.runtime.attention_backend)
    finally:
        torch.set_default_dtype(dtype)
    model.initialization = INITIALIZATION
    return model
