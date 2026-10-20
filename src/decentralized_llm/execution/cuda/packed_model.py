"""Fused CUDA forward over canonical packed parameters, for GH200 and A100."""

from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from decentralized_llm.models.llama import Block
from decentralized_llm.models.packed import PackedLlama, linear

from .pipeline import selected_norm
from .rope import packed_rope
from .swiglu import paired_swiglu


def forward(model: PackedLlama, inputs: Tensor) -> Tensor:
    n, batch, length = inputs.shape
    cfg = model.config
    workers = torch.arange(n, device=inputs.device)[:, None, None]
    hidden = model.embedding.weight[workers, inputs]
    blocks = cast(list[Block], list(model.blocks))
    a100 = model.execution_backend == "a100"
    rows = 8 if cfg.width == 320 and batch * length >= 4096 else 4
    hidden, normalized = selected_norm(
        hidden, None, blocks[0].attention_norm.weight, cfg.norm_eps, rows=rows, a100=a100
    )
    for index, block in enumerate(blocks):
        attn = block.attention
        if model.projection_fusion:
            weight = torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=1)
            qkv = linear(normalized, weight)
            if model.fused_rope:
                q, k, v = packed_rope(qkv.flatten(0, 1), attn._rope_cos, attn._rope_sin, cfg.heads)
            else:
                q, k, v = qkv.reshape(n * batch, length, 3, cfg.heads, cfg.width // cfg.heads).unbind(2)
                q, k = attn._rotary(q.transpose(1, 2)), attn._rotary(k.transpose(1, 2))
                v = v.transpose(1, 2)
        else:
            q, k, v = [
                linear(normalized, projection.weight)
                .reshape(n * batch, length, cfg.heads, cfg.width // cfg.heads)
                .transpose(1, 2)
                for projection in (attn.q_proj, attn.k_proj, attn.v_proj)
            ]
            q, k = attn._rotary(q), attn._rotary(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        branch = linear(attended.transpose(1, 2).contiguous().view(n, batch, length, cfg.width), attn.out_proj.weight)
        hidden, normalized = selected_norm(branch, hidden, block.ffn_norm.weight, cfg.norm_eps, rows=rows, a100=a100)
        ffn = block.ffn
        if model.projection_fusion:
            activated = paired_swiglu(linear(normalized, torch.cat((ffn.gate_proj.weight, ffn.up_proj.weight), dim=1)))
        else:
            activated = F.silu(linear(normalized, ffn.gate_proj.weight)) * linear(normalized, ffn.up_proj.weight)
        branch = linear(activated, ffn.down_proj.weight)
        norm = blocks[index + 1].attention_norm if index + 1 < len(blocks) else model.norm
        hidden, normalized = selected_norm(branch, hidden, norm.weight, cfg.norm_eps, rows=rows, a100=a100)
    return linear(normalized, model.embedding.weight)
