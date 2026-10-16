"""Fused first-order forward over unchanged canonical Llama parameters."""

from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from decentralized_llm.model import Block, Llama

from .gh200_pipeline import selected_norm
from .gh200_rope import packed_rope
from .gh200_swiglu import paired_swiglu


def forward(model: Llama, inputs: Tensor) -> Tensor:
    batch, length = inputs.shape
    cfg = model.config
    blocks = cast(list[Block], list(model.blocks))
    hidden = model.embedding(inputs)
    a100 = model.execution_backend == "a100"
    rows = 8 if cfg.width == 320 and batch * length >= 4096 else 4
    hidden, normalized = selected_norm(
        hidden, None, blocks[0].attention_norm.weight, cfg.norm_eps, rows=rows, a100=a100
    )
    for index, block in enumerate(blocks):
        attn = block.attention
        if model.projection_fusion:
            qkv = F.linear(normalized, torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)))
            if model.fused_rope:
                q, k, v = packed_rope(qkv, attn._rope_cos, attn._rope_sin, cfg.heads)
            else:
                q, k, v = qkv.view(batch, length, 3, cfg.heads, cfg.width // cfg.heads).unbind(2)
                q, k = attn._rotary(q.transpose(1, 2)), attn._rotary(k.transpose(1, 2))
                v = v.transpose(1, 2)
        else:
            q, k, v = [
                p(normalized).view(batch, length, cfg.heads, cfg.width // cfg.heads).transpose(1, 2)
                for p in (attn.q_proj, attn.k_proj, attn.v_proj)
            ]
            q, k = attn._rotary(q), attn._rotary(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        branch = attn.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, cfg.width))
        hidden, normalized = selected_norm(branch, hidden, block.ffn_norm.weight, cfg.norm_eps, rows=rows, a100=a100)
        ffn = block.ffn
        if model.projection_fusion:
            gate_up = F.linear(normalized, torch.cat((ffn.gate_proj.weight, ffn.up_proj.weight)))
            activated = paired_swiglu(gate_up)
        else:
            activated = F.silu(ffn.gate_proj(normalized)) * ffn.up_proj(normalized)
        branch = ffn.down_proj(activated)
        norm = blocks[index + 1].attention_norm if index + 1 < len(blocks) else model.norm
        hidden, normalized = selected_norm(branch, hidden, norm.weight, cfg.norm_eps, rows=rows, a100=a100)
    return F.linear(normalized, model.embedding.weight)
