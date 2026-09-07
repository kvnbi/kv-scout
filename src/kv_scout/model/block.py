from __future__ import annotations

import torch
import torch.nn as nn

from kv_scout.config import ModelConfig
from kv_scout.model.attention import GroupedQueryAttention
from kv_scout.model.layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.kind = cfg.layer_kind(layer)
        self.uses_rope = True

        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden(layer))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.uses_rope:
            cos = sin = None
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
