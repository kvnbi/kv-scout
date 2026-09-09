from __future__ import annotations

import math

import torch
import torch.nn as nn

from kv_scout.config import ModelConfig
from kv_scout.model.attention import GroupedQueryAttention
from kv_scout.model.gdn import GatedDeltaNet
from kv_scout.model.layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.kind = cfg.layer_kind(layer)
        self.uses_rope = not (cfg.nope_on_anchor_layers and self.kind == "anchor")

        self.norm_scale = (
            1.0 / math.sqrt(layer) if cfg.layernorm_scaling else 1.0
        )
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, self.norm_scale)
        self.is_linear = self.kind == "linear"
        self.attn = (
            GatedDeltaNet(cfg, layer)
            if self.is_linear
            else GroupedQueryAttention(cfg, layer)
        )
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, self.norm_scale)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden(layer))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        cache=None,
    ):
        if not self.uses_rope:
            cos = sin = None
        result = self.attn(self.attn_norm(x), cos, sin, v_first, cache)
        attended, source = result[0], result[1]
        x = x + attended
        x = x + self.ffn(self.ffn_norm(x))
        return x, source
