from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.layers import apply_rope, repeat_kv


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_query_heads = cfg.n_query_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.kv_group_size

        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_query_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, self.n_query_heads * self.head_dim)
        return self.o_proj(out)
