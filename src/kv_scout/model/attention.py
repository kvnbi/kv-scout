from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.layers import RMSNorm, apply_rope, repeat_kv
from kv_scout.model.sinks import visibility_mask

GATE_OPEN_BIAS = 3.0


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int = 1) -> None:
        super().__init__()
        self.layer = layer
        self.cache_group = cfg.cache_group(layer) if cfg.caches_keys(layer) else layer
        self.owns_cache = self.cache_group == layer
        self.windowed = cfg.windows_attention(layer)
        self.sink_tokens = cfg.sink_tokens
        self.attention_window = cfg.attention_window
        self.n_query_heads = cfg.n_query_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.kv_group_size
        self.attention_scale = cfg.attention_scale

        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        if cfg.normalized_value_residual and layer > 1:
            self.value_mix = nn.Parameter(torch.zeros(1))
        else:
            self.value_mix = None

        if cfg.per_head_gated_attention:
            self.head_gate = nn.Linear(cfg.d_model, cfg.n_query_heads, bias=True)
            nn.init.zeros_(self.head_gate.weight)
            nn.init.constant_(self.head_gate.bias, GATE_OPEN_BIAS)
        else:
            self.head_gate = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        cache=None,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_query_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        source = v
        if self.value_mix is not None and v_first is not None:
            alpha = torch.sigmoid(self.value_mix).to(v.dtype)
            v = (1.0 - alpha) * v + alpha * v_first

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        offset = cache.length if cache is not None else 0
        if cache is None:
            positions = torch.arange(offset, offset + k.shape[2], device=k.device)
        elif self.owns_cache:
            k, v, positions = cache.append(self.cache_group, k, v)
        else:
            k, v, positions = cache.read(self.cache_group)

        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)

        mask = None
        causal = k.shape[2] == q.shape[2]
        if self.windowed:
            mask = visibility_mask(
                positions, t, offset, self.sink_tokens, self.attention_window
            )
            causal = False

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, scale=self.attention_scale
        )

        if self.head_gate is not None:
            gate = torch.sigmoid(self.head_gate(x)).to(out.dtype)
            out = out * gate.transpose(1, 2).unsqueeze(-1)

        out = out.transpose(1, 2).reshape(b, t, self.n_query_heads * self.head_dim)
        return self.o_proj(out), source
