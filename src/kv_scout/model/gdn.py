from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.layers import RMSNorm, apply_rope, repeat_kv

DECAY_OPEN_BIAS = 3.0


def delta_rule_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    write: torch.Tensor,
    state: torch.Tensor | None = None,
):
    batch, heads, length, key_dim = key.shape
    value_dim = value.shape[-1]
    if state is None:
        state = torch.zeros(
            batch, heads, value_dim, key_dim, device=key.device, dtype=torch.float32
        )

    query = query.float()
    key = key.float()
    value = value.float()
    decay = decay.float()
    write = write.float()

    outputs = []
    for step in range(length):
        k = key[:, :, step]
        v = value[:, :, step]
        q = query[:, :, step]
        a = decay[:, :, step].unsqueeze(-1)
        b = write[:, :, step].unsqueeze(-1)

        predicted = torch.einsum("bhvk,bhk->bhv", state, k)
        error = v - a * predicted
        state = a.unsqueeze(-1) * state + torch.einsum(
            "bhv,bhk->bhvk", b * error, k
        )
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q))

    return torch.stack(outputs, dim=2), state


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int = 1) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads = cfg.n_query_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.kv_group_size

        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.decay_proj = nn.Linear(cfg.d_model, cfg.n_query_heads, bias=True)
        self.write_proj = nn.Linear(cfg.d_model, cfg.n_query_heads, bias=True)
        self.gate_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.gdn_output_gate
            else None
        )
        self.out_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        nn.init.constant_(self.decay_proj.bias, DECAY_OPEN_BIAS)
        nn.init.zeros_(self.write_proj.bias)

        if cfg.normalized_value_residual and layer > 1:
            self.value_mix = nn.Parameter(torch.zeros(1))
        else:
            self.value_mix = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        cache=None,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        source = v

        if self.value_mix is not None and v_first is not None:
            alpha = torch.sigmoid(self.value_mix).to(v.dtype)
            v = (1.0 - alpha) * v + alpha * v_first

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        decay = torch.sigmoid(self.decay_proj(x)).transpose(1, 2)
        write = torch.sigmoid(self.write_proj(x)).transpose(1, 2)

        previous = cache.get_state(self.layer) if cache is not None else None
        out, state = delta_rule_recurrence(q, k, v, decay, write, previous)
        if cache is not None:
            cache.set_state(self.layer, state)
        out = self.out_norm(out.to(x.dtype))
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out), source, state
