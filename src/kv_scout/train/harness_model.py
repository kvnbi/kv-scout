from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import HarnessModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


class HarnessAttention(nn.Module):
    def __init__(self, cfg: HarnessModelConfig, dropout: float) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class HarnessMLP(nn.Module):
    def __init__(self, cfg: HarnessModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HarnessBlock(nn.Module):
    def __init__(self, cfg: HarnessModelConfig, dropout: float) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = HarnessAttention(cfg, dropout)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = HarnessMLP(cfg)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.drop(self.attn(self.norm_attn(x), cos, sin))
        x = x + self.drop(self.mlp(self.norm_mlp(x)))
        return x


class HarnessGPT(nn.Module):
    def __init__(self, cfg: HarnessModelConfig, dropout: float = 0.1) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [HarnessBlock(cfg, dropout) for _ in range(cfg.n_layers)]
        )
        self.norm_out = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        seen = set()
        total = 0
        for param in self.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
        return total

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        if t > self.cfg.seq_len:
            raise ValueError("sequence longer than the configured context")
        x = self.embed(tokens)
        cos = self.rope_cos[:t].to(x.dtype)
        sin = self.rope_sin[:t].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm_out(x))


def loss_with_z(
    logits: torch.Tensor, targets: torch.Tensor, z_loss_weight: float = 0.0
):
    flat = logits.reshape(-1, logits.shape[-1]).float()
    cross_entropy = F.cross_entropy(flat, targets.reshape(-1))
    if z_loss_weight <= 0.0:
        return cross_entropy, cross_entropy
    z = torch.logsumexp(flat, dim=-1)
    total = cross_entropy + z_loss_weight * z.pow(2).mean()
    return total, cross_entropy
