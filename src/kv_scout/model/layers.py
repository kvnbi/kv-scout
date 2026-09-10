from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, scale: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.scale = scale
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * norm * self.weight.float()
        if self.scale != 1.0:
            out = out * self.scale
        return out.to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def inverse_frequencies(head_dim: int, theta: float, device=None) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    exponents = torch.arange(0, head_dim, 2, device=device).float() / head_dim
    return 1.0 / (theta**exponents)


def rope_frequencies(head_dim: int, seq_len: int, theta: float, device=None):
    inverse = inverse_frequencies(head_dim, theta, device)
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inverse)
    return torch.cos(angles), torch.sin(angles)


def rope_shift(head_dim: int, shift: int, theta: float, device=None):
    angles = inverse_frequencies(head_dim, theta, device) * float(shift)
    return torch.cos(angles), torch.sin(angles)


def rotate_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    left, right = x[..., :half], x[..., half:]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return rotate_rope(
        x,
        cos[None, None, : x.shape[-2], :],
        sin[None, None, : x.shape[-2], :],
    )


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, groups, t, d).reshape(b, h * groups, t, d)
