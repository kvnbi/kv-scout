from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.block import TransformerBlock
from kv_scout.model.layers import RMSNorm, rope_frequencies


class KVScout(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer) for layer in range(1, cfg.n_layers + 1)]
        )
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        cos, sin = rope_frequencies(cfg.head_dim, cfg.context_max, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o_proj.weight.mul_(scale)
                block.ffn.down.weight.mul_(scale)

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen = set()
        total = 0
        for param in self.parameters():
            if id(param) in seen or (trainable_only and not param.requires_grad):
                continue
            seen.add(id(param))
            total += param.numel()
        return total

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.cfg.context_max:
            raise ValueError("sequence longer than the configured context")

        x = self.embed(tokens)
        cos = self.rope_cos[:length].to(x.dtype)
        sin = self.rope_sin[:length].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.final_norm(x))


def language_model_loss(
    logits: torch.Tensor, targets: torch.Tensor, z_loss_weight: float = 0.0
):
    flat = logits.reshape(-1, logits.shape[-1]).float()
    cross_entropy = F.cross_entropy(flat, targets.reshape(-1))
    if z_loss_weight <= 0.0:
        return cross_entropy, cross_entropy
    z = torch.logsumexp(flat, dim=-1)
    return cross_entropy + z_loss_weight * z.pow(2).mean(), cross_entropy
