from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.block import TransformerBlock
from kv_scout.model.cache import Cache
from kv_scout.model.layers import RMSNorm, rope_frequencies
from kv_scout.model.mtp import MTPHead

BASE_INIT_STD = 0.02


class KVScout(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer) for layer in range(1, cfg.n_layers + 1)]
        )
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mtp = nn.ModuleList([MTPHead(cfg) for _ in range(cfg.mtp_heads)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        self.readout_divisor = (
            math.sqrt(cfg.width_multiplier)
            if cfg.tie_embeddings
            else cfg.width_multiplier
        )

        cos, sin = rope_frequencies(cfg.head_dim, cfg.context_max, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        nn.init.normal_(self.embed.weight, mean=0.0, std=BASE_INIT_STD)
        if not cfg.tie_embeddings:
            nn.init.normal_(
                self.head.weight,
                mean=0.0,
                std=BASE_INIT_STD / cfg.width_multiplier,
            )
        self._scale_residual_projections()
        self._tag_mup_groups()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = BASE_INIT_STD / math.sqrt(self.cfg.width_multiplier)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=BASE_INIT_STD)

    def _hidden_parameters(self):
        for block in self.blocks:
            modules = [
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
            ]
            gate = getattr(block.attn, "gate_proj", None)
            if gate is not None:
                modules.append(gate)
            for module in modules:
                yield module.weight

            hidden = getattr(block.ffn, "hidden_weights", None)
            if hidden is not None:
                yield from hidden()
            else:
                yield from (
                    block.ffn.gate.weight,
                    block.ffn.up.weight,
                    block.ffn.down.weight,
                )

        for head in self.mtp:
            yield from head.hidden_weights()

    def _tag_mup_groups(self) -> None:
        scale = 1.0 / self.cfg.width_multiplier
        for param in self.parameters():
            param.mup_lr_scale = 1.0
        for param in self._hidden_parameters():
            param.mup_lr_scale = scale

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o_proj.weight.mul_(scale)
                outputs = getattr(block.ffn, "output_weights", None)
                if outputs is None:
                    block.ffn.down.weight.mul_(scale)
                else:
                    for weight in outputs():
                        weight.mul_(scale)

            head_scale = 1.0 / math.sqrt(2)
            for head in self.mtp:
                for weight in head.output_weights():
                    weight.mul_(head_scale)

    def backbone_parameters(self) -> int:
        mtp = sum(p.numel() for head in self.mtp for p in head.parameters())
        return self.num_parameters() - mtp

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen = set()
        total = 0
        for param in self.parameters():
            if id(param) in seen or (trainable_only and not param.requires_grad):
                continue
            seen.add(id(param))
            total += param.numel()
        return total

    def readout(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.final_norm(hidden))
        if self.cfg.use_mup:
            logits = logits / self.readout_divisor
        return logits

    def predict_ahead(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> list[torch.Tensor]:
        predictions = []
        carried = hidden
        for depth, head in enumerate(self.mtp, start=1):
            usable = carried.shape[1] - 1
            if usable < 1 or depth >= tokens.shape[1]:
                break
            following = self.embed(tokens[:, depth : depth + usable])
            usable = min(usable, following.shape[1])
            carried = head(
                carried[:, :usable], following[:, :usable], cos[:usable], sin[:usable]
            )
            predictions.append(self.readout(carried))
        return predictions

    def forward(self, tokens: torch.Tensor, cache=None, predict_ahead: bool = False):
        length = tokens.shape[1]
        if length > self.cfg.context_max:
            raise ValueError("sequence longer than the configured context")

        base = cache.rebase(length) if cache is not None else 0
        store = cache if cache is not None else Cache(self.cfg, rolling=False)

        x = self.embed(tokens)
        cos = self.rope_cos[base : base + length].to(x.dtype)
        sin = self.rope_sin[base : base + length].to(x.dtype)
        v_first = None
        for block in self.blocks:
            x, source = block(x, cos, sin, v_first, store)
            if v_first is None:
                v_first = source
        if cache is not None:
            cache.trim()
            cache.advance(length)

        logits = self.readout(x)
        if not predict_ahead or not self.mtp:
            return logits
        return logits, self.predict_ahead(x, tokens, cos, sin)


def language_model_loss(
    logits: torch.Tensor, targets: torch.Tensor, z_loss_weight: float = 0.0
):
    flat = logits.reshape(-1, logits.shape[-1]).float()
    cross_entropy = F.cross_entropy(flat, targets.reshape(-1))
    if z_loss_weight <= 0.0:
        return cross_entropy, cross_entropy
    z = torch.logsumexp(flat, dim=-1)
    return cross_entropy + z_loss_weight * z.pow(2).mean(), cross_entropy
