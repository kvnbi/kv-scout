from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.block import TransformerBlock
from kv_scout.model.layers import RMSNorm


def mtp_block_config(cfg: ModelConfig) -> ModelConfig:
    return replace(
        cfg,
        use_gdn=False,
        use_moe=False,
        dense_ffn_hidden=cfg.mtp_ffn_hidden,
        layernorm_scaling=False,
        cross_layer_kv_sharing=False,
        attention_sinks=False,
        normalized_value_residual=False,
    )


class MTPHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.hidden_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.token_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.merge = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = TransformerBlock(mtp_block_config(cfg), 1)

    def hidden_weights(self):
        yield self.merge.weight
        block = self.block
        for module in (
            block.attn.q_proj,
            block.attn.k_proj,
            block.attn.v_proj,
            block.attn.o_proj,
            block.ffn.gate,
            block.ffn.up,
            block.ffn.down,
        ):
            yield module.weight
        gate = getattr(block.attn, "gate_proj", None)
        if gate is not None:
            yield gate.weight

    def output_weights(self):
        yield self.block.attn.o_proj.weight
        yield self.block.ffn.down.weight

    def forward(
        self,
        hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        merged = self.merge(
            torch.cat(
                [self.hidden_norm(hidden), self.token_norm(token_embeddings)], dim=-1
            )
        )
        out, _ = self.block(merged, cos, sin)
        return out


def multi_token_loss(
    predictions: list[torch.Tensor],
    targets: torch.Tensor,
    weight: float = 1.0,
):
    if not predictions or weight <= 0.0:
        zero = targets.new_zeros((), dtype=torch.float32)
        return zero, []

    losses = []
    for depth, logits in enumerate(predictions, start=1):
        shifted = targets[:, depth:]
        usable = min(logits.shape[1], shifted.shape[1])
        if usable < 1:
            continue
        losses.append(
            F.cross_entropy(
                logits[:, :usable].reshape(-1, logits.shape[-1]).float(),
                shifted[:, :usable].reshape(-1),
            )
        )
    if not losses:
        zero = targets.new_zeros((), dtype=torch.float32)
        return zero, []
    return weight * torch.stack(losses).mean(), losses
