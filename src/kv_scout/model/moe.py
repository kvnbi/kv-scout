from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig
from kv_scout.model.layers import SwiGLU

WEIGHT_EPS = 1e-9


class Router(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_experts = cfg.moe.num_experts
        self.top_k = cfg.moe.top_k
        self.routing = cfg.moe.routing
        self.balancing = cfg.moe.aux_loss_free_balancing
        self.update_rate = cfg.moe.balance_update_rate
        self.gate = nn.Linear(cfg.d_model, cfg.moe.num_experts, bias=False)
        self.register_buffer("balance_bias", torch.zeros(cfg.moe.num_experts))
        self.load_decay = cfg.moe.load_decay
        self.register_buffer(
            "expert_counts", torch.zeros(cfg.moe.num_experts, dtype=torch.long)
        )
        self.register_buffer(
            "recent_load", torch.full((cfg.moe.num_experts,), 1.0 / cfg.moe.num_experts)
        )

    def affinities(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.gate.weight.float())
        if self.routing == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor):
        scores = self.affinities(x)
        selection = scores
        if self.balancing:
            selection = scores + self.balance_bias.to(scores.dtype)
        _, indices = torch.topk(selection, self.top_k, dim=-1)
        chosen = torch.gather(scores, -1, indices)
        weights = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(WEIGHT_EPS)

        with torch.no_grad():
            counts = torch.bincount(
                indices.reshape(-1), minlength=self.num_experts
            ).to(self.expert_counts.device)
            self.expert_counts += counts
            self.observe(counts)
            if self.training and self.balancing:
                self.rebalance(counts)

        return weights.to(x.dtype), indices, scores

    @torch.no_grad()
    def rebalance(self, counts: torch.Tensor) -> None:
        total = float(counts.sum())
        if total <= 0:
            return
        target = total / self.num_experts
        error = target - counts.to(self.balance_bias.dtype)
        self.balance_bias += self.update_rate * torch.sign(error)
        self.balance_bias -= self.balance_bias.mean()

    @torch.no_grad()
    def observe(self, counts: torch.Tensor) -> None:
        total = float(counts.sum())
        if total <= 0:
            return
        share = counts.to(self.recent_load.dtype) / total
        self.recent_load.mul_(self.load_decay).add_(share, alpha=1.0 - self.load_decay)

    def reset_counts(self) -> None:
        self.expert_counts.zero_()
        self.recent_load.fill_(1.0 / self.num_experts)

    def balance(self) -> dict:
        from kv_scout.train.instrument import balance_summary

        return balance_summary(self.recent_load)

    def lifetime_balance(self) -> dict:
        from kv_scout.train.instrument import balance_summary

        return balance_summary(self.expert_counts)


class ExpertBank(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_experts = cfg.moe.num_experts
        self.experts = nn.ModuleList(
            [
                SwiGLU(cfg.d_model, cfg.moe.expert_ffn_hidden)
                for _ in range(cfg.moe.num_experts)
            ]
        )

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=-2)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        chosen = indices.reshape(-1, indices.shape[-1])
        share = weights.reshape(-1, weights.shape[-1]).to(x.dtype)
        out = torch.zeros_like(flat)

        for expert in range(self.num_experts):
            tokens, slots = (chosen == expert).nonzero(as_tuple=True)
            if tokens.numel() == 0:
                continue
            computed = self.experts[expert](flat.index_select(0, tokens))
            scaled = computed * share[tokens, slots].unsqueeze(-1)
            out.index_add_(0, tokens, scaled.to(out.dtype))

        return out.reshape(shape)
