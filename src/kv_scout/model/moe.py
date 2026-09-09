from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_scout.config import ModelConfig

WEIGHT_EPS = 1e-9


class Router(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_experts = cfg.moe.num_experts
        self.top_k = cfg.moe.top_k
        self.routing = cfg.moe.routing
        self.gate = nn.Linear(cfg.d_model, cfg.moe.num_experts, bias=False)

    def affinities(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.gate.weight.float())
        if self.routing == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor):
        scores = self.affinities(x)
        _, indices = torch.topk(scores, self.top_k, dim=-1)
        chosen = torch.gather(scores, -1, indices)
        weights = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(WEIGHT_EPS)
        return weights.to(x.dtype), indices, scores
