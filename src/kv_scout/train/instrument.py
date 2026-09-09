from __future__ import annotations

import math

import torch

STARVATION_FRACTION = 0.1


class RouterMonitor:
    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.counts = torch.zeros(num_experts, dtype=torch.long)

    def record(self, expert_indices: torch.Tensor) -> None:
        flat = expert_indices.detach().reshape(-1).to("cpu", torch.long)
        self.counts += torch.bincount(flat, minlength=self.num_experts)

    def reset(self) -> None:
        self.counts.zero_()

    def summary(self) -> dict:
        return balance_summary(self.counts)


def balance_summary(counts) -> dict:
    values = torch.as_tensor(counts, dtype=torch.float64)
    total = float(values.sum())
    experts = int(values.numel())
    if total <= 0 or experts == 0:
        return {
            "experts": experts,
            "assignments": 0,
            "max_over_mean": 0.0,
            "min_over_mean": 0.0,
            "coefficient_of_variation": 0.0,
            "normalised_entropy": 0.0,
            "starved_experts": experts,
        }
    share = values / total
    mean = 1.0 / experts
    entropy = float(-(share[share > 0] * share[share > 0].log()).sum())
    return {
        "experts": experts,
        "assignments": int(total),
        "max_over_mean": float(share.max()) / mean,
        "min_over_mean": float(share.min()) / mean,
        "coefficient_of_variation": float(share.std(unbiased=False)) / mean,
        "normalised_entropy": entropy / math.log(experts) if experts > 1 else 1.0,
        "starved_experts": int((share < STARVATION_FRACTION * mean).sum()),
    }


def collect_router_stats(model) -> list[dict]:
    stats = []
    for name, module in model.named_modules():
        counts = getattr(module, "recent_load", None)
        if counts is None:
            counts = getattr(module, "expert_counts", None)
        if counts is None:
            continue
        summary = balance_summary(counts)
        summary["module"] = name
        lifetime = getattr(module, "expert_counts", None)
        if lifetime is not None:
            summary["assignments"] = int(torch.as_tensor(lifetime).sum())
        stats.append(summary)
    return stats


def router_stats_are_healthy(stats: list[dict], max_over_mean: float = 3.0) -> bool:
    if not stats:
        return True
    return all(
        s["max_over_mean"] <= max_over_mean and s["starved_experts"] == 0
        for s in stats
    )
