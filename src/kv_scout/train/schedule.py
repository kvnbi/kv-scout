from __future__ import annotations

import math

from kv_scout.config import OptimConfig


def wsd_peak_from_cosine_peak(cosine_peak_lr: float, cfg: OptimConfig) -> float:
    return cosine_peak_lr * cfg.stable_fraction_of_peak


def wsd_lr(step: int, total_steps: int, cfg: OptimConfig) -> float:
    if step < 0:
        raise ValueError("step must not be negative")
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if cfg.schedule == "constant":
        return cfg.peak_lr

    warmup = min(cfg.warmup_steps, total_steps)
    if warmup > 0 and step < warmup:
        return cfg.peak_lr * (step + 1) / warmup

    decay_steps = max(1, int(round(cfg.decay_fraction * total_steps)))
    decay_start = max(warmup, total_steps - decay_steps)
    if step < decay_start:
        return cfg.peak_lr

    span = max(1, total_steps - decay_start)
    progress = min(1.0, (step - decay_start + 1) / span)
    if cfg.decay_profile == "inv_sqrt":
        scale = 1.0 - math.sqrt(progress)
    elif cfg.decay_profile == "linear":
        scale = 1.0 - progress
    elif cfg.decay_profile == "cosine":
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError("unsupported decay_profile")
    return cfg.peak_lr * max(scale, cfg.min_lr_fraction)
