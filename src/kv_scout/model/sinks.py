from __future__ import annotations

import torch


def sink_window_mask(
    query_length: int,
    key_length: int,
    sinks: int,
    window: int,
    offset: int = 0,
    device=None,
) -> torch.Tensor:
    queries = torch.arange(offset, offset + query_length, device=device).unsqueeze(1)
    keys = torch.arange(key_length, device=device).unsqueeze(0)
    causal = keys <= queries
    recent = keys > queries - window
    sink = keys < sinks
    return causal & (recent | sink)


def evict(
    keys: torch.Tensor, values: torch.Tensor, sinks: int, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    length = keys.shape[2]
    if length <= sinks + window:
        return keys, values
    head_k, head_v = keys[:, :, :sinks], values[:, :, :sinks]
    tail_k, tail_v = keys[:, :, -window:], values[:, :, -window:]
    return (
        torch.cat([head_k, tail_k], dim=2),
        torch.cat([head_v, tail_v], dim=2),
    )


def retained_positions(length: int, sinks: int, window: int) -> list[int]:
    if length <= sinks + window:
        return list(range(length))
    return list(range(sinks)) + list(range(length - window, length))


def visibility_mask(
    key_positions: torch.Tensor,
    query_length: int,
    offset: int,
    sinks: int,
    window: int,
) -> torch.Tensor:
    queries = torch.arange(
        offset, offset + query_length, device=key_positions.device
    ).unsqueeze(1)
    keys = key_positions.unsqueeze(0)
    causal = keys <= queries
    recent = keys > queries - window
    sink = keys < sinks
    return (causal & (recent | sink)).unsqueeze(0).unsqueeze(0)
