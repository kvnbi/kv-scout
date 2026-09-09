from __future__ import annotations

import torch

from kv_scout.config import ModelConfig


class Cache:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.states: dict[int, torch.Tensor] = {}
        self.length = 0

    def append(self, group: int, key: torch.Tensor, value: torch.Tensor):
        if group in self.keys:
            self.keys[group] = torch.cat([self.keys[group], key], dim=2)
            self.values[group] = torch.cat([self.values[group], value], dim=2)
        else:
            self.keys[group] = key
            self.values[group] = value
        return self.keys[group], self.values[group]

    def read(self, group: int):
        if group not in self.keys:
            raise KeyError(f"cache group {group} has not been written yet")
        return self.keys[group], self.values[group]

    def set_state(self, layer: int, state: torch.Tensor) -> None:
        self.states[layer] = state

    def get_state(self, layer: int) -> torch.Tensor | None:
        return self.states.get(layer)

    def advance(self, tokens: int) -> None:
        self.length += tokens

    def reset(self) -> None:
        self.keys.clear()
        self.values.clear()
        self.states.clear()
        self.length = 0

    @property
    def groups(self) -> int:
        return len(self.keys)

    def entries(self) -> int:
        return sum(k.shape[2] for k in self.keys.values())

    def bytes(self) -> int:
        total = 0
        for group in self.keys:
            total += self.keys[group].numel() * self.keys[group].element_size()
            total += self.values[group].numel() * self.values[group].element_size()
        for layer in self.states:
            total += self.states[layer].numel() * self.states[layer].element_size()
        return total
