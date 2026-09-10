from __future__ import annotations

import torch

from kv_scout.config import ModelConfig
from kv_scout.model.layers import rope_shift, rotate_rope

MAX_STRIDE = 256


class Cache:
    def __init__(self, cfg: ModelConfig, rolling: bool = True) -> None:
        self.cfg = cfg
        self.rolling = rolling
        self.storage_keys: dict[int, torch.Tensor] = {}
        self.storage_values: dict[int, torch.Tensor] = {}
        self.storage_positions: dict[int, torch.Tensor] = {}
        self.fill: dict[int, int] = {}
        self.states: dict[int, torch.Tensor] = {}
        self.length = 0
        self.origin = 0
        self.evictions = 0
        self.rebases = 0
        self._shifts: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def sinks(self, group: int) -> int:
        return self.cfg.sink_tokens if self.cfg.attention_sinks else 0

    def window(self, group: int) -> int:
        if self.cfg.windows_attention(group):
            return self.cfg.attention_window
        return max(1, self.cfg.context_max - self.sinks(group))

    def budget(self, group: int) -> int:
        return self.sinks(group) + self.window(group)

    def stride(self, group: int) -> int:
        return max(1, min(self.budget(group) // 8, MAX_STRIDE))

    def capacity(self, group: int) -> int:
        return self.budget(group) + self.stride(group)

    def geometry(self, group: int) -> tuple[int, int]:
        return self.sinks(group), self.window(group)

    def carries_rope(self, group: int) -> bool:
        return self.cfg.uses_rope(group)

    def base(self) -> int:
        return self.length - self.origin

    def rebase(self, tokens: int) -> int:
        limit = self.cfg.context_max
        step = max(1, limit // 2)
        while self.length - self.origin + tokens > limit and self.origin < self.length:
            self._slide(min(step, self.length - self.origin))
        base = self.length - self.origin
        if base < 0 or base + tokens > limit:
            raise ValueError("rotary frame fell outside the context")
        return base

    def _slide(self, shift: int) -> None:
        for group in self.storage_keys:
            sinks = self.sinks(group)
            fill = self.fill[group]
            if fill <= sinks or not self.carries_rope(group):
                continue
            keys = self.storage_keys[group]
            keys[:, :, sinks:fill] = self._rotate_back(keys[:, :, sinks:fill], shift)
        for layer in self.states:
            if self.carries_rope(layer):
                self.states[layer] = self._rotate_back(self.states[layer], shift)
        self.origin += shift
        self.rebases += 1

    def _allocate(self, group: int, key: torch.Tensor, value: torch.Tensor, size: int):
        b, h, _, d = key.shape
        self.storage_keys[group] = torch.zeros(
            b, h, size, d, device=key.device, dtype=key.dtype
        )
        self.storage_values[group] = torch.zeros(
            b, value.shape[1], size, value.shape[3],
            device=value.device, dtype=value.dtype,
        )
        self.storage_positions[group] = torch.zeros(
            size, device=key.device, dtype=torch.long
        )
        self.fill[group] = 0

    def _grow(self, group: int, size: int) -> None:
        keys, values = self.storage_keys[group], self.storage_values[group]
        positions = self.storage_positions[group]
        fill = self.fill[group]
        wider_k = torch.zeros(
            keys.shape[0], keys.shape[1], size, keys.shape[3],
            device=keys.device, dtype=keys.dtype,
        )
        wider_v = torch.zeros(
            values.shape[0], values.shape[1], size, values.shape[3],
            device=values.device, dtype=values.dtype,
        )
        wider_p = torch.zeros(size, device=positions.device, dtype=torch.long)
        wider_k[:, :, :fill] = keys[:, :, :fill]
        wider_v[:, :, :fill] = values[:, :, :fill]
        wider_p[:fill] = positions[:fill]
        self.storage_keys[group] = wider_k
        self.storage_values[group] = wider_v
        self.storage_positions[group] = wider_p

    def _reserve(self, group: int, key: torch.Tensor, value: torch.Tensor) -> None:
        count = key.shape[2]
        if group not in self.storage_keys:
            self._allocate(group, key, value, max(self.capacity(group), count))
            return
        held = self.storage_keys[group]
        if held.shape[0] != key.shape[0] or held.shape[1] != key.shape[1]:
            raise ValueError("cached keys do not match the incoming batch shape")
        if held.dtype != key.dtype or held.device != key.device:
            raise ValueError("cached keys do not match the incoming dtype or device")
        needed = self.fill[group] + count
        if needed > held.shape[2]:
            self._grow(group, needed)

    def append(self, group: int, key: torch.Tensor, value: torch.Tensor):
        count = key.shape[2]
        if not self.rolling and group not in self.storage_keys:
            self.storage_keys[group] = key
            self.storage_values[group] = value
            self.storage_positions[group] = torch.arange(
                self.length, self.length + count, device=key.device
            )
            self.fill[group] = count
            return self.read(group)
        self._reserve(group, key, value)
        start = self.fill[group]
        stop = start + count
        self.storage_keys[group][:, :, start:stop] = key
        self.storage_values[group][:, :, start:stop] = value
        self.storage_positions[group][start:stop] = torch.arange(
            self.length, self.length + count, device=key.device
        )
        self.fill[group] = stop
        return self.read(group)

    def _rotate_back(self, keys: torch.Tensor, shift: int) -> torch.Tensor:
        signature = (shift, str(keys.device), str(keys.dtype))
        pair = self._shifts.get(signature)
        if pair is None:
            cos, sin = rope_shift(
                self.cfg.head_dim, shift, self.cfg.rope_theta, keys.device
            )
            pair = (cos.to(keys.dtype), sin.to(keys.dtype))
            self._shifts[signature] = pair
        return rotate_rope(keys, pair[0], -pair[1])

    def _evict(self, group: int, drop: int) -> None:
        sinks = self.sinks(group)
        fill = self.fill[group]
        drop = min(drop, max(0, fill - sinks))
        if drop < 1:
            return
        keys = self.storage_keys[group]
        values = self.storage_values[group]
        positions = self.storage_positions[group]
        kept = fill - sinks - drop
        keys[:, :, sinks : sinks + kept] = keys[:, :, sinks + drop : fill].clone()
        values[:, :, sinks : sinks + kept] = values[:, :, sinks + drop : fill].clone()
        positions[sinks : sinks + kept] = positions[sinks + drop : fill].clone()
        self.fill[group] = sinks + kept
        self.evictions += 1

    def _shrink(self, group: int) -> None:
        size = self.capacity(group)
        keys, values = self.storage_keys[group], self.storage_values[group]
        self.storage_keys[group] = keys[:, :, :size].clone()
        self.storage_values[group] = values[:, :, :size].clone()
        self.storage_positions[group] = self.storage_positions[group][:size].clone()

    def trim(self) -> None:
        if not self.rolling:
            return
        for group in list(self.storage_keys):
            if self.fill[group] >= self.capacity(group):
                self._evict(group, self.fill[group] - self.budget(group))
            if self.storage_keys[group].shape[2] > self.capacity(group):
                self._shrink(group)

    def read(self, group: int):
        if group not in self.storage_keys:
            raise KeyError(f"cache group {group} has not been written yet")
        fill = self.fill[group]
        return (
            self.storage_keys[group][:, :, :fill],
            self.storage_values[group][:, :, :fill],
            self.storage_positions[group][:fill],
        )

    def set_state(self, layer: int, state: torch.Tensor) -> None:
        self.states[layer] = state

    def get_state(self, layer: int) -> torch.Tensor | None:
        return self.states.get(layer)

    def advance(self, tokens: int) -> None:
        self.length += tokens

    def reset(self) -> None:
        self.storage_keys.clear()
        self.storage_values.clear()
        self.storage_positions.clear()
        self.fill.clear()
        self.states.clear()
        self.length = 0
        self.origin = 0

    @property
    def keys(self) -> dict[int, torch.Tensor]:
        return {g: self.read(g)[0] for g in self.storage_keys}

    @property
    def values(self) -> dict[int, torch.Tensor]:
        return {g: self.read(g)[1] for g in self.storage_keys}

    @property
    def positions(self) -> dict[int, torch.Tensor]:
        return {g: self.read(g)[2] for g in self.storage_keys}

    @property
    def groups(self) -> int:
        return len(self.storage_keys)

    def entries(self) -> int:
        return sum(self.fill.values())

    def bytes(self) -> int:
        total = 0
        for group in self.storage_keys:
            keys, values = self.storage_keys[group], self.storage_values[group]
            total += keys.numel() * keys.element_size()
            total += values.numel() * values.element_size()
        for layer in self.states:
            total += self.states[layer].numel() * self.states[layer].element_size()
        return total
