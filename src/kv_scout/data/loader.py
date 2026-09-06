from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kv_scout.data.shards import ShardIndex


@dataclass
class LoaderState:
    epoch: int
    position: int

    def to_payload(self) -> dict:
        return {"epoch": self.epoch, "position": self.position}

    @classmethod
    def from_payload(cls, payload: dict) -> "LoaderState":
        return cls(epoch=int(payload["epoch"]), position=int(payload["position"]))


class ResumableTokenLoader:
    def __init__(
        self,
        index: ShardIndex | str | Path,
        seq_len: int,
        batch_size: int,
        seed: int = 7,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if isinstance(index, (str, Path)):
            index = ShardIndex.read(index)
        self.index = index
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.shard_sequences = [
            max(0, (entry.tokens - 1) // seq_len) for entry in index.shards
        ]
        self.cumulative = np.cumsum([0] + self.shard_sequences)
        self.sequences_per_epoch = int(self.cumulative[-1])
        if self.sequences_per_epoch == 0:
            raise ValueError("shard index holds no full sequences at this seq_len")
        if self.drop_last and self.sequences_per_epoch < batch_size:
            raise ValueError("not enough sequences for a single batch")

        self.epoch = 0
        self.position = 0
        self._shard_order: np.ndarray | None = None
        self._local_order: dict[int, np.ndarray] = {}
        self._memmaps: dict[int, np.memmap] = {}
        self._build_epoch(0)

    def _rng(self, *salt: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, *salt])

    def _build_epoch(self, epoch: int) -> None:
        n = len(self.index.shards)
        order = np.arange(n)
        if self.shuffle:
            self._rng(epoch, 0).shuffle(order)
        order = order[[self.shard_sequences[i] > 0 for i in order]]
        self._shard_order = order
        self._epoch_cumulative = np.cumsum(
            [0] + [self.shard_sequences[i] for i in order]
        )
        self._local_order = {}
        self.epoch = epoch

    def _local_indices(self, shard: int) -> np.ndarray:
        cached = self._local_order.get(shard)
        if cached is not None:
            return cached
        count = self.shard_sequences[shard]
        order = np.arange(count)
        if self.shuffle:
            self._rng(self.epoch, 1, shard).shuffle(order)
        self._local_order = {shard: order}
        return order

    def _memmap(self, shard: int) -> np.memmap:
        cached = self._memmaps.get(shard)
        if cached is None:
            cached = self.index.open_memmap(shard)
            self._memmaps = {shard: cached}
        return cached

    def _sequence(self, position: int) -> np.ndarray:
        slot = int(np.searchsorted(self._epoch_cumulative, position, side="right") - 1)
        shard = int(self._shard_order[slot])
        local = int(self._local_indices(shard)[position - self._epoch_cumulative[slot]])
        start = local * self.seq_len
        window = self._memmap(shard)[start : start + self.seq_len + 1]
        return np.asarray(window, dtype=np.int64)

    def next_batch(self, device: torch.device | str = "cpu"):
        rows = []
        for _ in range(self.batch_size):
            if self.position >= self.sequences_per_epoch:
                self._build_epoch(self.epoch + 1)
                self.position = 0
            rows.append(self._sequence(self.position))
            self.position += 1
        block = torch.from_numpy(np.stack(rows))
        inputs = block[:, :-1].contiguous().to(device)
        targets = block[:, 1:].contiguous().to(device)
        return inputs, targets

    def state(self) -> LoaderState:
        return LoaderState(epoch=self.epoch, position=self.position)

    def load_state(self, state: LoaderState | dict) -> None:
        if isinstance(state, dict):
            state = LoaderState.from_payload(state)
        if state.epoch != self.epoch:
            self._build_epoch(state.epoch)
        if not 0 <= state.position <= self.sequences_per_epoch:
            raise ValueError("loader position is outside the epoch")
        self.position = state.position
