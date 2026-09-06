from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHARD_DTYPE = np.uint16
INDEX_VERSION = 1
MAX_TOKEN_ID = 65535


@dataclass(frozen=True)
class ShardEntry:
    name: str
    tokens: int
    sha256: str


class ShardIndex:
    def __init__(
        self,
        root: Path,
        shards: list[ShardEntry],
        vocab_size: int,
        eos_id: int | None = None,
        tokenizer_sha256: str | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        self.root = Path(root)
        self.shards = shards
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.tokenizer_sha256 = tokenizer_sha256
        self.sources = sources or []

    @property
    def total_tokens(self) -> int:
        return sum(entry.tokens for entry in self.shards)

    def path(self, i: int) -> Path:
        return self.root / self.shards[i].name

    def open_memmap(self, i: int) -> np.memmap:
        entry = self.shards[i]
        return np.memmap(
            self.path(i), dtype=SHARD_DTYPE, mode="r", shape=(entry.tokens,)
        )

    def to_payload(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "dtype": np.dtype(SHARD_DTYPE).name,
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "total_tokens": self.total_tokens,
            "sources": self.sources,
            "shards": [
                {"name": e.name, "tokens": e.tokens, "sha256": e.sha256}
                for e in self.shards
            ],
        }

    def write(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else self.root / "index.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_payload(), indent=2) + "\n")
        os.replace(tmp, target)
        return target

    @classmethod
    def read(cls, path: str | Path) -> "ShardIndex":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("version") != INDEX_VERSION:
            raise ValueError("unsupported shard index version")
        if payload.get("dtype") != np.dtype(SHARD_DTYPE).name:
            raise ValueError("shard index declares an unsupported dtype")
        shards = [
            ShardEntry(name=s["name"], tokens=int(s["tokens"]), sha256=s.get("sha256", ""))
            for s in payload["shards"]
        ]
        return cls(
            root=path.parent,
            shards=shards,
            vocab_size=int(payload["vocab_size"]),
            eos_id=payload.get("eos_id"),
            tokenizer_sha256=payload.get("tokenizer_sha256"),
            sources=payload.get("sources", []),
        )

    def verify(self) -> None:
        for i, entry in enumerate(self.shards):
            p = self.path(i)
            if not p.exists():
                raise FileNotFoundError(f"missing shard {entry.name}")
            expected = entry.tokens * np.dtype(SHARD_DTYPE).itemsize
            actual = p.stat().st_size
            if actual != expected:
                raise ValueError(
                    f"shard {entry.name} is {actual} bytes, index expects {expected}"
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardWriter:
    def __init__(
        self,
        root: str | Path,
        shard_tokens: int = 100_000_000,
        prefix: str = "shard",
        vocab_size: int = 32768,
        eos_id: int | None = None,
        tokenizer_sha256: str | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        if shard_tokens < 1:
            raise ValueError("shard_tokens must be at least 1")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.prefix = prefix
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.tokenizer_sha256 = tokenizer_sha256
        self.sources = sources or []
        self.entries: list[ShardEntry] = []
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._handle = None
        self._open_tokens = 0
        self._open_path: Path | None = None

    def _open(self) -> None:
        name = f"{self.prefix}_{len(self.entries):05d}.bin"
        self._open_path = self.root / name
        self._handle = open(self._open_path, "wb")
        self._open_tokens = 0

    def _close(self) -> None:
        if self._handle is None or self._open_path is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self.entries.append(
            ShardEntry(
                name=self._open_path.name,
                tokens=self._open_tokens,
                sha256=_sha256_file(self._open_path),
            )
        )
        self._handle = None
        self._open_path = None
        self._open_tokens = 0

    def add(self, tokens) -> None:
        array = np.asarray(tokens)
        if array.size == 0:
            return
        if array.min() < 0 or array.max() > MAX_TOKEN_ID:
            raise ValueError("token ids do not fit in uint16")
        self._buffer.append(array.astype(SHARD_DTYPE, copy=False))
        self._buffered += int(array.size)
        if self._buffered >= 1 << 20:
            self._drain()

    def _drain(self) -> None:
        if not self._buffer:
            return
        block = np.concatenate(self._buffer)
        self._buffer = []
        self._buffered = 0
        offset = 0
        while offset < block.size:
            if self._handle is None:
                self._open()
            room = self.shard_tokens - self._open_tokens
            take = min(room, block.size - offset)
            block[offset : offset + take].tofile(self._handle)
            self._open_tokens += take
            offset += take
            if self._open_tokens >= self.shard_tokens:
                self._close()

    @property
    def tokens_written(self) -> int:
        return sum(e.tokens for e in self.entries) + self._open_tokens + self._buffered

    def close(self) -> ShardIndex:
        self._drain()
        self._close()
        index = ShardIndex(
            root=self.root,
            shards=self.entries,
            vocab_size=self.vocab_size,
            eos_id=self.eos_id,
            tokenizer_sha256=self.tokenizer_sha256,
            sources=self.sources,
        )
        index.write()
        return index

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
