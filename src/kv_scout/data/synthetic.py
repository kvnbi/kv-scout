from __future__ import annotations

from pathlib import Path

import numpy as np

from kv_scout.data.shards import ShardIndex, ShardWriter


def phrase_stream(
    n_tokens: int,
    vocab_size: int,
    n_phrases: int = 64,
    phrase_len: int = 32,
    seed: int = 0,
    noise: float = 0.02,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phrases = rng.integers(
        0, vocab_size, size=(n_phrases, phrase_len), dtype=np.uint16
    )
    repeats = int(np.ceil(n_tokens / phrase_len))
    picks = rng.integers(0, n_phrases, size=repeats)
    stream = phrases[picks].reshape(-1)[:n_tokens].copy()
    if noise > 0.0:
        mask = rng.random(stream.size) < noise
        stream[mask] = rng.integers(
            0, vocab_size, size=int(mask.sum()), dtype=np.uint16
        )
    return stream


def write_synthetic_corpus(
    root: str | Path,
    n_tokens: int = 1_000_000,
    vocab_size: int = 2048,
    shard_tokens: int = 250_000,
    seed: int = 0,
    n_phrases: int = 64,
    phrase_len: int = 32,
) -> ShardIndex:
    stream = phrase_stream(
        n_tokens=n_tokens,
        vocab_size=vocab_size,
        n_phrases=n_phrases,
        phrase_len=phrase_len,
        seed=seed,
    )
    with ShardWriter(
        root=root,
        shard_tokens=shard_tokens,
        vocab_size=vocab_size,
        eos_id=None,
        sources=[
            {
                "name": "synthetic_phrases",
                "seed": seed,
                "tokens": int(n_tokens),
                "n_phrases": n_phrases,
                "phrase_len": phrase_len,
            }
        ],
    ) as writer:
        writer.add(stream)
    return ShardIndex.read(Path(root) / "index.json")
