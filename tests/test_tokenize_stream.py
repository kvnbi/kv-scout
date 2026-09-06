from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from kv_scout.data.shards import ShardIndex
from kv_scout.data.tokenize_stream import _batches, _resolve_field, run

DOCS = [
    {"text": "alpha beta"},
    {"text": ""},
    {"text": "gamma"},
    {"nested": {"body": "delta"}},
]


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    vocab = 2048
    eos = 7

    @classmethod
    def from_file(cls, path):
        return cls()

    def get_vocab_size(self):
        return self.vocab

    def token_to_id(self, token):
        return self.eos if token == "<|endoftext|>" else None

    def encode_batch(self, texts):
        return [FakeEncoding([len(t) % 100 + 1] * len(t)) for t in texts]


@pytest.fixture
def stub_modules(monkeypatch):
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *a, **k: iter(DOCS)
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)


def test_resolve_field_handles_nesting_and_absence():
    assert _resolve_field({"text": "a"}, "text") == "a"
    assert _resolve_field({"nested": {"body": "b"}}, "nested.body") == "b"
    assert _resolve_field({"text": 3}, "text") == ""
    assert _resolve_field({}, "missing.deeper") == ""


def test_batches_yields_a_short_final_batch():
    assert list(_batches(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_stream_writes_shards_and_index(tmp_path, stub_modules):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    out = tmp_path / "shards"

    total = run(
        dataset="fake",
        tokenizer_path=tokenizer,
        out_dir=out,
        shard_tokens=8,
        batch_docs=2,
    )

    index = ShardIndex.read(out / "index.json")
    index.verify()
    assert index.total_tokens == total
    assert index.vocab_size == 2048
    assert index.eos_id == 7
    assert index.tokenizer_sha256
    assert len(index.shards) > 1

    joined = np.concatenate(
        [np.asarray(index.open_memmap(i)) for i in range(len(index.shards))]
    )
    assert total == len("alpha beta") + len("gamma") + 2
    assert joined[-1] == 7
    assert int((joined == 7).sum()) == 2


def test_stream_honours_max_tokens(tmp_path, stub_modules):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    out = tmp_path / "shards"
    total = run(
        dataset="fake",
        tokenizer_path=tokenizer,
        out_dir=out,
        max_tokens=5,
        shard_tokens=64,
        batch_docs=2,
    )
    assert total == len("alpha beta") + 1


def test_stream_rejects_an_oversized_vocabulary(tmp_path, stub_modules, monkeypatch):
    monkeypatch.setattr(FakeTokenizer, "vocab", 131072)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    with pytest.raises(ValueError):
        run(dataset="fake", tokenizer_path=tokenizer, out_dir=tmp_path / "shards")


def test_stream_requires_an_eos_entry(tmp_path, stub_modules):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    with pytest.raises(ValueError):
        run(
            dataset="fake",
            tokenizer_path=tokenizer,
            out_dir=tmp_path / "shards",
            eos_token="<|absent|>",
        )
