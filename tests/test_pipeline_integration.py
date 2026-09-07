from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kv_scout.config import HarnessModelConfig
from kv_scout.data.loader import ResumableTokenLoader
from kv_scout.data.shards import ShardIndex, ShardWriter
from kv_scout.train.harness_model import HarnessGPT, loss_with_z

TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"

DOCUMENTS = [
    "The colour of the centre was a testament to her behaviour.",
    "Die Farbe der Mitte war ein Zeugnis ihres Verhaltens.",
    "def compute(self, x):\n    return [i ** 2 for i in x]\n",
    "She realised the defence had a licence to practise, and travelled on.",
    "cost 5 euros € and a \U0001f600 face",
]

pytestmark = pytest.mark.skipif(
    not TOKENIZER.exists(), reason="frozen tokenizer not present"
)


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


@pytest.fixture(scope="module")
def shards(tokenizer, tmp_path_factory):
    root = tmp_path_factory.mktemp("shards")
    eos = tokenizer.token_to_id("<|endoftext|>")
    with ShardWriter(
        root, shard_tokens=900, vocab_size=tokenizer.get_vocab_size(), eos_id=eos
    ) as writer:
        for _ in range(60):
            for text in DOCUMENTS:
                writer.add(tokenizer.encode(text).ids + [eos])
    return ShardIndex.read(root / "index.json")


def test_index_is_consistent_with_the_files(shards):
    shards.verify()
    assert len(shards.shards) > 1
    assert shards.total_tokens == sum(s.tokens for s in shards.shards)


def test_every_token_fits_the_vocabulary(shards):
    for i in range(len(shards.shards)):
        block = np.asarray(shards.open_memmap(i))
        if block.size:
            assert block.min() >= 0
            assert block.max() < shards.vocab_size
            assert block.dtype == np.uint16


def test_documents_survive_the_round_trip(shards, tokenizer):
    block = np.asarray(shards.open_memmap(0))
    breaks = np.where(block == shards.eos_id)[0]
    assert len(breaks) >= len(DOCUMENTS)

    start = 0
    for position, index in enumerate(breaks[: len(DOCUMENTS)]):
        ids = block[start:index].tolist()
        assert tokenizer.decode(ids) == DOCUMENTS[position]
        assert tokenizer.encode(tokenizer.decode(ids)).ids == ids
        start = index + 1


def test_loader_yields_shifted_targets(shards):
    loader = ResumableTokenLoader(shards, seq_len=64, batch_size=4, seed=7)
    inputs, targets = loader.next_batch()
    assert inputs.shape == (4, 64)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert int(inputs.max()) < shards.vocab_size


def test_loader_skips_shards_too_small_for_a_sequence(shards):
    loader = ResumableTokenLoader(shards, seq_len=512, batch_size=1, seed=7)
    assert any(count == 0 for count in loader.shard_sequences)
    assert loader.sequences_per_epoch > 0
    loader.next_batch()


def test_model_trains_a_step_on_real_tokens(shards):
    torch.manual_seed(0)
    cfg = HarnessModelConfig(
        vocab_size=shards.vocab_size, d_model=96, n_layers=2, n_heads=3,
        ffn_hidden=192, seq_len=64,
    )
    model = HarnessGPT(cfg, dropout=0.0)
    loader = ResumableTokenLoader(shards, seq_len=64, batch_size=2, seed=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = None
    for _ in range(5):
        inputs, targets = loader.next_batch()
        total, cross_entropy = loss_with_z(model(inputs), targets, 0.0)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        last = float(cross_entropy.detach())
        if first is None:
            first = last

    assert first == pytest.approx(np.log(shards.vocab_size), abs=1.0)
    assert last < first
