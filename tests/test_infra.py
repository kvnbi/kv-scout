from __future__ import annotations

import numpy as np
import pytest
import torch

from kv_scout import checkpoint as ckpt
from kv_scout.config import (
    HarnessModelConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TokenizerConfig,
)
from kv_scout.data.loader import ResumableTokenLoader
from kv_scout.data.shards import ShardIndex, ShardWriter
from kv_scout.data.synthetic import write_synthetic_corpus
from kv_scout.train.harness_model import HarnessGPT
from kv_scout.train.schedule import wsd_lr


def test_spec_section_two_configuration():
    cfg = ModelConfig()
    assert cfg.d_model == 1920
    assert cfg.n_layers == 30
    assert cfg.dense_warmup_layers == 2
    assert cfg.n_anchor_layers == 8
    assert cfg.n_linear_layers == 20
    assert cfg.n_query_heads == 15
    assert cfg.n_kv_heads == 5
    assert cfg.head_dim == 128
    assert cfg.kv_group_size == 3
    assert cfg.vocab_size == 32768
    assert cfg.moe.num_experts == 12
    assert cfg.moe.top_k == 2
    assert cfg.moe.shared_experts == 1
    assert cfg.moe.expert_ffn_hidden == 1280
    assert cfg.moe.shared_ffn_hidden == 2560
    assert (cfg.context_min, cfg.context_max) == (1024, 8192)
    assert cfg.precision == "bfloat16"
    assert cfg.languages.english == 0.95
    RunConfig()


def test_reflection_slots_are_reserved():
    cfg = TokenizerConfig()
    ids = cfg.reflection_token_ids
    assert len(ids) == 32
    assert max(ids) == cfg.vocab_size - 1


def test_configs_reject_inconsistent_shapes():
    with pytest.raises(ValueError):
        ModelConfig(n_query_heads=15, n_kv_heads=7)
    with pytest.raises(ValueError):
        ModelConfig(head_dim=127)
    with pytest.raises(ValueError):
        TokenizerConfig(vocab_size=131072)
    with pytest.raises(ValueError):
        OptimConfig(decay_fraction=1.5)


def test_harness_model_is_about_five_million_params():
    cfg = HarnessModelConfig()
    model = HarnessGPT(cfg)
    counted = model.num_parameters()
    assert counted == cfg.param_count
    assert 4.5e6 < counted < 6.0e6


def test_shards_round_trip(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    with ShardWriter(tmp_path, shard_tokens=300, vocab_size=2048) as writer:
        writer.add(tokens)
    index = ShardIndex.read(tmp_path / "index.json")
    index.verify()
    assert index.total_tokens == 1000
    assert len(index.shards) == 4
    joined = np.concatenate(
        [np.asarray(index.open_memmap(i)) for i in range(len(index.shards))]
    )
    assert np.array_equal(joined, tokens)


def test_shard_writer_rejects_out_of_range_tokens(tmp_path):
    with pytest.raises(ValueError):
        with ShardWriter(tmp_path, vocab_size=2048) as writer:
            writer.add(np.array([70000], dtype=np.int64))


def test_loader_position_round_trip(tmp_path):
    index = write_synthetic_corpus(
        tmp_path, n_tokens=60_000, vocab_size=256, shard_tokens=20_000
    )
    loader = ResumableTokenLoader(index, seq_len=64, batch_size=4, seed=7)
    for _ in range(3):
        loader.next_batch()
    state = loader.state()
    expected, _ = loader.next_batch()

    other = ResumableTokenLoader(index, seq_len=64, batch_size=4, seed=7)
    other.load_state(state)
    actual, _ = other.next_batch()
    assert torch.equal(expected, actual)


def test_loader_rolls_into_the_next_epoch(tmp_path):
    index = write_synthetic_corpus(
        tmp_path, n_tokens=20_000, vocab_size=256, shard_tokens=20_000
    )
    loader = ResumableTokenLoader(index, seq_len=64, batch_size=8, seed=7)
    batches = loader.sequences_per_epoch // 8 + 2
    for _ in range(batches):
        loader.next_batch()
    assert loader.epoch == 1


def test_checkpoint_round_trip(tmp_path):
    cfg = HarnessModelConfig(vocab_size=128, d_model=32, n_layers=1, n_heads=2)
    model = HarnessGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tokens = torch.randint(0, 128, (2, 16))
    model(tokens).sum().backward()
    optimizer.step()

    class Position:
        def to_payload(self):
            return {"epoch": 2, "position": 37}

    ckpt.save(
        tmp_path,
        step=5,
        model=model,
        optimizer=optimizer,
        loader_state=Position(),
        config={"note": "unit"},
        keep_last=2,
    )

    restored = HarnessGPT(cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    loaded = ckpt.load(
        ckpt.latest_path(tmp_path), model=restored, optimizer=restored_optimizer
    )

    assert loaded.step == 5
    assert loaded.loader == {"epoch": 2, "position": 37}
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)
    assert restored_optimizer.state_dict()["state"].keys() == (
        optimizer.state_dict()["state"].keys()
    )


def test_checkpoint_pruning_keeps_the_newest(tmp_path):
    cfg = HarnessModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2)
    model = HarnessGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class Position:
        def to_payload(self):
            return {"epoch": 0, "position": 0}

    for step in (10, 20, 30, 40):
        ckpt.save(
            tmp_path,
            step=step,
            model=model,
            optimizer=optimizer,
            loader_state=Position(),
            config={},
            keep_last=2,
        )
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_00000030.pt", "step_00000040.pt"]
    assert ckpt.latest_path(tmp_path).name == "step_00000040.pt"


def test_rng_state_round_trip():
    torch.manual_seed(3)
    state = ckpt.capture_rng_state()
    expected = torch.randn(5)
    ckpt.restore_rng_state(state)
    assert torch.equal(expected, torch.randn(5))


def test_wsd_schedule_shape():
    cfg = OptimConfig(peak_lr=1.0, warmup_steps=10, decay_fraction=0.2)
    total = 100
    assert wsd_lr(0, total, cfg) == pytest.approx(0.1)
    assert wsd_lr(9, total, cfg) == pytest.approx(1.0)
    assert wsd_lr(50, total, cfg) == pytest.approx(1.0)
    assert wsd_lr(79, total, cfg) == pytest.approx(1.0)
    assert wsd_lr(total - 1, total, cfg) == pytest.approx(0.0)
    decay = [wsd_lr(s, total, cfg) for s in range(80, total)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))


def test_warmup_never_swallows_the_decay_phase():
    from kv_scout.train.schedule import wsd_lr

    cfg = OptimConfig()
    for total in (100, 1000, 20000, 100000):
        assert wsd_lr(total - 1, total, cfg) == pytest.approx(0.0)
        peak_reached = any(
            wsd_lr(step, total, cfg) == pytest.approx(cfg.peak_lr)
            for step in range(0, total, max(1, total // 50))
        )
        assert peak_reached, f"run of {total} steps never reaches peak learning rate"
