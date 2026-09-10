from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kv_scout.train.ema import (
    EMA_NAME,
    average_state_dicts,
    build_ema,
    checkpoint_steps,
    decay_phase_steps,
    ema_weights,
)


def write_checkpoint(directory: Path, step: int, value: float, counts: int = 1):
    payload = {
        "version": 1,
        "step": step,
        "model": {
            "weight": torch.full((3, 3), value),
            "bias": torch.full((3,), value * 2),
            "counter": torch.full((3,), counts, dtype=torch.long),
        },
        "optimizer": {"state": step},
        "loader": {"epoch": 0, "position": step},
        "rng": {},
        "config": {"note": "unit"},
        "extra": {"device": "cpu"},
    }
    torch.save(payload, directory / f"step_{step:08d}.pt")


def test_checkpoints_are_found_in_order(tmp_path):
    for step in (300, 100, 200):
        write_checkpoint(tmp_path, step, 1.0)
    found = checkpoint_steps(tmp_path)
    assert [step for step, _ in found] == [100, 200, 300]
    assert all(path.exists() for _, path in found)


def test_unrelated_files_are_ignored(tmp_path):
    write_checkpoint(tmp_path, 100, 1.0)
    (tmp_path / "latest.json").write_text("{}")
    (tmp_path / "losses.jsonl").write_text("")
    (tmp_path / "notes.pt").write_bytes(b"")
    assert len(checkpoint_steps(tmp_path)) == 1


def test_decay_phase_selects_the_final_fraction():
    steps = [(s, Path(f"step_{s}.pt")) for s in range(50, 501, 50)]
    chosen = decay_phase_steps(steps, total_steps=500, decay_fraction=0.2)
    assert [s for s, _ in chosen] == [450, 500]
    wider = decay_phase_steps(steps, total_steps=500, decay_fraction=0.5)
    assert [s for s, _ in wider] == [300, 350, 400, 450, 500]


def test_decay_phase_never_returns_nothing():
    steps = [(100, Path("a")), (200, Path("b"))]
    chosen = decay_phase_steps(steps, total_steps=10_000, decay_fraction=0.01)
    assert len(chosen) == 1
    assert chosen[0][0] == 200


def test_weights_sum_to_one_and_favour_recent():
    weights = ema_weights([100, 200, 300], 0.99)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]


def test_a_decay_of_one_is_a_plain_average():
    weights = ema_weights([10, 50, 900], 1.0)
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_a_fast_decay_collapses_onto_the_last_checkpoint():
    weights = ema_weights([100, 200, 300], 0.9)
    assert weights[-1] > 0.99


def test_weights_are_validated():
    with pytest.raises(ValueError):
        ema_weights([], 0.999)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            ema_weights([1, 2], bad)


def test_averaging_is_a_weighted_mean():
    states = [
        {"w": torch.tensor([0.0, 10.0])},
        {"w": torch.tensor([2.0, 20.0])},
    ]
    merged = average_state_dicts(states, [0.25, 0.75])
    assert torch.allclose(merged["w"], torch.tensor([1.5, 17.5]))


def test_integer_tensors_take_the_last_value_rather_than_an_average():
    states = [
        {"n": torch.tensor([10, 20], dtype=torch.long)},
        {"n": torch.tensor([30, 40], dtype=torch.long)},
    ]
    merged = average_state_dicts(states, [0.5, 0.5])
    assert torch.equal(merged["n"], torch.tensor([30, 40], dtype=torch.long))


def test_dtype_is_preserved():
    states = [
        {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
        {"w": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)},
    ]
    merged = average_state_dicts(states, [0.5, 0.5])
    assert merged["w"].dtype == torch.bfloat16
    assert torch.allclose(merged["w"].float(), torch.tensor([2.0, 3.0]), atol=1e-2)


def test_mismatched_checkpoints_are_refused():
    with pytest.raises(ValueError):
        average_state_dicts([{"a": torch.zeros(2)}, {"b": torch.zeros(2)}], [0.5, 0.5])
    with pytest.raises(ValueError):
        average_state_dicts([{"a": torch.zeros(2)}, {"a": torch.zeros(3)}], [0.5, 0.5])
    with pytest.raises(ValueError):
        average_state_dicts([{"a": torch.zeros(2)}], [0.5, 0.5])
    with pytest.raises(ValueError):
        average_state_dicts([], [])


def test_build_ema_writes_a_loadable_checkpoint(tmp_path):
    for index, step in enumerate((100, 200, 300)):
        write_checkpoint(tmp_path, step, float(index), counts=index)
    report = build_ema(tmp_path, decay=1.0, last_n=3)

    assert report["checkpoints"] == 3
    assert report["steps"] == [100, 200, 300]
    payload = torch.load(tmp_path / EMA_NAME, map_location="cpu", weights_only=False)
    assert payload["step"] == 300
    assert torch.allclose(payload["model"]["weight"], torch.full((3, 3), 1.0))
    assert torch.equal(payload["model"]["counter"], torch.full((3,), 2, dtype=torch.long))
    assert payload["extra"]["ema_decay"] == 1.0
    assert payload["extra"]["ema_steps"] == [100, 200, 300]


def test_build_ema_keeps_the_latest_metadata(tmp_path):
    for step in (100, 200):
        write_checkpoint(tmp_path, step, 1.0)
    build_ema(tmp_path, decay=1.0, last_n=2)
    payload = torch.load(tmp_path / EMA_NAME, map_location="cpu", weights_only=False)
    assert payload["loader"]["position"] == 200
    assert payload["optimizer"]["state"] == 200
    assert payload["config"] == {"note": "unit"}


def test_last_n_overrides_the_decay_phase(tmp_path):
    for step in range(100, 1001, 100):
        write_checkpoint(tmp_path, step, 1.0)
    report = build_ema(tmp_path, decay=1.0, last_n=3, total_steps=1000)
    assert report["steps"] == [800, 900, 1000]


def test_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_ema(tmp_path)


def test_averaging_one_checkpoint_reproduces_it(tmp_path):
    write_checkpoint(tmp_path, 100, 7.0)
    build_ema(tmp_path, decay=0.999, last_n=1)
    payload = torch.load(tmp_path / EMA_NAME, map_location="cpu", weights_only=False)
    assert torch.allclose(payload["model"]["weight"], torch.full((3, 3), 7.0))


def test_the_averaged_model_loads_into_the_real_architecture(tmp_path):
    from dataclasses import replace as dataclass_replace

    from kv_scout import checkpoint as ckpt
    from kv_scout.config import DataConfig, OptimConfig, TrainConfig, proxy_config
    from kv_scout.data.synthetic import write_synthetic_corpus
    from kv_scout.model import KVScout
    from kv_scout.train.loop import train

    index = write_synthetic_corpus(
        tmp_path / "corpus", n_tokens=60_000, vocab_size=2048, shard_tokens=20_000
    )
    cfg = proxy_config(
        d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1, head_dim=96,
        dense_ffn_hidden=384, attention_anchor_layers=(3,),
        context_max=64, context_min=64,
    )
    run = tmp_path / "run"
    train(
        out_dir=run,
        data_index=index.root / "index.json",
        model_cfg=cfg,
        optim_cfg=dataclass_replace(
            OptimConfig(), matrix_optimizer="adamw", peak_lr=3e-4, warmup_steps=2
        ),
        data_cfg=DataConfig(seq_len=64, batch_size=2, seed=1),
        train_cfg=TrainConfig(
            steps=12, checkpoint_every=3, keep_last_checkpoints=10, seed=1,
            device="cpu", dtype="float32", out_dir=str(run),
        ),
        dropout=0.0,
    )
    report = build_ema(run, decay=0.999, total_steps=12, decay_fraction=0.5)
    assert report["checkpoints"] >= 2

    payload = torch.load(run / EMA_NAME, map_location="cpu", weights_only=False)
    model = KVScout(cfg)
    model.load_state_dict(payload["model"])
    with torch.no_grad():
        out = model(torch.randint(0, cfg.vocab_size, (1, 16)))
    assert out.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(out).all()


def test_the_average_differs_from_the_final_checkpoint(tmp_path):
    write_checkpoint(tmp_path, 100, 0.0)
    write_checkpoint(tmp_path, 200, 4.0)
    build_ema(tmp_path, decay=1.0, last_n=2)
    payload = torch.load(tmp_path / EMA_NAME, map_location="cpu", weights_only=False)
    assert torch.allclose(payload["model"]["weight"], torch.full((3, 3), 2.0))


def test_averaging_streams_rather_than_holding_every_checkpoint(tmp_path):
    from kv_scout.train.ema import Accumulator

    accumulator = Accumulator()
    for value, weight in ((0.0, 0.25), (4.0, 0.75)):
        accumulator.add({"w": torch.full((2, 2), value)}, weight)
    merged = accumulator.result()
    assert torch.allclose(merged["w"], torch.full((2, 2), 3.0))


def test_the_accumulator_rejects_mismatched_checkpoints():
    from kv_scout.train.ema import Accumulator

    accumulator = Accumulator()
    accumulator.add({"a": torch.zeros(2)}, 0.5)
    with pytest.raises(ValueError):
        accumulator.add({"b": torch.zeros(2)}, 0.5)

    other = Accumulator()
    other.add({"a": torch.zeros(2)}, 0.5)
    with pytest.raises(ValueError):
        other.add({"a": torch.zeros(3)}, 0.5)


def test_an_empty_accumulator_is_an_error():
    from kv_scout.train.ema import Accumulator

    with pytest.raises(ValueError):
        Accumulator().result()


def test_memory_does_not_grow_with_the_number_of_checkpoints(tmp_path):
    import gc
    import resource

    for step in range(1, 25):
        write_checkpoint(tmp_path, step * 10, float(step))

    def run(count):
        gc.collect()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        build_ema(tmp_path, decay=1.0, last_n=count)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before

    small = run(2)
    large = run(24)
    assert large <= max(small * 4, 64 * 1024 * 1024)


def test_averaging_many_checkpoints_stays_accurate(tmp_path):
    for step in range(1, 21):
        write_checkpoint(tmp_path, step * 10, float(step))
    build_ema(tmp_path, decay=1.0, last_n=20)
    payload = torch.load(tmp_path / EMA_NAME, map_location="cpu", weights_only=False)
    assert torch.allclose(payload["model"]["weight"], torch.full((3, 3), 10.5), atol=1e-6)


def test_tied_embeddings_stay_identical_after_averaging(tmp_path):
    from dataclasses import replace as dataclass_replace

    from kv_scout.config import DataConfig, OptimConfig, TrainConfig, proxy_config
    from kv_scout.data.synthetic import write_synthetic_corpus
    from kv_scout.model import KVScout
    from kv_scout.train.loop import train

    index = write_synthetic_corpus(
        tmp_path / "corpus", n_tokens=60_000, vocab_size=2048, shard_tokens=20_000
    )
    cfg = proxy_config(
        d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1, head_dim=96,
        dense_ffn_hidden=384, attention_anchor_layers=(3,),
        context_max=64, context_min=64,
    )
    run = tmp_path / "run"
    train(
        out_dir=run,
        data_index=index.root / "index.json",
        model_cfg=cfg,
        optim_cfg=dataclass_replace(
            OptimConfig(), matrix_optimizer="adamw", peak_lr=3e-4, warmup_steps=2
        ),
        data_cfg=DataConfig(seq_len=64, batch_size=2, seed=1),
        train_cfg=TrainConfig(
            steps=9, checkpoint_every=3, keep_last_checkpoints=10, seed=1,
            device="cpu", dtype="float32", out_dir=str(run),
        ),
        dropout=0.0,
    )
    build_ema(run, decay=0.999, last_n=3)
    payload = torch.load(run / EMA_NAME, map_location="cpu", weights_only=False)
    assert torch.equal(payload["model"]["embed.weight"], payload["model"]["head.weight"])

    model = KVScout(cfg)
    model.load_state_dict(payload["model"])
    assert model.head.weight is model.embed.weight
    assert torch.equal(model.head.weight, payload["model"]["embed.weight"])
