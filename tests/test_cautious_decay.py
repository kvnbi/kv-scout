from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kv_scout.config import OptimConfig, proxy_config
from kv_scout.model import KVScout, language_model_loss
from kv_scout.train.normuon import (
    CautiousAdamW,
    CombinedOptimizer,
    NorMuon,
    build_optimizer,
)


@pytest.fixture
def small():
    return proxy_config(
        d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1, head_dim=96,
        dense_ffn_hidden=384, attention_anchor_layers=(3,),
        context_max=64, context_min=64,
    )


def test_decay_is_decoupled_from_the_inner_optimizer():
    param = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = CautiousAdamW([param], lr=1e-3, weight_decay=0.1)
    group = optimizer.param_groups[0]
    assert group["weight_decay"] == 0.0
    assert group["decoupled_decay"] == 0.1


def test_cautious_decay_only_pulls_agreeing_coordinates():
    param = torch.nn.Parameter(torch.tensor([[2.0, 2.0, -2.0, -2.0]]))
    optimizer = CautiousAdamW([param], lr=1.0, weight_decay=0.1, betas=(0.0, 0.999))
    param.grad = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    before = param.detach().clone()
    optimizer.step()

    momentum = optimizer.state[param]["exp_avg"]
    agrees = (momentum * before) > 0
    assert agrees.tolist() == [[True, False, False, True]]

    plain = torch.nn.Parameter(before.clone())
    reference = CautiousAdamW([plain], lr=1.0, weight_decay=0.1, betas=(0.0, 0.999), cautious=False)
    plain.grad = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    reference.step()

    same = torch.isclose(param.detach(), plain.detach(), atol=1e-6)
    assert same.tolist() == [[True, False, False, True]]


def test_zero_decay_leaves_the_step_untouched():
    torch.manual_seed(0)
    grad = torch.randn(4, 6)
    results = []
    for cautious in (True, False):
        param = torch.nn.Parameter(torch.ones(4, 6))
        optimizer = CautiousAdamW([param], lr=1e-2, weight_decay=0.0, cautious=cautious)
        param.grad = grad.clone()
        optimizer.step()
        results.append(param.detach().clone())
    assert torch.allclose(results[0], results[1])


def test_cautious_decay_shrinks_less_than_plain_decay():
    torch.manual_seed(0)
    grad = torch.randn(32, 32)
    norms = {}
    for cautious in (True, False):
        param = torch.nn.Parameter(torch.randn(32, 32))
        torch.manual_seed(1)
        param.data.normal_()
        optimizer = CautiousAdamW([param], lr=1e-2, weight_decay=0.5, cautious=cautious)
        for _ in range(20):
            param.grad = grad.clone()
            optimizer.step()
        norms[cautious] = float(param.detach().norm())
    assert norms[True] > norms[False]


def test_cautious_fraction_is_a_proportion():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(16, 16))
    optimizer = CautiousAdamW([param], lr=1e-3, weight_decay=0.1)
    assert optimizer.cautious_fraction() == 0.0
    param.grad = torch.randn(16, 16)
    optimizer.step()
    fraction = optimizer.cautious_fraction()
    assert 0.0 < fraction < 1.0


def test_build_optimizer_uses_cautious_adamw_on_both_paths(small):
    model = KVScout(small)
    flat = build_optimizer(model, replace(OptimConfig(), matrix_optimizer="adamw"))
    assert isinstance(flat, CautiousAdamW)

    combined = build_optimizer(model, OptimConfig())
    assert isinstance(combined, CombinedOptimizer)
    assert isinstance(combined.vector, CautiousAdamW)
    assert isinstance(combined.matrix, NorMuon)
    assert combined.matrix.param_groups[0]["cautious"] is True


def test_the_cautious_flag_reaches_both_optimizers(small):
    model = KVScout(small)
    for flag in (True, False):
        combined = build_optimizer(
            model, replace(OptimConfig(), cautious_weight_decay=flag)
        )
        assert combined.matrix.param_groups[0]["cautious"] is flag
        assert combined.vector.param_groups[0]["cautious"] is flag


def test_normuon_cautious_decay_skips_disagreeing_coordinates():
    param = torch.nn.Parameter(torch.tensor([[3.0, -3.0], [3.0, -3.0]]))
    optimizer = NorMuon([param], lr=0.01, weight_decay=0.5, cautious=True, momentum=0.0)
    param.grad = torch.eye(2)
    before = param.detach().clone()
    optimizer.step()
    assert not torch.equal(param.detach(), before)


def test_z_loss_penalises_large_logits():
    logits = torch.zeros(2, 4, 100)
    logits[..., 0] = 20.0
    targets = torch.zeros(2, 4, dtype=torch.long)
    plain, plain_ce = language_model_loss(logits, targets, 0.0)
    total, ce = language_model_loss(logits, targets, 1e-3)
    assert float(ce) == pytest.approx(float(plain_ce))
    assert float(total) > float(plain)


def test_z_loss_grows_with_logit_magnitude():
    targets = torch.zeros(2, 4, dtype=torch.long)
    penalties = []
    for magnitude in (1.0, 10.0, 30.0):
        logits = torch.zeros(2, 4, 100)
        logits[..., 0] = magnitude
        total, ce = language_model_loss(logits, targets, 1e-3)
        penalties.append(float(total) - float(ce))
    assert penalties[0] < penalties[1] < penalties[2]


def test_z_loss_is_off_when_the_weight_is_zero():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 50) * 5
    targets = torch.randint(0, 50, (2, 4))
    total, ce = language_model_loss(logits, targets, 0.0)
    assert float(total) == pytest.approx(float(ce))


def test_training_runs_with_both_features(small):
    torch.manual_seed(0)
    model = KVScout(small)
    optimizer = build_optimizer(model, replace(OptimConfig(), matrix_optimizer="adamw", peak_lr=3e-3))
    tokens = torch.randint(0, small.vocab_size, (2, 32))
    first = last = None
    for _ in range(25):
        total, ce = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:], 1e-4)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        last = float(ce.detach())
        if first is None:
            first = last
    assert last < first * 0.6


def test_weight_decay_is_lost_in_pure_bfloat16(small):
    tokens = torch.randint(0, small.vocab_size, (2, 16))
    norms = {}
    for decay in (0.0, 1.0):
        torch.manual_seed(0)
        model = KVScout(small).to(dtype=torch.bfloat16)
        optimizer = build_optimizer(
            model,
            replace(
                OptimConfig(),
                matrix_optimizer="adamw",
                weight_decay=decay,
                cautious_weight_decay=False,
                peak_lr=3e-4,
            ),
        )
        for _ in range(10):
            model(tokens).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        norms[decay] = float(model.blocks[1].ffn.gate.weight.detach().float().norm())
    assert norms[0.0] == norms[1.0]


def test_weight_decay_survives_in_float32(small):
    tokens = torch.randint(0, small.vocab_size, (2, 16))
    norms = {}
    for decay in (0.0, 1.0):
        torch.manual_seed(0)
        model = KVScout(small)
        optimizer = build_optimizer(
            model,
            replace(
                OptimConfig(),
                matrix_optimizer="adamw",
                weight_decay=decay,
                cautious_weight_decay=False,
                peak_lr=3e-4,
            ),
        )
        for _ in range(10):
            model(tokens).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        norms[decay] = float(model.blocks[1].ffn.gate.weight.detach().norm())
    assert norms[1.0] < norms[0.0]


def test_the_trainer_keeps_parameters_in_float32(tmp_path):
    from dataclasses import replace as dataclass_replace

    from kv_scout.config import DataConfig, TrainConfig
    from kv_scout.data.synthetic import write_synthetic_corpus
    from kv_scout.train.loop import train

    index = write_synthetic_corpus(
        tmp_path / "corpus", n_tokens=60_000, vocab_size=2048, shard_tokens=20_000
    )
    cfg = proxy_config(
        d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1, head_dim=96,
        dense_ffn_hidden=384, attention_anchor_layers=(3,),
        context_max=64, context_min=64,
    )
    out = tmp_path / "run"
    train(
        out_dir=out,
        data_index=index.root / "index.json",
        model_cfg=cfg,
        optim_cfg=dataclass_replace(
            OptimConfig(), matrix_optimizer="adamw", peak_lr=3e-4, warmup_steps=2
        ),
        data_cfg=DataConfig(seq_len=64, batch_size=2, seed=1),
        train_cfg=TrainConfig(
            steps=4, checkpoint_every=4, seed=1, device="cpu",
            dtype="bfloat16", out_dir=str(out),
        ),
        dropout=0.0,
    )
    payload = torch.load(out / "step_00000004.pt", map_location="cpu", weights_only=False)
    assert all(v.dtype == torch.float32 for v in payload["model"].values())
    assert payload["config"]["parameter_dtype"] == "float32"
    assert payload["config"]["compute_dtype"] == "bfloat16"
