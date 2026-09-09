from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from kv_scout.config import OptimConfig, proxy_config
from kv_scout.model import KVScout
from kv_scout.model.transformer import BASE_INIT_STD
from kv_scout.train.normuon import build_optimizer, group_by_lr_scale


def width(q: int, **overrides):
    d = q * 96
    base = dict(
        use_mup=True, d_model=d, n_query_heads=q, n_kv_heads=q // 3, head_dim=96,
        dense_ffn_hidden=d * 8 // 3, context_max=64, context_min=64,
    )
    base.update(overrides)
    return proxy_config(**base)


def test_disabled_by_default():
    cfg = proxy_config()
    assert cfg.use_mup is False
    assert cfg.width_multiplier == 1.0
    assert cfg.attention_scale is None


def test_width_multiplier_is_relative_to_the_base():
    assert width(3).width_multiplier == pytest.approx(0.5)
    assert width(6).width_multiplier == pytest.approx(1.0)
    assert width(18).width_multiplier == pytest.approx(3.0)


def test_attention_uses_one_over_head_dim():
    cfg = width(6)
    assert cfg.attention_scale == pytest.approx(1.0 / cfg.head_dim)
    assert proxy_config().attention_scale is None


def test_mup_base_width_is_validated():
    with pytest.raises(ValueError):
        proxy_config(mup_base_d_model=0)


def test_embedding_init_is_width_invariant():
    for q in (3, 6, 18):
        model = KVScout(width(q))
        with torch.no_grad():
            assert float(model.embed.weight.std()) == pytest.approx(BASE_INIT_STD, rel=0.05)


def test_hidden_init_shrinks_with_the_square_root_of_width():
    for q in (3, 6, 18):
        cfg = width(q)
        model = KVScout(cfg)
        expected = BASE_INIT_STD / math.sqrt(cfg.width_multiplier)
        with torch.no_grad():
            assert float(model.blocks[3].ffn.gate.weight.std()) == pytest.approx(
                expected, rel=0.05
            )


def test_hidden_learning_rate_scales_inversely_with_width():
    for q, expected in ((3, 2.0), (6, 1.0), (18, 1.0 / 3.0)):
        model = KVScout(width(q))
        optimizer = build_optimizer(
            model, replace(OptimConfig(), matrix_optimizer="adamw", peak_lr=1e-3)
        )
        scales = {round(g["lr_scale"], 4) for g in optimizer.param_groups}
        assert round(expected, 4) in scales
        assert 1.0 in scales


def test_embedding_and_norms_keep_the_base_learning_rate():
    model = KVScout(width(18))
    assert getattr(model.embed.weight, "mup_lr_scale") == 1.0
    assert getattr(model.blocks[3].attn_norm.weight, "mup_lr_scale") == 1.0
    assert getattr(model.blocks[3].ffn.gate.weight, "mup_lr_scale") == pytest.approx(1 / 3)


def test_tied_readout_uses_the_square_root_divisor():
    tied = KVScout(width(18))
    assert tied.readout_divisor == pytest.approx(math.sqrt(3.0))
    untied = KVScout(width(18, tie_embeddings=False))
    assert untied.readout_divisor == pytest.approx(3.0)


def test_logit_scale_is_width_invariant():
    values = []
    for q in (3, 6, 9, 18):
        cfg = width(q)
        torch.manual_seed(0)
        model = KVScout(cfg).eval()
        with torch.no_grad():
            logits = model(torch.randint(0, cfg.vocab_size, (2, 32)))
        values.append(float(logits.std()))
    assert max(values) / min(values) < 1.1


def test_activations_are_width_invariant_at_initialisation():
    deepest = []
    for q in (3, 6, 9, 18):
        cfg = width(q)
        torch.manual_seed(0)
        model = KVScout(cfg).eval()
        captured = {}
        handle = model.blocks[11].register_forward_hook(
            lambda m, inp, out: captured.__setitem__("x", float(out[0].detach().abs().mean()))
        )
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (2, 32)))
        handle.remove()
        deepest.append(captured["x"])
    assert max(deepest) / min(deepest) < 1.2


def test_without_mup_activations_grow_with_width():
    deepest = []
    for q in (3, 18):
        cfg = width(q, use_mup=False)
        torch.manual_seed(0)
        model = KVScout(cfg).eval()
        captured = {}
        handle = model.blocks[11].register_forward_hook(
            lambda m, inp, out: captured.__setitem__("x", float(out[0].detach().abs().mean()))
        )
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (2, 32)))
        handle.remove()
        deepest.append(captured["x"])
    assert deepest[1] / deepest[0] > 2.0


def test_group_by_lr_scale_partitions_parameters():
    a = torch.nn.Parameter(torch.randn(2, 2))
    b = torch.nn.Parameter(torch.randn(2, 2))
    c = torch.nn.Parameter(torch.randn(2))
    a.mup_lr_scale = 0.5
    b.mup_lr_scale = 0.5
    groups = group_by_lr_scale([a, b, c])
    assert len(groups) == 2
    by_scale = {g["mup_scale"]: g for g in groups}
    assert len(by_scale[0.5]["params"]) == 2
    assert len(by_scale[1.0]["params"]) == 1


def test_normuon_combines_mup_and_matrix_multipliers():
    model = KVScout(width(18))
    cfg = replace(OptimConfig(), matrix_optimizer="normuon", matrix_lr_multiplier=30.0)
    optimizer = build_optimizer(model, cfg)
    matrix_scales = {round(g["lr_scale"], 4) for g in optimizer.matrix.param_groups}
    assert round(30.0 / 3.0, 4) in matrix_scales
    assert all(g["lr_scale"] == 1.0 for g in optimizer.vector.param_groups)


def test_parameter_counts_are_unaffected():
    for q in (3, 6, 18):
        cfg = width(q)
        assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()


def test_mup_model_trains():
    from kv_scout.model import language_model_loss

    cfg = width(6, n_layers=3, attention_anchor_layers=(3,))
    torch.manual_seed(0)
    model = KVScout(cfg)
    optimizer = build_optimizer(
        model, replace(OptimConfig(), matrix_optimizer="adamw", peak_lr=1e-3)
    )
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    first = last = None
    for _ in range(25):
        total, ce = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        last = float(ce.detach())
        if first is None:
            first = last
    assert last < first * 0.6
