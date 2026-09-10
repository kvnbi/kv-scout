from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kv_scout.config import ModelConfig, OptimConfig, proxy_config
from kv_scout.model import KVScout, language_model_loss
from kv_scout.model.mtp import MTPHead, mtp_block_config, multi_token_loss


def small(**overrides):
    base = dict(
        mtp_heads=2, d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1,
        head_dim=96, dense_ffn_hidden=384, mtp_ffn_hidden=256,
        attention_anchor_layers=(3,), context_max=64, context_min=64,
    )
    base.update(overrides)
    return proxy_config(**base)


def test_the_spec_asks_for_two_heads():
    assert ModelConfig().mtp_heads == 2


def test_configuration_is_validated():
    with pytest.raises(ValueError):
        ModelConfig(mtp_heads=-1)
    with pytest.raises(ValueError):
        ModelConfig(mtp_ffn_hidden=0)
    with pytest.raises(ValueError):
        OptimConfig(mtp_loss_weight=-0.1)


def test_heads_can_be_switched_off():
    cfg = small(mtp_heads=0)
    model = KVScout(cfg)
    assert len(model.mtp) == 0
    assert cfg.mtp_parameter_estimate() == 0
    assert model.num_parameters() == cfg.parameter_estimate()
    assert cfg.parameter_estimate() == cfg.backbone_parameter_estimate()


def test_parameter_count_matches_the_estimate():
    for heads in (0, 1, 2, 3):
        cfg = small(mtp_heads=heads)
        assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()
        assert cfg.parameter_estimate() == (
            cfg.backbone_parameter_estimate() + cfg.mtp_parameter_estimate()
        )


def test_the_spec_budget_is_measured_on_the_backbone():
    cfg = ModelConfig()
    assert 3.25e9 < cfg.backbone_parameter_estimate() < 3.35e9
    assert cfg.parameter_estimate() > cfg.backbone_parameter_estimate()
    assert cfg.active_parameter_estimate() < cfg.backbone_parameter_estimate()


def test_the_backbone_count_excludes_the_heads():
    with_heads = KVScout(small(mtp_heads=2))
    without = KVScout(small(mtp_heads=0))
    assert with_heads.backbone_parameters() == without.num_parameters()
    assert with_heads.num_parameters() > with_heads.backbone_parameters()


def test_mtp_blocks_are_plain_dense_layers():
    cfg = mtp_block_config(small(use_gdn=True, use_moe=True))
    assert cfg.use_gdn is False
    assert cfg.use_moe is False
    assert cfg.attention_sinks is False
    assert cfg.cross_layer_kv_sharing is False
    assert cfg.layer_kind(1) == "dense"


def test_asking_for_predictions_does_not_change_the_main_output():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        plain = model(tokens)
        logits, ahead = model(tokens, predict_ahead=True)
    assert torch.equal(plain, logits)
    assert len(ahead) == cfg.mtp_heads


def test_a_model_without_heads_returns_only_logits():
    cfg = small(mtp_heads=0)
    model = KVScout(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, cfg.vocab_size, (1, 16)), predict_ahead=True)
    assert isinstance(out, torch.Tensor)


def test_each_depth_predicts_one_token_further_ahead():
    cfg = small()
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        _, ahead = model(tokens, predict_ahead=True)
    assert ahead[0].shape == (2, 31, cfg.vocab_size)
    assert ahead[1].shape == (2, 30, cfg.vocab_size)


def test_predictions_do_not_see_tokens_they_should_not():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 24))
    for boundary in (6, 12, 18):
        changed = tokens.clone()
        changed[:, boundary:] = (changed[:, boundary:] + 137) % cfg.vocab_size
        with torch.no_grad():
            _, before = model(tokens, predict_ahead=True)
            _, after = model(changed, predict_ahead=True)
        for depth, (a, b) in enumerate(zip(before, after), start=1):
            visible = boundary - depth
            assert torch.allclose(a[:, :visible], b[:, :visible], atol=1e-5)
            assert not torch.allclose(a[:, visible:], b[:, visible:], atol=1e-4)


def test_the_loss_aligns_each_depth_with_the_right_target():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    block = torch.randint(0, cfg.vocab_size, (2, 33))
    inputs, targets = block[:, :-1], block[:, 1:]
    with torch.no_grad():
        _, ahead = model(inputs, predict_ahead=True)
    _, parts = multi_token_loss(ahead, targets, 1.0)
    assert len(parts) == 2
    for value in parts:
        assert torch.isfinite(value)


def test_the_loss_is_zero_when_disabled():
    targets = torch.randint(0, 100, (2, 8))
    fake = [torch.randn(2, 7, 100)]
    total, parts = multi_token_loss(fake, targets, 0.0)
    assert float(total) == 0.0
    assert parts == []
    total, parts = multi_token_loss([], targets, 1.0)
    assert float(total) == 0.0


def test_the_loss_scales_with_its_weight():
    torch.manual_seed(0)
    targets = torch.randint(0, 100, (2, 8))
    fake = [torch.randn(2, 7, 100), torch.randn(2, 6, 100)]
    one, _ = multi_token_loss(fake, targets, 1.0)
    half, _ = multi_token_loss(fake, targets, 0.5)
    assert float(half) == pytest.approx(float(one) * 0.5, rel=1e-5)


def test_short_sequences_are_handled():
    cfg = small()
    model = KVScout(cfg).eval()
    for length in (1, 2, 3):
        with torch.no_grad():
            out = model(
                torch.randint(0, cfg.vocab_size, (1, length)), predict_ahead=True
            )
        logits, ahead = out if isinstance(out, tuple) else (out, [])
        assert logits.shape[1] == length
        assert len(ahead) <= cfg.mtp_heads


def test_gradients_reach_every_head():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    block = torch.randint(0, cfg.vocab_size, (2, 33))
    logits, ahead = model(block[:, :-1], predict_ahead=True)
    total, _ = language_model_loss(logits, block[:, 1:])
    extra, _ = multi_token_loss(ahead, block[:, 1:], 1.0)
    (total + extra).backward()
    for head in model.mtp:
        assert head.merge.weight.grad is not None
        assert float(head.merge.weight.grad.abs().sum()) > 0
        assert float(head.block.ffn.down.weight.grad.abs().sum()) > 0


def test_the_heads_learn_to_predict_ahead():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    block = torch.randint(0, cfg.vocab_size, (2, 33))
    inputs, targets = block[:, :-1], block[:, 1:]

    first = last = None
    for _ in range(40):
        logits, ahead = model(inputs, predict_ahead=True)
        total, _ = language_model_loss(logits, targets)
        extra, parts = multi_token_loss(ahead, targets, 1.0)
        optimizer.zero_grad(set_to_none=True)
        (total + extra).backward()
        optimizer.step()
        last = [float(p.detach()) for p in parts]
        if first is None:
            first = last
    for start, end in zip(first, last):
        assert end < start * 0.5


def test_training_ignores_the_heads_by_default():
    assert OptimConfig().mtp_loss_weight == 0.0


def test_heads_are_droppable_from_a_checkpoint():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    state = model.state_dict()
    stripped = {k: v for k, v in state.items() if not k.startswith("mtp.")}

    bare = KVScout(small(mtp_heads=0))
    bare.load_state_dict(stripped)
    assert bare.num_parameters() == model.backbone_parameters()
    with torch.no_grad():
        out = bare(torch.randint(0, cfg.vocab_size, (1, 16)))
    assert torch.isfinite(out).all()


def wide(**overrides):
    base = dict(
        use_mup=True, mtp_heads=2, d_model=1152, n_query_heads=12, n_kv_heads=4,
        head_dim=96, dense_ffn_hidden=3072, mtp_ffn_hidden=768,
        context_max=64, context_min=64,
    )
    base.update(overrides)
    return proxy_config(**base)


def test_mtp_hidden_weights_carry_the_mup_scale():
    cfg = wide()
    model = KVScout(cfg)
    expected = 1.0 / cfg.width_multiplier
    assert expected != 1.0
    for head in model.mtp:
        for weight in head.hidden_weights():
            assert getattr(weight, "mup_lr_scale") == pytest.approx(expected)


def test_mtp_norms_keep_the_base_learning_rate():
    model = KVScout(wide())
    for head in model.mtp:
        assert getattr(head.hidden_norm.weight, "mup_lr_scale") == 1.0
        assert getattr(head.token_norm.weight, "mup_lr_scale") == 1.0


def test_the_optimizer_sees_the_mtp_scale():
    from dataclasses import replace as dataclass_replace

    from kv_scout.train.normuon import build_optimizer

    cfg = wide()
    model = KVScout(cfg)
    optimizer = build_optimizer(
        model, dataclass_replace(OptimConfig(), matrix_optimizer="adamw", peak_lr=1e-3)
    )
    scales = sorted({round(g["lr_scale"], 4) for g in optimizer.param_groups})
    assert round(1.0 / cfg.width_multiplier, 4) in scales
    assert 1.0 in scales


def test_mtp_output_projections_are_damped():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    for head in model.mtp:
        with torch.no_grad():
            assert float(head.block.attn.o_proj.weight.std()) < float(
                head.block.attn.q_proj.weight.std()
            )
            assert float(head.block.ffn.down.weight.std()) < float(
                head.block.ffn.gate.weight.std()
            )


def test_mtp_initialisation_scales_with_width():
    from kv_scout.model.transformer import BASE_INIT_STD
    import math

    narrow = KVScout(wide(d_model=576, n_query_heads=6, n_kv_heads=2,
                          dense_ffn_hidden=1536, mtp_ffn_hidden=384))
    broad = KVScout(wide())
    with torch.no_grad():
        a = float(narrow.mtp[0].merge.weight.std())
        b = float(broad.mtp[0].merge.weight.std())
    assert a > b
    assert b == pytest.approx(BASE_INIT_STD / math.sqrt(2.0), rel=0.1)


def test_generation_with_a_cache_is_unaffected_by_the_heads():
    from kv_scout.model import Cache

    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 20))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        step = torch.cat([model(tokens[:, i : i + 1], cache) for i in range(20)], dim=1)
    assert torch.allclose(full, step, atol=1e-3)


def test_a_single_token_input_produces_no_predictions():
    cfg = small()
    model = KVScout(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, cfg.vocab_size, (1, 1)), predict_ahead=True)
    logits, ahead = out if isinstance(out, tuple) else (out, [])
    assert logits.shape[1] == 1
    assert ahead == []
