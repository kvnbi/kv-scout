from __future__ import annotations

import pytest

from kv_scout.config import MoEConfig, ModelConfig, PROXY_ANCHORS, proxy_config


def test_proxy_sits_in_the_phase_a_size_band():
    cfg = proxy_config()
    params = cfg.parameter_estimate()
    assert 55e6 < params < 200e6
    assert cfg.active_parameter_estimate() == params


def test_proxy_shape():
    cfg = proxy_config()
    assert cfg.d_model == 576
    assert cfg.n_layers == 12
    assert cfg.n_query_heads * cfg.head_dim == cfg.d_model
    assert cfg.kv_group_size == 3
    assert cfg.attention_anchor_layers == PROXY_ANCHORS


def test_proxy_starts_as_the_bare_baseline():
    cfg = proxy_config()
    for flag in (
        cfg.use_gdn,
        cfg.use_moe,
        cfg.qk_norm,
        cfg.normalized_value_residual,
        cfg.layernorm_scaling,
        cfg.per_head_gated_attention,
        cfg.nope_on_anchor_layers,
        cfg.cross_layer_kv_sharing,
        cfg.attention_sinks,
    ):
        assert flag is False


def test_layer_kinds_follow_the_stack():
    cfg = proxy_config(use_gdn=True)
    kinds = [cfg.layer_kind(i) for i in range(1, cfg.n_layers + 1)]
    assert kinds[:2] == ["dense", "dense"]
    assert [i for i, k in enumerate(kinds, 1) if k == "anchor"] == list(PROXY_ANCHORS)
    assert kinds.count("linear") == cfg.n_linear_layers


def test_linear_layers_become_attention_when_gdn_is_off():
    cfg = proxy_config(use_gdn=False)
    kinds = [cfg.layer_kind(i) for i in range(1, cfg.n_layers + 1)]
    assert "linear" not in kinds
    assert kinds.count("attention") == cfg.n_linear_layers


def test_layer_kind_rejects_out_of_range():
    cfg = proxy_config()
    for bad in (0, 13):
        with pytest.raises(ValueError):
            cfg.layer_kind(bad)


def test_ffn_hidden_switches_with_moe():
    dense = proxy_config()
    assert dense.ffn_hidden(1) == dense.dense_ffn_hidden
    assert dense.ffn_hidden(5) == dense.dense_ffn_hidden

    moe = proxy_config(use_moe=True)
    assert moe.ffn_hidden(1) == moe.dense_ffn_hidden
    assert moe.ffn_hidden(5) == moe.moe.expert_ffn_hidden


def test_moe_raises_total_but_not_active():
    dense = proxy_config()
    moe = proxy_config(use_moe=True)
    assert moe.parameter_estimate() > dense.parameter_estimate()
    assert moe.active_parameter_estimate() < moe.parameter_estimate()


def test_overrides_are_applied_and_validated():
    assert proxy_config(d_model=1152, head_dim=192).d_model == 1152
    with pytest.raises(ValueError):
        proxy_config(head_dim=128)
    with pytest.raises(ValueError):
        proxy_config(attention_anchor_layers=(1,))
    with pytest.raises(ValueError):
        proxy_config(n_kv_heads=5)


def test_full_config_active_params_match_the_spec():
    cfg = ModelConfig()
    active = cfg.active_parameter_estimate()
    assert 1.1e9 < active < 1.35e9
    assert cfg.linear_to_anchor_ratio == pytest.approx(2.5)


def test_interleave_ratio_is_close_to_the_spec_target():
    assert 2.0 < proxy_config().linear_to_anchor_ratio < 3.0


def test_moe_config_validation():
    with pytest.raises(ValueError):
        MoEConfig(num_experts=2, top_k=2)
