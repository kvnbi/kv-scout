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


def test_full_config_matches_the_spec_headline_numbers():
    cfg = ModelConfig(gdn_output_gate=False)
    total = cfg.parameter_estimate()
    active = cfg.active_parameter_estimate()
    assert 1.15e9 < active < 1.30e9
    assert 3.25e9 < total < 3.35e9
    assert 6.5 < total * 2 / 1e9 < 6.7
    assert cfg.linear_to_anchor_ratio == pytest.approx(2.5)


def test_the_gdn_output_gate_costs_one_matrix_per_linear_layer():
    with_gate = ModelConfig(gdn_output_gate=True)
    without = ModelConfig(gdn_output_gate=False)
    extra = with_gate.parameter_estimate() - without.parameter_estimate()
    assert extra == with_gate.d_model**2 * with_gate.n_linear_layers
    assert with_gate.active_parameter_estimate() > without.active_parameter_estimate()


def test_expert_count_is_the_reconciled_value():
    from kv_scout.config import MoEConfig

    assert MoEConfig().num_experts == 12
    assert MoEConfig().top_k == 2
    assert MoEConfig().shared_experts == 1
    assert MoEConfig().expert_ffn_hidden == 1280


def test_interleave_ratio_is_close_to_the_spec_target():
    assert 2.0 < proxy_config().linear_to_anchor_ratio < 3.0


def test_moe_config_validation():
    with pytest.raises(ValueError):
        MoEConfig(num_experts=2, top_k=2)


def test_config_round_trips_through_a_checkpoint_payload():
    from kv_scout.config import from_dict, to_dict

    for cfg in (ModelConfig(), proxy_config(), proxy_config(use_moe=True, use_gdn=True)):
        payload = to_dict(cfg)
        restored = from_dict(ModelConfig, payload)
        assert restored == cfg
        assert isinstance(restored.tokenizer, type(cfg.tokenizer))
        assert isinstance(restored.moe, type(cfg.moe))
        assert restored.attention_anchor_layers == cfg.attention_anchor_layers
        assert restored.parameter_estimate() == cfg.parameter_estimate()


def test_from_dict_ignores_unknown_keys_and_rejects_non_dataclasses():
    from kv_scout.config import from_dict, to_dict

    payload = to_dict(proxy_config())
    payload["not_a_field"] = 1
    assert from_dict(ModelConfig, payload) == proxy_config()
    with pytest.raises(TypeError):
        from_dict(dict, {})
