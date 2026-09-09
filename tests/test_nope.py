from __future__ import annotations

import pytest
import torch

from kv_scout.config import ModelConfig, proxy_config
from kv_scout.model import KVScout
from kv_scout.model.block import TransformerBlock
from kv_scout.model.layers import rope_frequencies


@pytest.fixture
def hybrid():
    return proxy_config(use_gdn=True, nope_on_anchor_layers=True)


def test_only_anchor_layers_lose_position(hybrid):
    for layer in range(1, hybrid.n_layers + 1):
        kind = hybrid.layer_kind(layer)
        expected = kind != "anchor"
        assert TransformerBlock(hybrid, layer).uses_rope is expected, f"layer {layer} is {kind}"


def test_dense_warmup_keeps_position(hybrid):
    for layer in (1, 2):
        assert hybrid.layer_kind(layer) == "dense"
        assert TransformerBlock(hybrid, layer).uses_rope is True


def test_linear_layers_keep_position(hybrid):
    linear = [i for i in range(1, hybrid.n_layers + 1) if hybrid.layer_kind(i) == "linear"]
    assert len(linear) == hybrid.n_linear_layers
    for layer in linear:
        assert TransformerBlock(hybrid, layer).uses_rope is True


def test_switching_nope_off_restores_position_everywhere():
    cfg = proxy_config(use_gdn=True, nope_on_anchor_layers=False)
    for layer in range(1, cfg.n_layers + 1):
        assert TransformerBlock(cfg, layer).uses_rope is True


def test_an_anchor_ignores_the_rope_tables(hybrid):
    torch.manual_seed(0)
    anchor = TransformerBlock(hybrid, 6).eval()
    assert anchor.kind == "anchor"
    x = torch.randn(1, 12, hybrid.d_model)
    cos, sin = rope_frequencies(hybrid.head_dim, 12, hybrid.rope_theta)
    with torch.no_grad():
        without, _ = anchor(x)
        with_tables, _ = anchor(x, cos, sin)
    assert torch.allclose(without, with_tables, atol=1e-6)


def test_a_linear_layer_does_not_ignore_them(hybrid):
    torch.manual_seed(0)
    linear = TransformerBlock(hybrid, 5).eval()
    assert linear.kind == "linear"
    x = torch.randn(1, 12, hybrid.d_model)
    cos, sin = rope_frequencies(hybrid.head_dim, 12, hybrid.rope_theta)
    with torch.no_grad():
        without, _ = linear(x)
        with_tables, _ = linear(x, cos, sin)
    assert not torch.allclose(without, with_tables, atol=1e-4)


def test_an_anchor_is_still_order_sensitive_through_the_stack(hybrid):
    torch.manual_seed(0)
    small = proxy_config(
        use_gdn=True, nope_on_anchor_layers=True, d_model=192, n_layers=5,
        n_query_heads=2, n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(5,), context_max=32, context_min=32,
    )
    model = KVScout(small).eval()
    tokens = torch.randint(0, small.vocab_size, (1, 16))
    shuffled = tokens[:, torch.randperm(16)]
    with torch.no_grad():
        a = model(tokens)
        b = model(shuffled)
    assert not torch.allclose(a.mean(dim=1), b.mean(dim=1), atol=1e-3)


def test_the_full_model_stays_causal_with_nope():
    torch.manual_seed(0)
    small = proxy_config(
        use_gdn=True, nope_on_anchor_layers=True, d_model=192, n_layers=5,
        n_query_heads=2, n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(5,), context_max=32, context_min=32,
    )
    model = KVScout(small).eval()
    tokens = torch.randint(0, small.vocab_size, (1, 20))
    with torch.no_grad():
        base = model(tokens)
        changed = tokens.clone()
        changed[:, 12:] = (changed[:, 12:] + 7) % small.vocab_size
        after = model(changed)
    assert torch.allclose(base[:, :12], after[:, :12], atol=1e-4)


def test_nope_costs_no_parameters(hybrid):
    plain = proxy_config(use_gdn=True, nope_on_anchor_layers=False)
    assert hybrid.parameter_estimate() == plain.parameter_estimate()
    assert KVScout(hybrid).num_parameters() == KVScout(plain).num_parameters()


def test_the_full_spec_stack_applies_nope_to_eight_anchors():
    cfg = ModelConfig()
    assert cfg.nope_on_anchor_layers is True
    anchors = [i for i in range(1, cfg.n_layers + 1) if cfg.layer_kind(i) == "anchor"]
    assert anchors == list(cfg.attention_anchor_layers)
    assert len(anchors) == 8
    keeping = [i for i in range(1, cfg.n_layers + 1) if cfg.layer_kind(i) != "anchor"]
    assert len(keeping) == 22


def test_nope_applies_to_anchors_even_without_gdn():
    cfg = proxy_config(use_gdn=False, nope_on_anchor_layers=True)
    for layer in range(1, cfg.n_layers + 1):
        expected = cfg.layer_kind(layer) != "anchor"
        assert TransformerBlock(cfg, layer).uses_rope is expected


def test_the_model_runs_at_a_longer_context_than_it_was_configured_to_train_on():
    cfg = proxy_config(
        use_gdn=True, nope_on_anchor_layers=True, d_model=192, n_layers=5,
        n_query_heads=2, n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(5,), context_max=128, context_min=32,
    )
    model = KVScout(cfg).eval()
    with torch.no_grad():
        for length in (32, 64, 128):
            out = model(torch.randint(0, cfg.vocab_size, (1, length)))
            assert out.shape == (1, length, cfg.vocab_size)
