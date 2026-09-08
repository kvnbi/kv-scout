from __future__ import annotations

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import KVScout
from kv_scout.model.attention import GroupedQueryAttention
from kv_scout.model.block import TransformerBlock
from kv_scout.model.layers import rope_frequencies


@pytest.fixture
def cfg():
    return proxy_config(normalized_value_residual=True)


def test_first_layer_has_no_mixing_parameter(cfg):
    assert GroupedQueryAttention(cfg, layer=1).value_mix is None
    assert GroupedQueryAttention(cfg, layer=2).value_mix is not None


def test_disabled_by_default(cfg):
    assert GroupedQueryAttention(proxy_config(), layer=5).value_mix is None


def test_parameter_cost_is_one_scalar_per_later_layer(cfg):
    plain = proxy_config()
    extra = cfg.parameter_estimate() - plain.parameter_estimate()
    assert extra == cfg.n_layers - 1
    assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()


def test_attention_returns_its_own_values_as_the_source(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, layer=3)
    x = torch.randn(2, 6, cfg.d_model)
    _, source = attn(x)
    expected = attn.v_proj(x).view(2, 6, cfg.n_kv_heads, cfg.head_dim)
    assert torch.allclose(source, expected)


def test_the_source_is_the_unmixed_value(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, layer=3)
    x = torch.randn(2, 6, cfg.d_model)
    v_first = torch.randn(2, 6, cfg.n_kv_heads, cfg.head_dim)
    _, without = attn(x)
    _, with_residual = attn(x, v_first=v_first)
    assert torch.allclose(without, with_residual)


def test_mixing_is_a_convex_combination(cfg):
    attn = GroupedQueryAttention(cfg, layer=2)
    with torch.no_grad():
        attn.v_proj.weight.zero_()
    v_first = torch.ones(1, 4, cfg.n_kv_heads, cfg.head_dim)
    x = torch.randn(1, 4, cfg.d_model)

    for logit, expected in ((0.0, 0.5), (-4.0, 0.0180), (4.0, 0.9820)):
        with torch.no_grad():
            attn.value_mix.fill_(logit)
        alpha = float(torch.sigmoid(attn.value_mix.detach()))
        assert alpha == pytest.approx(expected, abs=1e-3)
        mixed = (1 - alpha) * torch.zeros_like(v_first) + alpha * v_first
        assert float(mixed.max()) <= 1.0


def test_mixing_starts_balanced(cfg):
    attn = GroupedQueryAttention(cfg, layer=2)
    assert float(torch.sigmoid(attn.value_mix.detach())) == pytest.approx(0.5)


def test_residual_changes_the_output(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, layer=2).eval()
    x = torch.randn(1, 6, cfg.d_model)
    v_first = torch.randn(1, 6, cfg.n_kv_heads, cfg.head_dim) * 3.0
    with torch.no_grad():
        plain, _ = attn(x)
        mixed, _ = attn(x, v_first=v_first)
    assert not torch.allclose(plain, mixed, atol=1e-4)


def test_block_threads_the_source_through(cfg):
    block = TransformerBlock(cfg, 4)
    x = torch.randn(2, 5, cfg.d_model)
    out, source = block(x)
    assert out.shape == x.shape
    assert source.shape == (2, 5, cfg.n_kv_heads, cfg.head_dim)


def test_model_feeds_the_first_layer_values_to_every_later_layer(cfg):
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    seen = []
    original = model.blocks[3].attn.forward

    def spy(x, cos=None, sin=None, v_first=None):
        seen.append(v_first)
        return original(x, cos, sin, v_first)

    model.blocks[3].attn.forward = spy
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 16)))

    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0].shape == (1, 16, cfg.n_kv_heads, cfg.head_dim)


def test_model_stays_causal_with_the_residual(cfg):
    torch.manual_seed(0)
    small = proxy_config(
        normalized_value_residual=True, d_model=192, n_layers=4, n_query_heads=2,
        n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(4,), context_max=64, context_min=64,
    )
    model = KVScout(small).eval()
    tokens = torch.randint(0, small.vocab_size, (1, 20))
    with torch.no_grad():
        base = model(tokens)
        changed = tokens.clone()
        changed[:, 12:] = (changed[:, 12:] + 7) % small.vocab_size
        after = model(changed)
    assert torch.allclose(base[:, :12], after[:, :12], atol=1e-5)


def test_gradients_reach_every_mixing_parameter(cfg):
    from kv_scout.model import language_model_loss

    small = proxy_config(
        normalized_value_residual=True, d_model=192, n_layers=4, n_query_heads=2,
        n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(4,), context_max=64, context_min=64,
    )
    model = KVScout(small)
    tokens = torch.randint(0, small.vocab_size, (2, 32))
    total, _ = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
    total.backward()
    for index, block in enumerate(model.blocks):
        if block.attn.value_mix is None:
            assert index == 0
            continue
        assert block.attn.value_mix.grad is not None
        assert float(block.attn.value_mix.grad.abs()) > 0
