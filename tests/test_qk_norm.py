from __future__ import annotations

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import GroupedQueryAttention, KVScout
from kv_scout.model.attention import GroupedQueryAttention as Attention
from kv_scout.model.layers import rope_frequencies


@pytest.fixture
def plain():
    return proxy_config(qk_norm=False)


@pytest.fixture
def normed():
    return proxy_config(qk_norm=True)


def test_disabled_by_default_in_the_proxy_baseline(plain):
    attn = GroupedQueryAttention(plain)
    assert attn.q_norm is None
    assert attn.k_norm is None


def test_enabled_creates_one_norm_for_queries_and_one_for_keys(normed):
    attn = GroupedQueryAttention(normed)
    assert attn.q_norm is not None and attn.k_norm is not None
    assert attn.q_norm.weight.shape == (normed.head_dim,)
    assert attn.k_norm.weight.shape == (normed.head_dim,)


def test_parameter_cost_is_two_vectors_per_layer(plain, normed):
    extra = normed.parameter_estimate() - plain.parameter_estimate()
    assert extra == 2 * normed.head_dim * normed.n_layers
    assert KVScout(normed).num_parameters() == normed.parameter_estimate()
    assert extra / normed.parameter_estimate() < 1e-4


def test_normalisation_is_per_head_not_across_the_model(normed):
    attn = Attention(normed)
    x = torch.randn(2, 7, normed.d_model)
    q = attn.q_proj(x).view(2, 7, normed.n_query_heads, normed.head_dim)
    out = attn.q_norm(q)
    scale = attn.q_norm.weight
    magnitudes = (out / scale).pow(2).mean(-1).sqrt()
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-4)


def test_it_tames_an_outlier_query(normed):
    attn = Attention(normed)
    q = torch.randn(1, 4, normed.n_query_heads, normed.head_dim)
    q[0, 2] *= 200.0
    before = q.norm(dim=-1)
    after = attn.q_norm(q).norm(dim=-1)
    assert before.max() / before.min() > 50
    assert after.max() / after.min() < 2


def test_output_shape_and_causality_are_unchanged(normed):
    torch.manual_seed(0)
    attn = Attention(normed).eval()
    x = torch.randn(1, 10, normed.d_model)
    cos, sin = rope_frequencies(normed.head_dim, 10, normed.rope_theta)
    with torch.no_grad():
        base, _ = attn(x, cos, sin)
        changed = x.clone()
        changed[:, 6:] += 5.0
        after, _ = attn(changed, cos, sin)
    assert base.shape == x.shape
    assert torch.allclose(base[:, :6], after[:, :6], atol=1e-5)
    assert not torch.allclose(base[:, 6:], after[:, 6:], atol=1e-3)


def test_gradients_reach_the_norms(normed):
    attn = Attention(normed)
    cos, sin = rope_frequencies(normed.head_dim, 6, normed.rope_theta)
    attn(torch.randn(2, 6, normed.d_model), cos, sin)[0].square().mean().backward()
    for name in ("q_norm", "k_norm"):
        grad = getattr(attn, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_values_are_left_alone(normed):
    torch.manual_seed(0)
    attn = Attention(normed)
    x = torch.randn(1, 5, normed.d_model)
    with torch.no_grad():
        v = attn.v_proj(x)
        again = attn.v_proj(x)
    assert torch.equal(v, again)
    assert not hasattr(attn, "v_norm")


def test_the_full_model_trains_with_qk_norm(normed):
    from kv_scout.model import language_model_loss

    torch.manual_seed(0)
    small = proxy_config(
        qk_norm=True, d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1,
        head_dim=96, dense_ffn_hidden=384, attention_anchor_layers=(3,),
        context_max=64, context_min=64,
    )
    model = KVScout(small)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(0, small.vocab_size, (2, 32))
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
