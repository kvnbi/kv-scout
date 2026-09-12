from __future__ import annotations

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import KVScout
from kv_scout.model.attention import GATE_OPEN_BIAS, GroupedQueryAttention
from kv_scout.model.layers import rope_frequencies


@pytest.fixture
def cfg():
    return proxy_config(per_head_gated_attention=True)


def test_disabled_by_default():
    assert GroupedQueryAttention(proxy_config(), 3).head_gate is None


def test_one_gate_per_query_head(cfg):
    gate = GroupedQueryAttention(cfg, 3).head_gate
    assert gate.out_features == cfg.n_query_heads
    assert gate.in_features == cfg.d_model


def test_gates_start_almost_open(cfg):
    attn = GroupedQueryAttention(cfg, 3)
    values = torch.sigmoid(attn.head_gate(torch.randn(4, 9, cfg.d_model)))
    expected = float(torch.sigmoid(torch.tensor(GATE_OPEN_BIAS)))
    assert torch.allclose(values, torch.full_like(values, expected), atol=1e-5)
    assert expected > 0.95


def test_gates_survive_model_assembly(cfg):
    torch.manual_seed(0)
    model = KVScout(proxy_config(per_head_gated_attention=True, mtp_heads=1, mtp_ffn_hidden=128))
    gates = [m for m in model.modules() if isinstance(m, GroupedQueryAttention)]
    assert len(gates) > 1
    for attn in gates:
        assert torch.equal(attn.head_gate.weight, torch.zeros_like(attn.head_gate.weight))
        assert torch.equal(attn.head_gate.bias, torch.full_like(attn.head_gate.bias, GATE_OPEN_BIAS))


def test_parameter_cost(cfg):
    plain = proxy_config()
    extra = cfg.parameter_estimate() - plain.parameter_estimate()
    per_layer = cfg.d_model * cfg.n_query_heads + cfg.n_query_heads
    assert extra == per_layer * cfg.n_layers
    assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()
    assert extra / plain.parameter_estimate() < 0.001


def test_a_closed_gate_silences_only_its_own_head(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, 3).eval()
    with torch.no_grad():
        attn.head_gate.bias.fill_(GATE_OPEN_BIAS)
        attn.head_gate.bias[2] = -30.0
        attn.o_proj.weight.copy_(torch.eye(cfg.d_model))

    x = torch.randn(1, 6, cfg.d_model)
    with torch.no_grad():
        out, _ = attn(x)
    per_head = out.view(1, 6, cfg.n_query_heads, cfg.head_dim)
    magnitudes = per_head.abs().mean(dim=(0, 1, 3))

    assert magnitudes[2] < 1e-6
    assert (magnitudes[torch.arange(cfg.n_query_heads) != 2] > 1e-4).all()


def test_heads_can_receive_different_gate_values(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, 3)
    with torch.no_grad():
        attn.head_gate.weight.normal_(0.0, 0.5)
    values = torch.sigmoid(attn.head_gate(torch.randn(1, 5, cfg.d_model)))
    assert values.std(dim=-1).min() > 1e-3


def test_gating_is_applied_after_attention_not_to_the_values(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, 3).eval()
    x = torch.randn(1, 6, cfg.d_model)
    with torch.no_grad():
        _, source_open = attn(x)
        attn.head_gate.bias.fill_(-30.0)
        _, source_closed = attn(x)
    assert torch.allclose(source_open, source_closed)


def test_causality_is_preserved(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg, 3).eval()
    with torch.no_grad():
        attn.head_gate.weight.normal_(0.0, 0.5)
    x = torch.randn(1, 10, cfg.d_model)
    cos, sin = rope_frequencies(cfg.head_dim, 10, cfg.rope_theta)
    with torch.no_grad():
        base, _ = attn(x, cos, sin)
        changed = x.clone()
        changed[:, 6:] += 5.0
        after, _ = attn(changed, cos, sin)
    assert torch.allclose(base[:, :6], after[:, :6], atol=1e-5)
    assert not torch.allclose(base[:, 6:], after[:, 6:], atol=1e-3)


def test_gradients_reach_the_gate(cfg):
    attn = GroupedQueryAttention(cfg, 3)
    cos, sin = rope_frequencies(cfg.head_dim, 6, cfg.rope_theta)
    attn(torch.randn(2, 6, cfg.d_model), cos, sin)[0].square().mean().backward()
    assert attn.head_gate.weight.grad is not None
    assert attn.head_gate.weight.grad.abs().sum() > 0
    assert attn.head_gate.bias.grad.abs().sum() > 0


def test_all_four_interventions_compose():
    cfg = proxy_config(
        qk_norm=True,
        normalized_value_residual=True,
        layernorm_scaling=True,
        per_head_gated_attention=True,
    )
    model = KVScout(cfg)
    assert model.num_parameters() == cfg.parameter_estimate()
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    assert model(tokens).shape == (2, 32, cfg.vocab_size)


def test_the_full_set_trains():
    from kv_scout.model import language_model_loss

    torch.manual_seed(0)
    small = proxy_config(
        qk_norm=True, normalized_value_residual=True, layernorm_scaling=True,
        per_head_gated_attention=True, d_model=192, n_layers=4, n_query_heads=2,
        n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(4,), context_max=64, context_min=64,
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


def test_the_gate_is_a_near_identity_at_initialisation(cfg):
    torch.manual_seed(0)
    gated = GroupedQueryAttention(cfg, 3).eval()
    plain = GroupedQueryAttention(proxy_config(), 3).eval()
    with torch.no_grad():
        plain.load_state_dict(
            {k: v for k, v in gated.state_dict().items() if not k.startswith("head_gate")}
        )
        x = torch.randn(1, 6, cfg.d_model)
        with_gate, _ = gated(x)
        without, _ = plain(x)
    expected = float(torch.sigmoid(torch.tensor(GATE_OPEN_BIAS)))
    assert torch.allclose(with_gate, without * expected, atol=1e-4)
