from __future__ import annotations

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import KVScout
from kv_scout.model.block import TransformerBlock
from kv_scout.model.gdn import DECAY_OPEN_BIAS, GatedDeltaNet, delta_rule_recurrence
from kv_scout.model.layers import rope_frequencies


@pytest.fixture
def cfg():
    return proxy_config(use_gdn=True)


def test_layers_become_linear_when_gdn_is_on(cfg):
    kinds = [cfg.layer_kind(i) for i in range(1, cfg.n_layers + 1)]
    assert kinds.count("linear") == cfg.n_linear_layers
    assert kinds.count("anchor") == cfg.n_anchor_layers
    assert kinds[:2] == ["dense", "dense"]


def test_block_picks_the_right_module(cfg):
    assert isinstance(TransformerBlock(cfg, 3).attn, GatedDeltaNet)
    assert not isinstance(TransformerBlock(cfg, 6).attn, GatedDeltaNet)
    assert not isinstance(TransformerBlock(cfg, 1).attn, GatedDeltaNet)


def test_state_size_does_not_depend_on_sequence_length(cfg):
    layer = GatedDeltaNet(cfg, 3)
    sizes = []
    for length in (16, 64, 256):
        _, _, state = layer(torch.randn(2, length, cfg.d_model))
        sizes.append(tuple(state.shape))
    assert len(set(sizes)) == 1
    assert sizes[0] == (2, cfg.n_query_heads, cfg.head_dim, cfg.head_dim)


def test_the_recurrence_is_causal(cfg):
    torch.manual_seed(0)
    layer = GatedDeltaNet(cfg, 3).eval()
    x = torch.randn(1, 12, cfg.d_model)
    with torch.no_grad():
        base, _, _ = layer(x)
        changed = x.clone()
        changed[:, 7:] += 4.0
        after, _, _ = layer(changed)
    assert torch.allclose(base[:, :7], after[:, :7], atol=1e-4)
    assert not torch.allclose(base[:, 7:], after[:, 7:], atol=1e-3)


def test_resuming_from_a_carried_state_matches_one_pass():
    torch.manual_seed(0)
    b, h, t, d = 2, 3, 20, 16
    q, k, v = (torch.randn(b, h, t, d) for _ in range(3))
    k = torch.nn.functional.normalize(k, dim=-1)
    q = torch.nn.functional.normalize(q, dim=-1)
    decay = torch.sigmoid(torch.randn(b, h, t))
    write = torch.sigmoid(torch.randn(b, h, t))

    whole, final = delta_rule_recurrence(q, k, v, decay, write)
    first, carried = delta_rule_recurrence(
        q[:, :, :8], k[:, :, :8], v[:, :, :8], decay[:, :, :8], write[:, :, :8]
    )
    second, after = delta_rule_recurrence(
        q[:, :, 8:], k[:, :, 8:], v[:, :, 8:], decay[:, :, 8:], write[:, :, 8:], carried
    )
    assert torch.allclose(whole, torch.cat([first, second], dim=2), atol=1e-5)
    assert torch.allclose(final, after, atol=1e-5)


def test_the_delta_rule_replaces_rather_than_accumulates():
    b, h, d = 1, 1, 8
    key = torch.zeros(1, 1, 2, d)
    key[..., 0] = 1.0
    value = torch.zeros(1, 1, 2, d)
    value[:, :, 0, 1] = 1.0
    value[:, :, 1, 1] = 5.0
    query = key.clone()
    decay = torch.ones(b, h, 2)
    write = torch.ones(b, h, 2)

    out, _ = delta_rule_recurrence(query, key, value, decay, write)
    assert float(out[0, 0, 1, 1]) == pytest.approx(5.0, abs=1e-5)


def test_full_decay_forgets_the_state():
    b, h, d = 1, 1, 4
    key = torch.zeros(1, 1, 2, d)
    key[..., 0] = 1.0
    value = torch.zeros(1, 1, 2, d)
    value[:, :, 0, 1] = 9.0
    query = key.clone()
    write = torch.zeros(b, h, 2)
    write[:, :, 0] = 1.0

    kept, _ = delta_rule_recurrence(query, key, value, torch.ones(b, h, 2), write)
    forgotten, _ = delta_rule_recurrence(query, key, value, torch.zeros(b, h, 2), write)
    assert float(kept[0, 0, 1, 1]) == pytest.approx(9.0, abs=1e-5)
    assert float(forgotten[0, 0, 1, 1]) == pytest.approx(0.0, abs=1e-5)


def test_decay_starts_close_to_remembering_everything(cfg):
    torch.manual_seed(0)
    layer = GatedDeltaNet(cfg, 3)
    with torch.no_grad():
        values = torch.sigmoid(layer.decay_proj(torch.randn(64, 32, cfg.d_model)))
    centre = float(torch.sigmoid(torch.tensor(DECAY_OPEN_BIAS)))
    assert centre > 0.95
    assert float(values.mean()) == pytest.approx(centre, abs=0.02)
    assert float(values.min()) > 0.5


def test_heads_start_with_a_spread_of_memory_timescales(cfg):
    torch.manual_seed(0)
    layer = GatedDeltaNet(cfg, 3)
    with torch.no_grad():
        values = torch.sigmoid(layer.decay_proj(torch.randn(32, 16, cfg.d_model)))
    assert float(values.std()) > 0.005
    assert float(values.max() - values.min()) > 0.05


def test_write_strength_starts_balanced(cfg):
    torch.manual_seed(0)
    layer = GatedDeltaNet(cfg, 3)
    with torch.no_grad():
        values = torch.sigmoid(layer.write_proj(torch.randn(64, 32, cfg.d_model)))
    assert float(values.mean()) == pytest.approx(0.5, abs=0.02)


def test_gdn_returns_its_values_for_the_residual(cfg):
    layer = GatedDeltaNet(cfg, 3)
    x = torch.randn(2, 6, cfg.d_model)
    _, source, _ = layer(x)
    assert source.shape == (2, 6, cfg.n_kv_heads, cfg.head_dim)


def test_value_residual_reaches_gdn_layers(cfg):
    with_residual = proxy_config(use_gdn=True, normalized_value_residual=True)
    assert GatedDeltaNet(with_residual, 3).value_mix is not None
    assert GatedDeltaNet(with_residual, 1).value_mix is None
    assert GatedDeltaNet(cfg, 3).value_mix is None


def test_rope_is_applied_on_linear_layers(cfg):
    torch.manual_seed(0)
    layer = GatedDeltaNet(cfg, 3).eval()
    x = torch.randn(1, 10, cfg.d_model)
    cos, sin = rope_frequencies(cfg.head_dim, 10, cfg.rope_theta)
    with torch.no_grad():
        without, _, _ = layer(x)
        with_rope, _, _ = layer(x, cos, sin)
    assert not torch.allclose(without, with_rope, atol=1e-4)


def test_gradients_reach_every_gdn_parameter(cfg):
    layer = GatedDeltaNet(cfg, 3)
    layer(torch.randn(2, 8, cfg.d_model))[0].square().mean().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_parameter_count_matches_the_estimate(cfg):
    assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()


def test_the_hybrid_stack_trains():
    from kv_scout.model import language_model_loss

    torch.manual_seed(0)
    small = proxy_config(
        use_gdn=True, d_model=192, n_layers=5, n_query_heads=2, n_kv_heads=1,
        head_dim=96, dense_ffn_hidden=384, attention_anchor_layers=(5,),
        context_max=32, context_min=32,
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


def test_the_output_gate_can_be_switched_off():
    gated = proxy_config(use_gdn=True, gdn_output_gate=True)
    plain = proxy_config(use_gdn=True, gdn_output_gate=False)
    assert GatedDeltaNet(gated, 3).gate_proj is not None
    assert GatedDeltaNet(plain, 3).gate_proj is None
    assert KVScout(plain).num_parameters() == plain.parameter_estimate()
    saved = gated.parameter_estimate() - plain.parameter_estimate()
    assert saved == gated.d_model**2 * gated.n_linear_layers


def test_the_ungated_layer_still_runs(cfg):
    plain = proxy_config(use_gdn=True, gdn_output_gate=False)
    layer = GatedDeltaNet(plain, 3)
    out, _, state = layer(torch.randn(2, 8, plain.d_model))
    assert out.shape == (2, 8, plain.d_model)
    assert state.shape == (2, plain.n_query_heads, plain.head_dim, plain.head_dim)
