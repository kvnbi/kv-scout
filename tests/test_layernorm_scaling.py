from __future__ import annotations

import math

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import KVScout, RMSNorm
from kv_scout.model.block import TransformerBlock


@pytest.fixture
def cfg():
    return proxy_config(layernorm_scaling=True)


def test_disabled_by_default():
    assert TransformerBlock(proxy_config(), 7).norm_scale == 1.0


def test_scale_is_one_over_root_layer(cfg):
    for layer in (1, 4, 9, 12):
        assert TransformerBlock(cfg, layer).norm_scale == pytest.approx(
            1.0 / math.sqrt(layer)
        )


def test_first_layer_is_unscaled(cfg):
    assert TransformerBlock(cfg, 1).norm_scale == 1.0


def test_scaling_decreases_monotonically_with_depth(cfg):
    scales = [TransformerBlock(cfg, i).norm_scale for i in range(1, cfg.n_layers + 1)]
    assert all(a > b for a, b in zip(scales, scales[1:]))
    assert scales[-1] < 0.3


def test_both_norms_in_a_block_are_scaled(cfg):
    block = TransformerBlock(cfg, 9)
    assert block.attn_norm.scale == pytest.approx(1 / 3)
    assert block.ffn_norm.scale == pytest.approx(1 / 3)


def test_it_costs_no_parameters(cfg):
    assert cfg.parameter_estimate() == proxy_config().parameter_estimate()
    assert KVScout(cfg).num_parameters() == KVScout(proxy_config()).num_parameters()


def test_rmsnorm_scale_multiplies_the_normalised_output():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    plain = RMSNorm(16)
    scaled = RMSNorm(16, scale=0.25)
    with torch.no_grad():
        scaled.weight.copy_(plain.weight)
    assert torch.allclose(scaled(x), plain(x) * 0.25, atol=1e-6)


def test_deeper_layers_contribute_less_to_the_residual(cfg):
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.d_model)
    shallow = TransformerBlock(cfg, 2).eval()
    deep = TransformerBlock(cfg, 12).eval()
    with torch.no_grad():
        deep.load_state_dict(shallow.state_dict())
        shallow_delta = (shallow(x)[0] - x).norm()
        deep_delta = (deep(x)[0] - x).norm()
    assert deep_delta < shallow_delta


def test_the_final_norm_is_not_scaled(cfg):
    assert KVScout(cfg).final_norm.scale == 1.0


def test_model_trains_with_scaling(cfg):
    from kv_scout.model import language_model_loss

    torch.manual_seed(0)
    small = proxy_config(
        layernorm_scaling=True, d_model=192, n_layers=4, n_query_heads=2,
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
