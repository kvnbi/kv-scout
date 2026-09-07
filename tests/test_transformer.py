from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import KVScout, language_model_loss
from kv_scout.train.loop import build_model


@pytest.fixture(scope="module")
def small():
    return proxy_config(
        d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1, head_dim=96,
        dense_ffn_hidden=512, attention_anchor_layers=(3,), context_max=128,
        context_min=128,
    )


def test_parameter_count_matches_the_config_estimate():
    cfg = proxy_config()
    assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()


def test_forward_shape(small):
    model = KVScout(small)
    tokens = torch.randint(0, small.vocab_size, (2, 32))
    assert model(tokens).shape == (2, 32, small.vocab_size)


def test_embeddings_are_tied(small):
    model = KVScout(small)
    assert model.head.weight is model.embed.weight

    untied = KVScout(replace(small, tie_embeddings=False))
    assert untied.head.weight is not untied.embed.weight
    assert untied.num_parameters() > model.num_parameters()


def test_model_is_causal(small):
    torch.manual_seed(0)
    model = KVScout(small).eval()
    tokens = torch.randint(0, small.vocab_size, (1, 20))
    with torch.no_grad():
        base = model(tokens)
        changed = tokens.clone()
        changed[:, 12:] = (changed[:, 12:] + 101) % small.vocab_size
        after = model(changed)
    assert torch.allclose(base[:, :12], after[:, :12], atol=1e-5)
    assert not torch.allclose(base[:, 12:], after[:, 12:], atol=1e-3)


def test_rejects_sequences_longer_than_the_context(small):
    model = KVScout(small)
    with pytest.raises(ValueError):
        model(torch.zeros(1, small.context_max + 1, dtype=torch.long))


def test_initial_loss_is_near_uniform(small):
    torch.manual_seed(0)
    model = KVScout(small)
    tokens = torch.randint(0, small.vocab_size, (4, 64))
    with torch.no_grad():
        _, cross_entropy = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
    assert float(cross_entropy) == pytest.approx(math.log(small.vocab_size), abs=0.4)


def test_z_loss_adds_to_the_total_only(small):
    torch.manual_seed(0)
    model = KVScout(small)
    tokens = torch.randint(0, small.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(tokens[:, :-1])
        plain, plain_ce = language_model_loss(logits, tokens[:, 1:], 0.0)
        total, ce = language_model_loss(logits, tokens[:, 1:], 1e-3)
    assert float(plain) == pytest.approx(float(plain_ce))
    assert float(ce) == pytest.approx(float(plain_ce))
    assert float(total) > float(ce)


def test_gradients_reach_every_parameter(small):
    model = KVScout(small)
    tokens = torch.randint(0, small.vocab_size, (2, 32))
    total, _ = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
    total.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_residual_projections_are_scaled_down(small):
    torch.manual_seed(0)
    model = KVScout(small)
    for block in model.blocks:
        assert block.attn.o_proj.weight.std() < block.attn.q_proj.weight.std()
        assert block.ffn.down.weight.std() < block.ffn.gate.weight.std()


def test_the_model_can_overfit_a_single_batch(small):
    torch.manual_seed(0)
    model = KVScout(small)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(0, small.vocab_size, (2, 32))
    first = last = None
    for _ in range(30):
        total, ce = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        last = float(ce.detach())
        if first is None:
            first = last
    assert last < first * 0.5


def test_build_model_selects_by_config_type():
    proxy, proxy_loss = build_model(proxy_config(), 0.0)
    assert isinstance(proxy, KVScout)
    assert proxy_loss is language_model_loss
