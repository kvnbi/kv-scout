from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kv_scout.config import MoEConfig, ModelConfig, proxy_config
from kv_scout.model import Router


@pytest.fixture
def cfg():
    return proxy_config(use_moe=True)


def test_shapes(cfg):
    router = Router(cfg)
    weights, indices, scores = router(torch.randn(3, 11, cfg.d_model))
    assert weights.shape == (3, 11, cfg.moe.top_k)
    assert indices.shape == (3, 11, cfg.moe.top_k)
    assert scores.shape == (3, 11, cfg.moe.num_experts)


def test_the_gate_costs_one_row_per_expert(cfg):
    router = Router(cfg)
    total = sum(p.numel() for p in router.parameters())
    assert total == cfg.d_model * cfg.moe.num_experts


def test_indices_are_inside_the_expert_range(cfg):
    _, indices, _ = Router(cfg)(torch.randn(4, 9, cfg.d_model))
    assert int(indices.min()) >= 0
    assert int(indices.max()) < cfg.moe.num_experts


def test_no_token_is_routed_to_the_same_expert_twice(cfg):
    _, indices, _ = Router(cfg)(torch.randn(4, 16, cfg.d_model))
    flat = indices.reshape(-1, cfg.moe.top_k)
    for row in flat:
        assert len(set(row.tolist())) == cfg.moe.top_k


def test_top_k_selects_the_highest_scoring_experts(cfg):
    torch.manual_seed(0)
    router = Router(cfg)
    _, indices, scores = router(torch.randn(2, 8, cfg.d_model))
    expected = torch.topk(scores, cfg.moe.top_k, dim=-1).indices
    assert torch.equal(
        torch.sort(indices, dim=-1).values, torch.sort(expected, dim=-1).values
    )


def test_weights_sum_to_one(cfg):
    weights, _, _ = Router(cfg)(torch.randn(5, 13, cfg.d_model))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5, 13), atol=1e-6)


def test_weights_are_the_normalised_scores_of_the_chosen_experts(cfg):
    torch.manual_seed(0)
    weights, indices, scores = Router(cfg)(torch.randn(2, 6, cfg.d_model))
    chosen = torch.gather(scores, -1, indices)
    expected = chosen / chosen.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights, expected, atol=1e-6)


def test_weights_are_ordered_by_score(cfg):
    weights, _, _ = Router(cfg)(torch.randn(3, 10, cfg.d_model))
    assert (weights[..., 0] >= weights[..., 1]).all()


def test_sigmoid_scores_are_independent_per_expert(cfg):
    scores = Router(cfg).affinities(torch.randn(2, 6, cfg.d_model))
    assert cfg.moe.routing == "sigmoid"
    assert ((scores >= 0) & (scores <= 1)).all()
    assert not torch.allclose(scores.sum(dim=-1), torch.ones(2, 6), atol=1e-3)


def test_softmax_routing_normalises_across_experts():
    cfg = proxy_config(use_moe=True, moe=replace(MoEConfig(), routing="softmax"))
    scores = Router(cfg).affinities(torch.randn(2, 6, cfg.d_model))
    assert torch.allclose(scores.sum(dim=-1), torch.ones(2, 6), atol=1e-5)


def test_routing_is_deterministic(cfg):
    torch.manual_seed(0)
    router = Router(cfg).eval()
    x = torch.randn(2, 9, cfg.d_model)
    with torch.no_grad():
        first = router(x)
        second = router(x)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_routing_is_per_token(cfg):
    torch.manual_seed(0)
    router = Router(cfg).eval()
    x = torch.randn(1, 8, cfg.d_model)
    with torch.no_grad():
        _, indices, _ = router(x)
        changed = x.clone()
        changed[:, 3] = torch.randn(cfg.d_model) * 5
        _, after, _ = router(changed)
    assert torch.equal(indices[:, :3], after[:, :3])
    assert torch.equal(indices[:, 4:], after[:, 4:])


def test_every_expert_is_reachable(cfg):
    torch.manual_seed(0)
    _, indices, _ = Router(cfg)(torch.randn(64, 64, cfg.d_model))
    assert set(indices.reshape(-1).tolist()) == set(range(cfg.moe.num_experts))


def test_normalised_weights_carry_no_gradient_by_themselves(cfg):
    router = Router(cfg)
    weights, _, _ = router(torch.randn(2, 6, cfg.d_model))
    assert float(weights.detach().sum()) == pytest.approx(12.0, abs=1e-4)


def test_gradients_reach_the_gate_through_the_scores(cfg):
    torch.manual_seed(0)
    router = Router(cfg)
    _, _, scores = router(torch.randn(2, 6, cfg.d_model))
    (scores * torch.randn_like(scores)).sum().backward()
    grad = router.gate.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 1e-2


def test_gradients_reach_the_gate_through_a_weight_dependent_loss(cfg):
    torch.manual_seed(0)
    router = Router(cfg)
    weights, _, _ = router(torch.randn(2, 6, cfg.d_model))
    (weights * torch.randn_like(weights)).sum().backward()
    grad = router.gate.weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 1e-3


def test_scoring_stays_in_float32_under_autocast(cfg):
    router = Router(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, _, scores = router(x)
    assert scores.dtype == torch.float32


def test_a_bfloat16_router_still_scores_in_float32(cfg):
    router = Router(cfg).to(torch.bfloat16)
    weights, _, scores = router(torch.randn(2, 6, cfg.d_model, dtype=torch.bfloat16))
    assert scores.dtype == torch.float32
    assert weights.dtype == torch.bfloat16
    assert torch.allclose(
        weights.float().sum(dim=-1), torch.ones(2, 6), atol=1e-2
    )


def test_scoring_runs_in_float32_and_returns_the_input_dtype(cfg):
    router = Router(cfg)
    x = torch.randn(2, 5, cfg.d_model, dtype=torch.bfloat16)
    weights, _, scores = router(x)
    assert scores.dtype == torch.float32
    assert weights.dtype == torch.bfloat16


def test_top_k_of_one_still_normalises():
    cfg = proxy_config(use_moe=True, moe=replace(MoEConfig(), top_k=1))
    weights, indices, _ = Router(cfg)(torch.randn(2, 7, cfg.d_model))
    assert weights.shape == (2, 7, 1)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)
    assert indices.shape == (2, 7, 1)


def test_moe_configuration_is_validated():
    for bad in (
        dict(top_k=0),
        dict(top_k=12),
        dict(routing="linear"),
        dict(expert_ffn_hidden=0),
        dict(shared_ffn_hidden=0),
    ):
        with pytest.raises(ValueError):
            MoEConfig(**bad)


def test_the_spec_routing_defaults_are_unchanged():
    moe = ModelConfig().moe
    assert moe.num_experts == 12
    assert moe.top_k == 2
    assert moe.shared_experts == 1
    assert moe.routing == "sigmoid"
    assert moe.aux_loss_free_balancing is True
