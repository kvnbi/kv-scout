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


def skewed(cfg, factor=25.0, experts=2):
    torch.manual_seed(0)
    router = Router(cfg)
    with torch.no_grad():
        router.gate.weight[:experts] *= factor
    return router


def test_balancing_state_starts_neutral(cfg):
    router = Router(cfg)
    assert torch.equal(router.balance_bias, torch.zeros(cfg.moe.num_experts))
    assert int(router.expert_counts.sum()) == 0


def test_the_bias_is_not_a_learned_parameter(cfg):
    router = Router(cfg)
    names = {name for name, _ in router.named_parameters()}
    assert "balance_bias" not in names
    assert "expert_counts" not in names
    assert "balance_bias" in dict(router.named_buffers())


def test_counts_match_the_routing_that_happened(cfg):
    router = Router(cfg).eval()
    _, indices, _ = router(torch.randn(4, 20, cfg.d_model))
    expected = torch.bincount(indices.reshape(-1), minlength=cfg.moe.num_experts)
    assert torch.equal(router.expert_counts, expected)
    assert int(router.expert_counts.sum()) == 4 * 20 * cfg.moe.top_k


def test_counts_accumulate_and_can_be_reset(cfg):
    router = Router(cfg).eval()
    router(torch.randn(2, 10, cfg.d_model))
    first = int(router.expert_counts.sum())
    router(torch.randn(2, 10, cfg.d_model))
    assert int(router.expert_counts.sum()) == first * 2
    router.reset_counts()
    assert int(router.expert_counts.sum()) == 0


def test_evaluation_mode_never_moves_the_bias(cfg):
    router = skewed(cfg).eval()
    for _ in range(20):
        router(torch.randn(4, 32, cfg.d_model))
    assert torch.equal(router.balance_bias, torch.zeros(cfg.moe.num_experts))


def test_training_moves_the_bias(cfg):
    router = skewed(cfg).train()
    for _ in range(20):
        router(torch.randn(4, 32, cfg.d_model))
    assert float(router.balance_bias.abs().max()) > 0


def test_the_bias_stays_centred(cfg):
    router = skewed(cfg).train()
    for _ in range(30):
        router(torch.randn(4, 32, cfg.d_model))
    assert float(router.balance_bias.mean().abs()) < 1e-5


def test_balancing_recovers_a_badly_skewed_router(cfg):
    router = skewed(cfg, factor=25.0, experts=2).train()
    router(torch.randn(8, 64, cfg.d_model))
    before = router.lifetime_balance()
    router.reset_counts()

    for _ in range(400):
        router(torch.randn(8, 64, cfg.d_model))
    router.reset_counts()
    for _ in range(20):
        router(torch.randn(8, 64, cfg.d_model))
    after = router.lifetime_balance()

    assert before["max_over_mean"] > 2.0
    assert after["max_over_mean"] < 1.3
    assert after["normalised_entropy"] > before["normalised_entropy"]
    assert after["starved_experts"] == 0


def test_without_balancing_the_skew_persists(cfg):
    plain = proxy_config(
        use_moe=True, moe=replace(MoEConfig(), aux_loss_free_balancing=False)
    )
    router = skewed(plain, factor=25.0, experts=2).train()
    for _ in range(400):
        router(torch.randn(8, 64, plain.d_model))
    router.reset_counts()
    for _ in range(20):
        router(torch.randn(8, 64, plain.d_model))
    assert router.lifetime_balance()["max_over_mean"] > 2.0
    assert torch.equal(router.balance_bias, torch.zeros(plain.moe.num_experts))


def test_the_bias_changes_selection_but_not_the_weights(cfg):
    torch.manual_seed(0)
    router = Router(cfg).eval()
    x = torch.randn(2, 12, cfg.d_model)
    with torch.no_grad():
        _, before, scores = router(x)
        router.balance_bias[5] = 10.0
        weights, after, again = router(x)
    assert not torch.equal(before, after)
    assert torch.allclose(scores, again, atol=1e-6)
    chosen = torch.gather(again, -1, after)
    assert torch.allclose(
        weights.float(), chosen / chosen.sum(dim=-1, keepdim=True), atol=1e-6
    )


def starve(cfg, expert=7):
    torch.manual_seed(0)
    router = Router(cfg)
    with torch.no_grad():
        router.gate.weight[expert] *= 0.0
        router.gate.weight *= 10.0
    return router


def test_the_starved_setup_really_starves(cfg):
    router = starve(cfg).eval()
    router(torch.randn(8, 64, cfg.d_model))
    share = float(router.expert_counts[7]) / float(router.expert_counts.sum())
    assert share < 0.02
    assert router.lifetime_balance()["starved_experts"] >= 1


def test_a_starved_expert_is_pushed_back_into_use(cfg):
    router = starve(cfg).train()
    for _ in range(600):
        router(torch.randn(8, 64, cfg.d_model))
    router.reset_counts()
    for _ in range(20):
        router(torch.randn(8, 64, cfg.d_model))

    share = float(router.expert_counts[7]) / float(router.expert_counts.sum())
    target = 1.0 / cfg.moe.num_experts
    assert share == pytest.approx(target, abs=0.03)
    assert float(router.balance_bias[7]) > 0
    assert router.lifetime_balance()["starved_experts"] == 0


def test_a_starved_expert_stays_starved_without_balancing(cfg):
    plain = proxy_config(
        use_moe=True, moe=replace(MoEConfig(), aux_loss_free_balancing=False)
    )
    router = starve(plain).train()
    for _ in range(600):
        router(torch.randn(8, 64, plain.d_model))
    router.reset_counts()
    for _ in range(20):
        router(torch.randn(8, 64, plain.d_model))
    share = float(router.expert_counts[7]) / float(router.expert_counts.sum())
    assert share < 0.02


def test_a_dominant_expert_is_brought_back_to_its_share(cfg):
    torch.manual_seed(0)
    router = Router(cfg).train()
    with torch.no_grad():
        router.gate.weight[7] *= 0.0
        router.gate.weight[7] -= 5.0

    router(torch.randn(8, 64, cfg.d_model))
    before = float(router.expert_counts[7]) / float(router.expert_counts.sum())
    router.reset_counts()

    for _ in range(400):
        router(torch.randn(8, 64, cfg.d_model))
    router.reset_counts()
    for _ in range(20):
        router(torch.randn(8, 64, cfg.d_model))
    after = float(router.expert_counts[7]) / float(router.expert_counts.sum())

    assert before > 0.2
    assert after == pytest.approx(1.0 / cfg.moe.num_experts, abs=0.03)


def test_balancing_state_survives_a_state_dict_round_trip(cfg):
    router = skewed(cfg).train()
    for _ in range(15):
        router(torch.randn(4, 32, cfg.d_model))
    payload = router.state_dict()
    assert "balance_bias" in payload
    assert "expert_counts" in payload

    restored = Router(cfg)
    restored.load_state_dict(payload)
    assert torch.equal(restored.balance_bias, router.balance_bias)
    assert torch.equal(restored.expert_counts, router.expert_counts)


def test_the_instrument_finds_the_router(cfg):
    from kv_scout.train.instrument import collect_router_stats, router_stats_are_healthy

    model = torch.nn.Module()
    router = Router(cfg).eval()
    router(torch.randn(8, 64, cfg.d_model))
    model.add_module("layer3", router)
    stats = collect_router_stats(model)
    assert len(stats) == 1
    assert stats[0]["module"] == "layer3"
    assert stats[0]["experts"] == cfg.moe.num_experts
    assert router_stats_are_healthy(stats)


def test_the_instrument_reports_an_unbalanced_router(cfg):
    from kv_scout.train.instrument import collect_router_stats, router_stats_are_healthy

    model = torch.nn.Module()
    torch.manual_seed(0)
    router = Router(cfg).eval()
    with torch.no_grad():
        router.gate.weight[1:] -= 50.0
    for _ in range(600):
        router(torch.randn(8, 64, cfg.d_model))
    model.add_module("layer3", router)
    stats = collect_router_stats(model)
    assert stats[0]["max_over_mean"] > 3.0
    assert not router_stats_are_healthy(stats)


def test_the_update_rate_is_validated():
    with pytest.raises(ValueError):
        MoEConfig(balance_update_rate=0.0)
    with pytest.raises(ValueError):
        MoEConfig(balance_update_rate=-1.0)


def test_a_faster_rate_corrects_sooner(cfg):
    results = {}
    for rate in (1e-4, 1e-2):
        scaled = proxy_config(
            use_moe=True, moe=replace(MoEConfig(), balance_update_rate=rate)
        )
        router = skewed(scaled, factor=25.0, experts=2).train()
        for _ in range(60):
            router(torch.randn(8, 64, scaled.d_model))
        router.reset_counts()
        for _ in range(10):
            router(torch.randn(8, 64, scaled.d_model))
        results[rate] = router.balance()["max_over_mean"]
    assert results[1e-2] < results[1e-4]


def test_recent_load_starts_uniform(cfg):
    router = Router(cfg)
    expected = torch.full((cfg.moe.num_experts,), 1.0 / cfg.moe.num_experts)
    assert torch.allclose(router.recent_load, expected)
    assert router.balance()["max_over_mean"] == pytest.approx(1.0)


def test_recent_load_tracks_the_current_routing(cfg):
    router = Router(cfg).eval()
    for _ in range(200):
        router(torch.randn(4, 32, cfg.d_model))
    assert router.balance()["max_over_mean"] < 1.3
    assert router.balance()["starved_experts"] == 0


def test_recent_load_reports_a_late_collapse(cfg):
    torch.manual_seed(0)
    router = Router(cfg).eval()
    for _ in range(200):
        router(torch.randn(4, 32, cfg.d_model))
    healthy = router.balance()

    with torch.no_grad():
        router.gate.weight[1:] -= 50.0
    for _ in range(600):
        router(torch.randn(4, 32, cfg.d_model))
    collapsed = router.balance()

    assert healthy["max_over_mean"] < 1.3
    assert collapsed["max_over_mean"] > 4.0
    assert collapsed["starved_experts"] > 5


def test_lifetime_counts_hide_a_late_collapse(cfg):
    torch.manual_seed(0)
    router = Router(cfg).eval()
    for _ in range(200):
        router(torch.randn(4, 32, cfg.d_model))
    with torch.no_grad():
        router.gate.weight[1:] -= 50.0
    for _ in range(600):
        router(torch.randn(4, 32, cfg.d_model))

    assert router.lifetime_balance()["starved_experts"] == 0
    assert router.balance()["starved_experts"] > 5
    assert router.balance()["max_over_mean"] > router.lifetime_balance()["max_over_mean"]


def test_the_instrument_reads_the_recent_load(cfg):
    from kv_scout.train.instrument import collect_router_stats, router_stats_are_healthy

    torch.manual_seed(0)
    router = Router(cfg).eval()
    with torch.no_grad():
        router.gate.weight[1:] -= 50.0
    for _ in range(600):
        router(torch.randn(4, 32, cfg.d_model))

    model = torch.nn.Module()
    model.add_module("layer5", router)
    stats = collect_router_stats(model)
    assert len(stats) == 1
    assert not router_stats_are_healthy(stats)


def test_resetting_returns_the_monitor_to_neutral(cfg):
    router = Router(cfg).eval()
    with torch.no_grad():
        router.gate.weight[1:] -= 50.0
    for _ in range(100):
        router(torch.randn(4, 32, cfg.d_model))
    router.reset_counts()
    assert int(router.expert_counts.sum()) == 0
    assert router.balance()["max_over_mean"] == pytest.approx(1.0)


def test_recent_load_survives_a_state_dict_round_trip(cfg):
    router = Router(cfg).eval()
    for _ in range(30):
        router(torch.randn(4, 32, cfg.d_model))
    restored = Router(cfg)
    restored.load_state_dict(router.state_dict())
    assert torch.equal(restored.recent_load, router.recent_load)


def test_load_decay_is_validated():
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError):
            MoEConfig(load_decay=bad)


def test_the_bias_accumulates_in_bfloat16(cfg):
    counts = torch.tensor([100, 0] + [50] * 10)
    moved = {}
    for dtype in (torch.float32, torch.bfloat16):
        router = Router(cfg).to(dtype).train()
        with torch.no_grad():
            for _ in range(200):
                router.rebalance(counts)
        moved[dtype] = float(router.balance_bias[1])
    assert moved[torch.float32] > 0.15
    assert moved[torch.bfloat16] > 0.15
    assert abs(moved[torch.bfloat16] - moved[torch.float32]) < 0.05


def test_routing_works_on_the_accelerator(cfg):
    if not torch.backends.mps.is_available():
        pytest.skip("mps not available")
    router = Router(cfg).to("mps").eval()
    weights, indices, _ = router(torch.randn(2, 16, cfg.d_model, device="mps"))
    assert int(router.expert_counts.sum()) == 2 * 16 * cfg.moe.top_k
    assert weights.device.type == "mps"
    assert indices.device.type == "mps"
