from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kv_scout.config import MoEConfig, ModelConfig, OptimConfig, proxy_config
from kv_scout.model import KVScout, MoEFeedForward, SwiGLU
from kv_scout.model.block import TransformerBlock
from kv_scout.train.instrument import collect_router_stats
from kv_scout.train.normuon import build_optimizer


def moe(**overrides):
    base = dict(use_gdn=True, use_moe=True)
    base.update(overrides)
    return proxy_config(**base)


def small(**overrides):
    base = dict(
        use_gdn=True, use_moe=True, d_model=192, n_layers=5, n_query_heads=2,
        n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(5,), context_max=32, context_min=32,
        moe=replace(MoEConfig(), num_experts=6, expert_ffn_hidden=128, shared_ffn_hidden=256),
    )
    base.update(overrides)
    return proxy_config(**base)


def test_only_layers_from_the_third_use_moe():
    cfg = moe()
    for layer in range(1, cfg.n_layers + 1):
        expected = layer >= cfg.moe.first_moe_layer
        assert cfg.uses_moe(layer) is expected
        block = TransformerBlock(cfg, layer)
        assert isinstance(block.ffn, MoEFeedForward) is expected
        assert isinstance(block.ffn, SwiGLU) is not expected


def test_the_first_moe_layer_can_be_moved():
    cfg = moe(moe=replace(MoEConfig(), first_moe_layer=7))
    assert not cfg.uses_moe(6)
    assert cfg.uses_moe(7)
    assert isinstance(TransformerBlock(cfg, 6).ffn, SwiGLU)
    assert isinstance(TransformerBlock(cfg, 7).ffn, MoEFeedForward)


def test_switching_moe_off_leaves_every_layer_dense():
    cfg = moe(use_moe=False)
    for layer in range(1, cfg.n_layers + 1):
        assert cfg.uses_moe(layer) is False
        assert isinstance(TransformerBlock(cfg, layer).ffn, SwiGLU)


def test_layer_range_is_validated():
    cfg = moe()
    for bad in (0, cfg.n_layers + 1):
        with pytest.raises(ValueError):
            cfg.uses_moe(bad)


def test_moe_configuration_is_validated():
    with pytest.raises(ValueError):
        MoEConfig(first_moe_layer=0)
    with pytest.raises(ValueError):
        MoEConfig(shared_experts=-1)


def test_the_shared_expert_always_contributes():
    cfg = small()
    torch.manual_seed(0)
    ffn = MoEFeedForward(cfg).eval()
    x = torch.randn(2, 6, cfg.d_model)
    with torch.no_grad():
        before = ffn(x)
        for expert in ffn.shared:
            expert.down.weight.mul_(0.0)
        without = ffn(x)
    assert not torch.allclose(before, without, atol=1e-5)


def test_the_shared_expert_is_added_to_the_routed_output():
    cfg = small()
    torch.manual_seed(0)
    ffn = MoEFeedForward(cfg).eval()
    x = torch.randn(2, 6, cfg.d_model)
    with torch.no_grad():
        weights, indices, _ = ffn.router(x)
        routed = ffn.experts(x, weights, indices)
        shared = sum(expert(x) for expert in ffn.shared)
        assert torch.allclose(ffn(x), routed + shared, atol=1e-5)


def test_the_shared_expert_can_be_removed():
    cfg = small(moe=replace(MoEConfig(), num_experts=6, expert_ffn_hidden=128,
                            shared_ffn_hidden=256, shared_experts=0))
    ffn = MoEFeedForward(cfg)
    assert len(ffn.shared) == 0
    assert ffn(torch.randn(2, 4, cfg.d_model)).shape == (2, 4, cfg.d_model)
    assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()


def test_parameter_counts_match_the_estimate():
    for cfg in (moe(), moe(use_moe=False), small(), ModelConfig(n_layers=6, attention_anchor_layers=(6,))):
        assert KVScout(cfg).num_parameters() == cfg.parameter_estimate()


def test_moe_raises_total_parameters_but_not_active():
    dense, sparse = moe(use_moe=False), moe()
    assert sparse.parameter_estimate() > dense.parameter_estimate() * 1.5
    assert sparse.active_parameter_estimate() < sparse.parameter_estimate()


def test_the_full_spec_model_still_hits_its_budget():
    cfg = ModelConfig()
    assert 3.25e9 < cfg.parameter_estimate() < 3.35e9
    assert 1.15e9 < cfg.active_parameter_estimate() < 1.30e9


def test_the_model_runs_and_stays_causal():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        base = model(tokens)
        changed = tokens.clone()
        changed[:, 12:] = (changed[:, 12:] + 7) % cfg.vocab_size
        after = model(changed)
    assert base.shape == (1, 20, cfg.vocab_size)
    assert torch.allclose(base[:, :12], after[:, :12], atol=1e-4)


def test_the_moe_model_trains():
    from kv_scout.model import language_model_loss

    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    first = last = None
    for _ in range(30):
        total, ce = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        last = float(ce.detach())
        if first is None:
            first = last
    assert last < first * 0.6


def test_every_expert_receives_gradient_over_a_batch():
    from kv_scout.model import language_model_loss

    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (16, 32))
    total, _ = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
    total.backward()
    for block in model.blocks:
        if not block.uses_moe:
            continue
        for expert in block.ffn.experts.experts:
            assert float(expert.down.weight.grad.abs().sum()) > 0
        for expert in block.ffn.shared:
            assert float(expert.down.weight.grad.abs().sum()) > 0


def test_the_instrument_finds_every_router_in_the_model():
    cfg = small()
    model = KVScout(cfg).eval()
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (4, 16)))
    stats = collect_router_stats(model)
    expected = sum(1 for i in range(1, cfg.n_layers + 1) if cfg.uses_moe(i))
    assert len(stats) == expected
    for entry in stats:
        assert entry["experts"] == cfg.moe.num_experts
        assert "ffn.router" in entry["module"]


def test_expert_weights_carry_the_mup_scale():
    cfg = small(use_mup=True, d_model=384, n_query_heads=4, n_kv_heads=2, head_dim=96)
    model = KVScout(cfg)
    block = next(b for b in model.blocks if b.uses_moe)
    expected = 1.0 / cfg.width_multiplier
    for weight in block.ffn.hidden_weights():
        assert getattr(weight, "mup_lr_scale") == pytest.approx(expected)
    assert getattr(model.embed.weight, "mup_lr_scale") == 1.0


def test_the_optimizer_groups_expert_weights_by_scale():
    cfg = small(use_mup=True, d_model=384, n_query_heads=4, n_kv_heads=2, head_dim=96)
    model = KVScout(cfg)
    optimizer = build_optimizer(
        model, replace(OptimConfig(), matrix_optimizer="adamw", peak_lr=1e-3)
    )
    scales = sorted({round(g["lr_scale"], 4) for g in optimizer.param_groups})
    assert round(1.0 / cfg.width_multiplier, 4) in scales
    assert 1.0 in scales


def test_residual_projections_are_scaled_for_experts_too():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    block = next(b for b in model.blocks if b.uses_moe)
    for expert in block.ffn.experts.experts:
        with torch.no_grad():
            assert float(expert.down.weight.std()) < float(expert.gate.weight.std())
    for expert in block.ffn.shared:
        with torch.no_grad():
            assert float(expert.down.weight.std()) < float(expert.gate.weight.std())


def full_stack(**overrides):
    base = dict(
        use_gdn=True, use_moe=True, attention_sinks=True, sink_tokens=2,
        attention_window=16, cross_layer_kv_sharing=True, d_model=192, n_layers=12,
        n_query_heads=2, n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(6, 9, 12), context_max=128, context_min=128,
        moe=replace(MoEConfig(), num_experts=6, expert_ffn_hidden=192, shared_ffn_hidden=384),
    )
    base.update(overrides)
    return proxy_config(**base)


def test_cached_generation_matches_uncached_with_the_whole_stack():
    from dataclasses import replace as dataclass_replace
    from pathlib import Path

    from kv_scout.generate import SamplingConfig, generate

    tokenizer_path = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        pytest.skip("tokenizer missing")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cfg = full_stack()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()

    settings = SamplingConfig(max_new_tokens=48, greedy=True)
    cached = generate(model, tokenizer, "The history of", dataclass_replace(settings, use_cache=True))
    plain = generate(model, tokenizer, "The history of", dataclass_replace(settings, use_cache=False))
    assert cached["tokens"] == plain["tokens"]


def test_the_router_does_not_move_during_evaluation():
    cfg = full_stack()
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    router = next(b.ffn.router for b in model.blocks if b.uses_moe)
    before = router.balance_bias.clone()
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (2, 32)))
    assert torch.equal(before, router.balance_bias)
    assert int(router.expert_counts.sum()) > 0


def test_the_router_moves_during_training():
    cfg = full_stack()
    torch.manual_seed(0)
    model = KVScout(cfg).train()
    router = next(b.ffn.router for b in model.blocks if b.uses_moe)
    for _ in range(5):
        model(torch.randint(0, cfg.vocab_size, (2, 32)))
    assert float(router.balance_bias.abs().max()) > 0


def test_moe_works_with_normuon():
    from kv_scout.model import language_model_loss

    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg)
    optimizer = build_optimizer(
        model, replace(OptimConfig(), matrix_optimizer="normuon", peak_lr=3e-4)
    )
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    first = last = None
    for _ in range(30):
        total, ce = language_model_loss(model(tokens[:, :-1]), tokens[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        last = float(ce.detach())
        if first is None:
            first = last
    assert last < first
    assert torch.isfinite(torch.tensor(last))


def test_router_buffers_ride_along_in_a_checkpoint():
    cfg = small()
    torch.manual_seed(0)
    model = KVScout(cfg).train()
    for _ in range(5):
        model(torch.randint(0, cfg.vocab_size, (2, 16)))

    restored = KVScout(cfg)
    restored.load_state_dict(model.state_dict())
    for a, b in zip(model.blocks, restored.blocks):
        if not a.uses_moe:
            continue
        assert torch.equal(a.ffn.router.balance_bias, b.ffn.router.balance_bias)
        assert torch.equal(a.ffn.router.expert_counts, b.ffn.router.expert_counts)
        assert torch.equal(a.ffn.router.recent_load, b.ffn.router.recent_load)
