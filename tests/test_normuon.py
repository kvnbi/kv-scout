from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kv_scout.config import OptimConfig, proxy_config
from kv_scout.model import KVScout
from kv_scout.train.normuon import (
    CombinedOptimizer,
    NorMuon,
    build_optimizer,
    orthogonalize,
    split_parameters,
    update_scale,
)


def singular_values(matrix):
    return torch.linalg.svdvals(matrix)


def test_orthogonalize_pushes_singular_values_towards_one():
    torch.manual_seed(0)
    raw = torch.randn(64, 192) * 5.0
    before = singular_values(raw)
    after = singular_values(orthogonalize(raw, steps=7))
    assert before.max() / before.min() > 3
    assert after.max() / after.min() < 2
    assert 0.5 < after.min() and after.max() < 1.5


def test_square_matrices_need_more_iterations_than_rectangular():
    torch.manual_seed(0)
    square = torch.randn(576, 576)
    wide = torch.randn(576, 1536)

    def conditioning(matrix, steps):
        values = singular_values(orthogonalize(matrix, steps=steps))
        return float(values.max() / values.min())

    assert conditioning(wide, 7) < 2.0
    assert conditioning(square, 7) > 5.0
    assert conditioning(square, 9) < 2.5
    assert conditioning(square, 9) < conditioning(square, 7)


def test_orthogonalize_handles_tall_and_wide(): 
    torch.manual_seed(0)
    for shape in ((128, 64), (64, 128), (576, 1536)):
        out = orthogonalize(torch.randn(*shape), steps=7)
        assert out.shape == shape
        values = singular_values(out)
        assert 0.5 < values.min() and values.max() < 1.5


def test_orthogonalize_is_scale_invariant():
    torch.manual_seed(0)
    raw = torch.randn(32, 48)
    small = orthogonalize(raw, steps=7)
    large = orthogonalize(raw * 1000.0, steps=7)
    assert torch.allclose(small, large, atol=1e-3)


def test_orthogonalize_preserves_dtype_and_rejects_non_matrices():
    out = orthogonalize(torch.randn(8, 8, dtype=torch.bfloat16), steps=3)
    assert out.dtype == torch.bfloat16
    with pytest.raises(ValueError):
        orthogonalize(torch.randn(8))


def test_update_scale_grows_for_tall_matrices():
    assert update_scale(torch.Size([64, 64])) == 1.0
    assert update_scale(torch.Size([64, 256])) == 1.0
    assert update_scale(torch.Size([256, 64])) == pytest.approx(2.0)


def test_optimizer_rejects_non_matrix_parameters():
    with pytest.raises(ValueError):
        NorMuon([torch.nn.Parameter(torch.randn(8))])


def test_optimizer_validates_hyperparameters():
    param = torch.nn.Parameter(torch.randn(4, 4))
    for kwargs in ({"lr": 0.0}, {"momentum": 1.0}, {"neuron_beta": 1.5}, {"newton_schulz_steps": 0}):
        with pytest.raises(ValueError):
            NorMuon([param], **kwargs)


def test_optimizer_minimises_a_quadratic():
    torch.manual_seed(0)
    target = torch.randn(16, 24)
    param = torch.nn.Parameter(torch.zeros(16, 24))
    optimizer = NorMuon([param], lr=0.05)
    first = last = None
    for _ in range(120):
        loss = (param - target).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert last < first * 0.2


def test_momentum_and_neuron_state_are_created():
    param = torch.nn.Parameter(torch.randn(6, 8))
    optimizer = NorMuon([param], lr=0.01)
    param.grad = torch.randn(6, 8)
    optimizer.step()
    state = optimizer.state[param]
    assert state["momentum"].shape == (6, 8)
    assert state["neuron"].shape == (6,)


def test_parameters_without_gradients_are_skipped():
    param = torch.nn.Parameter(torch.randn(4, 4))
    before = param.detach().clone()
    NorMuon([param], lr=0.1).step()
    assert torch.equal(param.detach(), before)


def test_split_sends_embeddings_and_vectors_to_adamw():
    model = KVScout(proxy_config())
    matrices, vectors = split_parameters(model)
    assert all(p.ndim == 2 for p in matrices)
    assert any(p is model.embed.weight for p in vectors)
    assert not any(p is model.embed.weight for p in matrices)
    assert len(matrices) + len(vectors) == len(list(model.parameters()))


def test_build_optimizer_returns_adamw_when_asked():
    model = KVScout(proxy_config())
    optimizer = build_optimizer(model, OptimConfig(matrix_optimizer="adamw"))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(g["lr_scale"] == 1.0 for g in optimizer.param_groups)


def test_build_optimizer_rejects_an_unknown_choice():
    with pytest.raises(ValueError):
        build_optimizer(KVScout(proxy_config()), OptimConfig(matrix_optimizer="lion"))


def test_combined_optimizer_applies_the_learning_rate_multiplier():
    cfg = replace(OptimConfig(), matrix_optimizer="normuon", peak_lr=1e-4, matrix_lr_multiplier=25.0)
    optimizer = build_optimizer(KVScout(proxy_config()), cfg)
    assert isinstance(optimizer, CombinedOptimizer)
    scales = sorted(g["lr_scale"] for g in optimizer.param_groups)
    assert scales == [1.0, 25.0]
    assert optimizer.matrix.param_groups[0]["lr"] == pytest.approx(2.5e-3)


def test_combined_optimizer_state_round_trips():
    torch.manual_seed(0)
    cfg = replace(OptimConfig(), matrix_optimizer="normuon", peak_lr=1e-4)
    model = KVScout(proxy_config(d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1,
                                 head_dim=96, dense_ffn_hidden=384,
                                 attention_anchor_layers=(3,), context_max=64, context_min=64))
    optimizer = build_optimizer(model, cfg)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 16))
    model(tokens).sum().backward()
    optimizer.step()

    payload = optimizer.state_dict()
    assert set(payload) == {"matrix", "vector"}

    fresh = build_optimizer(model, cfg)
    fresh.load_state_dict(payload)
    assert len(fresh.matrix.state) == len(optimizer.matrix.state)


def test_combined_optimizer_zero_grad_clears_both_sides():
    cfg = replace(OptimConfig(), matrix_optimizer="normuon")
    model = KVScout(proxy_config(d_model=192, n_layers=3, n_query_heads=2, n_kv_heads=1,
                                 head_dim=96, dense_ffn_hidden=384,
                                 attention_anchor_layers=(3,), context_max=64, context_min=64))
    optimizer = build_optimizer(model, cfg)
    model(torch.randint(0, model.cfg.vocab_size, (1, 8))).sum().backward()
    assert any(p.grad is not None for p in model.parameters())
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
