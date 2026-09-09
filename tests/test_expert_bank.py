from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kv_scout.config import MoEConfig, proxy_config
from kv_scout.model import ExpertBank, Router


@pytest.fixture
def cfg():
    return proxy_config(use_moe=True)


def dense_reference(bank, x, weights, indices, d_model):
    every = bank.dense(x)
    out = torch.zeros_like(x)
    for slot in range(indices.shape[-1]):
        picked = torch.gather(
            every, -2, indices[..., slot, None, None].expand(-1, -1, 1, d_model)
        ).squeeze(-2)
        out = out + weights[..., slot, None] * picked
    return out


def test_the_bank_holds_one_expert_per_slot(cfg):
    bank = ExpertBank(cfg)
    assert len(bank.experts) == cfg.moe.num_experts
    assert bank.num_experts == cfg.moe.num_experts


def test_parameter_count(cfg):
    bank = ExpertBank(cfg)
    total = sum(p.numel() for p in bank.parameters())
    assert total == cfg.moe.num_experts * 3 * cfg.d_model * cfg.moe.expert_ffn_hidden


def test_experts_are_independent(cfg):
    bank = ExpertBank(cfg)
    x = torch.randn(1, 4, cfg.d_model)
    outputs = bank.dense(x)
    assert outputs.shape == (1, 4, cfg.moe.num_experts, cfg.d_model)
    first, second = outputs[..., 0, :], outputs[..., 1, :]
    assert not torch.allclose(first, second, atol=1e-4)


def test_dispatch_matches_dense_computation(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    router = Router(cfg).eval()
    x = torch.randn(3, 17, cfg.d_model)
    weights, indices, _ = router(x)
    sparse = bank(x, weights, indices)
    assert torch.allclose(
        sparse, dense_reference(bank, x, weights, indices, cfg.d_model), atol=1e-5
    )


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_dispatch_matches_dense_at_several_k(top_k):
    cfg = proxy_config(use_moe=True, moe=replace(MoEConfig(), top_k=top_k))
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    router = Router(cfg).eval()
    x = torch.randn(2, 11, cfg.d_model)
    weights, indices, _ = router(x)
    sparse = bank(x, weights, indices)
    assert torch.allclose(
        sparse, dense_reference(bank, x, weights, indices, cfg.d_model), atol=1e-5
    )


def test_only_the_chosen_experts_affect_a_token(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    x = torch.randn(1, 1, cfg.d_model)
    indices = torch.tensor([[[0, 1]]])
    weights = torch.tensor([[[0.6, 0.4]]])
    before = bank(x, weights, indices)

    with torch.no_grad():
        bank.experts[5].down.weight.mul_(100.0)
    after = bank(x, weights, indices)
    assert torch.allclose(before, after, atol=1e-6)

    with torch.no_grad():
        bank.experts[1].down.weight.mul_(100.0)
    assert not torch.allclose(before, bank(x, weights, indices), atol=1e-4)


def test_weights_scale_the_contribution(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    x = torch.randn(1, 3, cfg.d_model)
    indices = torch.zeros(1, 3, 1, dtype=torch.long)
    full = bank(x, torch.ones(1, 3, 1), indices)
    half = bank(x, torch.full((1, 3, 1), 0.5), indices)
    assert torch.allclose(half, full * 0.5, atol=1e-5)


def test_an_unused_expert_receives_no_gradient(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    indices = torch.zeros(2, 5, 1, dtype=torch.long)
    bank(x, torch.ones(2, 5, 1), indices).sum().backward()
    assert bank.experts[0].down.weight.grad.abs().sum() > 0
    for expert in range(1, cfg.moe.num_experts):
        grad = bank.experts[expert].down.weight.grad
        assert grad is None or float(grad.abs().sum()) == 0.0


def test_every_used_expert_receives_gradient(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    router = Router(cfg).eval()
    x = torch.randn(64, 32, cfg.d_model)
    weights, indices, _ = router(x)
    bank(x, weights, indices).sum().backward()
    used = set(indices.reshape(-1).tolist())
    assert len(used) == cfg.moe.num_experts
    for expert in range(cfg.moe.num_experts):
        assert float(bank.experts[expert].down.weight.grad.abs().sum()) > 0


def test_all_tokens_to_one_expert(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    indices = torch.full((2, 8, 1), 7, dtype=torch.long)
    out = bank(x, torch.ones(2, 8, 1), indices)
    assert torch.allclose(out, bank.experts[7](x), atol=1e-5)


def test_shape_is_preserved(cfg):
    bank = ExpertBank(cfg)
    for shape in ((1, 1), (4, 9), (7, 23)):
        x = torch.randn(*shape, cfg.d_model)
        weights = torch.rand(*shape, cfg.moe.top_k)
        indices = torch.randint(0, cfg.moe.num_experts, (*shape, cfg.moe.top_k))
        assert bank(x, weights, indices).shape == x.shape


def test_dispatch_is_deterministic(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg).eval()
    router = Router(cfg).eval()
    x = torch.randn(3, 12, cfg.d_model)
    weights, indices, _ = router(x)
    with torch.no_grad():
        first = bank(x, weights, indices)
        second = bank(x, weights, indices)
    assert torch.equal(first, second)


def test_dispatch_preserves_dtype(cfg):
    bank = ExpertBank(cfg).to(torch.bfloat16)
    x = torch.randn(2, 6, cfg.d_model, dtype=torch.bfloat16)
    weights = torch.rand(2, 6, cfg.moe.top_k, dtype=torch.bfloat16)
    indices = torch.randint(0, cfg.moe.num_experts, (2, 6, cfg.moe.top_k))
    assert bank(x, weights, indices).dtype == torch.bfloat16


def test_dispatch_runs_on_the_accelerator(cfg):
    if not torch.backends.mps.is_available():
        pytest.skip("mps not available")
    torch.manual_seed(0)
    bank = ExpertBank(cfg).to("mps")
    router = Router(cfg).to("mps").eval()
    x = torch.randn(2, 16, cfg.d_model, device="mps")
    weights, indices, _ = router(x)
    sparse = bank(x, weights, indices)
    assert sparse.device.type == "mps"
    assert torch.allclose(
        sparse, dense_reference(bank, x, weights, indices, cfg.d_model), atol=1e-4
    )


def test_dispatch_touches_only_the_tokens_an_expert_owns(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg)
    x = torch.randn(1, 6, cfg.d_model)
    indices = torch.tensor([[[0], [1], [0], [1], [0], [1]]])
    weights = torch.ones(1, 6, 1)
    out = bank(x, weights, indices)
    assert torch.allclose(out[0, 0], bank.experts[0](x[0, 0]), atol=1e-5)
    assert torch.allclose(out[0, 1], bank.experts[1](x[0, 1]), atol=1e-5)
    assert torch.allclose(out[0, 4], bank.experts[0](x[0, 4]), atol=1e-5)


def test_gradients_match_the_dense_path(cfg):
    torch.manual_seed(0)
    sparse_bank = ExpertBank(cfg)
    dense_bank = ExpertBank(cfg)
    dense_bank.load_state_dict(sparse_bank.state_dict())

    router = Router(cfg).eval()
    x = torch.randn(4, 24, cfg.d_model)
    with torch.no_grad():
        weights, indices, _ = router(x)
    target = torch.randn(4, 24, cfg.d_model)

    ((sparse_bank(x, weights, indices) - target) ** 2).mean().backward()
    reference = dense_reference(dense_bank, x, weights, indices, cfg.d_model)
    ((reference - target) ** 2).mean().backward()

    for a, b in zip(sparse_bank.parameters(), dense_bank.parameters()):
        assert a.grad is not None and b.grad is not None
        assert torch.allclose(a.grad, b.grad, atol=1e-8)


def test_dispatch_matches_dense_in_bfloat16(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg).to(torch.bfloat16).eval()
    router = Router(cfg).to(torch.bfloat16).eval()
    x = torch.randn(3, 16, cfg.d_model, dtype=torch.bfloat16)
    with torch.no_grad():
        weights, indices, _ = router(x)
        sparse = bank(x, weights, indices)
        reference = dense_reference(bank, x, weights, indices, cfg.d_model)
    assert torch.allclose(sparse, reference, atol=1e-3)


def test_dispatch_matches_dense_under_autocast(cfg):
    torch.manual_seed(0)
    bank = ExpertBank(cfg).eval()
    router = Router(cfg).eval()
    x = torch.randn(3, 16, cfg.d_model)
    with torch.no_grad():
        weights, indices, _ = router(x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            sparse = bank(x, weights, indices)
            reference = dense_reference(bank, x, weights, indices, cfg.d_model)
    assert torch.allclose(sparse, reference, atol=1e-3)


def test_an_empty_batch_is_handled(cfg):
    bank = ExpertBank(cfg).eval()
    with torch.no_grad():
        out = bank(
            torch.randn(0, 4, cfg.d_model),
            torch.rand(0, 4, cfg.moe.top_k),
            torch.zeros(0, 4, cfg.moe.top_k, dtype=torch.long),
        )
    assert out.shape == (0, 4, cfg.d_model)


def test_dispatch_is_deterministic_on_the_accelerator(cfg):
    if not torch.backends.mps.is_available():
        pytest.skip("mps not available")
    torch.manual_seed(0)
    bank = ExpertBank(cfg).to("mps").eval()
    router = Router(cfg).to("mps").eval()
    x = torch.randn(4, 32, cfg.d_model, device="mps")
    with torch.no_grad():
        weights, indices, _ = router(x)
        runs = [bank(x, weights, indices) for _ in range(5)]
    for run in runs[1:]:
        assert torch.equal(runs[0], run)
