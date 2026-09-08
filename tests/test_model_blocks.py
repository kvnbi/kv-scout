from __future__ import annotations

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import (
    GroupedQueryAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    apply_rope,
    repeat_kv,
    rope_frequencies,
)


@pytest.fixture
def cfg():
    return proxy_config()


def test_rmsnorm_matches_the_reference_formula():
    torch.manual_seed(0)
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    x = torch.randn(3, 5, 8)
    expected = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_preserves_dtype_and_applies_scale():
    norm = RMSNorm(8, scale=0.5)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(
        RMSNorm(8, scale=0.5)(x.float()), RMSNorm(8)(x.float()) * 0.5, atol=1e-5
    )


def test_swiglu_shape_and_parameter_count():
    mlp = SwiGLU(576, 1536)
    assert mlp(torch.randn(2, 7, 576)).shape == (2, 7, 576)
    assert sum(p.numel() for p in mlp.parameters()) == 3 * 576 * 1536


def test_rope_leaves_position_zero_unchanged():
    cos, sin = rope_frequencies(96, 8, 10000.0)
    x = torch.randn(1, 2, 8, 96)
    assert torch.allclose(apply_rope(x, cos, sin)[..., 0, :], x[..., 0, :], atol=1e-6)


def test_rope_preserves_vector_norm():
    cos, sin = rope_frequencies(96, 16, 10000.0)
    x = torch.randn(2, 3, 16, 96)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_dot_product_depends_only_on_relative_position():
    torch.manual_seed(0)
    dim, length = 96, 32
    cos, sin = rope_frequencies(dim, length, 10000.0)

    q = torch.randn(dim)
    k = torch.randn(dim)
    q_at = apply_rope(q.expand(1, 1, length, dim).contiguous(), cos, sin)[0, 0]
    k_at = apply_rope(k.expand(1, 1, length, dim).contiguous(), cos, sin)[0, 0]

    def score(i, j):
        return float(torch.dot(q_at[i], k_at[j]))

    for offset in (0, 1, 5, 11):
        values = [score(i + offset, i) for i in range(0, length - offset, 3)]
        assert max(values) == pytest.approx(min(values), abs=1e-4), (
            f"offset {offset} gave scores spanning {min(values)} to {max(values)}"
        )

    assert score(5, 0) != pytest.approx(score(1, 0), abs=1e-3)


def test_repeat_kv_expands_each_head_in_place():
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 3, 4)
    for head in range(2):
        for copy in range(3):
            assert torch.equal(out[:, head * 3 + copy], x[:, head])
    assert torch.equal(repeat_kv(x, 1), x)


def test_attention_shape_and_parameter_count(cfg):
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 12, cfg.d_model)
    assert attn(x)[0].shape == x.shape
    kv = cfg.n_kv_heads * cfg.head_dim
    expected = 2 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * kv
    assert sum(p.numel() for p in attn.parameters()) == expected


def test_attention_is_causal(cfg):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 10, cfg.d_model)
    cos, sin = rope_frequencies(cfg.head_dim, 10, cfg.rope_theta)

    with torch.no_grad():
        base, _ = attn(x, cos, sin)
        changed = x.clone()
        changed[:, 6:] += 5.0
        after, _ = attn(changed, cos, sin)

    assert torch.allclose(base[:, :6], after[:, :6], atol=1e-5)
    assert not torch.allclose(base[:, 6:], after[:, 6:], atol=1e-3)


def test_attention_runs_without_rope(cfg):
    attn = GroupedQueryAttention(cfg)
    assert attn(torch.randn(1, 5, cfg.d_model))[0].shape == (1, 5, cfg.d_model)


def test_block_matches_the_config_parameter_estimate(cfg):
    block = TransformerBlock(cfg, 3)
    kv = cfg.n_kv_heads * cfg.head_dim
    expected = (
        2 * cfg.d_model * cfg.d_model
        + 2 * cfg.d_model * kv
        + 3 * cfg.d_model * cfg.ffn_hidden(3)
        + 2 * cfg.d_model
    )
    assert sum(p.numel() for p in block.parameters()) == expected


def test_block_knows_its_layer_kind(cfg):
    assert TransformerBlock(cfg, 1).kind == "dense"
    assert TransformerBlock(cfg, 6).kind == "anchor"
    assert TransformerBlock(proxy_config(use_gdn=True), 5).kind == "linear"


def test_block_is_residual_and_causal(cfg):
    torch.manual_seed(0)
    block = TransformerBlock(cfg, 3).eval()
    x = torch.randn(1, 9, cfg.d_model)
    cos, sin = rope_frequencies(cfg.head_dim, 9, cfg.rope_theta)
    with torch.no_grad():
        out, _ = block(x, cos, sin)
        changed = x.clone()
        changed[:, 5:] += 3.0
        after, _ = block(changed, cos, sin)
    assert out.shape == x.shape
    assert torch.allclose(out[:, :5], after[:, :5], atol=1e-5)


def test_gradients_reach_every_parameter(cfg):
    block = TransformerBlock(cfg, 3)
    cos, sin = rope_frequencies(cfg.head_dim, 6, cfg.rope_theta)
    block(torch.randn(2, 6, cfg.d_model), cos, sin)[0].square().mean().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
