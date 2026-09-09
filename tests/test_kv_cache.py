from __future__ import annotations

import pytest
import torch

from kv_scout.config import ModelConfig, proxy_config
from kv_scout.model import Cache, KVScout


def hybrid(share: bool, **overrides):
    base = dict(
        use_gdn=True, cross_layer_kv_sharing=share, d_model=192, n_layers=12,
        n_query_heads=2, n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        context_max=64, context_min=64,
    )
    base.update(overrides)
    return proxy_config(**base)


def test_only_attention_layers_cache_keys():
    cfg = hybrid(True)
    for layer in range(1, cfg.n_layers + 1):
        expected = cfg.layer_kind(layer) in ("dense", "anchor", "attention")
        assert cfg.caches_keys(layer) is expected


def test_asking_for_a_group_on_a_linear_layer_is_an_error():
    cfg = hybrid(True)
    linear = next(i for i in range(1, cfg.n_layers + 1) if cfg.layer_kind(i) == "linear")
    with pytest.raises(ValueError):
        cfg.cache_group(linear)


def test_sharing_halves_the_anchor_caches_on_the_full_model():
    shared = ModelConfig()
    alone = ModelConfig(cross_layer_kv_sharing=False)
    a = len({shared.cache_group(i) for i in shared.attention_anchor_layers})
    b = len({alone.cache_group(i) for i in alone.attention_anchor_layers})
    assert b == 8
    assert a == 4
    assert a / b == 0.5


def test_group_size_controls_the_reduction():
    for size, expected in ((1, 8), (2, 4), (8, 1)):
        cfg = ModelConfig(kv_sharing_group_size=size)
        assert len({cfg.cache_group(i) for i in cfg.attention_anchor_layers}) == expected


def test_group_size_is_validated():
    with pytest.raises(ValueError):
        ModelConfig(kv_sharing_group_size=0)


def test_each_group_has_exactly_one_owner():
    cfg = ModelConfig()
    caching = [i for i in range(1, cfg.n_layers + 1) if cfg.caches_keys(i)]
    owners = [i for i in caching if cfg.owns_cache(i)]
    assert len(owners) == cfg.cache_group_count
    for layer in caching:
        assert cfg.cache_group(layer) in owners


def test_dense_layers_never_share():
    cfg = ModelConfig()
    for layer in (1, 2):
        assert cfg.layer_kind(layer) == "dense"
        assert cfg.owns_cache(layer) is True


@pytest.mark.parametrize("share", [False, True])
def test_token_by_token_matches_a_single_pass(share):
    cfg = hybrid(share, attention_anchor_layers=(6, 9, 12))
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 24))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        stepwise = torch.cat([model(tokens[:, i : i + 1], cache) for i in range(24)], dim=1)
    assert torch.allclose(full, stepwise, atol=1e-3)


@pytest.mark.parametrize("share", [False, True])
def test_a_prefill_then_continuation_matches_a_single_pass(share):
    cfg = hybrid(share, attention_anchor_layers=(6, 9, 12))
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 20))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        head = model(tokens[:, :12], cache)
        tail = torch.cat([model(tokens[:, i : i + 1], cache) for i in range(12, 20)], dim=1)
    assert torch.allclose(full, torch.cat([head, tail], dim=1), atol=1e-3)


def test_sharing_changes_the_model_during_training_too():
    tokens = torch.randint(0, 32768, (1, 16))
    outputs = []
    for share in (False, True):
        cfg = hybrid(share, attention_anchor_layers=(6, 9, 12))
        torch.manual_seed(0)
        with torch.no_grad():
            outputs.append(KVScout(cfg).eval()(tokens))
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-3)


def test_sharing_shrinks_the_cache():
    tokens = torch.randint(0, 32768, (1, 32))
    sizes = {}
    for share in (False, True):
        cfg = hybrid(share, attention_anchor_layers=(6, 9, 12))
        cache = Cache(cfg)
        torch.manual_seed(0)
        with torch.no_grad():
            KVScout(cfg).eval()(tokens, cache)
        sizes[share] = (cache.groups, cache.bytes())
    assert sizes[True][0] < sizes[False][0]
    assert sizes[True][1] < sizes[False][1]


def test_cache_tracks_its_own_length():
    cfg = hybrid(True, attention_anchor_layers=(6, 9, 12))
    model = KVScout(cfg).eval()
    cache = Cache(cfg)
    assert cache.length == 0
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 5)), cache)
        assert cache.length == 5
        model(torch.randint(0, cfg.vocab_size, (1, 3)), cache)
        assert cache.length == 8


def test_cache_reset_clears_everything():
    cfg = hybrid(True, attention_anchor_layers=(6, 9, 12))
    cache = Cache(cfg)
    with torch.no_grad():
        KVScout(cfg).eval()(torch.randint(0, cfg.vocab_size, (1, 6)), cache)
    assert cache.groups > 0 and cache.length == 6
    cache.reset()
    assert cache.groups == 0 and cache.length == 0 and cache.bytes() == 0


def test_gdn_state_is_carried_in_the_cache():
    cfg = hybrid(True, attention_anchor_layers=(6, 9, 12))
    model = KVScout(cfg).eval()
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 4)), cache)
    linear = [i for i in range(1, cfg.n_layers + 1) if cfg.layer_kind(i) == "linear"]
    for layer in linear:
        state = cache.get_state(layer)
        assert state is not None
        assert state.shape == (1, cfg.n_query_heads, cfg.head_dim, cfg.head_dim)


def test_reading_an_unwritten_group_raises():
    with pytest.raises(KeyError):
        Cache(ModelConfig()).read(99)


def test_the_cache_refuses_to_run_past_the_context():
    cfg = hybrid(True, attention_anchor_layers=(6, 9, 12), context_max=8, context_min=8)
    model = KVScout(cfg).eval()
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 6)), cache)
        with pytest.raises(ValueError):
            model(torch.randint(0, cfg.vocab_size, (1, 4)), cache)


def test_sharing_costs_no_parameters():
    a = hybrid(True, attention_anchor_layers=(6, 9, 12))
    b = hybrid(False, attention_anchor_layers=(6, 9, 12))
    assert a.parameter_estimate() == b.parameter_estimate()
    assert KVScout(a).num_parameters() == KVScout(b).num_parameters()
