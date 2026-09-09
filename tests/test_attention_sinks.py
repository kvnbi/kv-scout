from __future__ import annotations

import pytest
import torch

from kv_scout.config import ModelConfig, proxy_config
from kv_scout.model import Cache, KVScout
from kv_scout.model.sinks import (
    evict,
    retained_positions,
    sink_window_mask,
    visibility_mask,
)


def windowed(sinks=2, window=8, sink_on=True, **overrides):
    base = dict(
        use_gdn=True, attention_sinks=sink_on, sink_tokens=sinks,
        attention_window=window, d_model=192, n_layers=12, n_query_heads=2,
        n_kv_heads=1, head_dim=96, dense_ffn_hidden=384,
        attention_anchor_layers=(6, 9, 12), context_max=64, context_min=64,
    )
    base.update(overrides)
    return proxy_config(**base)


def test_mask_keeps_sinks_and_the_recent_window():
    mask = sink_window_mask(8, 8, sinks=2, window=3)
    assert mask[7].int().tolist() == [1, 1, 0, 0, 0, 1, 1, 1]
    assert mask[0].int().tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert mask[4].int().tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_mask_is_always_causal():
    mask = sink_window_mask(12, 12, sinks=3, window=4)
    for query in range(12):
        assert not mask[query, query + 1 :].any()
        assert mask[query, query]


def test_every_query_can_see_at_least_itself():
    for window in (1, 2, 9):
        mask = sink_window_mask(10, 10, sinks=0, window=window)
        assert mask.sum(dim=1).min() >= 1


def test_no_window_means_plain_causal():
    mask = sink_window_mask(6, 6, sinks=0, window=6)
    assert torch.equal(mask, torch.tril(torch.ones(6, 6, dtype=torch.bool)))


def test_evict_keeps_the_head_and_the_tail():
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1)
    values = keys.clone()
    trimmed_k, trimmed_v = evict(keys, values, sinks=2, window=3)
    assert trimmed_k.flatten().tolist() == [0, 1, 9, 10, 11]
    assert torch.equal(trimmed_k, trimmed_v)


def test_evict_is_a_no_op_below_the_budget():
    keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    trimmed, _ = evict(keys, keys.clone(), sinks=2, window=3)
    assert torch.equal(trimmed, keys)


def test_retained_positions_match_what_a_final_query_can_see():
    length, sinks, window = 20, 3, 5
    kept = retained_positions(length, sinks, window)
    mask = sink_window_mask(length, length, sinks, window)
    visible = [i for i in range(length) if mask[length - 1, i]]
    assert kept == visible


def test_visibility_mask_respects_evicted_positions():
    positions = torch.tensor([0, 1, 24, 25, 26])
    mask = visibility_mask(positions, query_length=1, offset=26, sinks=2, window=3)
    assert mask.shape == (1, 1, 1, 5)
    assert mask[0, 0, 0].int().tolist() == [1, 1, 1, 1, 1]


def test_visibility_mask_drops_a_key_that_has_aged_out():
    positions = torch.tensor([0, 1, 20, 21, 22])
    mask = visibility_mask(positions, query_length=1, offset=23, sinks=2, window=3)
    assert mask[0, 0, 0].int().tolist() == [1, 1, 0, 1, 1]


def test_only_anchors_are_windowed():
    cfg = windowed()
    for layer in range(1, cfg.n_layers + 1):
        expected = cfg.layer_kind(layer) == "anchor"
        assert cfg.windows_attention(layer) is expected


def test_sinks_can_be_switched_off():
    cfg = windowed(sink_on=False)
    for layer in range(1, cfg.n_layers + 1):
        assert cfg.windows_attention(layer) is False


def test_configuration_is_validated():
    with pytest.raises(ValueError):
        ModelConfig(sink_tokens=-1)
    with pytest.raises(ValueError):
        ModelConfig(attention_window=0)


def test_generation_matches_a_full_pass_with_windowing():
    cfg = windowed(sinks=2, window=8)
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 32))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        step = torch.cat([model(tokens[:, i : i + 1], cache) for i in range(32)], dim=1)
    assert torch.allclose(full, step, atol=1e-3)


def test_prefill_then_continue_matches_a_full_pass():
    cfg = windowed(sinks=2, window=8)
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 28))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        head = model(tokens[:, :16], cache)
        tail = torch.cat(
            [model(tokens[:, i : i + 1], cache) for i in range(16, 28)], dim=1
        )
    assert torch.allclose(full, torch.cat([head, tail], dim=1), atol=1e-3)


def test_the_anchor_cache_stops_growing():
    cfg = windowed(sinks=2, window=8)
    model = KVScout(cfg).eval()
    cache = Cache(cfg)
    sizes = []
    with torch.no_grad():
        for i in range(40):
            model(torch.randint(0, cfg.vocab_size, (1, 1)), cache)
            sizes.append(int(cache.keys[6].shape[2]))
    assert max(sizes) == cfg.sink_tokens + cfg.attention_window
    assert sizes[-1] == sizes[-5]


def test_dense_layers_keep_the_whole_history():
    cfg = windowed(sinks=2, window=8)
    model = KVScout(cfg).eval()
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 40)), cache)
    assert cache.keys[1].shape[2] == 40
    assert cache.keys[6].shape[2] == 10


def test_the_first_tokens_are_never_evicted():
    cfg = windowed(sinks=3, window=6)
    model = KVScout(cfg).eval()
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 50)), cache)
    kept = cache.positions[6].tolist()
    assert kept[:3] == [0, 1, 2]
    assert kept[-1] == 49


def test_windowing_costs_no_parameters():
    on, off = windowed(sink_on=True), windowed(sink_on=False)
    assert on.parameter_estimate() == off.parameter_estimate()
    assert KVScout(on).num_parameters() == KVScout(off).num_parameters()


def test_windowing_and_sharing_work_together():
    cfg = windowed(sinks=2, window=8, cross_layer_kv_sharing=True)
    torch.manual_seed(0)
    model = KVScout(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 30))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        step = torch.cat([model(tokens[:, i : i + 1], cache) for i in range(30)], dim=1)
    assert torch.allclose(full, step, atol=1e-3)
    anchors = {cfg.cache_group(i) for i in cfg.attention_anchor_layers}
    assert len(anchors) == 2
    for group in anchors:
        assert cache.keys[group].shape[2] == 10
