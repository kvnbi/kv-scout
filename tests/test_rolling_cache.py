from __future__ import annotations

import copy

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import Cache, KVScout
from kv_scout.model.attention import GroupedQueryAttention
from kv_scout.model.layers import apply_rope, rope_frequencies, rope_shift, rotate_rope
from kv_scout.model.sinks import evict, retained_positions


def streaming(**overrides):
    base = dict(
        use_gdn=True, d_model=192, n_layers=12, n_query_heads=2, n_kv_heads=1,
        head_dim=96, dense_ffn_hidden=384, attention_anchor_layers=(6, 9, 12),
        context_max=64, context_min=64, cross_layer_kv_sharing=True,
        attention_sinks=True, sink_tokens=3, attention_window=12,
    )
    base.update(overrides)
    return proxy_config(**base)


def built(**overrides):
    cfg = streaming(**overrides)
    torch.manual_seed(0)
    return cfg, KVScout(cfg).eval()


def stream(model, cfg, cache, steps, width=1):
    with torch.no_grad():
        for _ in range(steps):
            out = model(torch.randint(0, cfg.vocab_size, (1, width)), cache)
    return out


def test_rotating_a_key_back_matches_a_rope_at_the_shifted_position():
    head_dim, theta = 96, 10000.0
    cos, sin = rope_frequencies(head_dim, 400, theta)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, head_dim)
    for start, shift in ((40, 17), (200, 128), (5, 1)):
        direct = apply_rope(x, cos[start - shift :], sin[start - shift :])
        shifted = rope_shift(head_dim, shift, theta)
        stepped = rotate_rope(
            apply_rope(x, cos[start:], sin[start:]), shifted[0], -shifted[1]
        )
        assert torch.allclose(direct, stepped, atol=1e-4)


def test_a_slide_leaves_the_next_token_untouched():
    cfg, model = built(attention_sinks=False, use_gdn=False)
    warm = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 20)), warm)
        slid = copy.deepcopy(warm)
        slid._slide(7)
        nxt = torch.randint(0, cfg.vocab_size, (1, 1))
        assert torch.allclose(model(nxt, warm), model(nxt, slid), atol=1e-3)


def test_a_slide_carries_the_recurrent_state_with_it():
    cfg, model = built(attention_sinks=False, use_gdn=True)
    warm = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 20)), warm)
        slid = copy.deepcopy(warm)
        slid._slide(7)
        nxt = torch.randint(0, cfg.vocab_size, (1, 1))
        assert torch.allclose(model(nxt, warm), model(nxt, slid), atol=1e-3)


def test_a_long_stream_never_loses_a_live_key():
    cfg, model = built()
    cache = Cache(cfg)
    with torch.no_grad():
        for _ in range(400):
            model(torch.randint(0, cfg.vocab_size, (1, 1)), cache)
            for group in cache.fill:
                if not cfg.windows_attention(group):
                    continue
                sinks, window = cache.geometry(group)
                wanted = retained_positions(cache.length, sinks, window)
                held = set(cache.positions[group].tolist())
                assert set(wanted).issubset(held)


def test_decoding_never_reallocates_the_cache():
    cfg, model = built()
    cache = Cache(cfg)
    stream(model, cfg, cache, 90)
    settled = {g: cache.storage_keys[g].data_ptr() for g in cache.fill}
    stream(model, cfg, cache, 200)
    assert cache.evictions > 0
    assert {g: cache.storage_keys[g].data_ptr() for g in cache.fill} == settled


def test_memory_plateaus_across_a_long_stream():
    cfg, model = built()
    cache = Cache(cfg)
    sizes = []
    for _ in range(6):
        stream(model, cfg, cache, 50)
        sizes.append(cache.bytes())
    assert len(set(sizes[1:])) == 1
    for group in cache.fill:
        assert cache.fill[group] <= cache.capacity(group)
        assert cache.storage_keys[group].shape[2] == cache.capacity(group)


def test_the_stream_rolls_rather_than_starting_over():
    cfg, model = built()
    cache = Cache(cfg)
    stream(model, cfg, cache, 300)
    assert cache.length == 300
    assert cache.evictions > 0
    assert cache.rebases > 0
    assert cache.base() <= cfg.context_max


def test_the_first_tokens_still_reach_a_query_far_past_the_context():
    cfg, model = built()

    def run(first):
        torch.manual_seed(7)
        ids = torch.randint(0, cfg.vocab_size, (1, 300))
        ids[0, 0] = first
        cache = Cache(cfg)
        with torch.no_grad():
            for i in range(300):
                out = model(ids[:, i : i + 1], cache)
        return out[0, -1]

    assert not torch.allclose(run(11), run(9999), atol=1e-3)


def test_a_chunked_continuation_matches_a_single_pass():
    cfg, model = built(attention_sinks=False, context_max=128, context_min=128)
    tokens = torch.randint(0, cfg.vocab_size, (1, 22))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        pieces = [model(tokens[:, lo:hi], cache) for lo, hi in ((0, 10), (10, 17), (17, 22))]
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-3)


@pytest.mark.parametrize("batch", [1, 3])
def test_a_batched_stream_matches_a_single_pass(batch):
    cfg, model = built()
    tokens = torch.randint(0, cfg.vocab_size, (batch, 26))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        step = torch.cat([model(tokens[:, i : i + 1], cache) for i in range(26)], dim=1)
    assert torch.allclose(full, step, atol=1e-3)


def test_a_batch_shape_change_is_refused():
    cfg, model = built()
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (2, 4)), cache)
        with pytest.raises(ValueError):
            model(torch.randint(0, cfg.vocab_size, (5, 1)), cache)


def test_a_bfloat16_stream_stays_finite():
    cfg, model = built(precision="bfloat16")
    model = model.to(torch.bfloat16)
    cache = Cache(cfg)
    out = stream(model, cfg, cache, 200)
    assert out.dtype is torch.bfloat16
    assert bool(torch.isfinite(out).all())
    assert cache.rebases > 0


def test_the_recurrent_state_survives_the_whole_stream():
    cfg, model = built()
    cache = Cache(cfg)
    stream(model, cfg, cache, 300)
    linear = [i for i in range(1, cfg.n_layers + 1) if cfg.layer_kind(i) == "linear"]
    assert linear
    for layer in linear:
        state = cache.get_state(layer)
        assert state is not None
        assert bool(torch.isfinite(state).all())


def test_the_training_store_holds_only_the_sequence():
    cfg, model = built(context_max=512, context_min=512)
    seen = {}
    original = Cache.append

    def watched(self, group, key, value):
        result = original(self, group, key, value)
        seen[group] = self.storage_keys[group].shape[2]
        return result

    Cache.append = watched
    try:
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (1, 32)))
    finally:
        Cache.append = original
    assert seen
    assert set(seen.values()) == {32}


def test_multi_token_heads_keep_working_past_the_roll():
    cfg, model = built(mtp_heads=2, mtp_ffn_hidden=128)
    cache = Cache(cfg)
    stream(model, cfg, cache, 150)
    with torch.no_grad():
        logits, ahead = model(
            torch.randint(0, cfg.vocab_size, (1, 4)), cache, predict_ahead=True
        )
    assert len(ahead) == 2
    assert logits.shape[1] == 4
    assert all(bool(torch.isfinite(p).all()) for p in ahead)


def test_in_place_eviction_agrees_with_the_reference():
    cfg = streaming()
    cache = Cache(cfg)
    torch.manual_seed(0)
    group = 6
    sinks, window = cache.geometry(group)
    keys = torch.randn(1, 1, 40, cfg.head_dim)
    values = torch.randn(1, 1, 40, cfg.head_dim)
    cache.length = 0
    cache.append(group, keys, values)
    cache.advance(40)
    cache.trim()
    wanted_k, wanted_v = evict(keys, values, sinks, window)
    held_k, held_v, held_p = cache.read(group)
    assert torch.equal(held_k, wanted_k)
    assert torch.equal(held_v, wanted_v)
    assert held_p.tolist() == retained_positions(40, sinks, window)


def test_a_large_append_onto_a_full_cache_matches_a_single_pass():
    cfg, model = built(attention_window=12, sink_tokens=3)
    tokens = torch.randint(0, cfg.vocab_size, (1, 50))
    cache = Cache(cfg)
    with torch.no_grad():
        full = model(tokens)
        head = model(tokens[:, :20], cache)
        grown = max(cache.storage_keys[g].shape[2] for g in cache.fill)
        tail = model(tokens[:, 20:], cache)
    assert torch.allclose(full, torch.cat([head, tail], dim=1), atol=1e-3)
    for group in cache.fill:
        assert cache.storage_keys[group].shape[2] == cache.capacity(group)
        assert cache.fill[group] <= cache.capacity(group)
    assert grown > 0


def test_growing_for_one_big_append_does_not_leave_the_buffer_wide():
    cfg, model = built(attention_window=12, sink_tokens=3)
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 10)), cache)
        model(torch.randint(0, cfg.vocab_size, (1, 50)), cache)
    for group in cache.fill:
        assert cache.storage_keys[group].shape[2] == cache.capacity(group)
        assert cache.storage_positions[group].shape[0] == cache.capacity(group)
        assert cache.storage_values[group].shape[2] == cache.capacity(group)


def test_one_rolling_layer_matches_a_fresh_pass_over_the_window():
    cfg = streaming(use_gdn=False, attention_sinks=False, context_max=32, context_min=32)
    torch.manual_seed(0)
    layer = GroupedQueryAttention(cfg, 1).eval()
    cos, sin = rope_frequencies(cfg.head_dim, 4 * cfg.context_max, cfg.rope_theta)
    cache = Cache(cfg)
    window = cache.window(1)
    x = torch.randn(1, 140, cfg.d_model)
    rolling = []
    with torch.no_grad():
        for i in range(140):
            base = cache.rebase(1)
            out, _ = layer(
                x[:, i : i + 1], cos[base : base + 1], sin[base : base + 1], None, cache
            )
            cache.trim()
            cache.advance(1)
            rolling.append(out[0, -1])
        assert cache.rebases > 0
        for p in range(window, 140):
            fresh, _ = layer(
                x[:, p - window + 1 : p + 1], cos[:window], sin[:window], None, None
            )
            assert torch.allclose(rolling[p], fresh[0, -1], atol=1e-5)


def test_the_training_store_adopts_the_tensors_instead_of_copying():
    cfg, model = built(context_max=512, context_min=512)
    adopted = []
    original = Cache.append

    def watched(self, group, key, value):
        result = original(self, group, key, value)
        adopted.append(self.storage_keys[group].data_ptr() == key.data_ptr())
        return result

    Cache.append = watched
    try:
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (1, 32)))
    finally:
        Cache.append = original
    assert adopted
    assert all(adopted)


def test_an_impossible_rotary_frame_is_refused():
    cfg = streaming()
    cache = Cache(cfg)
    cache.length = 5
    with pytest.raises(ValueError):
        cache.rebase(cfg.context_max + 10)


def test_the_sinks_stay_pinned_across_a_long_stream():
    cfg, model = built()
    cache = Cache(cfg)
    stream(model, cfg, cache, 900)
    assert cache.rebases > 0
    for group in cache.fill:
        sinks = cache.sinks(group)
        if sinks and cache.fill[group] > sinks:
            assert cache.positions[group][:sinks].tolist() == list(range(sinks))
        assert 0 <= cache.base() <= cfg.context_max


def test_repeated_rotation_matches_one_large_rotation():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 40, 96)
    one = rope_shift(96, 1, 10000.0)
    stepped = x
    for _ in range(500):
        stepped = rotate_rope(stepped, one[0], -one[1])
    bulk = rope_shift(96, 500, 10000.0)
    assert torch.allclose(stepped, rotate_rope(x, bulk[0], -bulk[1]), atol=1e-3)


def test_a_reset_cache_can_be_streamed_again():
    cfg, model = built()
    cache = Cache(cfg)
    stream(model, cfg, cache, 120)
    assert cache.rebases > 0
    cache.reset()
    assert cache.length == 0 and cache.origin == 0 and cache.bytes() == 0
    out = stream(model, cfg, cache, 120)
    assert bool(torch.isfinite(out).all())
    assert cache.length == 120


def test_two_identical_streams_agree_bit_for_bit():
    cfg, model = built()
    results = []
    for _ in range(2):
        torch.manual_seed(11)
        cache = Cache(cfg)
        results.append(stream(model, cfg, cache, 200))
    assert torch.equal(results[0], results[1])


def test_back_to_back_full_context_calls_keep_rolling():
    cfg, model = built()
    cache = Cache(cfg)
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, cfg.context_max)), cache)
        out = model(torch.randint(0, cfg.vocab_size, (1, cfg.context_max)), cache)
    assert cache.length == 2 * cfg.context_max
    assert cache.rebases > 0
    assert bool(torch.isfinite(out).all())
