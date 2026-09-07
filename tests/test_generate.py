from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kv_scout.generate import (
    DEFAULT_BANNED,
    SamplingConfig,
    SequenceBan,
    byte_level_alphabet,
    generate,
    token_bytes,
)

TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer.json"
DASHES = "\u2014\u2013\u2015\u2012"

pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="tokenizer missing")


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER))


@pytest.fixture(scope="module")
def ban(tok):
    return SequenceBan(tok)


class Puppet(torch.nn.Module):
    def __init__(self, wanted, vocab=32768):
        super().__init__()
        self.cfg = type("C", (), {"context_max": 64})()
        self.wanted = wanted
        self.step = 0
        self.vocab = vocab

    def forward(self, tokens):
        logits = torch.full((1, tokens.shape[1], self.vocab), -10.0)
        logits[0, -1, self.wanted[self.step % len(self.wanted)]] = 50.0
        self.step += 1
        return logits


def test_byte_alphabet_covers_every_byte():
    alphabet = byte_level_alphabet()
    assert len(alphabet) == 256
    assert sorted(alphabet.values()) == list(range(256))


def test_token_bytes_reconstructs_multi_token_characters(tok):
    table = token_bytes(tok)
    ids = tok.encode("\u2014").ids
    assert len(ids) > 1
    assert b"".join(table[i] for i in ids).decode("utf-8") == "\u2014"


def test_no_single_token_contains_a_dash(tok):
    table = token_bytes(tok)
    for index, raw in table.items():
        assert not any(d.encode("utf-8") in raw for d in DASHES)


def test_ban_builds_states_for_every_pattern_prefix(ban):
    assert b"" in ban.states
    assert b"\xe2\x80" in ban.states
    assert b" --" in ban.states
    assert ban.blocked[b""] == set()
    assert len(ban.blocked[b"\xe2\x80"]) > 0


def test_ban_blocks_a_model_that_insists_on_dashes(tok, ban):
    wanted = tok.encode("a \u2014 b \u2014 c").ids
    plain = generate(
        Puppet(wanted), tok, "x", SamplingConfig(max_new_tokens=40, greedy=True)
    )
    assert sum(plain["text"].count(d) for d in DASHES) > 0

    guarded = generate(
        Puppet(wanted), tok, "x", SamplingConfig(max_new_tokens=40, greedy=True), ban=ban
    )
    assert sum(guarded["text"].count(d) for d in DASHES) == 0
    assert guarded["ban_events"] > 0


def test_every_banned_pattern_is_actually_blocked(tok, ban):
    for pattern in DEFAULT_BANNED:
        wanted = tok.encode(f"x{pattern}y").ids
        out = generate(
            Puppet(wanted),
            tok,
            "x",
            SamplingConfig(max_new_tokens=48, greedy=True),
            ban=ban,
        )
        assert pattern not in out["text"], f"{pattern!r} survived the ban"


def test_ban_leaves_ordinary_punctuation_alone(tok, ban):
    for text in ("well-known", "a python --flag here", "one, two. three", "x - y"):
        wanted = tok.encode(text).ids
        out = generate(
            Puppet(wanted),
            tok,
            "x",
            SamplingConfig(max_new_tokens=len(wanted), greedy=True),
            ban=ban,
        )
        assert out["ban_events"] == 0, f"{text!r} triggered the ban"
        assert out["text"].strip() == text.strip()


def test_ban_never_blocks_a_token_in_ordinary_text(tok, ban):
    for text in ("well-known", "a python --flag here", "one, two. three", "x - y"):
        state = ban.start()
        for token in tok.encode(text).ids:
            assert token not in ban.blocked[state], f"{text!r} blocked a token"
            state = ban.advance(state, token)


def test_hyphen_survives_but_the_spaced_double_does_not(tok, ban):
    table = token_bytes(tok)
    hyphen = tok.encode("-").ids[0]
    assert hyphen not in ban.blocked[b""]
    assert table[hyphen] == b"-"


def test_ban_state_tracks_a_prompt_that_already_contains_a_dash(tok, ban):
    assert ban.state_for(tok.encode("hello \u2014 there").ids) == b""
    partial = tok.encode("hello \u2014").ids[:-1]
    assert ban.state_for(partial) == b"\xe2\x80"


def test_generation_is_deterministic_under_a_seed(tok):
    wanted = list(range(300, 340))
    a = generate(Puppet(wanted), tok, "x", SamplingConfig(max_new_tokens=16, seed=3))
    b = generate(Puppet(wanted), tok, "x", SamplingConfig(max_new_tokens=16, seed=3))
    assert a["tokens"] == b["tokens"]


def test_greedy_picks_the_argmax(tok):
    wanted = [500]
    out = generate(
        Puppet(wanted), tok, "x", SamplingConfig(max_new_tokens=5, greedy=True)
    )
    assert out["tokens"] == [500] * 5


def test_generation_respects_the_token_budget(tok):
    out = generate(
        Puppet([700]), tok, "hello", SamplingConfig(max_new_tokens=12, greedy=True)
    )
    assert len(out["tokens"]) == 12
    assert out["prompt_tokens"] == len(tok.encode("hello").ids)


def test_empty_prompt_starts_from_end_of_text(tok):
    out = generate(Puppet([700]), tok, "", SamplingConfig(max_new_tokens=3, greedy=True))
    assert out["prompt_tokens"] == 1
