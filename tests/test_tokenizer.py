from __future__ import annotations

import json
from pathlib import Path

import pytest

from kv_scout.config import TokenizerConfig
from kv_scout.tokenizer import (
    BRITISH_SEED_FORMS,
    END_OF_TEXT,
    REFLECTION_SLOTS,
    REFLECTION_TOKENS,
    STRUCTURAL_TOKENS,
    reserved_tokens,
    train,
)

ARTIFACT = Path("data/tokenizer/tokenizer.json")


def test_reserved_slots_match_the_spec():
    reserved = reserved_tokens()
    assert len(reserved) == REFLECTION_SLOTS == 32
    assert reserved[: len(REFLECTION_TOKENS)] == REFLECTION_TOKENS
    assert len(set(reserved)) == len(reserved)
    assert REFLECTION_SLOTS == TokenizerConfig().reflection_slots


def test_every_reflection_family_is_present():
    prefixes = {t.split(":")[0] for t in REFLECTION_TOKENS}
    assert prefixes == {"[RETRIEVE", "[REL", "[SUP", "[USE", "[MORE"}
    assert sum(t.startswith("[USE") for t in REFLECTION_TOKENS) == 5


def test_structural_tokens_are_unique_and_include_eos():
    assert END_OF_TEXT in STRUCTURAL_TOKENS
    assert len(set(STRUCTURAL_TOKENS)) == len(STRUCTURAL_TOKENS)


def test_train_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train([tmp_path / "absent.txt"], tmp_path / "tokenizer.json")


def test_train_produces_an_exact_vocabulary(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("the colour of the centre and the defence of it\n" * 400)
        + ("def compute(self, x):\n    return x + 1\n" * 400)
    )
    out = tmp_path / "tok" / "tokenizer.json"
    report = train([corpus], out, vocab_size=1024, min_frequency=1)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(out))
    assert tok.get_vocab_size() == 1024 == report["vocab_size"]
    assert report["eos_id"] == tok.token_to_id(END_OF_TEXT)
    assert json.loads((out.parent / "tokenizer_report.json").read_text()) == report


@pytest.mark.skipif(not ARTIFACT.exists(), reason="tokenizer not trained yet")
class TestTrainedArtifact:
    @pytest.fixture(scope="class")
    @classmethod
    def tok(cls):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(ARTIFACT))

    def test_vocabulary_size_matches_the_config(self, tok):
        assert tok.get_vocab_size() == TokenizerConfig().vocab_size

    def test_reflection_tokens_are_single_ids(self, tok):
        for token in REFLECTION_TOKENS:
            assert len(tok.encode(token).ids) == 1

    def test_structural_tokens_are_single_ids(self, tok):
        for token in STRUCTURAL_TOKENS:
            assert len(tok.encode(token).ids) == 1

    def test_british_forms_never_cost_more_than_american(self, tok):
        pairs = [
            ("colour", "color"),
            ("realise", "realize"),
            ("realised", "realized"),
            ("centre", "center"),
            ("defence", "defense"),
            ("travelled", "traveled"),
            ("analyse", "analyze"),
            ("organise", "organize"),
            ("behaviour", "behavior"),
            ("licence", "license"),
        ]
        for british, american in pairs:
            nb = len(tok.encode(" " + british).ids)
            na = len(tok.encode(" " + american).ids)
            assert nb <= na, f"{british} costs {nb} against {american} at {na}"

    def test_every_seed_form_is_a_single_token(self, tok):
        for form in BRITISH_SEED_FORMS:
            assert len(tok.encode(" " + form).ids) == 1

    def test_round_trips_english_german_code_and_emoji(self, tok):
        samples = [
            "The colour of the centre was a testament to her behaviour.",
            "Die Farbe der Mitte war ein Zeugnis ihres Verhaltens.",
            "def compute(self, x):\n    return [i ** 2 for i in x]\n",
            "cost 5 euros € and a \U0001f600 face",
        ]
        for text in samples:
            assert tok.decode(tok.encode(text).ids) == text

    def test_ids_fit_in_uint16(self, tok):
        assert max(tok.get_vocab().values()) < 65536
