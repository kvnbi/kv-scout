from __future__ import annotations

import pytest

from kv_scout.data.normalise import DASHES, count_dashes, normalise_dashes


def test_spaced_dash_becomes_a_comma():
    assert normalise_dashes("something \u2014 offices") == "something, offices"


def test_numeric_ranges_become_hyphens():
    assert normalise_dashes("1914\u20131918") == "1914-1918"
    assert normalise_dashes("pages 10 \u2014 20") == "pages 10-20"


def test_intra_word_dash_becomes_a_hyphen():
    assert normalise_dashes("a well\u2014known fact") == "a well-known fact"


def test_dialogue_opener_is_left_alone():
    assert normalise_dashes("\u2014 I told you") == "\u2014 I told you"
    assert normalise_dashes("  \u2014 yes") == "  \u2014 yes"


def test_a_dash_later_in_a_dialogue_line_is_still_normalised():
    assert normalise_dashes("\u2014 yes \u2014 he said") == "\u2014 yes, he said"


def test_punctuation_artifacts_are_cleaned_up():
    assert normalise_dashes("he paused\u2014.") == "he paused."
    assert normalise_dashes("one, \u2014 two") == "one, two"
    assert normalise_dashes("wait\u2014!") == "wait!"


def test_text_without_dashes_is_returned_unchanged():
    text = "no dashes at all, just a hyphen-joined word"
    assert normalise_dashes(text) is text


def test_every_dash_character_is_handled():
    for dash in DASHES:
        out = normalise_dashes(f"left {dash} right")
        assert dash not in out, f"{dash!r} survived"


def test_line_structure_is_preserved():
    text = "line one \u2014 x\nline two \u2013 y\n"
    assert normalise_dashes(text) == "line one, x\nline two, y\n"


def test_count_dashes():
    assert count_dashes("a \u2014 b \u2013 c") == 2
    assert count_dashes("none here") == 0


@pytest.mark.parametrize(
    "text",
    [
        "The colour \u2014 of the centre",
        "1914\u20131918 \u2014 a long war",
        "\u2014 dialogue, then \u2014 an aside",
    ],
)
def test_normalising_twice_changes_nothing_further(text):
    once = normalise_dashes(text)
    assert normalise_dashes(once) == once
