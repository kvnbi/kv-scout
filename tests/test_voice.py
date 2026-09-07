from __future__ import annotations

import json
from pathlib import Path

import pytest

from kv_scout.voice import (
    COMPARED,
    LLM_VOCABULARY,
    compare,
    count_dashes,
    format_comparison,
    load_baseline,
    measure_files,
    measure_voice,
    opening_closing_overlap,
    save_baseline,
)

BASELINE = Path(__file__).resolve().parents[1] / "tokenizer" / "voice_baseline.json"

HUMAN = (
    "The bridge opened in 1894. It carried traffic across the river for eighty years "
    "before the council closed it for repairs. Local records show the cost ran well "
    "over the original estimate, and the contractor went bankrupt."
)

MACHINE = (
    "# Overview\n\n"
    "It is crucial to delve into the rich tapestry of this landscape of ideas. "
    "Moreover, we must leverage robust and seamless frameworks. Furthermore, this "
    "is not just a change, but a transformation.\n\n"
    "- **Key point one**\n- **Key point two**\n\n"
    "In conclusion, it is crucial to delve into this tapestry."
)


def test_counts_every_dash_character():
    assert count_dashes("a \u2014 b \u2013 c \u2015 d \u2012 e") == 4
    assert count_dashes("a - b") == 0


def test_separates_human_prose_from_machine_prose():
    human = measure_voice(HUMAN)
    machine = measure_voice(MACHINE)
    assert machine.llm_vocabulary_per_1k > human.llm_vocabulary_per_1k * 5
    assert machine.markdown_density_per_1k > human.markdown_density_per_1k
    assert machine.not_just_but_per_1k > 0
    assert human.llm_vocabulary_per_1k == 0


def test_vocabulary_hits_are_itemised():
    report = measure_voice(MACHINE)
    assert set(report.llm_vocabulary_counts) <= set(LLM_VOCABULARY)
    assert report.llm_vocabulary_counts["delve"] == 2
    assert report.llm_vocabulary_counts["crucial"] == 2


def test_word_boundaries_are_respected():
    assert measure_voice("crucially undelved").llm_vocabulary_counts == {}
    assert measure_voice("a crucial point").llm_vocabulary_counts == {"crucial": 1}


def test_markdown_counts_headings_bullets_and_bold():
    report = measure_voice("# One\n\n- a\n- b\n1. c\n\n**bold** and **more**")
    assert report.headings_per_1k > 0
    assert report.bullets_per_1k > 0
    assert report.bold_per_1k > 0
    assert report.markdown_density_per_1k == pytest.approx(
        report.headings_per_1k + report.bullets_per_1k + report.bold_per_1k
    )


def test_hedging_and_question_openers():
    report = measure_voice("Great question! Here is the answer.\n\nWhat about this?")
    assert report.hedging_openers == 1
    assert report.question_openers == 1


def test_opening_closing_overlap_detects_a_restated_conclusion():
    restated = "The bridge collapsed suddenly overnight.\n\nThe bridge collapsed suddenly overnight."
    varied = "The bridge collapsed overnight.\n\nNobody could explain what happened afterwards."
    assert opening_closing_overlap(restated) == pytest.approx(1.0)
    assert opening_closing_overlap(varied) < 0.3
    assert opening_closing_overlap("only one paragraph") == 0.0


def test_rates_are_per_thousand_tokens():
    once = measure_voice("delve " * 10 + "word " * 990)
    assert once.llm_vocabulary_per_1k == pytest.approx(10.0, abs=0.1)


def test_measure_files_reads_from_disk(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("we must delve into it")
    b.write_text("a plain sentence about bridges")
    report = measure_files([a, b])
    assert report.llm_vocabulary_counts == {"delve": 1}


def test_compare_and_format(tmp_path):
    baseline = measure_voice(HUMAN)
    path = save_baseline(baseline, tmp_path / "baseline.json")
    loaded = load_baseline(path)
    rows = compare(measure_voice(MACHINE), loaded)
    assert set(rows) == set(COMPARED)
    for row in rows.values():
        assert {"model", "human", "ratio"} == set(row)
    text = format_comparison(rows)
    assert "dashes_per_1k" in text and "ratio" in text


def test_compare_handles_a_zero_baseline():
    rows = compare(measure_voice(MACHINE), measure_voice("plain text here"))
    assert rows["llm_vocabulary_per_1k"]["ratio"] == float("inf")


@pytest.mark.skipif(not BASELINE.exists(), reason="baseline not built")
def test_saved_baseline_is_from_human_text():
    payload = json.loads(BASELINE.read_text())
    assert payload["tokens"] > 100_000
    assert payload["dashes_per_1k"] > 0
    assert payload["llm_vocabulary_per_1k"] < 1.0
