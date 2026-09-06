from __future__ import annotations

from pathlib import Path

import pytest

from kv_scout.tokenizer import train
from kv_scout.tokenizer_metrics import (
    BRITISH_AMERICAN_PAIRS,
    compression,
    format_report,
    measure,
    measure_british,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("metrics")
    corpus = root / "corpus.txt"
    corpus.write_text(
        ("the colour of the centre and the defence of it\n" * 400)
        + ("def compute(self, x):\n    return x + 1\n" * 400)
    )
    heldout = root / "heldout"
    heldout.mkdir()
    (heldout / "prose.txt").write_text("the colour of the centre\n" * 100)
    (heldout / "code.txt").write_text("def compute(self, x):\n" * 100)
    out = root / "tok" / "tokenizer.json"
    train([corpus], out, vocab_size=1024, min_frequency=1)
    return out, heldout


def test_compression_reports_ratios(trained):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained[0]))
    stats = compression(tok, "the colour of the centre")
    assert stats["tokens"] > 0
    assert stats["chars_per_token"] == stats["chars"] / stats["tokens"]
    assert stats["bytes_per_token"] > 0


def test_compression_handles_empty_text(trained):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained[0]))
    assert compression(tok, "")["tokens"] == 0


def test_measure_covers_every_heldout_file(trained):
    report = measure(*trained)
    assert set(report["sources"]) == {"prose", "code"}
    assert report["overall_chars_per_token"] > 1.0
    assert report["vocab_size"] == 1024


def test_british_table_is_complete(trained):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained[0]))
    table = measure_british(tok)
    assert len(table["pairs"]) == len(BRITISH_AMERICAN_PAIRS)
    assert 0 <= table["worse_than_american"] <= len(BRITISH_AMERICAN_PAIRS)


def test_format_report_is_printable(trained):
    text = format_report(measure(*trained))
    assert "chars/token" in text
    assert "british forms cost more" in text
