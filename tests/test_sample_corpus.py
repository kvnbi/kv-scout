from __future__ import annotations

import json
import sys
import types

import pytest

from kv_scout.data.sample_corpus import (
    DEFAULT_MIXTURE,
    Source,
    normalise_dashes,
    run,
)

MIXTURE = (
    Source("big", "fake/big", None, "train", "text", 0.75, "english"),
    Source("small", "fake/small", "cfg", "train", "body", 0.25, "german"),
)


@pytest.fixture
def stub_datasets(monkeypatch):
    def load_dataset(name, config=None, split=None, streaming=False):
        field = "text" if name == "fake/big" else "body"
        return ({field: f"{name} document {i} filler text\n"} for i in range(10_000))

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_default_mixture_weights_sum_to_one():
    assert sum(s.weight for s in DEFAULT_MIXTURE) == pytest.approx(1.0)
    kinds = {s.kind for s in DEFAULT_MIXTURE}
    assert kinds == {"english", "code", "german"}


def test_normalise_dashes():
    assert normalise_dashes("a \u2014 b") == "a, b"
    assert normalise_dashes("1914\u20131918") == "1914-1918"
    assert normalise_dashes("x \u2013 y") == "x, y"
    assert normalise_dashes("plain-hyphen stays") == "plain-hyphen stays"


def test_run_splits_bytes_by_weight(tmp_path, stub_datasets):
    manifest = run(tmp_path, total_bytes=20_000, mixture=MIXTURE)

    by_name = {e["name"]: e for e in manifest["sources"]}
    assert by_name["big"]["bytes"] >= 15_000
    assert by_name["small"]["bytes"] >= 5_000
    assert by_name["big"]["bytes"] > by_name["small"]["bytes"] * 2

    assert (tmp_path / "big.txt").exists()
    assert (tmp_path / "small.txt").exists()
    assert "fake/small document 0" in (tmp_path / "small.txt").read_text()

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved == manifest
    assert saved["total_bytes"] == sum(e["bytes"] for e in saved["sources"])
    assert saved["normalised_dashes"] is True


def test_run_truncates_long_documents(tmp_path, stub_datasets):
    run(tmp_path, total_bytes=2_000, mixture=MIXTURE, max_doc_bytes=10)
    for line in (tmp_path / "big.txt").read_text().splitlines():
        assert len(line) <= 10


def test_run_rejects_empty_weights(tmp_path, stub_datasets):
    empty = (Source("a", "fake/big", None, "train", "text", 0.0, "english"),)
    with pytest.raises(ValueError):
        run(tmp_path, total_bytes=100, mixture=empty)
