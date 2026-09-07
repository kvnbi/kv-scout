from __future__ import annotations

import json

import pytest
import torch

from kv_scout.train.ablation import (
    Variant,
    format_summary,
    steps_for_tokens,
    summarise,
)
from kv_scout.train.instrument import (
    RouterMonitor,
    balance_summary,
    collect_router_stats,
    router_stats_are_healthy,
)


def make_payload(rows):
    return {"tokens_per_run": 1000, "seeds": [1, 2], "results": rows}


def row(variant, seed, tail, grad=1.0, spikes=0):
    return {
        "variant": variant,
        "seed": seed,
        "steps": 10,
        "tokens": 1000,
        "final_loss": tail,
        "tail_loss": tail,
        "max_loss": tail + 1,
        "max_grad_norm": grad,
        "spikes": spikes,
        "wall_seconds": 1.0,
        "tokens_per_second": 1000.0,
        "router_stats": [],
    }


def test_steps_for_tokens():
    assert steps_for_tokens(250_000, 8, 256) == 122
    assert steps_for_tokens(100, 8, 256) == 1
    with pytest.raises(ValueError):
        steps_for_tokens(100, 0, 256)


def test_variant_defaults_are_independent():
    a, b = Variant("a"), Variant("b")
    a.overrides["x"] = 1
    assert b.overrides == {}


def test_summarise_reports_delta_against_the_baseline():
    payload = make_payload(
        [
            row("baseline", 1, 7.00),
            row("baseline", 2, 7.10),
            row("better", 1, 6.50),
            row("better", 2, 6.60),
        ]
    )
    summary = summarise(payload)
    assert summary["variants"]["baseline"]["delta"] == pytest.approx(0.0)
    assert summary["variants"]["better"]["delta"] == pytest.approx(-0.5)
    assert summary["variants"]["better"]["delta_percent"] == pytest.approx(
        -100 * 0.5 / 7.05
    )
    assert summary["variants"]["better"]["runs"] == 2


def test_summarise_flags_whether_a_delta_beats_seed_noise():
    noisy = make_payload(
        [row("baseline", 1, 7.0), row("baseline", 2, 7.4), row("tiny", 1, 7.15), row("tiny", 2, 7.25)]
    )
    assert summarise(noisy)["variants"]["tiny"]["beyond_noise"] is False

    clean = make_payload(
        [row("baseline", 1, 7.00), row("baseline", 2, 7.01), row("big", 1, 6.5), row("big", 2, 6.5)]
    )
    assert summarise(clean)["variants"]["big"]["beyond_noise"] is True


def test_summarise_requires_the_baseline_to_exist():
    with pytest.raises(ValueError):
        summarise(make_payload([row("other", 1, 7.0)]))


def test_single_seed_reports_unknown_significance():
    summary = summarise(make_payload([row("baseline", 1, 7.0), row("x", 1, 6.0)]))
    assert summary["noise"] == 0.0
    assert summary["variants"]["x"]["beyond_noise"] is None


def test_format_summary_is_printable():
    text = format_summary(
        summarise(make_payload([row("baseline", 1, 7.0), row("baseline", 2, 7.1)]))
    )
    assert "baseline" in text and "paired" in text


def test_balance_summary_on_a_uniform_router():
    stats = balance_summary(torch.ones(32) * 100)
    assert stats["max_over_mean"] == pytest.approx(1.0)
    assert stats["normalised_entropy"] == pytest.approx(1.0)
    assert stats["starved_experts"] == 0


def test_balance_summary_detects_collapse():
    counts = torch.zeros(32)
    counts[0] = 3200
    stats = balance_summary(counts)
    assert stats["max_over_mean"] == pytest.approx(32.0)
    assert stats["starved_experts"] == 31
    assert stats["normalised_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_balance_summary_on_empty_counts():
    stats = balance_summary(torch.zeros(8))
    assert stats["assignments"] == 0
    assert stats["starved_experts"] == 8


def test_router_monitor_accumulates_and_resets():
    monitor = RouterMonitor(4)
    monitor.record(torch.tensor([[0, 1], [1, 3]]))
    assert monitor.counts.tolist() == [1, 2, 0, 1]
    monitor.record(torch.tensor([2]))
    assert monitor.counts.tolist() == [1, 2, 1, 1]
    monitor.reset()
    assert monitor.counts.sum() == 0


def test_collect_router_stats_finds_modules_that_expose_counts():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.expert_counts = torch.ones(8) * 10

    model = torch.nn.Module()
    model.add_module("layer", Router())
    stats = collect_router_stats(model)
    assert len(stats) == 1
    assert stats[0]["module"] == "layer"
    assert router_stats_are_healthy(stats)


def test_collect_router_stats_is_empty_without_a_router():
    assert collect_router_stats(torch.nn.Linear(2, 2)) == []
    assert router_stats_are_healthy([]) is True


def test_unhealthy_router_is_reported():
    counts = torch.zeros(16)
    counts[0] = 100
    assert not router_stats_are_healthy([{**balance_summary(counts), "module": "r"}])


def test_paired_comparison_cancels_shared_seed_noise():
    payload = make_payload(
        [
            row("baseline", 1, 7.30),
            row("baseline", 2, 7.10),
            row("baseline", 3, 6.90),
            row("variant", 1, 7.20),
            row("variant", 2, 7.00),
            row("variant", 3, 6.80),
        ]
    )
    summary = summarise(payload)
    variant = summary["variants"]["variant"]

    assert variant["paired_delta"] == pytest.approx(-0.1)
    assert variant["paired_error"] == pytest.approx(0.0, abs=1e-9)
    assert summary["noise"] > 0.15
    assert variant["significant"] is True


def test_paired_error_is_smaller_than_unpaired_noise():
    payload = make_payload(
        [
            row("baseline", 1, 7.30),
            row("baseline", 2, 7.10),
            row("baseline", 3, 6.90),
            row("variant", 1, 7.24),
            row("variant", 2, 7.01),
            row("variant", 3, 6.83),
        ]
    )
    summary = summarise(payload)
    variant = summary["variants"]["variant"]
    assert variant["paired_error"] < summary["noise"]
    assert variant["significant"] is True
    assert variant["paired_delta"] < 0


def test_an_effect_inside_paired_noise_is_not_significant():
    payload = make_payload(
        [
            row("baseline", 1, 7.00),
            row("baseline", 2, 7.00),
            row("baseline", 3, 7.00),
            row("variant", 1, 7.05),
            row("variant", 2, 6.95),
            row("variant", 3, 7.01),
        ]
    )
    assert summarise(payload)["variants"]["variant"]["significant"] is False


def test_baseline_never_reports_significance():
    payload = make_payload([row("baseline", 1, 7.0), row("baseline", 2, 7.1)])
    assert summarise(payload)["variants"]["baseline"]["significant"] is None


def test_a_single_seed_cannot_establish_significance():
    payload = make_payload([row("baseline", 1, 7.0), row("variant", 1, 6.0)])
    assert summarise(payload)["variants"]["variant"]["significant"] is None


def test_an_identical_variant_is_not_significant():
    payload = make_payload(
        [
            row("baseline", 1, 7.0),
            row("baseline", 2, 7.2),
            row("variant", 1, 7.0),
            row("variant", 2, 7.2),
        ]
    )
    assert summarise(payload)["variants"]["variant"]["significant"] is False


def test_paired_ignores_seeds_the_baseline_did_not_run():
    payload = make_payload(
        [row("baseline", 1, 7.0), row("variant", 1, 6.5), row("variant", 9, 1.0)]
    )
    assert summarise(payload)["variants"]["variant"]["paired_delta"] == pytest.approx(-0.5)
