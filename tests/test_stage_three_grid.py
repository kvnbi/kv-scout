from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kv_scout.config import proxy_config
from kv_scout.model import KVScout, language_model_loss
from kv_scout.train.ablation import (
    RESULT_NAME,
    Variant,
    load_result,
    primary_metric,
    run_ablation,
    run_variant,
    stage_three_variants,
    summarise,
)
from kv_scout.train.ema import checkpoint_steps
from kv_scout.train.loop import evaluate_loss, freeze_parameters
from kv_scout.train.normuon import build_optimizer
from kv_scout.config import OptimConfig

TINY = dict(
    d_model=64, n_layers=4, n_query_heads=2, n_kv_heads=1, head_dim=32,
    dense_ffn_hidden=128, attention_anchor_layers=(3, 4),
)


def tiny_run(variant, corpus, out, **extra):
    settings = dict(
        tokens=512, seed=1, batch_size=2, seq_len=32, device="cpu", dtype="float32",
        heldout_index=corpus, eval_tokens=256,
    )
    settings.update(extra)
    tuned = Variant(
        variant.name, {**TINY, **variant.overrides}, variant.optim_overrides,
        variant.freeze, variant.ema,
    )
    return run_variant(tuned, corpus, out, **settings)


def test_freeze_matches_by_substring_and_optimizer_skips_them():
    cfg = proxy_config(per_head_gated_attention=True, **TINY)
    model = KVScout(cfg)
    frozen = freeze_parameters(model, ("head_gate",))
    assert frozen
    assert all("head_gate" in name for name in frozen)
    assert all(not p.requires_grad for n, p in model.named_parameters() if "head_gate" in n)
    optimizer = build_optimizer(model, OptimConfig(matrix_optimizer="adamw"))
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, param in model.named_parameters():
        assert (id(param) in held) is param.requires_grad


def test_evaluate_loss_is_deterministic_and_restores_training_mode(corpus):
    cfg = proxy_config(**TINY)
    torch.manual_seed(0)
    model = KVScout(cfg).train()
    device = torch.device("cpu")
    a = evaluate_loss(model, language_model_loss, corpus, 32, 2, device, "float32", 256)
    b = evaluate_loss(model, language_model_loss, corpus, 32, 2, device, "float32", 256)
    assert a == b
    assert 0.0 < a < 20.0
    assert model.training


def test_a_run_writes_a_result_and_is_not_repeated(corpus, tmp_path):
    first = tiny_run(Variant("baseline"), corpus, tmp_path)
    out = tmp_path / "baseline_seed1"
    assert (out / RESULT_NAME).exists()
    assert set(first.heldout) == {"heldout_32", "heldout_64"}
    assert first.wall_seconds > 0
    stamp = (out / RESULT_NAME).stat().st_mtime_ns
    again = tiny_run(Variant("baseline"), corpus, tmp_path)
    assert again == first
    assert (out / RESULT_NAME).stat().st_mtime_ns == stamp
    assert load_result(out) == first


def test_checkpoints_are_removed_after_evaluation(corpus, tmp_path):
    tiny_run(Variant("baseline"), corpus, tmp_path)
    assert checkpoint_steps(tmp_path / "baseline_seed1") == []
    tiny_run(Variant("kept"), corpus, tmp_path, keep_checkpoints=True)
    assert checkpoint_steps(tmp_path / "kept_seed1")


def test_the_frozen_gate_control_never_moves_its_gate(corpus, tmp_path):
    torch.manual_seed(0)
    frozen = Variant(
        "gated_attention_frozen", {"per_head_gated_attention": True}, freeze=("head_gate",)
    )
    tiny_run(frozen, corpus, tmp_path, keep_checkpoints=True)
    latest = checkpoint_steps(tmp_path / "gated_attention_frozen_seed1")[-1][1]
    state = torch.load(latest, map_location="cpu", weights_only=False)["model"]
    gates = {k: v for k, v in state.items() if "head_gate" in k}
    assert gates
    for name, value in gates.items():
        if name.endswith("weight"):
            assert torch.equal(value, torch.zeros_like(value))
        else:
            assert torch.equal(value, torch.full_like(value, 3.0))


def test_a_trained_gate_does_move(corpus, tmp_path):
    torch.manual_seed(0)
    live = Variant("gated_attention", {"per_head_gated_attention": True})
    tiny_run(live, corpus, tmp_path, keep_checkpoints=True)
    latest = checkpoint_steps(tmp_path / "gated_attention_seed1")[-1][1]
    state = torch.load(latest, map_location="cpu", weights_only=False)["model"]
    weights = [v for k, v in state.items() if "head_gate" in k and k.endswith("weight")]
    assert any(not torch.equal(w, torch.zeros_like(w)) for w in weights)


def test_the_ema_variant_reports_an_ema_loss_and_cleans_up(corpus, tmp_path):
    result = tiny_run(Variant("baseline", ema=True), corpus, tmp_path, tokens=2048)
    assert "ema_heldout_32" in result.heldout
    assert 0.0 < result.heldout["ema_heldout_32"] < 20.0
    out = tmp_path / "baseline_seed1"
    assert checkpoint_steps(out) == []
    assert not (out / "ema.pt").exists()


def test_the_grid_covers_every_open_claim():
    names = {v.name for v in stage_three_variants(1024)}
    assert names == {
        "baseline", "qk_norm", "nope_anchors", "window_no_sinks",
        "window_three_sinks", "gated_attention", "gated_attention_frozen", "kv_sharing",
    }
    by_name = {v.name: v for v in stage_three_variants(1024)}
    assert by_name["baseline"].ema is True
    assert by_name["gated_attention_frozen"].freeze == ("head_gate",)
    assert by_name["window_three_sinks"].overrides["attention_window"] == 256
    assert by_name["window_no_sinks"].overrides["sink_tokens"] == 0
    for variant in stage_three_variants(1024):
        proxy_config(context_max=2048, context_min=1024, **variant.overrides)


def test_summary_prefers_the_heldout_metric_when_present():
    def row(variant, seed, tail, held, long):
        return {
            "variant": variant, "seed": seed, "steps": 10, "tokens": 1000,
            "final_loss": tail, "tail_loss": tail, "max_loss": tail + 1,
            "max_grad_norm": 1.0, "spikes": 0, "wall_seconds": 1.0,
            "tokens_per_second": 1000.0, "router_stats": [],
            "heldout": {"heldout_32": held, "heldout_64": long},
        }
    payload = {"tokens_per_run": 1000, "seeds": [1, 2], "results": [
        row("baseline", 1, 5.0, 4.0, 4.5), row("baseline", 2, 5.1, 4.1, 4.6),
        row("x", 1, 5.0, 3.5, 4.5), row("x", 2, 5.1, 3.6, 4.6),
    ]}
    assert primary_metric(payload) == "heldout_32"
    summary = summarise(payload)
    assert summary["metric"] == "heldout_32"
    assert summary["variants"]["x"]["paired_delta"] == pytest.approx(-0.5)
    assert summary["variants"]["x"]["long_delta"] == pytest.approx(1.0)
    assert summary["variants"]["baseline"]["ema_delta"] is None
    bare = {"tokens_per_run": 1000, "seeds": [1], "results": [
        {**row("baseline", 1, 5.0, 4.0, 4.5), "heldout": {}},
    ]}
    assert primary_metric(bare) == "tail_loss"


def test_the_grid_runs_end_to_end_and_pairs_by_seed(corpus, tmp_path):
    variants = [
        Variant("baseline", TINY, ema=True),
        Variant("qk_norm", {**TINY, "qk_norm": True}),
    ]
    payload = run_ablation(
        variants, corpus, tmp_path, tokens=512, seeds=(1, 2), batch_size=2,
        seq_len=32, device="cpu", dtype="float32", heldout_index=corpus, eval_tokens=256,
    )
    assert len(payload["results"]) == 4
    order = [(r["seed"], r["variant"]) for r in payload["results"]]
    assert order == [(1, "baseline"), (1, "qk_norm"), (2, "baseline"), (2, "qk_norm")]
    summary = summarise(payload)
    assert summary["metric"] == "heldout_32"
    assert summary["variants"]["qk_norm"]["runs"] == 2
    assert summary["variants"]["baseline"]["ema_delta"] is not None
    assert (tmp_path / "ablation.json").exists()
