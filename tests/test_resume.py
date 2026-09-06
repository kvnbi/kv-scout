from __future__ import annotations

import json
import signal
import statistics
from pathlib import Path

import pytest

from kv_scout import checkpoint as ckpt
from kv_scout.train.loop import read_loss_log

STEPS = 200
KILL_AT = 100
CHECKPOINT_EVERY = 10
PRE_WINDOW = 25
POST_WINDOW = 5
SEAM_TOLERANCE = 1.5


@pytest.fixture(scope="module")
def reference_curve(tmp_path_factory, corpus, train_args, run_child):
    out = tmp_path_factory.mktemp("reference")
    run_child(train_args(corpus, out, STEPS))
    return [row["loss"] for row in read_loss_log(out / "losses.jsonl")]


@pytest.fixture(scope="module")
def interrupted_run(tmp_path_factory, corpus, train_args, run_child):
    out = tmp_path_factory.mktemp("interrupted")

    first = run_child(
        train_args(corpus, out, STEPS, kill_at=KILL_AT), expect_signal=signal.SIGKILL
    )
    before = read_loss_log(out / "losses.jsonl")
    latest = ckpt.latest_path(out)
    assert latest is not None
    checkpoint_step = json.loads((out / "latest.json").read_text())["step"]

    second = run_child(train_args(corpus, out, STEPS))
    after = read_loss_log(out / "losses.jsonl")

    return {
        "out": out,
        "first_stderr": first.stderr,
        "second_stderr": second.stderr,
        "before": before,
        "after": after,
        "checkpoint_step": checkpoint_step,
    }


def test_process_was_actually_killed(interrupted_run):
    assert "killing process at step 100" in interrupted_run["first_stderr"]
    steps = [row["step"] for row in interrupted_run["before"]]
    assert steps == list(range(1, KILL_AT + 1))


def test_resumed_from_last_checkpoint(interrupted_run):
    assert interrupted_run["checkpoint_step"] == KILL_AT
    assert f"resumed from step {KILL_AT}" in interrupted_run["second_stderr"]


def test_curve_is_complete_and_ordered(interrupted_run):
    steps = [row["step"] for row in interrupted_run["after"]]
    assert steps == list(range(1, STEPS + 1))


def test_loader_position_continues_across_the_seam(interrupted_run):
    rows = {row["step"]: row for row in interrupted_run["after"]}
    before = rows[KILL_AT]
    after = rows[KILL_AT + 1]
    assert after["epoch"] == before["epoch"]
    assert after["position"] == before["position"] + 8


def test_loss_curve_continues_smoothly(interrupted_run):
    losses = [row["loss"] for row in interrupted_run["after"]]
    assert len(losses) == STEPS

    deltas = [abs(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
    seam = KILL_AT - 1
    before = deltas[seam - PRE_WINDOW : seam]
    across = deltas[seam : seam + POST_WINDOW]

    allowed = max(before) * SEAM_TOLERANCE
    worst = max(across)
    assert worst <= allowed, (
        f"loss moved {worst:.4f} in a single step across the resume seam, "
        f"against a largest pre-interruption step change of {max(before):.4f} "
        f"over the {PRE_WINDOW} steps before it"
    )

    boundary = deltas[seam]
    assert boundary <= max(3.0 * statistics.median(before), 1e-6), (
        f"loss jumped {boundary:.4f} between the last step before the kill and "
        f"the first step after the resume"
    )


def test_curve_actually_descends(interrupted_run):
    losses = [row["loss"] for row in interrupted_run["after"]]
    first = statistics.mean(losses[:10])
    last = statistics.mean(losses[-10:])
    assert last < first - 0.5, (
        "the harness did not learn, so the smoothness assertion is vacuous"
    )


def test_resumed_curve_matches_an_uninterrupted_run(
    interrupted_run, reference_curve
):
    resumed = [row["loss"] for row in interrupted_run["after"]]
    assert len(reference_curve) == STEPS
    diffs = [abs(a - b) for a, b in zip(resumed, reference_curve)]
    worst = max(diffs)
    worst_after_seam = max(diffs[KILL_AT:])
    assert worst < 1e-4, (
        f"resumed run diverges from the uninterrupted run by {worst:.3e} "
        f"(post seam {worst_after_seam:.3e})"
    )
