from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(scope="session")
def child_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SRC + (os.pathsep + existing if existing else "")
    return env


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    from kv_scout.data.synthetic import write_synthetic_corpus

    root = tmp_path_factory.mktemp("corpus")
    index = write_synthetic_corpus(
        root, n_tokens=600_000, vocab_size=2048, shard_tokens=150_000, seed=0
    )
    index.verify()
    return root / "index.json"


@pytest.fixture(scope="session")
def train_args():
    def build(data: Path, out: Path, steps: int, kill_at: int | None = None):
        argv = [
            sys.executable,
            "-m",
            "kv_scout.train.loop",
            "--data",
            str(data),
            "--out",
            str(out),
            "--steps",
            str(steps),
            "--checkpoint-every",
            "10",
            "--seq-len",
            "128",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--seed",
            "1234",
        ]
        if kill_at is not None:
            argv += ["--kill-at", str(kill_at)]
        return argv

    return build


@pytest.fixture(scope="session")
def run_child(child_env):
    def run(argv, expect_signal=None):
        result = subprocess.run(argv, env=child_env, capture_output=True, text=True)
        if expect_signal is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode == -expect_signal, (
                f"expected signal {expect_signal}, got returncode "
                f"{result.returncode}\n{result.stderr}"
            )
        return result

    return run
