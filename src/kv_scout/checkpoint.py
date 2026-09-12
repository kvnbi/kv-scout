from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_VERSION = 1
LATEST_NAME = "latest.json"


def capture_rng_state() -> dict:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict) -> None:
    if "python" in state:
        value = state["python"]
        if isinstance(value, list):
            value = (value[0], tuple(value[1]), value[2])
        random.setstate(value)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].to(torch.uint8).cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["torch_mps"].to(torch.uint8).cpu())


@dataclass
class Checkpoint:
    step: int
    model: dict
    optimizer: dict
    loader: dict
    rng: dict
    config: dict
    extra: dict


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def save(
    out_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_state: Any,
    config: dict,
    extra: dict | None = None,
    keep_last: int = 3,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:08d}.pt"
    target = out_dir / name
    tmp = out_dir / (name + ".tmp")

    payload = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader": loader_state.to_payload()
        if hasattr(loader_state, "to_payload")
        else dict(loader_state),
        "rng": capture_rng_state(),
        "config": config,
        "extra": extra or {},
    }

    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    _fsync_dir(out_dir)

    _atomic_write_text(
        out_dir / LATEST_NAME,
        json.dumps({"step": step, "file": name}, indent=2) + "\n",
    )
    _prune(out_dir, keep_last)
    return target


def _prune(out_dir: Path, keep_last: int) -> None:
    if keep_last < 1:
        return
    files = sorted(out_dir.glob("step_*.pt"))
    for stale in files[:-keep_last]:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass


def latest_path(out_dir: str | Path) -> Path | None:
    out_dir = Path(out_dir)
    pointer = out_dir / LATEST_NAME
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text())
        except json.JSONDecodeError:
            payload = None
        if payload:
            candidate = out_dir / payload["file"]
            if candidate.exists():
                return candidate
    files = sorted(out_dir.glob("step_*.pt"))
    return files[-1] if files else None


def load(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> Checkpoint:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return Checkpoint(
        step=int(payload["step"]),
        model=payload["model"],
        optimizer=payload["optimizer"],
        loader=payload["loader"],
        rng=payload["rng"],
        config=payload["config"],
        extra=payload.get("extra", {}),
    )


def resume(
    out_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    map_location: str | torch.device = "cpu",
) -> int:
    path = latest_path(out_dir)
    if path is None:
        return 0
    checkpoint = load(
        path, model=model, optimizer=optimizer, map_location=map_location
    )
    loader.load_state(checkpoint.loader)
    return checkpoint.step
