from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")
EMA_NAME = "ema.pt"


def checkpoint_steps(out_dir: str | Path) -> list[tuple[int, Path]]:
    found = []
    for path in Path(out_dir).glob("step_*.pt"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def decay_phase_steps(
    steps: list[tuple[int, Path]], total_steps: int, decay_fraction: float
) -> list[tuple[int, Path]]:
    if not steps:
        return []
    first = total_steps - max(1, int(round(decay_fraction * total_steps)))
    inside = [row for row in steps if row[0] > first]
    return inside if inside else steps[-1:]


def ema_weights(steps: list[int], decay: float) -> list[float]:
    if not steps:
        raise ValueError("no checkpoints to weight")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must lie in (0, 1]")
    last = steps[-1]
    raw = [decay ** (last - step) for step in steps]
    total = sum(raw)
    return [value / total for value in raw]


def average_state_dicts(states: list[dict], weights: list[float]) -> dict:
    if len(states) != len(weights):
        raise ValueError("states and weights must be the same length")
    if not states:
        raise ValueError("nothing to average")

    accumulator = Accumulator()
    for state, weight in zip(states, weights):
        accumulator.add(state, weight)
    return accumulator.result()


class Accumulator:
    def __init__(self) -> None:
        self.running: dict[str, torch.Tensor] = {}
        self.dtypes: dict[str, torch.dtype] = {}
        self.integers: dict[str, torch.Tensor] = {}
        self.keys: set[str] | None = None

    def add(self, state: dict, weight: float) -> None:
        if self.keys is None:
            self.keys = set(state.keys())
        elif set(state.keys()) != self.keys:
            raise ValueError("checkpoints do not share the same parameters")

        for key, value in state.items():
            if not torch.is_floating_point(value):
                self.integers[key] = value.detach().clone()
                continue
            if key not in self.running:
                self.dtypes[key] = value.dtype
                self.running[key] = value.detach().to(torch.float64) * weight
                continue
            if value.shape != self.running[key].shape:
                raise ValueError(f"shape mismatch for {key}")
            self.running[key].add_(value.detach().to(torch.float64), alpha=weight)

    def result(self) -> dict:
        if self.keys is None:
            raise ValueError("nothing to average")
        merged = {
            key: value.to(self.dtypes[key]) for key, value in self.running.items()
        }
        merged.update(self.integers)
        return merged


def build_ema(
    out_dir: str | Path,
    decay: float = 0.999,
    total_steps: int | None = None,
    decay_fraction: float = 0.2,
    last_n: int | None = None,
    map_location: str = "cpu",
) -> dict:
    out_dir = Path(out_dir)
    found = checkpoint_steps(out_dir)
    if not found:
        raise FileNotFoundError(f"no checkpoints in {out_dir}")

    if last_n is not None:
        chosen = found[-last_n:]
    elif total_steps is not None:
        chosen = decay_phase_steps(found, total_steps, decay_fraction)
    else:
        chosen = found

    steps = [step for step, _ in chosen]
    weights = ema_weights(steps, decay)

    accumulator = Accumulator()
    latest = None
    for (step, path), weight in zip(chosen, weights):
        payload = torch.load(
            path, map_location=map_location, weights_only=False, mmap=True
        )
        accumulator.add(payload["model"], weight)
        if step == steps[-1]:
            latest = {
                "version": payload["version"],
                "step": payload["step"],
                "optimizer": payload["optimizer"],
                "loader": payload["loader"],
                "rng": payload["rng"],
                "config": payload["config"],
                "extra": dict(payload.get("extra", {})),
            }
        del payload

    output = {
        **latest,
        "model": accumulator.result(),
        "extra": {
            **latest["extra"],
            "ema_decay": decay,
            "ema_steps": steps,
            "ema_weights": weights,
        },
    }

    target = out_dir / EMA_NAME
    tmp = target.with_suffix(".pt.tmp")
    with open(tmp, "wb") as handle:
        torch.save(output, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)

    return {
        "path": str(target),
        "checkpoints": len(chosen),
        "steps": steps,
        "weights": weights,
        "decay": decay,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="kv-scout-ema")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--decay-fraction", type=float, default=0.2)
    parser.add_argument("--last-n", type=int, default=None)
    args = parser.parse_args(argv)

    report = build_ema(
        args.run,
        decay=args.decay,
        total_steps=args.total_steps,
        decay_fraction=args.decay_fraction,
        last_n=args.last_n,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
