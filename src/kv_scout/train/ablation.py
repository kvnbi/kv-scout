from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from kv_scout.config import (
    DataConfig,
    OptimConfig,
    TrainConfig,
    proxy_config,
)
from kv_scout.train.instrument import collect_router_stats
from kv_scout.train.loop import read_loss_log, train

TAIL_STEPS = 20


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict = field(default_factory=dict)
    optim_overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    variant: str
    seed: int
    steps: int
    tokens: int
    final_loss: float
    tail_loss: float
    max_loss: float
    max_grad_norm: float
    spikes: int
    wall_seconds: float
    tokens_per_second: float
    router_stats: list

    def to_payload(self) -> dict:
        return self.__dict__.copy()


def steps_for_tokens(tokens: int, batch_size: int, seq_len: int) -> int:
    per_step = batch_size * seq_len
    if per_step <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    return max(1, tokens // per_step)


def run_variant(
    variant: Variant,
    data_index: str | Path,
    out_root: str | Path,
    tokens: int,
    seed: int,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = "mps",
    dtype: str = "bfloat16",
    peak_lr: float = 3e-4,
    warmup_fraction: float = 0.1,
) -> RunResult:
    steps = steps_for_tokens(tokens, batch_size, seq_len)
    out_dir = Path(out_root) / f"{variant.name}_seed{seed}"

    model_cfg = proxy_config(
        context_max=seq_len, context_min=seq_len, **variant.overrides
    )
    optim_cfg = replace(
        OptimConfig(),
        peak_lr=peak_lr,
        warmup_steps=max(1, int(steps * warmup_fraction)),
        matrix_optimizer="adamw",
        **variant.optim_overrides,
    )
    data_cfg = DataConfig(seq_len=seq_len, batch_size=batch_size, seed=seed)
    train_cfg = TrainConfig(
        steps=steps,
        checkpoint_every=steps,
        keep_last_checkpoints=1,
        seed=seed,
        device=device,
        dtype=dtype,
        out_dir=str(out_dir),
    )

    started = time.time()
    train(
        out_dir=out_dir,
        data_index=data_index,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        dropout=0.0,
    )
    elapsed = time.time() - started

    rows = read_loss_log(out_dir / "losses.jsonl")
    losses = [r["loss"] for r in rows]
    grads = [r["grad_norm"] for r in rows]
    spikes = sum(1 for i in range(1, len(losses)) if losses[i] - losses[i - 1] > 1.0)

    return RunResult(
        variant=variant.name,
        seed=seed,
        steps=steps,
        tokens=steps * batch_size * seq_len,
        final_loss=losses[-1],
        tail_loss=statistics.mean(losses[-TAIL_STEPS:]),
        max_loss=max(losses),
        max_grad_norm=max(grads),
        spikes=spikes,
        wall_seconds=elapsed,
        tokens_per_second=steps * batch_size * seq_len / max(elapsed, 1e-6),
        router_stats=[],
    )


def run_ablation(
    variants: list[Variant],
    data_index: str | Path,
    out_root: str | Path,
    tokens: int = 250_000,
    seeds: tuple[int, ...] = (1, 2),
    **kwargs,
) -> dict:
    results = []
    for variant in variants:
        for seed in seeds:
            result = run_variant(
                variant, data_index, out_root, tokens, seed, **kwargs
            )
            results.append(result)
            print(
                f"{variant.name} seed {seed}: tail {result.tail_loss:.4f} "
                f"({result.wall_seconds:.0f}s)",
                flush=True,
            )

    payload = {
        "tokens_per_run": tokens,
        "seeds": list(seeds),
        "results": [r.to_payload() for r in results],
    }
    out = Path(out_root) / "ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def summarise(payload: dict, baseline: str = "baseline") -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in payload["results"]:
        grouped.setdefault(row["variant"], []).append(row)

    if baseline not in grouped:
        raise ValueError(f"no results for baseline variant {baseline}")

    base_values = [r["tail_loss"] for r in grouped[baseline]]
    base_mean = statistics.mean(base_values)
    noise = statistics.pstdev(base_values) if len(base_values) > 1 else 0.0

    base_by_seed = {r["seed"]: r["tail_loss"] for r in grouped[baseline]}

    table = {}
    for name, rows in grouped.items():
        values = [r["tail_loss"] for r in rows]
        mean = statistics.mean(values)
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        delta = mean - base_mean

        paired = [
            r["tail_loss"] - base_by_seed[r["seed"]]
            for r in rows
            if r["seed"] in base_by_seed
        ]
        paired_mean = statistics.mean(paired) if paired else 0.0
        paired_sd = statistics.pstdev(paired) if len(paired) > 1 else 0.0
        paired_error = paired_sd / math.sqrt(len(paired)) if len(paired) > 1 else 0.0
        if name == baseline or len(paired) < 2:
            significant = None
        elif paired_error > 0:
            significant = abs(paired_mean) > 2 * paired_error
        else:
            significant = paired_mean != 0.0

        table[name] = {
            "runs": len(rows),
            "tail_loss": mean,
            "spread": spread,
            "delta": delta,
            "delta_percent": 100.0 * delta / base_mean if base_mean else 0.0,
            "paired_delta": paired_mean,
            "paired_error": paired_error,
            "paired_percent": 100.0 * paired_mean / base_mean if base_mean else 0.0,
            "significant": significant,
            "beyond_noise": abs(delta) > 2 * noise if noise > 0 else None,
            "max_grad_norm": max(r["max_grad_norm"] for r in rows),
            "spikes": sum(r["spikes"] for r in rows),
            "tokens_per_second": statistics.mean(
                r["tokens_per_second"] for r in rows
            ),
        }
    return {"baseline": baseline, "noise": noise, "variants": table}


def format_summary(summary: dict) -> str:
    lines = [
        f"baseline {summary['baseline']}, unpaired seed noise {summary['noise']:.4f}",
        "",
        f"{'variant':22s} {'tail':>8s} {'paired':>9s} {'error':>8s} "
        f"{'pct':>8s} {'sig':>5s} {'spikes':>7s} {'tok/s':>8s}",
    ]
    for name, row in summary["variants"].items():
        flag = "" if row["significant"] is None else ("yes" if row["significant"] else "no")
        lines.append(
            f"{name:22s} {row['tail_loss']:>8.4f} {row['paired_delta']:>+9.4f} "
            f"{row['paired_error']:>8.4f} {row['paired_percent']:>+7.2f}% {flag:>5s} "
            f"{row['spikes']:>7d} {row['tokens_per_second']:>8.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="kv-scout-ablate")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=250_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    variants = [
        Variant("baseline"),
        Variant("narrow_ffn", {"dense_ffn_hidden": 768}),
    ]
    payload = run_ablation(
        variants,
        args.data,
        args.out,
        tokens=args.tokens,
        seeds=tuple(args.seeds),
        device=args.device,
        dtype=args.dtype,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print()
    print(format_summary(summarise(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
