from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from kv_scout.config import (
    DataConfig,
    OptimConfig,
    TrainConfig,
    proxy_config,
)
from kv_scout.train.ema import EMA_NAME, build_ema, checkpoint_steps
from kv_scout.train.loop import (
    build_model,
    evaluate_loss,
    read_loss_log,
    resolve_device,
    train,
)

TAIL_STEPS = 20
RESULT_NAME = "result.json"
EMA_CHECKPOINTS = 5


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict = field(default_factory=dict)
    optim_overrides: dict = field(default_factory=dict)
    freeze: tuple[str, ...] = ()
    ema: bool = False


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
    heldout: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return self.__dict__.copy()


def stage_three_variants(seq_len: int) -> list[Variant]:
    window = max(1, seq_len // 4)
    return [
        Variant("baseline", ema=True),
        Variant("qk_norm", {"qk_norm": True}),
        Variant("nope_anchors", {"nope_on_anchor_layers": True}),
        Variant(
            "window_no_sinks",
            {"attention_sinks": True, "sink_tokens": 0, "attention_window": window},
        ),
        Variant(
            "window_three_sinks",
            {"attention_sinks": True, "sink_tokens": 3, "attention_window": window},
        ),
        Variant("gated_attention", {"per_head_gated_attention": True}),
        Variant(
            "gated_attention_frozen",
            {"per_head_gated_attention": True},
            freeze=("head_gate",),
        ),
        Variant("kv_sharing", {"cross_layer_kv_sharing": True}),
    ]


def load_result(out_dir: str | Path) -> RunResult | None:
    path = Path(out_dir) / RESULT_NAME
    if not path.exists():
        return None
    return RunResult(**json.loads(path.read_text()))


def _evaluate(model, loss_fn, index, seq_len, batch_size, device, dtype, tokens):
    return evaluate_loss(
        model, loss_fn, index, seq_len, batch_size, device, dtype, tokens
    )


def _load_weights(model, path: Path, device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])


def _remove_checkpoints(out_dir: Path) -> None:
    for _, path in checkpoint_steps(out_dir):
        path.unlink(missing_ok=True)
    (out_dir / EMA_NAME).unlink(missing_ok=True)


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
    heldout_index: str | Path | None = None,
    eval_tokens: int = 2_000_000,
    keep_checkpoints: bool = False,
) -> RunResult:
    steps = steps_for_tokens(tokens, batch_size, seq_len)
    out_dir = Path(out_root) / f"{variant.name}_seed{seed}"
    finished = load_result(out_dir)
    if finished is not None:
        return finished

    model_cfg = proxy_config(
        context_max=2 * seq_len, context_min=seq_len, **variant.overrides
    )
    optim_fields = {
        "peak_lr": peak_lr,
        "warmup_steps": max(1, int(steps * warmup_fraction)),
        "matrix_optimizer": "adamw",
    }
    optim_fields.update(variant.optim_overrides)
    optim_cfg = replace(OptimConfig(), **optim_fields)
    data_cfg = DataConfig(seq_len=seq_len, batch_size=batch_size, seed=seed)
    decay_steps = max(1, int(round(optim_cfg.decay_fraction * steps)))
    checkpoint_every = (
        max(1, decay_steps // EMA_CHECKPOINTS) if variant.ema else steps
    )
    train_cfg = TrainConfig(
        steps=steps,
        checkpoint_every=checkpoint_every,
        keep_last_checkpoints=EMA_CHECKPOINTS + 2 if variant.ema else 1,
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
        freeze=variant.freeze,
    )
    elapsed = time.time() - started

    rows = read_loss_log(out_dir / "losses.jsonl")
    losses = [r["loss"] for r in rows]
    grads = [r["grad_norm"] for r in rows]
    spikes = sum(1 for i in range(1, len(losses)) if losses[i] - losses[i - 1] > 1.0)

    heldout: dict[str, float] = {}
    if heldout_index is not None:
        resolved = resolve_device(device)
        model, loss_fn = build_model(model_cfg, 0.0)
        model = model.to(resolved)
        latest = checkpoint_steps(out_dir)[-1][1]
        _load_weights(model, latest, resolved)
        for length in (seq_len, 2 * seq_len):
            rows_per_batch = max(1, batch_size * seq_len // length)
            heldout[f"heldout_{length}"] = _evaluate(
                model, loss_fn, heldout_index, length, rows_per_batch,
                resolved, dtype, eval_tokens,
            )
        if variant.ema:
            build_ema(
                out_dir,
                decay=optim_cfg.ema_decay,
                total_steps=steps,
                decay_fraction=optim_cfg.decay_fraction,
                map_location=str(resolved),
            )
            _load_weights(model, out_dir / EMA_NAME, resolved)
            heldout[f"ema_heldout_{seq_len}"] = _evaluate(
                model, loss_fn, heldout_index, seq_len, batch_size,
                resolved, dtype, eval_tokens,
            )
        del model
    if not keep_checkpoints:
        _remove_checkpoints(out_dir)

    result = RunResult(
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
        heldout=heldout,
    )
    (out_dir / RESULT_NAME).write_text(json.dumps(result.to_payload(), indent=2))
    return result


def run_ablation(
    variants: list[Variant],
    data_index: str | Path,
    out_root: str | Path,
    tokens: int = 250_000,
    seeds: tuple[int, ...] = (1, 2),
    **kwargs,
) -> dict:
    results = []
    for seed in seeds:
        for variant in variants:
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


def primary_metric(payload: dict) -> str:
    rows = payload["results"]
    keys = [k for k in rows[0].get("heldout", {}) if k.startswith("heldout_")] if rows else []
    if keys and all(k in r.get("heldout", {}) for r in rows for k in keys[:1]):
        return keys[0]
    return "tail_loss"


def metric_value(row: dict, metric: str) -> float:
    if metric == "tail_loss":
        return row["tail_loss"]
    return row["heldout"][metric]


def _mean_of(rows: list[dict], key: str) -> float | None:
    values = [r["heldout"][key] for r in rows if key in r.get("heldout", {})]
    return statistics.mean(values) if values else None


def summarise(payload: dict, baseline: str = "baseline", metric: str | None = None) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in payload["results"]:
        grouped.setdefault(row["variant"], []).append(row)

    if baseline not in grouped:
        raise ValueError(f"no results for baseline variant {baseline}")
    metric = metric or primary_metric(payload)

    base_values = [metric_value(r, metric) for r in grouped[baseline]]
    base_mean = statistics.mean(base_values)
    noise = statistics.pstdev(base_values) if len(base_values) > 1 else 0.0

    base_by_seed = {r["seed"]: metric_value(r, metric) for r in grouped[baseline]}
    long_key = next(
        (k for k in grouped[baseline][0].get("heldout", {}) if k.startswith("heldout_") and k != metric),
        None,
    )
    ema_key = next(
        (k for k in grouped[baseline][0].get("heldout", {}) if k.startswith("ema_")),
        None,
    )

    table = {}
    for name, rows in grouped.items():
        values = [metric_value(r, metric) for r in rows]
        mean = statistics.mean(values)
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        delta = mean - base_mean

        paired = [
            metric_value(r, metric) - base_by_seed[r["seed"]]
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

        long_loss = _mean_of(rows, long_key) if long_key else None
        ema_loss = _mean_of(rows, ema_key) if ema_key else None
        table[name] = {
            "runs": len(rows),
            "loss": mean,
            "tail_loss": statistics.mean(r["tail_loss"] for r in rows),
            "long_loss": long_loss,
            "long_delta": (long_loss - mean) if long_loss is not None else None,
            "ema_delta": (ema_loss - mean) if ema_loss is not None else None,
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
    return {"baseline": baseline, "metric": metric, "noise": noise, "variants": table}


def format_summary(summary: dict) -> str:
    lines = [
        f"baseline {summary['baseline']}, metric {summary['metric']}, "
        f"unpaired seed noise {summary['noise']:.4f}",
        "",
        f"{'variant':22s} {'loss':>8s} {'paired':>9s} {'error':>8s} "
        f"{'pct':>8s} {'sig':>5s} {'long':>8s} {'ema':>8s} {'spikes':>7s} {'tok/s':>8s}",
    ]
    for name, row in summary["variants"].items():
        flag = "" if row["significant"] is None else ("yes" if row["significant"] else "no")
        long_delta = "" if row["long_delta"] is None else f"{row['long_delta']:+.4f}"
        ema_delta = "" if row["ema_delta"] is None else f"{row['ema_delta']:+.4f}"
        lines.append(
            f"{name:22s} {row['loss']:>8.4f} {row['paired_delta']:>+9.4f} "
            f"{row['paired_error']:>8.4f} {row['paired_percent']:>+7.2f}% {flag:>5s} "
            f"{long_delta:>8s} {ema_delta:>8s} "
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
    parser.add_argument("--grid", choices=("smoke", "stage3"), default="smoke")
    parser.add_argument("--heldout", type=Path, default=None)
    parser.add_argument("--eval-tokens", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    if args.grid == "stage3":
        variants = stage_three_variants(args.seq_len)
    else:
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
        heldout_index=args.heldout,
        eval_tokens=args.eval_tokens,
    )
    print()
    print(format_summary(summarise(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
