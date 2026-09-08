from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from kv_scout import checkpoint as ckpt
from kv_scout.config import (
    DataConfig,
    HarnessModelConfig,
    HarnessRunConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    proxy_config,
    to_dict,
)
from kv_scout.data.loader import ResumableTokenLoader
from kv_scout.model import KVScout, language_model_loss
from kv_scout.train.normuon import build_optimizer
from kv_scout.train.harness_model import HarnessGPT, loss_with_z
from kv_scout.train.schedule import wsd_lr

LOSS_LOG = "losses.jsonl"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_loss_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def truncate_loss_log(path: str | Path, last_step: int) -> None:
    path = Path(path)
    rows = [row for row in read_loss_log(path) if row["step"] <= last_step]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def build_model(model_cfg, dropout: float):
    if isinstance(model_cfg, ModelConfig):
        return KVScout(model_cfg), language_model_loss
    return HarnessGPT(model_cfg, dropout=dropout), loss_with_z


def train(
    out_dir: str | Path,
    data_index: str | Path,
    model_cfg: HarnessModelConfig | ModelConfig,
    optim_cfg: OptimConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    dropout: float = 0.1,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(train_cfg.device)
    dtype = getattr(torch, train_cfg.dtype)

    seed_everything(train_cfg.seed)

    model, loss_fn = build_model(model_cfg, dropout)
    model = model.to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, optim_cfg)
    loader = ResumableTokenLoader(
        index=data_index,
        seq_len=data_cfg.seq_len,
        batch_size=data_cfg.batch_size,
        seed=data_cfg.seed,
        shuffle=data_cfg.shuffle,
        drop_last=data_cfg.drop_last,
    )

    config_snapshot = {
        "model": to_dict(model_cfg),
        "optim": to_dict(optim_cfg),
        "data": to_dict(data_cfg),
        "train": to_dict(train_cfg),
        "dropout": dropout,
    }

    start_step = ckpt.resume(
        out_dir, model=model, optimizer=optimizer, loader=loader, map_location=device
    )
    log_path = out_dir / LOSS_LOG
    truncate_loss_log(log_path, start_step)

    if start_step > 0:
        print(f"resumed from step {start_step}", file=sys.stderr, flush=True)

    log = open(log_path, "a")
    model.train()

    for step in range(start_step + 1, train_cfg.steps + 1):
        lr = wsd_lr(step - 1, train_cfg.steps, optim_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

        inputs, targets = loader.next_batch(device)
        logits = model(inputs)
        total_loss, cross_entropy = loss_fn(
            logits, targets, optim_cfg.z_loss_weight
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optim_cfg.grad_clip
        )
        optimizer.step()

        if step % train_cfg.log_every == 0:
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(cross_entropy.detach()),
                        "total_loss": float(total_loss.detach()),
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "epoch": loader.epoch,
                        "position": loader.position,
                    }
                )
                + "\n"
            )
            log.flush()

        if step % train_cfg.checkpoint_every == 0 or step == train_cfg.steps:
            log.flush()
            os.fsync(log.fileno())
            ckpt.save(
                out_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                loader_state=loader.state(),
                config=config_snapshot,
                extra={"device": str(device)},
                keep_last=train_cfg.keep_last_checkpoints,
            )

        if train_cfg.kill_at is not None and step == train_cfg.kill_at:
            log.flush()
            os.fsync(log.fileno())
            print(f"killing process at step {step}", file=sys.stderr, flush=True)
            os.kill(os.getpid(), signal.SIGKILL)

    log.close()
    return train_cfg.steps


def main(argv: list[str] | None = None) -> int:
    defaults = HarnessRunConfig()
    parser = argparse.ArgumentParser(prog="kv-scout-train")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=defaults.train.steps)
    parser.add_argument(
        "--checkpoint-every", type=int, default=defaults.train.checkpoint_every
    )
    parser.add_argument("--seed", type=int, default=defaults.train.seed)
    parser.add_argument("--data-seed", type=int, default=defaults.data.seed)
    parser.add_argument("--device", default=defaults.train.device)
    parser.add_argument("--dtype", default=defaults.train.dtype)
    parser.add_argument("--seq-len", type=int, default=defaults.data.seq_len)
    parser.add_argument("--batch-size", type=int, default=defaults.data.batch_size)
    parser.add_argument("--vocab-size", type=int, default=defaults.model.vocab_size)
    parser.add_argument("--lr", type=float, default=defaults.optim.peak_lr)
    parser.add_argument(
        "--warmup-steps", type=int, default=defaults.optim.warmup_steps
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kill-at", type=int, default=None)
    parser.add_argument("--model", choices=("harness", "proxy"), default="harness")
    args = parser.parse_args(argv)

    if args.model == "proxy":
        model_cfg = proxy_config(context_max=args.seq_len, context_min=args.seq_len)
    else:
        model_cfg = replace(
            defaults.model, vocab_size=args.vocab_size, seq_len=args.seq_len
        )
    optim_cfg = replace(
        defaults.optim, peak_lr=args.lr, warmup_steps=args.warmup_steps
    )
    data_cfg = replace(
        defaults.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.data_seed,
    )
    train_cfg = replace(
        defaults.train,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        out_dir=str(args.out),
        kill_at=args.kill_at,
    )

    train(
        out_dir=args.out,
        data_index=args.data,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        dropout=args.dropout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
