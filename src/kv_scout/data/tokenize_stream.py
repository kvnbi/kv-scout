from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from kv_scout.data.normalise import normalise_dashes
from kv_scout.data.shards import ShardWriter


def _tokenizer_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_field(record: dict, field: str) -> str:
    value = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return ""
        value = value[part]
    return value if isinstance(value, str) else ""


def _batches(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(
    dataset: str,
    tokenizer_path: Path,
    out_dir: Path,
    split: str = "train",
    config_name: str | None = None,
    text_field: str = "text",
    eos_token: str = "<|endoftext|>",
    max_tokens: int | None = None,
    shard_tokens: int = 100_000_000,
    batch_docs: int = 1000,
    log_every: int = 100_000_000,
    normalise: bool = True,
) -> int:
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError("tokenizer vocabulary does not fit in uint16 shards")

    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"tokenizer has no entry for {eos_token}")

    stream = load_dataset(
        dataset, config_name, split=split, streaming=True
    )

    writer = ShardWriter(
        root=out_dir,
        shard_tokens=shard_tokens,
        vocab_size=vocab_size,
        eos_id=eos_id,
        tokenizer_sha256=_tokenizer_digest(tokenizer_path),
        sources=[
            {
                "dataset": dataset,
                "config": config_name,
                "split": split,
                "text_field": text_field,
                "normalised_dashes": normalise,
            }
        ],
    )

    started = time.time()
    next_log = log_every
    docs = 0
    stop = False

    with writer:
        for batch in _batches(stream, batch_docs):
            texts = [_resolve_field(record, text_field) for record in batch]
            texts = [text for text in texts if text]
            if normalise:
                texts = [normalise_dashes(text) for text in texts]
            if not texts:
                continue
            for encoding in tokenizer.encode_batch(texts):
                ids = encoding.ids
                if not ids:
                    continue
                writer.add(ids + [eos_id])
                docs += 1
                if max_tokens is not None and writer.tokens_written >= max_tokens:
                    stop = True
                    break
            if writer.tokens_written >= next_log:
                elapsed = time.time() - started
                rate = writer.tokens_written / max(elapsed, 1e-6)
                print(
                    f"{writer.tokens_written} tokens, {docs} docs, "
                    f"{rate:,.0f} tokens/s",
                    file=sys.stderr,
                    flush=True,
                )
                next_log += log_every
            if stop:
                break

    index_path = Path(out_dir) / "index.json"
    payload = json.loads(index_path.read_text())
    print(
        f"wrote {payload['total_tokens']} tokens across "
        f"{len(payload['shards'])} shards to {index_path}",
        file=sys.stderr,
        flush=True,
    )
    return int(payload["total_tokens"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kv-scout-tokenize")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--batch-docs", type=int, default=1000)
    parser.add_argument("--keep-dashes", action="store_true")
    args = parser.parse_args(argv)

    run(
        dataset=args.dataset,
        tokenizer_path=args.tokenizer,
        out_dir=args.out,
        split=args.split,
        config_name=args.config_name,
        text_field=args.text_field,
        eos_token=args.eos_token,
        max_tokens=args.max_tokens,
        shard_tokens=args.shard_tokens,
        batch_docs=args.batch_docs,
        normalise=not args.keep_dashes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
