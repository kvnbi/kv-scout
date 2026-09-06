from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

EM_DASH = re.compile(r"\s*[\u2014\u2015]\s*")
EN_DASH_RANGE = re.compile(r"(?<=\d)\s*\u2013\s*(?=\d)")
EN_DASH = re.compile(r"\s*\u2013\s*")


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str | None
    split: str
    text_field: str
    weight: float
    kind: str


DEFAULT_MIXTURE = (
    Source(
        name="english_web",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        weight=0.35,
        kind="english",
    ),
    Source(
        name="english_synthetic",
        dataset="HuggingFaceTB/smollm-corpus",
        config="cosmopedia-v2",
        split="train",
        text_field="text",
        weight=0.20,
        kind="english",
    ),
    Source(
        name="code",
        dataset="codeparrot/codeparrot-clean",
        config=None,
        split="train",
        text_field="content",
        weight=0.25,
        kind="code",
    ),
    Source(
        name="math",
        dataset="HuggingFaceTB/finemath",
        config="finemath-3plus",
        split="train",
        text_field="text",
        weight=0.15,
        kind="english",
    ),
    Source(
        name="german",
        dataset="HuggingFaceFW/fineweb-2",
        config="deu_Latn",
        split="train",
        text_field="text",
        weight=0.05,
        kind="german",
    ),
)


def normalise_dashes(text: str) -> str:
    text = EM_DASH.sub(", ", text)
    text = EN_DASH_RANGE.sub("-", text)
    return EN_DASH.sub(", ", text)


def sample_source(
    source: Source,
    out_path: Path,
    target_bytes: int,
    max_doc_bytes: int,
    normalise: bool,
    shuffle_buffer: int,
) -> dict:
    from datasets import load_dataset

    stream = load_dataset(
        source.dataset, source.config, split=source.split, streaming=True
    )
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=7, buffer_size=shuffle_buffer)

    written = 0
    docs = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in stream:
            text = record.get(source.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            if len(text) > max_doc_bytes:
                text = text[:max_doc_bytes]
            if normalise:
                text = normalise_dashes(text)
            handle.write(text)
            handle.write("\n")
            written += len(text.encode("utf-8")) + 1
            docs += 1
            if written >= target_bytes:
                break
        handle.flush()
        os.fsync(handle.fileno())

    return {
        "name": source.name,
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "text_field": source.text_field,
        "kind": source.kind,
        "weight": source.weight,
        "file": out_path.name,
        "docs": docs,
        "bytes": written,
    }


def run(
    out_dir: str | Path,
    total_bytes: int,
    mixture: tuple[Source, ...] = DEFAULT_MIXTURE,
    max_doc_bytes: int = 100_000,
    normalise: bool = True,
    shuffle_buffer: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_weight = sum(s.weight for s in mixture)
    if total_weight <= 0:
        raise ValueError("mixture weights must sum above zero")

    entries = []
    for source in mixture:
        target = int(total_bytes * source.weight / total_weight)
        path = out_dir / f"{source.name}.txt"
        print(f"sampling {source.name} to {target} bytes", file=sys.stderr, flush=True)
        entry = sample_source(
            source, path, target, max_doc_bytes, normalise, shuffle_buffer
        )
        print(
            f"  {entry['docs']} docs, {entry['bytes']} bytes",
            file=sys.stderr,
            flush=True,
        )
        entries.append(entry)

    manifest = {
        "version": 1,
        "requested_bytes": total_bytes,
        "total_bytes": sum(e["bytes"] for e in entries),
        "total_docs": sum(e["docs"] for e in entries),
        "normalised_dashes": normalise,
        "max_doc_bytes": max_doc_bytes,
        "shuffle_buffer": shuffle_buffer,
        "sources": entries,
    }
    path = out_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kv-scout-sample")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--total-bytes", type=int, default=500_000_000)
    parser.add_argument("--max-doc-bytes", type=int, default=100_000)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--keep-dashes", action="store_true")
    args = parser.parse_args(argv)

    manifest = run(
        out_dir=args.out,
        total_bytes=args.total_bytes,
        max_doc_bytes=args.max_doc_bytes,
        normalise=not args.keep_dashes,
        shuffle_buffer=args.shuffle_buffer,
    )
    print(
        f"wrote {manifest['total_bytes']} bytes across "
        f"{len(manifest['sources'])} sources",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
