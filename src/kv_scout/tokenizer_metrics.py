from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BRITISH_AMERICAN_PAIRS = (
    ("colour", "color"),
    ("colours", "colors"),
    ("realise", "realize"),
    ("realised", "realized"),
    ("centre", "center"),
    ("centres", "centers"),
    ("defence", "defense"),
    ("travelled", "traveled"),
    ("analyse", "analyze"),
    ("analysed", "analyzed"),
    ("organise", "organize"),
    ("behaviour", "behavior"),
    ("favourite", "favorite"),
    ("neighbour", "neighbor"),
    ("labour", "labor"),
    ("licence", "license"),
    ("practise", "practice"),
    ("programme", "program"),
    ("catalogue", "catalog"),
    ("apologise", "apologize"),
)


def compression(tokenizer, text: str) -> dict:
    ids = tokenizer.encode(text).ids
    if not ids:
        return {"chars": len(text), "bytes": 0, "tokens": 0}
    return {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "tokens": len(ids),
        "chars_per_token": len(text) / len(ids),
        "bytes_per_token": len(text.encode("utf-8")) / len(ids),
    }


def measure_sources(tokenizer, heldout_dir: Path, limit_bytes: int = 5_000_000) -> dict:
    results = {}
    for path in sorted(heldout_dir.glob("*.txt")):
        text = path.read_text(errors="ignore")[:limit_bytes]
        results[path.stem] = compression(tokenizer, text)
    return results


def measure_british(tokenizer) -> dict:
    rows = []
    worse = 0
    for british, american in BRITISH_AMERICAN_PAIRS:
        nb = len(tokenizer.encode(" " + british).ids)
        na = len(tokenizer.encode(" " + american).ids)
        worse += nb > na
        rows.append(
            {"british": british, "american": american, "british_tokens": nb,
             "american_tokens": na}
        )
    return {"pairs": rows, "worse_than_american": worse}


def measure(tokenizer_path: Path, heldout_dir: Path) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sources = measure_sources(tokenizer, heldout_dir)
    total_chars = sum(s["chars"] for s in sources.values())
    total_tokens = sum(s["tokens"] for s in sources.values())
    return {
        "tokenizer": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "sources": sources,
        "overall_chars_per_token": total_chars / total_tokens,
        "british": measure_british(tokenizer),
    }


def format_report(report: dict) -> str:
    lines = [
        f"tokenizer {report['tokenizer']}",
        f"vocab {report['vocab_size']}",
        "",
        f"{'source':22s} {'chars/token':>12s} {'bytes/token':>12s} {'tokens':>12s}",
    ]
    for name, stats in report["sources"].items():
        lines.append(
            f"{name:22s} {stats['chars_per_token']:>12.3f} "
            f"{stats['bytes_per_token']:>12.3f} {stats['tokens']:>12,}"
        )
    lines.append("")
    lines.append(f"overall chars per token {report['overall_chars_per_token']:.3f}")
    lines.append("")
    british = report["british"]
    lines.append(f"{'british':14s} {'tok':>4s}  {'american':14s} {'tok':>4s}")
    for row in british["pairs"]:
        flag = "" if row["british_tokens"] <= row["american_tokens"] else "  WORSE"
        lines.append(
            f"{row['british']:14s} {row['british_tokens']:>4d}  "
            f"{row['american']:14s} {row['american_tokens']:>4d}{flag}"
        )
    lines.append("")
    lines.append(
        f"{british['worse_than_american']} of {len(british['pairs'])} "
        f"british forms cost more than the american form"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kv-scout-tokenizer-metrics")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = measure(args.tokenizer, args.heldout)
    print(format_report(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
