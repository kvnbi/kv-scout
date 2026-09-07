from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

DASHES = "\u2014\u2013\u2015\u2012"

LLM_VOCABULARY = (
    "delve",
    "tapestry",
    "testament to",
    "underscore",
    "underscores",
    "navigate the complexities",
    "in today's fast paced world",
    "it's important to note",
    "crucial",
    "leverage",
    "leverages",
    "leveraging",
    "robust",
    "seamless",
    "seamlessly",
    "moreover",
    "furthermore",
    "landscape of",
    "realm of",
    "embark",
    "unlock",
    "elevate",
    "harness",
)

HEDGING_OPENERS = (
    "great question",
    "that's a great question",
    "i'd be happy to",
    "i would be happy to",
    "certainly!",
    "of course!",
    "absolutely!",
)

NOT_JUST_BUT = re.compile(r"\bnot (?:just|only)\b[^.!?]{1,80}?\bbut\b", re.IGNORECASE)
TRICOLON = re.compile(r"\b\w+\s*,\s+\w+\s*,\s+and\s+\w+\b")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
BULLET = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+\.\s+)\S", re.MULTILINE)
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
QUESTION_OPENER = re.compile(r"^[^.!?\n]{1,120}\?")


def _word_pattern(phrase: str) -> re.Pattern:
    if " " in phrase:
        return re.compile(re.escape(phrase).replace(r"\ ", r"\s+"), re.IGNORECASE)
    return re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)


VOCABULARY_PATTERNS = {word: _word_pattern(word) for word in LLM_VOCABULARY}


def count_dashes(text: str) -> int:
    return sum(text.count(d) for d in DASHES)


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def opening_closing_overlap(text: str) -> float:
    parts = paragraphs(text)
    if len(parts) < 2:
        return 0.0
    first = set(re.findall(r"[a-z]{4,}", parts[0].lower()))
    last = set(re.findall(r"[a-z]{4,}", parts[-1].lower()))
    if not first or not last:
        return 0.0
    return len(first & last) / len(first | last)


@dataclass(frozen=True)
class VoiceReport:
    tokens: int
    characters: int
    dashes_per_1k: float
    llm_vocabulary_per_1k: float
    llm_vocabulary_counts: dict
    not_just_but_per_1k: float
    tricolon_per_1k: float
    hedging_openers: int
    question_openers: int
    headings_per_1k: float
    bullets_per_1k: float
    bold_per_1k: float
    markdown_density_per_1k: float
    opening_closing_overlap: float

    def to_payload(self) -> dict:
        return asdict(self)


def measure_voice(text: str, tokenizer=None) -> VoiceReport:
    if tokenizer is not None:
        tokens = max(1, len(tokenizer.encode(text).ids))
    else:
        tokens = max(1, len(text.split()))
    per_1k = 1000.0 / tokens

    counts = {}
    for word, pattern in VOCABULARY_PATTERNS.items():
        found = len(pattern.findall(text))
        if found:
            counts[word] = found
    vocabulary_total = sum(counts.values())

    headings = len(HEADING.findall(text))
    bullets = len(BULLET.findall(text))
    bold = len(BOLD.findall(text))

    lowered = text.lower()
    hedging = sum(lowered.count(opener) for opener in HEDGING_OPENERS)
    question_openers = sum(
        1 for part in paragraphs(text) if QUESTION_OPENER.match(part)
    )

    return VoiceReport(
        tokens=tokens,
        characters=len(text),
        dashes_per_1k=count_dashes(text) * per_1k,
        llm_vocabulary_per_1k=vocabulary_total * per_1k,
        llm_vocabulary_counts=counts,
        not_just_but_per_1k=len(NOT_JUST_BUT.findall(text)) * per_1k,
        tricolon_per_1k=len(TRICOLON.findall(text)) * per_1k,
        hedging_openers=hedging,
        question_openers=question_openers,
        headings_per_1k=headings * per_1k,
        bullets_per_1k=bullets * per_1k,
        bold_per_1k=bold * per_1k,
        markdown_density_per_1k=(headings + bullets + bold) * per_1k,
        opening_closing_overlap=opening_closing_overlap(text),
    )


def measure_files(paths, tokenizer=None, limit_bytes: int = 5_000_000) -> VoiceReport:
    chunks = []
    for path in paths:
        chunks.append(Path(path).read_text(errors="ignore")[:limit_bytes])
    return measure_voice("\n".join(chunks), tokenizer)


COMPARED = (
    "dashes_per_1k",
    "llm_vocabulary_per_1k",
    "not_just_but_per_1k",
    "tricolon_per_1k",
    "markdown_density_per_1k",
)


def compare(report: VoiceReport, baseline: VoiceReport | dict) -> dict:
    if isinstance(baseline, VoiceReport):
        baseline = baseline.to_payload()
    rows = {}
    for field in COMPARED:
        model = getattr(report, field)
        human = baseline[field]
        rows[field] = {
            "model": model,
            "human": human,
            "ratio": model / human if human > 0 else float("inf") if model else 1.0,
        }
    return rows


def format_comparison(rows: dict) -> str:
    lines = [f"{'metric':26s} {'model':>10s} {'human':>10s} {'ratio':>8s}"]
    for name, row in rows.items():
        ratio = row["ratio"]
        shown = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
        lines.append(f"{name:26s} {row['model']:>10.3f} {row['human']:>10.3f} {shown:>8s}")
    return "\n".join(lines)


def load_baseline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_baseline(report: VoiceReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_payload(), indent=2) + "\n")
    return path


def measure_model(
    checkpoint_dir: str | Path,
    tokenizer_path: str | Path,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    device: str = "cpu",
    ban_dashes: bool = True,
) -> tuple[VoiceReport, int]:
    import torch
    from tokenizers import Tokenizer

    from kv_scout import checkpoint as ckpt
    from kv_scout.config import ModelConfig, from_dict
    from kv_scout.generate import SamplingConfig, SequenceBan, generate
    from kv_scout.model import KVScout

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    path = ckpt.latest_path(checkpoint_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(ModelConfig, payload["config"]["model"])
    model = KVScout(cfg).to(device)
    model.load_state_dict(payload["model"])

    ban = SequenceBan(tokenizer) if ban_dashes else None
    pieces = []
    fired = 0
    for index, prompt in enumerate(prompts):
        out = generate(
            model,
            tokenizer,
            prompt,
            SamplingConfig(
                max_new_tokens=max_new_tokens, temperature=temperature, seed=index
            ),
            ban=ban,
            device=device,
        )
        pieces.append(prompt + out["text"])
        fired += out["ban_events"]
    return measure_voice("\n\n".join(pieces), tokenizer), fired


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="kv-scout-voice")
    parser.add_argument("--files", nargs="*", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("tokenizer/voice_baseline.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-baseline", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.files:
        from tokenizers import Tokenizer

        report = measure_files(args.files, Tokenizer.from_file(str(args.tokenizer)))
    elif args.checkpoint:
        report, fired = measure_model(
            args.checkpoint,
            args.tokenizer,
            DEFAULT_PROMPTS,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        print(f"dash ban fired {fired} times\n")
    else:
        raise SystemExit("pass either --files or --checkpoint")

    if args.save_baseline:
        save_baseline(report, args.save_baseline)
        print(f"saved baseline to {args.save_baseline}")

    print(f"measured {report.tokens:,} tokens\n")
    if args.baseline.exists():
        print(format_comparison(compare(report, load_baseline(args.baseline))))
    else:
        print(json.dumps(report.to_payload(), indent=2))
    return 0


DEFAULT_PROMPTS = [
    "The history of the city",
    "In 1914 the government",
    "She said that the",
    "The main advantage of",
    "According to the report",
    "Scientists have found that",
]


if __name__ == "__main__":
    raise SystemExit(main())
