from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

END_OF_TEXT = "<|endoftext|>"

STRUCTURAL_TOKENS = (
    END_OF_TEXT,
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|tool_end|>",
)

REFLECTION_TOKENS = (
    "[RETRIEVE:yes]",
    "[RETRIEVE:no]",
    "[RETRIEVE:continue]",
    "[REL:yes]",
    "[REL:partial]",
    "[REL:no]",
    "[SUP:full]",
    "[SUP:partial]",
    "[SUP:none]",
    "[USE:1]",
    "[USE:2]",
    "[USE:3]",
    "[USE:4]",
    "[USE:5]",
    "[MORE:yes]",
    "[MORE:no]",
)

REFLECTION_SLOTS = 32

BRITISH_SEED_FORMS = (
    "colour",
    "colours",
    "realise",
    "realised",
    "centre",
    "travelled",
    "defence",
    "organise",
    "organised",
    "recognise",
    "recognised",
    "behaviour",
    "favourite",
    "neighbour",
    "labour",
    "analyse",
    "analysed",
    "licence",
    "practise",
    "programme",
)


def reserved_tokens() -> tuple[str, ...]:
    spare = REFLECTION_SLOTS - len(REFLECTION_TOKENS)
    if spare < 0:
        raise ValueError("reflection tokens exceed the reserved slot count")
    padding = tuple(f"[RESERVED:{i:02d}]" for i in range(spare))
    return REFLECTION_TOKENS + padding


def _build(files: list[Path], vocab_size: int, min_frequency: int, extra: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    specials = list(STRUCTURAL_TOKENS) + list(reserved_tokens()) + extra
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=specials,
    )
    tokenizer.train([str(p) for p in files], trainer)
    return tokenizer


def _missing_british(tokenizer) -> list[str]:
    vocab = tokenizer.get_vocab()
    missing = []
    for form in BRITISH_SEED_FORMS:
        if "\u0120" + form not in vocab and form not in vocab:
            missing.append(" " + form)
    return missing


def train(
    files: list[Path],
    out_path: Path,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    seed_british: bool = True,
) -> dict:
    from tokenizers import AddedToken

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"missing training file {path}")

    seeded: list[str] = []
    tokenizer = _build(files, vocab_size, min_frequency, [])

    if seed_british:
        seeded = _missing_british(tokenizer)
        if seeded:
            tokenizer = _build(
                files, vocab_size - len(seeded), min_frequency, []
            )
            seeded = _missing_british(tokenizer)
            tokenizer.add_tokens(
                [AddedToken(t, single_word=False, normalized=False) for t in seeded]
            )

    shortfall = vocab_size - tokenizer.get_vocab_size()
    padding = [f"[RESERVED:{REFLECTION_SLOTS + i:02d}]" for i in range(max(0, shortfall))]
    if padding:
        tokenizer.add_special_tokens(padding)

    final = tokenizer.get_vocab_size()
    if final != vocab_size:
        raise ValueError(f"tokenizer has {final} entries, expected {vocab_size}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    vocab = tokenizer.get_vocab()

    report = {
        "vocab_size": final,
        "sha256": digest,
        "structural_tokens": list(STRUCTURAL_TOKENS),
        "reflection_slots": REFLECTION_SLOTS,
        "reflection_tokens": list(REFLECTION_TOKENS),
        "eos_id": vocab[END_OF_TEXT],
        "seeded_british_forms": seeded,
        "padding_tokens": len(padding),
        "min_frequency": min_frequency,
        "training_files": [str(p) for p in files],
    }
    (out_path.parent / "tokenizer_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kv-scout-train-tokenizer")
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--no-seed-british", action="store_true")
    args = parser.parse_args(argv)

    files = sorted(args.sample_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no text files in {args.sample_dir}")

    report = train(
        files=files,
        out_path=args.out,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        seed_british=not args.no_seed_british,
    )
    print(
        f"vocab {report['vocab_size']}, eos id {report['eos_id']}, "
        f"sha256 {report['sha256'][:16]}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
