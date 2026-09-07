from __future__ import annotations

import re

DASHES = "\u2014\u2013\u2015\u2012"

NUMERIC_RANGE = re.compile(rf"(?<=\d)\s*[{DASHES}]\s*(?=\d)")
INTRA_WORD = re.compile(rf"(?<=\w)[{DASHES}](?=\w)")
DIALOGUE_OPENER = re.compile(rf"^(\s*)[{DASHES}]\s*")
ANY_DASH = re.compile(rf"\s*[{DASHES}]\s*")

DOUBLE_COMMA = re.compile(r",\s*,")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
COMMA_BEFORE_STOP = re.compile(r",\s*([.;:!?])")
REPEATED_SPACE = re.compile(r"[ \t]{2,}")


def normalise_line(line: str) -> str:
    opener = DIALOGUE_OPENER.match(line)
    if opener:
        head = line[: opener.end()]
        tail = line[opener.end() :]
    else:
        head, tail = "", line

    tail = NUMERIC_RANGE.sub("-", tail)
    tail = INTRA_WORD.sub("-", tail)
    tail = ANY_DASH.sub(", ", tail)

    tail = DOUBLE_COMMA.sub(",", tail)
    tail = COMMA_BEFORE_STOP.sub(r"\1", tail)
    tail = SPACE_BEFORE_PUNCT.sub(r"\1", tail)
    tail = REPEATED_SPACE.sub(" ", tail)
    return head + tail


def normalise_dashes(text: str) -> str:
    if not any(d in text for d in DASHES):
        return text
    return "\n".join(normalise_line(line) for line in text.split("\n"))


def count_dashes(text: str) -> int:
    return sum(text.count(d) for d in DASHES)
