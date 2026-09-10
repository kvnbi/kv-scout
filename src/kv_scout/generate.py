from __future__ import annotations

from dataclasses import dataclass

import torch

from kv_scout.model.cache import Cache

EM_DASH = "\u2014"
EN_DASH = "\u2013"
HORIZONTAL_BAR = "\u2015"
FIGURE_DASH = "\u2012"
SPACED_DOUBLE_HYPHEN = " -- "

DEFAULT_BANNED = (
    EM_DASH,
    EN_DASH,
    HORIZONTAL_BAR,
    FIGURE_DASH,
    SPACED_DOUBLE_HYPHEN,
)


def byte_level_alphabet() -> dict[str, int]:
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): value for code, value in zip(codes, printable)}


def token_bytes(tokenizer) -> dict[int, bytes]:
    alphabet = byte_level_alphabet()
    table: dict[int, bytes] = {}
    for token, index in tokenizer.get_vocab().items():
        try:
            table[index] = bytes(alphabet[ch] for ch in token)
        except KeyError:
            table[index] = token.encode("utf-8")
    return table


class SequenceBan:
    def __init__(self, tokenizer, banned: tuple[str, ...] = DEFAULT_BANNED) -> None:
        self.patterns = [text.encode("utf-8") for text in banned]
        self.longest = max(len(p) for p in self.patterns)
        self.table = token_bytes(tokenizer)

        states = {b""}
        for pattern in self.patterns:
            for cut in range(1, len(pattern)):
                states.add(pattern[:cut])
        self.states = sorted(states, key=len)

        self.blocked: dict[bytes, set[int]] = {}
        self.transitions: dict[bytes, dict[int, bytes]] = {}
        for state in self.states:
            blocked = set()
            moves = {}
            for index, raw in self.table.items():
                joined = state + raw
                if any(pattern in joined for pattern in self.patterns):
                    blocked.add(index)
                else:
                    moves[index] = self._next_state(joined)
            self.blocked[state] = blocked
            self.transitions[state] = moves

    def _next_state(self, stream: bytes) -> bytes:
        tail = stream[-(self.longest - 1) :] if self.longest > 1 else b""
        for cut in range(len(tail), 0, -1):
            candidate = tail[-cut:]
            if candidate in self.states and candidate:
                return candidate
        return b""

    def start(self) -> bytes:
        return b""

    def advance(self, state: bytes, token: int) -> bytes:
        return self.transitions[state].get(token, b"")

    def state_for(self, tokens: list[int]) -> bytes:
        state = self.start()
        for token in tokens:
            if token in self.blocked[state]:
                state = b""
                continue
            state = self.advance(state, token)
        return state

    def mask(self, logits: torch.Tensor, state: bytes) -> torch.Tensor:
        blocked = self.blocked[state]
        if not blocked:
            return logits
        index = torch.tensor(sorted(blocked), device=logits.device, dtype=torch.long)
        return logits.index_fill(-1, index, float("-inf"))

    def total_blocked(self) -> int:
        return sum(len(v) for v in self.blocked.values())


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    greedy: bool = False
    seed: int | None = None
    use_cache: bool = True


def _filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(ordered, dim=-1) > top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter(-1, order, ordered)
    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str = "",
    config: SamplingConfig = SamplingConfig(),
    ban: SequenceBan | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    model.eval()
    context = getattr(model.cfg, "context_max", getattr(model.cfg, "seq_len", 1024))
    cache = None
    if config.use_cache and hasattr(model.cfg, "cache_group"):
        cache = Cache(model.cfg)

    ids = tokenizer.encode(prompt).ids if prompt else []
    if not ids:
        ids = [tokenizer.token_to_id("<|endoftext|>")]
    generated: list[int] = []

    state = ban.state_for(ids) if ban is not None else None
    blocked_events = 0

    rng = None
    if config.seed is not None:
        rng = torch.Generator(device="cpu").manual_seed(config.seed)

    pending = ids[-context:]
    for _ in range(config.max_new_tokens):
        tokens = torch.tensor([pending], dtype=torch.long, device=device)
        logits = model(tokens, cache)[0, -1].float() if cache is not None else model(tokens)[0, -1].float()

        if ban is not None:
            if int(torch.argmax(logits)) in ban.blocked[state]:
                blocked_events += 1
            logits = ban.mask(logits, state)

        if config.greedy:
            nxt = int(torch.argmax(logits))
        else:
            scaled = logits / max(config.temperature, 1e-6)
            scaled = _filter(scaled, config.top_k, config.top_p)
            probs = torch.softmax(scaled, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=rng))

        ids.append(nxt)
        generated.append(nxt)
        pending = [nxt] if cache is not None else ids[-context:]
        if ban is not None:
            state = ban.advance(state, nxt)

    return {
        "text": tokenizer.decode(generated),
        "tokens": generated,
        "prompt_tokens": len(ids) - len(generated),
        "ban_events": blocked_events,
        "cached": cache is not None,
        "evictions": cache.evictions if cache is not None else 0,
    }
