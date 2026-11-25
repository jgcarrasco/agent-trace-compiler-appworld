"""Prices for the models these experiments call, read from the providers' pages.

Peak DeepSeek rates by choice: the tier multiplies every rate by the same
factor, so it changes absolute dollars but not any ratio, and peak keeps the
numbers conservative. Cache reads are priced separately because DeepSeek bills
them at a fiftieth of a miss; Jev has no cache tier at all, encoded as `hit=None`
rather than a zero so a run that reports cache hits against it fails loudly.

Every number in this file was read from the source URL on `read_on`. If a model
is missing, pricing raises instead of quietly costing nothing — which is how the
native runs ended up reporting `cost: 0.0`.
"""

from dataclasses import dataclass
import hashlib
import json

SOURCE_DEEPSEEK = "https://api-docs.deepseek.com/quick_start/pricing"
SOURCE_TYPESAFE = "https://typesafe.ai/pricing"
READ_ON = "2026-10-21"


@dataclass(frozen=True)
class Price:
    """USD per token, peak rates."""

    miss: float
    hit: float | None
    output: float
    source: str
    note: str = ""


PRICES = {
    "deepseek-flash": Price(
        miss=0.30 / 1_000_000,
        hit=0.006 / 1_000_000,
        output=1.20 / 1_000_000,
        source=SOURCE_DEEPSEEK,
        note="peak hours, 01:00-04:00 and 06:00-10:00 UTC on weekdays",
    ),
    "jev-1.14": Price(
        miss=0.042 / 1_000_000,
        hit=None,
        output=0.0,
        source=SOURCE_TYPESAFE,
        note="output tokens are free and there is no cache tier",
    ),
}

# Providers name the snapshot that served the call; the table is keyed by the
# model generation those snapshots belong to.
ALIASES = {
    "jev-1.14.0": "jev-1.14",
    # The trace and compiler agent run DeepSeek V4.1 Flash through the
    # opencode-go provider; the records name it with and without the prefix.
    "deepseek-v4.1-flash": "deepseek-flash",
    "opencode-go/deepseek-v4.1-flash": "deepseek-flash",
    # The Decisions endpoint names the preview snapshot; it is the same model
    # generation as the released one, priced identically.
    "jev-preview": "jev-1.14",
}


def resolve(model: str) -> Price:
    key = ALIASES.get(model, model)
    if key not in PRICES:
        raise KeyError(f"no price for {model!r}: add it to bench/pricing.py")
    return PRICES[key]


def cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_hit_tokens: int = 0,
) -> float:
    """Price one call from the tokens the provider reported."""
    price = resolve(model)
    hit = min(cache_hit_tokens, input_tokens)
    if hit and price.hit is None:
        raise ValueError(f"{model} reports cache hits but has no cache tier")
    miss = input_tokens - hit
    total = miss * price.miss + output_tokens * price.output
    return total + hit * (price.hit or 0.0)


def fingerprint() -> str:
    """What a run records, so its dollars can be traced to a price table."""
    table = {
        name: {"miss": p.miss, "hit": p.hit, "output": p.output, "source": p.source}
        for name, p in PRICES.items()
    }
    payload = json.dumps({"read_on": READ_ON, "prices": table}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
