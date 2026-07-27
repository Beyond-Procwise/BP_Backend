"""Deterministic randomness.

Each generator draws from its own named stream so that adding, removing or
reordering a generator cannot shift another one's output. That property is what
makes the published answer key survive a regeneration.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence


def make_rng(seed: int, stream: str) -> random.Random:
    """A Random seeded from (seed, stream), independent of every other stream."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def weighted_apportion(weights: Sequence[float], total: int) -> list[int]:
    """Split `total` across `weights` so the result sums to exactly `total`.

    Largest-remainder method: floor every share, then hand the shortfall to the
    entries with the largest fractional parts. Ties break on index, so the result
    is deterministic.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive number")

    exact = [w * total / weight_sum for w in weights]
    floors = [int(value) for value in exact]
    shortfall = total - sum(floors)
    order = sorted(
        range(len(exact)), key=lambda i: (-(exact[i] - floors[i]), i)
    )
    for index in order[:shortfall]:
        floors[index] += 1
    return floors
