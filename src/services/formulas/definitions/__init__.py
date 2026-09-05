"""Importing this package registers every formula --- and runs every golden vector.

Import order is not significant: each module registers under its own names. What
IS significant is that an import failure here is a *correct* failure. If any
formula no longer reproduces its pinned behaviour, this package does not import,
and anything that depends on it stops rather than quietly computing new numbers.
"""
from __future__ import annotations

from . import (  # noqa: F401
    benchmarking,
    clustering,
    deal,
    duplicates,
    extraction,
    linking,
    negotiation,
    opportunity,
    price_outlier,
    ranking,
    risk,
    rivalry,
)

__all__ = [
    "benchmarking", "clustering", "deal", "duplicates", "extraction", "linking",
    "negotiation", "opportunity", "price_outlier", "ranking", "risk", "rivalry",
]
