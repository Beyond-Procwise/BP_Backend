"""The statistics behind a learning suggestion, and the guardrails (brief §6.2, §6.3).

Pure: no database, no settings lookup. A group is the decisions people made on one band of
values under one policy condition. Timeouts and system decisions never reach this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

Z95 = 1.959963984540054   # two-sided 95%


@dataclass(frozen=True)
class Decided:
    firing_id: int
    values: Dict[str, Any] = field(hash=False)
    approved: bool = False
    decided_by: str = ""
    asked_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    reversed: bool = False
    edited: bool = False


def wilson_lower(k: int, n: int) -> float:
    """Lower bound of the 95% Wilson score interval for k approvals out of n decisions."""
    if n <= 0:
        return 0.0
    p, z2 = k / n, Z95 * Z95
    return (p + z2 / (2 * n) - Z95 * sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)


def stats(group: Sequence[Decided]) -> Dict[str, Any]:
    n = len(group)
    if not n:
        return {"decisions": 0, "approved": 0, "wilsonLower": 0.0, "approvers": 0, "days": 0,
                "edited": 0, "reversed": 0, "medianSeconds": None}
    k = sum(1 for d in group if d.approved)
    when = [d.decided_at for d in group]
    return {
        "decisions": n,
        "approved": k,
        "wilsonLower": round(wilson_lower(k, n), 4),
        "approvers": len({str(d.decided_by).strip().lower() for d in group}),
        "days": (max(when) - min(when)).days,
        "edited": sum(1 for d in group if d.edited),
        "reversed": sum(1 for d in group if d.reversed),
        "medianSeconds": float(median((d.decided_at - d.asked_at).total_seconds() for d in group)),
    }


def _num(cfg: Dict[str, Any], key: str) -> float:
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"learning setting {key!r} is not a number")
    return value


def failed_guardrails(s: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """The guardrails that do not hold, in the brief's order. Empty = raise the suggestion."""
    failed: List[str] = []
    if s["decisions"] < _num(cfg, "min_decisions"):
        failed.append("min_decisions")
    if s["days"] < _num(cfg, "min_days"):
        failed.append("min_days")
    if s["approvers"] < _num(cfg, "min_approvers"):
        failed.append("min_approvers")
    if s["wilsonLower"] < _num(cfg, "wilson_lower"):
        failed.append("wilson_lower")
    if s["edited"] or s["reversed"]:
        failed.append("edited_or_reversed")
    floor = _num(cfg, "median_seconds_floor")
    if s["medianSeconds"] is None or s["medianSeconds"] < floor:
        failed.append("median_seconds_floor")
    return failed
