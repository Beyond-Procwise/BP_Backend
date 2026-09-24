"""Tolerances for triage: the ONLY place one is decided (spec §5.3).

Every value comes from governed-limit policy `triage_tolerances`. A missing or null
value refuses (LimitUnavailable) rather than falling back to a number in code — a
guard that cannot tell "policy missing" from "policy says 1%" is not a guard.

`resolve_tolerance(check, cfg, ctx)` takes a context it does not yet use. That is the
seam for the triage spec's leniency hierarchy (§7.3: customer, category, counterparty,
document, field), which arrives when the data has those dimensions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional

from src.services import governed_limits
from src.services.governed_limits import LimitUnavailable

POLICY = "triage_tolerances"


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


def _rates(value: Any) -> tuple:
    return tuple(Decimal(str(v)) for v in value)


KEYS: dict[str, Callable[[Any], Any]] = {
    "unit_price_over_pct": _dec, "unit_price_over_abs": _dec, "unit_price_combine": str,
    "unit_price_under_pct": _dec, "quantity_over_pct": _dec, "rounding_per_line": _dec,
    "cumulative_total_pct": _dec, "cumulative_total_abs": _dec,
    "cumulative_total_combine": str, "allowed_tax_rates": _rates,
    "min_link_confidence": float, "unlinked_below": float,
    "min_extraction_confidence": float, "description_min_similarity": float,
    "materiality_pct_of_total": _dec, "materiality_floor": _dec,
    "materiality_ceiling": _dec, "band_s1": float, "band_s2": float,
    "uplift_min_lines": int, "uplift_same_pct_within": _dec, "batch_size": int,
}

_COMBINES = ("min", "max", "pct_only", "abs_only")


@dataclass(frozen=True)
class TriageConfig:
    values: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({k: str(v) for k, v in sorted(self.values.items())}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(read: Optional[Callable[..., Any]] = None) -> TriageConfig:
    read = read or governed_limits.limit
    values = {}
    for key, cast in KEYS.items():
        value = read(POLICY, key, cast=cast)
        if value is None:
            raise LimitUnavailable(f"{POLICY} states null for {key!r}; triage needs a value")
        values[key] = value
    return TriageConfig(values)


@dataclass(frozen=True)
class Tolerance:
    pct: Decimal
    abs_gbp: Optional[Decimal]
    combine: str
    source: str

    def allowance(self, base: Decimal, fx_to_gbp: Optional[Decimal]) -> Decimal:
        """How far a value may differ from `base` (document currency) and still pass."""
        if self.combine not in _COMBINES:
            raise ValueError(f"unknown combine {self.combine!r}")
        pct_part = abs(base) * self.pct / Decimal(100)
        abs_part = (self.abs_gbp / fx_to_gbp
                    if self.abs_gbp is not None and fx_to_gbp else None)
        if self.combine == "pct_only" or abs_part is None:
            return pct_part
        if self.combine == "abs_only":
            return abs_part
        return min(pct_part, abs_part) if self.combine == "min" else max(pct_part, abs_part)

    def as_dict(self) -> dict:
        return {"pct": str(self.pct),
                "abs_gbp": None if self.abs_gbp is None else str(self.abs_gbp),
                "combine": self.combine, "source": self.source}


def resolve_tolerance(check: str, cfg: TriageConfig, ctx: Optional[dict] = None) -> Tolerance:
    src = f"bp_policy:{POLICY}"
    if check == "unit_price_over":
        return Tolerance(cfg["unit_price_over_pct"], cfg["unit_price_over_abs"],
                         cfg["unit_price_combine"], f"{src}.unit_price_over_*")
    if check == "unit_price_under":
        return Tolerance(cfg["unit_price_under_pct"], None, "pct_only",
                         f"{src}.unit_price_under_pct")
    if check == "quantity_over":
        return Tolerance(cfg["quantity_over_pct"], None, "pct_only",
                         f"{src}.quantity_over_pct")
    if check == "cumulative_total":
        return Tolerance(cfg["cumulative_total_pct"], cfg["cumulative_total_abs"],
                         cfg["cumulative_total_combine"], f"{src}.cumulative_total_*")
    raise KeyError(f"no tolerance defined for {check!r}")
