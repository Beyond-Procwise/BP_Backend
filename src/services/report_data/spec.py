"""A tile is a query spec over the registry. Validation is the allowlist: anything not named in
the registry is rejected with a reason, never interpreted and never partly executed."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import registry as R


class SpecRejected(ValueError):
    """The spec names something the registry does not allow. The message says what."""


@dataclass
class Tile:
    id: str
    metrics: List[str]                    # one, or several for a multi-series tile
    derive: Optional[Dict[str, str]]      # {"op": ratio|difference|sum|share, "a": key, "b": key}
    group_by: List[str]
    filters: Dict[str, List[str]]
    comparison: Optional[str]
    target: Optional[float]
    viz: str
    top: Optional[int] = None
    sort: str = "desc"                    # desc | asc | label
    data_mode: Optional[str] = None       # a tile that names a mode must match the request's
    period: Optional[Dict[str, Any]] = None   # tile-level override {from, to}

    @property
    def metric_keys(self) -> List[str]:
        keys = list(self.metrics)
        if self.derive:
            keys += [self.derive["a"]] + ([self.derive["b"]] if self.derive.get("b") else [])
        return keys


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def parse_tile(raw: Dict[str, Any]) -> Tile:
    if not isinstance(raw, dict):
        raise SpecRejected("a tile must be an object")
    tid = str(raw.get("id") or "")
    if not tid:
        raise SpecRejected("a tile needs an id")
    derive = raw.get("derive")
    metrics = _as_list(raw.get("metric"))
    if derive is not None:
        if metrics:
            raise SpecRejected(f"tile {tid}: give either metric or derive, not both")
        if not isinstance(derive, dict) or derive.get("op") not in R.DERIVE_OPS or not derive.get("a"):
            raise SpecRejected(f"tile {tid}: derive needs op in {R.DERIVE_OPS} and a metric 'a'")
        if derive["op"] != "share" and not derive.get("b"):
            raise SpecRejected(f"tile {tid}: derive {derive['op']} needs a second metric 'b'")
        derive = {k: str(v) for k, v in derive.items() if k in ("op", "a", "b")}
    elif not metrics:
        raise SpecRejected(f"tile {tid}: a tile needs a metric")
    viz = raw.get("viz") or "kpi"
    if viz not in R.VIZ:
        raise SpecRejected(f"tile {tid}: viz {viz!r} is not one of {R.VIZ}")
    comparison = raw.get("comparison")
    if comparison is not None and comparison not in R.COMPARISONS:
        raise SpecRejected(f"tile {tid}: comparison {comparison!r} is not one of {R.COMPARISONS}")
    group_by = _as_list(raw.get("groupBy", raw.get("group_by")))
    if len(group_by) > R.MAX_GROUP_BY:
        raise SpecRejected(f"tile {tid}: at most {R.MAX_GROUP_BY} group-by dimensions")
    if len(set(group_by)) != len(group_by):
        raise SpecRejected(f"tile {tid}: a dimension can be grouped once")
    filters_in = raw.get("filters") or {}
    if not isinstance(filters_in, dict):
        raise SpecRejected(f"tile {tid}: filters must be an object")
    filters = {str(k): _as_list(v) for k, v in filters_in.items()}
    target = raw.get("target")
    if target is not None:
        try:
            target = float(target)
        except (TypeError, ValueError):
            raise SpecRejected(f"tile {tid}: target must be a number")
    top = raw.get("top")
    if top is not None:
        if not isinstance(top, int) or isinstance(top, bool) or not (1 <= top <= 500):
            raise SpecRejected(f"tile {tid}: top must be a whole number from 1 to 500")
    sort = raw.get("sort", "desc")
    if sort not in ("desc", "asc", "label"):
        raise SpecRejected(f"tile {tid}: sort must be desc, asc or label")
    return Tile(tid, metrics, derive, group_by, filters, comparison, target, viz, top, sort,
                raw.get("data_mode"), raw.get("period") if isinstance(raw.get("period"), dict) else None)


def validate_tile(t: Tile, mode: str) -> None:
    """Reject anything the registry does not allow in this mode. Pure; touches no data."""
    if t.derive and t.viz == "kpi" and t.group_by:
        raise SpecRejected(f"tile {t.id}: a KPI is a single value; use line, bar or table to group")
    for key in t.metric_keys:
        m = R.METRICS.get(key)
        if m is None:
            raise SpecRejected(f"tile {t.id}: unknown metric {key!r}")
        if m.availability == R.UNAVAILABLE:
            raise SpecRejected(f"tile {t.id}: metric {key!r} is unavailable")
        allowed = m.dimensions if mode == "live" else tuple(d for d in tuple(m.dimensions) + tuple(m.presentation_dimensions) if d in R.PRESENTATION_DIMS)
        for d in t.group_by + list(t.filters):
            dim = R.DIMENSIONS.get(d)
            if dim is None:
                raise SpecRejected(f"tile {t.id}: unknown dimension {d!r}")
            if mode == "live" and dim.availability != R.LIVE:
                raise SpecRejected(f"tile {t.id}: dimension {d!r} has no live data")
            if d not in allowed:
                raise SpecRejected(f"tile {t.id}: metric {key!r} cannot be grouped or filtered by {d!r}")
    for d in t.filters:
        if R.DIMENSIONS[d].is_time:
            raise SpecRejected(f"tile {t.id}: filter by period, not by {d!r}")


def window(req: Dict[str, Any], today: Optional[dt.date] = None) -> Tuple[dt.date, dt.date]:
    """The inclusive [from, to] a request asks for. Dates only; one anchor (CURRENT_DATE)."""
    today = today or dt.date.today()
    try:
        f = dt.date.fromisoformat(str(req["from"]))
        t = dt.date.fromisoformat(str(req["to"]))
    except (KeyError, ValueError):
        raise SpecRejected("a request needs from and to as ISO dates")
    if t < f:
        raise SpecRejected("to is before from")
    if (t - f).days > 366 * 10:
        raise SpecRejected("the period is longer than ten years")
    return f, t
