"""Turn a list of tile specs into per-tile results, under the caller's rights and scope.

One entry point (``compute``) used by the endpoint, the export and the headless run, so a number
is never computed a second way. Each tile carries its own ``data_mode``, ``status`` and the
parameters it was computed with, so a page, a print and an export can all state them.
"""
from __future__ import annotations

import datetime as dt
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import live, presentation, registry as R
from .scope import Scope
from .spec import SpecRejected, Tile, parse_tile, validate_tile, window
from .live import Row

MARKER = "PRESENTATION DATA - NOT REAL"
TOLERANCE = 0.01   # a group sum may differ from its total by a rounding cent, no more


class ModeRefused(PermissionError):
    """The caller may not use the requested data mode."""


def default_mode() -> str:
    return os.getenv("REPORTS_DATA_SOURCE", "live").strip().lower() or "live"


def _provider(mode: str):
    return live if mode == "live" else presentation


def _num(v: Optional[float]) -> Optional[float]:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 2)


def _shift_year(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:                    # 29 Feb
        return d.replace(year=d.year + years, day=28)


def comparison_window(kind: str, start: dt.date, end: dt.date) -> Tuple[dt.date, dt.date]:
    """The window to compare against, the same length. 'end' is already capped at today when the
    period is still running, so a partial period is compared like-for-like."""
    if kind == "prior_year":
        return _shift_year(start, -1), _shift_year(end, -1)
    span = (end - start).days + 1
    return start - dt.timedelta(days=span), start - dt.timedelta(days=1)


def _key_str(k: Any) -> str:
    return k.isoformat() if isinstance(k, (dt.date, dt.datetime)) else str(k)


def _months_between(start: dt.date, end: dt.date) -> List[dt.date]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(dt.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


class _Ctx:
    def __init__(self, mode: str, scope: Scope, allowed: Dict[str, bool], today: dt.date):
        self.mode, self.scope, self.allowed, self.today = mode, scope, allowed, today
        self.prov = _provider(mode)

    def rows(self, metric: R.Metric, group_by, filters, start, end) -> List[Row]:
        if self.mode == "live":
            return live.series(metric, group_by, filters, start, end, self.scope)
        return presentation.series(metric, group_by, filters, start, end, None, self.today)


def _rows_to_map(rows: List[Row]) -> Dict[Tuple, Row]:
    return {r.keys: r for r in rows}


def _combine(op: str, a: Dict[Tuple, Row], b: Optional[Dict[Tuple, Row]]) -> List[Row]:
    out = []
    for k, ra in sorted(a.items(), key=lambda kv: tuple(_key_str(i) for i in kv[0])):
        va = ra.value
        if op == "share":
            tot = sum(r.value or 0 for r in a.values())
            v = None if not tot or va is None else 100.0 * va / tot
        else:
            vb = b.get(k).value if b and k in b else None
            if va is None or vb is None:
                v = None
            elif op == "ratio":
                v = None if vb == 0 else va / vb
            elif op == "difference":
                v = va - vb
            else:
                v = va + vb
        out.append(Row(k, ra.labels, v, None))
    return out


def _fill_months(rows: List[Row], group_by: List[str], start, end, additive: bool) -> List[Row]:
    """Empty months appear: as a zero for a count or a sum, as a gap (None) for a rate or an average,
    because 0 % would assert something the data did not say."""
    if not group_by or group_by[0] != "month":
        return rows
    have = {r.keys: r for r in rows}
    tail = len(group_by) - 1
    if tail:                              # month x other: only fill within groups already present
        return rows
    out = []
    for m in _months_between(start, end):
        r = have.get((m,)) or have.get((dt.datetime(m.year, m.month, 1),))
        out.append(r or Row((m,), (m.strftime("%b %Y"),), 0.0 if additive else None))
    return out


def _top(rows: List[Row], tile: Tile, additive: bool) -> List[Row]:
    if tile.sort == "label":
        rows = sorted(rows, key=lambda r: tuple(str(l) for l in r.labels))
    elif tile.sort in ("asc", "desc") and tile.group_by and tile.group_by[0] not in ("month", "quarter", "year"):
        rows = sorted(rows, key=lambda r: (r.value is None, (r.value or 0) * (-1 if tile.sort == "desc" else 1)))
    if tile.top and len(rows) > tile.top:
        kept, rest = rows[:tile.top], rows[tile.top:]
        if additive:
            kept = kept + [Row(("__other__",) * len(tile.group_by), ("Other",) + ("",) * (len(tile.group_by) - 1),
                               sum(r.value or 0 for r in rest), None)]
        return kept
    return rows


def _tile_period(tile: Tile, f: dt.date, t: dt.date) -> Tuple[dt.date, dt.date]:
    if tile.period:
        return window(tile.period)
    return f, t


def _compute_series(ctx: _Ctx, tile: Tile, metric_key: str, f, t, group_by=None) -> List[Row]:
    m = R.METRICS[metric_key]
    return ctx.rows(m, tile.group_by if group_by is None else group_by, tile.filters, f, t)


def _kpi_value(rows: List[Row]) -> Optional[float]:
    return rows[0].value if rows else None


def _result_for(ctx: _Ctx, tile: Tile, f: dt.date, t: dt.date, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The values for one tile. Raises LookupError when the live source cannot answer."""
    metric_keys = tile.metrics or []
    if tile.derive:
        d = tile.derive
        a = _rows_to_map(_compute_series(ctx, tile, d["a"], f, t))
        b = _rows_to_map(_compute_series(ctx, tile, d["b"], f, t)) if d.get("b") else None
        series = [("derived", f"{R.METRICS[d['a']].label} {d['op']}" + (f" {R.METRICS[d['b']].label}" if d.get("b") else ""),
                   _combine(d["op"], a, b), False, "ratio" if d["op"] == "ratio" else R.METRICS[d["a"]].format)]
    else:
        series = []
        for k in metric_keys:
            m = R.METRICS[k]
            rows = _compute_series(ctx, tile, k, f, t)
            series.append((k, m.label, rows, m.additive, m.format))
    additive_all = all(s[3] for s in series)

    out_series = []
    for key, label, rows, additive, fmt in series:
        rows = _fill_months(rows, tile.group_by, f, t, additive)
        # the groups must tie to the total, or the figure is withheld rather than published
        if tile.group_by and additive and not tile.derive:
            total_rows = _compute_series(ctx, tile, key, f, t, group_by=[])
            total = _kpi_value(total_rows) or 0.0
            gsum = sum(r.value or 0 for r in rows)
            if abs(gsum - total) > max(TOLERANCE, abs(total) * 1e-9):
                checks.append({"code": "groups_do_not_sum", "metric": key, "groups_sum": _num(gsum), "total": _num(total)})
                raise _Withheld()
            checks.append({"code": "groups_sum_to_total", "metric": key, "total": _num(total)})
        rows = _top(rows, tile, additive)
        out_series.append({"metric": key, "label": label, "format": fmt, "additive": additive,
                           "rows": [{"keys": [_key_str(k) for k in r.keys], "labels": [str(l) for l in r.labels],
                                     "value": _num(r.value), **({"assessed": r.n} if r.n is not None else {})} for r in rows]})
    return {"series": out_series, "group_by": tile.group_by}


class _Withheld(Exception):
    pass


def _comparison(ctx: _Ctx, tile: Tile, f, t, kind: str) -> Optional[Dict[str, Any]]:
    """Single-value comparison for a KPI tile."""
    if kind == "none" or tile.group_by or tile.derive or len(tile.metrics) != 1:
        return None
    cf, ct = comparison_window(kind, f, t)
    m = R.METRICS[tile.metrics[0]]
    cur = _kpi_value(ctx.rows(m, [], tile.filters, f, t))
    prev = _kpi_value(ctx.rows(m, [], tile.filters, cf, ct))
    delta = None if cur is None or prev is None else cur - prev
    pct = None if delta is None or not prev else 100.0 * delta / abs(prev)
    return {"kind": kind, "window": {"from": cf.isoformat(), "to": ct.isoformat()},
            "value": _num(prev), "delta": _num(delta), "delta_pct": _num(pct)}


def _viz_payload(ctx: _Ctx, tile: Tile, res: Dict[str, Any], f, t) -> Dict[str, Any]:
    s = res["series"]
    if tile.viz == "kpi":
        first = s[0]["rows"]
        value = first[0]["value"] if first else None
        spark = None
        if not tile.derive and len(tile.metrics) == 1:
            m = R.METRICS[tile.metrics[0]]
            if "month" in (m.dimensions or ()) or ctx.mode != "live":
                try:
                    sp = _fill_months(ctx.rows(m, ["month"], tile.filters, f, t), ["month"], f, t, m.additive)
                    spark = [_num(r.value) for r in sp]
                except Exception:
                    spark = None
        out = {"value": value, "sparkline": spark, "format": s[0]["format"]}
        if "assessed" in (first[0] if first else {}):
            out["assessed"] = first[0]["assessed"]
        return out
    if tile.viz in ("line", "bar"):
        first = s[0]
        if len(tile.group_by) == 2:
            # first-seen order: rows arrive ordered by their keys, so months stay chronological
            labels = list(dict.fromkeys(r["labels"][0] for r in first["rows"]))
            names = list(dict.fromkeys(r["labels"][1] for r in first["rows"]))
            data = {n: [next((r["value"] for r in first["rows"] if r["labels"][0] == l and r["labels"][1] == n), None)
                        for l in labels] for n in names}
            return {"labels": labels, "series": [{"name": n, "data": data[n]} for n in names], "format": first["format"]}
        return {"labels": [r["labels"][0] if r["labels"] else first["label"] for r in first["rows"]],
                "series": [{"name": x["label"], "data": [r["value"] for r in x["rows"]]} for x in s],
                "format": first["format"]}
    if tile.viz == "table":
        first = s[0]
        cols = [R.DIMENSIONS[d].label for d in tile.group_by] + [x["label"] for x in s]
        base = first["rows"]
        rows = []
        for i, r in enumerate(base):
            rows.append(list(r["labels"]) + [x["rows"][i]["value"] if i < len(x["rows"]) else None for x in s])
        total = None
        if first["additive"] and len(s) == 1 and tile.group_by:
            total = round(sum(r["value"] or 0 for r in base), 2)
        return {"columns": cols, "rows": rows, "total": total, "format": first["format"]}
    # findings: ranked by rule from numbers the server computed; words are templated, never generated
    first = s[0]
    ranked = sorted([r for r in first["rows"] if r["value"] is not None], key=lambda r: -abs(r["value"]))[:3]
    items = [{"text": f"{' / '.join(r['labels']) or first['label']}: {r['value']:,.2f}", "value": r["value"]} for r in ranked]
    return {"items": items}


def _marker_for(mode: str) -> Optional[str]:
    return MARKER if mode == "presentation" else None


def _envelope(tile: Tile, mode: str, status: str, f, t, ctx: _Ctx, **extra) -> Dict[str, Any]:
    partial = t > ctx.today
    env = {
        "id": tile.id, "data_mode": mode, "status": status, "viz": tile.viz,
        "params": {"metrics": tile.metrics or ([tile.derive["a"]] if tile.derive else []), "derive": tile.derive,
                   "group_by": tile.group_by, "filters": tile.filters, "top": tile.top, "sort": tile.sort,
                   "comparison": tile.comparison, "target": tile.target,
                   "period": {"from": f.isoformat(), "to": t.isoformat(), "partial": partial,
                              "as_of": ctx.today.isoformat(), "anchor": "CURRENT_DATE",
                              **({"effective_to": ctx.today.isoformat()} if partial else {})}},
        "checks": [], "marker": _marker_for(mode),
    }
    env.update(extra)
    return env


def compute(principal: Any, request: Dict[str, Any], *, scope: Optional[Scope] = None,
            authorise: Optional[Callable[[str], bool]] = None, mode_ok: Optional[Callable[[], bool]] = None,
            today: Optional[dt.date] = None) -> Dict[str, Any]:
    """The payload for a report. ``authorise(action)`` says whether the caller holds a read action;
    ``mode_ok()`` says whether presentation data is allowed for this caller and session."""
    today = today or dt.date.today()
    mode = str(request.get("data_mode") or default_mode()).lower()
    if mode not in ("live", "presentation"):
        raise SpecRejected(f"data_mode must be live or presentation, not {mode!r}")
    if mode == "presentation" and not (mode_ok and mode_ok()):
        raise ModeRefused("presentation data is not available")
    f, t = window(request, today)
    raw_tiles = request.get("tiles")
    if not isinstance(raw_tiles, list) or not raw_tiles:
        raise SpecRejected("a request needs a list of tiles")
    if len(raw_tiles) > 100:
        raise SpecRejected("at most 100 tiles per request")
    # one report is fully live or fully presentation: a tile naming the other mode refuses the request
    for rt in raw_tiles:
        if isinstance(rt, dict) and rt.get("data_mode") not in (None, mode):
            raise SpecRejected("a report cannot mix live and presentation tiles")
    from .scope import resolve
    scope = scope or (resolve(principal) if mode == "live" else Scope("presentation", True, ()))
    allowed: Dict[str, bool] = {}

    def may(action: str) -> bool:
        if mode != "live":
            return True
        if action not in allowed:
            allowed[action] = bool(authorise(action)) if authorise else False
        return allowed[action]

    ctx = _Ctx(mode, scope, allowed, today)
    tiles_out = []
    for raw in raw_tiles:
        tile = parse_tile(raw)               # a malformed spec rejects the whole request: it is a client bug
        try:
            validate_tile(tile, mode)
        except SpecRejected as exc:
            tiles_out.append(_envelope(tile, mode, "rejected", f, t, ctx, reason=str(exc)))
            continue
        tf, tt = _tile_period(tile, f, t)
        eff_t = min(tt, today)               # a running period is measured up to today, never padded
        env = _envelope(tile, mode, "ok", tf, tt, ctx)
        not_live = [k for k in tile.metric_keys if R.METRICS[k].availability != R.LIVE]
        if mode == "live" and not_live:
            env.update(status="no_data_source", reason=f"no live data source for {', '.join(not_live)}")
            tiles_out.append(env); continue
        if mode == "live":
            need = {R.SOURCES[R.METRICS[k].source].requires for k in tile.metric_keys}
            denied = sorted(a for a in need if not may(a))
            if denied:
                env.update(status="forbidden", reason="you do not have access to this data")
                tiles_out.append(env); continue
            if scope.assigned_nothing:
                env.update(status="empty_scope", reason="no deals are assigned to you")
                tiles_out.append(env); continue
        try:
            res = _result_for(ctx, tile, tf, eff_t, env["checks"])
            if tile.viz == "kpi" and (tile.comparison or R.METRICS[(tile.metrics or [tile.derive["a"]])[0]].default_comparison) != "none":
                kind = tile.comparison or R.METRICS[(tile.metrics or [tile.derive["a"]])[0]].default_comparison
                env["comparison"] = _comparison(ctx, tile, tf, eff_t, kind)
            env["result"] = _viz_payload(ctx, tile, res, tf, eff_t)
            if tile.target is not None:
                env["target"] = tile.target
            if tf <= today < tt or tt > today:
                env["checks"].append({"code": "partial_period", "as_of": today.isoformat(),
                                      "note": "the period is still running; compared like-for-like"})
            if mode == "live" and scope.all_rows and "finding" == (R.METRICS[tile.metric_keys[0]].source or ""):
                env["checks"].append({"code": "unattributed_findings", "count": live.unattributed_findings(tf, eff_t, R.METRICS[tile.metric_keys[0]].where),
                                      "note": "findings that map to no deal; visible to Admin only"})
        except _Withheld:
            env.update(status="withheld", reason="the groups do not add up to the total, so the figure is not published")
        except LookupError as exc:
            env.update(status="no_data_source", reason=str(exc))
        tiles_out.append(env)
    return {"data_mode": mode, "org": presentation.ORG if mode == "presentation" else None,
            "marker": _marker_for(mode), "as_of": today.isoformat(),
            "period": {"from": f.isoformat(), "to": t.isoformat()}, "tiles": tiles_out}
