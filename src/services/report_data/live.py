"""The live provider: registry fragments + bound parameters, run against the real tables.

``build_query`` is pure (returns SQL and params) so the rule "no SQL from user input" is
testable without a database: every request value appears in ``params``, never in ``sql``.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import registry as R
from .scope import Scope


@dataclass
class Row:
    keys: Tuple[Any, ...]
    labels: Tuple[Any, ...]
    value: Optional[float]
    n: Optional[int] = None          # how many the figure was taken over (3-state metrics)


def build_query(metric: R.Metric, group_by: List[str], filters: Dict[str, List[str]],
                start: dt.date, end_exclusive: dt.date, scope: Scope) -> Tuple[str, Dict[str, Any]]:
    if metric.availability != R.LIVE or not metric.source or not metric.measure:
        raise LookupError(f"{metric.key} has no live source")
    src = R.SOURCES[metric.source]
    params: Dict[str, Any] = {"t_from": start, "t_to": end_exclusive}
    where = [f"{src.time_expr} >= %(t_from)s", f"{src.time_expr} < %(t_to)s"]
    if metric.where:
        where.append(f"({metric.where})")
    if not scope.all_rows:
        params["buyers"] = list(scope.buyers)
        where.append(f"{src.scope_expr} = ANY(%(buyers)s)")
    for i, (dim, values) in enumerate(sorted(filters.items())):
        key_sql = src.dims[dim][0]            # KeyError = the registry does not offer it: a bug, not a user error
        params[f"flt_{i}"] = [str(v) for v in values]
        where.append(f"{key_sql}::text = ANY(%(flt_{i})s)")
    keys = [src.dims[d][0] for d in group_by]
    labels = [src.dims[d][1] for d in group_by]
    sel = ", ".join([f"{k} AS k{i}" for i, k in enumerate(keys)] + [f"{l} AS l{i}" for i, l in enumerate(labels)]
                    + [f"{metric.measure} AS v"] + ([f"{metric.assessed} AS n"] if metric.assessed else []))
    sql = f"SELECT {sel} FROM {src.from_sql} WHERE {' AND '.join(where)}"
    if keys:
        grp = ", ".join(keys + labels)
        sql += f" GROUP BY {grp} ORDER BY {', '.join(keys)}"
    return sql, params


def _num(v: Any) -> Optional[float]:
    return None if v is None else float(v)


def run_query(sql: str, params: Dict[str, Any], n_dims: int, has_n: bool) -> List[Row]:
    from src.services.db import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        out = []
        for r in cur.fetchall():
            keys = tuple(r[:n_dims]); labels = tuple(r[n_dims:2 * n_dims])
            v = r[2 * n_dims]
            n = r[2 * n_dims + 1] if has_n else None
            out.append(Row(keys, labels, _num(v), int(n) if n is not None else None))
        return out


def series(metric: R.Metric, group_by: List[str], filters: Dict[str, List[str]],
           start: dt.date, end_inclusive: dt.date, scope: Scope) -> List[Row]:
    if scope.assigned_nothing:
        return []
    sql, params = build_query(metric, group_by, filters, start, end_inclusive + dt.timedelta(days=1), scope)
    return run_query(sql, params, len(group_by), bool(metric.assessed))


def unattributed_findings(start: dt.date, end_inclusive: dt.date, where: Optional[str] = None) -> int:
    """Admin-only data-quality figure: findings that map to no deal, so no Buyer can ever see them.
    ``where`` is the metric's own fixed predicate, so the count is over the same findings the tile counts."""
    from src.services.db import get_conn
    src = R.SOURCES["finding"]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {src.from_sql} WHERE {src.time_expr} >= %s AND {src.time_expr} < %s "
                    "AND o.deal_id IS NULL" + (f" AND ({where})" if where else ""), (start, end_inclusive + dt.timedelta(days=1)))
        return int(cur.fetchone()[0])


def build_values_query(source_key: str, dim: str, q: Optional[str], scope: Scope, limit: int) -> Tuple[str, Dict[str, Any]]:
    """The distinct (key, label) pairs of one dimension, for a filter picker. Scoped exactly like a tile:
    a Buyer is offered only suppliers, deals and so on that are theirs. The search text is a bound
    parameter, never part of the SQL."""
    src = R.SOURCES[source_key]
    key_sql, label_sql = src.dims[dim]
    params: Dict[str, Any] = {"lim": int(limit)}
    where = [f"{key_sql} IS NOT NULL"]
    if not scope.all_rows:
        params["buyers"] = list(scope.buyers)
        where.append(f"{src.scope_expr} = ANY(%(buyers)s)")
    if q:
        params["q"] = f"%{q}%"
        where.append(f"({label_sql})::text ILIKE %(q)s")
    sql = (f"SELECT {key_sql}::text AS k, {label_sql}::text AS l FROM {src.from_sql} WHERE {' AND '.join(where)} "
           f"GROUP BY {key_sql}, {label_sql} ORDER BY {label_sql} LIMIT %(lim)s")
    return sql, params


def values(source_key: str, dim: str, q: Optional[str], scope: Scope, limit: int = 50) -> List[Tuple[str, str]]:
    if scope.assigned_nothing:
        return []
    sql, params = build_values_query(source_key, dim, q, scope, limit)
    from src.services.db import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [(r[0], r[1]) for r in cur.fetchall()]
