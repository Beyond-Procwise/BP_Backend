"""build_spend_cube: invoiced spend cut by one dimension, with the dashboards' currency rules.

Same table (``bp_invoice_trgt``), same measure (``invoice_amount``) and same conversion contract as the
SpendIQ supplier-spend answers, so a total here equals the total on screen. Only the dimensions the
invoices actually carry are offered; there is no category, and the tool says so rather than guess one.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.services.analytics.currency import DEFAULT_CURRENCY, DisplayCurrency
from src.services.analytics.period import Period, fiscal_year_to_date

log = logging.getLogger(__name__)

# The only SQL fragments that ever reach the query; a caller picks a key, never writes SQL.
DIMENSIONS: Dict[str, str] = {
    "supplier": "COALESCE(s.supplier_name, i.supplier_id::text)",
    "month": "to_char(i.invoice_date, 'YYYY-MM')",
    "currency": "i.currency",
    "country": "i.country",
    "region": "i.region",
    "buyer": "i.buyer_id::text",
    "payment_terms": "i.payment_terms",
    "status": "i.invoice_status",
}
NOT_RECORDED = "(not recorded)"
DEFAULT_TOP_N = 15
MAX_TOP_N = 50

_SQL = """
SELECT {dim} AS k, UPPER(COALESCE(i.currency, '')) AS currency,
       SUM(i.invoice_amount)::numeric AS amount, COUNT(*)::int AS invoices
  FROM proc.bp_invoice_trgt i
  LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
 WHERE i.invoice_date >= %s AND i.invoice_date <= %s AND i.invoice_amount IS NOT NULL
 GROUP BY 1, 2
"""


def sql_for(by: str) -> str:
    return _SQL.format(dim=DIMENSIONS[by])


def _fetch(by: str, period: Period) -> List[Tuple[Any, str, Any, int]]:
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql_for(by), (period.start, period.end))
        return list(cur.fetchall())


def _day(value: Any, default: date, name: str) -> date:
    if value is None or value == "":
        return default
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{name} must be a date like 2026-01-31.") from None


def build(by: Any = None, start: Any = None, end: Any = None, currency: Any = None, top_n: Any = None, *,
          fetch: Optional[Callable[[str, Period], Sequence[tuple]]] = None,
          display: Optional[DisplayCurrency] = None, today: Optional[date] = None) -> Dict[str, Any]:
    if by not in DIMENSIONS:
        return {"found": False, "note": f"by must be one of: {', '.join(DIMENSIONS)}. Invoices carry no category, "
                                        "so spend cannot be cut by category here."}
    default = fiscal_year_to_date(today or date.today())
    try:
        period = Period(_day(start, default.start, "start"), _day(end, default.end, "end"),
                        default.label if start is None and end is None else "custom")
    except ValueError as exc:
        return {"found": False, "note": str(exc)}
    if period.start > period.end:
        return {"found": False, "note": "start is after end."}
    try:
        limit = max(1, min(int(top_n), MAX_TOP_N)) if top_n is not None else DEFAULT_TOP_N
    except (TypeError, ValueError):
        limit = DEFAULT_TOP_N
    try:
        rows = list((fetch or _fetch)(by, period))
        if display is None:
            from src.services.analytics.repository import display_currency_from_request
            display = display_currency_from_request(str(currency or DEFAULT_CURRENCY))
    except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage, not "no spend"
        log.warning("build_spend_cube: unreadable: %s", exc)
        return {"found": False, "available": False,
                "note": "Spend could not be read just now. This does not mean there was no spend."}

    cells: Dict[str, Dict[str, Any]] = {}
    invoices = 0
    for key, ccy, amount, n in rows:
        cell = cells.setdefault(str(key) if key not in (None, "") else NOT_RECORDED,
                                {"by_currency": {}, "invoices": 0})
        cell["by_currency"][ccy or "unknown"] = cell["by_currency"].get(ccy or "unknown", Decimal(0)) + Decimal(str(amount or 0))
        cell["invoices"] += int(n or 0)
        invoices += int(n or 0)
    if not cells:
        return {"found": True, "by": by, "period": {"label": period.label, "start": str(period.start), "end": str(period.end)},
                "cells": [], "invoices": 0, "note": "No invoices in this period."}

    out_cells, excluded = [], []
    total = Decimal(0)
    for key, cell in cells.items():
        result = display.total([(a, c) for c, a in cell["by_currency"].items()])
        for code in result.excluded_currencies:
            if code not in excluded:
                excluded.append(code)
        if result.value is not None:
            total += result.value
        out_cells.append({by: key, "amount": None if result.value is None else float(round(result.value, 2)),
                          "invoices": cell["invoices"],
                          "as_billed": {c: float(round(a, 2)) for c, a in sorted(cell["by_currency"].items())}})
    out_cells.sort(key=lambda c: (c["amount"] is None, -(c["amount"] or 0), str(c[by])))
    for c in out_cells:
        c["share_pct"] = None if c["amount"] is None or total == 0 else float(round(Decimal(str(c["amount"])) / total * 100, 1))
    shown = out_cells[:limit]
    note = f"Invoiced amounts as stored, converted once per currency at {display.rate_note()}."
    if excluded:
        note += (f" No rate for {', '.join(excluded)}: those amounts are left out of the converted figures and "
                 "shown as billed.")
    if len(out_cells) > limit:
        note += f" Showing the top {limit} of {len(out_cells)}; the total covers all."
    return {"found": True, "by": by, "currency": display.target,
            "period": {"label": period.label, "start": str(period.start), "end": str(period.end)},
            "total": float(round(total, 2)), "invoices": invoices, "cell_count": len(out_cells),
            "cells": shown, "excluded_currencies": excluded, "note": note}
