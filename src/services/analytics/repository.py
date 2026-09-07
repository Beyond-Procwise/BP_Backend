"""The rows an analytic answer is computed from, and the rates it converts at.

Read-only, fixed SQL with no interpolation of user text, bounded by an explicit
period. It reads ``_trgt`` — the tier the product reads everywhere else — and
follows the conventions in ``services/corpus_facts``, which is where the ask
path already goes for counted rows.

Two departures from that module, both deliberate:

  * The aggregate is **grouped by supplier and currency**, never pre-summed. A
    supplier's GBP and USD invoices have to stay apart, or conversion cannot
    follow rule 1 of the display-currency contract.
  * The aggregate is **not truncated**. ``corpus_facts`` caps its lists at ten
    rows on purpose, but a share computed against a ten-row sample would report
    the leading supplier holding a third of all spend when it holds a fraction
    of that. The denominators here are the population, so the population is what
    is read; only the presented table is cut to the top N, and that happens
    after the arithmetic.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, List, Optional, Tuple

from src.services.analytics.currency import DEFAULT_CURRENCY, NATIVE, DisplayCurrency
from src.services.analytics.period import Period
from src.services.analytics.supplier_spend import SupplierSpendRow

logger = logging.getLogger(__name__)

_SUPPLIER_SPEND_SQL = """
SELECT i.supplier_id::text            AS supplier_id,
       s.supplier_name                AS supplier_name,
       i.currency                     AS currency,
       SUM(i.invoice_amount)::numeric AS amount,
       COUNT(*)::int                  AS invoices
  FROM proc.bp_invoice_trgt i
  JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
 WHERE i.invoice_date >= %s
   AND i.invoice_date <= %s
   AND i.invoice_amount IS NOT NULL
 GROUP BY i.supplier_id, s.supplier_name, i.currency
"""

_POPULATION_SQL = """
SELECT COUNT(DISTINCT supplier_id)::int AS suppliers,
       COUNT(*)::int                    AS invoices
  FROM proc.bp_invoice_trgt
 WHERE invoice_date >= %s
   AND invoice_date <= %s
   AND invoice_amount IS NOT NULL
"""


def fetch_supplier_spend(cur, period: Period) -> List[SupplierSpendRow]:
    """Every supplier's spend in the period, split by the currency it was billed in."""
    cur.execute(_SUPPLIER_SPEND_SQL, (period.start, period.end))
    columns = [d[0] for d in cur.description]
    rows: List[SupplierSpendRow] = []
    for record in cur.fetchall():
        values = dict(zip(columns, record))
        rows.append(SupplierSpendRow(
            supplier_id=str(values["supplier_id"]),
            supplier_name=values["supplier_name"] or "Unnamed supplier",
            currency=(values["currency"] or "").upper(),
            amount=Decimal(str(values["amount"] or 0)),
            invoices=int(values["invoices"] or 0),
        ))
    return rows


def fetch_population(cur, period: Period) -> Tuple[int, int]:
    """How many suppliers and invoices the period holds — counted, never tallied."""
    cur.execute(_POPULATION_SQL, (period.start, period.end))
    record = cur.fetchone() or (0, 0)
    return int(record[0] or 0), int(record[1] or 0)


def display_currency_from_request(
    target: Optional[str] = None,
    manual_rates: Optional[dict] = None,
    rates_payload: Optional[dict] = None,
) -> DisplayCurrency:
    """The reader's on-screen currency selection, as the server sees it.

    ``rates_payload`` is the shape ``GET /fx/rates`` serves and the client
    controller consumes, so both halves convert from the same batch. When it is
    not supplied it is loaded from the same repository that endpoint uses.
    """
    if rates_payload is None:
        rates_payload = _load_rates()

    selection = (target or DEFAULT_CURRENCY).strip()
    if selection.lower() == NATIVE:
        selection = NATIVE
    else:
        selection = selection.upper() or DEFAULT_CURRENCY

    return DisplayCurrency(
        target=selection,
        rates=(rates_payload or {}).get("rates") or {},
        manual=manual_rates or {},
        fetched_at=(rates_payload or {}).get("fetched_at"),
        stale=bool((rates_payload or {}).get("stale")),
    )


def _load_rates() -> dict:
    """The live USD-quoted batch, or nothing — never an invented rate.

    Shares ``fx_rate_repo`` with ``GET /fx/rates``, so a converted figure in an
    answer and the same figure on the dashboard came from one batch.
    """
    try:
        from repositories import fx_rate_repo

        result = fx_rate_repo.get_or_refresh_rates()
    except Exception:
        logger.exception("analytics: exchange rates unavailable")
        return {}
    if not result:
        return {}
    return {
        "rates": result.get("rates") or {},
        "fetched_at": result.get("fetched_at"),
        "stale": bool(result.get("stale")),
    }
