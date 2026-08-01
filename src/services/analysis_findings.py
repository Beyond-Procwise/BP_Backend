"""Builds the point-in-time findings snapshot stored on proc.bp_analysis.

Five sources, all read straight out of proc.* — no HTTP call out to the Node
gateway, which the listener thread cannot rely on reaching.

Three of the five sources (deals, opportunities, summaries) are keyed on the
deal_ids this analysis produced. The fourth, discrepancies, is different:
proc.bp_extraction_discrepancy has no deal_id column, so it is scoped by the
FILE PATHS this analysis actually read instead (source_file, which shares the
same S3-key format as proc.session_document_outcome.file_path and joins
against it exactly). That is also the more correct meaning for a frozen
analysis — it reports the problems found in ITS OWN documents — and it means
discrepancies still populate for a one-off analysis that formed no deal at
all. The fifth, benchmarks, stays keyed on deal_ids.

The portfolio compliance measures are deliberately NOT captured: they are
workspace-wide percentages that change for reasons unrelated to this upload.
See §5 of the spec.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

SOURCES = ("deals", "discrepancies", "opportunities", "summaries", "benchmarks")

# Verified against live bp_sqldb.
_DEALS = """
SELECT deal_id, deal_name, supplier_name, supplier_id, deal_date, currency,
       quote_count, po_count, invoice_count,
       quote_total, po_total, invoice_total,
       three_way_match, price_variance_pct, cycle_days_quote_to_po
  FROM proc.bp_deal_overview WHERE deal_id = ANY(%s)
"""

# proc.bp_extraction_discrepancy has NO deal_id column, so discrepancies are
# scoped by the file paths this analysis actually read instead. source_file
# and proc.session_document_outcome.file_path share the identical S3-key
# format ('documents/invoice/ORB-INV-9901_INVOICE.xlsx') and join exactly.
_DISCREPANCIES = """
SELECT discrepancy_id, doc_type, doc_pk_candidate, source_file, field_name,
       issue_type, severity, status, notes, blocks_promotion,
       resolution_outcome, recovered_amount
  FROM proc.bp_extraction_discrepancy WHERE source_file = ANY(%s)
"""

# proc.bp_opportunity has no title/opportunity_type/confidence columns.
_OPPORTUNITIES = """
SELECT opportunity_id, deal_id, detector_type, item_description, supplier_name,
       stage, financial_impact_gbp, realised_savings_gbp, ml_priority_score,
       detected_on
  FROM proc.bp_opportunity WHERE deal_id = ANY(%s)
"""

# Verified against live bp_sqldb.
_SUMMARIES = """
SELECT deal_id, summary, deal_value, currency, volume, unit_price,
       price_change_pct, volume_change_pct, efficiency_score, item_count
  FROM proc.bp_analysis_summary WHERE deal_id = ANY(%s) AND is_current
"""


def _rows(cur: Any, sql: str, params: list) -> list[dict]:
    cur.execute(sql, (params,))
    cols = [c.name for c in (cur.description or [])]
    return [dict(zip(cols, r)) for r in (cur.fetchall() or [])]


def _benchmark_for_deal(deal_id: str) -> Any:
    """Import lazily: the router module pulls in the FastAPI app graph, which
    the listener thread has no other reason to load."""
    from src.api.routers.benchmark import benchmark_by_deal

    # benchmark_by_deal is a FastAPI route handler: its min_points/method
    # parameters default to fastapi.params.Query sentinel objects, not plain
    # values, when the function is called directly outside of FastAPI's
    # dependency injection. Passing them explicitly avoids that trap.
    return benchmark_by_deal(deal_id, min_points=3, method="weighted")


def capture(deal_ids: list, *, file_paths: Optional[list] = None,
            conn: Optional[Any] = None) -> dict:
    """Snapshot everything this analysis found, per source.

    Each source is fetched independently so one failure cannot empty the
    others, and its status is recorded — an errored source reads as 'error',
    never as an empty result.

    deals, opportunities and summaries are keyed on deal_ids and are skipped
    (left as [], source 'ok') when deal_ids is empty. discrepancies is keyed
    on file_paths instead and is skipped the same way when file_paths is
    empty — a one-off analysis that formed no deal can still have documents,
    and its discrepancies still belong in the snapshot.
    """
    ids = [str(d) for d in (deal_ids or [])]
    paths = [str(p) for p in (file_paths or [])]
    out: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "sources": {name: "ok" for name in SOURCES},
    }
    for name in SOURCES:
        out[name] = []

    def _run(c: Any) -> None:
        cur = c.cursor()
        for name, sql in (("deals", _DEALS),
                          ("opportunities", _OPPORTUNITIES),
                          ("summaries", _SUMMARIES)):
            if not ids:
                continue
            try:
                out[name] = _rows(cur, sql, ids)
            except Exception:
                log.exception("findings capture failed for source=%s", name)
                out["sources"][name] = "error"

        if paths:
            try:
                out["discrepancies"] = _rows(cur, _DISCREPANCIES, paths)
            except Exception:
                log.exception("findings capture failed for source=discrepancies")
                out["sources"]["discrepancies"] = "error"

        for deal_id in ids:
            try:
                out["benchmarks"].append(
                    {"deal_id": deal_id, "benchmark": _benchmark_for_deal(deal_id)})
            except Exception:
                log.exception("benchmark capture failed for deal=%s", deal_id)
                out["sources"]["benchmarks"] = "error"

    if conn is not None:
        _run(conn)
    else:
        with get_conn() as own:
            _run(own)
    return out


def headline(findings: dict) -> tuple:
    """(value_found, currency) for the list view and the version delta.

    None rather than 0 when there is nothing to add up: zero means 'we looked
    and found nothing', None means 'there is no figure'. They are different
    facts and the list view renders them differently.
    """
    total = 0.0
    seen = False
    for opp in (findings.get("opportunities") or []):
        v = opp.get("financial_impact_gbp")
        if v is not None:
            total += float(v)
            seen = True
    for disc in (findings.get("discrepancies") or []):
        v = disc.get("recovered_amount")
        if v is not None:
            total += float(v)
            seen = True
    if not seen:
        return (None, None)
    currency = next((d.get("currency") for d in (findings.get("deals") or [])
                     if d.get("currency")), "GBP")
    return (round(total, 2), currency)
