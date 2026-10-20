"""Compliance "Invoice > PO variance": invoices billed beyond what their PO authorised.

This is not a second implementation of the rule. Triage already decides, against the
governed ``triage_tolerances`` policy, whether the invoices on a PO exceed it, and writes
an open ``cumulative_total`` finding when they do. This reads those findings, so the
Compliance screen and the Action Centre cannot disagree about the same invoice.

The £ figure is the leading £ amount of each finding's own ``delta`` (the same figure the
Action Centre and Value Found show). A finding triage could not convert (no FX rate at the
time) is still a case, but is left out of the money total and counted in
``excluded_no_gbp`` — never assumed 1:1 and never a fabricated 0.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from src.services.db import get_conn
from src.services.value_summary_service import parse_gbp_delta

RULE_ID = "cumulative_total"

_FINDINGS_SQL = """
SELECT finding_id, deal_id, delta
  FROM proc.bp_detection_finding
 WHERE rule_id = %s AND status = 'open'
   AND (%s::text IS NULL OR deal_id = %s)
"""

_INVOICED_POS_SQL = """
SELECT count(DISTINCT po_id)
  FROM proc.bp_invoice_trgt
 WHERE po_id IS NOT NULL AND TRIM(po_id) <> ''
   AND (%s::text IS NULL OR deal_id = %s)
"""


def _gbp_to_usd(value_gbp: Optional[float], rates: Optional[Mapping[str, Any]]) -> Optional[float]:
    if value_gbp is None:
        return None
    if value_gbp == 0:
        return 0
    try:
        gbp_per_usd = float((rates or {}).get("GBP"))
    except (TypeError, ValueError):
        return None
    return round(value_gbp / gbp_per_usd, 2) if gbp_per_usd > 0 else None


def summarise(findings: Iterable[dict], invoiced_pos: int,
              rates: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    cases = 0
    total = 0.0
    priced = 0
    for row in findings:
        cases += 1
        amount = parse_gbp_delta(row.get("delta"))
        if amount is not None:
            total += amount
            priced += 1
    value_gbp = round(total, 2) if (priced or not cases) else None
    return {
        "metric": "Invoice > PO Variance",
        "cases": cases,
        # None, not 0, when there are cases but none could be put in pounds.
        "value_at_risk_gbp": value_gbp,
        # The Compliance table's other rows are USD, so the same figure in USD. ``rates`` is
        # the USD-quoted batch (units per 1 USD), so 1 GBP = 1 / rates["GBP"] USD. No GBP
        # rate -> None, never an assumed parity.
        "value_at_risk_usd": _gbp_to_usd(value_gbp, rates),
        "excluded_no_gbp": cases - priced,
        "pct_of_invoiced_pos": round(cases * 100.0 / invoiced_pos, 2) if invoiced_pos else None,
        "source": "triage",
    }


def _tolerance() -> Optional[dict]:
    """The governed tolerance that decided these findings, for display. Never gates them."""
    try:
        from src.services.triage.tolerance import load_config, resolve_tolerance

        # Without ``source``: it names the policy table, which the output-safety filter withholds.
        return {k: v for k, v in resolve_tolerance(RULE_ID, load_config()).as_dict().items()
                if k != "source"}
    except Exception:
        return None


def _live_rates() -> Optional[Mapping[str, Any]]:
    try:
        from repositories import fx_rate_repo

        return (fx_rate_repo.get_or_refresh_rates() or {}).get("rates")
    except Exception:
        return None


def po_variance(deal_id: Optional[str] = None) -> dict[str, Any]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_FINDINGS_SQL, (RULE_ID, deal_id, deal_id))
        findings = [{"finding_id": r[0], "deal_id": r[1], "delta": r[2]} for r in cur.fetchall()]
        cur.execute(_INVOICED_POS_SQL, (deal_id, deal_id))
        invoiced_pos = int(cur.fetchone()[0] or 0)
    out = summarise(findings, invoiced_pos, _live_rates())
    out["tolerance"] = _tolerance()
    return out
