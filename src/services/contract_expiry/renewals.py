"""The renewals list: every contract the expiry rule has an opinion about.

Same evaluation as Today's brief (``detector.evaluate``), so the list and the
brief's card cannot disagree about what is expiring or in which bucket. The
summary IS the brief's signal, reused rather than recomputed.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from src.services.brief_signals_service import (
    _contract_name, _total_gbp, expiry_buckets_signal, get_rates, to_gbp,
)
from src.services.contract_expiry import detector
from src.services.contract_expiry.buckets import EXPIRED, NO_END_DATE
from src.services.db import get_conn


# The server's output-safety filter withholds any string that reads like an internal
# identifier, and NO_END_DATE does (measured 2026-10-05: it came back "[withheld]").
# The wire carries plain words; the stored alert keeps its own labels.
_WIRE_BUCKET = {EXPIRED: "expired", NO_END_DATE: "noEndDate"}


def _yes(flag: Any) -> bool:
    return str(flag or "").strip().lower() in ("yes", "true")


def _sort_key(item: Dict[str, Any]):
    """Soonest forward end date first; then already-expired, most recent first
    (the ones just missed matter more than the ones missed years ago); then the
    contracts with no end date. The screen filters by bucket, so this is only the
    order a person sees when nothing is filtered."""
    b = item["bucket"]
    group = 2 if b == _WIRE_BUCKET[NO_END_DATE] else 1 if b == _WIRE_BUCKET[EXPIRED] else 0
    days = item["daysToEnd"]
    return (group, (-days if group == 1 else (days or 0)), item["contractId"])


def renewals_payload(evaluation: Dict[str, Any], today: date, rates: Optional[dict]) -> Dict[str, Any]:
    contracts = evaluation["contracts"]
    items: List[Dict[str, Any]] = []
    for d in evaluation["desired"]:
        c = contracts[d.contract_id]
        value = c.get("total_contract_value")
        gbp = None
        if value is not None and rates:
            gbp, _ = to_gbp(float(value), c.get("currency"), rates)
        items.append({
            "contractId": d.contract_id,
            "title": _contract_name(c),
            "supplier": c.get("supplier_name"),
            "category": c.get("spend_category"),
            "endDate": d.end_date.isoformat() if d.end_date else None,
            "daysToEnd": d.days_to_end,
            "bucket": _WIRE_BUCKET.get(d.bucket, d.bucket),
            # 'inDemand' = an active demand item already covers it (the alert is held back)
            "state": "inDemand" if d.suppressed_by else "open",
            "demandId": d.suppressed_by,
            "autoRenew": _yes(c.get("auto_renew_flag")),
            "renewalTerm": c.get("renewal_term"),
            "value": float(value) if value is not None else None,
            "currency": c.get("currency"),
            "valueGbp": round(gbp, 2) if gbp is not None else None,
        })
    items.sort(key=_sort_key)
    return {
        "asOf": today.isoformat(),
        "summary": expiry_buckets_signal(evaluation, today, rates),
        "total": len(items),
        "items": items,
    }


def build(today: Optional[date] = None) -> Dict[str, Any]:
    today = today or date.today()
    rates = get_rates()
    with get_conn() as conn:
        evaluation = detector.evaluate(conn.cursor(), today, with_supplier=True)
    return renewals_payload(evaluation, today, rates)


_ACTIVE_CONTRACTS_SQL = """
    SELECT contract_id, contract_title, spend_category, contract_end_date,
           total_contract_value, currency, auto_renew_flag, cost_centre_id
      FROM proc.bp_contract_master
     WHERE COALESCE(contract_lifecycle_status, '') ILIKE 'active'
       AND contract_end_date >= %s
     ORDER BY contract_end_date, contract_id
"""


def active_contracts(today: Optional[date] = None) -> List[Dict[str, Any]]:
    """The live contracts the demand intake can ask "is this a renewal of that?" about.

    Real rows only, with no supplier name: the contract table carries a supplier id
    that points at a register outside both databases, and the intake already refuses
    to name a supplier it cannot know. Ending today still counts as live, which is
    why the comparison is >= and not >.
    """
    today = today or date.today()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_ACTIVE_CONTRACTS_SQL, (today,))
        cols = [c[0] for c in cur.description]
        out = []
        for row in cur.fetchall():
            r = dict(zip(cols, row))
            r["contract_end_date"] = r["contract_end_date"].isoformat()
            if r["total_contract_value"] is not None:
                r["total_contract_value"] = float(r["total_contract_value"])
            out.append(r)
    return out
