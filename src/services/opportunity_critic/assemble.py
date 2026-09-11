"""Build the critic's input envelope from a finding and the corpus.

Assembly, never judgement. What cannot be found is reported absent; nothing is
inferred, and nothing is fuzzy-matched across key spaces.

Two facts about this deployment shape the code (spec section 6):

  * Opportunity suppliers are name-derived slugs (SUP-MeridianSystems12);
    contract suppliers are coded ids (S9251). Zero of 308 opportunities resolve.
    The honest output is SUPPLIER_NOT_IN_CONTRACT_MASTER, not a guess.
  * There is no market index of any kind. index_pct is None and index_source is
    NONE_AVAILABLE -- stated, so the critic can raise it as a blocking gap
    rather than silently skipping the inflation test.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_FAMILY_BY_DETECTOR = {
    "Duplicate Invoice Recovery": "integrity",
    "Invoice Overbilling": "integrity",
    "Price Benchmark Variance": "position",
}

_CONTRACT_SQL = """
    SELECT contract_id, contract_start_date, contract_end_date, currency,
           jurisdiction, payment_terms, parent_contract_id
      FROM proc.bp_contract_master
     WHERE supplier_id = %s
     ORDER BY contract_start_date DESC NULLS LAST
     LIMIT 1
"""


def _confidence_for(facts_state: Optional[str]) -> str:
    """Map the finding's facts_state onto the evidence ladder.

    RESOLVED is ASSERTED, never CORROBORATED: it means numbers were parsed out
    of a document, not that a second source agreed with them. Upgrading here
    would let a parse masquerade as corroboration.
    """
    if str(facts_state or "").upper() == "RESOLVED":
        return "ASSERTED"
    return "UNASSESSED"


def _contract_context(supplier_id: Optional[str], conn) -> tuple[Dict[str, Any], str]:
    if not supplier_id or conn is None:
        return {}, "SUPPLIER_NOT_IN_CONTRACT_MASTER"
    try:
        cur = conn.cursor()
        cur.execute(_CONTRACT_SQL, (supplier_id,))
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - an unreadable corpus is a gap, not a crash
        logger.error("contract lookup failed for %s: %s", supplier_id, exc)
        return {}, "CONTRACT_LOOKUP_FAILED"

    if not row:
        return {}, "SUPPLIER_NOT_IN_CONTRACT_MASTER"

    return (
        {
            "contract_id": row[0],
            "start_date": row[1],
            "end_date": row[2],
            "currency": row[3],
            "jurisdiction": row[4],
            "payment_terms": row[5],
            "parent_contract_id": row[6],
            # Not columns anywhere in bp_contract_master. Stated as absent so
            # the critic raises them as gaps rather than treating silence as
            # "no escalation clause exists".
            "escalation_clause": None,
            "break_clause": None,
            "benchmarking_clause": None,
        },
        "RESOLVED",
    )


def assemble_candidate(finding: Dict[str, Any], conn) -> Dict[str, Any]:
    """Return the critic's input envelope for one finding."""
    calc = finding.get("calculation_details") or {}
    detector = finding.get("detector_type")
    supplier_id = finding.get("supplier_id")

    contract_context, resolution = _contract_context(supplier_id, conn)

    anchor_value = calc.get("benchmark_price")
    current_value = calc.get("actual_price")
    quantity = calc.get("quantity")

    return {
        "finding_id": finding.get("opportunity_ref_id"),
        "detector_id": detector,
        "detector_family": _FAMILY_BY_DETECTOR.get(detector, "unknown"),
        "claim": finding.get("claim"),
        "anchor": {
            "value": anchor_value,
            # The miner's anchor is grouped.groupby("item_id")["avg_price"].min()
            # -- the cheapest average any supplier ever charged, undated and
            # unnormalised. Naming it lets the critic test it properly.
            "kind": "cheapest_observed" if detector == "Price Benchmark Variance" else None,
            "date": None,
            "basis": None,
            "quantity": None,
            "line_value": None,
            "unit_price": anchor_value,
        },
        "current": {
            "value": current_value,
            "date": None,
            "basis": None,
            "quantity": quantity,
        },
        "delta": {
            "gap": (current_value - anchor_value)
            if isinstance(anchor_value, (int, float))
            and isinstance(current_value, (int, float))
            else None,
            "attributed_value": finding.get("financial_impact_gbp"),
            "currency": finding.get("currency"),
        },
        "evidence": {
            "source_records": list(finding.get("source_records") or []),
            "current_confidence": _confidence_for(finding.get("facts_state")),
            "anchor_confidence": "UNASSESSED",
            "contract_resolution": resolution,
            "facts_state": finding.get("facts_state"),
        },
        "contract_context": contract_context,
        "category_context": {
            "category_id": finding.get("category_id"),
            "index_pct": None,
            "index_source": "NONE_AVAILABLE",
        },
    }
