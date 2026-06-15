"""Persist mined opportunities into proc.bp_opportunity (stage-tracked) and keep
it in sync with opportunity_miner output + reject feedback.

The opportunity miner writes findings to opportunity_findings.json; the
Opportunities-page dashboard reads from the DB. ``sync_findings_from_json``
upserts the JSON into the table (idempotent), folding in reject feedback from
proc.opportunity_feedback. ``set_stage`` advances an opportunity through its
lifecycle (identified -> negotiation -> agreed -> realised / closed / rejected).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

_STAGES = ("identified", "negotiation", "agreed", "realised", "closed", "rejected")


def upsert_opportunity(cur, rec: dict) -> None:
    """Insert/update one mined finding. Preserves the existing stage unless the
    finding is rejected (is_rejected) — never demotes a progressed opportunity."""
    calc = rec.get("calculation_details") or {}
    item_desc = rec.get("item_description") or calc.get("item_description") or rec.get("item_id")
    stage = "rejected" if rec.get("is_rejected") else "identified"
    cur.execute(
        """
        insert into proc.bp_opportunity
          (opportunity_id, opportunity_ref_id, detector_type, policy_id, supplier_id,
           supplier_name, category_id, item_id, item_description, financial_impact_gbp,
           stage, ml_priority_score, weightage, calculation_details, source_records,
           detected_on, quote_id, po_id)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        on conflict (opportunity_id) do update set
          opportunity_ref_id=excluded.opportunity_ref_id,
          detector_type=excluded.detector_type, policy_id=excluded.policy_id,
          supplier_id=excluded.supplier_id, supplier_name=excluded.supplier_name,
          category_id=excluded.category_id, item_id=excluded.item_id,
          item_description=excluded.item_description,
          financial_impact_gbp=excluded.financial_impact_gbp,
          ml_priority_score=excluded.ml_priority_score, weightage=excluded.weightage,
          calculation_details=excluded.calculation_details,
          source_records=excluded.source_records, detected_on=excluded.detected_on,
          quote_id=excluded.quote_id, po_id=excluded.po_id,
          -- only force stage to 'rejected'; otherwise keep the progressed stage
          stage=case when excluded.stage='rejected' then 'rejected'
                     else proc.bp_opportunity.stage end,
          updated_at=now()
        """,
        (
            str(rec.get("opportunity_id")), rec.get("opportunity_ref_id"),
            rec.get("detector_type"), rec.get("policy_id"), rec.get("supplier_id"),
            rec.get("supplier_name"), rec.get("category_id"), rec.get("item_id"),
            item_desc, rec.get("financial_impact_gbp"), stage,
            rec.get("ml_priority_score"), rec.get("weightage"),
            json.dumps(calc), json.dumps(rec.get("source_records") or []),
            rec.get("detected_on"),
            rec.get("quote_id") or calc.get("quote_id"),
            rec.get("po_id") or calc.get("po_id"),
        ),
    )


def set_stage(opportunity_id: str, stage: str, realised_savings: Optional[float] = None,
              conn: Any = None) -> None:
    """Advance an opportunity to a new lifecycle stage."""
    if stage not in _STAGES:
        raise ValueError(f"invalid stage {stage!r}; must be one of {_STAGES}")

    def _run(c):
        cur = c.cursor()
        cur.execute(
            "update proc.bp_opportunity set stage=%s, "
            "realised_savings_gbp=coalesce(%s, realised_savings_gbp), "
            "stage_updated_at=now(), updated_at=now() where opportunity_id=%s",
            (stage, realised_savings, str(opportunity_id)))

    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                _run(own); own.commit()
            except Exception:
                own.rollback(); raise
    else:
        _run(conn)


def sync_findings_from_json(path: str, conn: Any = None) -> dict:
    """Upsert all findings from a miner JSON file into proc.bp_opportunity, then
    fold in reject feedback. Returns {"synced": n, "rejected": m}. Idempotent."""
    with open(path) as fh:
        findings = json.load(fh)

    def _run(c):
        cur = c.cursor()
        for rec in findings:
            upsert_opportunity(cur, rec)
        # fold in reject feedback (opportunity_feedback.status='rejected')
        rejected = 0
        try:
            cur.execute(
                "update proc.bp_opportunity o set stage='rejected', updated_at=now() "
                "from proc.opportunity_feedback f "
                "where f.opportunity_id=o.opportunity_id and f.status='rejected' "
                "and o.stage<>'rejected'")
            rejected = cur.rowcount or 0
        except Exception:  # feedback table optional
            log.debug("opportunity_feedback fold-in skipped", exc_info=True)
        return {"synced": len(findings), "rejected": rejected}

    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                r = _run(own); own.commit(); return r
            except Exception:
                own.rollback(); raise
    return _run(conn)
