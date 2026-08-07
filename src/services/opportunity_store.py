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
from src.services.facts.deprecation import read_calculation_detail

log = logging.getLogger(__name__)

_STAGES = ("identified", "negotiation", "agreed", "realised", "closed", "rejected")


def upsert_opportunity(cur, rec: dict) -> None:
    """Insert/update one mined finding. Preserves the existing stage unless the
    finding is rejected (is_rejected) — never demotes a progressed opportunity.

    Keyed on opportunity_ref_id, the content-derived identity
    (policy_detector_sourcehash_supplier_item). opportunity_id is a per-run counter
    the miner assigns while walking candidates, so it changes for the SAME finding
    between runs — keying on it meant every run inserted duplicates instead of
    updating, and a colliding id could overwrite an unrelated finding and inherit
    its lifecycle stage.
    """
    # Still written for one release, so the payload itself is kept for the
    # INSERT below -- but it is no longer the system of record. Reads go
    # through the shim, which prefers the structured column and logs every
    # JSONB fallback so the column can be retired on evidence, not assumption.
    calc = rec.get("calculation_details") or {}
    item_desc = (
        rec.get("item_description")
        or read_calculation_detail(rec, "item_description")
        or rec.get("item_id")
    )
    stage = "rejected" if rec.get("is_rejected") else "identified"
    # Pre-ref_id rows fall back to their own id so the identity is never blank.
    ref_id = str(rec.get("opportunity_ref_id") or rec.get("opportunity_id"))
    cur.execute(
        """
        insert into proc.bp_opportunity
          (opportunity_id, opportunity_ref_id, detector_type, policy_id, supplier_id,
           supplier_name, category_id, item_id, item_description, financial_impact_gbp,
           stage, ml_priority_score, weightage, calculation_details, source_records,
           detected_on, quote_id, po_id, invoice_id, deal_id)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        on conflict (opportunity_ref_id) do update set
          detector_type=excluded.detector_type, policy_id=excluded.policy_id,
          -- Re-detected: it is live again, whatever a previous run concluded.
          retired_at=null,
          supplier_id=excluded.supplier_id, supplier_name=excluded.supplier_name,
          category_id=excluded.category_id, item_id=excluded.item_id,
          item_description=excluded.item_description,
          financial_impact_gbp=excluded.financial_impact_gbp,
          ml_priority_score=excluded.ml_priority_score, weightage=excluded.weightage,
          calculation_details=excluded.calculation_details,
          source_records=excluded.source_records, detected_on=excluded.detected_on,
          quote_id=excluded.quote_id, po_id=excluded.po_id,
          invoice_id=excluded.invoice_id,
          -- a detector that already knows its deal_id wins; never blank out a
          -- value the quote-anchored backfill (opportunity_linkage.py) set earlier.
          deal_id=coalesce(excluded.deal_id, proc.bp_opportunity.deal_id),
          -- only force stage to 'rejected'; otherwise keep the progressed stage
          stage=case when excluded.stage='rejected' then 'rejected'
                     else proc.bp_opportunity.stage end,
          updated_at=now()
        """,
        (
            str(rec.get("opportunity_id")), ref_id,
            rec.get("detector_type"), rec.get("policy_id"), rec.get("supplier_id"),
            rec.get("supplier_name"), rec.get("category_id"), rec.get("item_id"),
            item_desc, rec.get("financial_impact_gbp"), stage,
            rec.get("ml_priority_score"), rec.get("weightage"),
            json.dumps(calc), json.dumps(rec.get("source_records") or []),
            rec.get("detected_on"),
            rec.get("quote_id") or read_calculation_detail(rec, "quote_id"),
            rec.get("po_id") or read_calculation_detail(rec, "po_id"),
            rec.get("invoice_id") or read_calculation_detail(rec, "invoice_id"),
            rec.get("deal_id") or read_calculation_detail(rec, "deal_id"),
        ),
    )


def retire_missing(cur, seen_ref_ids: Any, detector_types: Any,
                   min_impact: float = 0.0) -> int:
    """Close findings a full mining run no longer detects. Returns rows retired.

    Deliberately narrow, because this removes things from the user's screen with
    no human in the loop:

    * only rows still at ``identified`` — anything a human has progressed,
      realised, rejected or closed is never touched;
    * only detector types that produced findings in THIS run. A detector that
      returned nothing is indistinguishable from a detector that is broken (this
      corpus has no ``proc.contracts`` table at all, so the contract detectors
      always return empty), and clearing real findings because a detector failed
      would be worse than leaving a stale one;
    * only rows at or above this run's ``min_impact``. The threshold filters the
      run's own output, so a finding below it was never a candidate to be
      re-detected — retiring it would delete a live finding merely because
      someone ran an ad-hoc high-threshold scan;
    * never called at all unless the run covered the whole corpus — see
      ``_run_covers_whole_corpus`` in the miner.

    ``retired_at`` records that this was automatic. ``stage`` is set to ``closed``
    alongside it so every existing reader, which already excludes closed rows from
    open counts, needs no change.
    """
    detectors = [d for d in (detector_types or []) if d]
    if not detectors:
        return 0
    seen = [str(r) for r in (seen_ref_ids or []) if r]
    # `<> ALL(empty)` is true for every row, so an empty seen-list correctly
    # retires every identified finding for the detectors that ran and found none
    # of their previous ones — but `detectors` being non-empty means at least one
    # finding WAS produced, so this can never fire on a run that found nothing.
    cur.execute(
        """
        update proc.bp_opportunity
           set stage='closed', retired_at=now(), stage_updated_at=now(), updated_at=now()
         where stage='identified'
           and retired_at is null
           and detector_type = ANY(%s)
           and coalesce(financial_impact_gbp, 0) >= %s
           and opportunity_ref_id <> ALL(%s)
        """,
        (detectors, float(min_impact or 0.0), seen),
    )
    retired = cur.rowcount or 0
    if retired:
        log.info("retired %d opportunity finding(s) no longer detected", retired)
    return retired


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
