"""Persist mined opportunities into proc.bp_opportunity (stage-tracked) and keep
it in sync with opportunity_miner output + reject feedback.

The opportunity miner writes findings to opportunity_findings.json; the
Opportunities-page dashboard reads from the DB. ``sync_findings_from_json``
upserts the JSON into the table (idempotent), folding in reject feedback from
proc.opportunity_feedback. ``set_stage`` advances an opportunity through its
lifecycle (identified -> negotiation -> agreed -> realised / closed / rejected).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn
from src.services.facts.deprecation import read_calculation_detail
from src.services.lifecycle import IllegalTransition, refusal

log = logging.getLogger(__name__)

_STAGES = ("identified", "negotiation", "agreed", "realised", "closed", "rejected")


# Finished, with no way back to open: new evidence here raises a new row instead.
_HUMAN_FINAL = ("realised", "closed")


def _norm(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def subject_key(rec: dict) -> Optional[str]:
    """What an opportunity is ABOUT: the kind of saving, the supplier, and the item.

    Deliberately leaves out the supporting documents. opportunity_ref_id carries a
    fingerprint of them, so one more invoice against the same order made the same
    opportunity look new -- and it came back at 'identified' beside the row a person
    had already rejected.

    With no item the deal stands in for it (an overbilling finding is per deal, and
    one supplier has many deals); with neither, the category. A finding that names
    none of supplier, item, deal or category has no subject to speak of and returns
    None, which leaves it identified by opportunity_ref_id as before.

    The same expression backfills the column in
    deploy/sql/2026-10-13_opportunity_subject_review.sql; keep the two in step.
    """
    supplier = _norm(rec.get("supplier_id"))
    item = _norm(rec.get("item_id"))
    deal = _norm(rec.get("deal_id") or read_calculation_detail(rec, "deal_id"))
    category = _norm(rec.get("category_id"))
    if not (supplier or item or deal or category):
        return None
    if item:
        anchor = item
    elif deal:
        anchor = f"deal:{deal}"
    elif category:
        anchor = f"cat:{category}"
    else:
        anchor = ""
    return f"{_norm(rec.get('detector_type'))}|{supplier}|{anchor}".lower()


def _stored_id(rec: dict, ref_id: str) -> str:
    """The id a NEW row is stored under.

    The miner numbers its findings from 1 on every run, and opportunity_id is the
    table's primary key -- so a new finding numbered like an existing row was
    refused, and the refusal rolled back the whole run's save. A bare run counter is
    therefore replaced by an id derived from the reference, which is unique per row
    (not from the subject: a subject can have a finished row AND the fresh one raised
    after it). An id a detector chose itself (the duplicate-invoice detector's
    ``dupinv:...``) is already stable and is kept.
    """
    given = _norm(rec.get("opportunity_id"))
    if given and not given.isdigit():
        return given
    return "opp:" + hashlib.sha1(ref_id.encode("utf-8")).hexdigest()[:16]


def _as_list(value: Any) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_dict(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _pence(value: Any) -> Optional[float]:
    try:
        return round(float(value), 2) if value is not None else None
    except (TypeError, ValueError):
        return None


def evidence_change(old_sources: Any, old_impact: Any, new_sources: Any,
                    new_impact: Any, pending: Any = None) -> Optional[dict]:
    """What is new under an opportunity since it was last looked at, or None.

    New means a supporting record that was not there before, or a different figure.
    A record that merely dropped out, at the same figure, is not news.

    ``pending`` is the detail of a review already waiting. A second change before
    anyone has looked adds to it -- the person should see everything that arrived
    since they rejected, measured from the figure they rejected.
    """
    before = {str(s) for s in _as_list(old_sources)}
    added = sorted({str(s) for s in _as_list(new_sources)} - before)
    was, now = _pence(old_impact), _pence(new_impact)
    if not added and was == now:
        return None
    waiting = _as_dict(pending)
    if waiting:
        added = sorted(set(added) | {str(s) for s in _as_list(waiting.get("new_records"))})
        was = waiting.get("impact_before", was)
    return {"new_records": added, "impact_before": was, "impact_after": now}


def _find_existing(cur, key: Optional[str], ref_id: str):
    cols = ("opportunity_id, stage, source_records, financial_impact_gbp, "
            "review_detail, review_requested_at")
    if key:
        cur.execute(f"select {cols} from proc.bp_opportunity "
                    "where subject_key=%s for update", (key,))
        row = cur.fetchone()
        if row:
            return row
    # Rows from before subject_key, and findings with no subject at all.
    cur.execute(f"select {cols} from proc.bp_opportunity "
                "where opportunity_ref_id=%s for update", (ref_id,))
    return cur.fetchone()


def upsert_opportunity(cur, rec: dict) -> str:
    """Insert/update one mined finding; returns the id of the row it landed on.

    A finding lands on the row for its subject (``subject_key``), whatever documents
    support it today. What happens there depends on where a person left that row:

    * still open -- the figures and evidence are refreshed, the stage is kept;
    * rejected -- the row stays rejected. If the evidence has changed it is marked
      for review (``review_requested_at`` + what changed), so a person decides
      whether the rejection still stands. It is never silently re-raised;
    * realised or closed (by a person, or retired by an earlier run), and there are
      new supporting records -- that is a new occurrence, not the finished one
      coming back. The finished row keeps its history and gives up the subject; a
      fresh row is raised.

    A progressed opportunity is never demoted, and ``is_rejected`` only moves a row
    to 'rejected' where the lifecycle table allows it.
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
    key = subject_key(rec)
    sources = rec.get("source_records") or []
    impact = rec.get("financial_impact_gbp")
    fields = (
        rec.get("detector_type"), rec.get("policy_id"), rec.get("supplier_id"),
        rec.get("supplier_name"), rec.get("category_id"), rec.get("item_id"),
        item_desc, impact,
        rec.get("ml_priority_score"), rec.get("weightage"),
        json.dumps(calc), json.dumps(sources), rec.get("detected_on"),
        rec.get("quote_id") or read_calculation_detail(rec, "quote_id"),
        rec.get("po_id") or read_calculation_detail(rec, "po_id"),
        rec.get("invoice_id") or read_calculation_detail(rec, "invoice_id"),
        rec.get("deal_id") or read_calculation_detail(rec, "deal_id"),
    )

    existing = _find_existing(cur, key, ref_id)
    if existing is not None:
        row_id, row_stage, row_sources, row_impact, row_review, row_asked = existing
        change = evidence_change(row_sources, row_impact, sources, impact,
                                 pending=row_review if row_asked else None)
        if row_stage in _HUMAN_FINAL and change and change["new_records"]:
            cur.execute("update proc.bp_opportunity set subject_key=null, updated_at=now() "
                        "where opportunity_id=%s", (row_id,))
            existing = None

    if existing is None:
        new_id = _stored_id(rec, ref_id)
        cur.execute(
            """
            insert into proc.bp_opportunity
              (opportunity_id, opportunity_ref_id, subject_key, stage,
               detector_type, policy_id, supplier_id, supplier_name, category_id,
               item_id, item_description, financial_impact_gbp, ml_priority_score,
               weightage, calculation_details, source_records, detected_on,
               quote_id, po_id, invoice_id, deal_id)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (new_id, ref_id, key, stage) + fields,
        )
        return new_id

    review = change if (row_stage == "rejected" and change) else None
    cur.execute(
        """
        update proc.bp_opportunity set
          detector_type=%s, policy_id=%s, supplier_id=%s, supplier_name=%s,
          category_id=%s, item_id=%s, item_description=%s, financial_impact_gbp=%s,
          ml_priority_score=%s, weightage=%s, calculation_details=%s,
          source_records=%s, detected_on=%s, quote_id=%s, po_id=%s, invoice_id=%s,
          -- a detector that already knows its deal_id wins; never blank out a
          -- value the quote-anchored backfill (opportunity_linkage.py) set earlier.
          deal_id=coalesce(%s, deal_id),
          -- Re-detected: it is live again, whatever a previous run concluded.
          retired_at=null,
          subject_key=coalesce(%s, subject_key),
          -- only force stage to 'rejected', and only where that is a legal move:
          -- a realised/closed opportunity keeps its stage (the lifecycle trigger
          -- would otherwise refuse, aborting the whole sync)
          stage=case when %s and exists (
                         select 1 from proc.bp_lifecycle_transition t
                          where t.object_type='opportunity'
                            and t.from_state=proc.bp_opportunity.stage
                            and t.to_state='rejected')
                     then 'rejected'
                     else stage end,
          -- New evidence under a rejection asks a person to look again. The first
          -- request's time is kept: it is when the news arrived, not the latest run.
          review_requested_at=case when %s then coalesce(review_requested_at, now())
                                   else review_requested_at end,
          review_detail=case when %s then %s::jsonb else review_detail end,
          updated_at=now()
        where opportunity_id=%s
        """,
        fields + (key, stage == "rejected", review is not None, review is not None,
                  json.dumps(review) if review is not None else None, row_id),
    )
    return str(row_id)


def retire_missing(cur, seen_ids: Any, detector_types: Any,
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
    # Row ids, as returned by upsert_opportunity -- not opportunity_ref_id. A finding
    # whose evidence changed lands on its subject's existing row, whose reference is
    # the one it was first seen under; matching on the reference would retire the
    # very row the run had just refreshed.
    seen = [str(r) for r in (seen_ids or []) if r]
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
           and opportunity_id <> ALL(%s)
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
        try:
            cur.execute(
                "update proc.bp_opportunity set stage=%s, "
                "realised_savings_gbp=coalesce(%s, realised_savings_gbp), "
                # A person has now decided with the new evidence in front of them.
                "review_requested_at=null, "
                "stage_updated_at=now(), updated_at=now() where opportunity_id=%s",
                (stage, realised_savings, str(opportunity_id)))
        except Exception as exc:
            reason = refusal(exc)
            if reason:
                raise IllegalTransition(reason) from exc
            raise

    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                _run(own); own.commit()
            except Exception:
                own.rollback(); raise
    else:
        _run(conn)


_REVIEW_DECISIONS = {"reopen": "identified", "keep_rejected": "rejected"}


def resolve_review(opportunity_id: str, decision: str, decided_by: Optional[str] = None,
                   conn: Any = None) -> str:
    """A person's answer to "new information has arrived on something you rejected".

    ``reopen`` puts the opportunity back to 'identified'; ``keep_rejected`` leaves it
    rejected. Either way the request is cleared and the answer is kept beside what
    changed, so the next change is measured from the evidence this person saw.
    Returns the stage the opportunity is now at.
    """
    if decision not in _REVIEW_DECISIONS:
        raise ValueError(f"invalid decision {decision!r}; must be one of "
                         f"{tuple(_REVIEW_DECISIONS)}")
    stage = _REVIEW_DECISIONS[decision]

    def _run(c):
        cur = c.cursor()
        cur.execute("select stage from proc.bp_opportunity where opportunity_id=%s",
                    (str(opportunity_id),))
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"no opportunity {opportunity_id!r}")
        if row[0] != "rejected":
            raise IllegalTransition(
                f"opportunity {opportunity_id} is {row[0]}, not rejected; "
                "there is no rejection to review")
        set_stage(opportunity_id, stage, conn=c)
        cur.execute(
            "update proc.bp_opportunity set review_detail = "
            "coalesce(review_detail, '{}'::jsonb) || %s::jsonb where opportunity_id=%s",
            (json.dumps({"decision": decision, "decided_by": decided_by,
                         "decided_at": datetime.now(timezone.utc).isoformat()}),
             str(opportunity_id)))

    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                _run(own); own.commit()
            except Exception:
                own.rollback(); raise
    else:
        _run(conn)
    return stage


def _fold_in_rejections(cur) -> int:
    """Reject opportunities a person rejected in proc.opportunity_feedback.

    Only where rejection is a legal move: feedback does not unwind a realised saving,
    and the lifecycle trigger would refuse the whole statement if it tried.
    """
    cur.execute(
        "update proc.bp_opportunity o set stage='rejected', updated_at=now() "
        "from proc.opportunity_feedback f "
        "where f.opportunity_id=o.opportunity_id and f.status='rejected' "
        "and exists (select 1 from proc.bp_lifecycle_transition t "
        "where t.object_type='opportunity' and t.from_state=o.stage "
        "and t.to_state='rejected')")
    return cur.rowcount or 0


def sync_findings_from_json(path: str, conn: Any = None) -> dict:
    """Upsert all findings from a miner JSON file into proc.bp_opportunity, then
    fold in reject feedback. Returns {"synced": n, "rejected": m}. Idempotent."""
    with open(path) as fh:
        findings = json.load(fh)

    def _run(c):
        cur = c.cursor()
        for rec in findings:
            upsert_opportunity(cur, rec)
        rejected = 0
        try:
            rejected = _fold_in_rejections(cur)
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
