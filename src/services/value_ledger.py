"""The value ledger: money a finding or opportunity actually produced.

Pure rules first (validated, converted, derived) so they unit-test on plain values; the
SQL layer below them is thin. Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md
"""
from __future__ import annotations

import logging
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional

log = logging.getLogger(__name__)

OUTCOME_TYPES = frozenset({"avoided", "claimed", "recovered", "claim_dropped",
                           "realised_saving", "terms_improved", "cycle_time"})
# What a person may say when closing an open money finding. `accepted` closes it and
# records no row: no money moved.
FINDING_OPEN_OUTCOMES = ("avoided", "claimed", "accepted")
SETTLE_OUTCOMES = ("recovered", "claim_dropped")
SETTLED_STATES = frozenset({"avoided", "recovered", "claim_dropped", "realised_saving"})

_CCY = re.compile(r"^[A-Z]{3}$")
_PENNY = Decimal("0.01")


class LedgerError(ValueError):
    """A request the ledger refuses. ``code`` is what the route returns to the UI."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_outcome(outcome_type: str, amount: Any, currency: Optional[str],
                     evidence_ref: Optional[str]) -> tuple[Optional[Decimal], Optional[str]]:
    if outcome_type not in OUTCOME_TYPES:
        raise LedgerError("invalid_outcome", f"unknown outcome {outcome_type!r}")
    if outcome_type == "claim_dropped":
        return None, None
    try:
        value = Decimal(str(amount).strip())
    except (InvalidOperation, AttributeError):
        raise LedgerError("invalid_amount", f"amount {amount!r} is not a number")
    if not value.is_finite() or value <= 0:
        raise LedgerError("invalid_amount", "amount must be greater than zero")
    value = value.quantize(_PENNY, rounding=ROUND_HALF_UP)
    ccy = str(currency or "").strip().upper()
    if outcome_type != "cycle_time" and not _CCY.match(ccy):
        raise LedgerError("invalid_currency", f"currency {currency!r} is not a 3-letter code")
    if outcome_type == "recovered" and not str(evidence_ref or "").strip():
        raise LedgerError("evidence_required",
                          "a recovered amount needs its credit note or document reference")
    return value, (ccy or None)


def convert_to_gbp(amount: Decimal, currency: str, rates: Optional[dict]) -> dict:
    """GBP at record time, with the rate that produced it. No rate -> None, never a guess."""
    if currency == "GBP":
        return {"amount_gbp": amount, "fx_rate": None, "fx_as_of": None}
    if not rates or currency not in rates or "GBP" not in rates:
        return {"amount_gbp": None, "fx_rate": None, "fx_as_of": None}
    rate = Decimal(str(rates["GBP"])) / Decimal(str(rates[currency]))
    return {"amount_gbp": (amount * rate).quantize(_PENNY, rounding=ROUND_HALF_UP),
            "fx_rate": rate, "fx_as_of": rates.get("_fetched_at")}


def current_state(rows: list[dict]) -> Optional[dict]:
    """The latest row for one source that no other row supersedes."""
    replaced = {r.get("supersedes_id") for r in rows if r.get("supersedes_id")}
    live = [r for r in rows if r["outcome_id"] not in replaced]
    if not live:
        return None
    return max(live, key=lambda r: (r["recorded_at"], r["outcome_id"]))


# --------------------------------------------------------------------------
# SQL layer. Every writer takes ``conn``: given one, the caller owns the transaction;
# without one, a private connection is opened with autocommit OFF (get_conn() is
# AUTOCOMMIT, on which rollback is a no-op) and committed or rolled back here.
# --------------------------------------------------------------------------
from src.services.agent_actions import record_action_or_fail  # noqa: E402
from src.services.db import get_conn  # noqa: E402
from src.services.lifecycle import IllegalTransition, refusal  # noqa: E402

_INSERT = """
INSERT INTO proc.bp_value_outcome
    (source_type, source_id, outcome_type, amount, currency, amount_gbp, fx_rate, fx_as_of,
     evidence_ref, note, supersedes_id, recorded_by, valid_from)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, coalesce(%s::date, current_date))
RETURNING outcome_id
"""
_HISTORY = """
SELECT outcome_id, outcome_type, amount, currency, amount_gbp, evidence_ref, note,
       supersedes_id, recorded_by, valid_from, recorded_at
  FROM proc.bp_value_outcome
 WHERE source_type = %s AND source_id = %s
 ORDER BY recorded_at, outcome_id
"""


def _in_tx(conn, fn):
    if conn is not None:
        return fn(conn)
    with get_conn() as own:
        own.autocommit = False
        try:
            result = fn(own)
            own.commit()
            return result
        except Exception:
            own.rollback()
            raise


def _rows(cur, sql, params) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _rates() -> Optional[dict]:
    from src.services import value_summary_service
    return value_summary_service._get_rates()


def _write(cur, conn, *, source_type, source_id, outcome_type, amount, currency, actor,
           evidence_ref=None, note=None, valid_from=None, supersedes_id=None) -> dict:
    # Controller ruling R1: only call _rates() (which can trigger a live external FX
    # fetch) when the currency actually needs converting. GBP needs no rate at all.
    if amount is None or not currency:
        fx = {"amount_gbp": None, "fx_rate": None, "fx_as_of": None}
    elif currency == "GBP":
        fx = convert_to_gbp(amount, "GBP", None)
    else:
        fx = convert_to_gbp(amount, currency, _rates())
    cur.execute(_INSERT, (source_type, str(source_id), outcome_type, amount, currency,
                          fx["amount_gbp"], fx["fx_rate"], fx["fx_as_of"],
                          (evidence_ref or None), (note or None), supersedes_id, actor,
                          valid_from))
    outcome_id = cur.fetchone()[0]
    record_action_or_fail(
        phase="value", action_type="value.outcome_recorded", conn=conn,
        doc_type=source_type, doc_pk=str(source_id), agent=actor, status=outcome_type,
        summary=f"{outcome_type} {amount or ''} {currency or ''}".strip(),
        details={"outcome_id": outcome_id, "amount_gbp": fx["amount_gbp"],
                 "supersedes_id": supersedes_id})
    return {"outcome_id": outcome_id, "state": outcome_type, "amount_gbp": fx["amount_gbp"]}


def _lock_finding(cur, discrepancy_id: int) -> dict:
    rows = _rows(cur, "SELECT discrepancy_id, status, issue_type FROM "
                      "proc.bp_extraction_discrepancy WHERE discrepancy_id = %s FOR UPDATE",
                 (int(discrepancy_id),))
    if not rows:
        raise LedgerError("not_found", f"finding {discrepancy_id} does not exist")
    return rows[0]


def _money_types() -> tuple:
    from src.services import value_summary_service
    return value_summary_service.DISCREPANCY_VALUE_TYPES


def record_finding_outcome(discrepancy_id: int, outcome: str, amount, currency, *,
                           actor: str, valid_from: Optional[str] = None,
                           note: Optional[str] = None, conn=None) -> dict:
    if outcome not in FINDING_OPEN_OUTCOMES:
        raise LedgerError("invalid_outcome", f"{outcome!r} cannot close a finding")

    def run(c):
        cur = c.cursor()
        row = _lock_finding(cur, discrepancy_id)
        if row["status"] != "open":
            raise LedgerError("finding_already_moved",
                              f"finding {discrepancy_id} is already {row['status']}")
        if outcome != "accepted" and row["issue_type"] not in _money_types():
            raise LedgerError("not_a_money_finding",
                              f"a {row['issue_type']} finding carries no recoverable amount")
        clean_amount, clean_ccy = (None, None) if outcome == "accepted" else \
            validate_outcome(outcome, amount, currency, None)
        if outcome != "accepted":
            # R18: a superseded finding's money is already counted under a stronger,
            # live finding (dedupe / R6 / line-under-PO). Stopping or claiming it here
            # as well would count the same money twice. Accepting records no money, so
            # it stays allowed.
            from src.services import value_summary_service
            live_id = value_summary_service.superseded_by_for(discrepancy_id, c)
            if live_id:
                raise LedgerError("superseded",
                                  f"counted under {live_id}; record it there")
        # R17: accepting the charge is the gateway's dismiss -- 'ignored' ("set aside"),
        # which Value found excludes and triage syncs as accepted_risk. Money stopped or
        # claimed is 'resolved'.
        new_status = "ignored" if outcome == "accepted" else "resolved"
        try:
            cur.execute("UPDATE proc.bp_extraction_discrepancy SET status = %s, "
                        "resolved_by = %s, resolved_at = now() WHERE discrepancy_id = %s",
                        (new_status, actor, int(discrepancy_id)))
        except Exception as exc:
            reason = refusal(exc)
            if reason:
                raise LedgerError("finding_already_moved", reason) from exc
            raise
        if outcome == "accepted":
            record_action_or_fail(phase="value", action_type="value.outcome_recorded",
                                  conn=c, doc_type="finding", doc_pk=str(discrepancy_id),
                                  agent=actor, status="accepted",
                                  summary="charge accepted; no money moved")
            return {"outcome_id": None, "state": "accepted", "amount_gbp": None}
        return _write(cur, c, source_type="finding", source_id=discrepancy_id,
                      outcome_type=outcome, amount=clean_amount, currency=clean_ccy,
                      actor=actor, note=note, valid_from=valid_from)

    return _in_tx(conn, run)


def settle_claim(discrepancy_id: int, outcome: str, amount=None, currency=None, *,
                 actor: str, evidence_ref: Optional[str] = None,
                 valid_from: Optional[str] = None, note: Optional[str] = None,
                 conn=None) -> dict:
    if outcome not in SETTLE_OUTCOMES:
        raise LedgerError("invalid_outcome", f"{outcome!r} cannot settle a claim")

    def run(c):
        cur = c.cursor()
        _lock_finding(cur, discrepancy_id)          # serialises settles on this finding
        state = current_state(_rows(cur, _HISTORY, ("finding", str(discrepancy_id))))
        if not state or state["outcome_type"] != "claimed":
            raise LedgerError("no_open_claim", f"finding {discrepancy_id} has no open claim")
        clean_amount, clean_ccy = validate_outcome(outcome, amount, currency, evidence_ref)
        return _write(cur, c, source_type="finding", source_id=discrepancy_id,
                      outcome_type=outcome, amount=clean_amount, currency=clean_ccy,
                      actor=actor, evidence_ref=evidence_ref, note=note, valid_from=valid_from)

    return _in_tx(conn, run)


def realise_opportunity(opportunity_id: str, amount, currency, *, actor: str,
                        valid_from: Optional[str] = None, evidence_ref: Optional[str] = None,
                        note: Optional[str] = None, conn=None) -> dict:
    from src.services.opportunity_store import set_stage
    clean_amount, clean_ccy = validate_outcome("realised_saving", amount, currency, None)

    def run(c):
        cur = c.cursor()
        cur.execute("SELECT stage FROM proc.bp_opportunity WHERE opportunity_id = %s FOR UPDATE",
                    (str(opportunity_id),))
        found = cur.fetchone()
        if not found:
            raise LedgerError("not_found", f"opportunity {opportunity_id} does not exist")
        try:
            set_stage(str(opportunity_id), "realised", conn=c)
        except IllegalTransition as exc:
            raise LedgerError("opportunity_already_moved", str(exc)) from exc
        return _write(cur, c, source_type="opportunity", source_id=opportunity_id,
                      outcome_type="realised_saving", amount=clean_amount, currency=clean_ccy,
                      actor=actor, evidence_ref=evidence_ref, note=note, valid_from=valid_from)

    return _in_tx(conn, run)


def correct_outcome(outcome_id: int, amount, currency, *, actor: str, note: str,
                    evidence_ref: Optional[str] = None, conn=None) -> dict:
    if not str(note or "").strip():
        raise LedgerError("note_required", "say why the figure is being corrected")

    def run(c):
        cur = c.cursor()
        rows = _rows(cur, "SELECT * FROM proc.bp_value_outcome WHERE outcome_id = %s",
                     (int(outcome_id),))
        if not rows:
            raise LedgerError("not_found", f"outcome {outcome_id} does not exist")
        old = rows[0]
        cur.execute("SELECT 1 FROM proc.bp_value_outcome WHERE supersedes_id = %s",
                    (int(outcome_id),))
        if cur.fetchone():
            raise LedgerError("already_corrected", f"outcome {outcome_id} was already corrected")
        # R4: lock the source row so a concurrent correction (or a settle/claim
        # racing this one) serialises against it, then refuse unless this outcome
        # is still the source's CURRENT state. Without this, correcting an older
        # row (e.g. a claim that has since been settled recovered) would become
        # the new current state by recorded_at, silently reopening a claim a later
        # outcome already closed and letting it be settled a second time.
        if old["source_type"] == "finding":
            _lock_finding(cur, int(old["source_id"]))
        else:
            cur.execute("SELECT 1 FROM proc.bp_opportunity WHERE opportunity_id = %s FOR UPDATE",
                        (str(old["source_id"]),))
            if not cur.fetchone():
                raise LedgerError("not_found", f"opportunity {old['source_id']} does not exist")
        history = _rows(cur, _HISTORY, (old["source_type"], str(old["source_id"])))
        current = current_state(history)
        if not current or current["outcome_id"] != outcome_id:
            raise LedgerError("not_current", "only the latest figure can be corrected")
        ev = evidence_ref if evidence_ref is not None else old["evidence_ref"]
        clean_amount, clean_ccy = validate_outcome(old["outcome_type"], amount, currency, ev)
        return _write(cur, c, source_type=old["source_type"], source_id=old["source_id"],
                      outcome_type=old["outcome_type"], amount=clean_amount,
                      currency=clean_ccy, actor=actor, evidence_ref=ev, note=note,
                      valid_from=old["valid_from"].isoformat(), supersedes_id=int(outcome_id))

    return _in_tx(conn, run)


def finding_outcomes(discrepancy_id: int, conn=None) -> dict:
    from src.services import value_summary_service as vss

    def run(c):
        cur = c.cursor()
        history = _rows(cur, _HISTORY, ("finding", str(discrepancy_id)))
        state = current_state(history)
        # R16 (2026-09-25): the triage figure is the leading £ figure of the finding's
        # own bp_detection_finding.delta -- the SAME figure the Action Centre shows --
        # never one arbitrary bp_triage_result line (a finding has many: one per cause
        # line, plus a cumulative_total row, all sharing its finding_id). mirror_id and
        # finding_id are each unique, so this is a plain 1:1 LEFT JOIN.
        rows = _rows(cur, """
            SELECT e.issue_type, e.raw_value, e.expected_value, e.computed_value, i.currency,
                   f.delta AS triage_delta
              FROM proc.bp_extraction_discrepancy e
              LEFT JOIN proc.bp_invoice_trgt i
                     ON e.doc_type = 'invoice' AND i.invoice_id = e.doc_pk_candidate
              LEFT JOIN proc.bp_triage_finding m ON m.mirror_id = e.discrepancy_id
              LEFT JOIN proc.bp_detection_finding f ON f.finding_id = m.finding_id
             WHERE e.discrepancy_id = %s""", (int(discrepancy_id),))
        prefill = {"amount": None, "currency": None, "is_money": False}
        if rows:
            r = rows[0]
            prefill["is_money"] = r["issue_type"] in vss.DISCREPANCY_VALUE_TYPES
            exposure = vss.parse_gbp_delta(r.get("triage_delta"))
            if exposure is not None:
                prefill.update(amount=f"{exposure:.2f}", currency="GBP")
            elif r["issue_type"] not in vss.TRIAGE_VALUE_TYPES:
                # R20(a): a triage mirror's raw/expected values are quantities or unit
                # prices, never money -- with no £ figure the buyer types the amount.
                delta = vss.discrepancy_delta(r)
                if delta is not None:
                    prefill.update(amount=f"{delta:.2f}", currency=r.get("currency"))
        return {"state": state["outcome_type"] if state else None,
                "history": history, "prefill": prefill}

    return _in_tx(conn, run)
