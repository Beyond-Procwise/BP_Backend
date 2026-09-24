"""Persistence for triage (spec §8). The only module that writes.

One real transaction per batch: get_conn() hands out AUTOCOMMIT connections, where
rollback() is a no-op, so without switching autocommit off a crash mid-batch would
leave half a batch of findings behind.

A bp_detection_finding row is only ever moved from status 'open', and status and
lifecycle_status move together -- the gateway's stage gate reads lifecycle_status.
"""
from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import Iterable

from psycopg2.extras import execute_values

from .model import ACTION_CENTRE_SEVERITY, Finding, Result, Severity, money

#: The SpendIQ Action Centre reads proc.bp_extraction_discrepancy, so every finding also
#: gets a mirror row there. Triage-only issue types keep these rows out of the other
#: readers' lists. 'duplicate' is absent: its detector's own row is already in that table.
MIRROR_ISSUE_TYPE = {
    "unit_price": "unit_price_differs_from_po",
    "uniform_uplift": "prices_uplifted_across_lines",
    "quantity": "quantity_invoiced_above_po",
    "cumulative_total": "invoices_exceed_po_total",
    "tax_rate": "tax_rate_not_allowed",
    "currency": "currency_differs_from_po",
    "supplier": "supplier_differs_from_po",
    "invoice_date": "invoice_dated_before_po",
    "payment_terms": "payment_terms_differ_from_po",
    "unlinked_line": "invoice_line_not_on_po",
    "bad_po_ref": "invoice_cites_missing_po",
    "line_arithmetic": "line_amount_not_qty_x_price",
    "invoice_totals": "invoice_totals_do_not_add_up",
    "description": "line_description_differs_from_po",
    "no_po": "invoice_has_no_po",
    "rollup": "invoice_lines_rolled_up",
}

_EXISTING = """
SELECT m.fingerprint, m.finding_id, m.last_severity, f.status, m.mirror_id
  FROM proc.bp_triage_finding m
  JOIN proc.bp_detection_finding f ON f.finding_id = m.finding_id
 WHERE m.deal_id = ANY(%s)
"""
_INSERT_FINDING = """
INSERT INTO proc.bp_detection_finding
    (engine_run_id, rule_id, category, severity, doc_type, doc_pk, deal_id,
     pipeline_record_id, field_name, observed_value, expected_value, delta,
     blocks_promotion, confidence, notes, status, lifecycle_status)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', 'open')
RETURNING finding_id
"""
_UPDATE_FINDING = """
UPDATE proc.bp_detection_finding
   SET engine_run_id = %s, severity = %s, observed_value = %s, expected_value = %s,
       delta = %s, blocks_promotion = %s, confidence = %s, notes = %s
 WHERE finding_id = %s AND status = 'open'
"""
# Only an untouched finding (the same test rollback uses) is closed when its problem
# disappears: one a person owns, has dated, resolved or moved on -- here or on its
# Action Centre mirror row -- is theirs to close.
_SUPERSEDE = """
UPDATE proc.bp_detection_finding f
   SET status = 'superseded', lifecycle_status = 'resolved'
 WHERE f.finding_id = %s AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL
   AND NOT EXISTS (SELECT 1 FROM proc.bp_triage_finding m
                     JOIN proc.bp_extraction_discrepancy d ON d.discrepancy_id = m.mirror_id
                    WHERE m.finding_id = f.finding_id
                      AND (d.status <> 'open' OR d.resolved_by IS NOT NULL
                           OR d.query_sent_at IS NOT NULL))
"""
_SUPERSEDE_MIRROR = """
UPDATE proc.bp_extraction_discrepancy SET status = 'superseded'
 WHERE discrepancy_id = %s AND status = 'open' AND resolved_by IS NULL AND query_sent_at IS NULL
"""
# A decision a person made on the mirror row in the Action Centre becomes the finding's
# own (a 'flag' leaves the mirror open, so there is nothing to carry over).
_SYNC_DECISIONS = """
UPDATE proc.bp_detection_finding f
   SET status = d.status,
       lifecycle_status = CASE d.status WHEN 'resolved' THEN 'resolved' ELSE 'accepted_risk' END,
       resolved_by = coalesce(d.resolved_by, 'action-centre'),
       resolved_at = coalesce(d.resolved_at, now())
  FROM proc.bp_triage_finding m
  JOIN proc.bp_extraction_discrepancy d ON d.discrepancy_id = m.mirror_id
 WHERE f.finding_id = m.finding_id AND m.deal_id = ANY(%s)
   AND d.status IN ('resolved', 'ignored') AND f.status = 'open'
"""
_UPSERT_MAP = """
INSERT INTO proc.bp_triage_finding
    (fingerprint, finding_id, deal_id, first_run_id, last_run_id, last_severity,
     replaced_finding_id, replaced_severity, mirror_id, replaced_mirror_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (fingerprint) DO UPDATE
   SET finding_id = EXCLUDED.finding_id, first_run_id = EXCLUDED.first_run_id,
       last_run_id = EXCLUDED.last_run_id, last_severity = EXCLUDED.last_severity,
       replaced_finding_id = EXCLUDED.replaced_finding_id,
       replaced_severity = EXCLUDED.replaced_severity,
       mirror_id = EXCLUDED.mirror_id, replaced_mirror_id = EXCLUDED.replaced_mirror_id
"""
# The Action Centre's copy of a finding. raw_id stays NULL and blocks_promotion false, so
# promotion and the discrepancy triggers never act on it; field_name carries the finding
# id so every finding (a reopened one too) has its own key in the open-row unique index.
# computed_value stays NULL: the decision engine reads it as a value in expected_value's
# units (or a signed delta), which an exposure is not. The exposure is in the notes.
_INSERT_MIRROR = """
INSERT INTO proc.bp_extraction_discrepancy
    (doc_type, raw_id, source_file, doc_pk_candidate, field_name, raw_value, expected_value,
     computed_value, issue_type, severity, status, notes, blocks_promotion)
VALUES (%s, NULL, %s, %s, %s, %s, %s, NULL, %s, %s, 'open', %s, false)
RETURNING discrepancy_id
"""
# A mirror row a person has decided on is theirs: only an open one follows the finding.
_UPDATE_MIRROR = """
UPDATE proc.bp_extraction_discrepancy
   SET raw_value = %s, expected_value = %s, computed_value = NULL, severity = %s, notes = %s
 WHERE discrepancy_id = %s AND status = 'open'
"""
_SET_MAP_MIRROR = "UPDATE proc.bp_triage_finding SET mirror_id = %s WHERE fingerprint = %s"
_TOUCH_MAP = """
UPDATE proc.bp_triage_finding SET last_run_id = %s, last_severity = %s WHERE fingerprint = %s
"""
_TOUCH_MAP_RUN_ONLY = "UPDATE proc.bp_triage_finding SET last_run_id = %s WHERE fingerprint = %s"
_AUDIT = """
INSERT INTO proc.bp_triage_result
    (run_id, deal_id, rule_id, claim_doc, claim_line, auth_doc, auth_line, field_name,
     claim_value, auth_value, outcome, severity, exposure_gbp, score, score_inputs,
     tolerance, fingerprint, finding_id)
VALUES %s
"""
# One row per deal written successfully: what its documents hashed to and which
# tolerances judged them. The scheduler re-triages a deal whose row is missing or differs.
_UPSERT_STATE = """
INSERT INTO proc.bp_triage_deal_state (deal_id, content_hash, config_fingerprint, last_run_id)
SELECT %s, %s, config_fingerprint, run_id FROM proc.bp_triage_run WHERE run_id = %s
ON CONFLICT (deal_id) DO UPDATE
   SET content_hash = EXCLUDED.content_hash,
       config_fingerprint = EXCLUDED.config_fingerprint,
       last_run_id = EXCLUDED.last_run_id, triaged_at = now()
"""
_ROLLBACK_FINDINGS = """
DELETE FROM proc.bp_detection_finding f
 USING proc.bp_triage_finding m
 WHERE m.finding_id = f.finding_id AND m.first_run_id = %s
   AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL
   AND NOT EXISTS (SELECT 1 FROM proc.bp_extraction_discrepancy d
                    WHERE d.discrepancy_id = m.mirror_id
                      AND (d.status <> 'open' OR d.resolved_by IS NOT NULL
                           OR d.query_sent_at IS NOT NULL))
RETURNING f.finding_id, m.fingerprint, m.replaced_finding_id, m.replaced_severity,
          m.mirror_id, m.replaced_mirror_id
"""
_ROLLBACK_MIRRORS = """
DELETE FROM proc.bp_extraction_discrepancy
 WHERE discrepancy_id = ANY(%s) AND status = 'open' AND resolved_by IS NULL
   AND query_sent_at IS NULL
"""
_RESTORE_MAP = """
UPDATE proc.bp_triage_finding
   SET finding_id = %s, last_severity = %s, replaced_finding_id = NULL, replaced_severity = NULL,
       mirror_id = %s, replaced_mirror_id = NULL
 WHERE fingerprint = %s
"""


def deal_state(cur) -> dict[str, tuple[str, str]]:
    """deal_id -> (content_hash, config_fingerprint) of its last successful triage."""
    cur.execute("SELECT deal_id, content_hash, config_fingerprint "
                "FROM proc.bp_triage_deal_state")
    return {d: (h, c) for d, h, c in cur.fetchall()}


def start_run(conn, mode: str, cfg) -> str:
    run_id = str(uuid.uuid4())
    conn.cursor().execute(
        """INSERT INTO proc.bp_triage_run (run_id, mode, config_fingerprint, config_values)
           VALUES (%s, %s, %s, %s)""",
        (run_id, mode, cfg.fingerprint, json.dumps(dict(cfg.values), default=str)))
    return run_id


def finish_run(conn, run_id: str, report) -> None:
    data = report.to_dict()
    conn.cursor().execute(
        """UPDATE proc.bp_triage_run
              SET finished_at = now(), deal_count = %s, failed_deals = %s, report = %s
            WHERE run_id = %s""",
        (data["deals_done"], json.dumps(data["failed"]), json.dumps(data, default=str), run_id))


def finding_ids(cur, fingerprints: Iterable[str]) -> dict[str, int]:
    fps = list(fingerprints)
    if not fps:
        return {}
    cur.execute("SELECT fingerprint, finding_id FROM proc.bp_triage_finding "
                "WHERE fingerprint = ANY(%s)", (fps,))
    return {fp: fid for fp, fid in cur.fetchall()}


def _or_dash(value) -> str:
    return "-" if value is None else str(value)


def _finding_values(run_id: str, f: Finding) -> tuple:
    r = f.lead
    if f.exposure_gbp is not None:
        delta = money(f.exposure_gbp, "GBP")
        if (r.currency or "GBP").upper() != "GBP":
            delta += f" ({money(f.exposure, r.currency)})"
    else:
        delta = f"{money(f.exposure, r.currency)} (no FX rate)"
    return (run_id, f.rule_id, f.category, ACTION_CENTRE_SEVERITY[f.severity],
            "purchase_order" if f.rule_id == "cumulative_total" else "invoice",
            r.claim_doc, f.deal_id, f.deal_id, r.field_name,
            f"{r.claim_doc}: {_or_dash(r.claim_value)}",
            f"{r.auth_doc or 'expected'}: {_or_dash(r.auth_value)}", delta,
            f.severity == Severity.S1, round(f.confidence, 4), f.text)


def _insert(cur, run_id: str, f: Finding) -> int:
    cur.execute(_INSERT_FINDING, _finding_values(run_id, f))
    return cur.fetchone()[0]


def _mirror_values(f: Finding) -> tuple:
    """(raw_value, expected_value) for the Action Centre row.

    The Action Centre shows raw - expected as the money at stake and sums it per group,
    so where every cause carries its money and an FX rate these are GBP totals over the
    finding's causes. Otherwise (a currency or supplier mismatch, no FX rate) they stay
    the lead's own values, which the Action Centre cannot read as money.
    """
    causes = f.causes
    if f.lead.fx_to_gbp is not None and all(
            r.claim_amount is not None and r.auth_amount is not None
            and r.fx_to_gbp is not None for r in causes):
        cent = Decimal("0.01")
        claim = sum((r.claim_amount * r.fx_to_gbp for r in causes), Decimal("0"))
        auth = sum((r.auth_amount * r.fx_to_gbp for r in causes), Decimal("0"))
        return str(claim.quantize(cent)), str(auth.quantize(cent))
    return f.lead.claim_value, f.lead.auth_value


def _insert_mirror(cur, f: Finding, finding_id: int):
    """The finding's Action Centre row, or None for a rule that is not mirrored."""
    issue_type = MIRROR_ISSUE_TYPE.get(f.rule_id)
    if issue_type is None:
        return None
    r = f.lead
    cur.execute(_INSERT_MIRROR, (
        "purchase_order" if f.rule_id == "cumulative_total" else "invoice",
        f"triage:{f.deal_id}", r.claim_doc, f"{r.field_name} #{finding_id}",
        *_mirror_values(f), issue_type,
        ACTION_CENTRE_SEVERITY[f.severity], f.text))
    return cur.fetchone()[0]


def _audit_row(run_id: str, r: Result, fid) -> tuple:
    return (run_id, r.deal_id, r.rule_id, r.claim_doc, r.claim_line, r.auth_doc, r.auth_line,
            r.field_name, r.claim_value, r.auth_value, r.outcome.value,
            (r.severity or Severity.S0).name, r.exposure_gbp, r.score,
            json.dumps(r.score_inputs, default=str), json.dumps(r.tolerance, default=str),
            r.fingerprint, fid)


def write_batch(conn, run_id: str, outputs) -> dict:
    counts = dict(inserted=0, updated=0, unchanged=0, reopened=0, superseded=0, audit_rows=0)
    outputs = list(outputs)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        # F3: one deal must never be written by two batches at once -- held for the
        # whole transaction, released automatically on commit/rollback.
        for deal_id in sorted({o.deal_id for o in outputs}):
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (deal_id,))
        deal_ids = [o.deal_id for o in outputs]
        cur.execute(_SYNC_DECISIONS, (deal_ids,))
        cur.execute(_EXISTING, (deal_ids,))
        existing = {fp: (fid, sev, status, mirror_id)
                    for fp, fid, sev, status, mirror_id in cur.fetchall()}
        seen: set[str] = set()
        written: dict[str, int] = {}   # fingerprint -> finding_id already written this batch
        fid_of: dict[int, int] = {}
        for o in outputs:
            for f in o.findings:
                if f.severity < Severity.S2:
                    continue
                fp = f.fingerprint
                if fp in seen:
                    # F4: a second finding with the same fingerprint in this batch is
                    # never a new row -- it rides on the one already written.
                    fid = written[fp]
                else:
                    seen.add(fp)
                    prior = existing.get(fp)
                    # F1: 'superseded' carries no human decision -- treat it as absent.
                    if prior is None or prior[2] == "superseded":
                        fid = _insert(cur, run_id, f)
                        cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id,
                                                  f.severity.name, None, None,
                                                  _insert_mirror(cur, f, fid), None))
                        counts["inserted"] += 1
                    else:
                        old_fid, old_sev, status, mirror_id = prior
                        if status == "open":
                            v = _finding_values(run_id, f)
                            cur.execute(_UPDATE_FINDING, (v[0], v[3], v[9], v[10], v[11], v[12],
                                                          v[13], v[14], old_fid))
                            if cur.rowcount == 0:
                                # F5: a person closed it mid-batch -- do not resurrect it.
                                cur.execute(_TOUCH_MAP_RUN_ONLY, (run_id, fp))
                                fid = old_fid
                                counts["unchanged"] += 1
                            else:
                                cur.execute(_TOUCH_MAP, (run_id, f.severity.name, fp))
                                if mirror_id is not None:
                                    cur.execute(_UPDATE_MIRROR, (
                                        *_mirror_values(f), v[3], v[14], mirror_id))
                                else:   # written before mirrors existed
                                    mirror_id = _insert_mirror(cur, f, old_fid)
                                    if mirror_id is not None:
                                        cur.execute(_SET_MAP_MIRROR, (mirror_id, fp))
                                fid = old_fid
                                counts["updated"] += 1
                        elif f.severity > Severity[old_sev]:
                            f.text = (f"Reopened: finding {old_fid} was {status}; this is now "
                                      f"{f.severity.name}. {f.text}")
                            fid = _insert(cur, run_id, f)
                            # F2: record what this reopen replaces so a rollback can
                            # hand the fingerprint back to the person's decision.
                            cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id,
                                                      f.severity.name, old_fid, old_sev,
                                                      _insert_mirror(cur, f, fid), mirror_id))
                            counts["reopened"] += 1
                        else:
                            fid = old_fid
                            cur.execute(_TOUCH_MAP_RUN_ONLY, (run_id, fp))
                            counts["unchanged"] += 1
                    written[fp] = fid
                for r in (*f.causes, *f.effects):
                    fid_of[id(r)] = fid
        for fp, (fid, _sev, status, mirror_id) in existing.items():
            if fp not in seen and status == "open":
                cur.execute(_SUPERSEDE, (fid,))
                counts["superseded"] += cur.rowcount
                if cur.rowcount and mirror_id is not None:
                    cur.execute(_SUPERSEDE_MIRROR, (mirror_id,))
        for o in outputs:
            if o.content_hash:
                cur.execute(_UPSERT_STATE, (o.deal_id, o.content_hash, run_id))
                if cur.rowcount != 1:
                    raise RuntimeError(f"triage run {run_id} is not recorded; cannot write state")
            else:   # its documents have all gone: nothing left to compare against
                cur.execute("DELETE FROM proc.bp_triage_deal_state WHERE deal_id = %s",
                            (o.deal_id,))
        rows = [_audit_row(run_id, r, fid_of.get(id(r))) for o in outputs for r in o.results]
        if rows:
            execute_values(cur, _AUDIT, rows, page_size=1000)
        counts["audit_rows"] = len(rows)
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def rollback_run(conn, run_id: str) -> dict:
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute(_ROLLBACK_FINDINGS, (run_id,))
        deleted = cur.fetchall()
        removed = [row[0] for row in deleted]
        # F2: a removed finding that replaced an earlier one hands the fingerprint back
        # to that earlier finding -- the person's decision on it must not vanish too.
        to_delete = []
        for _fid, fp, replaced_finding_id, replaced_severity, _mid, replaced_mirror_id in deleted:
            if replaced_finding_id is not None:
                cur.execute(_RESTORE_MAP, (replaced_finding_id, replaced_severity,
                                           replaced_mirror_id, fp))
            else:
                to_delete.append(fp)
        # Their Action Centre rows go too, unless a person has already acted on one.
        mirrors = [row[4] for row in deleted if row[4] is not None]
        if mirrors:
            cur.execute(_ROLLBACK_MIRRORS, (mirrors,))
        if to_delete:
            cur.execute("DELETE FROM proc.bp_triage_finding WHERE fingerprint = ANY(%s)",
                        (to_delete,))
        cur.execute("SELECT count(*) FROM proc.bp_triage_finding WHERE first_run_id = %s",
                    (run_id,))
        kept = cur.fetchone()[0]
        cur.execute("DELETE FROM proc.bp_triage_result WHERE run_id = %s", (run_id,))
        audit = cur.rowcount
        # The deals this run triaged lose their state row, so the scheduler re-checks
        # them on its next pass (supersedes and in-place updates are not undone).
        cur.execute("DELETE FROM proc.bp_triage_deal_state WHERE last_run_id = %s", (run_id,))
        state = cur.rowcount
        cur.execute("UPDATE proc.bp_triage_run SET rolled_back_at = now() WHERE run_id = %s",
                    (run_id,))
        conn.commit()
        return {"findings_removed": len(removed), "findings_kept": kept,
                "audit_rows_removed": audit, "deal_states_removed": state}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True
