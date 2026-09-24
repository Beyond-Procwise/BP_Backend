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
from typing import Iterable

from psycopg2.extras import execute_values

from .model import ACTION_CENTRE_SEVERITY, Finding, Result, Severity, money

_EXISTING = """
SELECT m.fingerprint, m.finding_id, m.last_severity, f.status
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
# disappears: one a person owns, has dated, resolved or moved on is theirs to close.
_SUPERSEDE = """
UPDATE proc.bp_detection_finding
   SET status = 'superseded', lifecycle_status = 'resolved'
 WHERE finding_id = %s AND status = 'open' AND lifecycle_status = 'open'
   AND owner IS NULL AND due_date IS NULL AND resolved_by IS NULL
"""
_UPSERT_MAP = """
INSERT INTO proc.bp_triage_finding
    (fingerprint, finding_id, deal_id, first_run_id, last_run_id, last_severity,
     replaced_finding_id, replaced_severity)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (fingerprint) DO UPDATE
   SET finding_id = EXCLUDED.finding_id, first_run_id = EXCLUDED.first_run_id,
       last_run_id = EXCLUDED.last_run_id, last_severity = EXCLUDED.last_severity,
       replaced_finding_id = EXCLUDED.replaced_finding_id,
       replaced_severity = EXCLUDED.replaced_severity
"""
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
RETURNING f.finding_id, m.fingerprint, m.replaced_finding_id, m.replaced_severity
"""
_RESTORE_MAP = """
UPDATE proc.bp_triage_finding
   SET finding_id = %s, last_severity = %s, replaced_finding_id = NULL, replaced_severity = NULL
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
        cur.execute(_EXISTING, ([o.deal_id for o in outputs],))
        existing = {fp: (fid, sev, status) for fp, fid, sev, status in cur.fetchall()}
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
                                                  f.severity.name, None, None))
                        counts["inserted"] += 1
                    else:
                        old_fid, old_sev, status = prior
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
                                fid = old_fid
                                counts["updated"] += 1
                        elif f.severity > Severity[old_sev]:
                            f.text = (f"Reopened: finding {old_fid} was {status}; this is now "
                                      f"{f.severity.name}. {f.text}")
                            fid = _insert(cur, run_id, f)
                            # F2: record what this reopen replaces so a rollback can
                            # hand the fingerprint back to the person's decision.
                            cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id,
                                                      f.severity.name, old_fid, old_sev))
                            counts["reopened"] += 1
                        else:
                            fid = old_fid
                            cur.execute(_TOUCH_MAP_RUN_ONLY, (run_id, fp))
                            counts["unchanged"] += 1
                    written[fp] = fid
                for r in (*f.causes, *f.effects):
                    fid_of[id(r)] = fid
        for fp, (fid, _sev, status) in existing.items():
            if fp not in seen and status == "open":
                cur.execute(_SUPERSEDE, (fid,))
                counts["superseded"] += cur.rowcount
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
        for _finding_id, fp, replaced_finding_id, replaced_severity in deleted:
            if replaced_finding_id is not None:
                cur.execute(_RESTORE_MAP, (replaced_finding_id, replaced_severity, fp))
            else:
                to_delete.append(fp)
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
