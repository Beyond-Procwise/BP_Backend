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
_SUPERSEDE = """
UPDATE proc.bp_detection_finding
   SET status = 'superseded', lifecycle_status = 'resolved'
 WHERE finding_id = %s AND status = 'open'
"""
_UPSERT_MAP = """
INSERT INTO proc.bp_triage_finding
    (fingerprint, finding_id, deal_id, first_run_id, last_run_id, last_severity)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (fingerprint) DO UPDATE
   SET finding_id = EXCLUDED.finding_id, first_run_id = EXCLUDED.first_run_id,
       last_run_id = EXCLUDED.last_run_id, last_severity = EXCLUDED.last_severity
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
_ROLLBACK_FINDINGS = """
DELETE FROM proc.bp_detection_finding f
 USING proc.bp_triage_finding m
 WHERE m.finding_id = f.finding_id AND m.first_run_id = %s
   AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL
RETURNING f.finding_id
"""


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
            f"{r.claim_doc}: {r.claim_value}", f"{r.auth_doc}: {r.auth_value}", delta,
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
        cur.execute(_EXISTING, ([o.deal_id for o in outputs],))
        existing = {fp: (fid, sev, status) for fp, fid, sev, status in cur.fetchall()}
        seen: set[str] = set()
        fid_of: dict[int, int] = {}
        for o in outputs:
            for f in o.findings:
                if f.severity < Severity.S2:
                    continue
                fp = f.fingerprint
                seen.add(fp)
                prior = existing.get(fp)
                if prior is None:
                    fid = _insert(cur, run_id, f)
                    cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id, f.severity.name))
                    counts["inserted"] += 1
                else:
                    old_fid, old_sev, status = prior
                    if status == "open":
                        v = _finding_values(run_id, f)
                        cur.execute(_UPDATE_FINDING, (v[0], v[3], v[9], v[10], v[11], v[12],
                                                      v[13], v[14], old_fid))
                        cur.execute(_TOUCH_MAP, (run_id, f.severity.name, fp))
                        fid = old_fid
                        counts["updated"] += 1
                    elif f.severity > Severity[old_sev]:
                        f.text = (f"Reopened: finding {old_fid} was {status}; this is now "
                                  f"{f.severity.name}. {f.text}")
                        fid = _insert(cur, run_id, f)
                        cur.execute(_UPSERT_MAP, (fp, fid, f.deal_id, run_id, run_id,
                                                  f.severity.name))
                        counts["reopened"] += 1
                    else:
                        fid = old_fid
                        cur.execute(_TOUCH_MAP_RUN_ONLY, (run_id, fp))
                        counts["unchanged"] += 1
                for r in (*f.causes, *f.effects):
                    fid_of[id(r)] = fid
        for fp, (fid, _sev, status) in existing.items():
            if fp not in seen and status == "open":
                cur.execute(_SUPERSEDE, (fid,))
                counts["superseded"] += cur.rowcount
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
        removed = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_triage_finding WHERE finding_id = ANY(%s)", (removed,))
        cur.execute("SELECT count(*) FROM proc.bp_triage_finding WHERE first_run_id = %s",
                    (run_id,))
        kept = cur.fetchone()[0]
        cur.execute("DELETE FROM proc.bp_triage_result WHERE run_id = %s", (run_id,))
        audit = cur.rowcount
        cur.execute("UPDATE proc.bp_triage_run SET rolled_back_at = now() WHERE run_id = %s",
                    (run_id,))
        conn.commit()
        return {"findings_removed": len(removed), "findings_kept": kept,
                "audit_rows_removed": audit}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True
