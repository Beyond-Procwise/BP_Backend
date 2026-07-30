"""Session-completion housekeeping.

Runs after a session's fast promotion + deal-linking pass:

1. **Stale-PO reconcile** — `po_not_found` / `po_pending_review` discrepancies
   whose cited purchase order has since reached `_stg`/`_trgt` are resolved.
   Within one upload batch the invoice is often dispatched before its own PO,
   and until 2026-07-30 the resulting "PO does not exist" flag stayed open
   forever even though the PO was two rows away (ses-20260730-UJF3).

2. **Deal proposal generation** — every un-established batch label in the
   session is clustered into proposed deals (spec §Upload path change). The
   batch label itself is never a deal; a human confirms proposals, and only
   that confirm mints deal identity. Established labels (amend-mode: the user
   explicitly targeted a real deal) are left alone.

Both steps are best-effort: failures log and leave the scheduled sweep to
catch up, exactly like the promotion steps before them.
"""
from __future__ import annotations

import logging

from src.services.db import get_conn
from src.services.deal_assignment_service import is_established_deal
from src.services.linking_engine import _PO_NORM_SQL

log = logging.getLogger(__name__)

_NORM_D = _PO_NORM_SQL.format(col="d.raw_value")
_NORM_STG = _PO_NORM_SQL.format(col="p.po_id")
_NORM_TRGT = _PO_NORM_SQL.format(col="t.po_id")

_RECONCILE_SQL = f"""
UPDATE proc.bp_extraction_discrepancy d
   SET status = 'resolved',
       resolved_at = now(),
       resolution_action = 'dismiss',
       resolved_by = 'session_postprocess',
       notes = coalesce(d.notes, '')
               || ' [auto-resolved: the cited purchase order has since reached the system]'
 WHERE d.issue_type IN ('po_not_found', 'po_pending_review')
   AND coalesce(d.status, 'open') <> 'resolved'
   AND d.raw_value IS NOT NULL
   AND (EXISTS (SELECT 1 FROM proc.bp_purchase_order_stg p
                 WHERE {_NORM_STG} = {_NORM_D})
     OR EXISTS (SELECT 1 FROM proc.bp_purchase_order_trgt t
                 WHERE {_NORM_TRGT} = {_NORM_D}))
"""


def reconcile_po_discrepancies(cur) -> int:
    """Resolve open PO-citation discrepancies whose PO now exists. Returns
    the number of rows resolved."""
    cur.execute(_RECONCILE_SQL)
    return cur.rowcount or 0


def _generate_proposals(batch_deal_id: str, session_id: str) -> dict:
    """Cluster one batch into proposals (module-level so tests can stub it)."""
    from src.api.routers.deal_proposals import _generate
    return _generate(batch_deal_id, session_id)


def compose_session_summary(*, total: int, linked: int, held: int, duplicates: int,
                            proposals: list, critical: int, warnings: int,
                            top_issues: list) -> str:
    """Deterministic executive summary of an upload session. Every number is
    counted, never generated — this is what the report shows the moment the
    analysis is ready; no LLM call sits on the hot path."""
    parts = [f"{total} document{'s' if total != 1 else ''} were received in this upload: "
             f"{linked} analysed and linked"
             + (f", {held} held for data review" if held else "")
             + (f", {duplicates} duplicate{'s' if duplicates != 1 else ''} of earlier uploads"
                if duplicates else "") + "."]

    if proposals:
        lines = []
        for p in proposals:
            bits = [f"{p['bids']} bidder{'s' if p['bids'] != 1 else ''}"]
            if p.get("pos"):
                bits.append(f"{p['pos']} purchase order{'s' if p['pos'] != 1 else ''}")
            if p.get("invoices"):
                bits.append(f"{p['invoices']} invoice{'s' if p['invoices'] != 1 else ''}")
            conf = (f", {round(float(p['confidence']))}% confidence"
                    if p.get("confidence") is not None else "")
            lines.append(f"• {p['proposed_name']} ({', '.join(bits)}{conf})")
        parts.append("The documents group into "
                     f"{len(proposals)} proposed deal{'s' if len(proposals) != 1 else ''}, "
                     "awaiting your confirmation:\n" + "\n".join(lines))

    if critical or warnings:
        sev = []
        if critical:
            sev.append(f"{critical} critical issue{'s' if critical != 1 else ''}")
        if warnings:
            sev.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
        issues = "Findings: " + " and ".join(sev) + "."
        if top_issues:
            issues += "\n" + "\n".join(f"• {t}" for t in top_issues)
        parts.append(issues)
    else:
        parts.append("No critical issues were found in this upload.")

    if proposals:
        parts.append("Next step: review and confirm the proposed deals, then clear "
                     "any critical findings in Data Validation & Actions.")
    return "\n\n".join(parts)


# Plain-language labels for the issue codes worth naming in an executive
# summary (mirrors the UI's ISSUE_TITLE map for the same codes).
_ISSUE_LABEL = {
    "po_not_found": "cite a purchase order that is not in the system",
    "po_pending_review": "cite a purchase order still in extraction review",
    "missing_required": "have a required value that could not be read",
    "amount_over_po": "bill above their purchase order",
    "line_amount_over_po": "bill a line above the purchase order",
    "line_not_on_po": "charge a line that is not on the purchase order",
    "duplicate_document": "duplicate an already-analysed document",
    "net_exceeds_gross": "state a net amount above the gross amount",
}


def _session_facts(cur, session_id: str) -> dict:
    """Counted facts about a session: document outcomes and open findings."""
    cur.execute(
        "select coalesce(doc_action, 'processed'), count(*) "
        "from proc.process_monitor where session_id = %s group by 1", (session_id,))
    counts = dict(cur.fetchall())
    total = sum(counts.values())
    held = counts.get("needs_review", 0)
    duplicates = counts.get("duplicate", 0)
    linked = total - held - duplicates - counts.get("unsupported", 0)

    # Open findings for this session's documents, via the raw tier's
    # process_monitor linkage (the only reliable doc->session join).
    cur.execute(
        """
        with pks as (
          select doc_pk_candidate from proc.bp_quote_raw
           where process_monitor_id in (select id from proc.process_monitor where session_id = %(sid)s)
          union
          select doc_pk_candidate from proc.bp_invoice_raw
           where process_monitor_id in (select id from proc.process_monitor where session_id = %(sid)s)
          union
          select doc_pk_candidate from proc.bp_purchase_order_raw
           where process_monitor_id in (select id from proc.process_monitor where session_id = %(sid)s)
        )
        select d.issue_type, d.severity, count(*) as n,
               count(distinct d.doc_pk_candidate) as docs,
               min(d.raw_value) as sample_ref
          from proc.bp_extraction_discrepancy d
         where coalesce(d.status, 'open') <> 'resolved'
           and d.doc_pk_candidate in (select doc_pk_candidate from pks)
         group by d.issue_type, d.severity
         order by (d.severity = 'critical') desc, count(*) desc
        """, {"sid": session_id})
    rows = cur.fetchall()
    critical = sum(r[2] for r in rows if r[1] == "critical")
    warnings = sum(r[2] for r in rows if r[1] == "warning")
    top_issues = []
    _singular = {"cite": "cites", "have": "has", "bill": "bills", "charge": "charges",
                 "duplicate": "duplicates", "state": "states"}
    for issue_type, severity, n, docs, sample_ref in rows:
        if severity != "critical" or issue_type not in _ISSUE_LABEL:
            continue
        label = _ISSUE_LABEL[issue_type]
        if docs == 1:
            verb, _, rest = label.partition(" ")
            label = f"{_singular.get(verb, verb)} {rest}"
        line = f"{docs} document{'s' if docs != 1 else ''} {label}"
        if issue_type in ("po_not_found", "po_pending_review") and sample_ref:
            line += f" ({sample_ref})"
        top_issues.append(line)
    return {"total": total, "linked": linked, "held": held, "duplicates": duplicates,
            "critical": critical, "warnings": warnings, "top_issues": top_issues[:5]}


def store_session_summary(conn, batch_deal_id: str, session_id: str,
                          proposals_out: dict) -> bool:
    """Compose + persist the analysis summary for the batch label, so
    GET /deals/{batch}/summary answers immediately. One row per deal_id
    (same contract as deal_analysis_service.upsert_analysis_row). The sweep's
    LLM narrative still owns real confirmed deals; it skips deal_ids that
    already carry a current row, which is exactly right for a draft analysis."""
    cur = conn.cursor()
    facts = _session_facts(cur, session_id)
    proposals = []
    for label_out in (proposals_out or {}).values():
        for pid in label_out.get("proposal_ids", []):
            cur.execute(
                "select p.proposed_name, p.confidence, "
                "  count(*) filter (where m.doc_type='quote') as bids, "
                "  count(*) filter (where m.doc_type='po') as pos, "
                "  count(*) filter (where m.doc_type='invoice') as invoices "
                "from proc.bp_deal_proposal p "
                "join proc.bp_deal_proposal_member m on m.proposal_id = p.proposal_id "
                "where p.proposal_id = %s group by p.proposal_id, p.proposed_name, p.confidence",
                (pid,))
            row = cur.fetchone()
            if row:
                proposals.append({"proposed_name": row[0], "confidence": row[1],
                                  "bids": row[2], "pos": row[3], "invoices": row[4]})
    text = compose_session_summary(proposals=proposals, **facts)
    cur.execute("select deal_name from proc.process_monitor "
                "where session_id = %s and coalesce(deal_name,'') <> '' limit 1",
                (session_id,))
    row = cur.fetchone()
    deal_name = row[0] if row else batch_deal_id
    cur.execute("delete from proc.bp_analysis_summary where deal_id = %s", (batch_deal_id,))
    cur.execute(
        "insert into proc.bp_analysis_summary "
        "(analysis_id, deal_id, deal_name, summary, model, is_current, generated_at) "
        "values (gen_random_uuid(), %s, %s, %s, 'session-postprocess/deterministic-v1', "
        " true, now())",
        (batch_deal_id, deal_name, text))
    conn.commit()
    return True


def postprocess_session(session_id: str) -> dict:
    out: dict = {"po_discrepancies_resolved": 0, "proposals": {}}
    labels: list[str] = []
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            out["po_discrepancies_resolved"] = reconcile_po_discrepancies(cur)
            conn.commit()
        except Exception:  # noqa: BLE001
            log.exception("stale-PO reconcile failed for session %s", session_id)
            conn.rollback()
        cur.execute(
            "select distinct deal_id from proc.process_monitor "
            "where session_id = %s and coalesce(deal_id, '') <> ''",
            (session_id,))
        candidates = [r[0] for r in cur.fetchall()]
        labels = [l for l in candidates if not is_established_deal(cur, l)]

    for label in labels:
        try:
            r = _generate_proposals(label, session_id)
            out["proposals"][label] = {
                "proposal_ids": r.get("proposal_ids", []),
                "ungrouped": r.get("ungrouped", []),
            }
        except Exception as exc:  # noqa: BLE001
            log.exception("proposal generation failed for batch %s", label)
            out["proposals"][label] = {"error": str(exc)}

    # Executive summary, available the moment the analysis is ready. Draft
    # analyses only — an amend-mode upload into an established deal keeps the
    # deal-summary machinery as its owner.
    if labels:
        try:
            with get_conn() as conn:
                store_session_summary(conn, labels[0], session_id, out["proposals"])
            out["summary_stored"] = True
        except Exception:  # noqa: BLE001
            log.exception("session summary failed for %s", session_id)
            out["summary_stored"] = False
    return out
