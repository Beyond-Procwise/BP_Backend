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
import re
from typing import Optional

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
   -- Only open findings. One a person ignored stays ignored: the PO arriving later is
   -- not a reason to overrule them, and the lifecycle trigger would refuse the move.
   AND d.status = 'open'
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


# The stored-summary tag. v2: verdict first, one facts object, every finding type named.
SUMMARY_MODEL = "session-postprocess/deterministic-v2"

# Plain-language name for every finding type, as a lower-case noun phrase so it
# reads both inside a sentence ("mostly lines with no quantity or price") and,
# capitalised, as a bullet label. A type missing here still renders, from its
# code, so a new detector never produces a blank.
_ISSUE_LABEL = {
    "amount_over_po": "amounts billed above the purchase order",
    "contract_parent_proposed": "proposed links to a parent contract",
    "currency_ambiguous": "amounts in an unclear currency",
    "duplicate_document": "documents already analysed in an earlier upload",
    "duplicate_invoice": "possible duplicate invoices",
    "invariant_failed": "internal consistency checks that failed",
    "invoice_cites_missing_po": "invoices citing a purchase order that is missing",
    "invoices_exceed_po_total": "invoices above the purchase order total",
    "line_amount_not_qty_x_price": "line amounts that are not quantity × unit price",
    "line_amount_over_po": "lines billed above the purchase order",
    "line_missing_amount": "lines with no amount",
    "line_missing_numbers": "lines with no quantity or price",
    "line_not_on_po": "lines charged that are not on the purchase order",
    "line_sum_mismatch": "lines that do not sum to the document total",
    "line_total_mismatch": "line totals that do not add up",
    "missing_line_items": "documents with no line items",
    "missing_required": "required values that could not be read",
    "net_exceeds_gross": "net amounts above the gross amount",
    "po_line_not_billed": "purchase-order lines not yet billed",
    "po_not_found": "purchase orders cited that are not in the system",
    "po_pending_review": "purchase orders cited that are still in extraction review",
    "po_reference_missing": "documents with no purchase-order reference",
    "price_rises_unstated": "price rises the document does not state",
    "prices_uplifted_across_lines": "prices raised across several lines",
    "quantity_invoiced_above_po": "quantities invoiced above the purchase order",
    "reread_not_applied": "documents still holding their earlier read (a later read was not applied)",
    "sum_mismatch": "totals that do not add up",
    "tax_percent_mismatch": "tax rates that do not match",
    "unit_price_differs_from_po": "unit prices that differ from the purchase order",
    "uplift_above_stated": "price rises above the stated uplift",
    "value_derived": "values worked out rather than read",
}

# Findings whose sample value is a reference worth quoting (the PO number cited).
_QUOTE_SAMPLE = ("po_not_found", "po_pending_review", "invoice_cites_missing_po")

# How many finding types the summary names; the rest are counted, not dropped.
_TOP_ISSUES = 3

# Proposals named before 2026-10-09 carry their bidder count in the name
# ("… — 3 bidders"). The count is now a separate fact, so a stored name is shown without it.
_BIDDER_SUFFIX = re.compile(r"\s+—\s+\d+\s+bidders?\s*$")


def issue_label(issue_type: str) -> str:
    return _ISSUE_LABEL.get(issue_type) or issue_type.replace("_", " ")


def _n(count: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _gbp(amount: float) -> str:
    return f"£{amount:,.2f}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _bids(p: dict) -> str:
    return _n(p["bids"], "competing bid") if p["bids"] > 1 else _n(p["bids"], "bid")


def _proposal_bits(p: dict) -> str:
    bits = [_bids(p)]
    if p.get("pos"):
        bits.append(_n(p["pos"], "purchase order"))
    if p.get("invoices"):
        bits.append(_n(p["invoices"], "invoice"))
    if p.get("confidence") is not None:
        bits.append(f"{round(float(p['confidence']))}% grouping confidence")
    return ", ".join(bits)


def compose_session_summary(facts: dict) -> str:
    """Deterministic executive summary of an upload session.

    Every sentence is rendered from ``facts`` (see ``session_summary_facts``), so
    every number in the text is a counted one; no LLM call sits on the hot path.
    Same shape as the confirmed-deal summary (deal_summary._build_prompt): a lead,
    "Key Outcomes:" bullets, a one-sentence "Conclusion:", so a deal reads the same
    way before and after it is confirmed. Here the lead is the verdict: is there a
    deal to confirm, and does anything block it.
    """
    docs = facts["documents"]
    proposals = facts["proposals"]
    findings = facts["findings"]
    pending = [p for p in proposals if p["status"] == "proposed"]
    confirmed = [p for p in proposals if p["status"] == "confirmed"]
    critical = [f for f in findings if f["severity"] == "critical"]
    n_critical = sum(f["count"] for f in critical)
    n_warning = sum(f["count"] for f in findings if f["severity"] == "warning")
    var = facts.get("value_at_risk") or {}
    billed = var.get("billed_gbp") or {}
    uplift = var.get("uplift_up_to_gbp")
    unvalued = var.get("unvalued") or {}
    total_billed = round(sum(billed.values()), 2)

    # ---- lead: the verdict --------------------------------------------------
    if pending and n_critical:
        lead = (f"{_n(len(pending), 'proposed deal is', 'proposed deals are')} waiting for "
                f"you, but {_n(n_critical, 'critical finding')} should be cleared before you "
                f"confirm {'it' if len(pending) == 1 else 'them'}.")
    elif len(pending) == 1:
        lead = f"1 proposed deal is ready to confirm: {pending[0]['name']}, with {_bids(pending[0])}."
    elif pending:
        lead = f"{len(pending)} proposed deals are ready to confirm."
    elif len(confirmed) == 1:
        lead = f"This upload's deal has been confirmed: {confirmed[0]['name']}."
    elif confirmed:
        lead = f"{len(confirmed)} deals from this upload have been confirmed."
    elif docs["total"] and not docs["linked"]:
        lead = "No deal could be proposed: none of the documents could be analysed yet."
    else:
        lead = "No deal could be proposed: the documents did not group into a sourcing event."

    if n_critical and pending:   # the lead already states the critical count
        if n_warning:
            lead += f" There {'is' if n_warning == 1 else 'are'} also {_n(n_warning, 'data warning')}."
    elif n_critical:
        also = f", with {_n(n_warning, 'data warning')}" if n_warning else ""
        lead += (f" {_cap(_n(n_critical, 'critical finding'))} "
                 f"{'is' if n_critical == 1 else 'are'} open{also}.")
    elif n_warning:
        top = max((f for f in findings if f["severity"] == "warning"), key=lambda f: f["count"])
        mostly = "" if top["count"] == n_warning else "mostly "
        blocks = "none of them block confirmation" if pending else "none of them is critical"
        lead += (f" {_cap(_n(n_warning, 'data warning'))}, {mostly}{issue_label(top['issue_type'])}"
                 f"; {blocks}.")
    else:
        lead += " There are no open findings."
    if total_billed:
        lead += f" Value at risk: {_gbp(total_billed)}."
    if uplift:
        lead += (f" A bid rising faster than its stated uplift adds up to {_gbp(uplift)} "
                 "over the term.")

    # ---- key outcomes ---------------------------------------------------------
    bullets = []
    for p in pending:
        bullets.append(f"• Proposed deal: {p['name']} — {_proposal_bits(p)}")
    for p in confirmed:
        bullets.append(f"• Confirmed deal: {p['name']} — {_proposal_bits(p)}")
    # A type's money is stated once, on its first row (a type can have a critical and a
    # warning row); whatever is not on a named row is stated on the "Other findings" row.
    valued: set = set()

    def _money(issue_type: str) -> str:
        if issue_type in valued:
            return ""
        valued.add(issue_type)
        bits = []
        if billed.get(issue_type):
            bits.append(f"{_gbp(billed[issue_type])} at risk")
        if issue_type == "uplift_above_stated" and uplift:
            bits.append(f"up to {_gbp(uplift)} over the term")
        if unvalued.get(issue_type):
            bits.append(f"{unvalued[issue_type]} not valued")
        return "; " + ", ".join(bits) if bits else ""

    named = findings[:_TOP_ISSUES]   # ordered critical first, then by count
    for f in named:
        kind = {"critical": "critical finding", "warning": "warning"}.get(f["severity"], "note")
        line = (f"• {_cap(issue_label(f['issue_type']))}: {_n(f['count'], kind)} across "
                f"{_n(f['docs'], 'document')}")
        if f["issue_type"] in _QUOTE_SAMPLE and f.get("sample_ref"):
            line += f" (e.g. {f['sample_ref']})"
        bullets.append(line + _money(f["issue_type"]))
    rest = findings[_TOP_ISSUES:]
    if rest:
        line = (f"• Other findings: {_n(sum(f['count'] for f in rest), 'more finding')} "
                f"of {_n(len(rest), 'other type')}")
        rest_billed = round(sum(billed.get(t, 0.0) for t in {f["issue_type"] for f in rest}
                                if t not in valued), 2)
        if rest_billed:
            line += f"; {_gbp(rest_billed)} at risk"
        bullets.append(line)
    doc_bits = [f"{docs['linked']} analysed and linked"]
    if docs["held"]:
        doc_bits.append(f"{docs['held']} held for data review")
    if docs["duplicates"]:
        doc_bits.append(_n(docs["duplicates"], "duplicate") + " of earlier uploads")
    bullets.append(f"• Documents: {docs['total']} received — {', '.join(doc_bits)}")

    # ---- conclusion: the next step follows the findings -------------------------
    where = "in Data Validation & Actions"
    if n_critical:
        types = ", ".join(issue_label(f["issue_type"]) for f in critical[:_TOP_ISSUES])
        step = f"Clear the {_n(n_critical, 'critical finding')} ({types}) {where}"
        step += ", then confirm the deal." if len(pending) == 1 else (
            ", then confirm the deals." if pending else ".")
    elif pending:
        step = "Confirm the deal" if len(pending) == 1 else "Review and confirm the proposed deals"
        step += (f"; the warnings can be triaged later {where}." if n_warning else ".")
    elif n_warning:
        step = f"Nothing needs a decision; the warnings can be triaged {where}."
    else:
        step = "Nothing needs your attention."
    if docs["held"]:
        held = f"review the {_n(docs['held'], 'document')} held for data review"
        step = (_cap(held) + f" {where}." if step.startswith("Nothing")
                else step[:-1] + f", and {held}.")

    return "\n\n".join([lead, "Key Outcomes:\n" + "\n".join(bullets), "Conclusion:\n" + step])


def _session_documents(cur, session_id: str) -> dict:
    cur.execute(
        "select coalesce(doc_action, 'processed'), count(*) "
        "from proc.process_monitor where session_id = %s group by 1", (session_id,))
    counts = dict(cur.fetchall())
    total = sum(counts.values())
    held = counts.get("needs_review", 0)
    duplicates = counts.get("duplicate", 0)
    linked = total - held - duplicates - counts.get("unsupported", 0)
    return {"total": total, "linked": linked, "held": held, "duplicates": duplicates}


def _session_facts(cur, session_id: str) -> list[dict]:
    """Open findings for this session's documents, one row per issue type × severity,
    critical first then by count."""
    # Open findings for this session's documents, via the raw tier's
    # process_monitor linkage (the only reliable doc->session join). Only
    # status 'open', as the report's "Items to validate" tile counts: a finding
    # a person ignored is not one the summary should still announce.
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
         where coalesce(d.status, 'open') = 'open'
           and d.doc_pk_candidate in (select doc_pk_candidate from pks)
           -- Document-type findings are reporting notes, not problems with the
           -- document's data; keep in step with
           -- extraction.persistence.TYPE_FINDING_ISSUE_TYPES (a test enforces it).
           and d.issue_type not in ('document_type_disagreement',
                                    'unresolved_document_type')
         group by d.issue_type, d.severity
         order by (d.severity = 'critical') desc, (d.severity = 'warning') desc,
                  count(*) desc, d.issue_type
        """, {"sid": session_id})
    return [{"issue_type": t, "severity": sev, "count": n, "docs": docs, "sample_ref": ref}
            for t, sev, n, docs, ref in cur.fetchall()]


def _session_proposals(cur, batch_deal_id: str) -> list[dict]:
    """The batch's live proposals (proposed or confirmed; rejected and superseded
    ones are history), with the bid / PO / invoice counts read from their members."""
    cur.execute(
        "select p.proposal_id, p.proposed_name, p.confidence, p.status, p.deal_id, "
        "  count(*) filter (where m.doc_type='quote' "
        "    and m.role is distinct from 'earlier_round') as bids, "
        "  count(*) filter (where m.doc_type='po') as pos, "
        "  count(*) filter (where m.doc_type='invoice') as invoices "
        "from proc.bp_deal_proposal p "
        "join proc.bp_deal_proposal_member m on m.proposal_id = p.proposal_id "
        "where p.batch_deal_id = %s and p.status in ('proposed', 'confirmed') "
        "group by p.proposal_id order by p.confidence desc nulls last, p.proposal_id",
        (batch_deal_id,))
    return [{"proposal_id": pid, "name": _BIDDER_SUFFIX.sub("", name or "") or "Sourcing event",
             "confidence": conf, "status": status, "deal_id": deal_id,
             "bids": bids, "pos": pos, "invoices": inv}
            for pid, name, conf, status, deal_id, bids, pos, inv in cur.fetchall()]


_SESSION_PKS_SQL = """
  select doc_pk_candidate from proc.bp_quote_raw
   where process_monitor_id in (select id from proc.process_monitor where session_id = %s)
  union
  select doc_pk_candidate from proc.bp_invoice_raw
   where process_monitor_id in (select id from proc.process_monitor where session_id = %s)
  union
  select doc_pk_candidate from proc.bp_purchase_order_raw
   where process_monitor_id in (select id from proc.process_monitor where session_id = %s)
"""


def _session_billed_value(cur, session_id: str) -> tuple[dict, dict]:
    """Money billed above what was ordered, or possibly paid twice: the open findings of
    this session's documents, valued exactly as the Value Found screen values them
    (value_summary_service: its per-type figure conventions, its rules for not counting
    the same money twice, its FX to GBP). Returns ({issue_type: gbp}, {issue_type: n
    findings that could not be valued}); a finding with no figure, no currency or no
    rate is counted as not valued, never as £0."""
    from src.services import value_summary_service as vs
    rows = vs._rows(cur, vs._DISCREPANCY_SQL
                    + " AND e.status = 'open' AND e.doc_pk_candidate IN (" + _SESSION_PKS_SQL + ")",
                    (vs.DISCREPANCY_VALUE_TYPES, session_id, session_id, session_id))
    unvalued: dict = {}
    findings = []
    for row in rows:
        row["deal_id"] = row.get("deal_id") or None
        f = vs.classify_discrepancy(row)
        if f is None:
            unvalued[row["issue_type"]] = unvalued.get(row["issue_type"], 0) + 1
            continue
        findings.append(f)
    need_fx = any(f.get("currency") not in (None, "GBP") for f in findings)
    rates = vs._get_rates() if need_fx else None
    findings = [vs._apply_discrepancy_fx(f, rates) for f in findings]
    findings = vs.supersede_overlaps(findings)
    value: dict = {}
    for f in findings:
        if f.get("superseded_by"):
            continue          # its money is counted under the finding that superseded it
        if f.get("amount_gbp") is None:
            unvalued[f["issue_type"]] = unvalued.get(f["issue_type"], 0) + 1
            continue
        value[f["issue_type"]] = round(value.get(f["issue_type"], 0.0) + f["amount_gbp"], 2)
    return value, unvalued


def _session_uplift_value(cur, session_id: str) -> tuple[Optional[float], int]:
    """The largest extra cost over the term, against its own stated uplift, among the
    session's bids (uplift_above_stated.computed_value; price_schedule.escalation_findings).

    Only each bid's latest version counts: the finding is raised on every round of a bid,
    and a later round that no longer rises too fast clears the bid. Bids are alternatives
    (one is awarded), so the figure is the largest, never a sum. price_rises_unstated is
    left out on purpose: rises nobody agreed to are a negotiating point, not money billed
    against an agreed rate. Returns (largest £ or None, findings that could not be valued)."""
    from src.services import value_summary_service as vs
    from src.services.version_collapse import base_reference, version_ordinal
    cur.execute(
        "select q.quote_id from proc.bp_quote_raw r "
        "join proc.process_monitor m on m.id = r.process_monitor_id "
        "join proc.bp_quote_stg q on q.quote_id = r.doc_pk_candidate "
        "where m.session_id = %s", (session_id,))
    latest: dict = {}
    for (qid,) in cur.fetchall():
        b = base_reference(qid)
        latest[b] = max(latest.get(b, 0), version_ordinal(qid))
    cur.execute(
        "select e.doc_pk_candidate, e.computed_value, coalesce(t.currency, s.currency) "
        "from proc.bp_extraction_discrepancy e "
        "left join proc.bp_quote_trgt t on t.quote_id = e.doc_pk_candidate "
        "left join proc.bp_quote_stg s on s.quote_id = e.doc_pk_candidate "
        "where e.issue_type = 'uplift_above_stated' and e.status = 'open' "
        "  and e.doc_pk_candidate in (" + _SESSION_PKS_SQL + ")",
        (session_id, session_id, session_id))
    amounts, unvalued, rates = [], 0, None
    for pk, computed, currency in cur.fetchall():
        if version_ordinal(pk) < latest.get(base_reference(pk), 0):
            continue          # an earlier round of a bid that has a later one
        amount = vs.parse_amount(computed)
        if amount is None or amount <= 0:
            unvalued += 1
            continue
        if currency not in (None, "GBP") and rates is None:
            rates = vs._get_rates()
        gbp, _ = vs._to_gbp(amount, currency, rates)
        if gbp is None:
            unvalued += 1
        else:
            amounts.append(gbp)
    return (max(amounts) if amounts else None), unvalued


def _session_value_at_risk(cur, session_id: str) -> dict:
    billed, unvalued = _session_billed_value(cur, session_id)
    uplift, uplift_unvalued = _session_uplift_value(cur, session_id)
    if uplift_unvalued:
        unvalued["uplift_above_stated"] = uplift_unvalued
    return {"billed_gbp": billed, "uplift_up_to_gbp": uplift, "unvalued": unvalued}


def session_summary_facts(cur, batch_deal_id: str, session_id: str) -> dict:
    """The one facts object the executive summary is rendered from."""
    return {"documents": _session_documents(cur, session_id),
            "proposals": _session_proposals(cur, batch_deal_id),
            "findings": _session_facts(cur, session_id),
            "value_at_risk": _session_value_at_risk(cur, session_id)}


def store_session_summary(conn, batch_deal_id: str, session_id: str,
                          proposals_out: Optional[dict] = None) -> bool:
    """Compose + persist the analysis summary for the batch label, so
    GET /deals/{batch}/summary answers immediately. One row per deal_id
    (same contract as deal_analysis_service.upsert_analysis_row). The sweep's
    LLM narrative still owns real confirmed deals; it skips deal_ids that
    already carry a current row, which is exactly right for a draft analysis.

    Proposals are read back from proc.bp_deal_proposal (the rows
    ``proposals_out`` names were just written there), so a later re-render
    (``refresh_session_summary``) reads exactly what this one did."""
    cur = conn.cursor()
    text = compose_session_summary(session_summary_facts(cur, batch_deal_id, session_id))
    cur.execute("select deal_name from proc.process_monitor "
                "where session_id = %s and coalesce(deal_name,'') <> '' "
                "order by id limit 1",
                (session_id,))
    row = cur.fetchone()
    deal_name = row[0] if row else batch_deal_id
    cur.execute("delete from proc.bp_analysis_summary where deal_id = %s", (batch_deal_id,))
    cur.execute(
        "insert into proc.bp_analysis_summary "
        "(analysis_id, deal_id, deal_name, summary, model, is_current, generated_at) "
        "values (gen_random_uuid(), %s, %s, %s, %s, true, now())",
        (batch_deal_id, deal_name, text, SUMMARY_MODEL))
    conn.commit()
    return True


def refresh_session_summary(conn, deal_id: str) -> Optional[str]:
    """Re-render a stored draft summary from today's facts.

    The summary is written once, when the session completes; findings resolved and
    proposals confirmed afterwards left it announcing what was no longer true (an
    upload kept reading "87 warnings" after a rule closed all 87). Rendering is a
    few counted queries, so the live read re-renders and stores the result when it
    changed. Only rows this module wrote are touched; returns None for any other
    deal (the caller then serves the stored row as before).
    """
    cur = conn.cursor()
    cur.execute("select summary, model from proc.bp_analysis_summary "
                "where deal_id = %s and is_current limit 1", (deal_id,))
    row = cur.fetchone()
    if row is None or not str(row[1] or "").startswith("session-postprocess/"):
        return None
    cur.execute("select session_id from proc.process_monitor "
                "where deal_id = %s and coalesce(session_id, '') <> '' "
                "order by id limit 1", (deal_id,))
    sid = cur.fetchone()
    if sid is None:
        return None
    text = compose_session_summary(session_summary_facts(cur, deal_id, sid[0]))
    if text != row[0] or row[1] != SUMMARY_MODEL:
        cur.execute("update proc.bp_analysis_summary set summary = %s, model = %s, "
                    "generated_at = now() where deal_id = %s and is_current",
                    (text, SUMMARY_MODEL, deal_id))
        conn.commit()
    return text


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
