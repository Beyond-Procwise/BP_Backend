"""Value Found (W1): read-model aggregation over existing finding sources.

Pure functions do the classification/dedup/summing so they unit-test on dicts;
build_value_summary() does the (thin) SQL. Spec:
docs/superpowers/specs/2026-07-30-value-found-design.md

Schema note (verified live against bp_sqldb 2026-07-30): proc.bp_extraction_discrepancy
carries neither deal_id nor supplier_name -- those live on proc.bp_invoice_trgt (deal_id,
currency, supplier_id) and proc.bp_supplier (supplier_name), so the discrepancy loader
joins through both rather than the bp_deal_documents view (whose supplier_name column is
NULL for invoice/quote rows -- only populated for POs).

FX note: repositories.fx_rate_repo.get_or_refresh_rates() returns
``{"fetched_at": datetime, "base_currency": "USD", "rates": {ccy: rate, ...}, "stale": bool}``
where every rate is USD-quoted (rates[ccy] = units of ccy per 1 USD). GBP is one of the
keys in "rates", not the base. Converting amount(ccy) -> GBP is therefore
``amount / rates[ccy] * rates["GBP"]``.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn
from src.services.value_ledger import SETTLED_STATES, current_state

log = logging.getLogger(__name__)

# Triage-sourced money findings (mirrored into proc.bp_extraction_discrepancy by
# triage/writer.py): their figure is the leading £ figure of the finding's own
# bp_detection_finding.delta (R16), never the legacy computed_value/raw_value/
# expected_value delta, and never one arbitrary bp_triage_result line (see
# classify_discrepancy / parse_gbp_delta).
TRIAGE_VALUE_TYPES = ("quantity_invoiced_above_po", "invoices_exceed_po_total",
                      "unit_price_differs_from_po")
DISCREPANCY_VALUE_TYPES = ("amount_over_po", "line_amount_over_po", "duplicate_invoice",
                           *TRIAGE_VALUE_TYPES)
# A PO-level "invoices exceed PO total" finding already counts the overbilling once;
# the per-invoice/per-line triage findings on invoices against that PO are the same
# money seen line by line and are superseded under it (supersede_lines_under_overbilled_po).
_PO_LEVEL_TYPES = ("invoices_exceed_po_total",)
_EXCLUDED_STATUS = ("ignored", "superseded")
_STAGE_TIER = {"identified": "potential", "negotiation": "verified", "agreed": "verified",
               "realised": "verified", "rejected": None, "closed": None}
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
_GBP_DELTA_RE = re.compile(r"^£([0-9,]+\.[0-9]+)")


def parse_amount(v: Any) -> Optional[float]:
    """STRICT: whole-string numbers only — a SHA-256 raw_value must return None."""
    s = re.sub(r"[£$€¥₹,\s]", "", str(v if v is not None else "")).strip()
    if not _NUM_RE.match(s):
        return None
    return float(s)


def parse_gbp_delta(s: Any) -> Optional[float]:
    """The leading £ figure of a proc.bp_detection_finding.delta string -- the SAME
    figure the Action Centre shows (R16). ``money()`` (triage/model.py) writes this as
    e.g. '£226.78' or, for a non-GBP document, '£1,687.57 (2,230.94 USD)'; when no FX
    rate was available at triage time it instead writes the native amount with no leading
    £ at all ('315.21 USD (no FX rate)'). That case, and anything else that doesn't start
    with a £ figure, returns None -- never a fabricated 0."""
    m = _GBP_DELTA_RE.match(str(s if s is not None else ""))
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def discrepancy_delta(row: dict) -> Optional[float]:
    # convention 1: computed_value IS the signed delta ("+950.00") for these types
    d = parse_amount(row.get("computed_value"))
    if d is not None:
        return abs(d)
    o, e = parse_amount(row.get("raw_value")), parse_amount(row.get("expected_value"))
    if o is None or e is None:
        return None
    return round(abs(o - e), 2)


def _age_days(created_at) -> Optional[int]:
    if not isinstance(created_at, datetime):
        return None
    now = datetime.now(timezone.utc)
    ca = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    return max((now - ca).days, 0)


def classify_discrepancy(row: dict) -> Optional[dict]:
    if row.get("issue_type") not in DISCREPANCY_VALUE_TYPES:
        return None
    if row.get("status") in _EXCLUDED_STATUS:
        return None
    # A triage mirror's figure is the leading £ figure of its bp_detection_finding.delta
    # (R16) -- the SAME figure the Action Centre shows for this finding as a whole, not
    # one arbitrary bp_triage_result line among its many (a finding has one result row
    # per cause line, plus a cumulative_total row, all sharing its finding_id). The
    # mirror leaves computed_value NULL on purpose (triage/writer.py), and its raw/
    # expected values are not a currency basis to difference.
    if row.get("issue_type") in TRIAGE_VALUE_TYPES:
        exposure = parse_gbp_delta(row.get("triage_delta"))
        if exposure is None or exposure <= 0:
            return None
        delta, currency = round(exposure, 2), "GBP"
    else:
        delta = discrepancy_delta(row)
        if delta is None:
            return None
        currency = row.get("currency") or None
    return {
        "id": f"disc:{row.get('discrepancy_id')}",
        "tier": "verified",
        "source": "discrepancy",
        "amount_gbp": delta,           # converted later if currency != GBP
        # Recovered/avoided/realised/claimed now come only from the ledger
        # (apply_ledger), never from the legacy resolution_outcome/recovered_amount
        # columns -- see constraints.md.
        "recovered_gbp": None,
        "converted_from": None,
        # Never assume GBP for a foreign (or unknown-currency) document: only an
        # explicit "GBP" passes straight through. Missing/blank currency is left
        # None here on purpose so the FX step excludes it from GBP sums and flags
        # it, instead of silently treating an unlabelled amount as sterling.
        "currency": currency,
        "title": _disc_title(row, delta, currency),
        "supplier_name": row.get("supplier_name"),
        "deal_id": row.get("deal_id") or None,
        "doc_pk": row.get("doc_pk_candidate"),
        "issue_type": row.get("issue_type"),
        "po_id": row.get("po_id"),
        "found_at": row.get("created_at").isoformat() if isinstance(row.get("created_at"), datetime) else None,
        "age_days": _age_days(row.get("created_at")),
        "link": {"screen": "actions", "id": row.get("discrepancy_id")},
        "status": row.get("status"),
        # When the finding was closed. The weekly digest needs it to answer "what did we
        # recover THIS week" — found_at slices what we found, this slices what came back.
        "resolved_at": row.get("resolved_at").isoformat() if isinstance(row.get("resolved_at"), datetime) else None,
        # A triage-sourced finding is not queryable by email yet: value_query_service's
        # template reads computed_value/raw_value/expected_value, which for a triage
        # mirror are not a currency basis (see above) -- see value_query_service's
        # _require_queryable, which refuses the same three types for the same reason.
        "queryable": (row.get("status") == "open" and bool(row.get("supplier_name"))
                     and row.get("issue_type") not in TRIAGE_VALUE_TYPES),
        "query_sent_at": row.get("query_sent_at").isoformat() if isinstance(row.get("query_sent_at"), datetime) else None,
        "superseded_by": None,
    }


def _disc_title(row: dict, delta: float, currency: Optional[str] = None) -> str:
    """The delta here is the amount as BILLED, in ``currency`` — the row's amount_gbp is
    the converted figure. Naming the currency keeps the two from reading as two different
    numbers ("duplicate ... by 147,783.11" beside "£110,043.74"). An unlabelled document's
    amount stays bare rather than wearing a guessed symbol.

    ``currency`` defaults to the row's own currency column; a triage-sourced finding
    passes its already-GBP exposure explicitly, since the document's own currency
    (row["currency"]) would mislabel a sterling figure as e.g. "226.78 USD"."""
    kind = {"duplicate_invoice": "appears to duplicate another invoice"}.get(
        row.get("issue_type"), "bills over its purchase order")
    ccy = str((currency if currency is not None else row.get("currency")) or "").strip().upper()
    amount = f"{delta:,.2f}{' ' + ccy if ccy else ''}"
    return f"{row.get('doc_type', 'document').capitalize()} {row.get('doc_pk_candidate')} {kind} by {amount}"


def classify_opportunity(row: dict) -> Optional[dict]:
    stage = row.get("stage")
    if stage not in _STAGE_TIER:
        raise ValueError(f"unmapped bp_opportunity stage {stage!r} — assign it a tier")
    tier = _STAGE_TIER[stage]
    if tier is None:
        return None
    amount = parse_amount(row.get("financial_impact_gbp")) or 0.0
    if amount <= 0:
        return None
    return {
        "id": f"opp:{row.get('opportunity_id')}",
        "tier": tier,
        "source": "opportunity",
        "amount_gbp": amount,          # financial_impact_gbp is already native GBP
        # Realised/avoided/recovered now come only from the ledger (apply_ledger),
        # never from the legacy realised_savings_gbp column -- see constraints.md.
        "recovered_gbp": None,
        "converted_from": None,
        "title": f"Opportunity: {row.get('item_description') or row.get('supplier_name') or 'unnamed'}",
        "supplier_name": row.get("supplier_name"),
        "deal_id": row.get("deal_id"),
        # invoice_id first: an opportunity anchored to ONE invoice shares its document
        # with the discrepancy that found it, and dedupe() keys on (deal_id, doc_pk) — so
        # a duplicate-invoice recovery counts its money once, in the discrepancy, rather
        # than a second time as potential. Sourcing opportunities have no invoice_id and
        # fall through to po_id/quote_id exactly as before.
        "doc_pk": (row.get("doc_pk") or row.get("invoice_id")
                   or row.get("po_id") or row.get("quote_id")),
        "found_at": row.get("created_at").isoformat() if isinstance(row.get("created_at"), datetime) else None,
        "age_days": _age_days(row.get("created_at")),
        "link": {"screen": "opportunities", "id": row.get("opportunity_id")},
        "status": stage,
        # An opportunity has no resolved_at; the moment it reached its current stage is the
        # equivalent — that is when a realised one was realised.
        "resolved_at": row.get("stage_updated_at").isoformat() if isinstance(row.get("stage_updated_at"), datetime) else None,
        "queryable": False,
        "query_sent_at": None,
        "superseded_by": None,
    }


_PRECEDENCE = {"discrepancy": 0, "opportunity": 1, "benchmark": 2}


def _norm_item(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def dedupe(findings: list[dict]) -> list[dict]:
    """Same £ in several sources counts once, in the strongest source. Suppressed
    duplicates stay in the list flagged superseded_by (the drawer explains, never omits)."""
    best: dict[tuple, dict] = {}
    for f in sorted(findings, key=lambda f: _PRECEDENCE[f["source"]]):
        key = (f.get("deal_id"), f.get("doc_pk"), _norm_item(f.get("title") if f["source"] == "benchmark" else None))
        if f.get("deal_id") is None and f.get("doc_pk") is None:
            best[("solo", f["id"], "")] = f           # nothing to collide on
            continue
        if key in best:
            f["superseded_by"] = best[key]["id"]
        else:
            best[key] = f
    return findings


_KEY = {"finding": "disc", "opportunity": "opp"}


def _by_source(ledger_rows: list[dict]) -> dict:
    grouped: dict[str, list] = {}
    for r in ledger_rows:
        grouped.setdefault(f"{_KEY[r['source_type']]}:{r['source_id']}", []).append(r)
    return grouped


def _gbp(v) -> float:
    return round(float(v), 2) if v is not None else 0.0


def ledger_totals(ledger_rows: list[dict]) -> dict:
    """Saved money from each source's CURRENT state (a correction replaces what it
    supersedes). An unconvertible row (amount_gbp NULL) counts in no GBP total."""
    totals = {"avoided_gbp": 0.0, "recovered_gbp": 0.0, "realised_gbp": 0.0,
              "claimed_open_gbp": 0.0}
    months: dict[str, dict] = {}
    field = {"avoided": "avoided_gbp", "recovered": "recovered_gbp",
             "realised_saving": "realised_gbp"}
    for rows in _by_source(ledger_rows).values():
        state = current_state(rows)
        if state is None:
            continue
        kind = state["outcome_type"]
        if kind == "claimed":
            totals["claimed_open_gbp"] += _gbp(state["amount_gbp"])
        elif kind in field:
            totals[field[kind]] += _gbp(state["amount_gbp"])
            m = months.setdefault(state["valid_from"].strftime("%Y-%m"),
                                  {"avoided_gbp": 0.0, "recovered_gbp": 0.0, "realised_gbp": 0.0})
            m[field[kind]] = round(m[field[kind]] + _gbp(state["amount_gbp"]), 2)
    totals = {k: round(v, 2) for k, v in totals.items()}
    totals["saved_gbp"] = round(totals["avoided_gbp"] + totals["recovered_gbp"]
                                + totals["realised_gbp"], 2)
    totals["by_month"] = [{"month": k, **v} for k, v in sorted(months.items())]
    return totals


def apply_ledger(findings: list[dict], ledger_rows: list[dict]) -> list[dict]:
    """Stamp each finding with its ledger state. A finding with no ledger rows carries
    no state and no claim -- it is still purely "found", per classify_discrepancy /
    classify_opportunity above."""
    grouped = _by_source(ledger_rows)
    for f in findings:
        rows = grouped.get(f["id"], [])
        state = current_state(rows)
        kind = state["outcome_type"] if state else None
        f["ledger_state"] = kind
        f["claim"] = ({"amount": str(state["amount"]), "currency": state["currency"],
                       "amount_gbp": _gbp(state["amount_gbp"]) if state["amount_gbp"] is not None else None}
                      if kind == "claimed" else None)
        claimed = [r for r in rows if r["outcome_type"] == "claimed"]
        f["claimed_at"] = claimed[0]["recorded_at"].isoformat() if claimed else None
        f["recovered_gbp"] = _gbp(state["amount_gbp"]) if kind == "recovered" and state["amount_gbp"] is not None else None
        f["avoided_gbp"] = _gbp(state["amount_gbp"]) if kind == "avoided" and state["amount_gbp"] is not None else None
        f["realised_gbp"] = _gbp(state["amount_gbp"]) if kind == "realised_saving" and state["amount_gbp"] is not None else None
        f["settled_at"] = (state["valid_from"].isoformat()
                           if kind in SETTLED_STATES else None)
    return findings


def in_play_gbp(findings: list[dict]) -> float:
    """Money still to act on: open findings, live opportunities, and claims not yet settled."""
    total = 0.0
    for f in findings:
        if f.get("superseded_by") or f.get("amount_gbp") is None:
            continue
        if f.get("ledger_state") in SETTLED_STATES:
            continue
        live = (f.get("ledger_state") == "claimed"
                or (f["source"] == "discrepancy" and f.get("status") == "open")
                or (f["source"] == "opportunity" and f.get("status") in ("identified", "negotiation", "agreed")))
        if live:
            total += f["amount_gbp"]
    return round(total, 2)


def _id_num(finding_id: str) -> int:
    try:
        return int(str(finding_id).split(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def supersede_po_level_by_duplicate(findings: list[dict]) -> list[dict]:
    """Controller ruling R6: a PO carrying both an "invoices exceed PO total" finding and
    a live duplicate-invoice finding on an invoice against that PO double-counts the same
    overage -- the duplicate is the stronger, whole-invoice explanation of it, so the
    PO-level finding is superseded by the duplicate (largest amount_gbp; ties -> lowest
    id). For a CFO headline an undercount is safer than a double count.

    Must run BEFORE supersede_lines_under_overbilled_po: that function only supersedes a
    line under a PO-level finding that is still live, so once this has demoted a PO-level
    finding, its line findings correctly stay live (they are not the duplicate's money
    unless dedupe() already collapsed them onto the same invoice)."""
    dupes_by_po: dict[str, list[dict]] = {}
    for f in findings:
        if (f.get("issue_type") == "duplicate_invoice" and f.get("po_id")
                and f.get("superseded_by") is None):
            dupes_by_po.setdefault(f["po_id"], []).append(f)
    for f in findings:
        if (f.get("issue_type") in _PO_LEVEL_TYPES and f.get("po_id")
                and f.get("superseded_by") is None):
            candidates = dupes_by_po.get(f["po_id"])
            if not candidates:
                continue
            best = max(candidates, key=lambda d: (
                d["amount_gbp"] if d["amount_gbp"] is not None else float("-inf"),
                -_id_num(d["id"])))
            f["superseded_by"] = best["id"]
    return findings


def supersede_lines_under_overbilled_po(findings: list[dict]) -> list[dict]:
    """A PO whose invoices exceed its total already counts the overbilled money once; the
    line findings on invoices against that PO are the same money, seen line by line."""
    po_level = {f["po_id"]: f["id"] for f in findings
                if f.get("issue_type") in _PO_LEVEL_TYPES and f.get("po_id")
                and f.get("superseded_by") is None}
    for f in findings:
        if (f.get("issue_type") in TRIAGE_VALUE_TYPES and f.get("issue_type") not in _PO_LEVEL_TYPES
                and f.get("po_id") in po_level and f.get("superseded_by") is None):
            f["superseded_by"] = po_level[f["po_id"]]
    return findings


def summarise(findings: list[dict]) -> dict:
    # amount_gbp can be None on a live finding whose native currency couldn't
    # be converted (unknown currency, or FX rates unavailable) -- that's an
    # honest "we don't know the GBP figure", never a fabricated 0.0. Such a
    # finding still counts (it's real, and stays in finding_count / its
    # supplier's finding_count) but contributes nothing to any GBP total.
    live = [f for f in findings if f.get("superseded_by") is None]
    verified = [f for f in live if f["tier"] == "verified"]
    by_supplier: dict[str, dict] = {}
    for f in verified:
        name = f.get("supplier_name") or "Unknown supplier"
        g = by_supplier.setdefault(name, {"supplier_name": name, "verified_found_gbp": 0.0,
                                          "finding_count": 0})
        if f["amount_gbp"] is not None:
            g["verified_found_gbp"] = round(g["verified_found_gbp"] + f["amount_gbp"], 2)
        g["finding_count"] += 1
    return {
        "verified_found_gbp": round(sum(f["amount_gbp"] for f in verified if f["amount_gbp"] is not None), 2),
        "potential_gbp": round(sum(f["amount_gbp"] for f in live
                                   if f["tier"] == "potential" and f["amount_gbp"] is not None), 2),
        "finding_count": len(live),
        "by_supplier": sorted(by_supplier.values(),
                              key=lambda g: -g["verified_found_gbp"]),
    }


# --------------------------------------------------------------------------
# SQL loaders (thin) + build_value_summary
# --------------------------------------------------------------------------

def _rows(cur, sql, params=()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# Verified live 2026-07-30: proc.bp_extraction_discrepancy has no deal_id/supplier_name
# of its own. proc.bp_invoice_trgt DOES carry deal_id + currency inline (and
# supplier_id, but not supplier_name -- that needs proc.bp_supplier). A PO-level triage
# mirror (doc_type = 'purchase_order', from the cumulative_total rule) has no invoice to
# join through -- it joins proc.bp_purchase_order_trgt/bp_supplier instead, and its own
# doc_pk_candidate IS the PO id (verified live 2026-09-25).
#
# R16 (2026-09-25, found live during the Task 10 demo): the triage figure comes from
# proc.bp_detection_finding.delta, not proc.bp_triage_result -- a finding has MANY result
# rows sharing its finding_id (one per cause line, plus a cumulative_total row), so the
# earlier "LATERAL ... ORDER BY started_at DESC LIMIT 1" picked one arbitrary line's
# figure rather than the finding's own total (live: £56.60 off a £226.78 finding; summed
# across the corpus, ~£12.6M against the true ~£39.4M). bp_triage_finding.mirror_id and
# bp_detection_finding.finding_id are each unique, so this is a plain 1:1 LEFT JOIN --
# no LATERAL/ORDER BY/LIMIT needed. parse_gbp_delta() reads the leading £ figure out of
# delta (the same figure the Action Centre shows); a non-triage discrepancy has no
# bp_triage_finding row, so triage_delta is NULL for it.
_DISCREPANCY_SQL = """
SELECT e.discrepancy_id, e.doc_type, e.doc_pk_candidate, e.field_name, e.raw_value,
       e.expected_value, e.computed_value, e.issue_type, e.status, e.notes, e.created_at,
       e.query_sent_at, e.resolved_at,
       coalesce(i.deal_id, p.deal_id) AS deal_id,
       coalesce(s.supplier_name, ps.supplier_name) AS supplier_name,
       coalesce(i.currency, p.currency) AS currency,
       CASE WHEN e.doc_type = 'purchase_order' THEN e.doc_pk_candidate ELSE i.po_id END AS po_id,
       f.delta AS triage_delta
  FROM proc.bp_extraction_discrepancy e
  LEFT JOIN proc.bp_invoice_trgt i
         ON e.doc_type = 'invoice' AND i.invoice_id = e.doc_pk_candidate
  LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
  LEFT JOIN proc.bp_purchase_order_trgt p
         ON e.doc_type = 'purchase_order' AND p.po_id = e.doc_pk_candidate
  LEFT JOIN proc.bp_supplier ps ON ps.supplier_id = p.supplier_id
  LEFT JOIN proc.bp_triage_finding m ON m.mirror_id = e.discrepancy_id
  LEFT JOIN proc.bp_detection_finding f ON f.finding_id = m.finding_id
 WHERE e.issue_type IN %s
"""

_OPPORTUNITY_SQL = """
SELECT opportunity_id, stage, financial_impact_gbp, realised_savings_gbp,
       supplier_name, deal_id, po_id, quote_id, invoice_id, item_description, created_at,
       stage_updated_at
  FROM proc.bp_opportunity
"""

_LEDGER_SQL = """
SELECT outcome_id, source_type, source_id, outcome_type, amount, currency, amount_gbp,
       supersedes_id, valid_from, recorded_at
  FROM proc.bp_value_outcome
"""


def _load_discrepancies(cur) -> list[dict]:
    rows = _rows(cur, _DISCREPANCY_SQL, (DISCREPANCY_VALUE_TYPES,))
    for row in rows:
        # bp_invoice_trgt.deal_id is '' rather than NULL on undeal-assigned rows;
        # normalise so downstream dedup/display never treats "" as a real key.
        row["deal_id"] = row.get("deal_id") or None
    return rows


def _load_opportunities(cur) -> list[dict]:
    return _rows(cur, _OPPORTUNITY_SQL)


def _load_ledger(cur) -> list[dict]:
    return _rows(cur, _LEDGER_SQL)


def _load_benchmark() -> list[dict]:
    """Positive-delta benchmark findings, if a cheap cross-deal call existed.

    services.benchmark_live.benchmark_deal() is deal-scoped only (it takes a
    single deal_id and rescans the whole PO/invoice price-history pool per
    call) -- there is no cheap "every deal, positive deltas only" query to
    call here without iterating every deal's quote lines per request, which
    is not a cheap read-model op. Rather than re-implement or approximate its
    matching/gating logic here, this returns no findings; the source is
    reported "ok" (genuinely empty), never fabricated.
    """
    return []


def _to_gbp(amount: float, currency: Optional[str], rates: Optional[dict]) -> tuple[Optional[float], Optional[dict]]:
    if currency == "GBP":
        return amount, None
    if not currency:
        # Never assume GBP for an unlabelled amount: exclude it from GBP sums
        # and say plainly that the currency is unknown.
        return None, {"currency": "unknown", "amount": amount, "rate_date": None}
    if not rates or currency not in rates or "GBP" not in rates:
        return None, {"currency": currency, "amount": amount, "rate_date": None}
    gbp = round(amount / rates[currency] * rates["GBP"], 2)
    return gbp, {"currency": currency, "amount": amount, "rate_date": rates.get("_fetched_at")}


def _get_rates() -> Optional[dict]:
    try:
        from repositories import fx_rate_repo
        result = fx_rate_repo.get_or_refresh_rates()
    except Exception:
        log.exception("value_summary_service: fx rate fetch failed")
        return None
    if not result:
        return None
    rates = dict(result.get("rates") or {})
    fetched_at = result.get("fetched_at")
    rates["_fetched_at"] = fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at
    return rates


def _apply_discrepancy_fx(finding: dict, rates: Optional[dict]) -> dict:
    currency = finding.pop("currency", None)
    amount_gbp, converted_from = _to_gbp(finding["amount_gbp"], currency, rates)
    # Honest None, never a silent zero: an unconvertible foreign amount is
    # real and non-zero, just not GBP-denominated yet. summarise() treats
    # None as excluded from every sum instead of rendering a fabricated
    # "£0.00"; converted_from still carries the native amount/currency so the
    # UI can show that instead.
    finding["amount_gbp"] = amount_gbp
    finding["converted_from"] = converted_from
    if finding.get("recovered_gbp") is not None:
        recovered_gbp, _ = _to_gbp(finding["recovered_gbp"], currency, rates)
        finding["recovered_gbp"] = recovered_gbp  # None (excluded) if unconvertable
    return finding


def build_value_summary(conn=None) -> dict:
    """Assemble the Value Found read model: classify -> FX-convert discrepancy
    amounts -> dedupe -> summarise. Each source loads in isolation: a failure
    in one never blocks the others, and is reported honestly in ``sources``
    rather than silently zeroed."""
    sources: dict[str, str] = {}
    findings: list[dict] = []
    ledger_rows: list[dict] = []

    def _own_conn():
        return get_conn()

    ctx = _own_conn() if conn is None else _NullContext(conn)
    with ctx as active_conn:
        cur = active_conn.cursor()
        try:
            rows = _load_discrepancies(cur)
            rates = _get_rates()
            for row in rows:
                f = classify_discrepancy(row)
                if f is not None:
                    findings.append(_apply_discrepancy_fx(f, rates))
            sources["discrepancies"] = "ok"
        except Exception:
            log.exception("value_summary_service: discrepancy source failed")
            sources["discrepancies"] = "unavailable"

        try:
            rows = _load_opportunities(cur)
            for row in rows:
                f = classify_opportunity(row)
                if f is not None:
                    findings.append(f)
            sources["opportunities"] = "ok"
        except Exception:
            log.exception("value_summary_service: opportunity source failed")
            sources["opportunities"] = "unavailable"

        try:
            ledger_rows = _load_ledger(cur)
            sources["ledger"] = "ok"
        except Exception:
            log.exception("value_summary_service: ledger source failed")
            ledger_rows, sources["ledger"] = [], "unavailable"

        try:
            cur.close()
        except Exception:
            pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_load_benchmark)
            findings.extend(future.result(timeout=5))
        sources["benchmark"] = "ok"
    except Exception:
        log.exception("value_summary_service: benchmark source failed")
        sources["benchmark"] = "unavailable"

    findings = dedupe(findings)
    findings = supersede_po_level_by_duplicate(findings)     # R6: before the line pass
    findings = supersede_lines_under_overbilled_po(findings)
    findings = apply_ledger(findings, ledger_rows)
    summary = summarise(findings)
    summary.update(ledger_totals(ledger_rows))
    summary["in_play_gbp"] = in_play_gbp(findings)
    return {
        **summary,
        "findings": findings,
        "sources": sources,
        "since": None,
    }


class _NullContext:
    """Wrap an already-open connection (e.g. injected in tests) so
    build_value_summary can use the same ``with ... as conn`` shape whether it
    opened the connection itself or received one."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False
