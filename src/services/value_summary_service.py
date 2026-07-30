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

log = logging.getLogger(__name__)

DISCREPANCY_VALUE_TYPES = ("amount_over_po", "line_amount_over_po", "duplicate_invoice")
_EXCLUDED_STATUS = ("ignored", "superseded")
_STAGE_TIER = {"identified": "potential", "negotiation": "verified", "agreed": "verified",
               "realised": "verified", "rejected": None, "closed": None}
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


def parse_amount(v: Any) -> Optional[float]:
    """STRICT: whole-string numbers only — a SHA-256 raw_value must return None."""
    s = re.sub(r"[£$€¥₹,\s]", "", str(v if v is not None else "")).strip()
    if not _NUM_RE.match(s):
        return None
    return float(s)


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
    delta = discrepancy_delta(row)
    if delta is None:
        return None
    recovered = None
    if row.get("status") == "resolved" and row.get("resolution_outcome") == "recovered":
        recovered = parse_amount(row.get("recovered_amount"))
        if recovered is None:
            recovered = delta          # spec: reader falls back to the delta
    return {
        "id": f"disc:{row.get('discrepancy_id')}",
        "tier": "verified",
        "source": "discrepancy",
        "amount_gbp": delta,           # converted later if currency != GBP
        "recovered_gbp": recovered,
        "converted_from": None,
        # Never assume GBP for a foreign (or unknown-currency) document: only an
        # explicit "GBP" passes straight through. Missing/blank currency is left
        # None here on purpose so the FX step excludes it from GBP sums and flags
        # it, instead of silently treating an unlabelled amount as sterling.
        "currency": row.get("currency") or None,
        "title": _disc_title(row, delta),
        "supplier_name": row.get("supplier_name"),
        "deal_id": row.get("deal_id") or None,
        "doc_pk": row.get("doc_pk_candidate"),
        "found_at": row.get("created_at").isoformat() if isinstance(row.get("created_at"), datetime) else None,
        "age_days": _age_days(row.get("created_at")),
        "link": {"screen": "actions", "id": row.get("discrepancy_id")},
        "status": row.get("status"),
        "queryable": row.get("status") == "open" and bool(row.get("supplier_name")),
        "query_sent_at": row.get("query_sent_at").isoformat() if isinstance(row.get("query_sent_at"), datetime) else None,
        "superseded_by": None,
    }


def _disc_title(row: dict, delta: float) -> str:
    kind = {"duplicate_invoice": "appears to duplicate another invoice"}.get(
        row.get("issue_type"), "bills over its purchase order")
    return f"{row.get('doc_type', 'document').capitalize()} {row.get('doc_pk_candidate')} {kind} by {delta:,.2f}"


def classify_opportunity(row: dict) -> Optional[dict]:
    stage = row.get("stage")
    if stage not in _STAGE_TIER:
        raise ValueError(f"unmapped bp_opportunity stage {stage!r} — assign it a tier")
    tier = _STAGE_TIER[stage]
    if tier is None:
        return None
    amount = parse_amount(row.get("financial_impact_gbp")) or 0.0
    recovered = parse_amount(row.get("realised_savings_gbp")) if stage == "realised" else None
    if amount <= 0 and not recovered:
        return None
    return {
        "id": f"opp:{row.get('opportunity_id')}",
        "tier": tier,
        "source": "opportunity",
        "amount_gbp": amount,          # financial_impact_gbp is already native GBP
        "recovered_gbp": recovered,
        "converted_from": None,
        "title": f"Opportunity: {row.get('item_description') or row.get('supplier_name') or 'unnamed'}",
        "supplier_name": row.get("supplier_name"),
        "deal_id": row.get("deal_id"),
        "doc_pk": row.get("doc_pk") or row.get("po_id") or row.get("quote_id"),
        "found_at": row.get("created_at").isoformat() if isinstance(row.get("created_at"), datetime) else None,
        "age_days": _age_days(row.get("created_at")),
        "link": {"screen": "opportunities", "id": row.get("opportunity_id")},
        "status": stage,
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
        "recovered_gbp": round(sum(f["recovered_gbp"] or 0.0 for f in live), 2),
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
# supplier_id, but not supplier_name -- that needs proc.bp_supplier). Only doc_type
# = 'invoice' rows exist for these issue_types in the live corpus today, but the
# join is still gated on doc_type = 'invoice' so a future PO/quote-typed row of the
# same issue_type doesn't silently mismatch against an unrelated invoice_id.
_DISCREPANCY_SQL = """
SELECT e.discrepancy_id, e.doc_type, e.doc_pk_candidate, e.field_name, e.raw_value,
       e.expected_value, e.computed_value, e.issue_type, e.status, e.notes, e.created_at,
       e.resolution_outcome, e.recovered_amount, e.query_sent_at,
       i.deal_id, s.supplier_name, i.currency
  FROM proc.bp_extraction_discrepancy e
  LEFT JOIN proc.bp_invoice_trgt i
         ON e.doc_type = 'invoice' AND i.invoice_id = e.doc_pk_candidate
  LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
 WHERE e.issue_type IN %s
"""

_OPPORTUNITY_SQL = """
SELECT opportunity_id, stage, financial_impact_gbp, realised_savings_gbp,
       supplier_name, deal_id, po_id, quote_id, item_description, created_at
  FROM proc.bp_opportunity
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
    summary = summarise(findings)
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
