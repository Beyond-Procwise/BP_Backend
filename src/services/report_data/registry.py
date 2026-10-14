"""The metric registry: what can be asked, from where, grouped by what, for whom.

The allowlist is a safety boundary, not a size limit: it grows by reviewed entry. An entry
is admitted when it maps to real, populated columns, declares ``requires`` and ``scope``,
has a test showing its result equals direct SQL, and says what it shows when data is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from src.services.version_collapse import latest_quote_pred

LIVE, PRESENTATION_ONLY, UNAVAILABLE = "live", "presentation-only", "unavailable"

# The one conversion basis (matches the gateway's SpendIqService.USD / FX_JOIN so a report
# tile and the Analyse view agree): rates are USD-quoted, converting INTO dollars divides.
_FX = """LEFT JOIN (SELECT currency, rate FROM proc.bp_fx_rates
              WHERE fetched_at = (SELECT MAX(fetched_at) FROM proc.bp_fx_rates)) fx
         ON fx.currency = i.currency"""
_USD = "i.invoice_amount / NULLIF(COALESCE(fx.rate, 1.0 / NULLIF(i.exchange_rate_to_usd, 0)), 0)"

_DOCTYPE = "CASE WHEN {c} IN ('po','purchase_order') THEN 'po' ELSE {c} END"


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    availability: str = LIVE
    is_time: bool = False


@dataclass(frozen=True)
class Source:
    """One table (or reviewed join) a metric reads, with how to time it, scope it and group it."""
    key: str
    from_sql: str
    time_expr: str
    scope_expr: str                      # Buyer/Viewer rows: <expr> = ANY(%(buyers)s)
    dims: Dict[str, Tuple[str, str]]     # dimension key -> (key sql, label sql)
    requires: str                        # the read action the caller must hold


def _time_dims(t: str) -> Dict[str, Tuple[str, str]]:
    # ::date so a key is a plain date (DATE_TRUNC on a date returns a timestamptz, which neither
    # compares equal to a date nor survives a session time-zone change).
    return {"month": (f"DATE_TRUNC('month', {t})::date", f"TO_CHAR(DATE_TRUNC('month', {t}),'Mon YYYY')"),
            "quarter": (f"DATE_TRUNC('quarter', {t})::date", f"TO_CHAR(DATE_TRUNC('quarter', {t}),'YYYY \"Q\"Q')"),
            "year": (f"DATE_TRUNC('year', {t})::date", f"TO_CHAR(DATE_TRUNC('year', {t}),'YYYY')")}


def _sup(alias: str, sid: str) -> Tuple[str, str]:
    return (sid, f"COALESCE(s.supplier_name, {sid})")


_SUP_JOIN = "LEFT JOIN proc.bp_supplier s ON s.supplier_id = {sid}"

SOURCES: Dict[str, Source] = {
    "invoice": Source(
        "invoice", f"proc.bp_invoice_trgt i {_FX} " + _SUP_JOIN.format(sid="i.supplier_id"),
        "i.invoice_date", "i.buyer_id",
        {**_time_dims("i.invoice_date"), "supplier": _sup("s", "i.supplier_id"),
         "buyer": ("i.buyer_id",) * 2, "currency": ("i.currency",) * 2, "country": ("i.country",) * 2,
         "region": ("i.region",) * 2, "payment_terms": ("i.payment_terms",) * 2},
        "invoice.read"),
    "deal": Source(
        "deal", "proc.bp_deal_overview o",
        "COALESCE(o.deal_date, o.first_activity_date)", "o.buyer_id",
        {**_time_dims("COALESCE(o.deal_date, o.first_activity_date)"),
         "supplier": ("o.supplier_id", "COALESCE(o.supplier_name, o.supplier_id)"),
         "buyer": ("o.buyer_id",) * 2, "currency": ("o.currency",) * 2,
         "deal": ("o.deal_id", "COALESCE(o.deal_name, o.deal_id)")},
        "deal.read"),
    "quote": Source(
        "quote", "proc.bp_quote_trgt q " + _SUP_JOIN.format(sid="q.supplier_id"),
        "q.quote_date", "q.buyer_id",
        {**_time_dims("q.quote_date"), "supplier": _sup("s", "q.supplier_id"),
         "buyer": ("q.buyer_id",) * 2, "currency": ("q.currency",) * 2, "country": ("q.country",) * 2,
         "region": ("q.region",) * 2, "status": ("q.status",) * 2},
        "quote.read"),
    "opportunity": Source(
        "opportunity", "proc.bp_opportunity op LEFT JOIN proc.bp_deal_overview o ON o.deal_id = op.deal_id",
        "op.detected_on", "o.buyer_id",
        {**_time_dims("op.detected_on"),
         "supplier": ("op.supplier_id", "COALESCE(op.supplier_name, op.supplier_id)"),
         "buyer": ("o.buyer_id",) * 2, "currency": ("op.currency",) * 2, "deal": ("op.deal_id",) * 2,
         "stage": ("op.stage",) * 2, "detector_type": ("op.detector_type",) * 2},
        "deal.read"),
    # A finding names its DOCUMENT (doc_pk_candidate); bp_deal_documents maps document -> deal; the
    # deal's buyer scopes it. Findings with no matching document have a NULL buyer: Admin only.
    "finding": Source(
        "finding",
        "proc.bp_extraction_discrepancy d "
        "LEFT JOIN (SELECT DISTINCT " + _DOCTYPE.format(c="doc_type") + " AS dt, doc_pk, deal_id "
        "FROM proc.bp_deal_documents) dd ON dd.doc_pk = d.doc_pk_candidate AND dd.dt = "
        + _DOCTYPE.format(c="d.doc_type") + " LEFT JOIN proc.bp_deal_overview o ON o.deal_id = dd.deal_id",
        "d.created_at", "o.buyer_id",
        {**_time_dims("d.created_at"), "finding_type": ("d.issue_type",) * 2, "severity": ("d.severity",) * 2,
         "status": ("d.status",) * 2, "doc_type": ("d.doc_type",) * 2, "deal": ("o.deal_id",) * 2,
         "buyer": ("o.buyer_id",) * 2},
        "finding.read"),
}
# the PO source exists for PO-based metrics added later; kept out until a metric needs it.

DIMENSIONS: Dict[str, Dimension] = {d.key: d for d in [
    Dimension("month", "Month", is_time=True), Dimension("quarter", "Quarter", is_time=True),
    Dimension("year", "Year", is_time=True), Dimension("supplier", "Supplier"), Dimension("buyer", "Buyer"),
    Dimension("currency", "Currency"), Dimension("country", "Country"), Dimension("region", "Region"),
    Dimension("payment_terms", "Payment terms"), Dimension("status", "Status"), Dimension("deal", "Deal"),
    Dimension("stage", "Opportunity stage"), Dimension("detector_type", "Detector type"),
    Dimension("finding_type", "Finding type"), Dimension("severity", "Severity"),
    Dimension("doc_type", "Document type"),
    # No data behind these in the live corpus: bp_opportunity.category_id exists but is empty.
    Dimension("category", "Category", PRESENTATION_ONLY), Dimension("tail_band", "Spend band", PRESENTATION_ONLY),
    Dimension("risk_dimension", "Risk dimension", PRESENTATION_ONLY),
]}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    format: str                           # currency | percent | count | days | score
    availability: str
    source: Optional[str] = None
    measure: Optional[str] = None         # one aggregate expression, valid per group
    where: Optional[str] = None           # fixed extra predicate (no user input)
    additive: bool = False                # groups sum to the total (counts, sums)
    unit: Optional[str] = None            # e.g. USD, GBP
    dimensions: Tuple[str, ...] = ()
    default_comparison: str = "prior_period"
    target: Optional[float] = None
    note: Optional[str] = None
    # for 3-state metrics: a second aggregate counting what the figure was taken over
    assessed: Optional[str] = None
    presentation_dimensions: Tuple[str, ...] = ()
    why_unavailable: Optional[str] = None


_TIME = ("month", "quarter", "year")
_INV = _TIME + ("supplier", "buyer", "currency", "country", "region", "payment_terms")
_DEAL = _TIME + ("supplier", "buyer", "currency", "deal")
_QUOTE = _TIME + ("supplier", "buyer", "currency", "country", "region", "status")
_OPP = _TIME + ("supplier", "buyer", "currency", "deal", "stage", "detector_type")
_FIND = _TIME + ("finding_type", "severity", "status", "doc_type", "deal", "buyer")

_PO_EXISTS = "EXISTS (SELECT 1 FROM proc.bp_purchase_order_trgt p WHERE p.po_id = i.po_id)"

METRICS: Dict[str, Metric] = {m.key: m for m in [
    Metric("committed_spend", "Committed spend", "currency", LIVE, "invoice",
           f"SUM({_USD})", additive=True, unit="USD", dimensions=_INV,
           note="Invoiced spend, net of VAT, converted to USD on the latest FX snapshot."),
    Metric("off_contract_spend", "Off-contract spend", "currency", LIVE, "invoice",
           f"SUM(CASE WHEN NOT {_PO_EXISTS} THEN {_USD} ELSE 0 END)", additive=True, unit="USD", dimensions=_INV,
           note="Invoiced with no matching PO."),
    Metric("non_po_spend", "Non-PO spend share", "percent", LIVE, "invoice",
           f"100.0 * SUM(CASE WHEN NOT {_PO_EXISTS} THEN {_USD} ELSE 0 END) / NULLIF(SUM({_USD}), 0)",
           dimensions=_INV, note="Share of invoiced spend with no matching PO."),
    # Quotes, not quote records: each bid once, at its latest version.
    Metric("quote_volume", "Quote volume", "count", LIVE, "quote",
           f"COUNT(*) FILTER (WHERE {latest_quote_pred('q')})", additive=True, dimensions=_QUOTE),
    Metric("cycle_time_to_po", "Cycle time to PO", "days", LIVE, "deal",
           "AVG(o.cycle_days_quote_to_po)", where="o.cycle_days_quote_to_po IS NOT NULL", dimensions=_DEAL),
    Metric("value_reconciled_rate", "Value reconciled", "percent", LIVE, "deal",
           "100.0 * AVG(CASE WHEN o.value_reconciled THEN 1.0 ELSE 0.0 END)",
           where="o.value_reconciled IS NOT NULL", dimensions=_DEAL, assessed="COUNT(*)",
           note="Do the quote, PO and invoice AMOUNTS agree. Not a delivery check."),
    Metric("three_way_match_rate", "Three-way match", "percent", LIVE, "deal",
           "100.0 * AVG(CASE WHEN o.three_way_matched THEN 1.0 ELSE 0.0 END)",
           where="o.three_way_matched IS NOT NULL", dimensions=_DEAL, assessed="COUNT(*)",
           note="Did what was billed ARRIVE. NULL = not assessed (no goods receipt); taken over assessed deals only."),
    Metric("savings_secured", "Savings secured", "currency", LIVE, "opportunity",
           "SUM(COALESCE(op.realised_savings_gbp, 0))", where="op.retired_at IS NULL", additive=True, unit="GBP",
           dimensions=_OPP, note="Realised savings. Zero until opportunities are progressed to a realised stage."),
    Metric("opportunity_pipeline", "Opportunity pipeline", "currency", LIVE, "opportunity",
           "SUM(COALESCE(op.financial_impact_gbp, 0))", where="op.retired_at IS NULL", additive=True, unit="GBP",
           dimensions=_OPP, note="Savings identified."),
    Metric("in_flight_negotiations", "In-flight negotiations", "count", LIVE, "opportunity",
           "COUNT(*)", where="op.retired_at IS NULL AND op.stage IN ('negotiation','agreed')", additive=True,
           dimensions=_OPP),
    Metric("duplicate_risk", "Duplicate findings", "count", LIVE, "finding", "COUNT(*)",
           where="d.issue_type ILIKE '%%duplicate%%'", additive=True, dimensions=_FIND,
           note="Duplicate-document findings raised."),
    # --- not live: registered so they are never silently dropped ---------------------------------
    Metric("tail_spend_visibility", "Tail spend visibility", "percent", PRESENTATION_ONLY,
           presentation_dimensions=_TIME + ("category",),
           why_unavailable="needs a spend-category / contract taxonomy; none exists"),
    Metric("compliance_rate", "Compliance rate", "percent", PRESENTATION_ONLY,
           presentation_dimensions=_TIME, why_unavailable="spend is not linked to a contract"),
    Metric("tail_spend_breakdown", "Tail spend breakdown", "currency", PRESENTATION_ONLY, additive=True,
           presentation_dimensions=_TIME + ("tail_band",), why_unavailable="only maverick findings exist"),
    Metric("spend_by_category", "Spend by category", "currency", PRESENTATION_ONLY, additive=True,
           presentation_dimensions=_TIME + ("category",),
           why_unavailable="bp_opportunity.category_id exists but is empty; no category in any extracted table"),
    Metric("supplier_risk_profile", "Supplier risk profile", "score", PRESENTATION_ONLY,
           presentation_dimensions=("risk_dimension", "supplier"),
           why_unavailable="only a single risk_score exists; the profile needs six dimensions"),
]}

# What the synthetic data has attributes for. Anything else is refused in presentation mode rather
# than invented, so presentation data never claims a breakdown it did not generate.
PRESENTATION_DIMS = ("month", "quarter", "year", "supplier", "buyer", "currency", "country", "region",
                     "category", "tail_band", "risk_dimension")

VIZ = ("kpi", "line", "bar", "table", "findings")
COMPARISONS = ("prior_period", "prior_year", "none")
DERIVE_OPS = ("ratio", "difference", "sum", "share")
MAX_GROUP_BY = 2


def live_metrics() -> List[Metric]:
    return [m for m in METRICS.values() if m.availability == LIVE]
