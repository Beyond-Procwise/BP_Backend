"""Procurement document linking engine.

A single deterministic engine that (1) scores how confidently two procurement
documents are related using the math model in
``Procurement Relationship Math Models.pdf``, and (2) drives confidence-gated
promotion of staged rows (`_stg`) into the final target tables (`_trgt`).

Domain: a *deal* is a sourcing event holding multiple competing quotes, one or
more purchase orders (the anchor), and invoices. Relationships are
invoice→PO and quote→PO via ``po_id``. ``deal_id`` is assigned by a separate
SQL trigger — this engine never writes it; it only ensures linkage accuracy so
the trigger's grouping is sound.

The scorer is fully deterministic and auditable: every signal exposes its match
score, field quality, reliability, signed contribution, and status.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Optional

from src.services.db import get_conn
from src.services.agent_actions import record_action, PHASE_CONSOLIDATION

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (env-overridable)
# ---------------------------------------------------------------------------
# The floor below which a staged row is not trusted enough to publish.
#
# Read `promotion._compute_confidence_score` before touching this: the score is
# COMPLETENESS, not correctness. Required fields score 2, optional fields 1, and the total
# is a percentage of the schema. Its own docstring says it: "a row that has every required
# field filled but no secondaries lands at ~50%."
#
# It was set to 90, which demanded that a document carry ~90% of every field the schema
# defines. Real documents do not: a purchase order that states no incoterm, no requisition
# id and no delivery region is a perfectly ordinary purchase order, and it scored 70-85 and
# was held. On the SpendIQDocs corpus that silently held 14 of 25 documents -- every one of
# them extracted correctly. We were rejecting documents for not containing fields they were
# never going to contain.
#
# A row missing a REQUIRED field never gets this far: extraction raises a blocking
# `missing_required` discrepancy and the row never reaches _stg. So the floor's only job is
# to catch a near-empty extraction, and 50 -- the score of a document with all its required
# fields and no optional ones -- is where that line actually sits.
MIN_CONFIDENCE = float(os.getenv("PROMOTE_MIN_CONFIDENCE", "50"))   # _stg extraction conf
MIN_LINK_SCORE = float(os.getenv("PROMOTE_MIN_LINK_SCORE", "80"))   # F auto-promote gate
REVIEW_MIN = float(os.getenv("PROMOTE_REVIEW_MIN", "65"))           # F floor for human review

# Decision band thresholds (PDF Stage 7C)
_BAND_AUTO = 92.0
_BAND_WARN = 80.0
_BAND_REVIEW = 65.0
_BAND_WEAK = 45.0

# Cluster dampening (PDF Stage 2)
def _dampen(n_active: int) -> float:
    if n_active <= 1:
        return 1.0
    if n_active == 2:
        return 0.85
    return 0.70


# ---------------------------------------------------------------------------
# Comparators (PDF Stage 1A) — each returns a match score s in [0,1];
# status is OK / CONFLICT / MISSING / WEAK / LOW_QUALITY for the gap report.
# ---------------------------------------------------------------------------
def _norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = re.sub(r"[^a-z0-9]", "", str(v).lower())
    return s or None


def _norm_po(v: Any) -> Optional[str]:
    """Canonical PO number. Procurement documents reference the same PO in
    inconsistent formats: the PO table stores it bare ('506789'), quotes prefix
    it ('PO506789'), invoices are mixed ('PO507269' / '389948'). Strip
    non-alphanumerics AND a leading 'po' so every form maps to the bare number,
    so quote->PO and invoice->PO joins actually connect."""
    s = _norm_id(v)
    if s is None:
        return None
    s = re.sub(r"^po", "", s)
    return s or None


def cmp_po_ref(a: Any, b: Any) -> tuple[float, str]:
    na, nb = _norm_po(a), _norm_po(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def cmp_exact_ref(a: Any, b: Any) -> tuple[float, str]:
    na, nb = _norm_id(a), _norm_id(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


cmp_exact_id = cmp_exact_ref  # same normalized-equality semantics


def cmp_supplier(a: Any, b: Any) -> tuple[float, str]:
    """Supplier identity with a fuzzy fallback (PDF D.4 SUPPLIER_NAME_CANONICAL).

    Exact resolved-ID match -> strong (1.0). When resolved IDs differ but share a
    strong root (one normalized ID is a prefix of the other, e.g. SUP-Nexaspark
    vs SUP-NexasparkMarketingLtd — a supplier-master resolution drift), treat as
    WEAK supporting evidence (0.7) rather than a hard Tier-1 conflict, so it
    supports the link without imposing the 0.45 conflict cap. Genuinely different
    suppliers remain a CONFLICT.
    """
    na, nb = _norm_id(a), _norm_id(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    if na == nb:
        return 1.0, "OK"
    short, lng = sorted([na, nb], key=len)
    if len(short) >= 8 and lng.startswith(short):
        return 0.7, "WEAK"
    return 0.0, "CONFLICT"


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cmp_numeric_tol(a: Any, b: Any, tol: float = 0.01) -> tuple[float, str]:
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return 0.5, "MISSING"
    denom = max(abs(fa), abs(fb))
    if denom == 0:
        return (1.0, "OK") if fa == fb else (0.0, "CONFLICT")
    drift = abs(fa - fb) / denom
    if drift <= tol:
        return 1.0, "OK"
    # linear decay to 0 by 10% drift
    s = max(0.0, 1.0 - (drift - tol) / (0.10 - tol)) if drift < 0.10 else 0.0
    return s, ("WEAK" if s >= 0.5 else "CONFLICT")


def _tokens(text: Any) -> set[str]:
    if text is None:
        return set()
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t}


def _line_pair_score(a: dict, b: dict) -> float:
    """Per-line composite over the sub-signals that actually have data on both
    sides (description always; quantity / unit price only when present). Missing
    qty/price are neutral, not penalised (PDF: missing -> neutral, not conflict)."""
    parts: list[tuple[float, float]] = []  # (weight, score)
    desc_a, desc_b = _tokens(a.get("item_description")), _tokens(b.get("item_description"))
    if desc_a or desc_b:
        parts.append((0.5, len(desc_a & desc_b) / max(1, len(desc_a | desc_b))))
    qa, qb = _to_float(a.get("quantity")), _to_float(b.get("quantity"))
    if qa is not None and qb is not None:
        parts.append((0.25, 1.0 if abs(qa - qb) < 1e-9 else 0.0))
    pa, pb = _to_float(a.get("unit_price")), _to_float(b.get("unit_price"))
    if pa is not None and pb is not None:
        parts.append((0.25, 1.0 if abs(pa - pb) < 1e-6 else 0.0))
    if not parts:
        return 0.0
    return sum(w * s for w, s in parts) / sum(w for w, _ in parts)


def cmp_line_composite(src_lines: list[dict], tgt_lines: list[dict]) -> tuple[float, str]:
    if not src_lines or not tgt_lines:
        return 0.5, "MISSING"
    # best-match each source (e.g. invoice) line to a target (PO) line
    used: set[int] = set()
    scores: list[float] = []
    for sl in src_lines:
        best, best_j = 0.0, -1
        for j, tl in enumerate(tgt_lines):
            if j in used:
                continue
            sc = _line_pair_score(sl, tl)
            if sc > best:
                best, best_j = sc, j
        if best_j >= 0:
            used.add(best_j)
        scores.append(best)
    avg = sum(scores) / len(scores) if scores else 0.0
    # Coverage penalty ONLY for source-extra lines (invoice lines with no PO
    # match). A partial invoice covering a SUBSET of PO lines is legitimate
    # (split shipment / staged payment) and is not penalised (PDF LINE_COUNT_COVERAGE).
    extra = max(0, len(src_lines) - len(tgt_lines))
    coverage = 1.0 - (extra / len(src_lines)) if src_lines else 1.0
    s = avg * coverage
    return s, ("OK" if s >= 0.8 else "WEAK" if s >= 0.5 else "CONFLICT")


def _to_date(v: Any):
    if v is None:
        return None
    if hasattr(v, "toordinal"):
        return v
    try:
        from datetime import date
        return date.fromisoformat(str(v)[:10])
    except Exception:
        return None


def cmp_temporal(inv_date: Any, po_order: Any, po_due: Any) -> tuple[float, str]:
    d, o, u = _to_date(inv_date), _to_date(po_order), _to_date(po_due)
    if d is None or o is None:
        return 0.5, "MISSING"
    subs = []
    after_parent = 1.0 if d >= o else 0.0           # invoice/quote on/after PO date
    subs.append(after_parent)
    delta = (d - o).days
    if delta <= 365:
        within = 1.0
    elif delta <= 730:
        within = 0.7
    else:
        within = 0.0
    subs.append(within)
    if u is not None:
        subs.append(1.0 if d <= u else 0.0)         # not past expected/expiry
    s = min(subs)
    return s, ("OK" if s >= 0.7 else "WEAK" if s > 0 else "CONFLICT")


def cmp_location(a_country: Any, a_region: Any, b_country: Any, b_region: Any) -> tuple[float, str]:
    cs, _ = cmp_exact_ref(a_country, b_country)
    rs, _ = cmp_exact_ref(a_region, b_region)
    # treat MISSING (0.5) as partial support, not conflict
    s = 0.5 * cs + 0.5 * rs
    return s, ("OK" if s >= 0.8 else "WEAK" if s >= 0.4 else "CONFLICT")


# ---------------------------------------------------------------------------
# Profiles (PDF D.11 / D.12) — computable signals only.
# Each signal: id, cluster, tier, weight, applicability, conflict_cap, kind.
# ---------------------------------------------------------------------------
_COMMON_SIGNALS = [
    {"id": "po_ref",      "cluster": "reference",  "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "po_ref"},
    {"id": "supplier_id", "cluster": "identity",   "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "supplier_id"},
    {"id": "amount",      "cluster": "commercial", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.65, "kind": "amount"},
    {"id": "currency",    "cluster": "commercial", "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.80, "kind": "currency"},
    {"id": "line_set",    "cluster": "line",       "tier": 2, "weight": 4, "appl": 1.0, "cap": 0.70, "kind": "line_set"},
    {"id": "temporal",    "cluster": "temporal",   "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "temporal"},
    {"id": "location",    "cluster": "context",    "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "location"},
]

PROFILES = {
    "invoice_po": {"p0": 0.02, "alpha": 0.30, "floor": 0.55, "signals": _COMMON_SIGNALS,
                   "date_field": "invoice_date"},
    "quote_po":   {"p0": 0.02, "alpha": 0.30, "floor": 0.55, "signals": _COMMON_SIGNALS,
                   "date_field": "quote_date"},
}


def _signal_match(kind: str, src: dict, tgt: dict, src_lines, tgt_lines, date_field: str,
                  set_amount_usd: Optional[float] = None) -> tuple[float, str]:
    if kind == "po_ref":
        return cmp_po_ref(src.get("po_id"), tgt.get("po_id"))
    if kind == "supplier_id":
        return cmp_supplier(src.get("supplier_id"), tgt.get("supplier_id"))
    if kind == "amount":
        # Set-level for N:1 (PDF Appendix B): compare the aggregate source amount
        # (sum of sibling invoices sharing the PO) against the PO total, so a
        # legitimate partial/split invoice does not read as an amount conflict.
        src_amt = set_amount_usd if set_amount_usd is not None else src.get("converted_amount_usd")
        return cmp_numeric_tol(src_amt, tgt.get("converted_amount_usd"))
    if kind == "currency":
        return cmp_exact_ref(src.get("currency"), tgt.get("currency"))
    if kind == "line_set":
        return cmp_line_composite(src_lines, tgt_lines)
    if kind == "temporal":
        # PO has no expiry field; expected_delivery_date is a delivery target, NOT
        # an expiry, so a legitimate later (e.g. staged-payment) invoice must not
        # be flagged. Compare on order-precedence + plausibility window only.
        return cmp_temporal(src.get(date_field), tgt.get("order_date"), None)
    if kind == "location":
        return cmp_location(src.get("country"), src.get("region"),
                            tgt.get("ship_to_country"), tgt.get("delivery_region"))
    raise ValueError(f"unknown signal kind {kind}")


def _band(F: float) -> str:
    if F >= _BAND_AUTO:
        return "auto_link"
    if F >= _BAND_WARN:
        return "auto_link_with_warning"
    if F >= _BAND_REVIEW:
        return "review"
    if F >= _BAND_WEAK:
        return "weak_relation"
    return "block_or_exception"


def score_link(source_row: dict, target_row: dict, profile_name: str,
               source_lines: Optional[list] = None, target_lines: Optional[list] = None,
               set_amount_usd: Optional[float] = None) -> dict:
    """Deterministically score the relationship source→target. Returns the full
    auditable result (PDF stages 1A..7C).

    ``set_amount_usd`` (PDF Appendix B): when scoring an N:1 child (e.g. one of
    several invoices on a PO), pass the aggregate amount of the whole sibling set
    so the amount signal compares the set total against the parent, not the
    single child.
    """
    profile = PROFILES[profile_name]
    src_lines = source_lines or []
    tgt_lines = target_lines or []

    q_src = (_to_float(source_row.get("confidence_score")) or 0.0) / 100.0
    q_tgt = (_to_float(target_row.get("confidence_score")) or 0.0) / 100.0

    signals = []
    for spec in profile["signals"]:
        s, status = _signal_match(spec["kind"], source_row, target_row, src_lines, tgt_lines,
                                  profile["date_field"], set_amount_usd)
        # Stage 1B/1C: normalized/system fields (currency) trust = 1.0; else conf proxy.
        if spec["kind"] == "currency":
            q = 1.0
        else:
            q = min(q_src if q_src else 1.0, q_tgt if q_tgt else 1.0)
        if status == "MISSING":
            q = 0.0  # missing field carries no reliable evidence
        r = q * spec["appl"]
        c = spec["weight"] * r * (2.0 * s - 1.0)
        signals.append({**spec, "s": s, "q": q, "r": r, "c": c, "status": status})

    # Stage 2: cluster dampening
    clusters: dict[str, list[dict]] = {}
    for sig in signals:
        clusters.setdefault(sig["cluster"], []).append(sig)
    total_cluster_score = 0.0
    for sigs in clusters.values():
        n_active = sum(1 for x in sigs if x["status"] != "MISSING")
        total_cluster_score += _dampen(n_active) * sum(x["c"] for x in sigs)

    # Stage 3/4: log-odds → probability
    p0 = profile["p0"]
    L = math.log(p0 / (1 - p0)) + profile["alpha"] * total_cluster_score
    P_raw = 1.0 / (1.0 + math.exp(-L))

    # Stage 5: coverage
    sum_wr = sum(x["weight"] * x["r"] for x in signals)
    sum_w = sum(x["weight"] for x in signals)
    rho = sum_wr / sum_w if sum_w else 0.0
    C = profile["floor"] + (1 - profile["floor"]) * rho

    # Stage 6: separation — single explicitly-referenced candidate (1:1)
    S = 1.0

    # Stage 7: caps. Tier-1 conflict imposes the signal's conflict cap.
    F_cap = 1.0
    for sig in signals:
        if sig["status"] == "CONFLICT" and sig["tier"] == 1:
            F_cap = min(F_cap, sig["cap"])

    # Q: pipeline/extraction quality proxy
    Q = min(q_src if q_src else 1.0, q_tgt if q_tgt else 1.0)

    F = min(F_cap, P_raw * C * S * Q) * 100.0
    return {
        "F": round(F, 4),
        "decision": _band(F),
        "P_raw": round(P_raw, 6),
        "C": round(C, 6),
        "S": S,
        "Q": round(Q, 4),
        "F_cap": F_cap,
        "L": round(L, 6),
        "signals": [
            {"id": s["id"], "cluster": s["cluster"], "tier": s["tier"], "weight": s["weight"],
             "s": round(s["s"], 4), "q": round(s["q"], 4), "r": round(s["r"], 4),
             "c": round(s["c"], 4), "status": s["status"]}
            for s in signals
        ],
    }


# ---------------------------------------------------------------------------
# Gated promotion _stg -> _trgt
# ---------------------------------------------------------------------------
_DOC = {
    "invoice": {
        "stg": "proc.bp_invoice_stg", "trgt": "proc.bp_invoice_trgt", "pk": "invoice_id",
        "lines_stg": "proc.bp_invoice_line_items_stg", "lines_trgt": "proc.bp_invoice_line_items_trgt",
        "profile": "invoice_po",
    },
    "quote": {
        "stg": "proc.bp_quote_stg", "trgt": "proc.bp_quote_trgt", "pk": "quote_id",
        "lines_stg": "proc.bp_quote_line_items_stg", "lines_trgt": "proc.bp_quote_line_items_trgt",
        "profile": "quote_po",
    },
}
_PO = {"stg": "proc.bp_purchase_order_stg", "trgt": "proc.bp_purchase_order_trgt", "pk": "po_id"}
# Deal columns are owned by deal_assignment_service, NOT by promotion: never
# copy them stg->trgt, or a re-extraction would clobber assigned deal values
# (e.g. reset deal_date to the always-NULL staged value).
_DEAL_COLS = {"deal_id", "deal_name", "document_id", "deal_date"}


def _rows(cur, sql, params=()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _table_columns(cur, schema_table: str) -> list[str]:
    schema, table = schema_table.split(".", 1)
    cur.execute(
        "select column_name from information_schema.columns "
        "where table_schema=%s and table_name=%s order by ordinal_position",
        (schema, table))
    return [r[0] for r in cur.fetchall()]


# SQL fragment that normalizes a po_id column the same way as _norm_po()
# (strip non-alphanumerics, lowercase, drop a leading 'po').
_PO_NORM_SQL = "regexp_replace(regexp_replace(lower({col}), '[^a-z0-9]', '', 'g'), '^po', '')"


def _find_parent_po(cur, po_id) -> Optional[dict]:
    """Resolve the parent PO by canonical PO number, tolerant of PO-prefix and
    separator differences between the reference and the PO table."""
    npo = _norm_po(po_id)
    if npo is None:
        return None
    cond = _PO_NORM_SQL.format(col="po_id")
    for tbl in (_PO["trgt"], _PO["stg"]):
        rows = _rows(cur, f"select * from {tbl} where {cond} = %s limit 1", (npo,))
        if rows:
            return rows[0]
    return None


def _copyable_cols(cur, stg_table: str, trgt_table: str) -> list[str]:
    common = set(_table_columns(cur, stg_table)) & set(_table_columns(cur, trgt_table))
    return [c for c in _table_columns(cur, stg_table) if c in common and c not in _DEAL_COLS]


def _upsert(cur, trgt_table: str, pk: str, row: dict, cols: list[str]) -> None:
    """Insert if PK absent (lets the SQL trigger set deal_id); else UPDATE the
    non-deal columns, PRESERVING any trigger-assigned deal_id/deal_name/document_id."""
    pk_val = row.get(pk)
    cur.execute(f"select 1 from {trgt_table} where {pk} = %s limit 1", (pk_val,))
    exists = cur.fetchone() is not None
    if exists:
        set_cols = [c for c in cols if c != pk]
        cur.execute(
            f"update {trgt_table} set " + ", ".join(f"{c}=%s" for c in set_cols) + f" where {pk}=%s",
            [row.get(c) for c in set_cols] + [pk_val])
    else:
        cur.execute(
            f"insert into {trgt_table} (" + ", ".join(cols) + ") values (" +
            ", ".join(["%s"] * len(cols)) + ")",
            [row.get(c) for c in cols])


def _copy_lines(cur, pk: str, pk_val, lines_stg: str, lines_trgt: str,
                canonical_po: Optional[str] = None) -> int:
    cols = _copyable_cols(cur, lines_stg, lines_trgt)
    if pk not in cols:
        # line tables are keyed by the parent pk; ensure it's carried
        cols = [pk] + [c for c in cols if c != pk] if pk in _table_columns(cur, lines_trgt) else cols
    rows = _rows(cur, f"select * from {lines_stg} where {pk} = %s", (pk_val,))
    cur.execute(f"delete from {lines_trgt} where {pk} = %s", (pk_val,))
    for r in rows:
        if canonical_po is not None and "po_id" in cols:
            r = {**r, "po_id": canonical_po}  # canonicalize the line's PO reference
        cur.execute(
            f"insert into {lines_trgt} (" + ", ".join(cols) + ") values (" +
            ", ".join(["%s"] * len(cols)) + ")",
            [r.get(c) for c in cols])
    return len(rows)


def promote_ready(conn: Any = None, doc_types=("invoice", "quote"), limit: Optional[int] = None) -> dict:
    """Score every not-yet-promoted _stg invoice/quote against its parent PO and
    promote to _trgt when extraction confidence and link score both pass."""
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                result = _promote(own, doc_types, limit)
                own.commit()
                return result
            except Exception:
                own.rollback()
                raise
    return _promote(conn, doc_types, limit)


def _set_amount_for_invoice(cur, po_id) -> Optional[float]:
    """Sum of all invoices (stg+trgt, deduped by invoice_id) on this PO — for N:1.
    Matches on the canonical PO number so prefixed/bare invoice po_ids all count."""
    npo = _norm_po(po_id)
    inv_cond = _PO_NORM_SQL.format(col="po_id")
    cur.execute(
        "select coalesce(sum(amt),0) from ("
        "  select invoice_id, max(converted_amount_usd) amt from ("
        f"    select invoice_id, converted_amount_usd from proc.bp_invoice_stg where {inv_cond}=%s"
        "    union all"
        f"    select invoice_id, converted_amount_usd from proc.bp_invoice_trgt where {inv_cond}=%s"
        "  ) x group by invoice_id"
        ") y", (npo, npo))
    return _to_float(cur.fetchone()[0])


def _resolve_po_supplier_id(cur, po: dict) -> None:
    """Give the PO the supplier_id the link scorer compares against.

    The scorer matches src.supplier_id to tgt.supplier_id, and
    bp_purchase_order_stg HAS NO supplier_id COLUMN — it carries supplier_name. So the
    supplier signal came back MISSING (q=0.00) for every invoice/PO pair ever scored, and
    the strongest evidence in the whole comparison contributed nothing.

    Measured on a correct pair (invoice → its own PO, right supplier, matching lines):
    F = 15.4 against a promotion gate of 80. Nothing could ever link, so nothing could ever
    become a deal.

    Resolve the name to the master's id here, at comparison time. No schema change: the PO
    row is enriched in memory, and what gets written to _trgt is untouched.
    """
    if po.get("supplier_id") or not po.get("supplier_name"):
        return
    name = str(po["supplier_name"]).strip()
    if not name:
        return
    cur.execute(
        "SELECT supplier_id FROM proc.bp_supplier "
        "WHERE lower(supplier_name) = lower(%s) LIMIT 1",
        (name,),
    )
    hit = cur.fetchone()
    if hit:
        po["supplier_id"] = hit[0]
        return
    # Fuzzy fallback, so an "Ltd"/"Ld" difference between the PO and the invoice does not
    # silently kill the signal. Deliberately does NOT create a supplier: scoring a link is
    # a read, and an evaluation pass must not mint master data as a side effect.
    try:
        from rapidfuzz import fuzz

        from src.services.extraction_v3.supplier_resolver import _strip_biz_suffix

        cur.execute("SELECT supplier_id, supplier_name FROM proc.bp_supplier")
        stem = _strip_biz_suffix(name)
        best, best_score = None, 0.0
        for sid, sname in cur.fetchall():
            if not sname:
                continue
            score = fuzz.WRatio(stem, _strip_biz_suffix(sname))
            if score > best_score:
                best, best_score = sid, score
        if best and best_score >= 92:  # same bar the extractor uses to auto-link
            po["supplier_id"] = best
    except Exception:  # noqa: BLE001 — a resolver failure must not break promotion
        log.debug("could not resolve PO supplier %r to an id", name, exc_info=True)


def _evaluate(cur, doc_type: str, row: dict) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
    """Score one staged row against its parent PO. Returns (po, link, reason).
    ``reason`` is set when the row fails a gate (held); None means promotable."""
    cfg = _DOC[doc_type]
    pk, pk_val = cfg["pk"], row[cfg["pk"]]
    po = _find_parent_po(cur, row.get("po_id"))
    if po is not None:
        _resolve_po_supplier_id(cur, po)
    conf = _to_float(row.get("confidence_score")) or 0.0
    if row.get("po_id") is None:
        # A quote is raised BEFORE the PO exists, so carrying no PO reference is the normal
        # state of a standalone quote, not a defect. Holding for a parent that has not been
        # raised yet kept every uploaded quote out of _trgt permanently -- and _trgt is what
        # the product reads, so the document was invisible in the UI forever. There is
        # nothing to link against, so it promotes on extraction confidence alone.
        #
        # Invoices still require their PO: that reference is the three-way match, and an
        # invoice without one is a genuine exception for a human to look at.
        if doc_type == "quote":
            if conf < MIN_CONFIDENCE:
                return None, None, "low_extraction_confidence"
            return None, None, None
        return None, None, "no_parent_reference"
    if po is None:
        return None, None, "parent_not_found"
    src_lines = _rows(cur, f"select * from {cfg['lines_stg']} where {pk} = %s", (pk_val,))
    tgt_lines = _rows(cur, "select * from proc.bp_po_line_items_stg where po_id = %s", (po["po_id"],))
    set_amount = _set_amount_for_invoice(cur, po["po_id"]) if doc_type == "invoice" else None
    link = score_link(row, po, cfg["profile"], src_lines, tgt_lines, set_amount_usd=set_amount)
    if conf < MIN_CONFIDENCE:
        return po, link, "low_extraction_confidence"
    if link["F"] < MIN_LINK_SCORE:
        return po, link, "low_link_score"
    return po, link, None


def _ensure_po_in_trgt(cur, po: dict) -> None:
    """Make sure the parent PO (the deal anchor) is itself in _trgt, so a promoted
    invoice/quote is never an orphan. Copies from _stg if absent; preserves any
    trigger-set deal columns on update."""
    cfg = _PO
    rows = _rows(cur, f"select * from {cfg['stg']} where po_id = %s", (po["po_id"],))
    src = rows[0] if rows else po  # prefer the staged source; fall back to the row we have
    cols = _copyable_cols(cur, cfg["stg"], cfg["trgt"])
    cols = [c for c in cols if c in src]
    _upsert(cur, cfg["trgt"], cfg["pk"], src, cols)
    _copy_lines(cur, cfg["pk"], po["po_id"], "proc.bp_po_line_items_stg",
                "proc.bp_po_line_items_trgt")


def _do_copy(cur, doc_type: str, row: dict, po: Optional[dict] = None) -> int:
    """Copy a staged doc + its line items into _trgt (deal cols excluded). When a
    parent PO is given, canonicalize the doc's po_id to the PO's bare number so
    the whole deal shares one join key, and ensure the PO anchor is in _trgt."""
    cfg = _DOC[doc_type]
    canonical_po = None
    if po is not None:
        canonical_po = po["po_id"]
        row = {**row, "po_id": canonical_po}
        _ensure_po_in_trgt(cur, po)
    cols = _copyable_cols(cur, cfg["stg"], cfg["trgt"])
    _upsert(cur, cfg["trgt"], cfg["pk"], row, cols)
    return _copy_lines(cur, cfg["pk"], row[cfg["pk"]], cfg["lines_stg"], cfg["lines_trgt"],
                       canonical_po=canonical_po)


def _promote_purchase_orders(cur) -> tuple[int, int]:
    """Promote staged POs on their own extraction confidence. Returns (promoted, held).

    A purchase order has no parent to be scored against — it IS the parent, the anchor the
    whole deal hangs off. So it was never in the promotion config at all: a PO only reached
    _trgt as a side-effect of some invoice or quote promoting against it
    (`_ensure_po_in_trgt`). Upload 46 purchase orders and nothing else, and the Purchase
    Orders screen stays empty, because _trgt is the only tier the product reads.

    Same shape of bug as quotes being held for a PO that had not been raised yet: a document
    that is nobody's child was treated as though it must be somebody's child.
    """
    promoted = held = 0
    rows = _rows(cur,
                 f"select * from {_PO['stg']} s where s.po_id is not null "
                 f"and not exists (select 1 from {_PO['trgt']} t where t.po_id = s.po_id)")
    for row in rows:
        conf = _to_float(row.get("confidence_score")) or 0.0
        if conf < MIN_CONFIDENCE:
            held += 1
            record_action(
                phase=PHASE_CONSOLIDATION, action_type="promote_held",
                doc_type="purchase_order", doc_pk=str(row["po_id"]), agent="linking_engine",
                status="skipped", confidence=conf,
                summary=f"held purchase_order {row['po_id']}: low_extraction_confidence",
                details={"reason": "low_extraction_confidence", "confidence": conf},
                conn=cur.connection)
            continue
        _ensure_po_in_trgt(cur, row)
        promoted += 1
        record_action(
            phase=PHASE_CONSOLIDATION, action_type="promote_to_trgt",
            doc_type="purchase_order", doc_pk=str(row["po_id"]), agent="linking_engine",
            status="ok", confidence=conf,
            summary=f"promoted purchase_order {row['po_id']} (anchor: nothing to link against)",
            details={"decision": "anchor", "confidence": conf},
            conn=cur.connection)
    return promoted, held


def _promote(conn, doc_types, limit) -> dict:
    cur = conn.cursor()
    promoted, held = 0, 0
    by_reason: dict[str, int] = {}
    details: list[dict] = []

    # Anchors first: a child promoting needs its PO present, and a PO uploaded on its own
    # must still reach the product.
    po_promoted, po_held = _promote_purchase_orders(cur)
    promoted += po_promoted
    held += po_held
    if po_held:
        by_reason["low_extraction_confidence"] = by_reason.get("low_extraction_confidence", 0) + po_held
    if po_promoted or po_held:
        details.append({"doc_type": "purchase_order", "promoted": po_promoted, "held": po_held})

    for doc_type in doc_types:
        cfg = _DOC[doc_type]
        pk = cfg["pk"]
        cand = _rows(cur,
                     f"select * from {cfg['stg']} s where s.{pk} is not null "
                     f"and not exists (select 1 from {cfg['trgt']} t where t.{pk} = s.{pk})"
                     + (f" limit {int(limit)}" if limit else ""))
        for row in cand:
            pk_val = row[pk]
            po, link, reason = _evaluate(cur, doc_type, row)

            if reason is None:  # PROMOTE
                n_lines = _do_copy(cur, doc_type, row, po=po)
                promoted += 1
                # A standalone quote promotes with no parent, so there is no link to score.
                # Everything below used to dereference link/po unconditionally.
                warn = bool(link) and link["F"] < _BAND_AUTO
                if link is not None:
                    summary = f"promoted {doc_type} {pk_val} (F={link['F']}, {link['decision']})"
                    det = {"F": link["F"], "decision": link["decision"], "parent_po": po["po_id"],
                           "lines": n_lines, "signals": link["signals"],
                           "P_raw": link["P_raw"], "C": link["C"], "Q": link["Q"]}
                else:
                    summary = f"promoted {doc_type} {pk_val} (standalone: no parent PO raised yet)"
                    det = {"F": None, "decision": "unlinked", "parent_po": None,
                           "lines": n_lines}
                record_action(
                    phase=PHASE_CONSOLIDATION, action_type="promote_to_trgt",
                    doc_type=doc_type, doc_pk=str(pk_val), agent="linking_engine",
                    status="warn" if warn else "ok",
                    confidence=(link["F"] if link else None),
                    summary=summary, details=det, conn=conn)
                details.append({"doc_type": doc_type, "doc_pk": pk_val, "action": "promoted",
                                "F": (link["F"] if link else None),
                                "decision": (link["decision"] if link else "unlinked")})
            else:  # HELD
                held += 1
                by_reason[reason] = by_reason.get(reason, 0) + 1
                record_action(
                    phase=PHASE_CONSOLIDATION, action_type="promote_held",
                    doc_type=doc_type, doc_pk=str(pk_val), agent="linking_engine",
                    status="skipped", confidence=(link["F"] if link else None),
                    summary=f"held {doc_type} {pk_val}: {reason}",
                    details={"reason": reason,
                             "F": link["F"] if link else None,
                             "decision": link["decision"] if link else None,
                             "signals": link["signals"] if link else None},
                    conn=conn)
                details.append({"doc_type": doc_type, "doc_pk": pk_val, "action": "held",
                                "reason": reason, "F": link["F"] if link else None})

    return {"promoted": promoted, "held": held, "by_reason": by_reason, "details": details}


# ---------------------------------------------------------------------------
# Canonicalize PO references already in _trgt so a deal's docs share one join key
# ---------------------------------------------------------------------------
def canonicalize_po_references(conn: Any = None) -> dict:
    """Align po_id across _trgt to the bare PO number of the PO it resolves to,
    so quote.po_id == invoice.po_id == purchase_order.po_id for the same PO and
    the deal-grouping trigger can join cleanly. Only rewrites a value when it
    resolves to an existing PO and currently differs (e.g. 'PO506789' -> '506789').
    Never touches deal columns. Read-modify on _trgt only."""
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                result = _canon_po(own)
                own.commit()
                return result
            except Exception:
                own.rollback()
                raise
    return _canon_po(conn)


def _canon_po(conn) -> dict:
    cur = conn.cursor()
    updated: dict[str, int] = {}
    for doc_type, cfg in _DOC.items():
        pk = cfg["pk"]
        n = 0
        rows = _rows(cur, f"select {pk}, po_id from {cfg['trgt']} where po_id is not null")
        for r in rows:
            po = _find_parent_po(cur, r["po_id"])
            if po is None or po["po_id"] == r["po_id"]:
                continue
            canon = po["po_id"]
            cur.execute(f"update {cfg['trgt']} set po_id=%s where {pk}=%s", (canon, r[pk]))
            # line items carry po_id too on some tables
            if "po_id" in _table_columns(cur, cfg["lines_trgt"]):
                cur.execute(f"update {cfg['lines_trgt']} set po_id=%s where {pk}=%s", (canon, r[pk]))
            n += 1
        updated[doc_type] = n
    return {"updated": updated, "total": sum(updated.values())}


# ---------------------------------------------------------------------------
# Quote-anchored traversal: quote (sourcing) -> PO (award) -> invoices (billing)
# ---------------------------------------------------------------------------
def quote_chains(conn: Any = None) -> dict:
    """Procurement starts with the quote. For every distinct quote, resolve its
    PO (canonical PO number, prefix-tolerant) and the invoices billed against
    that PO, with the deterministic quote->PO link score. Returns the full
    quote->PO->invoice picture plus connectivity stats. Read-only."""
    if conn is None:
        with get_conn() as own:
            return _quote_chains(own)
    return _quote_chains(conn)


def _quote_chains(conn) -> dict:
    cur = conn.cursor()
    quotes = _rows(cur, """
        select quote_id,
               max(po_id) po_id, max(supplier_id) supplier_id,
               max(converted_amount_usd) converted_amount_usd, max(currency) currency,
               max(confidence_score) confidence_score, max(quote_date) quote_date,
               max(country) country, max(region) region
          from (
            select quote_id, po_id, supplier_id, converted_amount_usd, currency,
                   confidence_score, quote_date, country, region
              from proc.bp_quote_stg where quote_id is not null
            union all
            select quote_id, po_id, supplier_id, converted_amount_usd, currency,
                   confidence_score, quote_date, country, region
              from proc.bp_quote_trgt where quote_id is not null
          ) u group by quote_id order by quote_id""")

    inv_cond = _PO_NORM_SQL.format(col="po_id")
    chains, linked, orphan = [], 0, 0
    for qrow in quotes:
        qid = qrow["quote_id"]
        po = _find_parent_po(cur, qrow.get("po_id"))
        if po is None:
            orphan += 1
            chains.append({"quote_id": qid, "supplier_id": qrow.get("supplier_id"),
                           "po_ref": qrow.get("po_id"), "purchase_order": None,
                           "invoices": [], "link": None,
                           "status": "no_linked_po"})
            continue
        linked += 1
        invs = _rows(cur,
                     f"select invoice_id, converted_amount_usd, invoice_date from proc.bp_invoice_stg where {inv_cond}=%s "
                     f"union select invoice_id, converted_amount_usd, invoice_date from proc.bp_invoice_trgt where {inv_cond}=%s",
                     (_norm_po(po["po_id"]), _norm_po(po["po_id"])))
        qlines = _rows(cur, "select * from proc.bp_quote_line_items_stg where quote_id=%s", (qid,))
        plines = _rows(cur, "select * from proc.bp_po_line_items_stg where po_id=%s", (po["po_id"],))
        link = score_link(qrow, po, "quote_po", qlines, plines)
        chains.append({
            "quote_id": qid, "supplier_id": qrow.get("supplier_id"),
            "po_ref": qrow.get("po_id"),
            "purchase_order": {"po_id": po["po_id"], "supplier_id": po.get("supplier_id"),
                               "converted_amount_usd": _to_float(po.get("converted_amount_usd"))},
            "invoices": [{"invoice_id": i["invoice_id"],
                          "converted_amount_usd": _to_float(i.get("converted_amount_usd"))}
                         for i in invs],
            "link": {"F": link["F"], "decision": link["decision"]},
            "status": "linked",
        })
    return {
        "total_quotes": len(quotes),
        "linked_to_po": linked,
        "orphan_quotes": orphan,
        "chains": chains,
    }


# ---------------------------------------------------------------------------
# Human review queue
# ---------------------------------------------------------------------------
def review_queue(conn: Any = None, doc_types=("invoice", "quote"),
                 min_score: Optional[float] = None) -> list[dict]:
    """List not-yet-promoted staged docs in the REVIEW band — a verified parent
    link with REVIEW_MIN <= F < MIN_LINK_SCORE — with their gap report, for human
    approval. Read-only (no writes)."""
    floor = REVIEW_MIN if min_score is None else float(min_score)
    if conn is None:
        with get_conn() as own:
            return _review_queue(own, doc_types, floor)
    return _review_queue(conn, doc_types, floor)


def _review_queue(conn, doc_types, floor) -> list[dict]:
    cur = conn.cursor()
    out: list[dict] = []
    for doc_type in doc_types:
        cfg = _DOC[doc_type]
        pk = cfg["pk"]
        cand = _rows(cur,
                     f"select * from {cfg['stg']} s where s.{pk} is not null "
                     f"and not exists (select 1 from {cfg['trgt']} t where t.{pk} = s.{pk})")
        for row in cand:
            po, link, reason = _evaluate(cur, doc_type, row)
            if link is None or not (floor <= link["F"] < MIN_LINK_SCORE):
                continue
            # gap report: signals sorted by impact = |w * r * (2s-1)| (PDF Stage 8)
            gaps = sorted(link["signals"], key=lambda s: abs(s["c"]), reverse=True)
            out.append({
                "doc_type": doc_type, "doc_pk": row[pk], "parent_po": po["po_id"],
                "F": link["F"], "decision": link["decision"],
                "confidence_score": _to_float(row.get("confidence_score")),
                "supplier_id": row.get("supplier_id"),
                "amount_usd": _to_float(row.get("converted_amount_usd")),
                "weak_or_conflicting": [g for g in gaps if g["status"] not in ("OK",)],
                "gap_report": gaps,
            })
    out.sort(key=lambda x: x["F"], reverse=True)
    return out


def approve_promotion(doc_type: str, doc_pk: str, reviewer: Optional[str] = None,
                      note: Optional[str] = None, conn: Any = None) -> dict:
    """Human override: force-promote a specific staged doc to _trgt regardless of
    the F gate, recording who approved it and the score at approval time."""
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                result = _approve(own, doc_type, doc_pk, reviewer, note)
                own.commit()
                return result
            except Exception:
                own.rollback()
                raise
    return _approve(conn, doc_type, doc_pk, reviewer, note)


def _approve(conn, doc_type: str, doc_pk: str, reviewer, note) -> dict:
    if doc_type not in _DOC:
        return {"status": "error", "detail": f"unknown doc_type {doc_type}"}
    cfg = _DOC[doc_type]
    pk = cfg["pk"]
    cur = conn.cursor()
    rows = _rows(cur, f"select * from {cfg['stg']} where {pk} = %s", (doc_pk,))
    if not rows:
        return {"status": "not_found", "detail": f"{doc_type} {doc_pk} not in staging"}
    row = rows[0]
    po, link, reason = _evaluate(cur, doc_type, row)
    n_lines = _do_copy(cur, doc_type, row, po=po)
    record_action(
        phase=PHASE_CONSOLIDATION, action_type="promote_approved",
        doc_type=doc_type, doc_pk=str(doc_pk), agent=reviewer or "human_review",
        status="ok", confidence=(link["F"] if link else None),
        summary=f"human-approved promotion of {doc_type} {doc_pk}"
                + (f" (F={link['F']}, was {reason})" if link else ""),
        details={"approved_by": reviewer or "human_review", "note": note,
                 "F": link["F"] if link else None,
                 "decision": link["decision"] if link else None,
                 "held_reason": reason, "parent_po": po["po_id"] if po else None,
                 "lines": n_lines, "signals": link["signals"] if link else None},
        conn=conn)
    return {"status": "promoted", "doc_type": doc_type, "doc_pk": doc_pk,
            "F": link["F"] if link else None, "decision": link["decision"] if link else None,
            "approved_by": reviewer or "human_review", "lines": n_lines}
