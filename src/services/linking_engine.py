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
MIN_CONFIDENCE = float(os.getenv("PROMOTE_MIN_CONFIDENCE", "90"))   # _stg extraction conf
MIN_LINK_SCORE = float(os.getenv("PROMOTE_MIN_LINK_SCORE", "80"))   # F decision band

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


def cmp_exact_ref(a: Any, b: Any) -> tuple[float, str]:
    na, nb = _norm_id(a), _norm_id(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


cmp_exact_id = cmp_exact_ref  # same normalized-equality semantics


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
    """Per-line composite: description overlap, quantity, unit price."""
    desc_a, desc_b = _tokens(a.get("item_description")), _tokens(b.get("item_description"))
    if desc_a or desc_b:
        overlap = len(desc_a & desc_b) / max(1, len(desc_a | desc_b))
    else:
        overlap = 0.0
    qa, qb = _to_float(a.get("quantity")), _to_float(b.get("quantity"))
    qty = 1.0 if (qa is not None and qb is not None and abs(qa - qb) < 1e-9) else 0.0
    pa, pb = _to_float(a.get("unit_price")), _to_float(b.get("unit_price"))
    price = 1.0 if (pa is not None and pb is not None and abs(pa - pb) < 1e-6) else 0.0
    # weights: description 0.4, quantity 0.3, unit price 0.3
    return 0.4 * overlap + 0.3 * qty + 0.3 * price


def cmp_line_composite(src_lines: list[dict], tgt_lines: list[dict]) -> tuple[float, str]:
    if not src_lines or not tgt_lines:
        return 0.5, "MISSING"
    # greedy best-match each source line to a target line
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
    # count coverage: penalise extra/missing lines
    coverage = min(len(src_lines), len(tgt_lines)) / max(len(src_lines), len(tgt_lines))
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


def _signal_match(kind: str, src: dict, tgt: dict, src_lines, tgt_lines, date_field: str) -> tuple[float, str]:
    if kind == "po_ref":
        return cmp_exact_ref(src.get("po_id"), tgt.get("po_id"))
    if kind == "supplier_id":
        return cmp_exact_id(src.get("supplier_id"), tgt.get("supplier_id"))
    if kind == "amount":
        return cmp_numeric_tol(src.get("converted_amount_usd"), tgt.get("converted_amount_usd"))
    if kind == "currency":
        return cmp_exact_ref(src.get("currency"), tgt.get("currency"))
    if kind == "line_set":
        return cmp_line_composite(src_lines, tgt_lines)
    if kind == "temporal":
        return cmp_temporal(src.get(date_field), tgt.get("order_date"), tgt.get("expected_delivery_date"))
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
               source_lines: Optional[list] = None, target_lines: Optional[list] = None) -> dict:
    """Deterministically score the relationship source→target. Returns the full
    auditable result (PDF stages 1A..7C)."""
    profile = PROFILES[profile_name]
    src_lines = source_lines or []
    tgt_lines = target_lines or []

    q_src = (_to_float(source_row.get("confidence_score")) or 0.0) / 100.0
    q_tgt = (_to_float(target_row.get("confidence_score")) or 0.0) / 100.0

    signals = []
    for spec in profile["signals"]:
        s, status = _signal_match(spec["kind"], source_row, target_row, src_lines, tgt_lines,
                                  profile["date_field"])
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
_DEAL_COLS = {"deal_id", "deal_name", "document_id"}  # owned by the SQL trigger


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


def _find_parent_po(cur, po_id) -> Optional[dict]:
    if po_id is None:
        return None
    for tbl in (_PO["trgt"], _PO["stg"]):
        rows = _rows(cur, f"select * from {tbl} where po_id = %s limit 1", (po_id,))
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


def _copy_lines(cur, pk: str, pk_val, lines_stg: str, lines_trgt: str) -> int:
    cols = _copyable_cols(cur, lines_stg, lines_trgt)
    if pk not in cols:
        # line tables are keyed by the parent pk; ensure it's carried
        cols = [pk] + [c for c in cols if c != pk] if pk in _table_columns(cur, lines_trgt) else cols
    rows = _rows(cur, f"select * from {lines_stg} where {pk} = %s", (pk_val,))
    cur.execute(f"delete from {lines_trgt} where {pk} = %s", (pk_val,))
    for r in rows:
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


def _promote(conn, doc_types, limit) -> dict:
    cur = conn.cursor()
    promoted, held = 0, 0
    by_reason: dict[str, int] = {}
    details: list[dict] = []

    for doc_type in doc_types:
        cfg = _DOC[doc_type]
        pk = cfg["pk"]
        cand = _rows(cur,
                     f"select * from {cfg['stg']} s where s.{pk} is not null "
                     f"and not exists (select 1 from {cfg['trgt']} t where t.{pk} = s.{pk})"
                     + (f" limit {int(limit)}" if limit else ""))
        for row in cand:
            pk_val = row[pk]
            reason = None
            link = None
            po = _find_parent_po(cur, row.get("po_id"))
            conf = _to_float(row.get("confidence_score")) or 0.0

            if row.get("po_id") is None:
                reason = "no_parent_reference"
            elif po is None:
                reason = "parent_not_found"
            elif conf < MIN_CONFIDENCE:
                reason = "low_extraction_confidence"
            else:
                src_lines = _rows(cur, f"select * from {cfg['lines_stg']} where {pk} = %s", (pk_val,))
                tgt_lines = _rows(cur, "select * from proc.bp_po_line_items_stg where po_id = %s",
                                  (po["po_id"],))
                link = score_link(row, po, cfg["profile"], src_lines, tgt_lines)
                if link["F"] < MIN_LINK_SCORE:
                    reason = "low_link_score"

            if reason is None:  # PROMOTE
                cols = _copyable_cols(cur, cfg["stg"], cfg["trgt"])
                _upsert(cur, cfg["trgt"], pk, row, cols)
                n_lines = _copy_lines(cur, pk, pk_val, cfg["lines_stg"], cfg["lines_trgt"])
                promoted += 1
                warn = link["F"] < _BAND_AUTO
                record_action(
                    phase=PHASE_CONSOLIDATION, action_type="promote_to_trgt",
                    doc_type=doc_type, doc_pk=str(pk_val), agent="linking_engine",
                    status="warn" if warn else "ok", confidence=link["F"],
                    summary=f"promoted {doc_type} {pk_val} (F={link['F']}, {link['decision']})",
                    details={"F": link["F"], "decision": link["decision"], "parent_po": po["po_id"],
                             "lines": n_lines, "signals": link["signals"],
                             "P_raw": link["P_raw"], "C": link["C"], "Q": link["Q"]},
                    conn=conn)
                details.append({"doc_type": doc_type, "doc_pk": pk_val, "action": "promoted",
                                "F": link["F"], "decision": link["decision"]})
            else:  # HELD
                held += 1
                by_reason[reason] = by_reason.get(reason, 0) + 1
                record_action(
                    phase=PHASE_CONSOLIDATION, action_type="promote_held",
                    doc_type=doc_type, doc_pk=str(pk_val), agent="linking_engine",
                    status="skipped", confidence=(link["F"] if link else None),
                    summary=f"held {doc_type} {pk_val}: {reason}",
                    details={"reason": reason, "confidence_score": conf,
                             "F": link["F"] if link else None,
                             "decision": link["decision"] if link else None,
                             "signals": link["signals"] if link else None},
                    conn=conn)
                details.append({"doc_type": doc_type, "doc_pk": pk_val, "action": "held",
                                "reason": reason, "F": link["F"] if link else None})

    return {"promoted": promoted, "held": held, "by_reason": by_reason, "details": details}
