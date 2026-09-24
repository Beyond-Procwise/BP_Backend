"""The procure-to-pay checks (spec §5.2).

Each takes (ds, links, cfg) and returns Results, one outcome per compared pair. None
reads the database, the clock or the network.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Optional

from .model import Doc, DocumentSet, Links, Outcome, Result
from .normalise import norm_text, similarity, terms_days
from .tolerance import resolve_tolerance

ZERO = Decimal("0")


def _s(value) -> Optional[str]:
    return None if value is None else str(value)


def _doc_conf(*docs: Optional[Doc]) -> float:
    c = 1.0
    for d in docs:
        if d is not None and d.confidence is not None:
            c *= d.confidence
    return c


def _fail(cfg, link_conf: float, *docs: Optional[Doc]) -> Outcome:
    """CONFLICT, unless a misread number or a weak link could explain it."""
    low_doc = any(d is not None and d.confidence is not None
                  and d.confidence < cfg["min_extraction_confidence"] for d in docs)
    if low_doc or link_conf < cfg["min_link_confidence"]:
        return Outcome.UNVERIFIABLE
    return Outcome.CONFLICT


def _r(ds: DocumentSet, rule: str, cls: str, outcome: Outcome, claim: Doc, field: str,
       **kw) -> Result:
    kw.setdefault("currency", claim.currency)
    kw.setdefault("fx_to_gbp", claim.fx_to_gbp)
    kw.setdefault("fx_rate_date", claim.fx_rate_date)
    kw.setdefault("basis_total", claim.gross if claim.gross is not None else claim.net)
    return Result(deal_id=ds.deal_id, rule_id=rule, field_class=cls, outcome=outcome,
                  claim_doc=claim.doc_id, field_name=field, **kw)


# --- 1. unit price ---------------------------------------------------------------

def check_unit_price(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for lk in links.line_links:
        if lk.po_line is None or lk.rollup or lk.invoice.is_credit_note:
            continue
        claim = lk.inv_line.unit_price
        auth, auth_doc = lk.po_line.unit_price, lk.po.doc_id
        if auth is None:
            q = links.po_quote.get(lk.po.doc_id)
            ql = next((l for l in q.lines if l.item_id and l.item_id == lk.po_line.item_id),
                      None) if q else None
            if ql is not None and ql.unit_price is not None:
                auth, auth_doc = ql.unit_price, q.doc_id
        if claim is None and auth is None:
            continue
        common = dict(claim_line=lk.inv_line.line_ref, auth_doc=auth_doc,
                      auth_line=lk.po_line.line_ref, po_id=lk.po.doc_id,
                      claim_value=_s(claim), auth_value=_s(auth),
                      confidence=_doc_conf(lk.invoice, lk.po) * lk.confidence)
        if claim is None:
            out.append(_r(ds, "unit_price", "money", Outcome.ABSENT_SUBORDINATE,
                          lk.invoice, "unit_price", **common))
            continue
        if auth is None:
            out.append(_r(ds, "unit_price", "money", Outcome.ABSENT_AUTHORITATIVE,
                          lk.invoice, "unit_price", **common))
            continue
        diff = claim - auth
        if diff == 0:
            out.append(_r(ds, "unit_price", "money", Outcome.MATCH, lk.invoice,
                          "unit_price", delta=diff, **common))
            continue
        tol = resolve_tolerance("unit_price_over" if diff > 0 else "unit_price_under", cfg)
        allow = tol.allowance(auth, lk.invoice.fx_to_gbp)
        outcome = (Outcome.WITHIN_TOL if abs(diff) <= allow
                   else _fail(cfg, lk.confidence, lk.invoice, lk.po))
        qty = lk.inv_line.quantity or ZERO
        out.append(_r(ds, "unit_price", "money", outcome, lk.invoice, "unit_price",
                      delta=diff, exposure=abs(diff * qty),
                      tolerance={**tol.as_dict(), "allowance": str(allow)}, **common))
    return out


# --- 2. quantity (cumulative across invoices) -------------------------------------

def check_quantity(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    by_po_line = defaultdict(list)
    for lk in links.line_links:
        if lk.po_line is None or lk.rollup or lk.inv_line.quantity is None:
            continue
        by_po_line[(lk.po.doc_id, lk.po_line.line_ref)].append(lk)
    out = []
    for (po_id, ref), lks in by_po_line.items():
        po_line, po = lks[0].po_line, lks[0].po
        if po_line.quantity is None:
            continue
        cum = sum(((-abs(l.inv_line.quantity)) if l.invoice.is_credit_note
                   else l.inv_line.quantity) for l in lks)
        last = max(lks, key=lambda l: (l.invoice.doc_date or date.min, l.invoice.doc_id,
                                       l.inv_line.line_ref))
        over = cum - po_line.quantity
        link_conf = min(l.confidence for l in lks)
        invoices = ", ".join(sorted({l.invoice.doc_id for l in lks}))
        common = dict(claim_line=last.inv_line.line_ref, auth_doc=po_id, auth_line=ref,
                      po_id=po_id, claim_value=_s(cum), auth_value=_s(po_line.quantity),
                      delta=over, confidence=link_conf * _doc_conf(last.invoice, po))
        if over <= 0:
            if over == 0 and len(lks) == 1:
                outcome, note = Outcome.MATCH, ""
            elif over < 0:
                outcome, note = Outcome.EXPLAINED, f"partially invoiced: {cum} of {po_line.quantity}"
            else:
                outcome, note = Outcome.EXPLAINED, f"invoiced across {len(lks)} lines ({invoices})"
            out.append(_r(ds, "quantity", "quantity", outcome, last.invoice, "quantity",
                          note=note, **common))
            continue
        tol = resolve_tolerance("quantity_over", cfg)
        allow = tol.allowance(po_line.quantity, None)
        outcome = (Outcome.WITHIN_TOL if over <= allow
                   else _fail(cfg, link_conf, last.invoice, po))
        price = po_line.unit_price or last.inv_line.unit_price or ZERO
        out.append(_r(ds, "quantity", "quantity", outcome, last.invoice, "quantity",
                      exposure=abs(over * price), note=f"invoiced on {invoices}",
                      tolerance={**tol.as_dict(), "allowance": str(allow)}, **common))
    return out


# --- 3. line arithmetic -----------------------------------------------------------

def check_line_arithmetic(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    tol = cfg["rounding_per_line"]
    out = []
    for inv in ds.invoices:
        for l in inv.lines:
            if l.quantity is None or l.unit_price is None or l.line_amount is None:
                continue
            expected = abs(l.quantity * l.unit_price)
            diff = abs(l.line_amount) - expected
            outcome = (Outcome.MATCH if diff == 0 else
                       Outcome.WITHIN_TOL if abs(diff) <= tol else _fail(cfg, 1.0, inv))
            out.append(_r(ds, "line_arithmetic", "money", outcome, inv, "line_amount",
                          claim_line=l.line_ref, po_id=inv.po_ref,
                          claim_value=_s(l.line_amount), auth_value=_s(expected),
                          delta=diff, exposure=abs(diff), confidence=_doc_conf(inv),
                          tolerance={"rounding": str(tol)}))
    return out


# --- 11. description ----------------------------------------------------------------

def check_description(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for lk in links.line_links:
        if lk.po_line is None or lk.rollup or lk.confidence < 1.0 or lk.invoice.is_credit_note:
            continue
        a, b = lk.inv_line.description, lk.po_line.description
        if not a or not b:
            continue
        sim = similarity(a, b)
        outcome = Outcome.MATCH if sim >= cfg["description_min_similarity"] else Outcome.CONFLICT
        out.append(_r(ds, "description", "description", outcome, lk.invoice, "description",
                      claim_line=lk.inv_line.line_ref, auth_doc=lk.po.doc_id,
                      auth_line=lk.po_line.line_ref, po_id=lk.po.doc_id, claim_value=a,
                      auth_value=b, note=f"similarity {sim:.2f}",
                      confidence=_doc_conf(lk.invoice, lk.po)))
    return out


# --- 12. unlinked lines, and roll-ups -------------------------------------------------

def check_unlinked_lines(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for lk in links.line_links:
        if lk.rollup:
            out.append(_r(ds, "rollup", "reference", Outcome.EXPLAINED, lk.invoice, "line",
                          claim_line=lk.inv_line.line_ref, auth_doc=lk.po.doc_id,
                          auth_line=lk.po_line.line_ref, po_id=lk.po.doc_id,
                          claim_value=lk.inv_line.description,
                          note=f"itemises PO line {lk.po_line.line_ref}"))
        elif lk.po_line is None:
            out.append(_r(ds, "unlinked_line", "money", Outcome.ABSENT_AUTHORITATIVE,
                          lk.invoice, "line", claim_line=lk.inv_line.line_ref,
                          auth_doc=lk.po.doc_id, po_id=lk.po.doc_id,
                          claim_value=lk.inv_line.description or lk.inv_line.item_id,
                          exposure=abs(lk.inv_line.line_amount or ZERO),
                          confidence=_doc_conf(lk.invoice)))
    return out


LINE_CHECKS = (check_unit_price, check_quantity, check_line_arithmetic,
               check_description, check_unlinked_lines)


def run_checks(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    return [r for check in LINE_CHECKS for r in check(ds, links, cfg)]
