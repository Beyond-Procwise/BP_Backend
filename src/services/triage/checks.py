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
        amounts = ({} if lk.inv_line.quantity is None else
                   dict(claim_amount=claim * lk.inv_line.quantity,
                        auth_amount=auth * lk.inv_line.quantity))
        out.append(_r(ds, "unit_price", "money", outcome, lk.invoice, "unit_price",
                      delta=diff, exposure=abs(diff * qty), **amounts,
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
        invs = list({l.invoice.doc_id: l.invoice for l in lks}.values())
        common = dict(claim_line=last.inv_line.line_ref, auth_doc=po_id, auth_line=ref,
                      po_id=po_id, claim_value=_s(cum), auth_value=_s(po_line.quantity),
                      delta=over, confidence=link_conf * _doc_conf(po, *invs))
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
                   else _fail(cfg, link_conf, po, *invs))
        price = po_line.unit_price or last.inv_line.unit_price or ZERO
        # No price on either line: there is no money to show, only units.
        amounts = ({} if not price else
                   dict(claim_amount=cum * price, auth_amount=po_line.quantity * price))
        out.append(_r(ds, "quantity", "quantity", outcome, last.invoice, "quantity",
                      exposure=abs(over * price), note=f"invoiced on {invoices}", **amounts,
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
                          claim_amount=abs(l.line_amount), auth_amount=expected,
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
        elif lk.po_line is None and (lk.invoice.is_credit_note
                                     or (lk.inv_line.line_amount is not None
                                         and lk.inv_line.line_amount < 0)):
            # A credit line reduces what is owed; it never needs a PO line to cover it.
            out.append(_r(ds, "unlinked_line", "money", Outcome.ABSENT_SUBORDINATE,
                          lk.invoice, "line", claim_line=lk.inv_line.line_ref,
                          auth_doc=lk.po.doc_id, po_id=lk.po.doc_id,
                          claim_value=lk.inv_line.description or lk.inv_line.item_id,
                          exposure=ZERO, note="credit line with no PO line",
                          confidence=_doc_conf(lk.invoice)))
        elif lk.po_line is None:
            out.append(_r(ds, "unlinked_line", "money", Outcome.ABSENT_AUTHORITATIVE,
                          lk.invoice, "line", claim_line=lk.inv_line.line_ref,
                          auth_doc=lk.po.doc_id, po_id=lk.po.doc_id,
                          claim_value=lk.inv_line.description or lk.inv_line.item_id,
                          exposure=abs(lk.inv_line.line_amount or ZERO),
                          **({} if lk.inv_line.line_amount is None else
                             dict(claim_amount=abs(lk.inv_line.line_amount),
                                  auth_amount=ZERO)),
                          confidence=_doc_conf(lk.invoice)))
    return out


# --- 4. invoice totals ------------------------------------------------------------------

def check_invoice_totals(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    tol = cfg["rounding_per_line"]
    out = []
    for inv in ds.invoices:
        amounts = [l.line_amount for l in inv.lines]
        if inv.lines and inv.net is not None and all(a is not None for a in amounts):
            total = sum(amounts, ZERO)
            diff = abs(inv.net) - abs(total)
            allow = tol * max(1, len(inv.lines))
            outcome = (Outcome.MATCH if diff == 0 else
                       Outcome.WITHIN_TOL if abs(diff) <= allow else _fail(cfg, 1.0, inv))
            out.append(_r(ds, "invoice_totals", "money", outcome, inv, "net",
                          po_id=inv.po_ref, claim_value=_s(inv.net), auth_value=_s(total),
                          delta=diff, exposure=abs(diff), confidence=_doc_conf(inv),
                          claim_amount=abs(inv.net), auth_amount=abs(total),
                          tolerance={"rounding": str(allow)}))
        if inv.net is not None and inv.tax is not None and inv.gross is not None:
            expected = inv.net + inv.tax
            diff = inv.gross - expected
            outcome = (Outcome.MATCH if diff == 0 else
                       Outcome.WITHIN_TOL if abs(diff) <= tol else _fail(cfg, 1.0, inv))
            out.append(_r(ds, "invoice_totals", "money", outcome, inv, "gross",
                          po_id=inv.po_ref, claim_value=_s(inv.gross),
                          auth_value=_s(expected), delta=diff, exposure=abs(diff),
                          claim_amount=inv.gross, auth_amount=expected,
                          confidence=_doc_conf(inv), tolerance={"rounding": str(tol)}))
    return out


# --- 5. running total against the PO ---------------------------------------------------

def check_cumulative_total(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    by_po = defaultdict(list)
    for inv in ds.invoices:
        p = links.invoice_po.get(inv.doc_id)
        if p is not None:
            by_po[p.doc_id].append(inv)
    out = []
    for p in ds.pos:
        invs = by_po.get(p.doc_id)
        if not invs or p.net is None:
            continue
        total = sum((i.net for i in invs if i.net is not None), ZERO)
        over = total - p.net
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=_s(total),
                      auth_value=_s(p.net), delta=over, claim_amount=total,
                      auth_amount=p.net,
                      note="invoiced by " + ", ".join(sorted(i.doc_id for i in invs)),
                      confidence=_doc_conf(p, *invs))
        if over <= 0:
            outcome = Outcome.MATCH if over == 0 else Outcome.EXPLAINED
            out.append(_r(ds, "cumulative_total", "money", outcome, p, "net", **common))
            continue
        tol = resolve_tolerance("cumulative_total", cfg)
        allow = tol.allowance(p.net, p.fx_to_gbp)
        outcome = Outcome.WITHIN_TOL if over <= allow else _fail(cfg, 1.0, p, *invs)
        out.append(_r(ds, "cumulative_total", "money", outcome, p, "net", exposure=over,
                      tolerance={**tol.as_dict(), "allowance": str(allow)}, **common))
    return out


# --- 6. tax rate ----------------------------------------------------------------------

def check_tax_rate(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    rates = cfg["allowed_tax_rates"]
    out = []
    for inv in ds.invoices:
        if inv.net is None or inv.net == 0:
            continue
        if inv.tax is None:
            out.append(_r(ds, "tax_rate", "money", Outcome.UNVERIFIABLE, inv, "tax",
                          po_id=inv.po_ref, note="no tax amount", confidence=_doc_conf(inv)))
            continue
        implied = inv.tax / inv.net * 100
        nearest = min(rates, key=lambda r: abs(r - implied))
        expected = (inv.net * nearest / 100).quantize(Decimal("0.01"))
        diff = inv.tax - expected
        allow = cfg["rounding_per_line"] * max(1, len(inv.lines))
        outcome = (Outcome.MATCH if diff == 0 else
                   Outcome.WITHIN_TOL if abs(diff) <= allow else _fail(cfg, 1.0, inv))
        out.append(_r(ds, "tax_rate", "money", outcome, inv, "tax", po_id=inv.po_ref,
                      claim_value=f"{implied:.2f}%", auth_value=f"{nearest}%", delta=diff,
                      claim_amount=inv.tax, auth_amount=expected,
                      exposure=abs(diff), confidence=_doc_conf(inv),
                      tolerance={"allowed_rates": [str(r) for r in rates],
                                 "rounding": str(allow)}))
    return out


# --- 7-9, 13. header fields compared with the PO -----------------------------------------

def _against_po(ds: DocumentSet, links: Links):
    for inv in ds.invoices:
        p = links.invoice_po.get(inv.doc_id)
        if p is not None:
            yield inv, p


def check_currency(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        a, b = (inv.currency or "").strip().upper(), (p.currency or "").strip().upper()
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=a or None,
                      auth_value=b or None, confidence=_doc_conf(inv, p))
        if not b:
            continue                      # the PO states no currency: nothing to compare
        if not a:
            outcome, exposure = Outcome.ABSENT_SUBORDINATE, ZERO
        elif a != b:
            outcome, exposure = Outcome.CONFLICT, abs(inv.net or ZERO)
        else:
            outcome, exposure = Outcome.MATCH, ZERO
        out.append(_r(ds, "currency", "currency", outcome, inv, "currency",
                      exposure=exposure, **common))
    return out


def check_supplier(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        a, b = inv.supplier_id, p.supplier_id
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=a, auth_value=b,
                      confidence=_doc_conf(inv, p))
        if not b:
            continue                      # the PO names no supplier: nothing to compare
        if not a:
            outcome, exposure = Outcome.ABSENT_SUBORDINATE, ZERO
        elif str(a) != str(b):
            outcome, exposure = Outcome.CONFLICT, abs(inv.net or ZERO)
        else:
            outcome, exposure = Outcome.MATCH, ZERO
        out.append(_r(ds, "supplier", "party", outcome, inv, "supplier_id",
                      exposure=exposure, **common))
    return out


def check_invoice_date(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        if inv.doc_date is None or p.doc_date is None:
            continue
        outcome = Outcome.CONFLICT if inv.doc_date < p.doc_date else Outcome.MATCH
        out.append(_r(ds, "invoice_date", "date", outcome, inv, "invoice_date",
                      auth_doc=p.doc_id, po_id=p.doc_id, claim_value=_s(inv.doc_date),
                      auth_value=_s(p.doc_date), confidence=_doc_conf(inv, p)))
    return out


def check_payment_terms(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv, p in _against_po(ds, links):
        if not p.payment_terms:
            continue
        common = dict(auth_doc=p.doc_id, po_id=p.doc_id, claim_value=inv.payment_terms,
                      auth_value=p.payment_terms, confidence=_doc_conf(inv, p))
        if not inv.payment_terms:
            outcome = Outcome.ABSENT_SUBORDINATE
        else:
            da, db = terms_days(inv.payment_terms), terms_days(p.payment_terms)
            if da is not None and db is not None:
                outcome = Outcome.MATCH if da == db else Outcome.CONFLICT
            elif norm_text(inv.payment_terms) == norm_text(p.payment_terms):
                outcome = Outcome.MATCH
            else:
                outcome = Outcome.UNVERIFIABLE
        out.append(_r(ds, "payment_terms", "terms", outcome, inv, "payment_terms", **common))
    return out


# --- 10. duplicates (read from the duplicate detector) -----------------------------------

def check_duplicates(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    invs = {i.doc_id: i for i in ds.invoices}
    out = []
    for flag in ds.duplicates:
        inv = invs.get(flag.invoice_id)
        if inv is None:
            continue
        p = links.invoice_po.get(inv.doc_id)
        exposure = abs(inv.net) if inv.net is not None else abs(flag.amount or ZERO)
        out.append(_r(ds, "duplicate", "money", Outcome.CONFLICT, inv, "invoice_id",
                      auth_doc=flag.earlier_invoice_id, po_id=p.doc_id if p else None,
                      claim_value=inv.doc_id, auth_value=flag.earlier_invoice_id,
                      exposure=exposure, claim_amount=exposure, auth_amount=ZERO,
                      confidence=_doc_conf(inv),
                      note="flagged by the duplicate-invoice detector"))
    return out


# --- 14-15. invoices that do not reach a PO -----------------------------------------------

def check_po_links(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    out = []
    for inv in ds.invoices:
        if inv.doc_id in links.bad_refs:
            out.append(_r(ds, "bad_po_ref", "reference", Outcome.ABSENT_AUTHORITATIVE, inv,
                          "po_id", claim_value=inv.po_ref, exposure=abs(inv.net or ZERO),
                          **({} if inv.net is None else
                             dict(claim_amount=abs(inv.net), auth_amount=ZERO)),
                          confidence=_doc_conf(inv)))
        elif inv.doc_id in links.no_ref:
            out.append(_r(ds, "no_po", "reference", Outcome.ABSENT_AUTHORITATIVE, inv,
                          "po_id", exposure=abs(inv.net or ZERO), confidence=_doc_conf(inv)))
    return out


LINE_CHECKS = (check_unit_price, check_quantity, check_line_arithmetic,
               check_description, check_unlinked_lines)
DOC_CHECKS = (check_invoice_totals, check_cumulative_total, check_tax_rate, check_currency,
              check_supplier, check_invoice_date, check_payment_terms, check_duplicates,
              check_po_links)


def run_checks(ds: DocumentSet, links: Links, cfg) -> list[Result]:
    return [r for check in (*LINE_CHECKS, *DOC_CHECKS) for r in check(ds, links, cfg)]
