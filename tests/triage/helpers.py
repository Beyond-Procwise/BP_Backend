"""Hand-built deals for triage unit tests. No database, ever."""
from __future__ import annotations

from datetime import date
from decimal import Decimal as D
from typing import Optional

from src.services.triage.model import Doc, DocumentSet, Line
from src.services.triage.tolerance import load_config
from tests.conftest import GOVERNED_LIMIT_SEED


def make_cfg(**overrides):
    rules = dict(GOVERNED_LIMIT_SEED["triage_tolerances"], **overrides)
    return load_config(read=lambda policy, key, cast: cast(rules[key]))


def line(ref, item: Optional[str] = "ITEM-1", qty: Optional[str] = "10",
         price: Optional[str] = "12.00", amount: Optional[str] = None,
         desc: str = "Widget", po_id: Optional[str] = None) -> Line:
    q = D(qty) if qty is not None else None
    p = D(price) if price is not None else None
    if amount is not None:
        amt = D(amount)
    else:
        amt = q * p if q is not None and p is not None else None
    return Line(line_ref=str(ref), item_id=item, description=desc, quantity=q,
                uom="each", unit_price=p, line_amount=amt, po_id=po_id)


def _net(lines, net):
    if net is not None:
        return D(net)
    return sum((l.line_amount or D("0")) for l in lines)


def po(po_id="PO-1", lines=None, net=None, currency="GBP", supplier="SUP-1",
       order_date=date(2026, 1, 10), terms="Net 30", quote_ref=None, fx="1") -> Doc:
    lines = lines if lines is not None else [line(1)]
    n = _net(lines, net)
    tax = (n * D("0.2")).quantize(D("0.01"))
    return Doc(doc_id=po_id, doc_type="purchase_order", supplier_id=supplier,
               currency=currency, doc_date=order_date, net=n, tax=tax, gross=n + tax,
               payment_terms=terms, quote_ref=quote_ref,
               fx_to_gbp=D(fx) if fx else None, lines=lines)


def inv(inv_id="INV-1", po_id="PO-1", lines=None, net=None, tax=None, gross=None,
        currency="GBP", supplier="SUP-1", inv_date=date(2026, 2, 1), terms="30 days",
        fx="1", confidence=None) -> Doc:
    lines = lines if lines is not None else [line(1)]
    n = _net(lines, net)
    t = D(tax) if tax is not None else (n * D("0.2")).quantize(D("0.01"))
    g = D(gross) if gross is not None else n + t
    return Doc(doc_id=inv_id, doc_type="invoice", supplier_id=supplier, currency=currency,
               doc_date=inv_date, net=n, tax=t, gross=g, payment_terms=terms, po_id=po_id,
               confidence=confidence, fx_to_gbp=D(fx) if fx else None, lines=lines)


def quote(quote_id="Q-1", lines=None, currency="GBP", supplier="SUP-1", fx="1") -> Doc:
    lines = lines if lines is not None else [line(1)]
    n = _net(lines, None)
    return Doc(doc_id=quote_id, doc_type="quote", supplier_id=supplier, currency=currency,
               doc_date=date(2026, 1, 1), net=n, fx_to_gbp=D(fx) if fx else None, lines=lines)


def deal(*docs, duplicates=(), deal_id="DEAL-1") -> DocumentSet:
    ds = DocumentSet(deal_id)
    for d in docs:
        {"quote": ds.quotes, "purchase_order": ds.pos, "invoice": ds.invoices}[d.doc_type].append(d)
    ds.duplicates = list(duplicates)
    return ds


def scored(ds, cfg=None):
    """link -> checks -> score, as the engine does it."""
    from src.services.triage.checks import run_checks
    from src.services.triage.link import link
    from src.services.triage.score import score_result
    cfg = cfg or make_cfg()
    links = link(ds, cfg)
    results = run_checks(ds, links, cfg)
    for r in results:
        score_result(r, cfg)
    return links, results
