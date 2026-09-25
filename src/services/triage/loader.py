"""Batch loading of deal document sets from the _trgt tables (spec §4).

A fixed number of queries per batch whatever its size — deal assignment showed that a
query per document does not survive 5,000 deals. Read-only.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from src.services.extraction.po_revision import po_base
from src.services.facts.fx import resolve_fx

from .model import Doc, DocumentSet, DuplicateFlag, Line
from .normalise import to_confidence, to_decimal

BASE_CURRENCY = "GBP"

ALL_DEALS_SQL = """
SELECT deal_id FROM (
    SELECT deal_id FROM proc.bp_purchase_order_trgt
    UNION SELECT deal_id FROM proc.bp_invoice_trgt
    UNION SELECT deal_id FROM proc.bp_quote_trgt
) d WHERE deal_id IS NOT NULL ORDER BY deal_id
"""

_INVOICES = """
SELECT invoice_id, deal_id, po_id, supplier_id, currency, invoice_date,
       invoice_amount, tax_amount, invoice_total_incl_tax, payment_terms, confidence_score
  FROM proc.bp_invoice_trgt WHERE deal_id = ANY(%s)
"""
_INVOICE_LINES = """
SELECT invoice_id, COALESCE(line_no::text, invoice_line_id::text), item_id, item_description,
       quantity, unit_of_measure, unit_price, line_amount, po_id, delivery_date
  FROM proc.bp_invoice_line_items_trgt WHERE invoice_id = ANY(%s)
 ORDER BY invoice_id, line_no
"""
_POS = """
SELECT po_id, deal_id, supplier_id, currency, order_date, total_amount, tax_amount,
       total_amount_incl_tax, payment_terms, quote_reference, confidence_score,
       po_revision, approval_status
  FROM proc.bp_purchase_order_trgt
 WHERE deal_id = ANY(%s) OR po_id = ANY(%s)
    -- every revision of a PO an invoice names: the invoice prints the bare number
    OR regexp_replace(po_id, '\\s*\\(\\s*rev\\M.*$', '', 'i') = ANY(%s)
"""
_PO_LINES = """
SELECT po_id, COALESCE(line_number::text, po_line_id::text), item_id, item_description,
       quantity, unit_of_measure, unit_price, line_total
  FROM proc.bp_po_line_items_trgt WHERE po_id = ANY(%s)
 ORDER BY po_id, line_number
"""
_QUOTES = """
SELECT quote_id, deal_id, supplier_id, currency, quote_date, total_amount, tax_amount,
       total_amount_incl_tax, confidence_score
  FROM proc.bp_quote_trgt WHERE deal_id = ANY(%s) OR quote_id = ANY(%s)
"""
_QUOTE_LINES = """
SELECT quote_id, COALESCE(line_number::text, quote_line_id::text), item_id, item_description,
       quantity, unit_of_measure, unit_price, line_total
  FROM proc.bp_quote_line_items_trgt WHERE quote_id = ANY(%s)
 ORDER BY quote_id, line_number
"""
_DUPLICATES = """
SELECT doc_pk_candidate, raw_value, notes FROM proc.bp_extraction_discrepancy
 WHERE issue_type = 'duplicate_invoice' AND status = 'open' AND doc_pk_candidate = ANY(%s)
"""
_EARLIER = re.compile(r"possible duplicate of (\S+)")


def _s(value) -> Optional[str]:
    return None if value is None else str(value)


def list_deal_ids(cur) -> list[str]:
    cur.execute(ALL_DEALS_SQL)
    return [r[0] for r in cur.fetchall()]


def _line(row) -> Line:
    _doc, ref, item, desc, qty, uom, price, amount, *rest = row
    return Line(line_ref=str(ref), item_id=_s(item), description=desc,
                quantity=to_decimal(qty), uom=uom, unit_price=to_decimal(price),
                line_amount=to_decimal(amount),
                po_id=_s(rest[0]) if rest else None,
                delivery_date=rest[1] if len(rest) > 1 else None)


def _lines_by_doc(cur, sql: str, ids) -> dict[str, list[Line]]:
    out: dict[str, list[Line]] = {}
    if not ids:
        return out
    cur.execute(sql, (list(ids),))
    for row in cur.fetchall():
        out.setdefault(str(row[0]), []).append(_line(row))
    return out


def load_deal_sets(cur, deal_ids: Iterable[str]) -> dict[str, DocumentSet]:
    wanted = list(dict.fromkeys(str(d) for d in deal_ids if d))
    if not wanted:
        return {}
    sets = {d: DocumentSet(d) for d in wanted}

    # Invoices and their lines.
    cur.execute(_INVOICES, (wanted,))
    invoices: list[tuple[str, Doc]] = []
    for (inv_id, deal, po_id, sup, ccy, when, net, tax, gross, terms, conf) in cur.fetchall():
        invoices.append((str(deal), Doc(
            doc_id=str(inv_id), doc_type="invoice", supplier_id=_s(sup), currency=ccy,
            doc_date=when, net=to_decimal(net), tax=to_decimal(tax), gross=to_decimal(gross),
            payment_terms=terms, po_id=_s(po_id), confidence=to_confidence(conf))))
    inv_lines = _lines_by_doc(cur, _INVOICE_LINES, [d.doc_id for _, d in invoices])
    for deal, doc in invoices:
        doc.lines = inv_lines.get(doc.doc_id, [])
        sets[deal].invoices.append(doc)

    # POs in these deals, plus any PO an invoice here names.
    po_refs = sorted({d.po_ref for _, d in invoices if d.po_ref})
    cur.execute(_POS, (wanted, po_refs, sorted({po_base(r) for r in po_refs})))
    pos: dict[str, Doc] = {}
    po_deal: dict[str, Optional[str]] = {}
    for (po_id, deal, sup, ccy, when, net, tax, gross, terms, qref, conf, rev,
         approval) in cur.fetchall():
        pos[str(po_id)] = Doc(
            doc_id=str(po_id), doc_type="purchase_order", supplier_id=_s(sup), currency=ccy,
            doc_date=when, net=to_decimal(net), tax=to_decimal(tax), gross=to_decimal(gross),
            payment_terms=terms, quote_ref=_s(qref), confidence=to_confidence(conf),
            revision=rev, approval=approval)
        po_deal[str(po_id)] = _s(deal)
    po_lines = _lines_by_doc(cur, _PO_LINES, list(pos))
    po_owners: dict[str, set] = {}
    for po_id, doc in pos.items():
        doc.lines = po_lines.get(po_id, [])
        owners = {po_deal[po_id]} | {deal for deal, i in invoices
                                     if i.po_ref and po_base(i.po_ref) == po_base(po_id)}
        po_owners[po_id] = owners
        for owner in owners:
            if owner in sets:
                sets[owner].pos.append(doc)

    # Quotes in these deals, plus any quote a PO here references.
    quote_refs = sorted({d.quote_ref for d in pos.values() if d.quote_ref})
    cur.execute(_QUOTES, (wanted, quote_refs))
    quotes: dict[str, Doc] = {}
    quote_deal: dict[str, Optional[str]] = {}
    for (qid, deal, sup, ccy, when, net, tax, gross, conf) in cur.fetchall():
        quotes[str(qid)] = Doc(
            doc_id=str(qid), doc_type="quote", supplier_id=_s(sup), currency=ccy,
            doc_date=when, net=to_decimal(net), tax=to_decimal(tax), gross=to_decimal(gross),
            confidence=to_confidence(conf))
        quote_deal[str(qid)] = _s(deal)
    quote_lines = _lines_by_doc(cur, _QUOTE_LINES, list(quotes))
    for qid, doc in quotes.items():
        doc.lines = quote_lines.get(qid, [])
        owners = {quote_deal[qid]}
        for po_id, p in pos.items():
            if p.quote_ref == qid:
                owners |= po_owners[po_id]
        for owner in owners:
            if owner in sets:
                sets[owner].quotes.append(doc)

    # Open duplicate findings from the duplicate-invoice detector (read, never re-detected).
    inv_deal = {d.doc_id: deal for deal, d in invoices}
    if inv_deal:
        cur.execute(_DUPLICATES, (list(inv_deal),))
        for pk, raw, notes in cur.fetchall():
            m = _EARLIER.search(notes or "")
            sets[inv_deal[str(pk)]].duplicates.append(
                DuplicateFlag(str(pk), m.group(1) if m else None, to_decimal(raw)))

    # FX to GBP, once per currency per batch.
    rates = {}
    for s in sets.values():
        for doc in (*s.quotes, *s.pos, *s.invoices):
            ccy = (doc.currency or "").strip().upper()
            if not ccy:
                continue
            if ccy not in rates:
                rates[ccy] = resolve_fx(cur, ccy, BASE_CURRENCY)
            doc.fx_to_gbp, doc.fx_rate_date = rates[ccy].rate, rates[ccy].rate_date

    return {d: s for d, s in sets.items() if s.quotes or s.pos or s.invoices}
