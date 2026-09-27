"""Which invoice belongs to which PO, and which invoice line to which PO line (spec §5.1).

Also the group match of §6: an invoice's unlinked lines that together equal one PO line
nothing else linked to are a roll-up (EXPLAINED), not N unlinked lines. Pure.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from src.services.extraction.po_revision import latest_approved

from .model import Doc, DocumentSet, Line, LineLink, Links, TermLink
from .normalise import similarity


def _gap(a: Line, b: Line) -> Decimal:
    if a.line_amount is None or b.line_amount is None:
        return Decimal("Infinity")
    return abs(a.line_amount - b.line_amount)


def _link_line(inv: Doc, line: Line, po: Doc, cfg) -> LineLink:
    if line.item_id:
        same = [pl for pl in po.lines if pl.item_id == line.item_id]
        if same:
            exact = [pl for pl in same if pl.unit_price == line.unit_price]
            return LineLink(inv, line, po, (exact or same)[0], 1.0)
    best: Optional[Line] = None
    best_key = None
    for pl in po.lines:
        key = (similarity(line.description, pl.description), -_gap(line, pl))
        if best_key is None or key > best_key:
            best, best_key = pl, key
    score = best_key[0] if best_key else 0.0
    return LineLink(inv, line, po, best if score >= cfg["unlinked_below"] else None,
                    round(score, 4))


def _rollup(links: list[LineLink], po: Doc, cfg) -> list[LineLink]:
    unlinked = [lk for lk in links if lk.po_line is None]
    if len(unlinked) < 2:
        return links
    total = sum((lk.inv_line.line_amount or Decimal("0")) for lk in unlinked)
    used = {id(lk.po_line) for lk in links if lk.po_line is not None}
    tol = cfg["rounding_per_line"] * len(unlinked)
    for pl in po.lines:
        if id(pl) in used or pl.line_amount is None:
            continue
        if abs(pl.line_amount - total) <= tol:
            for lk in unlinked:
                lk.po_line, lk.confidence, lk.rollup = pl, 1.0, True
            break
    return links


def term_for(line: Line, contracts: list[Doc], cfg) -> Optional[tuple[Doc, Line, float]]:
    """The contract term governing this invoice line, with a confidence, or None.

    Same rule as the PO line link: an item-id match is certain, otherwise the best
    description similarity, and below `unlinked_below` there is no link at all.

    Two guards, both load-bearing and both tested by breaking them:

    * A term whose basis could not be read (`term_basis is None`, which is what
      extraction/contract_terms.py leaves rather than guessing) governs nothing. An
      unreadable term must never judge a charge.
    * Two item ids that differ are evidence of a *different* item, so such a term is
      rejected outright rather than falling through to the description. Without that,
      "Gadget" scores 0.67 against "Widget" and a line gets judged against another
      item's cap — which is how this was found.
    """
    best: Optional[tuple[Doc, Line, float]] = None
    for c in contracts:
        for t in c.lines:
            if t.term_basis is None:
                continue
            if line.item_id and t.item_id:
                if line.item_id != t.item_id:
                    continue
                return c, t, 1.0
            score = similarity(line.description, t.description)
            if best is None or score > best[2]:
                best = (c, t, round(score, 4))
    if best is None or best[2] < cfg["unlinked_below"]:
        return None
    return best


def link(ds: DocumentSet, cfg) -> Links:
    pos = {p.doc_id: p for p in ds.pos}
    quotes = {q.doc_id: q for q in ds.quotes}
    out = Links()
    for inv in ds.invoices:
        ref = inv.po_ref
        po = latest_approved(ref, pos) if ref else None
        out.invoice_po[inv.doc_id] = po
        if ref is None:
            out.no_ref.add(inv.doc_id)
            continue
        if po is None:
            out.bad_refs.add(inv.doc_id)
            continue
        out.line_links.extend(_rollup([_link_line(inv, l, po, cfg) for l in inv.lines], po, cfg))
    out.po_quote = {p.doc_id: (quotes.get(p.quote_ref) if p.quote_ref else None) for p in ds.pos}
    for inv in ds.invoices:
        for l in inv.lines:
            found = term_for(l, ds.contracts, cfg)
            if found is not None:
                contract, t, conf = found
                out.term_links.append(TermLink(inv, l, contract, t, conf))
    return out
