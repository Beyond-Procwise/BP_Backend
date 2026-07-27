"""Coherent document chains.

Requirement -> 2-5 competing quotes -> award -> PO -> 1-3 invoices. Deliberately
lossy: not every requirement reaches a PO and not every PO is fully invoiced.
That asymmetry is what test B3 (spend with no purchase order) measures, so it is
a design property rather than an accident.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping, Optional, Sequence

from scripts.testdata.catalogue import CatalogueItem, price_on
from scripts.testdata.org import ENTITIES, CostCentre
from scripts.testdata.rng import make_rng
from scripts.testdata.suppliers import Supplier

DOC_TYPES: tuple[str, ...] = (
    "Requirement", "Quote", "Purchase_Order", "Invoice", "Contract",
)

WINDOW_START = date(2023, 1, 1)
WINDOW_END = date(2026, 7, 31)
VAT_RATE = Decimal("0.20")

# Share of chains that stop before a PO is raised. The invoices on those chains
# become the "spend with no purchase order" population.
NO_PO_SHARE = 0.16


@dataclass(frozen=True)
class LineItem:
    line_number: int
    item_id: str
    description: str
    unit_of_measure: str
    quantity: Optional[Decimal]
    unit_price: Optional[Decimal]
    line_total: Decimal
    currency: str
    leaf_path: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    doc_date: date
    supplier_id: str
    org_id: str
    bu_id: str
    cc_id: str
    currency: str
    net_total: Decimal
    tax_amount: Decimal
    gross_total: Decimal
    lines: tuple[LineItem, ...]
    parent_doc_id: Optional[str]


@dataclass(frozen=True)
class Chain:
    requirement_id: str
    quotes: tuple[Document, ...]
    purchase_order: Optional[Document]
    invoices: tuple[Document, ...]
    awarded_supplier_id: str


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def convert(amount: Decimal, to_currency: str, fx: Mapping[str, float]) -> Decimal:
    """Sterling amount -> to_currency, via the USD base the FX table uses.

    Raises KeyError on an unknown currency: a missing rate must fail loudly
    rather than silently label a sterling figure as something else, which is
    the bug this function exists to fix.
    """
    if to_currency == "GBP":
        return _money(amount)
    usd = Decimal(str(amount)) / Decimal(str(fx["GBP"]))
    return _money(usd * Decimal(str(fx[to_currency])))


def _build_lines(
    rng, items: Sequence[CatalogueItem], when: date, seed: int, currency: str,
    fx: Mapping[str, float],
) -> tuple[LineItem, ...]:
    count = rng.randint(2, 9)
    lines: list[LineItem] = []
    for number in range(1, count + 1):
        item = items[rng.randrange(len(items))]
        unit_price = convert(price_on(item, when, seed=seed), currency, fx)
        quantity = Decimal(str(rng.choice([1, 1, 2, 3, 4, 5, 8, 10, 12, 25, 40])))
        line_total = _money(quantity * unit_price)
        lines.append(
            LineItem(
                line_number=number,
                item_id=item.item_id,
                description=item.description,
                unit_of_measure=item.unit_of_measure,
                quantity=quantity,
                unit_price=unit_price,
                line_total=line_total,
                currency=currency,
                leaf_path=item.leaf.path,
            )
        )
    return tuple(lines)


def _assemble(
    doc_id: str,
    doc_type: str,
    when: date,
    supplier_id: str,
    centre: CostCentre,
    lines: Sequence[LineItem],
    parent_doc_id: Optional[str],
) -> Document:
    net = _money(sum((line.line_total for line in lines), Decimal("0")))
    tax = _money(net * VAT_RATE)
    return Document(
        doc_id=doc_id,
        doc_type=doc_type,
        doc_date=when,
        supplier_id=supplier_id,
        org_id=centre.org_id,
        bu_id=centre.bu_id,
        cc_id=centre.cc_id,
        currency=centre.currency,
        net_total=net,
        tax_amount=tax,
        gross_total=_money(net + tax),
        lines=tuple(lines),
        parent_doc_id=parent_doc_id,
    )


def build_chains(
    seed: int,
    suppliers: Sequence[Supplier],
    catalogue_items: Sequence[CatalogueItem],
    cost_centres: Sequence[CostCentre],
    *,
    fx: Mapping[str, float],
    count: int = 6000,
) -> list[Chain]:
    """Build `count` requirement-to-invoice chains."""
    rng = make_rng(seed, "documents")
    window_days = (WINDOW_END - WINDOW_START).days
    chains: list[Chain] = []

    for index in range(1, count + 1):
        centre = cost_centres[rng.randrange(len(cost_centres))]
        requirement_date = WINDOW_START + timedelta(days=rng.randrange(window_days - 120))
        requirement_id = f"REQ{index:06d}"

        quote_count = rng.randint(2, 5)
        quoting = [suppliers[rng.randrange(len(suppliers))] for _ in range(quote_count)]

        quotes: list[Document] = []
        for position, supplier in enumerate(quoting, start=1):
            quote_date = requirement_date + timedelta(days=rng.randint(3, 21))
            lines = _build_lines(rng, catalogue_items, quote_date, seed, centre.currency, fx)
            quotes.append(
                _assemble(
                    doc_id=f"QUO{index:06d}-{position}",
                    doc_type="Quote",
                    when=quote_date,
                    supplier_id=supplier.bp_supplier_id,
                    centre=centre,
                    lines=lines,
                    parent_doc_id=requirement_id,
                )
            )

        # Award usually, but not always, to the lowest quote. Test C5 measures the
        # exceptions; defects.py converts a controlled number into planted D09 cases.
        awarded_quote = min(quotes, key=lambda quote: quote.net_total)
        awarded_supplier_id = awarded_quote.supplier_id

        purchase_order: Optional[Document] = None
        if rng.random() > NO_PO_SHARE:
            po_date = awarded_quote.doc_date + timedelta(days=rng.randint(2, 30))
            purchase_order = _assemble(
                doc_id=f"PO{index:06d}",
                doc_type="Purchase_Order",
                when=po_date,
                supplier_id=awarded_supplier_id,
                centre=centre,
                lines=awarded_quote.lines,
                parent_doc_id=awarded_quote.doc_id,
            )

        invoice_base = purchase_order.doc_date if purchase_order else awarded_quote.doc_date
        invoices: list[Document] = []
        for position in range(1, rng.randint(1, 3) + 1):
            invoice_date = invoice_base + timedelta(days=rng.randint(1, 75))
            if invoice_date > WINDOW_END:
                invoice_date = WINDOW_END
            invoices.append(
                _assemble(
                    doc_id=f"INV{index:06d}-{position}",
                    doc_type="Invoice",
                    when=invoice_date,
                    supplier_id=awarded_supplier_id,
                    centre=centre,
                    lines=awarded_quote.lines,
                    parent_doc_id=purchase_order.doc_id if purchase_order else None,
                )
            )

        chains.append(
            Chain(
                requirement_id=requirement_id,
                quotes=tuple(quotes),
                purchase_order=purchase_order,
                invoices=tuple(invoices),
                awarded_supplier_id=awarded_supplier_id,
            )
        )
    return chains
