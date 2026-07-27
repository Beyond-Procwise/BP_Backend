"""Map generated documents onto the six _trgt tables and bulk-load them.

Plan 1 generated 38,000 documents and discarded them. Without these rows the
price-history pool is empty and the benchmark engine gates on every line, so
persistence is what makes the engine testable at all.

Only columns the generator can actually fill are written. Anything else is left
NULL rather than invented. The purchase-order HEADER table is
bp_purchase_order_trgt; only the line table uses the `po` abbreviation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Sequence

from scripts.testdata.documents import Chain, Document, LineItem
from scripts.testdata.org import ENTITIES

MARKER = "testdata"

# The header tables carry country and region but no business-unit or
# cost-centre column, so entity geography is the most attribution that fits.
ENTITY_COUNTRY: dict[str, str] = {e.org_id: e.country for e in ENTITIES}
ENTITY_REGION: dict[str, str] = {
    "ORG-UK": "Europe",
    "ORG-DE": "Europe",
    "ORG-IE": "Europe",
    "ORG-US": "North America",
    "ORG-IN": "APAC",
    "ORG-AE": "Middle East",
}

COLUMNS: dict[str, tuple[str, ...]] = {
    "bp_quote_trgt": (
        "quote_id", "supplier_id", "buyer_id", "quote_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax", "po_id",
        "country", "region", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "bp_quote_line_items_trgt": (
        "quote_line_id", "quote_id", "line_number", "item_id",
        "item_description", "quantity", "unit_of_measure", "unit_price",
        "line_total", "currency", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "bp_purchase_order_trgt": (
        "po_id", "supplier_id", "buyer_id", "order_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax",
        "quote_reference", "ship_to_country", "delivery_region",
        "created_date", "created_by", "last_modified_by", "last_modified_date",
    ),
    "bp_po_line_items_trgt": (
        "po_line_id", "po_id", "line_number", "item_id", "item_description",
        "quantity", "unit_of_measure", "unit_price", "line_total", "currency",
        "quote_number", "created_date", "created_by", "last_modified_by",
        "last_modified_date",
    ),
    "bp_invoice_trgt": (
        "invoice_id", "po_id", "supplier_id", "buyer_id", "invoice_date",
        "currency", "invoice_amount", "tax_amount", "invoice_total_incl_tax",
        "country", "region", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "bp_invoice_line_items_trgt": (
        "invoice_line_id", "invoice_id", "line_no", "item_id",
        "item_description", "quantity", "unit_of_measure", "unit_price",
        "line_amount", "po_id", "country", "region", "created_date",
        "created_by", "last_modified_by", "last_modified_date",
    ),
}

TABLES: tuple[str, ...] = tuple(COLUMNS)


def _stamp(doc: Document) -> list:
    """created_date, created_by, last_modified_by, last_modified_date.

    Stamped from the document's own date rather than wall-clock time: the build
    must reproduce byte-for-byte from a seed, and now() would break that.

    created_date/last_modified_date and created_by/last_modified_by are each
    intentionally identical pairs (generated rows have no separate edit
    event) — that redundancy is by design, not a copy-paste bug, and a swap
    within either pair is deliberately unobservable from these values alone.
    """
    when = datetime.combine(doc.doc_date, datetime.min.time())
    return [when, MARKER, MARKER, when]


def _quote_rows(doc: Document, po_id: str | None) -> list:
    return [
        doc.doc_id, doc.supplier_id, doc.cc_id, doc.doc_date, doc.currency,
        doc.net_total, doc.tax_amount, doc.gross_total, po_id,
        ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id),
        *_stamp(doc),
    ]


def _quote_line_rows(doc: Document, line: LineItem) -> list:
    return [
        f"{doc.doc_id}-{line.line_number}", doc.doc_id, line.line_number,
        line.item_id, line.description, line.quantity, line.unit_of_measure,
        line.unit_price, line.line_total, line.currency, *_stamp(doc),
    ]


def _po_rows(doc: Document, quote_ref: str | None) -> list:
    return [
        doc.doc_id, doc.supplier_id, doc.cc_id, doc.doc_date, doc.currency,
        doc.net_total, doc.tax_amount, doc.gross_total, quote_ref,
        ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id),
        *_stamp(doc),
    ]


def _po_line_rows(doc: Document, line: LineItem, quote_ref: str | None) -> list:
    return [
        f"{doc.doc_id}-{line.line_number}", doc.doc_id, line.line_number,
        line.item_id, line.description, line.quantity, line.unit_of_measure,
        line.unit_price, line.line_total, line.currency, quote_ref,
        *_stamp(doc),
    ]


def _invoice_rows(doc: Document) -> list:
    return [
        doc.doc_id, doc.parent_doc_id, doc.supplier_id, doc.cc_id,
        doc.doc_date, doc.currency, doc.net_total, doc.tax_amount,
        doc.gross_total, ENTITY_COUNTRY.get(doc.org_id),
        ENTITY_REGION.get(doc.org_id), *_stamp(doc),
    ]


def _invoice_line_rows(doc: Document, line: LineItem) -> list:
    return [
        f"{doc.doc_id}-{line.line_number}", doc.doc_id, line.line_number,
        line.item_id, line.description, line.quantity, line.unit_of_measure,
        line.unit_price, line.line_total, doc.parent_doc_id,
        ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id),
        *_stamp(doc),
    ]


def rows_for(chains: Sequence[Chain]) -> dict[str, list[list]]:
    """Every table's rows, in COLUMNS order. Pure: no database contact."""
    out: dict[str, list[list]] = {table: [] for table in TABLES}

    for chain in chains:
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        po = chain.purchase_order
        po_id = po.doc_id if po else None

        for quote in chain.quotes:
            linked_po = po_id if quote.doc_id == awarded.doc_id else None
            out["bp_quote_trgt"].append(_quote_rows(quote, linked_po))
            for line in quote.lines:
                out["bp_quote_line_items_trgt"].append(
                    _quote_line_rows(quote, line))

        if po is not None:
            out["bp_purchase_order_trgt"].append(_po_rows(po, awarded.doc_id))
            for line in po.lines:
                out["bp_po_line_items_trgt"].append(
                    _po_line_rows(po, line, awarded.doc_id))

        for invoice in chain.invoices:
            out["bp_invoice_trgt"].append(_invoice_rows(invoice))
            for line in invoice.lines:
                out["bp_invoice_line_items_trgt"].append(
                    _invoice_line_rows(invoice, line))

    return out


# The document tables carry no NOT NULL constraint at all, so a mapping that
# wrote NULL into every invoice amount would be accepted in silence. These are
# the columns that must carry a value for a row to mean anything.
REQUIRED: dict[str, tuple[str, ...]] = {
    "bp_quote_trgt": (
        "quote_id", "supplier_id", "quote_date", "currency", "total_amount",
    ),
    "bp_quote_line_items_trgt": (
        "quote_line_id", "quote_id", "line_number", "item_id", "line_total",
        "currency",
    ),
    "bp_purchase_order_trgt": (
        "po_id", "supplier_id", "order_date", "currency", "total_amount",
    ),
    "bp_po_line_items_trgt": (
        "po_line_id", "po_id", "line_number", "item_id", "line_total", "currency",
    ),
    "bp_invoice_trgt": (
        "invoice_id", "supplier_id", "invoice_date", "currency", "invoice_amount",
    ),
    "bp_invoice_line_items_trgt": (
        "invoice_line_id", "invoice_id", "line_no", "item_id", "line_amount",
    ),
}

# Headers before their lines: the line tables reference the document above them.
LOAD_ORDER: tuple[str, ...] = (
    "bp_quote_trgt", "bp_quote_line_items_trgt",
    "bp_purchase_order_trgt", "bp_po_line_items_trgt",
    "bp_invoice_trgt", "bp_invoice_line_items_trgt",
)


# --- requirements -----------------------------------------------------------
# A Chain carries only a requirement_id. bp_requirement wants a title, category,
# budget and dates, so they are derived from the chain's own facts -- its awarded
# quote's category leaf, its cost centre's currency, its earliest quote date --
# rather than invented independently, which would leave the requirement
# describing something its own documents do not.

REQUIREMENT_COLUMNS: tuple[str, ...] = (
    "requirement_id", "session_id", "status", "created_by", "title", "category",
    "description", "quantity", "unit", "target_budget", "currency",
    "needed_by_date", "delivery_location", "priority", "specifications",
    "constraints", "completeness_score", "missing_fields", "seed_context",
    "created_at", "updated_at",
)

REQUIREMENT_REQUIRED: tuple[str, ...] = (
    "requirement_id", "status", "title", "category", "currency", "target_budget",
)

_PRIORITIES = ("Low", "Medium", "High")

# bp_requirement.status carries a CHECK constraint. These are the only values
# the schema accepts, and the mapping must use its vocabulary rather than one
# invented to describe the chain.
REQUIREMENT_STATUSES: tuple[str, ...] = (
    "draft", "gathering", "complete", "handed_off", "abandoned",
)


def requirement_rows(chains: Sequence[Chain]) -> list[list]:
    """One row per chain, consistent with the documents beneath it."""
    rows: list[list] = []
    for chain in chains:
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        first_quote = min(chain.quotes, key=lambda q: q.doc_date)
        leaf_path = awarded.lines[0].leaf_path if awarded.lines else "Uncategorised"
        category = leaf_path.split(" > ")[0]
        quantity = sum(
            (line.quantity for line in awarded.lines if line.quantity is not None),
            Decimal("0"),
        )
        # Raised before the first quote arrived, and wanted after the award.
        raised = first_quote.doc_date - timedelta(days=7)
        when = datetime.combine(raised, datetime.min.time())

        rows.append([
            chain.requirement_id, None,
            # Handed off once a purchase order was raised against it; otherwise
            # the requirement finished gathering but never converted.
            "handed_off" if chain.purchase_order is not None else "complete",
            MARKER,
            f"{category} requirement {chain.requirement_id}",
            leaf_path,
            f"Sourcing for {leaf_path} across {len(awarded.lines)} line items.",
            quantity or None,
            awarded.lines[0].unit_of_measure if awarded.lines else None,
            awarded.net_total, awarded.currency,
            awarded.doc_date + timedelta(days=30),
            None,
            _PRIORITIES[len(chain.quotes) % len(_PRIORITIES)],
            None, None, None, None, None,
            when, when,
        ])
    return rows
