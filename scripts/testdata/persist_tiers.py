"""Map documents onto the raw and staging tiers.

persist.py writes the promoted `_trgt` record. This writes the two tiers behind
it, so the promotion path has data and the outcome triggers on the `_trgt`
tables resolve instead of finding nothing.

There is no extraction step here, so `raw` is not a degraded version of the
document -- it is the same document plus the provenance a real ingestion would
have recorded: which file it came from, when, under which pipeline version, and
the candidate primary key the trigger matches on.

`doc_pk_candidate` and `source_file` are the two columns that matter beyond
storage: `trg_fn_invoice_trgt_outcome` looks the raw row up by
`doc_pk_candidate = NEW.invoice_id` and reads `source_file` from it.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Sequence

from scripts.testdata.documents import Chain, Document, LineItem
from scripts.testdata.loader import MissingRequiredValue, _check_required
from scripts.testdata.persist import ENTITY_COUNTRY, ENTITY_REGION, MARKER

__all__ = [
    "COLUMNS", "REQUIRED", "LOAD_ORDER", "MissingRequiredValue",
    "check_required", "rows_for_tiers", "source_file_for",
]

PIPELINE_VERSION = "testdata-1"
PROMOTION_STATUS = "promoted"

COLUMNS: dict[str, tuple[str, ...]] = {
    "bp_quote_raw": (
        "raw_id", "doc_pk_candidate", "source_file", "raw_payload", "extracted_at",
        "pipeline_version", "promotion_status", "promoted_at",
        "quote_id", "supplier_id", "buyer_id", "quote_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax", "po_id",
        "country", "region",
    ),
    "bp_quote_line_items_raw": (
        "line_raw_id", "raw_id", "line_number", "item_id", "item_description",
        "quantity", "unit_of_measure", "unit_price", "line_total", "currency",
    ),
    "bp_quote_stg": (
        "quote_id", "supplier_id", "buyer_id", "quote_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax", "po_id",
        "country", "region", "created_date", "created_by", "last_modified_by",
        "last_modified_date",
    ),
    "bp_quote_line_items_stg": (
        "quote_line_id", "quote_id", "line_number", "item_id", "item_description",
        "quantity", "unit_of_measure", "unit_price", "line_total", "currency",
        "created_date", "created_by", "last_modified_by", "last_modified_date",
    ),
    "bp_purchase_order_raw": (
        "raw_id", "doc_pk_candidate", "source_file", "raw_payload", "extracted_at",
        "pipeline_version", "promotion_status", "promoted_at",
        "po_id", "supplier_id", "buyer_id", "order_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax",
        "ship_to_country", "delivery_region", "quote_reference",
    ),
    "bp_po_line_items_raw": (
        "line_raw_id", "raw_id", "line_number", "item_id", "item_description",
        "quote_number", "quantity", "unit_price", "unit_of_measure", "currency",
        "line_total",
    ),
    "bp_purchase_order_stg": (
        "po_id", "supplier_id", "buyer_id", "order_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax",
        "ship_to_country", "delivery_region", "quote_reference",
        "created_date", "created_by", "last_modified_by", "last_modified_date",
    ),
    "bp_po_line_items_stg": (
        "po_id", "po_line_id", "line_number", "item_id", "item_description",
        "quote_number", "quantity", "unit_price", "unit_of_measure", "currency",
        "line_total", "created_date", "created_by", "last_modified_by",
        "last_modified_date",
    ),
    "bp_invoice_raw": (
        "raw_id", "doc_pk_candidate", "source_file", "raw_payload", "extracted_at",
        "pipeline_version", "promotion_status", "promoted_at",
        "invoice_id", "po_id", "supplier_id", "buyer_id", "invoice_date",
        "currency", "invoice_amount", "tax_amount", "invoice_total_incl_tax",
        "country", "region",
    ),
    "bp_invoice_line_items_raw": (
        "line_raw_id", "raw_id", "line_no", "item_id", "item_description",
        "quantity", "unit_of_measure", "unit_price", "line_amount", "po_id",
        "country", "region",
    ),
    "bp_invoice_stg": (
        "invoice_id", "po_id", "supplier_id", "buyer_id", "invoice_date",
        "currency", "invoice_amount", "tax_amount", "invoice_total_incl_tax",
        "country", "region", "created_date", "created_by", "last_modified_by",
        "last_modified_date",
    ),
    "bp_invoice_line_items_stg": (
        "invoice_line_id", "invoice_id", "line_no", "item_id", "item_description",
        "quantity", "unit_of_measure", "unit_price", "line_amount", "po_id",
        "country", "region", "created_date", "created_by", "last_modified_by",
        "last_modified_date",
    ),
}

_PROVENANCE_REQUIRED = (
    "raw_id", "doc_pk_candidate", "source_file", "raw_payload",
    "extracted_at", "pipeline_version", "promotion_status",
)

REQUIRED: dict[str, tuple[str, ...]] = {
    "bp_quote_raw": (*_PROVENANCE_REQUIRED, "quote_id", "supplier_id", "total_amount"),
    "bp_quote_line_items_raw": ("line_raw_id", "raw_id", "line_number", "item_id", "line_total"),
    "bp_quote_stg": ("quote_id", "supplier_id", "quote_date", "currency", "total_amount"),
    "bp_quote_line_items_stg": ("quote_line_id", "quote_id", "line_number", "item_id", "line_total"),
    "bp_purchase_order_raw": (*_PROVENANCE_REQUIRED, "po_id", "supplier_id", "total_amount"),
    "bp_po_line_items_raw": ("line_raw_id", "raw_id", "line_number", "item_id", "line_total"),
    "bp_purchase_order_stg": ("po_id", "supplier_id", "order_date", "currency", "total_amount"),
    "bp_po_line_items_stg": ("po_id", "po_line_id", "line_number", "item_id", "line_total"),
    "bp_invoice_raw": (*_PROVENANCE_REQUIRED, "invoice_id", "supplier_id", "invoice_amount"),
    "bp_invoice_line_items_raw": ("line_raw_id", "raw_id", "line_no", "item_id", "line_amount"),
    "bp_invoice_stg": ("invoice_id", "supplier_id", "invoice_date", "currency", "invoice_amount"),
    "bp_invoice_line_items_stg": ("invoice_line_id", "invoice_id", "line_no", "item_id", "line_amount"),
}

# raw before its lines (the lines reference raw_id), then staging.
LOAD_ORDER: tuple[str, ...] = (
    "bp_quote_raw", "bp_quote_line_items_raw",
    "bp_purchase_order_raw", "bp_po_line_items_raw",
    "bp_invoice_raw", "bp_invoice_line_items_raw",
    "bp_quote_stg", "bp_quote_line_items_stg",
    "bp_purchase_order_stg", "bp_po_line_items_stg",
    "bp_invoice_stg", "bp_invoice_line_items_stg",
)

_EXTENSION = {"Quote": "pdf", "Purchase_Order": "pdf", "Invoice": "pdf"}


def check_required(table: str, rows: Sequence[Sequence]) -> None:
    _check_required(table, COLUMNS[table], REQUIRED[table], rows)


def source_file_for(doc: Document) -> str:
    """The file the document would have arrived as. The outcome triggers read it."""
    folder = doc.doc_type.lower()
    return f"s3://bp-testdata/{folder}/{doc.doc_id}.{_EXTENSION.get(doc.doc_type, 'pdf')}"


class _Ids:
    """raw_id and line_raw_id are bigint with sequences behind them, so they
    must be numbers. They are allocated in generation order, which is
    deterministic, and written explicitly rather than left to the sequence so a
    line row can name its parent without a round trip."""

    def __init__(self) -> None:
        self._raw: dict[str, int] = {}
        self._line = 0

    def raw(self, doc: Document) -> int:
        if doc.doc_id not in self._raw:
            self._raw[doc.doc_id] = len(self._raw) + 1
        return self._raw[doc.doc_id]

    def line(self) -> int:
        self._line += 1
        return self._line


def _when(doc: Document) -> datetime:
    return datetime.combine(doc.doc_date, datetime.min.time())


def _payload(doc: Document) -> str:
    """What an extractor would have emitted for this document."""
    return json.dumps(
        {
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "doc_date": doc.doc_date.isoformat(),
            "supplier_id": doc.supplier_id,
            "currency": doc.currency,
            "net_total": str(doc.net_total),
            "tax_amount": str(doc.tax_amount),
            "gross_total": str(doc.gross_total),
            "line_count": len(doc.lines),
        },
        sort_keys=True,
    )


def _provenance(doc: Document, raw_id: int) -> list:
    when = _when(doc)
    return [
        raw_id, doc.doc_id, source_file_for(doc), _payload(doc), when,
        PIPELINE_VERSION, PROMOTION_STATUS, when,
    ]


def _geo(doc: Document) -> list:
    return [ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id)]


def _stamp(doc: Document) -> list:
    when = _when(doc)
    return [when, MARKER, MARKER, when]


def _line_id(doc: Document, line: LineItem) -> str:
    return f"{doc.doc_id}-{line.line_number}"


def rows_for_tiers(chains: Sequence[Chain]) -> dict[str, list[list]]:
    """Every raw and staging row, in COLUMNS order. Pure: no database contact."""
    out: dict[str, list[list]] = {table: [] for table in COLUMNS}
    ids = _Ids()

    for chain in chains:
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        po = chain.purchase_order
        po_id = po.doc_id if po else None

        for quote in chain.quotes:
            linked = po_id if quote.doc_id == awarded.doc_id else None
            quote_raw_id = ids.raw(quote)
            out["bp_quote_raw"].append([
                *_provenance(quote, quote_raw_id), quote.doc_id, quote.supplier_id, quote.cc_id,
                quote.doc_date, quote.currency, quote.net_total, quote.tax_amount,
                quote.gross_total, linked, *_geo(quote),
            ])
            out["bp_quote_stg"].append([
                quote.doc_id, quote.supplier_id, quote.cc_id, quote.doc_date,
                quote.currency, quote.net_total, quote.tax_amount,
                quote.gross_total, linked, *_geo(quote), *_stamp(quote),
            ])
            for line in quote.lines:
                out["bp_quote_line_items_raw"].append([
                    ids.line(), quote_raw_id, line.line_number,
                    line.item_id, line.description, line.quantity,
                    line.unit_of_measure, line.unit_price, line.line_total,
                    line.currency,
                ])
                out["bp_quote_line_items_stg"].append([
                    _line_id(quote, line), quote.doc_id, line.line_number,
                    line.item_id, line.description, line.quantity,
                    line.unit_of_measure, line.unit_price, line.line_total,
                    line.currency, *_stamp(quote),
                ])

        if po is not None:
            po_raw_id = ids.raw(po)
            out["bp_purchase_order_raw"].append([
                *_provenance(po, po_raw_id), po.doc_id, po.supplier_id, po.cc_id,
                po.doc_date, po.currency, po.net_total, po.tax_amount,
                po.gross_total, *_geo(po), awarded.doc_id,
            ])
            out["bp_purchase_order_stg"].append([
                po.doc_id, po.supplier_id, po.cc_id, po.doc_date, po.currency,
                po.net_total, po.tax_amount, po.gross_total, *_geo(po),
                awarded.doc_id, *_stamp(po),
            ])
            for line in po.lines:
                out["bp_po_line_items_raw"].append([
                    ids.line(), po_raw_id, line.line_number,
                    line.item_id, line.description, awarded.doc_id, line.quantity,
                    line.unit_price, line.unit_of_measure, line.currency,
                    line.line_total,
                ])
                out["bp_po_line_items_stg"].append([
                    po.doc_id, _line_id(po, line), line.line_number, line.item_id,
                    line.description, awarded.doc_id, line.quantity,
                    line.unit_price, line.unit_of_measure, line.currency,
                    line.line_total, *_stamp(po),
                ])

        for invoice in chain.invoices:
            invoice_raw_id = ids.raw(invoice)
            out["bp_invoice_raw"].append([
                *_provenance(invoice, invoice_raw_id), invoice.doc_id, invoice.parent_doc_id,
                invoice.supplier_id, invoice.cc_id, invoice.doc_date,
                invoice.currency, invoice.net_total, invoice.tax_amount,
                invoice.gross_total, *_geo(invoice),
            ])
            out["bp_invoice_stg"].append([
                invoice.doc_id, invoice.parent_doc_id, invoice.supplier_id,
                invoice.cc_id, invoice.doc_date, invoice.currency,
                invoice.net_total, invoice.tax_amount, invoice.gross_total,
                *_geo(invoice), *_stamp(invoice),
            ])
            for line in invoice.lines:
                out["bp_invoice_line_items_raw"].append([
                    ids.line(), invoice_raw_id, line.line_number,
                    line.item_id, line.description, line.quantity,
                    line.unit_of_measure, line.unit_price, line.line_total,
                    invoice.parent_doc_id, *_geo(invoice),
                ])
                out["bp_invoice_line_items_stg"].append([
                    _line_id(invoice, line), invoice.doc_id, line.line_number,
                    line.item_id, line.description, line.quantity,
                    line.unit_of_measure, line.unit_price, line.line_total,
                    invoice.parent_doc_id, *_geo(invoice), *_stamp(invoice),
                ])

    return out
