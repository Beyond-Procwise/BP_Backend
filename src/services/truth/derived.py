"""Fields the document does not contain as text, so grounding cannot judge them.

A field wrongly ON this list is excused from measurement forever. A field
wrongly OFF it is reported as a failure the model had no way to avoid. So the
list is short, explicit, and defaults to NOT derived: a field nobody has
classified is measured, which fails loudly rather than quietly.

``tax_percent`` is deliberately absent. Where a document prints "VAT (20%)" the
figure is on the page and should be verified. Where it prints only an amount and
a subtotal, verification reports it unsupported -- which is the honest answer,
and the signal that the pipeline computed the rate rather than read it.
OPEN FOR REVIEW: the business may rule otherwise.
"""
from __future__ import annotations

DERIVED_FIELDS: frozenset[str] = frozenset({
    # Currency conversion happens in the pipeline against an FX table.
    "converted_amount_usd",
    "exchange_rate_to_usd",
    # Identity and bookkeeping the pipeline assigns.
    "deal_id",
    "deal_name",
    "document_id",
    # Surrogate keys the pipeline MINTS. No document contains
    # 'SUP-THRIVESTUDIOSLLC' or 'QTE-2026-00487-1' -- they are constructed from
    # a resolved supplier or a parent id plus a line number. Scoring them as
    # extraction errors blamed the model for 1,235 values it could not have read
    # off a page: 32% of every 'wrong' verdict in the first baseline.
    #
    # Note what this costs: supplier CORRECTNESS is no longer measured here at
    # all. Whether the right supplier was identified is a resolution question,
    # answered against proc.bp_supplier, not by grounding against page text.
    "supplier_id",
    "buyer_id",
    "quote_line_id",
    "invoice_line_id",
    "po_line_id",
    "created_date",
    "created_by",
    "last_modified_by",
    "last_modified_date",
    "confidence_score",
    "accuracy_score",
    # Routing and classification decided by the pipeline, not printed on the page.
    "doc_type",
})

# `region` was here and has been removed. It holds "West Sussex" in this corpus
# -- an address component printed on the page, not pipeline routing -- so
# excusing it hid 280 field instances from measurement forever. Exactly the
# hazard the docstring above warns about, committed in the first draft.


def is_derived(field: str) -> bool:
    return field in DERIVED_FIELDS
