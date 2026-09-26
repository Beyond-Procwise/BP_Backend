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
    "created_date",
    "created_by",
    "last_modified_by",
    "last_modified_date",
    "confidence_score",
    "accuracy_score",
    # Routing and classification decided by the pipeline, not printed on the page.
    "doc_type",
    "region",
})


def is_derived(field: str) -> bool:
    return field in DERIVED_FIELDS
