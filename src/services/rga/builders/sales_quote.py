"""The pack for one customer quote (QDA B2).

Built from the customer record and nothing else. ``sell_side.quotes.get_quote`` reads a
named list of columns with no cost or margin among them, and that record is the only thing
this module is handed -- so what we pay cannot reach a pack, and through it a page, by any
route that starts here. A record that does carry an internal field is refused outright
rather than filtered: it means the wrong reader was used.

What is NOT in a quote pack yet, and why:

  * **A quote number.** Nothing mints one (the allocator is B6). ``quote_ref`` is an
    internal draft reference and is carried as that, never as the number.
  * **Tax.** No tax rate is configured anywhere; the only total is the one the quote row
    holds, excluding tax.
  * **Terms.** No terms library exists (B9), and legal wording is never written here.

A caller that has already read the customer record passes it as ``source`` and it is not
read again.

Scope: ``sales_quote_id``. Pass the quote's own date as ``as_of`` so the same quote always
builds the same pack.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

from src.services.analytics.formatting import format_money_exact
from src.services.analytics.models import Confidence
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import FindingCode, FormatHint, Severity
from src.services.rga.quote_models import QuoteIdentity, QuoteLine, QuoteNote, QuotePack
from src.services.sell_side.quote_render import INTERNAL_FIELDS

REPORT_TYPE_ID = "sales_quote"


def _read_quote(sales_quote_id: int) -> Dict[str, Any]:
    """The customer record. Never ``get_quote_internal``."""
    from src.services.sell_side import quotes
    from src.services.sell_side._db import transactional_conn

    with transactional_conn() as conn:
        return quotes.get_quote(conn, sales_quote_id)


def _refuse_internal(row: Dict[str, Any], where: str) -> None:
    found = sorted(INTERNAL_FIELDS & set(row))
    if found:
        raise ValueError(f"{where} carries {', '.join(found)}; a quote pack is built from "
                         "the customer record, which has no cost or margin in it")


@register(REPORT_TYPE_ID, pack_class=QuotePack)
def build(fb: FactBuilder) -> None:
    quote = fb.source if fb.source is not None else _read_quote(
        int(fb.scope["sales_quote_id"]))
    quote_id = quote["sales_quote_id"]
    if quote_id != int(fb.scope["sales_quote_id"]):
        raise ValueError(f"handed quote {quote_id} for scope {fb.scope['sales_quote_id']}")
    currency = quote["currency"]
    _refuse_internal(quote, f"quote {quote_id}")

    lines = []
    for row in quote["lines"]:
        _refuse_internal(row, f"quote {quote_id} line {row['line_no']}")
        lines.append(QuoteLine(
            line_id=f"L{row['sales_quote_line_id']}", line_no=row["line_no"],
            sku=row.get("distributor_sku"), mpn=row.get("mpn"),
            description=row["item_description"],
            unit_of_measure=row.get("unit_of_measure"), quantity=row["quantity"],
            unit_price=row["unit_price"], line_total=row["line_total"],
            currency=row["currency"],
            provenance_id=f"proc.bp_sales_quote_line:{row['sales_quote_line_id']}"))

    total = fb.add(
        label="Total excluding tax", value=Decimal(quote["total_ex_tax"]),
        derivation="sales_quote.total_ex_tax", confidence=Confidence.ASSERTED,
        format_hint=FormatHint.MONEY_EXACT, currency=currency,
        provenance_id=f"proc.bp_sales_quote:{quote_id}#total_ex_tax")

    added = sum((l.line_total for l in lines), Decimal(0))
    if added != total.value:
        fb.finding(
            code=FindingCode.QUOTE_TOTAL_MISMATCH, severity=Severity.HIGH,
            detail=(f"the lines add up to {format_money_exact(added, currency)} but the "
                    f"quote's total is {total.display}"),
            fact_id=total.fact_id)

    fb.extras.update(
        identity=QuoteIdentity(
            quote_ref=quote["quote_ref"], status=quote["status"],
            account_name=quote["account_name"], contact_name=quote.get("contact_name"),
            quote_date=quote["quote_date"], valid_until=quote["valid_until"],
            currency=currency, provenance_id=f"proc.bp_sales_quote:{quote_id}"),
        lines=lines,
        # The reader already keeps only these; an unsafe one that reached here is dropped
        # again rather than trusted.
        notes=[QuoteNote(note_id=f"N{j['justification_id']}", kind=j["kind"],
                         text=j["claim"],
                         provenance_id=f"proc.bp_sales_justification:{j['justification_id']}")
               for j in quote.get("justifications") or [] if j.get("customer_safe") is True],
        total_fact_id=total.fact_id,
    )
