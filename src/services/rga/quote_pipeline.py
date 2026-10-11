"""From a quote on record to the page or PDF a customer may be sent.

One path: the customer record becomes a pack, the pack a document, the document a page,
and the page goes through the gate (``postcheck.run_quote``). A page that does not pass is
not returned. The PDF is printed from the page that passed.

No model is called anywhere here. The document is the standard one below -- every line, the
total, and whatever notes and terms the pack holds -- until something else is asked for.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.services.rga import builders  # noqa: F401  (registers the quote pack builder)
from src.services.rga import postcheck
from src.services.rga.builders.sales_quote import REPORT_TYPE_ID
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import Finding, FindingCode
from src.services.rga.quote_models import (
    LineTableBlock,
    NotesBlock,
    PaymentBlock,
    QuoteAST,
    QuotePack,
    TermsBlock,
    TotalsBlock,
)
from src.services.rga.render import RenderedArtefact, quote_html
from src.services.rga.style import resolve_style_brief
from src.services.sell_side._db import StateConflict
from src.services.sell_side.quote_render import InternalFieldLeak, NotCustomerReady

FORMATS = ("html", "pdf")


class QuoteBlocked(StateConflict):
    """The quote did not pass its gate. Carries every finding that blocked it."""

    def __init__(self, findings: List[Finding]) -> None:
        self.findings = findings
        super().__init__("the quote cannot be sent: " + "; ".join(
            f"{f.code.value}: {f.detail}" for f in findings))


def default_document(pack: QuotePack) -> QuoteAST:
    """Every line in order, the total, then the notes and terms the pack holds."""
    blocks: list = [LineTableBlock(line_refs=[l.line_id for l in pack.lines]),
                    TotalsBlock(fact_refs=[pack.total_fact_id])]
    if pack.notes:
        blocks.append(NotesBlock(note_refs=[n.note_id for n in pack.notes]))
    general = [t.term_id for t in pack.terms if t.kind == "general"]
    if general:
        blocks.append(TermsBlock(term_refs=general))
    blocks.extend(PaymentBlock(term_ref=t.term_id) for t in pack.terms if t.kind == "payment")
    return QuoteAST(blocks=blocks)


def produce(quote: Dict[str, Any], internal: Dict[str, Any], *, fmt: str = "html",
            ast: Optional[QuoteAST] = None, emit_audit: bool = False) -> RenderedArtefact:
    """The customer's page (or PDF) for one quote, checked.

    ``quote`` is the customer record (``quotes.get_quote``) and the only thing the page is
    drawn from. ``internal`` (``quotes.get_quote_internal``) is handed to the gate so it
    knows which figures must not appear, and goes nowhere else.

    Raises ``NotCustomerReady`` for a quote that is not approved, ``InternalFieldLeak`` if
    anything we pay is on the page, and ``QuoteBlocked`` for every other failed check.
    """
    if fmt not in FORMATS:
        raise ValueError(f"a quote is produced as one of {', '.join(FORMATS)}, not {fmt!r}")
    if quote.get("status") not in postcheck.QUOTE_READY:
        raise NotCustomerReady(f"quote is {quote.get('status')}; only approved or issued "
                               "quotes have a customer view")

    pack = build_fact_pack(
        REPORT_TYPE_ID, scope={"sales_quote_id": quote["sales_quote_id"]},
        as_of=quote["quote_date"].isoformat(), source=quote, emit_audit=emit_audit)
    document = ast if ast is not None else default_document(pack)
    page = quote_html.render(document, pack, resolve_style_brief(REPORT_TYPE_ID))

    result = postcheck.run_quote(page, pack, document, internal, emit_audit=emit_audit)
    for finding in result.blocking:
        if finding.code is FindingCode.INTERNAL_FIELD_LEAK:
            raise InternalFieldLeak(finding.detail)
    if not result.passed:
        raise QuoteBlocked(result.blocking)
    return page if fmt == "html" else quote_html.to_pdf(page)
