"""The pack for one customer quote (QDA B2).

Built from the customer record and nothing else. ``sell_side.quotes.get_quote`` reads a
named list of columns with no cost or margin among them, and that record is the only thing
this module is handed -- so what we pay cannot reach a pack, and through it a page, by any
route that starts here. A record that does carry an internal field is refused outright
rather than filtered: it means the wrong reader was used.

What is NOT in a quote pack yet, and why:

  * **A quote number, until the quote is issued.** It is minted at issue
    (``sell_side.numbering``). ``quote_ref`` is an internal draft reference and is carried
    as that, never as the number.
  * **Legal terms.** No terms library exists (B9), and legal wording is never written
    here. The terms a Quote builder quote carries (payment, start, length, renewal,
    service level) are the seller's own words and are printed as they were typed.
  * **Approval conditions.** They are written by the approver for the seller, not for the
    customer.

What a Quote builder quote adds, each read from the quote or the company's own profile:
the seller's letterhead and logo, the scope, assumptions and text sections the seller
wrote (as plain paragraphs, under the layout's headings and in its order), each line's
discount, and tax at the rate the quote was priced at with the total including it.

A caller that has already read the customer record passes it as ``source`` and it is not
read again.

Scope: ``sales_quote_id``. Pass the quote's own date as ``as_of`` so the same quote always
builds the same pack.
"""

from __future__ import annotations

import base64
import hashlib
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from src.services.analytics.formatting import format_money_exact
from src.services.analytics.models import Confidence
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import FindingCode, FormatHint, Severity
from src.services.rga.quote_models import (
    QuoteIdentity,
    QuoteLine,
    QuoteNote,
    QuotePack,
    QuoteParagraph,
    QuoteSection,
    QuoteSeller,
    QuoteTerm,
)
from src.services.sell_side.money import q2
from src.services.sell_side.quote_render import INTERNAL_FIELDS

REPORT_TYPE_ID = "sales_quote"


def _read_quote(sales_quote_id: int) -> Dict[str, Any]:
    """The customer record. Never ``get_quote_internal``."""
    from src.services.sell_side import quotes
    from src.services.sell_side._db import transactional_conn

    with transactional_conn() as conn:
        return quotes.get_quote(conn, sales_quote_id)


class _Paragraphs(HTMLParser):
    """The seller's rich text (already cut to p/b/i/u/ul/ol/li/br by builder.clean_html)
    as plain paragraphs: one per <p> or <li>, a <br> a line break inside it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[QuoteParagraph] = []
        self._text: List[str] = []
        self._bullet = False

    def _flush(self) -> None:
        text = "\n".join(" ".join(part.split()) for part in "".join(self._text).split("\n"))
        text = "\n".join(part for part in text.split("\n") if part)
        if text:
            self.out.append(QuoteParagraph(text=text, bullet=self._bullet))
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "li", "ul", "ol"):
            self._flush()
            self._bullet = tag == "li"
        elif tag == "br":
            self._text.append("\n")

    def handle_endtag(self, tag):
        if tag in ("p", "li", "ul", "ol"):
            self._flush()
            self._bullet = False

    def handle_data(self, data):
        self._text.append(data)


def paragraphs(html: Optional[str]) -> List[QuoteParagraph]:
    parser = _Paragraphs()
    parser.feed(html or "")
    parser.close()
    parser._flush()
    return parser.out


# The words a builder quote's terms are printed under. Each value is the seller's own.
_TERM_LABELS = (("start", "Start"), ("lengthMonths", "Term"), ("renewal", "Renewal"),
                ("sla", "Service level"))


def _terms(quote_id: int, terms: Dict[str, Any]) -> List[QuoteTerm]:
    out: List[QuoteTerm] = []

    def add(kind: str, key: str, text: str) -> None:
        out.append(QuoteTerm(term_id=f"T{len(out) + 1:04d}", kind=kind, text=text,
                             provenance_id=f"proc.bp_sales_quote:{quote_id}#terms.{key}"))

    for key, label in _TERM_LABELS:
        value = terms.get(key)
        if value is None or value == "":
            continue
        if key == "lengthMonths":
            value = f"{value} month" + ("" if int(value) == 1 else "s")
        add("general", key, f"{label}: {value}")
    if terms.get("payment"):
        add("payment", "payment", str(terms["payment"]))
    return out


def _read_seller(sections: List[Dict[str, Any]]) -> Optional[QuoteSeller]:
    """The company's profile, and the logo the letterhead names (else the profile's)."""
    from src.services import brand_logos, entity_profile
    from src.services.sell_side._db import transactional_conn

    with transactional_conn() as conn:
        profile = entity_profile.load(conn)
        if not profile.get("legalName"):
            return None      # nobody has said who the company is; not invented here
        logo_id = next((s.get("logoId") for s in sections
                        if s.get("kind") == "letterhead" and s.get("logoId")), None) \
            or profile.get("logoId")
        logo = brand_logos.image(conn, int(logo_id)) if logo_id else None
    return seller_from(profile, logo)


def seller_from(profile: Dict[str, Any], logo: Optional[tuple]) -> QuoteSeller:
    """``logo`` is (bytes, media type) or None."""
    bank = profile.get("bank") or {}
    data_uri = sha = None
    if logo:
        data, media = logo
        data_uri = f"data:{media};base64,{base64.b64encode(data).decode('ascii')}"
        sha = hashlib.sha256(data).hexdigest()
    return QuoteSeller(
        legal_name=profile["legalName"], trading_name=profile.get("tradingName"),
        address=profile.get("address"), email=profile.get("email"),
        phone=profile.get("phone"), web=profile.get("web"),
        company_number=profile.get("companyNumber"), vat_number=profile.get("vatNumber"),
        bank_name=bank.get("name"), sort_code=bank.get("sortCode"),
        account_number=bank.get("accountNumber"), iban=bank.get("iban"),
        footer_note=profile.get("footerNote"), logo_data_uri=data_uri, logo_sha256=sha,
        provenance_id="proc.bp_admin_config:entity_profile")


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
            unit_price=row["unit_price"],
            # a catalog line's discount is how its price was reached from list, not an
            # offer to the customer; only the builder's entered discount is printed
            discount_pct=row.get("entered_discount_pct") or None,
            line_total=row["line_total"],
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

    tax_ids: List[str] = []
    rate = quote.get("tax_rate_pct")
    if rate is not None:
        rate = Decimal(rate)
        shown = f"{rate.normalize():f}%"
        tax = fb.add(
            label=f"Tax at {shown}", value=q2(total.value * rate / Decimal(100)),
            derivation="sales_quote.total_ex_tax * sales_quote.tax_rate_pct / 100",
            confidence=Confidence.ASSERTED, format_hint=FormatHint.MONEY_EXACT,
            currency=currency, provenance_id=f"proc.bp_sales_quote:{quote_id}#tax_rate_pct")
        whole = fb.add(
            label="Total including tax", value=total.value + tax.value,
            derivation="sales_quote.total_ex_tax + tax", confidence=Confidence.ASSERTED,
            format_hint=FormatHint.MONEY_EXACT, currency=currency,
            provenance_id=f"proc.bp_sales_quote:{quote_id}#tax_rate_pct")
        tax_ids = [tax.fact_id, whole.fact_id]

    # The layout: the quote's own when it has one (a builder quote always does), else the
    # catalog quote's fixed order. Prose sections become passages of the pack, in place.
    from src.services.sell_side import templates
    builder_quote = quote.get("sections") is not None or bool(
        quote.get("scope_html") or quote.get("assumptions_html") or quote.get("terms"))
    layout_in = (quote.get("sections") or templates.default_sections()) if builder_quote else []
    written = {"scope": quote.get("scope_html"), "assumptions": quote.get("assumptions_html")}
    sections: List[QuoteSection] = []
    layout: List[str] = []
    for sec in layout_in:
        kind = sec["kind"]
        if kind in ("scope", "assumptions", "text"):
            said = paragraphs(sec.get("html") if kind == "text" else written[kind])
            if said:
                sid = f"S{len(sections) + 1:04d}"
                sections.append(QuoteSection(
                    section_id=sid, heading=sec["heading"], paragraphs=said,
                    provenance_id=(f"proc.bp_sales_quote:{quote_id}#sections.{sec['key']}"
                                   if kind == "text" else
                                   f"proc.bp_sales_quote:{quote_id}#{kind}_html")))
                layout.append(sid)
        elif kind in ("letterhead", "lines", "totals", "terms", "signature"):
            layout.append(kind)
        # "customer" is the page's own header; "conditions" are the approver's, not the
        # customer's
    seller = None
    if "letterhead" in layout or "signature" in layout:
        seller = quote["seller"] if "seller" in quote else _read_seller(layout_in)
    if seller is None and "letterhead" in layout:
        layout.remove("letterhead")

    fb.extras.update(
        identity=QuoteIdentity(
            quote_ref=quote["quote_ref"], number=quote.get("quote_number"),
            status=quote["status"],
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
        terms=_terms(quote_id, quote.get("terms") or {}),
        sections=sections, seller=seller, tax_fact_ids=tax_ids, layout=layout,
        total_fact_id=total.fact_id,
    )
