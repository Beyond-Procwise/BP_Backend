"""The customer's quote page, its PDF, and the gate they pass (QDA B3, B5).

No database, no model. Built from the two SYNTHETIC quotes under ``tests/sell_side/fixtures``.

Most tests here damage a finished page -- drop a row, change a price, slip one of our own
figures in -- and require the gate to object by name. The page is damaged rather than the
tree because the gate reads the page: that is what the customer would be sent.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from src.services.rga import postcheck, quote_pipeline
from src.services.rga.builders import sales_quote
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import FindingCode as Code
from src.services.rga.quote_models import (
    LineTableBlock,
    PaymentBlock,
    QuoteAST,
    QuoteTerm,
    SignatureBlock,
    TermsBlock,
    TotalsBlock,
)
from src.services.rga.render import quote_html
from src.services.rga.style import resolve_style_brief
from src.services.sell_side._db import StateConflict
from src.services.sell_side.quote_render import InternalFieldLeak, NotCustomerReady
from tests.sell_side.fixtures.synthetic_quotes import goods_quote, unknown_cost_quote

BRIEF = resolve_style_brief(sales_quote.REPORT_TYPE_ID)
OURS = ("1543.75", "1,543.75", "633.25", "27.35", "420", "125.6667", "0.2909", "29.09")


def pack_of(quote):
    return build_fact_pack(sales_quote.REPORT_TYPE_ID,
                           scope={"sales_quote_id": quote["sales_quote_id"]},
                           as_of=quote["quote_date"].isoformat(), source=quote,
                           emit_audit=False)


class Made:
    """One quote taken all the way to a page, with the pieces kept for damaging."""

    def __init__(self, records=None, *, change=None, document=None):
        self.quote, self.internal = records or goods_quote()
        self.pack = pack_of(self.quote)
        if change:
            self.pack = change(self.pack)
        self.document = document(self.pack) if document else \
            quote_pipeline.default_document(self.pack)
        self.page = quote_html.render(self.document, self.pack, BRIEF)

    @property
    def html(self) -> str:
        return self.page.content.decode("utf-8")

    def damaged(self, old: str, new: str, count: int = 1) -> "Made":
        assert self.html.count(old) >= 1, f"{old!r} is not on the page to damage"
        self.page = dataclasses.replace(
            self.page, content=self.html.replace(old, new, count).encode("utf-8"))
        return self

    def without_row(self, description: str) -> "Made":
        start = self.html.rindex("<tr data-chunk>", 0, self.html.index(description))
        end = self.html.index("</tr>", start) + len("</tr>")
        return self.damaged(self.html[start:end], "")

    def blocking(self):
        return postcheck.run_quote(self.page, self.pack, self.document, self.internal,
                                   emit_audit=False).blocking

    def codes(self):
        return [f.code for f in self.blocking()]

    def details(self):
        return [f.detail for f in self.blocking()]


def identity(**changes):
    return lambda pack: pack.model_copy(
        update={"identity": pack.identity.model_copy(update=changes)})


def with_terms(pack):
    return pack.model_copy(update={"terms": [
        QuoteTerm(term_id="T0001", kind="general", provenance_id="synthetic:term#1",
                  text="Synthetic term: goods remain ours until paid for."),
        QuoteTerm(term_id="T0002", kind="payment", provenance_id="synthetic:term#2",
                  text="Synthetic term: payment within 30 days of invoice.")]})


class TestThePage:
    def test_a_whole_quote_passes(self):
        assert Made().codes() == []
        assert Made(unknown_cost_quote()).codes() == []

    def test_it_says_who_it_is_for_and_until_when(self):
        html = Made().html
        assert "Reference SQ-SYNTHETIC-A" in html
        assert "Dated 14 September 2026 · Valid until 13 November 2026" in html
        assert "Prices in GBP, excluding tax" in html

    def test_every_word_is_escaped(self):
        html = Made().html
        assert "Synthetic Fixture Co &lt;Ltd&gt;" in html and "bracket &amp; plate" in html
        assert "<Ltd>" not in html

    def test_a_unit_price_is_printed_in_full(self):
        """173.6667 x 3 is the 521 on the line. 173.67 x 3 is not."""
        html = Made().html
        assert "£173.6667" in html and "£41.20" in html and "£58<" in html
        assert "£2,177" in html

    def test_nothing_we_pay_is_on_it(self):
        html = Made().html
        for ours in OURS:
            assert ours not in html.replace("£1,450", ""), ours

    def test_a_price_that_equals_a_cost_on_another_line_still_prints(self):
        """41.20 is line 1's cost and line 2's price. The customer's own figure prints."""
        made = Made()
        assert made.internal["lines"][0]["unit_cost"] == Decimal("41.2000")
        assert "£41.20" in made.html and made.codes() == []

    def test_it_carries_no_marker_badge_or_hash_and_passes_without_them(self):
        """A report must show each figure's marker and badge. A customer's quote shows
        none, and its gate does not ask for one."""
        made = Made()
        for internal_chrome in ("[F0001]", "F0001", "ASSERTED", "CORROBORATED", "UNASSESSED",
                                "LEGACY_UNVERIFIED", made.pack.hash[:12], made.pack.pack_id,
                                "platform_default", "bp_sales_quote"):
            assert internal_chrome not in made.html, internal_chrome
        assert Code.MISSING_ORIGIN_BADGE not in made.codes()

    def test_the_customer_safe_note_is_printed_word_for_word(self):
        made = Made()
        assert "<li data-chunk>Line 3 replaces a withdrawn part</li>" in made.html

    def test_an_unsafe_note_handed_to_the_builder_is_not_printed(self):
        quote, internal = goods_quote()
        quote["justifications"] = internal["justifications"]   # as if the reader had not filtered
        made = Made((quote, internal))
        assert "We clear" not in made.html and made.codes() == []

    def test_terms_payment_and_signatures_are_drawn_from_the_pack(self):
        def document(pack):
            return QuoteAST(blocks=[
                *quote_pipeline.default_document(pack).blocks,
                SignatureBlock(party="supplier"), SignatureBlock(party="customer")])
        made = Made(change=with_terms, document=document)
        assert "goods remain ours until paid for" in made.html
        assert "payment within 30 days of invoice" in made.html
        assert "Signed for Synthetic Fixture Co &lt;Ltd&gt;" in made.html
        assert "Signed for the supplier" in made.html
        assert made.codes() == []

    def test_the_same_quote_draws_the_same_bytes(self):
        assert Made().page.content == Made().page.content

    def test_no_script_and_nothing_fetched(self):
        html = Made().html
        for outside in ("<script", "<link", "<img", "http://", "https://", "url("):
            assert outside not in html
        assert "default-src 'none'" in html


class TestTheGate:
    # -- every line -----------------------------------------------------------------
    def test_a_line_left_out_of_the_document_is_caught(self):
        made = Made(document=lambda pack: QuoteAST(blocks=[
            LineTableBlock(line_refs=["L501", "L503"]),
            TotalsBlock(fact_refs=[pack.total_fact_id])]))
        assert made.codes() == [Code.QUOTE_LINE_OMITTED]
        assert "not in the document" in made.details()[0]

    def test_a_line_dropped_from_the_page_is_caught(self):
        made = Made().without_row("Synthetic fastener kit")
        assert made.codes() == [Code.QUOTE_LINE_OMITTED]
        assert "line 2 (Synthetic fastener kit)" in made.details()[0]
        assert "not drawn on the page" in made.details()[0]

    def test_a_document_with_no_lines_at_all_is_caught(self):
        made = Made(document=lambda pack: QuoteAST(
            blocks=[TotalsBlock(fact_refs=[pack.total_fact_id])]))
        assert made.codes() == [Code.QUOTE_LINE_OMITTED] * 3

    def test_one_row_cannot_stand_for_two_identical_lines(self):
        quote, internal = unknown_cost_quote()
        quote["lines"][1].update(quote["lines"][0], sales_quote_line_id=512, line_no=2)
        quote["total_ex_tax"] = Decimal("2900.00")
        assert Made((quote, internal)).codes() == []
        dropped = Made((quote, internal)).without_row("Synthetic bracket")
        assert dropped.codes() == [Code.QUOTE_LINE_OMITTED]

    def test_a_line_drawn_at_another_price_is_caught(self):
        codes = Made().damaged(">£58<", ">£57<").codes()
        assert Code.QUOTE_LINE_OMITTED in codes and Code.REPORT_UNTRACED_FIGURE in codes

    def test_a_line_drawn_with_another_lines_price_is_caught(self):
        """£173.6667 is a figure the quote holds, so the untraced scan passes it. The row
        check does not: it is not line 1's price."""
        assert Made().damaged(">£58<", ">£173.6667<").codes() == [Code.QUOTE_LINE_OMITTED]

    # -- the total ------------------------------------------------------------------
    def test_a_total_that_reads_differently_from_the_record_is_caught(self):
        codes = Made().damaged("£2,177", "£2,077").codes()
        assert Code.QUOTE_TOTAL_MISMATCH in codes

    def test_a_total_swapped_for_a_line_total_is_caught(self):
        """£1,450 is on the quote, so only the total check can see this."""
        assert Made().damaged('<span class="val">£2,177', '<span class="val">£1,450').codes() \
            == [Code.QUOTE_TOTAL_MISMATCH]

    def test_a_document_that_states_no_total_is_caught(self):
        made = Made(document=lambda pack: QuoteAST(
            blocks=[LineTableBlock(line_refs=[l.line_id for l in pack.lines])]))
        assert made.codes() == [Code.QUOTE_TOTAL_MISMATCH]

    def test_a_total_missing_from_the_page_is_caught(self):
        made = Made()
        start = made.html.index('<p class="total"')
        made.damaged(made.html[start:made.html.index("</p>", start) + 4], "")
        assert made.codes() == [Code.QUOTE_TOTAL_MISMATCH]

    def test_lines_that_do_not_add_up_block_the_quote(self):
        quote, internal = goods_quote()
        quote["total_ex_tax"] = Decimal("2000.00")
        assert Made((quote, internal)).codes() == [Code.QUOTE_TOTAL_MISMATCH]

    # -- a figure from nowhere --------------------------------------------------------
    def test_a_figure_the_quote_does_not_hold_is_caught(self):
        made = Made().damaged("</header>", "</header><p data-chunk>Deposit £500 on order</p>")
        assert made.codes() == [Code.REPORT_UNTRACED_FIGURE]

    def test_a_figure_drawn_outside_any_chunk_is_still_read(self):
        made = Made().damaged("</header>", "</header><p>Deposit £500 on order</p>")
        assert made.codes() == [Code.REPORT_UNTRACED_FIGURE]

    def test_a_digit_in_a_description_or_a_term_is_not_a_figure(self):
        quote, internal = goods_quote()
        quote["lines"][0]["item_description"] = "Synthetic bracket M8 x 60"
        assert Made((quote, internal), change=with_terms).codes() == []

    # -- what we pay ------------------------------------------------------------------
    @pytest.mark.parametrize("sentence", [
        "We clear 633.25 on this one",        # total margin
        "we clear £1,543.75 less than that",  # total cost, with a thousands separator
        "that is 29% for us",                 # 0.2909 as a person writes it
        "each one costs us 27.35",            # 27.3500 to two places
    ])
    def test_our_figure_in_a_note_is_caught(self, sentence):
        """A note is words, so the untraced scan does not read its digits. The leak check
        is the only thing between a sentence like this and the customer."""
        quote, internal = goods_quote()
        quote["justifications"][0]["claim"] = sentence
        assert Made((quote, internal)).codes() == [Code.INTERNAL_FIELD_LEAK]

    def test_our_figure_in_a_description_is_caught(self):
        quote, internal = goods_quote()
        quote["lines"][1]["item_description"] = "Synthetic fastener kit (27.35 to us)"
        assert Made((quote, internal)).codes() == [Code.INTERNAL_FIELD_LEAK]

    def test_a_cost_that_is_also_a_price_is_caught_the_second_time(self):
        """41.20 may print once, as line 2's price. Written again, it is ours."""
        quote, internal = goods_quote()
        quote["justifications"][0]["claim"] = "the bracket is 41.20 to us"
        assert Made((quote, internal)).codes() == [Code.INTERNAL_FIELD_LEAK]

    def test_our_figure_where_the_total_goes_is_caught(self):
        codes = Made().damaged("£2,177", "£1,543.75").codes()
        assert Code.INTERNAL_FIELD_LEAK in codes and Code.QUOTE_TOTAL_MISMATCH in codes

    def test_our_figure_under_a_price_is_caught(self):
        codes = Made().damaged(">£41.20<", ">£27.35<").codes()
        assert Code.INTERNAL_FIELD_LEAK in codes

    @pytest.mark.parametrize("hidden", [
        '<p data-chunk>see the Unit Cost column</p>',
        '<p data-note="total_margin"></p>',
        "<!-- line_margin_pct -->",
    ])
    def test_the_name_of_an_internal_field_anywhere_in_the_file_is_caught(self, hidden):
        assert Made().damaged("</header>", "</header>" + hidden).codes() \
            == [Code.INTERNAL_FIELD_LEAK]

    # -- its number and its standing ----------------------------------------------------
    def test_a_number_before_issue_is_caught(self):
        made = Made(change=identity(number="Q-2026-0007"))
        assert "Quote number Q-2026-0007" in made.html
        assert made.codes() == [Code.QUOTE_NUMBER_BEFORE_ISSUE]

    def test_an_issued_quote_shows_its_number(self):
        made = Made(change=identity(number="Q-2026-0007", status="issued"))
        assert "Quote number Q-2026-0007" in made.html and made.codes() == []

    def test_a_page_that_shows_a_number_the_quote_does_not_have_is_caught(self):
        made = Made().damaged("</header>", "<p data-chunk>Quote number Q-NINE</p></header>")
        assert made.codes() == [Code.QUOTE_NUMBER_BEFORE_ISSUE]

    @pytest.mark.parametrize("status", ["draft", "in_review", "expired", "superseded"])
    def test_a_quote_that_is_not_approved_is_caught(self, status):
        assert Made(change=identity(status=status)).codes() == [Code.QUOTE_NOT_APPROVED]

    # -- references and the record --------------------------------------------------------
    def test_a_reference_to_nothing_is_caught_and_shown(self):
        made = Made(document=lambda pack: QuoteAST(blocks=[
            LineTableBlock(line_refs=[l.line_id for l in pack.lines] + ["L777"]),
            TotalsBlock(fact_refs=[pack.total_fact_id]),
            TermsBlock(term_refs=["T0001"]), PaymentBlock(term_ref="T0002")]))
        assert "[unresolved L777]" in made.html and "[unresolved T0002]" in made.html
        assert made.codes().count(Code.UNKNOWN_FACT_REF) == 3

    def test_a_page_made_from_another_pack_is_caught(self):
        made = Made()
        made.page = dataclasses.replace(made.page, pack_hash="0" * 64)
        assert made.codes() == [Code.MISSING_HASH_RECORD]

    def test_a_page_with_no_record_is_caught(self):
        made = Made()
        made.page = dataclasses.replace(made.page, ast_hash="")
        assert made.codes() == [Code.MISSING_HASH_RECORD]


class TestThePdf:
    def test_the_pdf_is_printed_from_the_page_and_carries_the_same_record(self):
        made = Made()
        pdf = quote_html.to_pdf(made.page)
        assert pdf.content[:5] == b"%PDF-" and pdf.media_type == "application/pdf"
        assert (pdf.ast_hash, pdf.pack_hash, pdf.style_version) == (
            made.page.ast_hash, made.page.pack_hash, made.page.style_version)
        assert pdf.recorded()

    def test_the_pdf_reads_what_the_page_reads(self):
        import io

        import pdfplumber

        made = Made()
        with pdfplumber.open(io.BytesIO(quote_html.to_pdf(made.page).content)) as pdf:
            text = " ".join(page.extract_text() for page in pdf.pages)
        for shown in ("SQ-SYNTHETIC-A", "£173.6667", "£2,177", "Synthetic enclosure"):
            assert shown in text
        for ours in OURS:
            assert ours not in text.replace("£1,450", ""), ours

    def test_a_pdf_is_not_printed_from_anything_but_a_quote_page(self):
        made = Made()
        with pytest.raises(ValueError, match="quote_html"):
            quote_html.to_pdf(dataclasses.replace(made.page, renderer="html"))

    def test_a_pdf_cannot_be_put_through_the_gate_as_if_it_were_the_page(self):
        made = Made()
        made.page = quote_html.to_pdf(made.page)
        with pytest.raises(ValueError, match="no text reader"):
            made.codes()


class TestTheWholePath:
    def test_a_quote_on_record_becomes_a_checked_page(self):
        page = quote_pipeline.produce(*goods_quote())
        assert page.renderer == "quote_html" and b"SQ-SYNTHETIC-A" in page.content

    def test_the_page_and_the_pdf_come_from_one_document(self):
        page = quote_pipeline.produce(*goods_quote())
        pdf = quote_pipeline.produce(*goods_quote(), fmt="pdf")
        assert pdf.content[:5] == b"%PDF-"
        assert pdf.ast_hash == page.ast_hash and pdf.pack_hash == page.pack_hash

    def test_the_page_cannot_be_asked_for_without_the_record_the_gate_searches_for(self):
        with pytest.raises(TypeError):
            quote_pipeline.produce(goods_quote()[0])

    def test_a_quote_that_is_not_approved_has_no_page(self):
        with pytest.raises(NotCustomerReady):
            quote_pipeline.produce(*goods_quote(status="draft"))

    def test_our_figure_on_the_page_stops_it_and_names_the_figure(self):
        quote, internal = goods_quote()
        quote["justifications"][0]["claim"] = "each one costs us 27.35"
        with pytest.raises(InternalFieldLeak, match="line 2 unit_cost"):
            quote_pipeline.produce(quote, internal)

    def test_any_other_failed_check_stops_it_and_says_why(self):
        quote, internal = goods_quote()
        quote["total_ex_tax"] = Decimal("2000.00")
        with pytest.raises(quote_pipeline.QuoteBlocked, match="QUOTE_TOTAL_MISMATCH") as stopped:
            quote_pipeline.produce(quote, internal)
        assert isinstance(stopped.value, StateConflict)

    def test_a_blocked_quote_has_no_pdf_either(self):
        quote, internal = goods_quote()
        quote["total_ex_tax"] = Decimal("2000.00")
        with pytest.raises(quote_pipeline.QuoteBlocked):
            quote_pipeline.produce(quote, internal, fmt="pdf")

    def test_only_a_page_or_a_pdf_can_be_asked_for(self):
        with pytest.raises(ValueError, match="docx"):
            quote_pipeline.produce(*goods_quote(), fmt="docx")

    def test_the_record_handed_in_must_be_the_quote_asked_for(self):
        quote = goods_quote()[0]
        with pytest.raises(ValueError, match="9001"):
            build_fact_pack(sales_quote.REPORT_TYPE_ID, scope={"sales_quote_id": 1},
                            as_of="2026-09-14", source=quote, emit_audit=False)
