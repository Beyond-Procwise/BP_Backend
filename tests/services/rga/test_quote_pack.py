"""The quote pack and the quote document tree (QDA B2).

No database. The pack is built from the two SYNTHETIC quotes under
``tests/sell_side/fixtures`` -- there is no real sell-side quote to test against.

What is held here:
  * the pack is built from the customer record alone, so nothing we pay is in it;
  * a quote document names things in the pack by id and holds nothing else -- no
    figure, no legal wording, no name -- so a hand-written number is not constructible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.services.rga import factpack
from src.services.rga.builders import sales_quote
from src.services.rga.models import FactPack, FindingCode
from src.services.rga.quote_models import (
    LineTableBlock,
    PaymentBlock,
    QuoteAST,
    QuotePack,
    QuoteTerm,
    SignatureBlock,
    TermsBlock,
    TotalsBlock,
    unresolved_refs,
)
from tests.sell_side.fixtures.synthetic_quotes import goods_quote, unknown_cost_quote

NOW = datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc)


def build(quote, monkeypatch, *, now=NOW) -> QuotePack:
    monkeypatch.setattr(sales_quote, "_read_quote", lambda quote_id: quote)
    return factpack.build_fact_pack(
        sales_quote.REPORT_TYPE_ID, scope={"sales_quote_id": quote["sales_quote_id"]},
        as_of=quote["quote_date"].isoformat(), now=now, emit_audit=False)


@pytest.fixture
def pack(monkeypatch) -> QuotePack:
    return build(goods_quote()[0], monkeypatch)


def with_terms(pack: QuotePack) -> QuotePack:
    """The same pack with two terms in it. No terms library exists yet (B9), so a real
    pack carries none; these are invented to exercise the references."""
    terms = [QuoteTerm(term_id="T0001", kind="general", text="Synthetic general term.",
                       provenance_id="synthetic:term#1"),
             QuoteTerm(term_id="T0002", kind="payment", text="Synthetic payment term.",
                       provenance_id="synthetic:term#2")]
    return pack.model_copy(update={"terms": terms})


def whole_quote(pack: QuotePack) -> QuoteAST:
    return QuoteAST(blocks=[
        LineTableBlock(line_refs=[l.line_id for l in pack.lines]),
        TotalsBlock(fact_refs=[pack.total_fact_id]),
        TermsBlock(term_refs=["T0001"]),
        PaymentBlock(term_ref="T0002"),
        SignatureBlock(party="supplier"),
        SignatureBlock(party="customer"),
    ])


class TestThePack:
    def test_it_is_a_fact_pack_registered_like_any_other(self, pack):
        assert isinstance(pack, QuotePack) and isinstance(pack, FactPack)
        assert sales_quote.REPORT_TYPE_ID in factpack.registered_types()

    def test_every_line_is_in_it_under_its_own_id(self, pack):
        assert [l.line_id for l in pack.lines] == ["L501", "L502", "L503"]
        line = pack.line("L503")
        assert (line.line_no, line.sku, line.description) == (3, "SYN-C", "Synthetic enclosure")
        assert (line.quantity, line.unit_price, line.line_total) == (
            Decimal("3"), Decimal("173.6667"), Decimal("521.00"))
        assert pack.line("L999") is None

    def test_every_line_says_which_row_it_was_read_from(self, pack):
        assert pack.line("L502").provenance_id == "proc.bp_sales_quote_line:502"

    def test_the_total_is_a_fact_with_evidence(self, pack):
        total = pack.fact(pack.total_fact_id)
        assert total.value == Decimal("2177.00") and total.currency == "GBP"
        assert total.display == "£2,177"
        assert total.provenance_id == "proc.bp_sales_quote:9001#total_ex_tax"

    def test_who_the_quote_is_for_comes_from_the_record(self, pack):
        who = pack.identity
        assert (who.quote_ref, who.account_name, who.contact_name) == (
            "SQ-SYNTHETIC-A", "Synthetic Fixture Co <Ltd>", "Sam Sample")
        assert (who.quote_date.isoformat(), who.valid_until.isoformat()) == (
            "2026-09-14", "2026-11-13")

    def test_it_has_no_number_until_it_is_given_one(self, pack):
        """The draft reference is not the quote's number."""
        assert pack.identity.number is None

    def test_its_number_is_the_one_on_the_record(self, monkeypatch):
        quote = goods_quote("issued")[0]
        quote["quote_number"] = "Q-2026-0007"
        numbered = build(quote, monkeypatch)
        assert numbered.identity.number == "Q-2026-0007"
        assert numbered.hash != build(goods_quote("issued")[0], monkeypatch).hash

    def test_it_carries_no_terms_because_no_library_exists(self, pack):
        assert pack.terms == []

    def test_nothing_we_pay_is_in_it(self, monkeypatch):
        """The builder is handed the customer record only, so this holds by construction;
        the search is here so that a later change to what it is handed goes red."""
        quote, internal = goods_quote()
        stored = str(build(quote, monkeypatch).stored())
        for ours in ("1543.75", "633.25", "27.35", "420.00", "125.6667", "0.2909"):
            assert ours not in stored
        for name in ("unit_cost", "total_cost", "margin", "cost_tier"):
            assert name not in stored

    def test_a_cost_handed_to_the_builder_by_mistake_is_refused(self, monkeypatch):
        quote, internal = goods_quote()
        quote["total_cost"] = internal["total_cost"]
        with pytest.raises(ValueError, match="total_cost"):
            build(quote, monkeypatch)
        quote, internal = goods_quote()
        quote["lines"][0]["unit_cost"] = internal["lines"][0]["unit_cost"]
        with pytest.raises(ValueError, match="unit_cost"):
            build(quote, monkeypatch)

    def test_the_same_quote_packs_to_the_same_hash(self, monkeypatch):
        first = build(goods_quote()[0], monkeypatch)
        later = build(goods_quote()[0], monkeypatch,
                      now=datetime(2027, 1, 1, tzinfo=timezone.utc))
        assert first.hash == later.hash

    def test_a_changed_line_changes_the_hash(self, monkeypatch):
        """The hash covers the lines, not only the facts: the total below is unchanged."""
        quote = goods_quote()[0]
        changed = goods_quote()[0]
        changed["lines"][0]["item_description"] = "Synthetic bracket, no plate"
        assert build(quote, monkeypatch).hash != build(changed, monkeypatch).hash

    def test_a_reworded_note_changes_the_hash(self, monkeypatch):
        changed = goods_quote()[0]
        changed["justifications"][0]["claim"] = "Line 3 replaces a part"
        assert build(goods_quote()[0], monkeypatch).hash != build(changed, monkeypatch).hash

    def test_a_different_customer_changes_the_hash(self, monkeypatch):
        changed = goods_quote()[0]
        changed["account_name"] = "Somebody Else"
        assert build(goods_quote()[0], monkeypatch).hash != build(changed, monkeypatch).hash

    def test_it_survives_being_stored_and_a_tampered_copy_does_not(self, pack):
        stored = with_terms(pack).stored()
        again = QuotePack.from_stored(stored)
        assert again.hash == stored["pack_hash"] and again.line("L501").quantity == Decimal("25")
        stored["pack"]["lines"][0]["unit_price"] = "1.0000"
        with pytest.raises(ValueError, match="altered"):
            QuotePack.from_stored(stored)

    def test_lines_that_do_not_add_up_to_the_total_block_the_quote(self, monkeypatch):
        quote = goods_quote()[0]
        quote["total_ex_tax"] = Decimal("2000.00")
        blocking = build(quote, monkeypatch).blocking_findings()
        assert [f.code for f in blocking] == [FindingCode.QUOTE_TOTAL_MISMATCH]
        assert "2,177" in blocking[0].detail and "2,000" in blocking[0].detail

    def test_a_quote_that_adds_up_raises_nothing(self, pack, monkeypatch):
        assert pack.findings == []
        assert build(unknown_cost_quote()[0], monkeypatch).findings == []

    def test_two_lines_cannot_share_an_id(self, pack):
        with pytest.raises(ValidationError, match="L501"):
            QuotePack.model_validate({**dict(pack), "lines": [pack.lines[0], pack.lines[0]]})

    def test_a_report_pack_hashes_exactly_as_it_did(self):
        """The hash was restructured so a quote pack can add to it. A report pack already
        stored under its old hash must still open: this value was taken before the change."""
        plain = FactPack(pack_id="FP-fixed", report_type_id="t", scope={"a": 1},
                         as_of="2026-01-01", generated_at=NOW, generated_by="test")
        assert plain.hash == (
            "281ce7fb6c8c9dac617dccde8b67f70af903e2cd8e9501943bafc985c69aa33a")


class TestTheDocument:
    def test_a_hand_written_document_for_one_quote_is_valid(self, pack):
        terms = with_terms(pack)
        document = whole_quote(terms)
        assert unresolved_refs(document, terms) == []
        assert QuoteAST.model_validate(document.model_dump(mode="json")) == document

    def test_the_same_document_as_plain_json_is_valid(self):
        QuoteAST.model_validate({"blocks": [
            {"type": "line_table", "line_refs": ["L501", "L502"]},
            {"type": "totals", "fact_refs": ["F0001"]},
            {"type": "terms", "term_refs": ["T0001"]},
            {"type": "payment", "term_ref": "T0002"},
            {"type": "signature", "party": "customer"},
        ]})

    @pytest.mark.parametrize("block", [
        # a figure under a name of its own
        {"type": "totals", "fact_refs": ["F0001"], "total": "2177.00"},
        {"type": "line_table", "line_refs": ["L501"], "unit_price": 58},
        # a figure where an id belongs
        {"type": "totals", "fact_refs": ["2177.00"]},
        {"type": "line_table", "line_refs": ["58.0000"]},
        {"type": "line_table", "line_refs": [501]},
        # legal wording where an id belongs
        {"type": "terms", "term_refs": ["Payment within 30 days"]},
        {"type": "payment", "term_ref": "Net 30"},
        {"type": "payment", "term_ref": "T0002", "due_days": 30},
        # a name where a party belongs
        {"type": "signature", "party": "Sam Sample"},
        {"type": "signature", "party": "customer", "name": "Sam Sample"},
        # a heading of the writer's own
        {"type": "line_table", "line_refs": ["L501"], "columns": ["Our price: 58"]},
        {"type": "line_table", "line_refs": ["L501"], "title": "Lines"},
    ])
    def test_a_free_figure_name_or_wording_is_not_constructible(self, block):
        with pytest.raises(ValidationError):
            QuoteAST.model_validate({"blocks": [block]})

    def test_the_document_itself_takes_no_free_field(self):
        with pytest.raises(ValidationError):
            QuoteAST.model_validate({"blocks": [], "note": "about 2,000"})

    def test_a_block_with_no_type_is_refused_not_guessed(self):
        with pytest.raises(ValidationError):
            QuoteAST.model_validate({"blocks": [{"line_refs": ["L501"]}]})

    def test_a_report_block_is_not_a_quote_block(self):
        """Prose is where a model writes. A customer quote has none."""
        with pytest.raises(ValidationError):
            QuoteAST.model_validate({"blocks": [{"type": "narrative", "text": "A fair price."}]})

    @pytest.mark.parametrize("block", [
        {"type": "line_table", "line_refs": []},
        {"type": "totals", "fact_refs": []},
        {"type": "terms", "term_refs": []},
    ])
    def test_an_empty_block_is_refused(self, block):
        with pytest.raises(ValidationError):
            QuoteAST.model_validate({"blocks": [block]})

    @pytest.mark.parametrize("dropped", ["description", "quantity", "unit_price", "line_total"])
    def test_a_line_table_cannot_leave_out_what_makes_it_an_offer(self, dropped):
        columns = [c for c in ("description", "quantity", "unit_price", "line_total")
                   if c != dropped]
        with pytest.raises(ValidationError, match=dropped):
            LineTableBlock(line_refs=["L501"], columns=columns)

    def test_a_line_table_may_leave_out_the_rest(self):
        LineTableBlock(line_refs=["L501"],
                       columns=["description", "quantity", "unit_price", "line_total"])

    def test_a_note_is_named_by_id_like_everything_else(self, pack):
        from src.services.rga.quote_models import NotesBlock

        assert [n.note_id for n in pack.notes] == ["N801"]
        assert pack.note("N801").provenance_id == "proc.bp_sales_justification:801"
        assert unresolved_refs(QuoteAST(blocks=[NotesBlock(note_refs=["N801", "N9"])]),
                               pack) == ["N9"]
        with pytest.raises(ValidationError):
            NotesBlock(note_refs=["Line 3 replaces a withdrawn part"])

    def test_a_line_cannot_be_listed_twice(self):
        with pytest.raises(ValidationError, match="L501"):
            LineTableBlock(line_refs=["L501", "L501"])

    def test_a_reference_to_something_not_in_the_pack_is_reported(self, pack):
        document = QuoteAST(blocks=[
            LineTableBlock(line_refs=["L501", "L777"]),
            TotalsBlock(fact_refs=["F0099"]),
            TermsBlock(term_refs=["T0001"]),
            PaymentBlock(term_ref="T0002"),
        ])
        assert unresolved_refs(document, pack) == ["F0099", "L777", "T0001", "T0002"]

    def test_a_general_term_cannot_stand_in_for_the_payment_term(self, pack):
        document = QuoteAST(blocks=[PaymentBlock(term_ref="T0001")])
        assert unresolved_refs(document, with_terms(pack)) == ["T0001"]

    def test_the_report_tree_is_unchanged(self):
        """The composer's grammar is built from the report blocks. Quote blocks are a
        separate union so a model writing a report is never offered one."""
        from src.services.rga.compose import CompositionDraft

        schema = str(CompositionDraft.model_json_schema())
        for name in ("line_table", "signature", "term_ref"):
            assert name not in schema
