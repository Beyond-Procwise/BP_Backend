"""The contract: a model cannot originate a number, and a fact cannot exist
without evidence.

These are the type-level halves of DoD2 and DoD7. Each asserts a *refusal* —
that the object cannot be constructed — rather than that a checker later
notices. A validation that can be skipped by a caller who forgot to call it is
not the guarantee this design rests on.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.services.analytics.models import Confidence
from src.services.rga.models import (
    FactEntry,
    FormatHint,
    MetricBlock,
    NarrativeBlock,
    Origin,
    ReportAST,
    Section,
    TableBlock,
    canonical_hash,
)

from tests.services.rga.conftest import make_fact, make_pack


class TestFactEntry:
    def test_a_fact_without_provenance_cannot_be_built(self):
        with pytest.raises(ValidationError, match="provenance_id"):
            FactEntry(
                fact_id="F0001", label="Invoiced spend", value=Decimal("1"),
                format_hint=FormatHint.MONEY, confidence=Confidence.ASSERTED,
                origin=Origin.OBSERVED, provenance_id="", derivation="d")

    def test_whitespace_is_not_provenance(self):
        """``min_length=1`` alone would accept a space, which evidences nothing."""
        with pytest.raises(ValidationError, match="must carry a provenance_id"):
            FactEntry(
                fact_id="F0001", label="Invoiced spend", value=Decimal("1"),
                format_hint=FormatHint.MONEY, confidence=Confidence.ASSERTED,
                origin=Origin.OBSERVED, provenance_id="   ", derivation="d")

    def test_fact_id_shape_is_enforced(self):
        with pytest.raises(ValidationError, match="F0042"):
            FactEntry(
                fact_id="spend", label="x", value=None, format_hint=FormatHint.TEXT,
                confidence=Confidence.UNASSESSED, origin=Origin.OBSERVED,
                provenance_id="p", derivation="d")

    def test_an_absent_value_renders_as_a_dash_not_a_zero(self):
        """A measure nobody could take is not a measured nothing."""
        entry = make_fact(4, "Realised savings", None, FormatHint.TEXT,
                          Confidence.UNASSESSED)
        assert entry.display == "—"
        assert "0" not in entry.display

    def test_display_comes_from_the_shared_formatter(self):
        entry = make_fact(1, "Spend", Decimal("5833817.90"), FormatHint.MONEY,
                          Confidence.CORROBORATED, currency="GBP")
        assert entry.display == "£5.8M"
        assert entry.tokens() == {"5.8"}

    def test_confidence_and_origin_stay_separate_fields(self):
        """Merging them would render a corroborated aggregate over unverified
        rows as something else entirely."""
        entry = make_fact(1, "Spend", Decimal("1000"), FormatHint.MONEY,
                          Confidence.CORROBORATED, currency="GBP",
                          origin=Origin.LEGACY_UNVERIFIED)
        assert entry.confidence is Confidence.CORROBORATED
        assert entry.origin is Origin.LEGACY_UNVERIFIED


class TestNarrativeCannotCarryAFigure:
    def test_a_literal_number_in_prose_is_refused(self):
        with pytest.raises(ValidationError, match="literal number"):
            NarrativeBlock(text="Spend rose to 5.8M this quarter.")

    def test_a_placeholder_is_the_only_way_a_number_reaches_a_sentence(self):
        block = NarrativeBlock(text="Spend was {{F0001}}.", fact_refs=["F0001"])
        assert block.placeholders() == ["F0001"]

    def test_even_a_bare_digit_is_refused(self):
        """'the top 3 suppliers' is a figure nobody measured."""
        with pytest.raises(ValidationError):
            NarrativeBlock(text="The top 3 suppliers dominate.")

    def test_a_table_cell_may_not_carry_a_figure(self):
        with pytest.raises(ValidationError, match="literal figure"):
            TableBlock(columns=["Measure", "Value"], rows=[["Spend", "£5.8M"]])

    def test_a_table_cell_may_carry_a_fact_id_or_a_label(self):
        table = TableBlock(columns=["Measure", "Value"],
                           rows=[["Three-way match rate", "F0003"]])
        assert table.rows[0][1] == "F0003"

    def test_a_table_with_no_columns_is_refused(self):
        """It reached python-pptx live and came back as ZeroDivisionError —
        the renderer divides the width by the column count."""
        with pytest.raises(ValidationError, match="at least one column"):
            TableBlock(columns=[], rows=[])

    def test_a_ragged_row_is_refused(self):
        with pytest.raises(ValidationError, match="ragged row"):
            TableBlock(columns=["Measure", "Value"],
                       rows=[["Match rate", "F0003"], ["Cycle"]])

    def test_a_placeholder_in_a_cell_is_normalised_to_the_bare_id(self):
        """One syntax, not two. A composer told to write {{F0003}} in prose
        writes it in a cell too, and rejecting that threw away an otherwise
        correct report with a misleading message."""
        table = TableBlock(columns=["Measure", "Value"],
                           rows=[["Match rate", "{{F0003}}"]])

        assert table.rows[0][1] == "F0003"

    def test_the_digit_error_quotes_the_offending_sentence(self):
        """"001, 03, 87" alone is unactionable — they are fragments of an
        identifier the composer wrote, and it needs to see which words to cut."""
        with pytest.raises(ValidationError, match="Prior spend was FND001"):
            NarrativeBlock(text="Prior spend was FND001 on the ledger.")


class TestPackIdentity:
    def test_same_content_hashes_the_same(self, facts):
        assert make_pack(facts).hash == make_pack(facts).hash

    def test_the_hash_ignores_the_clock(self, facts):
        """Two builds of one snapshot must not differ because time passed."""
        from datetime import datetime, timezone

        early = make_pack(facts)
        late = early.model_copy(update={
            "generated_at": datetime(2027, 1, 1, tzinfo=timezone.utc)})
        assert late.hash == early.hash

    def test_a_changed_figure_changes_the_hash(self, facts):
        moved = list(facts)
        moved[0] = make_fact(1, "Invoiced spend (GBP)", Decimal("9999999.00"),
                             FormatHint.MONEY, Confidence.CORROBORATED,
                             currency="GBP")
        assert make_pack(moved).hash != make_pack(facts).hash

    def test_quotable_tokens_cover_the_facts_and_the_scope(self, pack):
        tokens = pack.quotable_tokens()
        assert "5.8" in tokens        # the money figure as rendered
        assert "376" in tokens        # the count
        assert "2026" in tokens       # the period, from the scope line
        assert "9999999" not in tokens

    def test_ast_collects_refs_from_every_block_kind(self, ast):
        assert ast.fact_refs() == {"F0001", "F0002", "F0003", "F0004"}


def test_canonical_hash_is_order_independent():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
