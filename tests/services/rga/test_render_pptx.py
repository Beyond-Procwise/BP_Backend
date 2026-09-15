"""The deck: reproducible bytes, and disclosures the renderer draws itself.

The badge and footnote tests matter more than they look. §5 says confidence
badges and provenance footnotes are rendered by the renderer, not by the model —
a disclosure a composer can choose to leave out is not a disclosure. These
assert the renderer emits them from the pack, with no cooperation from anything
upstream.
"""

from __future__ import annotations

import io
import zipfile
from decimal import Decimal

from src.services.analytics.models import Confidence
from src.services.rga.models import (
    FactPack,
    FormatHint,
    MetricBlock,
    Origin,
    ReportAST,
    Section,
)
from src.services.rga.render import pptx as renderer
from src.services.rga.style import resolve_style_brief

from tests.services.rga.conftest import make_fact, make_pack


def text_of(content: bytes) -> str:
    return "\n".join(renderer.extract_text(content))


class TestReproducibility:
    def test_two_renders_of_one_report_are_byte_identical(self, ast, pack, brief):
        assert renderer.render(ast, pack, brief).content == \
               renderer.render(ast, pack, brief).content

    def test_every_zip_entry_carries_a_fixed_timestamp(self, ast, pack, brief):
        """Without this the same deck differs between two seconds, for reasons
        that have nothing to do with the report."""
        content = renderer.render(ast, pack, brief).content
        stamps = {i.date_time for i in zipfile.ZipFile(io.BytesIO(content)).infolist()}

        assert stamps == {(1980, 1, 1, 0, 0, 0)}

    def test_it_regenerates_from_stored_inputs_with_no_model_in_the_loop(
            self, ast, pack, brief):
        """DoD12. Nothing here calls a language model; the AST is an input."""
        original = renderer.render(ast, pack, brief)

        reloaded_pack = FactPack.from_stored(pack.stored())
        reloaded_ast = ReportAST.model_validate(ast.model_dump(mode="json"))
        again = renderer.render(reloaded_ast, reloaded_pack, brief)

        assert again.content == original.content
        assert again.ast_hash == original.ast_hash
        assert again.pack_hash == original.pack_hash

    def test_the_artefact_carries_a_complete_reproducibility_record(
            self, ast, pack, brief):
        artefact = renderer.render(ast, pack, brief)

        assert artefact.recorded()
        assert artefact.pack_hash == pack.hash
        assert artefact.style_version == brief.version()
        assert artefact.renderer_version == renderer.RENDERER_VERSION

    def test_a_changed_figure_changes_the_bytes(self, ast, facts, brief):
        moved = list(facts)
        moved[0] = make_fact(1, "Invoiced spend (GBP)", Decimal("9999999.00"),
                             FormatHint.MONEY, Confidence.CORROBORATED,
                             currency="GBP")

        assert renderer.render(ast, make_pack(moved), brief).content != \
               renderer.render(ast, make_pack(facts), brief).content


class TestTheRendererDrawsTheDisclosures:
    def test_confidence_badges_come_from_the_pack(self, ast, pack, brief):
        shown = text_of(renderer.render(ast, pack, brief).content)

        assert "CORROBORATED" in shown
        assert "ASSERTED" in shown

    def test_provenance_footnotes_name_the_derivation_and_the_audit_row(
            self, ast, pack, brief):
        shown = text_of(renderer.render(ast, pack, brief).content)

        assert "[F0001]" in shown
        assert "fixture.q1" in shown
        assert "bp_agent_actions:FP-fixture#F0001" in shown

    def test_badges_can_be_switched_off_but_origin_cannot(self, ast, facts, brief):
        """A LEGACY_UNVERIFIED figure that looks verified is the specific
        misreading the origin badge exists to prevent, so it is not governed by
        the confidence-badge switch."""
        legacy = list(facts)
        legacy[0] = make_fact(1, "Invoiced spend (GBP)", Decimal("5833817.90"),
                              FormatHint.MONEY, Confidence.CORROBORATED,
                              currency="GBP", origin=Origin.LEGACY_UNVERIFIED)
        quiet = resolve_style_brief("exec_procurement_summary")
        object.__setattr__(quiet, "values",
                           dict(quiet.values, **{"report.style.show_confidence_badges": False}))

        shown = text_of(renderer.render(ast, make_pack(legacy), quiet).content)

        assert "CORROBORATED" not in shown
        assert "LEGACY_UNVERIFIED" in shown

    def test_an_unassessed_figure_renders_as_a_dash_never_a_zero(
            self, ast, pack, brief):
        shown = text_of(renderer.render(ast, pack, brief).content)

        assert "—" in shown
        assert "UNASSESSED" in shown

    def test_the_front_page_counts_what_could_not_be_measured(
            self, ast, pack, brief):
        artefact = renderer.render(ast, pack, brief)

        assert artefact.coverage_disclosure == "1 of 4 measures not assessed"
        assert artefact.coverage_disclosure in text_of(artefact.content)

    def test_the_page_admits_the_style_was_never_scoped(self, ast, pack, brief):
        shown = text_of(renderer.render(ast, pack, brief).content)

        assert "platform_default" in shown

    def test_a_narrative_placeholder_is_substituted_from_the_pack(
            self, pack, brief):
        from src.services.rga.models import NarrativeBlock

        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            NarrativeBlock(text="Spend was {{F0001}}.", fact_refs=["F0001"])])])

        shown = text_of(renderer.render(ast, pack, brief).content)

        assert "Spend was £5.8M" in shown
        assert "{{F0001}}" not in shown

    def test_a_fact_a_sentence_rests_on_is_stamped_even_if_not_quoted(
            self, pack, brief):
        """"No opportunities were identified" rests on a count of zero without
        printing it. The reliance still has to be visible, or the claim is
        unsourced on the page — live, three such facts were referenced by the
        report and drawn nowhere in it."""
        from src.services.rga.models import NarrativeBlock

        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            NarrativeBlock(text="No realised savings were recorded.",
                           fact_refs=["F0004"])])])

        shown = text_of(renderer.render(ast, pack, brief).content)

        assert "{{" not in shown          # nothing to substitute
        assert "[F0004]" in shown         # but the reliance is stamped
        assert "UNASSESSED" in shown
        assert "fixture.q4" in shown      # and footnoted

    def test_an_unresolvable_reference_stays_visible_rather_than_blanking(
            self, pack, brief):
        """A sentence that lost its number silently reads as finished prose."""
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            MetricBlock(fact_ref="F9999")])])

        assert "F9999" in text_of(renderer.render(ast, pack, brief).content)
