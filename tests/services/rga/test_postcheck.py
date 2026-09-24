"""The gate, proved by breaking it.

Every test here asserts a BLOCK. A post-check that has only ever been run
against a correct report is a post-check nobody has tested — this codebase has
shipped guards that were green because they were checking nothing, so each
failure class in §6 gets a case that deliberately triggers it, and the clean
control at the bottom proves the gate is not simply refusing everything.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from src.services.analytics.models import Confidence
from src.services.rga import postcheck
from src.services.rga.models import (
    FindingCode,
    FormatHint,
    MetricBlock,
    NarrativeBlock,
    Origin,
    ReportAST,
    Section,
    TableBlock,
)
from src.services.rga.render import pptx as renderer
from src.services.rga.style import resolve_style_brief

from tests.services.rga.conftest import make_fact, make_pack


def codes(result) -> set[str]:
    return {f.code for f in result.blocking}


def check(ast, pack, brief, *, artefact=None):
    artefact = artefact if artefact is not None else renderer.render(ast, pack, brief)
    return postcheck.run(artefact, pack, ast, brief, emit_audit=False)


class TestTheCleanReportPasses:
    def test_a_correct_report_is_not_blocked(self, ast, pack, brief):
        """The control. Without it every other test in this file could pass
        because the gate refuses everything."""
        result = check(ast, pack, brief)

        assert result.passed, [f.detail for f in result.blocking]
        assert result.findings == []


class TestAReportMustSaySomething:
    def test_a_report_referencing_no_facts_blocks_release(self, pack, brief):
        """Found live. Every other check in §6 is conditional on a figure being
        present, so a report with none passes them all and releases clean —
        which is a gate that is green because it checked nothing."""
        empty = ReportAST.model_validate({"sections": [
            {"id": "s", "title": "Executive Summary", "blocks": [
                {"type": "finding_list", "finding_refs": []} for _ in range(5)]}]})

        result = check(empty, pack, brief)

        assert not result.passed
        assert FindingCode.REPORT_STATES_NOTHING in codes(result)

    def test_an_empty_pack_is_not_held_to_it(self, brief):
        """Nothing measured means nothing to state, and that is not the
        report's fault."""
        empty_pack = make_pack([], findings=[])
        ast = ReportAST.model_validate({"sections": [
            {"id": "s", "title": "S", "blocks": [
                {"type": "finding_list", "finding_refs": []}]}]})

        assert check(ast, empty_pack, brief).passed


class TestUntracedFigures:
    def test_a_fabricated_figure_blocks_release(self, pack, brief):
        """A column header carrying a figure nothing measured. Table headers are
        author text, so the type cannot refuse this — the gate must."""
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            MetricBlock(fact_ref="F0001"),
            TableBlock(columns=["Target £9.9M"], rows=[["F0001"]])])])

        result = check(ast, pack, brief)

        assert not result.passed
        assert FindingCode.REPORT_UNTRACED_FIGURE in codes(result)
        assert any("9.9" in f.detail for f in result.blocking)

    def test_an_altered_figure_blocks_release(self, ast, pack, brief):
        """The artefact is drawn from one pack and checked against another —
        which is what a figure edited after rendering looks like from here."""
        artefact = renderer.render(ast, pack, brief)
        altered = make_pack([make_fact(1, "Invoiced spend (GBP)",
                                       Decimal("9999999.00"), FormatHint.MONEY,
                                       Confidence.CORROBORATED, currency="GBP")])

        result = postcheck.run(artefact, altered, ast, brief, emit_audit=False)

        assert not result.passed
        assert FindingCode.REPORT_UNTRACED_FIGURE in codes(result)

    def test_the_reports_own_provenance_is_not_mistaken_for_a_figure(
            self, ast, pack, brief):
        """A sha256 is mostly digits. Failing a report for disclosing its own
        hash would make the honest thing the expensive one."""
        artefact = renderer.render(ast, pack, brief)
        shown = "\n".join(renderer.extract_text(artefact.content))

        assert artefact.pack_hash[:12] in shown          # it really is on the page
        assert check(ast, pack, brief).passed            # and it does not trip the scan


class TestReferences:
    def test_a_reference_to_a_fact_not_in_the_pack_blocks_release(self, pack, brief):
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            MetricBlock(fact_ref="F9999")])])

        result = check(ast, pack, brief)

        assert not result.passed
        assert FindingCode.UNKNOWN_FACT_REF in codes(result)


class TestUnassessedFigures:
    def test_an_unassessed_fact_may_not_underpin_a_recommendation(self, pack, brief):
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            NarrativeBlock(text="Prioritise savings capture, given {{F0004}}.",
                           fact_refs=["F0004"], role="recommendation")])])

        result = check(ast, pack, brief)

        assert not result.passed
        assert FindingCode.UNASSESSED_IN_RECOMMENDATION in codes(result)

    def test_but_it_may_be_stated_as_a_fact(self, pack, brief):
        """Saying a measure is unavailable is what an honest report does."""
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            NarrativeBlock(text="Realised savings stand at {{F0004}}.",
                           fact_refs=["F0004"], role="statement")])])

        assert check(ast, pack, brief).passed


class TestBadgeLocality:
    def test_a_figure_drawn_without_its_badge_blocks_release(
            self, ast, pack, brief):
        """Rendered with badges off, checked under a brief that requires them.

        This is the case the first version of the check missed: it looked for
        the badge string anywhere in the deck, so one fact's disclosure covered
        for every other fact's missing one.
        """
        import dataclasses as _dc

        quiet = resolve_style_brief("exec_procurement_summary")
        object.__setattr__(quiet, "values", dict(
            quiet.values, **{"report.style.show_confidence_badges": False}))
        artefact = renderer.render(ast, pack, quiet)

        # ...but the governing brief says badges are shown.
        result = postcheck.run(artefact, pack, ast, brief, emit_audit=False)

        assert not result.passed
        assert FindingCode.MISSING_ORIGIN_BADGE in codes(result)

    def test_an_unassessed_figure_in_a_table_carries_its_badge(
            self, ast, pack, brief):
        """F0004 appears only in a table cell. It rendered as a bare dash until
        the renderer drew badges outside metric blocks."""
        artefact = renderer.render(ast, pack, brief)
        cells = [c for c in renderer.extract_text(artefact.content)
                 if "[F0004]" in c]

        assert cells, "F0004 is drawn nowhere"
        assert any("UNASSESSED" in c for c in cells)


class TestOriginBadge:
    def test_a_legacy_unverified_fact_without_its_badge_blocks_release(
            self, ast, facts, brief):
        """Drawn while the fact was OBSERVED, checked once it is known to be
        LEGACY_UNVERIFIED — so the badge the renderer would have drawn is
        genuinely absent from the bytes."""
        artefact = renderer.render(ast, make_pack(facts), brief)
        legacy = list(facts)
        legacy[0] = make_fact(1, "Invoiced spend (GBP)", Decimal("5833817.90"),
                              FormatHint.MONEY, Confidence.CORROBORATED,
                              currency="GBP", origin=Origin.LEGACY_UNVERIFIED)

        result = postcheck.run(artefact, make_pack(legacy), ast, brief,
                               emit_audit=False)

        assert FindingCode.MISSING_ORIGIN_BADGE in codes(result)


class TestProvenanceFootnotes:
    def test_a_figure_printed_without_its_footnote_blocks_release(
            self, ast, pack, brief):
        """Rendered with footnotes suppressed, checked under a brief that
        requires them. The badge and the footnote are different disclosures —
        a report can keep one and lose the other."""
        quiet = resolve_style_brief("exec_procurement_summary")
        object.__setattr__(quiet, "values", dict(
            quiet.values, **{"report.style.show_provenance_footnotes": False}))
        artefact = renderer.render(ast, pack, quiet)

        result = postcheck.run(artefact, pack, ast, brief, emit_audit=False)

        assert not result.passed
        assert FindingCode.MISSING_PROVENANCE_FOOTNOTE in codes(result)

    def test_the_footnote_names_the_audit_row_the_figure_came_from(
            self, ast, pack, brief):
        artefact = renderer.render(ast, pack, brief)
        shown = renderer.extract_text(artefact.content)

        assert any("[F0001]" in c and "bp_agent_actions:FP-fixture#F0001" in c
                   for c in shown)


class TestTheArtefactRecord:
    def test_a_missing_style_provenance_chain_blocks_release(
            self, ast, pack, brief):
        """Without it, 'why did this report look like this' has no answer."""
        artefact = dataclasses.replace(renderer.render(ast, pack, brief),
                                       style_provenance={})

        result = postcheck.run(artefact, pack, ast, brief, emit_audit=False)

        assert FindingCode.MISSING_STYLE_PROVENANCE in codes(result)

    def test_an_incomplete_reproducibility_record_blocks_release(
            self, ast, pack, brief):
        artefact = dataclasses.replace(renderer.render(ast, pack, brief),
                                       ast_hash="")

        result = postcheck.run(artefact, pack, ast, brief, emit_audit=False)

        assert FindingCode.MISSING_HASH_RECORD in codes(result)

    def test_a_record_naming_the_wrong_pack_blocks_release(
            self, ast, pack, brief):
        artefact = dataclasses.replace(renderer.render(ast, pack, brief),
                                       pack_hash="0" * 64)

        result = postcheck.run(artefact, pack, ast, brief, emit_audit=False)

        assert FindingCode.MISSING_HASH_RECORD in codes(result)


class TestThePrintablePage:
    """The same §6 checks, read through the page's own renderer (2026-09-24)."""

    def _page(self, ast, pack, brief, mutate=None):
        from src.services.rga.render import html as page_renderer
        art = page_renderer.render(ast, pack, brief)
        if mutate:
            art = dataclasses.replace(art, content=mutate(art.content.decode("utf-8")).encode("utf-8"))
        return art

    def test_the_page_passes_the_same_checks_as_the_deck(self, ast, pack, brief):
        result = check(ast, pack, brief, artefact=self._page(ast, pack, brief))
        assert result.passed, [f.detail for f in result.findings]

    def test_an_untraced_figure_on_the_page_blocks(self, ast, pack, brief):
        art = self._page(ast, pack, brief,
                         lambda t: t.replace("</body>", '<p data-chunk>A spare £9,999 appeared.</p></body>'))
        assert "REPORT_UNTRACED_FIGURE" in codes(check(ast, pack, brief, artefact=art))

    def test_a_page_figure_without_its_badge_blocks(self, ast, pack, brief):
        # Every drawing of F0003 loses its badge text: the page shows the figure bare.
        art = self._page(ast, pack, brief, lambda t: t.replace("[F0003] · CORROBORATED", "[F0003]"))
        assert "MISSING_ORIGIN_BADGE" in codes(check(ast, pack, brief, artefact=art))

    def test_an_unknown_renderer_is_refused(self, ast, pack, brief):
        import pytest
        from src.services.rga.render import extract_text_for
        art = dataclasses.replace(self._page(ast, pack, brief), renderer="pdf")
        with pytest.raises(ValueError):
            extract_text_for(art)
