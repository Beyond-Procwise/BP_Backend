"""The Style Brief, and the honesty about what it cannot do.

The most important test in this file is the last one. The brief specifies a
scope-resolving settings resolver (D1) that does not exist in this platform, and
the risk is not that scoping is missing — it is that a later reader assumes it
works. So the brief is asserted to *say* it is unscoped, and the registry is
asserted to fail closed rather than invent a value.
"""

from __future__ import annotations

import pytest

from src.services.rga.style import (
    PLATFORM_DEFAULT,
    REGISTRY,
    UNRESOLVED_SCOPES,
    UnknownStyleKey,
    resolve_style_brief,
)


class TestRegistry:
    def test_every_registered_key_has_a_default(self):
        """A key with no default is a report that cannot be rendered."""
        for key, spec in REGISTRY.items():
            assert spec.default is not None, f"{key} has no fail-closed default"

    def test_the_brief_spec_keys_are_all_registered(self):
        for key in (
            "report.style.tone", "report.style.lead_with",
            "report.style.exec_summary.max_bullets", "report.style.section_order",
            "report.style.chart.preferred", "report.style.show_confidence_badges",
            "report.style.show_provenance_footnotes", "report.style.palette",
            "report.style.font.body", "report.style.font.mono",
            "report.style.unassessed_treatment",
        ):
            assert key in REGISTRY

    def test_unassessed_is_never_hidden(self):
        assert REGISTRY["report.style.unassessed_treatment"].default == "surface_as_finding"

    def test_the_design_tokens_are_the_products_own(self):
        palette = REGISTRY["report.style.palette"].default
        assert palette["ocean_dark"] == "#09608B"
        assert palette["teal"] == "#00BEA9"
        assert palette["violet"] == "#6E64FF"
        assert REGISTRY["report.style.font.body"].default == "DM Sans"
        assert REGISTRY["report.style.font.mono"].default == "DM Mono"


class TestResolution:
    def test_an_unknown_key_raises_rather_than_defaulting(self, brief):
        """A typo that resolved to something plausible is how a report comes to
        be styled by a key that governs nothing."""
        with pytest.raises(UnknownStyleKey):
            brief.get("report.style.tone_of_voice")

    def test_every_key_resolves_and_records_where_from(self, brief):
        for key in REGISTRY:
            assert brief.get(key) is not None
            assert brief.chain(key) == PLATFORM_DEFAULT

    def test_a_report_type_may_set_its_own_section_order(self):
        brief = resolve_style_brief("board_paper",
                                    section_order=["cover", "recommendation"])
        assert brief.get("report.style.section_order") == ["cover", "recommendation"]
        assert brief.chain("report.style.section_order") == "report_type:board_paper"

    def test_the_version_changes_when_a_value_does(self):
        a = resolve_style_brief("exec_procurement_summary")
        b = resolve_style_brief("exec_procurement_summary",
                                section_order=["only_this"])
        assert a.version() != b.version()

    def test_the_version_is_stable_for_the_same_values(self):
        assert (resolve_style_brief("x").version()
                == resolve_style_brief("x").version())


def test_the_brief_admits_it_resolved_no_scopes(brief):
    """D1 does not exist. This asserts the brief says so rather than implying a
    scope chain was consulted and came back empty.

    If a real settings resolver is ever wired in, this test should fail — and
    that failure is the reminder to update the disclosure rather than to delete
    the assertion.
    """
    disclosure = brief.disclosure()
    assert PLATFORM_DEFAULT in disclosure
    for scope in ("region", "legal_entity", "business_unit", "category", "site", "user"):
        assert scope in disclosure
    assert brief.unresolved_scopes == UNRESOLVED_SCOPES
    assert brief.resolver == "rga.style.defaults_only"
