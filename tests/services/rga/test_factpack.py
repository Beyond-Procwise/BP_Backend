"""Building a pack: the registry, the provenance rule, and determinism.

No database. The exec-summary builder's real queries are covered by
``test_exec_summary_live.py``; what is covered here is the machinery every
builder runs through, which must be testable without one.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.services.analytics.models import Confidence
from src.services.rga import factpack
from src.services.rga.factpack import (
    FactBuilder,
    NoBuilderRegistered,
    build_fact_pack,
    register,
    registered_types,
)
from src.services.rga.models import FindingCode, FormatHint, Origin


@pytest.fixture
def stub_type():
    """Register a builder for the duration of one test, then remove it.

    The registry is module-level state; a test that left an entry behind would
    change what ``registered_types()`` reports for every test after it.
    """
    name = "_test_report_type"

    @register(name)
    def _build(fb: FactBuilder) -> None:
        fb.add(label="Deals in period", value=Decimal(376),
               derivation="stub.deals", confidence=Confidence.ASSERTED,
               format_hint=FormatHint.INT, unit="deals")
        fb.unmeasured(label="Realised savings", derivation="stub.realised",
                      reason="nothing in the corpus records one")

    yield name
    factpack._BUILDERS.pop(name, None)


SCOPE = {"period_label": "2026 Q1", "period_start": "2026-01-01",
         "period_end": "2026-03-31", "currency": "GBP"}


class TestRegistry:
    def test_an_unregistered_report_type_raises(self):
        """Not an empty pack. An empty pack renders as 'nothing to report'."""
        with pytest.raises(NoBuilderRegistered):
            build_fact_pack("no_such_report", scope=SCOPE, as_of="2026-03-31",
                            emit_audit=False)

    def test_the_seed_report_type_is_registered_by_importing_the_package(self):
        import src.services.rga  # noqa: F401

        assert "exec_procurement_summary" in registered_types()


class TestProvenanceRule:
    def test_a_fact_without_provenance_is_dropped_and_raises_a_finding(self):
        """DoD2. Two mechanisms: the type refuses it, and the builder records
        why the figure is missing instead of dying."""
        fb = FactBuilder(pack_id="FP-test", scope={}, as_of="2026-03-31")
        result = fb.add(label="unevidenced", value=Decimal(1),
                        derivation="d", confidence=Confidence.ASSERTED,
                        provenance_id="   ")

        assert result is None
        assert fb.facts == []
        assert [f.code for f in fb.findings] == [FindingCode.FACT_WITHOUT_PROVENANCE]
        assert fb.findings[0].blocks_release is True

    def test_a_good_fact_gets_a_provenance_pointer_that_resolves(self):
        fb = FactBuilder(pack_id="FP-test", scope={}, as_of="2026-03-31")
        entry = fb.add(label="Deals", value=Decimal(1), derivation="stub.deals",
                       confidence=Confidence.ASSERTED, format_hint=FormatHint.INT)

        # The pointer names the audit row this build writes (trace_id = pack_id)
        # and the fact within it.
        assert entry.provenance_id == "bp_agent_actions:FP-test#F0001"
        assert entry.derivation == "stub.deals"


class TestUnmeasured:
    def test_an_unmeasurable_figure_is_recorded_not_omitted(self, stub_type):
        """A report that silently drops a measure reads as though it were not
        relevant."""
        pack = build_fact_pack(stub_type, scope=SCOPE, as_of="2026-03-31",
                               emit_audit=False)
        realised = [f for f in pack.facts if f.label == "Realised savings"]

        assert len(realised) == 1
        assert realised[0].value is None
        assert realised[0].confidence is Confidence.UNASSESSED
        assert realised[0].display == "—"

    def test_it_raises_a_visible_but_non_blocking_finding(self, stub_type):
        pack = build_fact_pack(stub_type, scope=SCOPE, as_of="2026-03-31",
                               emit_audit=False)
        unavailable = [f for f in pack.findings
                       if f.code is FindingCode.MEASURE_UNAVAILABLE]

        assert len(unavailable) == 1
        # Not being able to measure something does not block the report; it is
        # disclosed on it.
        assert unavailable[0].blocks_release is False
        assert pack.blocking_findings() == []


class TestDeterminism:
    def test_same_scope_and_as_of_produce_the_same_hash(self, stub_type):
        """DoD3."""
        first = build_fact_pack(stub_type, scope=SCOPE, as_of="2026-03-31",
                                emit_audit=False)
        second = build_fact_pack(stub_type, scope=SCOPE, as_of="2026-03-31",
                                 emit_audit=False)

        assert first.hash == second.hash
        assert first.pack_id == second.pack_id

    def test_a_different_period_is_a_different_pack(self, stub_type):
        other = dict(SCOPE, period_start="2025-01-01", period_label="2025 Q1")
        a = build_fact_pack(stub_type, scope=SCOPE, as_of="2026-03-31",
                            emit_audit=False)
        b = build_fact_pack(stub_type, scope=other, as_of="2026-03-31",
                            emit_audit=False)

        assert a.pack_id != b.pack_id
        assert a.hash != b.hash


class TestStorage:
    def test_a_pack_round_trips_through_storage(self, pack):
        from src.services.rga.models import FactPack

        reloaded = FactPack.from_stored(pack.stored())

        assert reloaded.hash == pack.hash
        assert [f.display for f in reloaded.facts] == [f.display for f in pack.facts]

    def test_an_altered_stored_pack_is_refused(self, pack):
        """Reproducible and wrong is worse than not reproducible."""
        from src.services.rga.models import FactPack

        payload = pack.stored()
        payload["pack"]["facts"][0]["value"] = "9999999.00"

        with pytest.raises(ValueError, match="altered since it was written"):
            FactPack.from_stored(payload)
