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


class TestUnmeasuredReasonsCarryNoNumbers:
    """An unmeasured reason is shown to the composer, which may quote it, and a
    narrative may not contain a literal number. Live 2026-09-24, run
    FP-1f076acf7aa2: the reason "...which is not £0" was quoted into prose and
    blocked the whole report on both attempts. A reason with a digit in it is a
    report that cannot release, so the builder refuses it at the point of the
    mistake."""

    def test_a_reason_with_a_number_is_refused(self):
        fb = FactBuilder(pack_id="FP-test", scope={}, as_of="2026-03-31")
        with pytest.raises(ValueError, match="number"):
            fb.unmeasured(label="Identified value", derivation="d",
                          reason="no value to state, which is not £0")

    def test_a_reason_in_words_is_recorded(self):
        fb = FactBuilder(pack_id="FP-test", scope={}, as_of="2026-03-31")
        fb.unmeasured(label="Identified value", derivation="d",
                      reason="no value to state, which is not the same as zero")
        assert fb.findings[0].code == FindingCode.MEASURE_UNAVAILABLE

    @pytest.mark.parametrize("rates_unavailable", [True, False])
    def test_every_exec_summary_gap_is_stated_in_words(self, monkeypatch,
                                                       rates_unavailable):
        """Drive the real builder down every unmeasured branch: an empty
        period, and spend that is either unconvertible or rate-less."""
        from src.services.rga.builders import exec_procurement_summary as ex

        rows = {
            ex._DEAL_SHAPE: [(0, 0, 0, 0, None, 0)],
            ex._DEAL_AMOUNTS: [(Decimal("10"), "XXX")],
            ex._OPPORTUNITIES: [(0, None)],
            ex._REALISED: [(Decimal("0"), 0, 0)],
            ex._SAVED: [],
        }
        monkeypatch.setattr(ex, "_fetch", lambda sql, params: rows[sql])
        monkeypatch.setattr(
            ex, "_rates",
            lambda: ({}, None, True) if rates_unavailable
            else ({"USD": Decimal("1"), "GBP": Decimal("0.8")}, None, False))

        fb = FactBuilder(pack_id="FP-test", scope={
            "period_start": "2026-01-01", "period_end": "2026-03-31",
            "currency": "GBP"}, as_of="2026-03-31")
        ex.build(fb)  # raises if any reason carries a number

        gaps = [f for f in fb.findings if f.code == FindingCode.MEASURE_UNAVAILABLE]
        assert len(gaps) == 6  # spend, match rate, cycle, identified, realised, saved

    def test_unpriced_only_rows_do_not_fabricate_a_measured_zero(self, monkeypatch):
        """An unconvertible-currency ledger row (amount_gbp NULL) must not inflate
        realised_n/saved_n -- the counts that gate CORROBORATED vs unmeasured. A period
        holding only unpriced rows must go unmeasured, never report a fabricated £0.00
        at CORROBORATED confidence."""
        from src.services.rga.builders import exec_procurement_summary as ex

        rows = {
            ex._DEAL_SHAPE: [(0, 0, 0, 0, None, 0)],
            ex._DEAL_AMOUNTS: [],
            ex._OPPORTUNITIES: [(0, None)],
            ex._REALISED: [(Decimal("0"), 0, 3)],          # 3 unpriced, 0 priced
            ex._SAVED: [("avoided", Decimal("0"), 0, 2)],  # 2 unpriced, 0 priced
        }
        monkeypatch.setattr(ex, "_fetch", lambda sql, params: rows[sql])
        monkeypatch.setattr(ex, "_rates", lambda: ({}, None, True))

        fb = FactBuilder(pack_id="FP-test", scope={
            "period_start": "2026-01-01", "period_end": "2026-03-31",
            "currency": "GBP"}, as_of="2026-03-31")
        ex.build(fb)  # raises if any reason carries a number

        realised = next(f for f in fb.facts if f.label == "Realised savings (GBP)")
        saved = next(f for f in fb.facts if f.label == "Saved (GBP)")
        assert realised.value is None and realised.confidence is Confidence.UNASSESSED
        assert saved.value is None and saved.confidence is Confidence.UNASSESSED

    def test_a_mix_of_priced_and_unpriced_rows_sums_only_the_priced(self, monkeypatch):
        """A mix of priced and unpriced rows in the same period sums only what could be
        converted to GBP, and discloses how many further rows it excludes (in the
        derivation, since an unmeasured reason may carry no digit)."""
        from src.services.rga.builders import exec_procurement_summary as ex

        rows = {
            ex._DEAL_SHAPE: [(0, 0, 0, 0, None, 0)],
            ex._DEAL_AMOUNTS: [],
            ex._OPPORTUNITIES: [(0, None)],
            ex._REALISED: [(Decimal("500"), 1, 2)],        # 1 priced (£500), 2 unpriced
            ex._SAVED: [("avoided", Decimal("300"), 1, 0),
                       ("recovered", Decimal("0"), 0, 4)],  # 1 priced (£300), 4 unpriced
        }
        monkeypatch.setattr(ex, "_fetch", lambda sql, params: rows[sql])
        monkeypatch.setattr(ex, "_rates", lambda: ({}, None, True))

        fb = FactBuilder(pack_id="FP-test", scope={
            "period_start": "2026-01-01", "period_end": "2026-03-31",
            "currency": "GBP"}, as_of="2026-03-31")
        ex.build(fb)  # raises if any reason carries a number

        realised = next(f for f in fb.facts if f.label == "Realised savings (GBP)")
        saved = next(f for f in fb.facts if f.label == "Saved (GBP)")
        assert realised.value == Decimal("500")
        assert realised.confidence is Confidence.CORROBORATED
        assert "2" in realised.derivation and "could not be converted" in realised.derivation
        assert saved.value == Decimal("300")
        assert saved.confidence is Confidence.CORROBORATED
        assert "4" in saved.derivation and "could not be converted" in saved.derivation


def test_a_pack_id_can_be_known_before_the_pack_is_built(stub_type):
    """So a job can carry its run id from the start, and a run interrupted before
    its pack was built still has a trace id to close its trail under."""
    pack = build_fact_pack(stub_type, scope=SCOPE, as_of="2026-03-31", emit_audit=False)
    assert factpack.pack_id_for(stub_type, SCOPE, "2026-03-31") == pack.pack_id
