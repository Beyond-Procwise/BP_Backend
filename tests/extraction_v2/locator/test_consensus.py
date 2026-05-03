"""Tests for the consensus voting runner. Voting policy must be:

- ≥ 2 locators agreeing → commit
- Single locator with confidence ≥ 0.95 → commit (capped at 0.9 output)
- Otherwise → abstain (residual)

A buggy locator that raises must not break the runner — its output is
treated as abstain. A buggy locator that returns wrong values must be
outvoted by the others.
"""
import pytest
from dataclasses import dataclass

from services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from services.extraction_v2.locator.consensus import ConsensusResult, run_locators


@dataclass
class FakeLocator:
    name: str
    field: str
    out: LocatorOutput | None
    raises: bool = False

    def locate(self, doc):
        if self.raises:
            raise RuntimeError("simulated bug")
        return self.out


def _out(value, confidence=0.9, why="test"):
    return LocatorOutput(
        value=value,
        confidence=confidence,
        evidence=AnchorRef(raw_text=str(value)),
        why=why,
    )


class TestVotingPolicy:
    def test_two_agreeing_commits(self):
        loc1 = FakeLocator("L1", "invoice_id", _out("INV600820", 0.8))
        loc2 = FakeLocator("L2", "invoice_id", _out("INV600820", 0.7))
        loc3 = FakeLocator("L3", "invoice_id", _out("INV600821", 0.6))  # disagrees
        r = run_locators("invoice_id", [loc1, loc2, loc3], doc=None)
        assert not r.abstained
        assert r.value == "INV600820"
        assert r.confidence > 0.5
        assert "2/3" in r.why or "agreed" in r.why

    def test_single_locator_high_confidence_commits(self):
        loc1 = FakeLocator("L1", "invoice_id", _out("INV600820", 0.96))
        r = run_locators("invoice_id", [loc1], doc=None)
        assert not r.abstained
        assert r.value == "INV600820"
        assert r.confidence <= 0.9   # capped per policy

    def test_single_locator_low_confidence_abstains(self):
        loc1 = FakeLocator("L1", "invoice_id", _out("INV600820", 0.7))
        r = run_locators("invoice_id", [loc1], doc=None)
        assert r.abstained
        assert r.value is None
        # Original candidate must still be visible for review
        assert len(r.candidates) == 1

    def test_no_consensus_abstains(self):
        # Three locators, all different values
        loc1 = FakeLocator("L1", "invoice_id", _out("INV600820", 0.8))
        loc2 = FakeLocator("L2", "invoice_id", _out("INV600821", 0.8))
        loc3 = FakeLocator("L3", "invoice_id", _out("INV600822", 0.8))
        r = run_locators("invoice_id", [loc1, loc2, loc3], doc=None)
        assert r.abstained
        assert r.value is None
        assert len(r.candidates) == 3   # all visible to reviewer

    def test_all_abstain_returns_abstain(self):
        loc1 = FakeLocator("L1", "f", None)
        loc2 = FakeLocator("L2", "f", None)
        r = run_locators("f", [loc1, loc2], doc=None)
        assert r.abstained
        assert r.candidates == ()

    def test_no_locators_registered(self):
        r = run_locators("f", [], doc=None)
        assert r.abstained
        assert "no_locators_registered" in r.why


class TestRobustness:
    def test_locator_exception_is_swallowed(self):
        loc1 = FakeLocator("L1", "f", _out("ok", 0.8))
        loc2 = FakeLocator("L2", "f", _out("ok", 0.8))
        loc3 = FakeLocator("L3", "f", None, raises=True)  # crashes
        r = run_locators("f", [loc1, loc2, loc3], doc=None)
        assert not r.abstained
        assert r.value == "ok"
        # Crashed locator did NOT prevent consensus

    def test_canonical_string_match_is_case_insensitive(self):
        loc1 = FakeLocator("L1", "supplier_id", _out("SUP-AssurityLtd", 0.8))
        loc2 = FakeLocator("L2", "supplier_id", _out("sup-assurityltd", 0.8))
        r = run_locators("supplier_id", [loc1, loc2], doc=None)
        assert not r.abstained
        # Either casing might win — vote-equality is canonical


class TestConsensusResultStructure:
    def test_committed_includes_all_candidates(self):
        loc1 = FakeLocator("L1", "f", _out("a", 0.8))
        loc2 = FakeLocator("L2", "f", _out("a", 0.8))
        loc3 = FakeLocator("L3", "f", _out("b", 0.6))
        r = run_locators("f", [loc1, loc2, loc3], doc=None)
        assert len(r.candidates) == 3   # winners + dissenter

    def test_abstained_result_has_clear_reason(self):
        loc1 = FakeLocator("L1", "f", _out("x", 0.5))
        r = run_locators("f", [loc1], doc=None)
        assert r.abstained
        assert r.why  # non-empty rationale

    def test_field_name_threaded_through(self):
        loc1 = FakeLocator("L1", "po_id", _out("PO123", 0.96))
        r = run_locators("po_id", [loc1], doc=None)
        assert r.field_name == "po_id"
