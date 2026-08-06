"""L1 parity for contract.yaml against synthetic prose fixtures.

NOT REAL-WORLD VALIDATION. proc.bp_contracts has 0 rows in both databases and
the repo contains no contract PDFs, so these fixtures are hand-written prose in
the shapes contracts normally use. They prove the regexes do what they claim on
text; they do not prove the regexes survive real contract layout. Re-run this
against a real corpus when blocker B1 is resolved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.pattern_registry import PatternRegistry, clear_cache  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "contracts"


def setup_function():
    clear_cache()


def _load_fixture(name: str) -> tuple[str, dict]:
    text = (FIXTURES / f"{name}.txt").read_text()
    expected = json.loads((FIXTURES / f"{name}.expected.json").read_text())
    return text, expected


def _best_hit(text: str, field: str) -> str | None:
    """Highest-prior pattern that matches, mirroring how L1 orders candidates.

    patterns_for() returns them already sorted by prior_confidence descending
    (PatternRegistry._compile), so the first match is the highest-prior one.
    """
    reg = PatternRegistry("contract")
    for cp in reg.patterns_for(field):
        for m in cp.anchor_re.finditer(text):
            window = text[m.end():m.end() + cp.max_span_after_anchor_chars]
            vm = cp.value_re.search(window)
            if vm:
                return vm.group(1) if vm.lastindex else vm.group(0)
    return None


ALLOCATION_FIELDS = ["contract_id", "parent_contract_id", "cost_centre_id", "is_amendment"]


@pytest.mark.parametrize("field", ALLOCATION_FIELDS)
def test_allocation_fields_extract_from_fixture(field):
    text, expected = _load_fixture("msa_with_amendment")
    assert _best_hit(text, field) == expected[field]


def test_parent_contract_is_not_confused_with_contract_id():
    """Both are MSA-prefixed identifiers on adjacent lines. Getting these the
    wrong way round would invert every amendment chain."""
    text, expected = _load_fixture("msa_with_amendment")
    assert _best_hit(text, "contract_id") != _best_hit(text, "parent_contract_id")


TERM_FIELDS = [
    "amendment_ref",
    "document_version",
    "term_months",
    "billing_frequency",
    "escalator_pct",
    "escalator_basis",
    "escalator_cap_pct",
]


@pytest.mark.parametrize("field", TERM_FIELDS)
def test_commercial_term_fields_extract_from_fixture(field):
    text, expected = _load_fixture("msa_with_amendment")
    assert _best_hit(text, field) == expected[field]


@pytest.mark.parametrize("field", ["term_months", "escalator_pct", "escalator_cap_pct"])
def test_decimal_fields_capture_a_bindable_number(field):
    """parse_amount returns None for '3.5%' and '36 months', and the decimal
    fallback float() then raises — the value would be discarded as a
    type_bind_error. The capture group must yield the bare number."""
    from src.services.extraction_v2.parsers.amounts import parse_amount

    text, _ = _load_fixture("msa_with_amendment")
    hit = _best_hit(text, field)
    assert hit is not None
    assert parse_amount(hit) is not None, f"{field} captured {hit!r}, which will not bind to decimal"


def test_escalator_and_cap_are_not_the_same_number():
    """3.5% escalation capped at 5% — reading the cap as the rate would
    understate every uplift check by 43%."""
    text, _ = _load_fixture("msa_with_amendment")
    assert _best_hit(text, "escalator_pct") != _best_hit(text, "escalator_cap_pct")


def test_term_stated_in_years_is_not_silently_converted():
    """A three-year term must yield NULL, not 36. Conversion is arithmetic, and
    the extractor does not do arithmetic."""
    text = "Initial Term: three (3) years from the Commencement Date."
    assert _best_hit(text, "term_months") is None
