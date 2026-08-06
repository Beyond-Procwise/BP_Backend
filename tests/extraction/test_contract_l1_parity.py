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
