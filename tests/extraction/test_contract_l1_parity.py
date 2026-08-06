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
    wrong way round would invert every amendment chain.

    Asserting only that the two differ passes on any fixture where the failure
    cannot occur — which is exactly how the parent-capture defect survived. Both
    exact values are asserted here, and the adversarial orderings live below.
    """
    text, _ = _load_fixture("msa_with_amendment")
    assert _best_hit(text, "contract_id") == "MSA-2024-0087"
    assert _best_hit(text, "parent_contract_id") == "MSA-2024-0001"


# --- adversarial: contract_id must never resolve to the PARENT contract ------
#
# contract_id is the upsert key for proc.bp_contracts (promotion.py:56-60,
# 857-860: ON CONFLICT (contract_id) DO UPDATE SET <every other column>). An
# amendment that resolves to its master's identifier does not error — it
# silently overwrites the master row's dates, value and escalator terms.


def test_amendment_naming_only_its_master_yields_no_contract_id():
    """The only identifier in the text belongs to the PARENT. contract_id must
    stay NULL rather than adopt it."""
    text = "This Amendment is made under Master Agreement No: MSA-2024-0001."
    assert _best_hit(text, "contract_id") is None
    assert _best_hit(text, "parent_contract_id") == "MSA-2024-0001"


def test_parent_stated_before_child_does_not_win_on_document_order():
    """persistence.py:93 keeps the highest-confidence candidate with a strict
    '>', so on equal priors the FIRST occurrence wins. A parent line above the
    child line must still not be read as contract_id."""
    text = "Parent Contract Number: MSA-2024-0001\nContract Number: MSA-2024-0087"
    assert _best_hit(text, "contract_id") == "MSA-2024-0087"
    assert _best_hit(text, "parent_contract_id") == "MSA-2024-0001"


@pytest.mark.parametrize("qualifier", ["Parent", "Master", "Principal"])
def test_qualified_contract_label_is_not_read_as_contract_id(qualifier):
    text = f"{qualifier} Contract Number: MSA-2024-0001"
    assert _best_hit(text, "contract_id") is None


@pytest.mark.parametrize("qualifier", ["Parent", "Master", "Principal"])
def test_qualified_agreement_label_is_not_read_as_contract_id(qualifier):
    text = f"{qualifier} Agreement No: MSA-2024-0001"
    assert _best_hit(text, "contract_id") is None


@pytest.mark.parametrize("verb", ["amends", "supplements", "varies"])
def test_amending_verb_target_is_not_read_as_contract_id(verb):
    text = f"This document {verb} Contract No: MSA-2024-0001."
    assert _best_hit(text, "contract_id") is None
    assert _best_hit(text, "parent_contract_id") == "MSA-2024-0001"


def test_unqualified_labels_still_produce_contract_id():
    """The lookbehind guard must not disarm the field it protects."""
    assert _best_hit("Contract Number: CTR-2025-0042", "contract_id") == "CTR-2025-0042"
    assert _best_hit("Contract No: CTR/2025/119", "contract_id") == "CTR/2025/119"
    assert _best_hit("Agreement No: AGR-4471", "contract_id") == "AGR-4471"
    assert _best_hit("Contract Reference: FRM-2023-88", "contract_id") == "FRM-2023-88"


def test_parent_contract_patterns_still_capture_their_own_forms():
    assert _best_hit("Parent Contract: MSA-2024-0001", "parent_contract_id") == "MSA-2024-0001"
    assert _best_hit("Master Agreement No: MSA-2024-0001", "parent_contract_id") == "MSA-2024-0001"


# --- adversarial: the identifier value must be a real identifier ------------


@pytest.mark.parametrize("field", ["contract_id", "parent_contract_id"])
@pytest.mark.parametrize("placeholder", ["N/A", "NA", "TBC", "TBD", "NONE", "SEE ATTACHED SCHEDULE"])
def test_placeholder_tokens_are_not_captured_as_an_identifier(field, placeholder):
    """'N/A' as contract_id becomes the bp_contracts primary key and collides
    with every other 'N/A' contract under the same upsert."""
    label = "Contract No" if field == "contract_id" else "Parent Contract"
    assert _best_hit(f"{label}: {placeholder}", field) is None


def test_prose_label_lookalike_does_not_fire():
    """'Contract Notes:' contains 'No'. Only a mandatory connector stops the
    label matcher walking into the middle of an unrelated word."""
    assert _best_hit("Contract Notes: SEE ATTACHED SCHEDULE", "contract_id") is None


@pytest.mark.parametrize("field,text,expected", [
    ("contract_id", "Contract No: MSA-9921.", "MSA-9921"),
    ("contract_id", "Contract No: MSA-9921,", "MSA-9921"),
    ("parent_contract_id", "Parent Contract: MSA-9921.", "MSA-9921"),
    ("parent_contract_id", "Master Agreement No: CTR/2025/119.", "CTR/2025/119"),
])
def test_trailing_punctuation_is_not_part_of_the_identifier(field, text, expected):
    """'MSA-9921.' != 'MSA-9921' silently breaks the later join to
    bp_contracts — no error, just an orphan."""
    assert _best_hit(text, field) == expected


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


def test_money_labels_are_not_bound_to_a_percentage_field():
    """'Not to Exceed' is total_contract_value's label. Sharing it with
    escalator_cap_pct numeric(9,4) would overflow the column the moment the
    label-driven extractors are re-enabled."""
    from src.services.extraction_v3.yaml_schema.loader import load_doc_schema

    by_name = {f.name: f for f in load_doc_schema("contract").fields}
    assert "Not to Exceed" not in by_name["escalator_cap_pct"].canonical_labels
    assert "Not to Exceed" in by_name["total_contract_value"].canonical_labels


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


# --- adversarial: the cap is not the rate -----------------------------------


def test_a_cap_alone_does_not_become_the_escalation_rate():
    """'shall not exceed 4%' states a ceiling, not a rate. Recording 4 as the
    escalator invents a contractual uplift the document never granted."""
    text = "Any such increase shall not exceed 4% in any year."
    assert _best_hit(text, "escalator_pct") is None
    assert _best_hit(text, "escalator_cap_pct") == "4"


def test_cap_stated_before_the_rate_does_not_win_on_document_order():
    """The existing fixture happens to state the rate first. Reversed, the cap
    was captured as the rate — a 43% understatement of every uplift check."""
    text = (
        "Any increase shall be capped at 5%.\n"
        "Annual price escalation: 3.5% per annum, indexed to CPI."
    )
    assert _best_hit(text, "escalator_pct") == "3.5"
    assert _best_hit(text, "escalator_cap_pct") == "5"


def test_an_indexed_increase_with_a_cap_yields_no_fixed_rate():
    """The rate here is CPI, not a number. The only number in the sentence is
    the cap."""
    text = "Charges shall increase by CPI, capped at 5%."
    assert _best_hit(text, "escalator_pct") is None
    assert _best_hit(text, "escalator_cap_pct") == "5"


def test_legitimate_escalation_phrasings_still_extract():
    assert _best_hit("Annual price escalation: 3.5% per annum.", "escalator_pct") == "3.5"
    assert _best_hit("Price escalation of 4% each year.", "escalator_pct") == "4"
    assert _best_hit("Charges shall increase by 2.5% annually.", "escalator_pct") == "2.5"
    assert _best_hit("Annual increase: 3%", "escalator_pct") == "3"


# --- adversarial: a renewal term is not the contract term --------------------


def test_renewal_term_is_not_recorded_as_the_contract_term():
    """A 60-month initial term with 12-month renewals must not record as a
    12-month contract."""
    text = "Renewal Term: 12 months following the Initial Term."
    assert _best_hit(text, "term_months") is None


def test_extension_term_is_not_recorded_as_the_contract_term():
    text = "Extension Term: 24 months at the Buyer's option."
    assert _best_hit(text, "term_months") is None


def test_initial_term_wins_over_a_renewal_term_stated_first():
    text = "Renewal Term: 12 months.\nInitial Term: 60 months."
    assert _best_hit(text, "term_months") == "60"


def test_legitimate_term_phrasings_still_extract():
    assert _best_hit("Initial Term: 36 months", "term_months") == "36"
    assert _best_hit("Minimum Term of 18 months", "term_months") == "18"
    assert _best_hit("for a period of 24 months", "term_months") == "24"
