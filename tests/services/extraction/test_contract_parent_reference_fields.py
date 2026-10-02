"""The references the vocabulary declares must be extractable.

proc.bp_document_type says a call-off points at its framework through
framework_ref. That field did not exist in extraction_schemas/contract.yaml, so
the pointer was declared and never filled.

Offline — the registry compiles from YAML with no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_parent_reference_fields.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.pattern_extractor import run_pattern_extractor  # noqa: E402
from src.services.extraction.pattern_registry import get_registry           # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return get_registry("contract")


class _Parsed:
    """The only thing run_pattern_extractor needs off a parsed document.

    Its signature is run_pattern_extractor(parsed, doc_type) and it reads
    `parsed.full_text` (pattern_extractor.py:101). It does NOT take a registry
    -- it loads one for the doc_type itself.
    """

    def __init__(self, text: str) -> None:
        self.full_text = text


def _value(text, field):
    """The value the L1 pattern layer reads for one field, or None.

    run_pattern_extractor returns list[Candidate] (src/services/extraction/
    types.py:34) with .field, .value, .span, .source, .pattern_name and
    .confidence. Higher-prior patterns are tried first per field, so the first
    candidate for a field is the highest-prior match.
    """
    for c in run_pattern_extractor(_Parsed(text), "contract"):
        if c.field == field:
            return c.value
    return None


def test_both_fields_exist_and_are_optional(registry):
    names = {f.name for f in registry.schema.fields}
    assert {"framework_ref", "parent_agreement_ref"} <= names, sorted(names)
    for field in registry.schema.fields:
        if field.name in ("framework_ref", "parent_agreement_ref"):
            assert field.required is False, (
                f"{field.name} must be optional: most contracts name no parent, and "
                "a required field would block promotion on every standalone agreement"
            )
            assert field.db_column == field.name


def test_a_framework_reference_is_read(registry):
    text = "ORDER FORM\n\nThis Order Form is made under Framework Agreement No: FW-2024-0012.\n"
    assert _value(text, "framework_ref") == "FW-2024-0012"


def test_the_made_under_phrasing_is_read(registry):
    text = "CALL-OFF CONTRACT\n\nmade under Framework Agreement RM6187.\n"
    assert _value(text, "framework_ref") == "RM6187"


def test_a_parent_agreement_reference_is_read(registry):
    text = "STATEMENT OF WORK\n\nThis SOW is issued under Master Agreement MSA-4417.\n"
    assert _value(text, "parent_agreement_ref") == "MSA-4417"


def test_a_placeholder_is_refused_not_stored(registry):
    """'N/A' is not a reference. Storing it would make the maths link on a word.

    Matches the rejection branch parent_contract_id already carries.
    """
    for placeholder in ("N/A", "NA", "TBC", "TBD", "NONE"):
        text = f"ORDER FORM\n\nFramework Agreement No: {placeholder}\n"
        assert _value(text, "framework_ref") is None, placeholder


def test_a_sow_naming_its_msa_does_not_populate_parent_contract_id(registry):
    """Sitting under an agreement is not amending it.

    parent_contract_id anchors on amends/supplements/varies. If a plain "issued
    under" filled it too, the maths could not tell a child from an amendment.
    """
    text = "STATEMENT OF WORK\n\nThis SOW is issued under Master Agreement MSA-4417.\n"
    assert _value(text, "parent_agreement_ref") == "MSA-4417"
    assert _value(text, "parent_contract_id") is None


def test_an_amendment_still_populates_parent_contract_id(registry):
    """The existing behaviour must not move."""
    text = "VARIATION\n\nThis deed amends Contract MSA-4417.\n"
    assert _value(text, "parent_contract_id") == "MSA-4417"


def test_a_contract_naming_no_parent_reads_neither_field(registry):
    text = "MASTER AGREEMENT\n\nbetween Acme Ltd and Beta Ltd, numbered clauses.\n"
    assert _value(text, "framework_ref") is None
    assert _value(text, "parent_agreement_ref") is None


def test_the_colon_form_fills_both_reference_fields_known_overlap(registry):
    """"Master Agreement: X" populates parent_contract_id AND parent_agreement_ref.

    PRE-EXISTING, not introduced here: parent_contract_id's `anchored_parent_contract`
    pattern has always matched "(parent|master|principal) (contract|agreement)" followed
    by a colon, and this task's parent_agreement_ref matches the same text. The two
    forms that carry distinct meaning stay clean -- "issued under Master Agreement X"
    fills only parent_agreement_ref, and "amends Contract X" fills only
    parent_contract_id -- so the distinction the two fields exist for survives.

    Harmless for the hierarchy scoring in the next task, which reads all three
    reference fields as equal candidates and only asks whether the parent's id is
    among them; a duplicated value changes no score.

    Narrowing parent_contract_id's anchor to fix this would change extraction on
    live documents for a case that has behaved this way for months, so it is
    recorded rather than changed.
    """
    text = "STATEMENT OF WORK\n\nMaster Agreement: MSA-4417\n"
    assert _value(text, "parent_agreement_ref") == "MSA-4417"
    assert _value(text, "parent_contract_id") == "MSA-4417"
