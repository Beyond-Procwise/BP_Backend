"""contract_id must be declared on all three transaction schemas, and its L1
patterns must fire on the labelled forms that appear on real POs and invoices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.pattern_registry import PatternRegistry, clear_cache  # noqa: E402
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema  # noqa: E402

DOC_TYPES = ["invoice", "purchase_order", "quote"]


def setup_function():
    clear_cache()


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_schema_declares_contract_id(doc_type):
    schema = load_doc_schema(doc_type)
    by_name = {f.name: f for f in schema.fields}
    assert "contract_id" in by_name, f"{doc_type} must declare contract_id"
    f = by_name["contract_id"]
    assert f.db_column == "contract_id"
    assert f.type == "string"
    assert f.required is False
    assert f.patterns, "contract_id needs L1 patterns; the VLM alone has never populated it"


def _hits(doc_type: str, field: str, text: str) -> list[str]:
    """Every value the field's L1 patterns extract from `text`."""
    reg = PatternRegistry(doc_type)
    out: list[str] = []
    for cp in reg.patterns_for(field):  # sorted by prior_confidence desc
        for m in cp.anchor_re.finditer(text):
            window = text[m.end():m.end() + cp.max_span_after_anchor_chars]
            vm = cp.value_re.search(window)
            if vm:
                out.append(vm.group(1) if vm.lastindex else vm.group(0))
    return out


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("text,expected", [
    ("Contract No: MSA-2024-0087", "MSA-2024-0087"),
    ("Contract Number: CTR/2025/119", "CTR/2025/119"),
    ("Agreement No. AGR-4471", "AGR-4471"),
    ("Contract Reference: FRM-2023-88", "FRM-2023-88"),
    ("Issued under Master Agreement MSA-9921", "MSA-9921"),
])
def test_contract_id_patterns_extract_the_identifier(doc_type, text, expected):
    got = _hits(doc_type, "contract_id", text)
    assert expected in got, f"{doc_type}: no pattern extracted {expected!r} from {text!r}"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_contract_id_patterns_do_not_fire_on_boilerplate(doc_type):
    """A false contract link is worse than none — it would attach a transaction
    to an agreement that does not govern it."""
    noise = "This contract is subject to our standard terms and conditions."
    got = _hits(doc_type, "contract_id", noise)
    assert not got, f"{doc_type}: patterns fired on boilerplate: {got}"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("text,expected", [
    ("Issued under Master Agreement MSA-9921.", "MSA-9921"),
    ("Contract No: MSA-2024-0087.", "MSA-2024-0087"),
    ("Contract Reference: CTR/2025/119,", "CTR/2025/119"),
])
def test_trailing_punctuation_is_not_part_of_the_identifier(doc_type, text, expected):
    """'MSA-9921.' != 'MSA-9921' silently breaks the join to bp_contracts —
    the transaction is left orphaned with no error anywhere."""
    got = _hits(doc_type, "contract_id", text)
    assert got, f"{doc_type}: nothing extracted from {text!r}"
    assert all(g == expected for g in got), f"{doc_type}: got {got}, expected {expected!r}"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("placeholder", ["N/A", "NA", "TBC", "TBD", "NONE"])
def test_placeholder_tokens_are_not_captured_as_a_contract_link(doc_type, placeholder):
    got = _hits(doc_type, "contract_id", f"Contract No: {placeholder}")
    assert not got, f"{doc_type}: captured placeholder {got}"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_prose_label_lookalike_does_not_fire(doc_type):
    """'Contract Notes:' contains 'No'. An optional connector let the label
    matcher walk into the middle of an unrelated word and take the next token."""
    got = _hits(doc_type, "contract_id", "Contract Notes: SEE ATTACHED SCHEDULE")
    assert not got, f"{doc_type}: fired on a label lookalike: {got}"
