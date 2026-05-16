"""Tests for YAML schema patterns + confidence_threshold extensions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction_v3.yaml_schema.loader import load_doc_schema, Pattern


def test_invoice_schema_loads():
    schema = load_doc_schema("invoice")
    assert schema.doc_type == "invoice"
    assert schema.fields, "no fields loaded"


def test_invoice_id_has_patterns():
    schema = load_doc_schema("invoice")
    by_name = {f.name: f for f in schema.fields}
    f = by_name["invoice_id"]
    assert f.patterns, "invoice_id must have at least one pattern"
    for p in f.patterns:
        assert isinstance(p, Pattern)
        assert p.anchor and p.value
        assert 0.0 <= p.prior_confidence <= 1.0
    # ordered by intent: anchored before bareword
    assert f.patterns[0].prior_confidence >= f.patterns[-1].prior_confidence
    assert f.confidence_threshold == 0.70


def test_supplier_name_has_patterns_and_ner():
    schema = load_doc_schema("invoice")
    by_name = {f.name: f for f in schema.fields}
    f = by_name["supplier_name"]
    assert f.patterns, "supplier_name must have patterns"
    assert f.judge.ner_type_check == "ORG"
    assert f.confidence_threshold == 0.65


def test_invoice_amount_has_patterns():
    schema = load_doc_schema("invoice")
    by_name = {f.name: f for f in schema.fields}
    f = by_name["invoice_amount"]
    assert f.patterns
    # check a real-looking number can match value regex
    import re
    val_re = re.compile(f.patterns[0].value)
    m = val_re.search("1,250.00")
    assert m, "value regex must accept comma-thousand decimal"


def test_fields_without_patterns_default_to_empty_list():
    """Most fields don't have patterns yet; loader must default to [], not error."""
    schema = load_doc_schema("invoice")
    by_name = {f.name: f for f in schema.fields}
    f = by_name["invoice_paid_date"]
    assert f.patterns == []
    assert f.confidence_threshold == 0.70  # default
