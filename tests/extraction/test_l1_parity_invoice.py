"""L1 parity test: pattern_extractor against the 7 invoice fixtures.

For each fixture, parse → run L1 → assert the highest-confidence candidate
for the three Task-5 fields matches the expected.json ground truth.

These tests drive YAML pattern iteration: a failing assertion means we need
to add or refine a pattern in extraction_schemas/invoice.yaml.

NOTE: supplier_name is intentionally NOT in COMPARE_FIELDS yet — L1 alone
cannot find it for docs without a 'Supplier:' label (relies on L2 NER).
That's added back when L2 ner_validator lands.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.services.extraction.parser import parse  # noqa: E402
from src.services.extraction.pattern_extractor import run_pattern_extractor  # noqa: E402
from src.services.extraction.pattern_registry import clear_cache  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "extraction_v3" / "fixtures" / "invoices"
# Only include fixtures that have a matching expected.json (INV-007 has none).
FIXTURES = sorted(
    p for p in FIXTURES_DIR.glob("INV-*.pdf")
    if p.with_suffix(".expected.json").exists()
)
COMPARE_FIELDS = ["invoice_id", "invoice_amount"]


def setup_function():
    clear_cache()


def _money(s) -> float:
    """Parse a money-shaped string '£1,250.00' → 1250.00."""
    if s is None:
        return float("nan")
    if isinstance(s, (int, float)):
        return float(s)
    cleaned = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def _expected(pdf_path: Path) -> dict:
    return json.loads(pdf_path.with_suffix(".expected.json").read_text())


def _best_by_field(cands) -> dict:
    """Highest-confidence candidate per field."""
    by = {}
    for c in cands:
        if c.field not in by or c.confidence > by[c.field].confidence:
            by[c.field] = c
    return by


@pytest.mark.parametrize("pdf_path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_invoice_id_l1_matches(pdf_path):
    expected = _expected(pdf_path)
    parsed = parse(str(pdf_path))
    cands = run_pattern_extractor(parsed, "invoice")
    best = _best_by_field(cands)
    exp_id = expected["header"].get("invoice_id")
    if not exp_id:
        pytest.skip("no expected invoice_id")
    got = best.get("invoice_id")
    if got is None:
        pytest.fail(f"[{pdf_path.name}] L1 produced no invoice_id (expected {exp_id!r})")
    assert got.value.strip() == str(exp_id).strip(), (
        f"[{pdf_path.name}] invoice_id mismatch: got {got.value!r} expected {exp_id!r}"
    )
    assert got.span.text in parsed.full_text, "grounding broken"


@pytest.mark.parametrize("pdf_path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_invoice_amount_l1_matches(pdf_path):
    expected = _expected(pdf_path)
    parsed = parse(str(pdf_path))
    cands = run_pattern_extractor(parsed, "invoice")
    best = _best_by_field(cands)
    exp_amt = expected["header"].get("invoice_amount")
    if exp_amt is None:
        pytest.skip("no expected invoice_amount")
    got = best.get("invoice_amount")
    if got is None:
        pytest.fail(f"[{pdf_path.name}] L1 produced no invoice_amount (expected {exp_amt})")
    got_v = _money(got.value)
    exp_v = float(exp_amt)
    assert abs(got_v - exp_v) < 0.01, (
        f"[{pdf_path.name}] invoice_amount mismatch: got {got_v} expected {exp_v}"
    )
