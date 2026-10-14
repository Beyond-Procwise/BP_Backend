"""Invariant results must reach the discrepancy queue.

Twelve validators are registered and none of them has ever produced a finding.
Live database, 2026-08-11: `SELECT count(*) FROM proc.bp_extraction_discrepancy
WHERE issue_type = 'invariant_failed'` returns 0, against 821 rows of every
other type.

Three defects, and they have to be fixed together:

  A  dispatch.py passes ``line_items=[]`` to run_invariants although the list is
     in scope and populated, so line_arithmetic, subtotal_closure and
     line_sum_closure all return not_applicable on every document.

  B  run_invariants returns lowercase severities ("warning") and dispatch
     compares against uppercase ("WARNING"), so no branch ever matches.

  C  InvariantResult drops ``passed``, and ValidatorResult.ok() inherits the
     dataclass default severity=WARNING. So a PASSING check reports "warning" —
     meaning a fix for B alone would file every satisfied invariant as a
     failure. The declared contract in the InvariantResult docstring is
     "PASS" | "INFO" | "WARNING" | "CRITICAL" | "NA"; the implementation never
     honoured it.

These tests assert the contract, so all three have to hold at once.
"""
from __future__ import annotations

import pytest

from src.services.extraction_v3.binding.invariants_runner import run_invariants
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema


@pytest.fixture(scope="module")
def schema():
    return load_doc_schema("invoice")


# A document whose arithmetic is internally consistent.
_CLEAN_HEADER = {
    "invoice_amount": 1000.00, "tax_amount": 200.00, "tax_percent": 20.0,
    "invoice_total_incl_tax": 1200.00, "currency": "GBP",
    "invoice_date": "2026-01-15",
}
_CLEAN_LINES = [
    {"item_description": "Widget", "quantity": 2, "unit_price": 250.00,
     "line_amount": 500.00},
    {"item_description": "Gadget", "quantity": 5, "unit_price": 100.00,
     "line_amount": 500.00},
]


def _by_name(results):
    return {r.name: r for r in results}


# --------------------------------------------------------------------------
# C — a passing invariant must not look like a failure
# --------------------------------------------------------------------------

def test_a_satisfied_invariant_reports_pass_not_warning(schema):
    """ValidatorResult.ok() inherits severity=WARNING from the dataclass
    default. Passing that through means every satisfied check files a
    discrepancy the moment the severity comparison is fixed."""
    results = _by_name(run_invariants(_CLEAN_HEADER, _CLEAN_LINES, schema))

    tax = results["tax_closure"]
    assert tax.severity == "PASS", (
        f"tax_closure holds (1000 x 20% = 200) but reports {tax.severity!r}. "
        f"A passing check must not be indistinguishable from a failing one."
    )


def test_a_not_applicable_invariant_is_distinct_from_a_pass(schema):
    """Abstaining is not the same as passing, and neither is a failure."""
    results = _by_name(run_invariants(_CLEAN_HEADER, [], schema))
    assert results["line_arithmetic"].severity == "NA"


# --------------------------------------------------------------------------
# B — a failing invariant must be reported at a severity dispatch recognises
# --------------------------------------------------------------------------

def test_a_failing_invariant_reports_an_uppercase_severity(schema):
    """dispatch.py compares against "CRITICAL" / "WARNING". Lowercase values
    match neither, which is why the bridge has never fired."""
    broken = dict(_CLEAN_HEADER, invoice_total_incl_tax=9999.99)
    results = _by_name(run_invariants(broken, _CLEAN_LINES, schema))

    grand = results["grand_total_closure"]
    assert grand.severity in ("WARNING", "CRITICAL"), (
        f"grand_total_closure failed (1000 + 200 != 9999.99) but reports "
        f"{grand.severity!r}; dispatch.py matches on uppercase"
    )
    assert grand.message, "a failure must carry its reason"


@pytest.mark.parametrize("severity", ["PASS", "NA", "INFO", "WARNING", "CRITICAL"])
def test_every_severity_is_one_of_the_declared_values(severity, schema):
    """The InvariantResult docstring declares the vocabulary. Anything outside
    it is a value some consumer will silently not match."""
    allowed = {"PASS", "NA", "INFO", "WARNING", "CRITICAL"}
    for header, lines in ((_CLEAN_HEADER, _CLEAN_LINES),
                          (dict(_CLEAN_HEADER, tax_amount=999.0), _CLEAN_LINES),
                          (_CLEAN_HEADER, [])):
        for r in run_invariants(header, lines, schema):
            assert r.severity in allowed, f"{r.name} reported {r.severity!r}"


# --------------------------------------------------------------------------
# A — the line invariants must actually see the lines
# --------------------------------------------------------------------------

def test_line_arithmetic_catches_a_line_whose_maths_is_wrong(schema):
    """2 x 584.79 booked as 584.79 — the exact shape of the unit-price-as-total
    bug that shipped at 94% confidence."""
    lines = [{"item_description": "Acer TravelMate P2", "quantity": 2,
              "unit_price": 584.79, "line_amount": 584.79}]
    header = dict(_CLEAN_HEADER, invoice_amount=1169.58)
    results = _by_name(run_invariants(header, lines, schema))

    arith = results["line_arithmetic"]
    assert arith.severity in ("WARNING", "CRITICAL"), (
        "line_arithmetic passed a line where quantity x unit_price != amount"
    )
    assert "1 of 1" in (arith.message or "")


def test_subtotal_closure_catches_lines_that_do_not_sum_to_the_header(schema):
    lines = [{"item_description": "Widget", "quantity": 1,
              "unit_price": 100.00, "line_amount": 100.00}]
    results = _by_name(run_invariants(_CLEAN_HEADER, lines, schema))
    assert results["subtotal_closure"].severity in ("WARNING", "CRITICAL")


# --------------------------------------------------------------------------
# The bridge itself: what dispatch would file
# --------------------------------------------------------------------------

def _discrepancies_dispatch_would_file(results):
    """The exact filter dispatch.py:563-579 applies."""
    out = []
    for ir in results:
        if ir.severity == "CRITICAL":
            out.append((ir.name, "critical", True))
        elif ir.severity == "WARNING":
            out.append((ir.name, "warning", False))
    return out


def test_a_clean_document_files_no_discrepancies(schema):
    filed = _discrepancies_dispatch_would_file(
        run_invariants(_CLEAN_HEADER, _CLEAN_LINES, schema)
    )
    assert filed == [], (
        f"a document with consistent arithmetic filed {len(filed)} "
        f"invariant discrepancies: {filed}"
    )


def test_a_broken_document_files_discrepancies(schema):
    """The whole point. A known-broken invoice must reach the queue."""
    broken = dict(_CLEAN_HEADER, invoice_total_incl_tax=9999.99,
                  tax_percent=50.0)
    lines = [{"item_description": "Acer", "quantity": 2,
              "unit_price": 584.79, "line_amount": 584.79}]
    filed = _discrepancies_dispatch_would_file(
        run_invariants(broken, lines, schema)
    )
    names = {n for n, _s, _b in filed}
    assert filed, "a knowingly-broken invoice filed no invariant discrepancies"
    assert "line_arithmetic" in names
    assert "grand_total_closure" in names or "tax_closure" in names
