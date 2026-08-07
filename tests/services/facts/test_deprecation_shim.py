"""The one-release shim over calculation_details.

calculation_details stops being the system of record in this phase but is still
written, so readers need a single chokepoint that prefers the structured column
and falls back to the JSONB. Every fallback is logged: a shim nobody can
measure cannot be retired, and "we think nothing uses it now" is not evidence.
"""
from __future__ import annotations

import logging
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.deprecation import read_calculation_detail  # noqa: E402


class _Finding:
    """The miner passes objects, the store passes dict rows."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_the_structured_column_wins_when_present(caplog):
    rec = {"opportunity_id": "O-1", "currency": "USD",
           "calculation_details": {"currency": "GBP"}}
    with caplog.at_level(logging.WARNING):
        assert read_calculation_detail(rec, "currency") == "USD"
    assert not caplog.records, "reading the column is not a deprecation event"


def test_it_falls_back_to_the_jsonb_and_logs_the_hit(caplog):
    rec = {"opportunity_id": "O-2", "calculation_details": {"currency": "GBP"}}
    with caplog.at_level(logging.WARNING):
        assert read_calculation_detail(rec, "currency") == "GBP"
    assert caplog.records, "every fallback must be logged or the shim cannot be retired"
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "currency" in msg
    assert "O-2" in msg, "the log must identify which opportunity still needs the JSONB"


def test_a_missing_key_returns_the_default_without_inventing_a_value():
    rec = {"opportunity_id": "O-3", "calculation_details": {}}
    assert read_calculation_detail(rec, "currency") is None
    assert read_calculation_detail(rec, "currency", default="X") == "X"


def test_it_reads_objects_as_well_as_dict_rows(caplog):
    finding = _Finding(opportunity_id="O-4", unit_price=Decimal("10"),
                       calculation_details={"actual_price": 99})
    with caplog.at_level(logging.WARNING):
        assert read_calculation_detail(finding, "unit_price") == Decimal("10")
    assert not caplog.records


def test_an_aliased_key_resolves_to_the_same_structured_column(caplog):
    """The benchmark detector writes actual_price; the column is unit_price.
    Without the alias the shim would fall back forever and the log would never
    go quiet."""
    finding = _Finding(opportunity_id="O-5", unit_price=Decimal("55.5"),
                       calculation_details={"actual_price": 99})
    with caplog.at_level(logging.WARNING):
        assert read_calculation_detail(finding, "actual_price") == Decimal("55.5")
    assert not caplog.records


def test_a_key_with_no_structured_column_falls_back_without_pretending(caplog):
    """item_description was never migrated to a new column. It must still read,
    and must still be logged, so the retirement decision is based on data."""
    rec = {"opportunity_id": "O-6", "calculation_details": {"band": "high"}}
    with caplog.at_level(logging.WARNING):
        assert read_calculation_detail(rec, "band") == "high"
    assert caplog.records


def test_a_null_structured_column_does_not_mask_a_real_jsonb_value(caplog):
    """An INDETERMINATE row has NULL columns. Treating NULL as authoritative
    would silently drop values the JSONB still holds during the shim release."""
    rec = {"opportunity_id": "O-7", "currency": None,
           "calculation_details": {"currency": "EUR"}}
    with caplog.at_level(logging.WARNING):
        assert read_calculation_detail(rec, "currency") == "EUR"
    assert caplog.records


def test_a_missing_calculation_details_payload_is_survivable():
    assert read_calculation_detail({"opportunity_id": "O-8"}, "currency") is None
    assert read_calculation_detail({"calculation_details": None}, "currency") is None
    assert read_calculation_detail(None, "currency") is None
