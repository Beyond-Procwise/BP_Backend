"""FX must be resolved once, stamped onto the fact, and never looked up again
at render time — otherwise the same report renders differently tomorrow.

bp_fx_rates has no historical dimension: its columns are
(base_currency, currency, rate, fetched_at) and every row shares one
fetched_at. So rate_date is the snapshot's fetched_at, NOT the transaction
date. These tests pin that behaviour so the limitation stays visible.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.fx import FX_UNAVAILABLE, resolve_fx  # noqa: E402

_T = datetime(2026, 7, 16, 15, 35, 57, tzinfo=timezone.utc)


class _Cur:
    def __init__(self, rows):
        self._rows = rows
        self.description = [("rate",), ("fetched_at",), ("base_currency",)]
    def execute(self, sql, params=None):
        self.sql, self.params = sql, params
    def fetchone(self):
        return self._rows[0] if self._rows else None


def test_same_currency_is_rate_one_without_a_lookup():
    r = resolve_fx(_Cur([]), "GBP", "GBP")
    assert r.rate == Decimal("1")
    assert FX_UNAVAILABLE not in r.reason_codes


def test_resolved_rate_carries_its_date_and_source():
    r = resolve_fx(_Cur([(Decimal("1.2734"), _T, "USD")]), "USD", "GBP")
    assert r.rate == Decimal("1.2734")
    assert r.rate_date == _T
    assert r.source and "bp_fx_rates" in r.source


def test_missing_pair_fails_closed():
    r = resolve_fx(_Cur([]), "XYZ", "GBP")
    assert r.rate is None and r.rate_date is None
    assert FX_UNAVAILABLE in r.reason_codes


def test_rate_is_decimal_not_float():
    """Float FX rates reintroduce the rounding drift the benchmark engine went
    to some trouble to eliminate."""
    r = resolve_fx(_Cur([(Decimal("1.2734"), _T, "USD")]), "USD", "GBP")
    assert isinstance(r.rate, Decimal)


def test_unknown_currency_codes_do_not_raise():
    assert resolve_fx(_Cur([]), "", "GBP").rate is None
    assert resolve_fx(_Cur([]), None, "GBP").rate is None  # type: ignore[arg-type]


def test_the_latest_snapshot_wins_deterministically():
    """bp_sqldb holds 7 fetched_at snapshots of the same pairs (1,162 rows),
    not the single snapshot the plan's F4 recorded. Without an explicit
    ordering the resolver would return an arbitrary duplicate, so two runs of
    the same report could stamp different rates. Pin the ordering in the SQL.
    """
    cur = _Cur([(Decimal("1.2734"), _T, "USD")])
    resolve_fx(cur, "USD", "GBP")
    sql = " ".join(cur.sql.lower().split())
    assert "order by fetched_at desc" in sql
    assert "limit 1" in sql


def test_a_float_rate_from_the_driver_is_not_silently_kept_as_float():
    """psycopg2 returns NUMERIC as Decimal, but a schema change or a fake store
    could hand back a float. Coerce rather than propagate the drift."""
    r = resolve_fx(_Cur([(1.2734, _T, "USD")]), "USD", "GBP")
    assert isinstance(r.rate, Decimal)
