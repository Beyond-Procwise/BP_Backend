"""Pins the guarantees around dispatch._apply_currency_resolution / _resolve_bare_dollar_currency_hint.

These two are the only guard between the "learn this supplier's currency" feature and
`persistence.write_raw`, which inserts every key in `columns` with no whitelist filtering.
Three things have to hold no matter what:

  (a) the "supplier_default_currency" hint never survives past the block, on the ordinary
      path where everything resolves cleanly;
  (b) it never survives even when something inside the block raises;
  (c) a supplier name that matches more than one bp_supplier row must not silently hand
      back either one's currency — bp_supplier has no uniqueness constraint on
      supplier_name/trading_name, so a naive LIMIT 1 would pick a company at random.

These exercise the real dispatch.py functions (not a reimplementation), with a fake DB
connection so no live database is required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction import dispatch as dispatch_mod  # noqa: E402
from src.services.extraction.persistence import Discrepancy  # noqa: E402
import src.services.db as db_mod  # noqa: E402


class _FakeCursor:
    """Dispatches by substring match on the SQL text, most-specific first."""

    def __init__(self, script: list[tuple[str, list[tuple]]]):
        self._script = script
        self._last: list[tuple] = []
        self.description = [("col",)]

    def execute(self, sql, params=None):
        for needle, rows in self._script:
            if needle in sql:
                self._last = rows
                return
        self._last = []

    def fetchone(self):
        return self._last[0] if self._last else None

    def fetchall(self):
        return list(self._last)


class _FakeConn:
    def __init__(self, cur: _FakeCursor):
        self._cur = cur

    def cursor(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_registry(fields=("currency", "supplier_name")):
    field_objs = [SimpleNamespace(db_column=f) for f in fields]
    return SimpleNamespace(schema=SimpleNamespace(fields=field_objs))


@pytest.fixture(autouse=True)
def _reset_learned_currency_cache(monkeypatch):
    # The 15-minute learned-currency cache is process-global; force a fresh load in every
    # test so one test's fake DB can't leak into another's.
    monkeypatch.setattr(dispatch_mod, "_LEARNED_CCY_CACHE", {}, raising=False)
    monkeypatch.setattr(dispatch_mod, "_LEARNED_CCY_CACHED_AT", 0.0, raising=False)


def _patch_conn(monkeypatch, script):
    monkeypatch.setattr(db_mod, "get_conn", lambda: _FakeConn(_FakeCursor(script)))


# Order matters: bp_supplier_alias / bp_extraction_verdict / default_currency are all
# checked before the bare "bp_supplier" exact-match fallback, since each is also a
# substring match against a query that mentions "bp_supplier".
_NO_ALIAS = ("bp_supplier_alias", [])
_NO_VERDICTS = ("bp_extraction_verdict", [])


def test_hint_is_absent_after_the_block_runs_normally(monkeypatch):
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _NO_VERDICTS,
        ("default_currency", [("USD",)]),
        ("bp_supplier", [("SUP-ONE",)]),  # exact-name match: exactly one row
    ])
    columns = {"supplier_name": "Acme Corp"}
    discrepancies: list[Discrepancy] = []
    dispatch_mod._apply_currency_resolution(columns, "Total $500.00", _fake_registry(), discrepancies)

    assert "supplier_default_currency" not in columns
    # And prove the hint was actually exercised, not merely never set: it resolved the
    # bare "$" via the supplier master rather than raising an ambiguity discrepancy.
    assert columns.get("currency") == "USD"
    assert discrepancies == []


def test_hint_is_absent_even_when_the_resolver_raises(monkeypatch):
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _NO_VERDICTS,
        ("default_currency", [("CAD",)]),
        ("bp_supplier", [("SUP-ONE",)]),
    ])

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated failure inside the currency block")

    monkeypatch.setattr(
        "src.services.extraction.context_layer.resolve_dollar_currency", _boom,
    )

    columns = {"supplier_name": "Acme Corp"}
    with pytest.raises(RuntimeError, match="simulated failure"):
        dispatch_mod._apply_currency_resolution(
            columns, "Total $500.00", _fake_registry(), [],
        )

    # The exception propagated (asserted above) but the hint must still be gone — this is
    # the entire point of the try/finally: persistence.write_raw has no column whitelist.
    assert "supplier_default_currency" not in columns


def test_an_ambiguous_supplier_name_yields_no_hint(monkeypatch):
    # Two different real suppliers share this name (bp_supplier has no uniqueness
    # constraint on supplier_name/trading_name) — LIMIT 2 comes back with both rows.
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _NO_VERDICTS,
        ("default_currency", [("EUR",)]),
        ("bp_supplier", [("SUP-ONE",), ("SUP-TWO",)]),
    ])
    columns = {"supplier_name": "Ambiguous Traders Ltd"}
    dispatch_mod._resolve_bare_dollar_currency_hint(columns)

    assert "supplier_default_currency" not in columns


def test_an_exact_single_match_still_resolves(monkeypatch):
    # Companion to the ambiguous case: one match through the same code path DOES yield a
    # hint, so the ambiguity test above is proving something real, not a query that always
    # returns nothing.
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _NO_VERDICTS,
        ("default_currency", [("SGD",)]),
        ("bp_supplier", [("SUP-ONE",)]),
    ])
    columns = {"supplier_name": "Unambiguous Co"}
    dispatch_mod._resolve_bare_dollar_currency_hint(columns)

    assert columns["supplier_default_currency"] == "SGD"
