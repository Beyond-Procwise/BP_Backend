"""Pins the guarantees around dispatch._apply_currency_resolution / _resolve_bare_dollar_currency_hint.

Four things have to hold no matter what:

  (a) an UNLEARNED supplier resolves nothing. proc.bp_supplier.default_currency must not
      settle a bare "$" on its own: 481 live suppliers carry a dollar default, and letting
      it answer would auto-resolve documents that stopped for a human before this feature
      existed. The governing rule is that anything short of certainty goes to a person, and
      confidence is earned back from what that person says — a static column on the vendor
      master is not somebody agreeing with us;
  (b) a LEARNED supplier does resolve — three people correcting the same supplier to the
      same currency is exactly the confidence that was supposed to be earned;
  (c) the "supplier_default_currency" hint never survives past the block, including when
      something inside it raises. `persistence.write_raw` inserts every key in `columns`
      with no whitelist filtering;
  (d) a supplier name that matches more than one bp_supplier row must not silently hand
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
    """Dispatches by substring match on the SQL text, most-specific first.

    A script entry is (needle, rows) or (needle, rows, column_names) — the third element
    matters for the learned-currency query, whose result is zipped against
    cursor.description to build dicts.
    """

    def __init__(self, script: list[tuple]):
        self._script = script
        self._last: list[tuple] = []
        self.description = [("col",)]

    def execute(self, sql, params=None):
        for entry in self._script:
            needle, rows = entry[0], entry[1]
            if needle in sql:
                self._last = rows
                self.description = [(c,) for c in entry[2]] if len(entry) > 2 else [("col",)]
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
_VERDICT_COLS = ["supplier_id", "corrected_value", "verdict"]


def _learned(supplier="SUP-ONE", code="SGD", n=3):
    """A verdict script entry: `n` humans corrected `supplier` to `code`."""
    return ("bp_extraction_verdict",
            [(supplier, code, "corrected")] * n,
            _VERDICT_COLS)


def test_an_unlearned_supplier_resolves_nothing_and_stops_for_a_human(monkeypatch):
    """The supplier master says USD. Nobody has ever confirmed it. The document must
    still go to the Action page — that is the behaviour that existed before this branch,
    and 'the vendor master has a guess' is not the human-earned confidence that is
    allowed to change it."""
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
    assert columns.get("currency") is None
    assert [d.issue_type for d in discrepancies] == ["currency_ambiguous"]
    assert discrepancies[0].blocks_promotion is True


def test_a_supplier_people_have_settled_does_resolve(monkeypatch):
    """The payoff, and the only way the hint is ever populated: three corrections to SGD
    mean the fourth invoice stops asking. Note the supplier master says USD here — proof
    the answer came from the humans, not from the vendor record."""
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _learned("SUP-ONE", "SGD"),
        ("default_currency", [("USD",)]),
        ("bp_supplier", [("SUP-ONE",)]),
    ])
    columns = {"supplier_name": "Acme Corp"}
    discrepancies: list[Discrepancy] = []
    dispatch_mod._apply_currency_resolution(columns, "Total $500.00", _fake_registry(), discrepancies)

    assert columns.get("currency") == "SGD"
    assert discrepancies == []
    # ...and the hint itself is still gone before anything can try to persist it.
    assert "supplier_default_currency" not in columns


def test_hint_is_absent_even_when_the_resolver_raises(monkeypatch):
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _learned("SUP-ONE", "CAD"),
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
        _learned("SUP-ONE", "SGD"),
        ("default_currency", [("SGD",)]),
        ("bp_supplier", [("SUP-ONE",)]),
    ])
    columns = {"supplier_name": "Unambiguous Co"}
    dispatch_mod._resolve_bare_dollar_currency_hint(columns)

    assert columns["supplier_default_currency"] == "SGD"


def test_two_corrections_are_not_enough_to_earn_it(monkeypatch):
    # MIN_AGREEMENTS is 3. Two people is an opinion; the third makes it a policy.
    _patch_conn(monkeypatch, [
        _NO_ALIAS,
        _learned("SUP-ONE", "SGD", n=2),
        ("default_currency", [("SGD",)]),
        ("bp_supplier", [("SUP-ONE",)]),
    ])
    columns = {"supplier_name": "Unambiguous Co"}
    dispatch_mod._resolve_bare_dollar_currency_hint(columns)

    assert "supplier_default_currency" not in columns
