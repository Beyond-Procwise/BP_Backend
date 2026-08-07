"""uom.py loads its vocabulary from proc.bp_uom_canonical at runtime.

Three hazards this must not have, each pinned below:

  * An empty or failed load must NOT blank the vocabulary. A normaliser that
    suddenly recognises nothing marks every unit UOM_UNMAPPED, and those
    absences are then written into the fact base as though the documents had
    said nothing. That is far worse than a slightly stale map.
  * No query in the per-value hot path. assemble_line_facts calls normalise_uom
    once per line; a lookup per call is the N+1 pattern that already cost this
    codebase an information_schema query per document.
  * status='proposed' must never resolve. It is an observation awaiting a
    human, and if it silently started normalising, confirming it would be
    pointless.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts import uom as uom_mod  # noqa: E402
from src.services.facts.uom import (  # noqa: E402
    UOM_UNMAPPED,
    Vocabulary,
    active_vocabulary,
    build_vocabulary,
    ensure_vocabulary,
    normalise_uom,
    reset_vocabulary,
)

COLS = ["uom_code", "dimension", "aliases", "factor_days", "factor_convention"]


class FakeCur:
    """Returns rows for the vocabulary query and counts how often it is asked."""

    def __init__(self, rows, fail=False):
        self._rows, self._fail = rows, fail
        self.description = [(c,) for c in COLS]
        self.calls = 0

    def execute(self, sql, params=None):
        self.calls += 1
        if self._fail:
            raise RuntimeError("database unavailable")

    def fetchall(self):
        return list(self._rows)


def _row(code, dim="count", aliases=None, factor=None, convention=None):
    return [code, dim, aliases or [], factor, convention]


@pytest.fixture(autouse=True)
def _isolate():
    """Every test starts from the built-in seed and leaves it that way."""
    reset_vocabulary()
    yield
    reset_vocabulary()


def test_the_builtin_seed_is_the_default_before_any_load():
    assert active_vocabulary().source == "builtin-seed"
    assert normalise_uom("each").canonical == "each"


def test_a_unit_only_in_the_table_resolves_after_loading():
    assert normalise_uom("pallet").canonical is None
    cur = FakeCur([_row("pallet", "count", ["pallets"])])
    ensure_vocabulary(cur)
    assert normalise_uom("pallet").canonical == "pallet"
    assert normalise_uom("pallets").canonical == "pallet"
    assert active_vocabulary().source.startswith("bp_uom_canonical")


def test_the_query_asks_only_for_active_rows():
    """Filtering in Python would mean a proposed unit was ever a candidate."""
    captured = {}

    class Cur(FakeCur):
        def execute(self, sql, params=None):
            captured["sql"] = " ".join(sql.lower().split())
            self.calls += 1

    ensure_vocabulary(Cur([_row("pallet")]))
    assert "status" in captured["sql"] and "'active'" in captured["sql"]


def test_an_empty_table_does_not_blank_the_vocabulary():
    """The catastrophic case. Zero active rows must be treated as a failed
    load, not as 'no units exist' — otherwise every unit in the corpus silently
    becomes UOM_UNMAPPED and the absences are written into facts."""
    ensure_vocabulary(FakeCur([]))
    assert normalise_uom("each").canonical == "each"
    assert active_vocabulary().source == "builtin-seed"


def test_rows_that_are_not_vocabulary_rows_do_not_blank_it():
    """Non-empty rows that parse to nothing usable — a changed column list, or
    a cursor answering a different query — must be treated as a failed load.
    Checking the raw row count instead of the built vocabulary let exactly this
    install an empty vocabulary and silently unmap the whole corpus."""
    class WrongShape(FakeCur):
        def __init__(self):
            super().__init__([["INV-1", "GBP", "SUP-1"]])
            self.description = [("invoice_id",), ("currency",), ("supplier_id",)]

    ensure_vocabulary(WrongShape())
    assert normalise_uom("each").canonical == "each"
    assert active_vocabulary().source == "builtin-seed"


def test_a_failed_load_keeps_the_previous_vocabulary():
    ensure_vocabulary(FakeCur([_row("pallet")]))
    assert normalise_uom("pallet").canonical == "pallet"

    ensure_vocabulary(FakeCur([], fail=True), ttl_seconds=0)
    assert normalise_uom("pallet").canonical == "pallet", (
        "a database outage must not silently shrink the vocabulary"
    )


def test_a_failed_first_load_falls_back_to_the_seed_not_to_nothing():
    ensure_vocabulary(FakeCur([], fail=True))
    assert normalise_uom("each").canonical == "each"
    assert UOM_UNMAPPED not in normalise_uom("each").reason_codes


def test_the_vocabulary_is_cached_not_queried_per_value():
    cur = FakeCur([_row("pallet")])
    ensure_vocabulary(cur)
    for _ in range(50):
        normalise_uom("pallet")
        ensure_vocabulary(cur)
    assert cur.calls == 1, f"expected one query, made {cur.calls}"


def test_ttl_zero_forces_a_reload():
    cur = FakeCur([_row("pallet")])
    ensure_vocabulary(cur)
    ensure_vocabulary(cur, ttl_seconds=0)
    assert cur.calls == 2


def test_a_reload_replaces_rather_than_merges():
    """The table is authoritative. A unit removed from it must stop resolving,
    or the code would accumulate units nobody can find in the source of truth."""
    ensure_vocabulary(FakeCur([_row("pallet"), _row("crate")]))
    assert normalise_uom("crate").canonical == "crate"
    ensure_vocabulary(FakeCur([_row("pallet")]), ttl_seconds=0)
    assert normalise_uom("crate").canonical is None
    assert normalise_uom("pallet").canonical == "pallet"


def test_factors_and_conventions_survive_the_load():
    cur = FakeCur([_row("fortnight", "time", ["fortnights"],
                        Decimal("14"), "CALENDAR_CONVENTION_30D_365D")])
    ensure_vocabulary(cur)
    r = normalise_uom("fortnight")
    assert r.dimension == "time"
    assert r.factor == Decimal("14")
    assert any("CALENDAR_CONVENTION" in c for c in r.reason_codes)


def test_a_loaded_unit_with_no_convention_carries_no_convention_code():
    ensure_vocabulary(FakeCur([_row("crate", "count")]))
    assert normalise_uom("crate").reason_codes == ()


def test_normalise_uom_stays_pure_when_given_a_vocabulary():
    """The function remains testable without touching module state, which is
    what keeps the rest of the suite deterministic."""
    vocab = build_vocabulary(
        [{"uom_code": "widget", "dimension": "count", "aliases": ["widgets"],
          "factor_days": None, "factor_convention": None}],
        source="test",
    )
    assert normalise_uom("widget", vocabulary=vocab).canonical == "widget"
    # module state untouched
    assert normalise_uom("widget").canonical is None
    assert active_vocabulary().source == "builtin-seed"


def test_a_float_factor_from_the_driver_becomes_decimal():
    ensure_vocabulary(FakeCur([_row("fortnight", "time", [], 14.0, "X")]))
    assert isinstance(normalise_uom("fortnight").factor, Decimal)


def test_exact_matching_survives_the_load():
    """The single most important rule in the module must not be lost when the
    vocabulary comes from elsewhere: 'transition 7 weeks' contains 'week'."""
    ensure_vocabulary(FakeCur([_row("week", "time", ["weeks"], Decimal("7"))]))
    assert normalise_uom("transition 7 weeks").canonical is None
    assert UOM_UNMAPPED in normalise_uom("transition 7 weeks").reason_codes


def test_a_vocabulary_is_frozen():
    with pytest.raises(Exception):
        active_vocabulary().source = "tampered"  # type: ignore[misc]
