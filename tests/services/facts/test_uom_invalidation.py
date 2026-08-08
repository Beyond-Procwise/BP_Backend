"""Invalidation: a confirmed unit must take effect without waiting out the TTL.

Two mechanisms, because they solve different halves of the problem:

  * invalidate_vocabulary() drops the cache in THIS process. Immediate, but a
    function call cannot reach the other workers.
  * A cheap version probe lets every other process notice on its own. Without
    it, invalidation would only ever work in whichever worker happened to
    receive the call — which looks like it works when tested on one process and
    silently does not in production.

The probe reads count and max(recorded_at) over a ~24-row table, not the whole
vocabulary, and only every probe_seconds. Reloading in full on every check
would reintroduce the per-call query the TTL exists to avoid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.uom import (  # noqa: E402
    active_vocabulary,
    ensure_vocabulary,
    invalidate_vocabulary,
    normalise_uom,
    reset_vocabulary,
)

LOAD_COLS = ["uom_code", "dimension", "aliases", "factor_days",
             "factor_convention", "recorded_at"]


class Cur:
    """Answers the load query and the probe query separately, counting each."""

    def __init__(self, units, stamp="t0"):
        self.units, self.stamp = list(units), stamp
        self.loads = self.probes = 0
        self.description, self._rows, self._one = [], [], None

    def execute(self, sql, params=None):
        s = " ".join(sql.lower().split())
        if "count(*)" in s:
            self.probes += 1
            self._one = (len(self.units), self.stamp)
        else:
            self.loads += 1
            self.description = [(c,) for c in LOAD_COLS]
            self._rows = [[u, "count", [], None, None, self.stamp]
                          for u in self.units]

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._one


@pytest.fixture(autouse=True)
def _isolate():
    reset_vocabulary()
    yield
    reset_vocabulary()


def test_invalidate_forces_a_reload_inside_the_ttl():
    cur = Cur(["pallet"])
    ensure_vocabulary(cur)
    assert cur.loads == 1

    cur.units.append("crate")
    ensure_vocabulary(cur)          # cached; must not see the new unit
    assert normalise_uom("crate").canonical is None

    invalidate_vocabulary()
    ensure_vocabulary(cur)
    assert cur.loads == 2
    assert normalise_uom("crate").canonical == "crate"


def test_invalidate_before_any_load_is_harmless():
    invalidate_vocabulary()
    assert active_vocabulary().source == "builtin-seed"
    cur = Cur(["pallet"])
    ensure_vocabulary(cur)
    assert normalise_uom("pallet").canonical == "pallet"


def test_within_the_probe_interval_nothing_is_queried_at_all():
    cur = Cur(["pallet"])
    ensure_vocabulary(cur)
    for _ in range(50):
        ensure_vocabulary(cur, probe_seconds=300)
    assert (cur.loads, cur.probes) == (1, 0)


def test_a_probe_that_sees_no_change_does_not_reload():
    """The common case, and the one that must stay cheap."""
    cur = Cur(["pallet"])
    ensure_vocabulary(cur)
    for _ in range(10):
        ensure_vocabulary(cur, probe_seconds=0)
    assert cur.probes == 10
    assert cur.loads == 1, "an unchanged vocabulary must not be re-read"


def test_another_worker_changing_the_table_is_picked_up_by_the_probe():
    """The half a function call cannot reach."""
    cur = Cur(["pallet"], stamp="t0")
    ensure_vocabulary(cur)
    assert normalise_uom("crate").canonical is None

    cur.units.append("crate")       # someone else confirmed a unit
    cur.stamp = "t1"

    ensure_vocabulary(cur, probe_seconds=0)
    assert cur.loads == 2
    assert normalise_uom("crate").canonical == "crate"


def test_a_change_that_only_moves_the_timestamp_is_still_noticed():
    """Editing a row without adding one leaves the count identical, so a
    count-only version would miss it entirely."""
    cur = Cur(["pallet"], stamp="t0")
    ensure_vocabulary(cur)
    cur.stamp = "t1"
    ensure_vocabulary(cur, probe_seconds=0)
    assert cur.loads == 2


def test_a_failing_probe_keeps_the_vocabulary_and_does_not_raise():
    class Failing(Cur):
        def execute(self, sql, params=None):
            if "count(*)" in " ".join(sql.lower().split()):
                raise RuntimeError("probe failed")
            super().execute(sql, params)

    cur = Failing(["pallet"])
    ensure_vocabulary(cur)
    ensure_vocabulary(cur, probe_seconds=0)
    assert normalise_uom("pallet").canonical == "pallet"


def test_a_probe_returning_nothing_does_not_blank_the_vocabulary():
    class Empty(Cur):
        def fetchone(self):
            return None

    cur = Empty(["pallet"])
    ensure_vocabulary(cur)
    ensure_vocabulary(cur, probe_seconds=0)
    assert normalise_uom("pallet").canonical == "pallet"
