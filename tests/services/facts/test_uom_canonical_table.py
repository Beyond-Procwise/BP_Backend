"""The unit table and uom.py must agree, and unconfirmed units must not resolve.

The vocabulary now exists twice: as the seed map in src/services/facts/uom.py
and as rows in proc.bp_uom_canonical. Duplication between code and data is only
safe when something fails loudly the moment the two diverge — otherwise this
becomes the second vocabulary that B3 was trying to avoid, which is precisely
how the original drift went unnoticed.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/facts/test_uom_canonical_table.py
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.uom import (  # noqa: E402
    _ALIASES,
    _CANONICAL,
    UOM_UNMAPPED,
    normalise_uom,
)

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def rows():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT uom_code, dimension, aliases, factor_days, factor_convention,
                   status, is_billing_basis, non_unit_kind
              FROM proc.bp_uom_canonical
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _active(rows):
    return {r["uom_code"]: r for r in rows if r["status"] == "active"}


def test_every_python_unit_exists_in_the_table(rows):
    active = _active(rows)
    missing = set(_CANONICAL) - set(active)
    assert not missing, (
        f"units in uom.py with no active row in bp_uom_canonical: {sorted(missing)}"
    )


def test_every_active_table_unit_exists_in_python(rows):
    """The other direction. Without this, a unit could be activated in the
    database and silently never normalise, which looks like the table works."""
    active = _active(rows)
    extra = set(active) - set(_CANONICAL)
    assert not extra, (
        f"active units in bp_uom_canonical that uom.py cannot resolve: {sorted(extra)}"
    )


def test_dimensions_and_factors_agree(rows):
    for code, row in _active(rows).items():
        canonical, dimension, factor = _CANONICAL[code]
        assert row["dimension"] == dimension, f"{code}: dimension differs"
        table_factor = row["factor_days"]
        if factor is None:
            assert table_factor is None, f"{code}: table has a factor, uom.py does not"
        else:
            # hour is 1/24, which the table stores as NULL rather than a
            # repeating decimal; every other factor must match exactly.
            if table_factor is not None:
                assert Decimal(str(table_factor)) == factor, f"{code}: factor differs"


def test_aliases_in_the_table_actually_resolve(rows):
    """An alias recorded in the table but absent from uom.py would resolve in
    one place and not the other."""
    for row in rows:
        if row["status"] != "active":
            continue
        for alias in row["aliases"] or []:
            resolved = normalise_uom(alias).canonical
            assert resolved == row["uom_code"], (
                f"alias {alias!r} of {row['uom_code']!r} resolved to {resolved!r}"
            )


def test_python_aliases_are_recorded_in_the_table(rows):
    table_aliases = {a for r in rows for a in (r["aliases"] or [])}
    missing = set(_ALIASES) - table_aliases
    assert not missing, f"aliases in uom.py absent from the table: {sorted(missing)}"


def test_nothing_but_an_active_unit_normalises(rows):
    """The load-bearing rule of the status column. A proposed unit is an
    observation awaiting a human, not a decision; a rejected one is a decision
    that it is not a unit at all. Neither may resolve — if they did, the whole
    review step would be decoration.

    Asserted over whatever is currently proposed rather than requiring some to
    exist: the queue is meant to empty, and a test that fails when the backlog
    is cleared punishes the desired outcome.
    """
    for row in rows:
        if row["status"] == "active":
            continue
        code = row["uom_code"]
        r = normalise_uom(code)
        assert r.canonical is None, (
            f"{row['status']} unit {code!r} resolved to {r.canonical!r}"
        )
        assert UOM_UNMAPPED in r.reason_codes


def test_the_service_engagement_bases_are_rejected_not_units(rows):
    """Decided 2026-08-08. They are deliverable types, not a measurement
    dimension: 'audit' was attached to a laser printer and 'programme' to a
    mouse (extraction noise), while the genuine ones state their real basis in
    the description — 'HR Advisory Retainer (Quarterly)' is billed per QUARTER,
    a unit already mapped. An 'engagement' dimension would have blessed the
    noise and frozen a deliverable noun in place of the real period."""
    by_code = {r["uom_code"]: r for r in rows}
    for code in ("service", "programme", "retainer", "audit"):
        assert code in by_code, f"{code} should be recorded as observed"
        assert by_code[code]["status"] == "rejected"
        assert by_code[code]["is_billing_basis"] is False
        assert by_code[code]["non_unit_kind"] == "deliverable_type"


def test_every_rejected_value_records_why(rows):
    """'Fix the extractor' and 'this is a real lump sum' are different
    problems. Without the kind they look identical, and a data-quality defect
    stays hidden inside a modelling decision."""
    unclassified = [r["uom_code"] for r in rows
                    if r["status"] == "rejected" and not r["non_unit_kind"]]
    assert not unclassified, f"rejected with no reason recorded: {sorted(unclassified)}"


def test_an_active_unit_carries_no_rejection_reason(rows):
    """A unit cannot simultaneously be a unit and a reason for not being one."""
    for row in rows:
        if row["status"] == "active":
            assert row["non_unit_kind"] is None, (
                f"active unit {row['uom_code']!r} carries non_unit_kind "
                f"{row['non_unit_kind']!r}"
            )


def test_the_canonical_master_vocabulary_is_now_covered(rows):
    """The point of the exercise: every unit_of_measure in the authoritative
    product master should now be either active or explicitly proposed —
    nothing silently unaccounted for."""
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT lower(btrim(unit_of_measure))
              FROM proc.bp_product_master
             WHERE btrim(coalesce(unit_of_measure, '')) <> ''
        """)
        observed = {r[0] for r in cur.fetchall()}

    known = {r["uom_code"] for r in rows} | {a for r in rows for a in (r["aliases"] or [])}
    unaccounted = observed - known
    assert not unaccounted, (
        f"canonical units neither active nor proposed: {sorted(unaccounted)}"
    )
