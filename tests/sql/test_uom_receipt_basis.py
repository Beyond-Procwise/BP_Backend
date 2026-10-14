"""Every unit declares whether a human can take delivery of what it measures.

`bp_uom_canonical.dimension` describes the PHYSICAL MEASURE and is close but
wrong for this question: `licence`, `seat` and `module` all sit under `count`
while being no more deliverable than an hour. A three-way match that read
`dimension` would quietly expect a goods receipt for a software licence, find
none, and report the line as unreceived -- a manufactured failure.

The second test is the one that matters most: a unit with no `receipt_basis`
silently skips the match, so an unclassified unit is a hole in the control
rather than a gap in a reference table.
"""
from __future__ import annotations

import os

import pytest

from src.services.db import get_conn

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live db")


@pytest.mark.parametrize("uom,expected", [
    ("each", "goods_receipt"), ("box", "goods_receipt"), ("tonne", "goods_receipt"),
    ("metre", "goods_receipt"), ("case", "goods_receipt"), ("pack", "goods_receipt"),
    ("hour", "service_entry"), ("day", "service_entry"), ("month", "service_entry"),
    ("licence", "service_entry"),   # counted, but nobody takes delivery of one
    ("seat", "service_entry"), ("module", "service_entry"),
])
def test_every_unit_declares_whether_it_can_be_received(uom, expected):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT receipt_basis FROM proc.bp_uom_canonical WHERE uom_code=%s",
                    (uom,))
        row = cur.fetchone()
        assert row is not None, f"{uom!r} is not in the canonical unit table"
        assert row[0] == expected


def test_no_unit_is_left_unclassified():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT uom_code FROM proc.bp_uom_canonical WHERE receipt_basis IS NULL")
        assert cur.fetchall() == [], "an unclassified unit silently skips the match"


def test_the_extraction_noise_can_prove_nothing():
    """The rows whose dimension is NULL are not units -- they are extraction
    noise that reached a reference table ('30 days from invoice',
    'implementation (one-off, fixed)'). They read as 'none': nothing can prove
    a line measured in one.

    They are also already `status='rejected'` with a `non_unit_kind`, set by
    2026-08-08_uom_non_unit_kind.sql. The plan asked for `status='retired'`
    here; that is not an allowed status and the stamp would have been redundant
    with the one the table already carries.
    """
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT uom_code, receipt_basis, status, non_unit_kind "
                    "FROM proc.bp_uom_canonical WHERE dimension IS NULL")
        rows = cur.fetchall()
        assert rows, "expected the noise rows to still be present"
        bad = [r for r in rows if r[1] != "none" or r[2] != "rejected" or not r[3]]
        assert bad == [], f"noise rows not marked unprovable non-units: {bad}"


def test_the_column_refuses_a_value_outside_the_three():
    """A fourth spelling would be read by Task 8's branch as 'not
    goods_receipt' and silently excluded from the match."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT uom_code FROM proc.bp_uom_canonical LIMIT 1")
        victim = cur.fetchone()[0]
        with pytest.raises(Exception) as exc:
            cur.execute("UPDATE proc.bp_uom_canonical SET receipt_basis='maybe' "
                        "WHERE uom_code=%s", (victim,))
        assert "ck_bp_uom_canonical_receipt_basis" in str(exc.value)
