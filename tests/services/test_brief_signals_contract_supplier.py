"""The supplier join in `_EXPIRING_SQL` matches nothing, and is kept on purpose.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/test_brief_signals_contract_supplier.py

The ruling is already recorded above `_EXPIRING_SQL` in
`src/services/brief_signals_service.py`, and this file exists to hold it in a
form that runs:

    "The supplier join currently matches NOTHING, and is kept rather than
    dropped. ... a contract cannot be named by its supplier today. The reading
    falls back to the contract's own title and invents nobody; the join starts
    paying the moment those ids are crosswalked."

A comment cannot tell you when its premise expires. These tests can.

**What the measurement says**, re-taken 2026-10-04 and agreeing with the comment:
`proc.bp_contract_master.supplier_id` holds at least four different things —
2,981 rows in an ``S9251`` namespace no other table in the schema uses, a second
``SI000703`` namespace, bare numbers like ``505697``, and extraction debris
(``'Edwards Sarah Thompson'``, ``'policy.'``, ``'Opportunity Finder Agent - Hard
Rule Policies v2.0'``). `proc.bp_supplier.supplier_id` is
``SUP-CopperleafSystems``. Zero of the 3,008 match an id, a supplier NAME, or a
row in `bp_supplier_id_crosswalk`; 19 of 3,051 reach `bp_supplier_master`.

**Why the join stays.** It is a LEFT JOIN, so it removes no contract and changes
no reading: `_contract_name` already falls back to the title, and
`expiring_signal` prefers a supplier name only where one exists. Dropping it
would buy nothing and would mean a code change on the day the crosswalk lands.
Keeping it means the brief begins naming suppliers by itself.

**What these tests add.** A self-healing join heals silently, and silence is the
problem: nobody would know the brief had changed. The tripwire turns that into
an event, and the positive control is what makes its zero mean something.

See also `project_contract_supplier_untrustworthy` — a contract's supplier field
has been found holding the BUYER's name on every live document checked by hand,
so a crosswalk that lands is worth reading before it is trusted.
"""
from __future__ import annotations

import os
from datetime import date

import pytest

from src.services import brief_signals_service as bss
from src.services.db import get_conn

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on",
)

_RESOLVES = """
    SELECT count(*), count(s.supplier_id)
      FROM proc.{table} x
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = x.supplier_id
"""


def _resolution(table: str) -> tuple[int, int]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_RESOLVES.format(table=table))
        return cur.fetchone()


def test_the_expiring_query_still_carries_the_supplier_join():
    """The ruling, asserted. Runs without a database so it holds even in a suite
    run that skips every live test.

    Removing the join is the plausible-looking tidy-up this guards against: it
    matches nothing today, so it reads like dead code. It is not — it is the
    thing that makes the brief start naming suppliers the day the ids are
    crosswalked, with no release. Read the comment above `_EXPIRING_SQL` and the
    tripwire below before taking it out.
    """
    sql = bss._EXPIRING_SQL.lower()
    assert "left join proc.bp_supplier" in sql
    assert "s.supplier_name" in sql


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_every_other_document_type_resolves_its_supplier():
    """The positive control, without which the tripwire's zero proves nothing.

    If invoices, POs and quotes also failed to resolve, a zero for contracts
    would mean the join is wrong rather than the data. They resolve completely.
    """
    for table in ("bp_invoice_trgt", "bp_purchase_order_trgt", "bp_quote_trgt"):
        total, resolved = _resolution(table)
        assert total > 0, f"{table} is empty; the control proves nothing"
        assert resolved == total, (
            f"{table}: only {resolved} of {total} suppliers resolve -- the join "
            f"mechanism itself is suspect, so read the tripwire's zero carefully"
        )


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_no_contract_resolves_its_supplier_TRIPWIRE():
    """RED means the premise expired: the ids have been crosswalked.

    Asserting a defect is deliberate. On the day this fails the brief has
    silently begun naming suppliers on expiring contracts, and somebody should
    read what it is naming them before trusting it -- the column has held a
    person's name and an agent's name, so a crosswalk is not the same as a
    correct supplier. Then delete this test, and the paragraph it pins above
    `_EXPIRING_SQL`.
    """
    total, resolved = _resolution("bp_contract_master")
    assert total > 0
    assert resolved == 0, (
        f"{resolved} of {total} contract supplier ids now resolve to "
        f"proc.bp_supplier. The join above _EXPIRING_SQL has started paying: "
        f"check what Today's brief is now naming, then retire this test."
    )


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_brief_names_every_expiring_contract_despite_the_empty_join():
    """The reading is unnamed by supplier, never missing. Without this, the join
    matching nothing could be mistaken for the signal being broken.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(bss._EXPIRING_SQL, (bss.EXPIRY_WINDOW_DAYS,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    assert rows, "no contract expires in the window; this proves nothing today"
    assert all(r["supplier_name"] is None for r in rows), "see the tripwire"

    signal = bss.expiring_signal(rows, date.today(), bss.EXPIRY_WINDOW_DAYS, None)
    assert signal is not None
    assert signal["count"] == len(rows)
    # Named by title or id, never blank, and never a fabricated supplier.
    assert all(i["name"] for i in signal["items"])
    assert signal["nearest"]["name"]
