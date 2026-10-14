# Deal-less Documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the 14 quote rounds stranded in `_trgt` with `deal_id` NULL, stop counting superseded rounds in deal money totals, and report every remaining deal-less document so the silence that hid them cannot return.

**Architecture:** One new pure module holds the SQL definition of a quote round, next to the Python definition it must agree with. The attach pass and the deal-overview view both read from it. A read-only endpoint reports what is still unattached, splitting the actionable few from the standalone many.

**Tech Stack:** Python 3, psycopg2, PostgreSQL (schema `proc`), FastAPI, pytest.

**Spec:** `specs/2026-09-30-deal-less-documents-design.md`

## Global Constraints

- Plans and specs live in `specs/`. **`docs/` is gitignored in this repo** — nothing written there can be committed.
- This checkout is **shared with another session**. Never `git add -A`, never a bare `git commit`. Commit through a private index:
  `GIT_INDEX_FILE=<tmp> git read-tree HEAD` → `git add <only my paths>` → `git write-tree` → `git commit-tree` → `git update-ref refs/heads/Development` → `git reset -q HEAD -- <my paths>`.
- All new tables/indexes follow the `bp_` prefix convention. No new tables are created by this plan.
- `get_conn()` returns an **autocommit** connection. A test that needs rollback must set `conn.autocommit = False` first, as `tests/services/test_deal_save_live.py` does.
- DB-backed tests require `PROCWISE_TEST_LIVE_DB=1`; without it pytest uses a fake connection and the assertions are meaningless.
- Run test suites with `CUDA_VISIBLE_DEVICES=""` and a dead Ollama port. **Never run two pytest suites concurrently** — they contend on the live DB and both go spuriously red.
- Use `./venv/bin/python` for tests (`.venv` is the runtime; `venv` is the test env).
- Absent data stays NULL. Nothing in this plan fabricates a deal for a document that has no provable one.
- Every guard is watched go red before it is made to pass. A guard that has only ever been seen green has not been tested.

## Review Focus

Five conditions the spec implies but does not pin with a test. Each has a test added to the task that owns the code.

1. **A base whose only deal-bearing rows are `not_awarded` rivals.** Attaching a round with `award_status` NULL would make it count in `quote_total` while its own siblings deliberately do not — inflating the deal by a losing bid. The attached round must inherit the anchor's `award_status`. → Task 2, Step 9.
2. **Two-digit round ordinals.** `X (V10)` must outrank `X (V9)`. String ordering gets this wrong; the SQL casts to `int`. → Task 1, Step 1.
3. **A quote id containing an apostrophe or a regex metacharacter.** `document_id` is minted by `||` concatenation and the base is computed by `regexp_replace`; neither may break or silently mangle the id. → Task 1, Step 1.
4. **Anchor rows that disagree on `deal_name` while agreeing on `deal_id`.** `min(deal_name)` picks one; the attached row must not end up with a name no other row on the deal uses. → Task 2, Step 9.
5. **A deal-less quote that already carries a `document_id`.** Partial prior state must be overwritten to match the deal it is now on, not left pointing at nothing. → Task 2, Step 9.

---

### Task 1: The shared definition of a quote round

Two implementations of "which round is this" already exist in spirit — `version_collapse` in Python, ad-hoc regexes in SQL. This task makes the SQL one explicit, puts it beside the Python one, and pins them together with a test. Everything later reads from here.

**Files:**
- Create: `src/services/quote_rounds.py`
- Test: `tests/test_quote_rounds_parity.py`

**Interfaces:**
- Consumes: `src/services/version_collapse.py` — `base_reference(quote_id) -> str`, `version_ordinal(quote_id) -> int` (existing, unchanged)
- Produces:
  - `base_reference_sql(col: str = "quote_id") -> str`
  - `version_ordinal_sql(col: str = "quote_id") -> str`
  - `REVISION_CANDIDATES_SQL: str` — a complete SELECT returning columns `quote_id, supplier_id, total_amount, currency, candidate_deal_id, candidate_deal_name, candidate_award_status`

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_quote_rounds_parity.py`:

```python
"""The SQL and Python definitions of a quote round must agree.

version_collapse answers "which round is this" in Python, for clustering.
bp_deal_overview and the attach pass need the same answer inside SQL, on tables
too large to pull into Python. Two implementations of one rule drift silently —
the drift shows up as a deal quietly counting two rounds of one negotiation —
so both live in one place and this test holds them together.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/test_quote_rounds_parity.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.services.db import get_conn
from src.services.quote_rounds import base_reference_sql, version_ordinal_sql
from src.services.version_collapse import base_reference, version_ordinal

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

# Every shape the corpus and the extractor actually produce, plus the two
# Review Focus cases: a two-digit ordinal, and ids carrying characters that
# break naive quoting or regex handling.
FIXTURES = [
    "CPS-Q-3380",
    "CPS-Q-3380 (V2)",
    "ORB-Q-6612 (V3)",
    "ORB-Q-6612 (V3 (BAFO))",
    "Q-1 ( v2 )",
    "Q-1 (V10)",
    "Q-1 (V9)",
    "NXF-2024-441",
    "O'Brien-Q-1 (V2)",
    "A+B(C)-Q-1 (V3)",
    "quote (not a version)",
]


def test_sql_and_python_agree_on_base_and_ordinal():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT v, {base_reference_sql('v')}, {version_ordinal_sql('v')} "
            f"  FROM unnest(%s::text[]) AS t(v)",
            (FIXTURES,),
        )
        rows = cur.fetchall()

    assert len(rows) == len(FIXTURES)
    for quote_id, sql_base, sql_ordinal in rows:
        assert sql_base == base_reference(quote_id), f"base disagrees for {quote_id!r}"
        assert sql_ordinal == version_ordinal(quote_id), f"ordinal disagrees for {quote_id!r}"


def test_a_two_digit_round_outranks_a_single_digit_one():
    """String ordering puts '(V9)' above '(V10)'. The ordinal is an int."""
    assert version_ordinal("Q-1 (V10)") > version_ordinal("Q-1 (V9)")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {version_ordinal_sql('v')} FROM unnest(%s::text[]) AS t(v)",
            (["Q-1 (V10)", "Q-1 (V9)"],),
        )
        ten, nine = [r[0] for r in cur.fetchall()]
    assert ten > nine
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/test_quote_rounds_parity.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.services.quote_rounds'`. Confirm the failure is the missing module, not a DB connection error — a connection error means the test would also "fail" after the fix and proves nothing.

- [ ] **Step 3: Create the module**

Create `src/services/quote_rounds.py`:

```python
"""One definition of what a quote round is, shared by Python and SQL.

``version_collapse`` answers "which round is this" in Python and is what the
clustering code uses. ``bp_deal_overview`` and the revision-attach pass need the
same answer inside SQL, against tables too large to pull into Python.

Two implementations of one rule drift, and this particular drift is invisible:
it shows up as a deal quietly counting two rounds of one negotiation as two
competing bids. So both live here, side by side, and
tests/test_quote_rounds_parity.py holds them together against a shared fixture
set. Pure strings — no DB, no imports from the services that use them.
"""
from __future__ import annotations

# Mirrors version_collapse.base_reference: strip a trailing "(V<n>...)".
_BASE_REFERENCE = r"regexp_replace({col}, '\s*\(\s*[vV][0-9]+.*\)\s*$', '')"

# Mirrors version_collapse.version_ordinal. An UNVERSIONED id is round 1, not
# round 0 and not NULL -- a bare quote number is the opening bid. The ::int cast
# is load-bearing: as text, '(V9)' sorts above '(V10)'.
_VERSION_ORDINAL = (
    r"coalesce(nullif(substring({col} from '\(\s*[vV]([0-9]+)'), '')::int, 1)"
)


def base_reference_sql(col: str = "quote_id") -> str:
    """SQL expression for the base quote reference of ``col``."""
    return _BASE_REFERENCE.format(col=col)


def version_ordinal_sql(col: str = "quote_id") -> str:
    """SQL expression for the round ordinal of ``col``."""
    return _VERSION_ORDINAL.format(col=col)


# A deal-less quote, and the deal its own base reference already sits on.
#
# This proposes an ATTACHMENT ONLY where the evidence is an identity, never a
# resemblance. Inferring a link from supplier and amount is the mechanism behind
# the open mis-grouping bug, where an 81k invoice was absorbed into a 638 deal.
# Here the quote number and the supplier are the same recorded facts; the suffix
# is a round marker this codebase mints itself in
# context_layer.canonical_quote_revision.
#
# Three conditions, all required:
#   n_deals = 1      -- two candidate deals is a coin toss, not evidence. Hold.
#   n_suppliers = 1  -- a quote number reused across suppliers is a collision.
#   supplier matches -- the candidate's own supplier equals the anchor's.
#
# Direction-free by construction: it joins on base reference, so it works both
# when the deal holds the opening bid and the revision is stranded, and when the
# deal holds the latest round and the earlier ones are stranded. The second is
# the majority case on real data (10 of the 14).
#
# candidate_award_status carries the anchor's outcome so an attached round can
# inherit it. A base whose deal-bearing rows are all 'not_awarded' rivals must
# not contribute a counted quote to that deal through one of its other rounds.
REVISION_CANDIDATES_SQL = f"""
WITH anchors AS (
    SELECT {base_reference_sql('quote_id')}   AS base,
           min(deal_id)                        AS deal_id,
           min(deal_name)                      AS deal_name,
           min(award_status)                   AS award_status,
           count(DISTINCT deal_id)             AS n_deals,
           count(DISTINCT deal_name)           AS n_names,
           count(DISTINCT supplier_id)         AS n_suppliers,
           count(DISTINCT award_status)        AS n_award_statuses,
           min(supplier_id)                    AS supplier_id
      FROM proc.bp_quote_trgt
     WHERE deal_id IS NOT NULL
     GROUP BY 1
)
SELECT q.quote_id,
       q.supplier_id,
       q.total_amount,
       q.currency,
       a.deal_id                                          AS candidate_deal_id,
       CASE WHEN a.n_names = 1 THEN a.deal_name END       AS candidate_deal_name,
       CASE WHEN a.n_award_statuses = 1 THEN a.award_status END
                                                          AS candidate_award_status
  FROM proc.bp_quote_trgt q
  JOIN anchors a
    ON a.base = {base_reference_sql('q.quote_id')}
 WHERE q.deal_id IS NULL
   AND a.n_deals = 1
   AND a.n_suppliers = 1
   AND a.supplier_id IS NOT DISTINCT FROM q.supplier_id
 ORDER BY q.quote_id
"""
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/test_quote_rounds_parity.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Break it on purpose, then restore**

Temporarily change `_VERSION_ORDINAL` to drop the `::int` cast (compare as text). Re-run. `test_a_two_digit_round_outranks_a_single_digit_one` must FAIL. Restore the cast and confirm green again. A guard only seen green has not been tested.

- [ ] **Step 6: Commit**

```bash
export GIT_INDEX_FILE=$(mktemp /tmp/idx.XXXXXX)
git read-tree HEAD
git add src/services/quote_rounds.py tests/test_quote_rounds_parity.py
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p HEAD -m 'feat(quote-rounds): one definition of a round, in SQL and in Python

version_collapse already answered "which round is this" for clustering. The
deal overview and the attach pass need the same answer inside SQL, and two
implementations of one rule drift into a deal counting two rounds of one
negotiation as two competing bids. Both now live in one module with a parity
test over the shapes the corpus really produces.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>')
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/quote_rounds.py tests/test_quote_rounds_parity.py
```

---

### Task 2: Attach quote revisions to their base version's deal

**Files:**
- Modify: `src/services/deal_assignment_service.py` (add `_attach_quote_revisions`, call it from `_run` at line ~1147, add its key to the returned dict)
- Test: `tests/services/test_quote_revision_attach_live.py`

**Interfaces:**
- Consumes: `src.services.quote_rounds.REVISION_CANDIDATES_SQL` (Task 1)
- Produces: `_attach_quote_revisions(cur) -> int`; `_run()` gains the result key `quote_revisions_attached`

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_quote_revision_attach_live.py`:

```python
"""A quote's other rounds belong on the deal that quote's own round formed.

Four quotes extracted correctly on 2026-09-23 -- including ORB-Q-6612 (V3), the
awarded best-and-final offer at GBP 1,096,000 on TESTDEAL2026072901 -- reached
proc.bp_quote_trgt with deal_id NULL. They were on no deal, on no screen and in
no queue, and stayed that way for a week.

Every test runs inside a transaction that is rolled back, on synthetic ids that
cannot collide with corpus data.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_quote_revision_attach_live.py
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from src.services.db import get_conn
from src.services.deal_assignment_service import _attach_quote_revisions

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_P = "ZZTEST-QR"   # synthetic prefix; nothing in the corpus starts with this


def _insert(cur, quote_id, *, deal_id=None, deal_name=None, supplier_id="SUP-ZZTest",
            award_status=None, document_id=None, total=1000.00):
    cur.execute(
        "INSERT INTO proc.bp_quote_trgt "
        "(quote_id, deal_id, deal_name, document_id, supplier_id, award_status, "
        " total_amount, currency, quote_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,'GBP',DATE '2025-01-01')",
        (quote_id, deal_id, deal_name, document_id, supplier_id, award_status, total))


def _read(cur, quote_id):
    cur.execute("SELECT deal_id, deal_name, document_id, award_status "
                "FROM proc.bp_quote_trgt WHERE quote_id = %s", (quote_id,))
    return cur.fetchone()


@pytest.fixture
def cur():
    with get_conn() as conn:
        conn.autocommit = False          # get_conn is autocommit; rollback needs this
        try:
            yield conn.cursor()
        finally:
            conn.rollback()


def test_a_stranded_revision_joins_its_base_versions_deal(cur):
    _insert(cur, f"{_P}-A", deal_id="ZZDEAL-1", deal_name="Zed")
    _insert(cur, f"{_P}-A (V2)")

    assert _attach_quote_revisions(cur) == 1

    deal_id, deal_name, document_id, _ = _read(cur, f"{_P}-A (V2)")
    assert deal_id == "ZZDEAL-1"
    assert deal_name == "Zed"
    assert document_id == f"ZZDEAL-1::quote::{_P}-A (V2)"


def test_the_deal_may_hold_the_later_round_and_the_earlier_ones_be_stranded(cur):
    """The majority case on real data: 10 of the 14 look like this.

    Fails if the rule is implemented as 'child joins parent' rather than as a
    match on base reference.
    """
    _insert(cur, f"{_P}-B (V3)", deal_id="ZZDEAL-2", deal_name="Zed Two")
    _insert(cur, f"{_P}-B")
    _insert(cur, f"{_P}-B (V2)")

    assert _attach_quote_revisions(cur) == 2

    assert _read(cur, f"{_P}-B")[0] == "ZZDEAL-2"
    assert _read(cur, f"{_P}-B (V2)")[0] == "ZZDEAL-2"


def test_two_candidate_deals_is_a_coin_toss_so_it_holds(cur):
    _insert(cur, f"{_P}-C", deal_id="ZZDEAL-3", deal_name="Three")
    _insert(cur, f"{_P}-C (V2)", deal_id="ZZDEAL-4", deal_name="Four")
    _insert(cur, f"{_P}-C (V3)")

    assert _attach_quote_revisions(cur) == 0
    assert _read(cur, f"{_P}-C (V3)")[0] is None


def test_a_different_supplier_is_a_collision_not_a_revision(cur):
    _insert(cur, f"{_P}-D", deal_id="ZZDEAL-5", deal_name="Five",
            supplier_id="SUP-ZZOne")
    _insert(cur, f"{_P}-D (V2)", supplier_id="SUP-ZZTwo")

    assert _attach_quote_revisions(cur) == 0
    assert _read(cur, f"{_P}-D (V2)")[0] is None


def test_an_attached_round_is_not_marked_not_awarded(cur):
    """A later round of the deal's own quote is part of the transaction.

    A rival bid gets award_status='not_awarded' to keep it out of
    bp_deal_documents. A revision must not, or the deal loses its own quote.
    """
    _insert(cur, f"{_P}-E", deal_id="ZZDEAL-6", deal_name="Six")
    _insert(cur, f"{_P}-E (V2)")

    _attach_quote_revisions(cur)

    assert _read(cur, f"{_P}-E (V2)")[3] is None
```

- [ ] **Step 2: Run the tests and watch them fail**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_quote_revision_attach_live.py -v
```

Expected: 5 errors, `ImportError: cannot import name '_attach_quote_revisions'`.

- [ ] **Step 3: Add the import**

In `src/services/deal_assignment_service.py`, after the existing `from src.services.linking_engine import (...)` block, add:

```python
from src.services.quote_rounds import REVISION_CANDIDATES_SQL
```

- [ ] **Step 4: Write the function**

Add to `src/services/deal_assignment_service.py`, immediately **above** `def _attach_rival_quotes(cur) -> int:`:

```python
def _attach_quote_revisions(cur) -> int:
    """Put a quote's other rounds on the deal that quote's own round formed.

    'ORB-Q-6612 (V3)' and 'ORB-Q-6612' are not two documents that resemble each
    other. They carry one quote number and one supplier, and the suffix is a round
    marker this codebase mints itself (context_layer.canonical_quote_revision). So
    this reads an identity that is already recorded rather than inferring a link
    from supplier and amount -- the mechanism behind the open mis-grouping bug.

    Nine quotes uploaded on 2026-07-29 became five rows because three V3/BAFO
    files were written under their V1's quote number and last-write-wins kept one.
    The extraction side of that was fixed; the rounds it then recovered reached
    _trgt with deal_id NULL and belonged to nothing. This is the other half.

    Unlike a rival bid (_attach_rival_quotes), a later round is part of the
    transaction, so award_status is NOT forced to 'not_awarded'. It is inherited
    from the anchor instead: where a base's deal-bearing rows are all losing bids,
    its other rounds lost too, and must not slip into the deal's quote_total
    through the back door. Where the anchors disagree, it is left NULL rather
    than guessed.

    Counting every round would treble-count one negotiation. bp_deal_overview
    handles that by counting only the latest round -- not by hiding the earlier
    ones, which stay listed as the deal's negotiation history.
    """
    cur.execute(
        f"""
        UPDATE proc.bp_quote_trgt q
           SET deal_id      = c.candidate_deal_id,
               deal_name    = c.candidate_deal_name,
               document_id  = c.candidate_deal_id || '::quote::' || q.quote_id,
               award_status = c.candidate_award_status
          FROM ({REVISION_CANDIDATES_SQL}) c
         WHERE q.quote_id = c.quote_id
           AND q.deal_id IS NULL
        RETURNING q.quote_id
        """
    )
    attached = [r[0] for r in (cur.fetchall() or [])]
    if not attached:
        return 0
    # The lines follow their header, and only the headers just moved.
    cur.execute(
        """
        UPDATE proc.bp_quote_line_items_trgt l
           SET deal_id = q.deal_id, deal_name = q.deal_name,
               document_id = q.document_id
          FROM proc.bp_quote_trgt q
         WHERE l.quote_id = q.quote_id
           AND l.quote_id = ANY(%s)
        """,
        (attached,),
    )
    return len(attached)
```

- [ ] **Step 5: Run the tests and watch them pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_quote_revision_attach_live.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Break each guard on purpose**

One at a time, restore after each:

| Remove from `REVISION_CANDIDATES_SQL` | Test that must go RED |
|---|---|
| `AND a.n_deals = 1` | `test_two_candidate_deals_is_a_coin_toss_so_it_holds` |
| `AND a.supplier_id IS NOT DISTINCT FROM q.supplier_id` | `test_a_different_supplier_is_a_collision_not_a_revision` |

If either stays green with its condition removed, the test is not testing it — fix the test before continuing.

- [ ] **Step 7: Wire it into the run**

In `src/services/deal_assignment_service.py`, in `_run()`, add the call **before** `_attach_rival_quotes` and add the key to the returned dict:

```python
    # Before the rival pass: a revision that gets its deal here is no longer
    # deal_id NULL, so the rival pass correctly skips it. A later round of the
    # deal's own quote is not a rival bid.
    revisions = _attach_quote_revisions(cur)
    rivals = _attach_rival_quotes(cur)
```

and in the return dict, after `"propagated": prop,`:

```python
            "quote_revisions_attached": revisions,
```

- [ ] **Step 8: Verify the wiring reports a number**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.db import get_conn
from src.services.deal_assignment_service import _run
with get_conn() as conn:
    conn.autocommit = False
    try:
        print(_run(conn.cursor(), include_look_back=False)['quote_revisions_attached'])
    finally:
        conn.rollback()
"
```

Expected: `14`. Rolled back, so nothing is written yet.

- [ ] **Step 9: Add the Review Focus tests**

Append to `tests/services/test_quote_revision_attach_live.py`:

```python
def test_a_round_of_a_losing_bid_does_not_become_a_counted_quote(cur):
    """Review Focus 1.

    bp_deal_documents excludes award_status='not_awarded', which is how a rival
    bid stays out of the deal's quote_total. The attach pass reads bp_quote_trgt
    directly, so without inheritance it would attach a sibling round with NULL
    award_status -- and a losing bid would land in the deal's money via a door
    its own siblings are shut out of.
    """
    _insert(cur, f"{_P}-F", deal_id="ZZDEAL-7", deal_name="Seven",
            award_status="not_awarded")
    _insert(cur, f"{_P}-F (V2)")

    _attach_quote_revisions(cur)

    assert _read(cur, f"{_P}-F (V2)")[3] == "not_awarded"


def test_anchors_disagreeing_on_the_name_leave_the_name_null(cur):
    """Review Focus 4. One deal_id, two deal_names: do not invent a third answer."""
    _insert(cur, f"{_P}-G", deal_id="ZZDEAL-8", deal_name="Eight")
    _insert(cur, f"{_P}-G (V2)", deal_id="ZZDEAL-8", deal_name="Eight renamed")
    _insert(cur, f"{_P}-G (V3)")

    assert _attach_quote_revisions(cur) == 1

    deal_id, deal_name, _, _ = _read(cur, f"{_P}-G (V3)")
    assert deal_id == "ZZDEAL-8"
    assert deal_name is None


def test_a_stale_document_id_is_overwritten(cur):
    """Review Focus 5. Partial prior state must not survive the attach."""
    _insert(cur, f"{_P}-H", deal_id="ZZDEAL-9", deal_name="Nine")
    _insert(cur, f"{_P}-H (V2)", document_id="STALE::quote::whatever")

    _attach_quote_revisions(cur)

    assert _read(cur, f"{_P}-H (V2)")[2] == f"ZZDEAL-9::quote::{_P}-H (V2)"


def test_an_id_with_an_apostrophe_survives_intact(cur):
    """Review Focus 3. document_id is minted by || and the base by regexp_replace."""
    _insert(cur, f"{_P}-O'Brien", deal_id="ZZDEAL-10", deal_name="Ten")
    _insert(cur, f"{_P}-O'Brien (V2)")

    assert _attach_quote_revisions(cur) == 1
    assert _read(cur, f"{_P}-O'Brien (V2)")[2] == f"ZZDEAL-10::quote::{_P}-O'Brien (V2)"
```

- [ ] **Step 10: Run the full file**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_quote_revision_attach_live.py -v
```

Expected: 9 passed. If `test_a_round_of_a_losing_bid_does_not_become_a_counted_quote` fails, `candidate_award_status` is not being applied — fix Task 1's SQL or Step 4's `SET` clause, not the test.

- [ ] **Step 11: Commit**

```bash
export GIT_INDEX_FILE=$(mktemp /tmp/idx.XXXXXX)
git read-tree HEAD
git add src/services/deal_assignment_service.py tests/services/test_quote_revision_attach_live.py
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p HEAD -m 'feat(deals): a quote'"'"'s other rounds join the deal its own round formed

Four quotes extracted correctly on 2026-09-23 -- including ORB-Q-6612 (V3), the
awarded BAFO at GBP 1,096,000 -- reached _trgt with deal_id NULL and belonged to
nothing. 14 rows across four deals are in that state.

Matches on base reference, so it works whether the deal holds the opening bid or
the latest round; 10 of the 14 are the latter. Holds where two deals or two
suppliers share a base. Inherits the anchor'"'"'s award_status, so a losing bid
cannot reach a deal'"'"'s quote_total through a sibling round.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>')
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/deal_assignment_service.py tests/services/test_quote_revision_attach_live.py
```

---

### Task 3: Count a negotiation once, but date it from its opening bid

**Files:**
- Create: `scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql`
- Test: `tests/services/test_deal_overview_quote_rounds_live.py`

**Interfaces:**
- Consumes: `src.services.quote_rounds` (Task 1) — the SQL expressions are repeated literally in the migration, and Task 1's parity test is what keeps them honest
- Produces: `proc.bp_deal_overview` with unchanged column names and types; `quote_count`, `quote_total` and `converted_total_usd` now dedupe rounds

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_deal_overview_quote_rounds_live.py`:

```python
"""A negotiation counts once, but is dated from the bid that opened it.

Counting V1, V2 and V3 of one negotiation as three quotes inflates quote_count
and quote_total. collapse_versions has existed for exactly this reason since the
clustering work; the deal overview never applied it.

The dates are the other half, and they go the OTHER way (Nick's ruling,
2026-09-30): a sourcing event starts when the first bid arrives, so
first_activity_date and cycle_days_quote_to_po read every round. Only the money
dedupes.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_deal_overview_quote_rounds_live.py
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from src.services.db import get_conn

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_D = "ZZDEAL-OVERVIEW"
_P = "ZZTEST-OV"


@pytest.fixture
def cur():
    with get_conn() as conn:
        conn.autocommit = False
        try:
            yield conn.cursor()
        finally:
            conn.rollback()


def _quote(cur, quote_id, amount, date, *, deal=_D, usd=None):
    cur.execute(
        "INSERT INTO proc.bp_quote_trgt "
        "(quote_id, deal_id, deal_name, supplier_id, total_amount, "
        " converted_amount_usd, currency, quote_date) "
        "VALUES (%s,%s,'Zed Overview','SUP-ZZTest',%s,%s,'GBP',%s)",
        (quote_id, deal, amount, usd if usd is not None else amount, date))


def _po(cur, po_id, amount, date):
    cur.execute(
        "INSERT INTO proc.bp_purchase_order_trgt "
        "(po_id, deal_id, deal_name, supplier_name, total_amount, "
        " converted_amount_usd, currency, order_date) "
        "VALUES (%s,%s,'Zed Overview','ZZ Supplier',%s,%s,'GBP',%s)",
        (po_id, _D, amount, amount, date))


def _overview(cur):
    cur.execute(
        "SELECT quote_count, quote_total, converted_total_usd, "
        "       first_activity_date, cycle_days_quote_to_po "
        "  FROM proc.bp_deal_overview WHERE deal_id = %s", (_D,))
    return cur.fetchone()


def test_three_rounds_of_one_quote_count_once_at_the_latest_round(cur):
    _quote(cur, f"{_P}-A", 1000, "2025-03-15")
    _quote(cur, f"{_P}-A (V2)", 900, "2025-04-01")
    _quote(cur, f"{_P}-A (V3)", 800, "2025-04-19")

    count, total, usd, _, _ = _overview(cur)

    assert count == 1
    assert float(total) == 800.0
    assert float(usd) == 800.0


def test_every_round_is_still_listed_on_the_deal(cur):
    """The rounds stop counting; they do not stop existing."""
    _quote(cur, f"{_P}-B", 1000, "2025-03-15")
    _quote(cur, f"{_P}-B (V2)", 900, "2025-04-01")
    _quote(cur, f"{_P}-B (V3)", 800, "2025-04-19")

    cur.execute("SELECT count(*) FROM proc.bp_deal_documents "
                " WHERE deal_id = %s AND doc_type = 'quote'", (_D,))
    assert cur.fetchone()[0] == 3


def test_two_distinct_quotes_are_not_collapsed_into_one(cur):
    _quote(cur, f"{_P}-C", 1000, "2025-03-15")
    _quote(cur, f"{_P}-D", 2000, "2025-03-16")

    count, total, _, _, _ = _overview(cur)

    assert count == 2
    assert float(total) == 3000.0


def test_the_cycle_runs_from_the_opening_bid(cur):
    """Nick's ruling, 2026-09-30.

    Goes RED if the dedupe is applied to the date aggregates as well as the
    money ones -- which is what the first draft of the spec called for, so this
    is the guard against implementing the superseded design. From the surviving
    round the cycle would read 17 days; from the opening bid it is 52.
    """
    _quote(cur, f"{_P}-E", 1000, "2025-03-15")
    _quote(cur, f"{_P}-E (V2)", 900, "2025-04-19")
    _po(cur, f"{_P}-PO", 900, "2025-05-06")

    _, _, _, first_activity, cycle = _overview(cur)

    assert str(first_activity) == "2025-03-15"
    assert cycle == 52


def test_a_deal_holding_only_the_latest_round_is_unchanged_when_earlier_ones_arrive(cur):
    """The DEALV3-77 case: attaching history must not move the money."""
    _quote(cur, f"{_P}-F (V3)", 800, "2025-04-19")
    before = _overview(cur)

    _quote(cur, f"{_P}-F", 1000, "2025-03-15")
    _quote(cur, f"{_P}-F (V2)", 900, "2025-04-01")
    after = _overview(cur)

    assert before[0] == after[0] == 1
    assert float(before[1]) == float(after[1]) == 800.0
    # ...but the deal is now correctly dated from the opening bid.
    assert str(before[3]) == "2025-04-19"
    assert str(after[3]) == "2025-03-15"
```

- [ ] **Step 2: Run the tests and watch them fail**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_deal_overview_quote_rounds_live.py -v
```

Expected: `test_three_rounds_of_one_quote_count_once_at_the_latest_round` fails with `count == 3`, `test_a_deal_holding_only_the_latest_round...` fails, and `test_every_round_is_still_listed_on_the_deal`, `test_two_distinct_quotes...` and `test_the_cycle_runs_from_the_opening_bid` PASS (today's view already behaves correctly on those three). Record which passed — they are the regression guards for this task.

- [ ] **Step 3: Write the migration**

Create `scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql`:

```sql
-- A negotiation counts once, but is dated from the bid that opened it.
--
-- WHY
-- A quote goes through rounds: CPS-Q-3380, then (V2), then (V3 (BAFO)). Those are
-- one negotiation with one supplier, not three competing bids, and counting them
-- as three inflates both quote_count and quote_total. src/services/version_collapse.py
-- has existed for exactly this reason since the clustering work -- its docstring
-- calls collapsing rounds "a prerequisite for correct clustering" -- but the deal
-- overview never applied it.
--
-- This became urgent rather than merely wrong when the stranded rounds were
-- attached to their deals: on TESTDEAL2026072901 that would have taken quote_total
-- from GBP 5,327,600 to GBP 11,796,510, treble-counting three negotiations.
--
-- THE DATES GO THE OTHER WAY
-- Superseded rounds are NOT removed from the aggregate's input. They are marked
-- and then excluded from the MONEY aggregates only, because "what did this deal
-- cost" and "when did this deal start" are different questions. A sourcing event
-- begins when the first bid arrives, so first_activity_date and
-- cycle_days_quote_to_po read every round (Nick's ruling, 2026-09-30).
--
-- The effect is real and visible: DEALV3-77 moves from 17 days to 52 and
-- DEALV3-78 from 13 to 48, because their opening bids were stranded with
-- deal_id NULL and the cycle has been reading five weeks short.
--
-- The regexes mirror src/services/quote_rounds.py, which mirrors version_collapse.
-- tests/test_quote_rounds_parity.py holds the SQL and the Python together; if you
-- change one, that test tells you about the other.
--
-- bp_deal_documents is deliberately NOT changed. Every round stays listed, so the
-- deal screen still shows the full negotiation history. Only the arithmetic moves.

CREATE OR REPLACE VIEW proc.bp_deal_overview AS
WITH d AS (
    SELECT deal_id, deal_name, document_id, doc_type, doc_pk, doc_number,
           doc_date, deal_date, supplier_id, supplier_name, buyer_id, currency,
           amount, amount_incl_tax, converted_amount_usd, country, region,
           confidence_score, status, created_date,
           -- True for the highest round of each (deal, doc_type, base reference).
           -- Non-quote documents have no version suffix, so each is alone in its
           -- partition and is always true.
           row_number() OVER (
               PARTITION BY deal_id, doc_type,
                            regexp_replace(doc_pk, '\s*\(\s*[vV][0-9]+.*\)\s*$', '')
               ORDER BY coalesce(nullif(substring(doc_pk from '\(\s*[vV]([0-9]+)'), '')::int, 1) DESC,
                        doc_date DESC NULLS LAST,
                        doc_pk DESC
           ) = 1 AS is_latest_round
      FROM proc.bp_deal_documents
     WHERE deal_id IS NOT NULL AND deal_id <> ''::text
), agg AS (
    SELECT d.deal_id,
           max(d.deal_name::text)   AS deal_name,
           max(d.supplier_id)       AS supplier_id,
           max(d.supplier_name)     AS supplier_name,
           max(d.buyer_id)          AS buyer_id,
           max(d.deal_date)         AS deal_date,
           -- Every round: the opening bid dates the sourcing event.
           min(d.doc_date)          AS first_activity_date,
           max(d.doc_date)          AS last_activity_date,
           -- Money: the surviving round only.
           count(*) FILTER (WHERE d.doc_type = 'quote'::text AND d.is_latest_round)   AS quote_count,
           count(*) FILTER (WHERE d.doc_type = 'po'::text)                            AS po_count,
           count(*) FILTER (WHERE d.doc_type = 'invoice'::text)                       AS invoice_count,
           sum(d.amount) FILTER (WHERE d.doc_type = 'quote'::text AND d.is_latest_round) AS quote_total,
           sum(d.amount) FILTER (WHERE d.doc_type = 'po'::text)                       AS po_total,
           sum(d.amount) FILTER (WHERE d.doc_type = 'invoice'::text)                  AS invoice_total,
           max(d.currency::text)    AS currency,
           -- Consistent with quote_total, or the two would contradict each other.
           sum(d.converted_amount_usd)
               FILTER (WHERE d.doc_type <> 'quote'::text OR d.is_latest_round)        AS converted_total_usd,
           -- Every round again: measured from the opening bid.
           max(d.doc_date) FILTER (WHERE d.doc_type = 'po'::text)
             - min(d.doc_date) FILTER (WHERE d.doc_type = 'quote'::text)              AS cycle_days_quote_to_po,
           max(d.doc_date) FILTER (WHERE d.doc_type = 'invoice'::text)
             - min(d.doc_date) FILTER (WHERE d.doc_type = 'po'::text)                 AS cycle_days_po_to_invoice
      FROM d
     GROUP BY d.deal_id
)
SELECT deal_id,
       deal_name,
       supplier_id,
       supplier_name,
       buyer_id,
       deal_date,
       first_activity_date,
       last_activity_date,
       quote_count,
       po_count,
       invoice_count,
       quote_total,
       po_total,
       invoice_total,
       currency,
       converted_total_usd,
       quote_count > 0 AND po_count > 0 AND invoice_count > 0 AND po_total > 0::numeric
         AND (abs(COALESCE(invoice_total, 0::numeric) - po_total) / NULLIF(po_total, 0::numeric)) <= 0.10
           AS three_way_match,
       CASE WHEN po_total > 0::numeric
            THEN round(100.0 * abs(COALESCE(invoice_total, 0::numeric) - po_total) / NULLIF(po_total, 0::numeric), 2)
            ELSE NULL::numeric
       END AS price_variance_pct,
       cycle_days_quote_to_po,
       cycle_days_po_to_invoice,
       quote_count > 0 AS has_quote_anchor,
       quote_count = 0 AND (po_count > 0 OR invoice_count > 0) AS orphaned
  FROM agg;
```

- [ ] **Step 4: Snapshot every deal's figures before applying**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.services.db import get_conn
with get_conn() as c:
    cur=c.cursor()
    cur.execute('select deal_id, quote_count, quote_total, converted_total_usd, first_activity_date, cycle_days_quote_to_po from proc.bp_deal_overview order by 1')
    json.dump([[str(x) for x in r] for r in cur.fetchall()], open('/tmp/overview_before.json','w'))
print('snapshot written')
"
```

- [ ] **Step 5: Apply the migration**

```bash
./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.db import get_conn
sql = open('scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql').read()
with get_conn() as c:
    c.cursor().execute(sql)
print('applied')
"
```

- [ ] **Step 6: Run the tests and watch them pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_deal_overview_quote_rounds_live.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Diff the snapshot — the blast radius must be exactly two deals**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.services.db import get_conn
before = {r[0]: r for r in json.load(open('/tmp/overview_before.json'))}
with get_conn() as c:
    cur=c.cursor()
    cur.execute('select deal_id, quote_count, quote_total, converted_total_usd, first_activity_date, cycle_days_quote_to_po from proc.bp_deal_overview order by 1')
    for row in cur.fetchall():
        now=[str(x) for x in row]
        was=before.get(now[0])
        if was and was!=now: print('CHANGED', was, '->', now)
"
```

Expected: exactly two lines, `TESTDEAL2026072901` (5→3, 5327600→3190600) and `TESTDATA_3007262026073025` (6→3, 1164750→574396). Note the Test Deal total here is **3,190,600, not 3,110,000** — Task 2's attach has not been committed to the database yet, so only the dedupe has happened. The 3,110,000 figure arrives in Task 5.

- [ ] **Step 8: Break it on purpose**

Temporarily change `min(d.doc_date)` in `first_activity_date` to
`min(d.doc_date) FILTER (WHERE d.doc_type <> 'quote' OR d.is_latest_round)`, re-apply, and re-run. `test_the_cycle_runs_from_the_opening_bid` must go RED. Restore the migration, re-apply, confirm green.

- [ ] **Step 9: Commit**

```bash
export GIT_INDEX_FILE=$(mktemp /tmp/idx.XXXXXX)
git read-tree HEAD
git add scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql tests/services/test_deal_overview_quote_rounds_live.py
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p HEAD -m 'feat(deals): a negotiation counts once, dated from its opening bid

Three rounds of one quote were three competing bids to bp_deal_overview. With
the stranded rounds attached that would have taken TESTDEAL2026072901 from GBP
5,327,600 to GBP 11,796,510.

Only the money dedupes. first_activity_date and cycle_days_quote_to_po read
every round, because a sourcing event starts when the first bid arrives -- which
moves DEALV3-77 from 17 days to 52 and DEALV3-78 from 13 to 48, correcting a
cycle that read five weeks short while the opening bids sat on no deal.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>')
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- scripts/migrations/2026-09-30-deal-overview-quote-rounds.sql tests/services/test_deal_overview_quote_rounds_live.py
```

---

### Task 4: Report what is still unattached

**Files:**
- Modify: `src/services/linking_engine.py` (add `unattached_documents` at the end, beside `review_queue`)
- Modify: `src/api/routers/promotion.py` (import it; add the route after `get_review_queue`, line ~100)
- Test: `tests/services/test_unattached_documents_live.py`

**Interfaces:**
- Consumes: `src.services.quote_rounds.REVISION_CANDIDATES_SQL` (Task 1); `linking_engine._rows`, `linking_engine.get_conn` (existing)
- Produces: `unattached_documents(conn: Any = None) -> dict` returning `{"counts": {"quote": int, "invoice": int, "po": int, "total": int}, "items": list[dict]}`

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_unattached_documents_live.py`:

```python
"""A document on no deal must be reported, even when nothing can be done about it.

review_queue lists documents the promotion gate HELD. Four quotes promoted
cleanly and then landed with deal_id NULL, so they were held by nothing, listed
by nothing, and invisible for a week. bp_deal_orphans does not help: it covers
po and invoice only, and reads through bp_deal_documents, which a NULL deal_id
already excludes.

items carries only what a person can act on. counts states the whole number, so
the 5,536 standalone documents are visible as a figure without burying the few
that matter -- the same argument /promotion/link-proposals already makes with
its `considered` field.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_unattached_documents_live.py
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from src.services.db import get_conn
from src.services.linking_engine import unattached_documents

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_P = "ZZTEST-UN"


@pytest.fixture
def conn():
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _quote(cur, quote_id, *, deal_id=None, supplier_id="SUP-ZZTest"):
    cur.execute(
        "INSERT INTO proc.bp_quote_trgt "
        "(quote_id, deal_id, deal_name, supplier_id, total_amount, currency, quote_date) "
        "VALUES (%s,%s,%s,%s,500,'GBP',DATE '2025-01-01')",
        (quote_id, deal_id, "Zed Un" if deal_id else None, supplier_id))


def test_an_actionable_document_appears_in_items(conn):
    cur = conn.cursor()
    _quote(cur, f"{_P}-A", deal_id="ZZDEAL-UN-1")
    _quote(cur, f"{_P}-A (V2)")

    result = unattached_documents(conn=conn)

    hit = [i for i in result["items"] if i["doc_pk"] == f"{_P}-A (V2)"]
    assert len(hit) == 1
    assert hit[0]["doc_type"] == "quote"
    assert hit[0]["candidate_deal_id"] == "ZZDEAL-UN-1"
    assert hit[0]["evidence"]


def test_a_standalone_document_is_counted_but_not_itemised(conn):
    cur = conn.cursor()
    before = unattached_documents(conn=conn)["counts"]["quote"]
    _quote(cur, f"{_P}-LONELY")          # no sibling anywhere
    after = unattached_documents(conn=conn)

    assert after["counts"]["quote"] == before + 1
    assert not [i for i in after["items"] if i["doc_pk"] == f"{_P}-LONELY"]


def test_counts_cover_all_three_document_types_and_total(conn):
    result = unattached_documents(conn=conn)
    counts = result["counts"]

    assert set(counts) == {"quote", "invoice", "po", "total"}
    assert counts["total"] == counts["quote"] + counts["invoice"] + counts["po"]
    assert all(isinstance(v, int) for v in counts.values())


def test_no_field_names_an_internal_route_or_table(conn):
    """Output-safety: a JSON field naming an internal route or table comes back
    '[withheld]' from the gateway, so ids only -- never URLs, never table names."""
    result = unattached_documents(conn=conn)
    blob = repr(result).lower()

    for banned in ("proc.bp_", "/promotion", "_trgt", "_stg"):
        assert banned not in blob, banned
```

- [ ] **Step 2: Run the tests and watch them fail**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_unattached_documents_live.py -v
```

Expected: 4 errors, `ImportError: cannot import name 'unattached_documents'`.

- [ ] **Step 3: Add the reader**

At the top of `src/services/linking_engine.py`, add to the imports:

```python
from src.services.quote_rounds import REVISION_CANDIDATES_SQL
```

Append to `src/services/linking_engine.py`, after `review_queue`:

```python
_UNATTACHED_COUNTS_SQL = """
SELECT (SELECT count(*) FROM proc.bp_quote_trgt          WHERE deal_id IS NULL) AS quote,
       (SELECT count(*) FROM proc.bp_invoice_trgt        WHERE deal_id IS NULL) AS invoice,
       (SELECT count(*) FROM proc.bp_purchase_order_trgt WHERE deal_id IS NULL) AS po
"""


def unattached_documents(conn: Any = None) -> dict:
    """Documents that reached the final tier and belong to no deal.

    This is the alarm that did not exist. `review_queue` lists documents the
    promotion gate HELD; a document that promoted cleanly and then landed with
    deal_id NULL was held by nothing and listed by nothing. `bp_deal_orphans`
    does not cover it either -- po and invoice only, read through
    bp_deal_documents, which a NULL deal_id has already excluded.

    `items` carries ONLY documents with a provable candidate deal. That is a
    deliberate choice, not an oversight: 5,536 of the 5,550 are standalone
    documents with no deal partner to attach to, and a list containing all of
    them would bury the few that are real defects and be ignored within a week.

    `counts` states the whole number so nothing is hidden. Same argument
    /promotion/link-proposals already makes with its `considered` field: a bare
    empty list reads as "everything is attached" when the truth is "5,536 are
    not, and none of them can be".

    In steady state, once the attach pass has run, `items` is empty. A non-empty
    `items` means the attach pass has something it could not take.

    Read-only. Nothing is written by this call.
    """
    if conn is None:
        with get_conn() as own:
            return _unattached_documents(own)
    return _unattached_documents(conn)


def _unattached_documents(conn) -> dict:
    cur = conn.cursor()
    counts = _rows(cur, _UNATTACHED_COUNTS_SQL)[0]
    counts = {k: int(v or 0) for k, v in counts.items()}
    counts["total"] = counts["quote"] + counts["invoice"] + counts["po"]

    items = []
    for row in _rows(cur, REVISION_CANDIDATES_SQL):
        items.append({
            "doc_type": "quote",
            "doc_pk": row["quote_id"],
            "supplier_id": row["supplier_id"],
            "amount": _to_float(row.get("total_amount")),
            "currency": row.get("currency"),
            "candidate_deal_id": row["candidate_deal_id"],
            # Plain words, no identifiers a gateway would withhold.
            "evidence": "an earlier or later round of this quote is on that deal",
        })
    return {"counts": counts, "items": items}
```

- [ ] **Step 4: Add the route**

In `src/api/routers/promotion.py`, extend the `linking_engine` import:

```python
from src.services.linking_engine import (
    promote_ready, review_queue, approve_promotion, quote_chains,
    canonicalize_po_references, unattached_documents,
)
```

and add after `get_review_queue`:

```python
@router.get("/unattached", summary="Documents in the final tier that belong to no deal")
def get_unattached() -> dict[str, Any]:
    """Read-only. `items` are the ones a person can act on; `counts` is the whole
    population, including the standalone documents nothing can attach.

    A document that promoted cleanly and then landed on no deal was previously
    reported by nothing at all -- not the review queue, which lists only held
    documents, and not bp_deal_orphans, which covers po and invoice only.
    """
    try:
        return unattached_documents()
    except Exception as exc:
        logger.exception("unattached failed")
        raise HTTPException(status_code=500, detail=str(exc))
```

- [ ] **Step 5: Run the tests and watch them pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_unattached_documents_live.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Check the route is reachable**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.api.routers.promotion import router
print([r.path for r in router.routes if 'unattached' in r.path])
"
```

Expected: `['/promotion/unattached']`.

- [ ] **Step 7: Commit**

```bash
export GIT_INDEX_FILE=$(mktemp /tmp/idx.XXXXXX)
git read-tree HEAD
git add src/services/linking_engine.py src/api/routers/promotion.py tests/services/test_unattached_documents_live.py
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p HEAD -m 'feat(promotion): report documents that reached _trgt and belong to no deal

review_queue lists what the gate HELD. Four quotes promoted cleanly, landed with
deal_id NULL, and were therefore held by nothing and listed by nothing for a
week. bp_deal_orphans covers po and invoice only.

items carries only documents with a provable candidate deal; counts states the
whole population. Listing all 5,550 would bury the 14 that are real defects
among 5,536 standalone documents nothing can ever attach.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>')
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/linking_engine.py src/api/routers/promotion.py tests/services/test_unattached_documents_live.py
```

---

### Task 5: Run it for real and verify on the live server

Tests prove the units. This proves the product. Nothing here is a code change; if a check fails, go back to the task that owns it.

**Files:** none modified.

- [ ] **Step 1: Snapshot every deal, before anything is written**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.services.db import get_conn
with get_conn() as c:
    cur=c.cursor()
    cur.execute('select deal_id, quote_count, quote_total, first_activity_date, cycle_days_quote_to_po from proc.bp_deal_overview order by 1')
    json.dump([[str(x) for x in r] for r in cur.fetchall()], open('/tmp/final_before.json','w'))
    cur.execute('''select (select count(*) from proc.bp_quote_trgt where deal_id is null),
                          (select count(*) from proc.bp_invoice_trgt where deal_id is null),
                          (select count(*) from proc.bp_purchase_order_trgt where deal_id is null)''')
    print('deal-less before:', cur.fetchone())
"
```

Expected: `deal-less before: (3395, 2154, 1)`.

- [ ] **Step 2: Run deal assignment for real (committed, not rolled back)**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.deal_assignment_service import assign_deals
print(assign_deals())
"
```

Expected: the result dict contains `'quote_revisions_attached': 14`.

- [ ] **Step 3: The four stranded quotes are on Test Deal**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.db import get_conn
with get_conn() as c:
    cur=c.cursor()
    cur.execute('''select doc_pk, supplier_name, amount from proc.bp_deal_documents
                    where deal_id=%s and doc_type='quote' order by doc_pk''', ('TESTDEAL2026072901',))
    for r in cur.fetchall(): print(r)
"
```

Expected: 9 quote rows, including `ORB-Q-6612 (V3)` at 1096000.00 attributed to Orbis Platform Solutions Ltd.

- [ ] **Step 4: Test Deal's figures**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.db import get_conn
with get_conn() as c:
    cur=c.cursor()
    cur.execute('''select deal_id, quote_count, quote_total, first_activity_date, cycle_days_quote_to_po
                     from proc.bp_deal_overview
                    where deal_id in ('TESTDEAL2026072901','TESTDATA_3007262026073025','DEALV3-76','DEALV3-77','DEALV3-78')
                    order by 1''')
    for r in cur.fetchall(): print(r)
"
```

Expected, all five lines:

| deal | quote_count | quote_total | first_activity | cycle |
|---|---|---|---|---|
| `DEALV3-76` | unchanged | unchanged | 2025-04-08 | NULL |
| `DEALV3-77` | unchanged | unchanged | **2025-03-15** | **52** |
| `DEALV3-78` | unchanged | unchanged | **2025-03-13** | **48** |
| `TESTDATA_3007262026073025` | 3 | 574396.00 | 2025-03-04 | 55 |
| `TESTDEAL2026072901` | 3 | **3110000.00** | 2024-03-08 | 54 |

If `DEALV3-77`'s cycle is still 17, the opening-bid ruling is not in effect — Task 3.
If its `quote_total` moved, the dedupe is keeping the wrong round — Task 3.

- [ ] **Step 5: Nothing else moved**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys, json; sys.path.insert(0,'.')
from src.services.db import get_conn
before={r[0]:r for r in json.load(open('/tmp/final_before.json'))}
changed=[]
with get_conn() as c:
    cur=c.cursor()
    cur.execute('select deal_id, quote_count, quote_total, first_activity_date, cycle_days_quote_to_po from proc.bp_deal_overview order by 1')
    for row in cur.fetchall():
        now=[str(x) for x in row]
        if before.get(now[0]) and before[now[0]]!=now: changed.append(now[0])
print('deals changed:', changed)
"
```

Expected exactly: `['DEALV3-77', 'DEALV3-78', 'TESTDATA_3007262026073025', 'TESTDEAL2026072901']`. Any other deal id here is a regression — stop and investigate before continuing.

- [ ] **Step 6: The census closed by 14**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.linking_engine import unattached_documents
r = unattached_documents()
print('counts:', r['counts'])
print('items:', len(r['items']))
"
```

Expected: `counts: {'quote': 3381, 'invoice': 2154, 'po': 1, 'total': 5536}` and `items: 0`.

- [ ] **Step 7: The endpoint answers on the running server**

The service is managed by systemd and is already active. Restart it so the new route is loaded, then call it. **Never `pkill -f uvicorn`** — it kills the shared stack.

```bash
sudo systemctl restart procwise
sleep 5
curl -s localhost:8000/promotion/unattached | head -c 400
```

Expected: JSON with `counts.total` 5536 and an empty `items`.

- [ ] **Step 8: Confirm the report screen**

Open the Test Deal in SpendIQ and confirm the deal lists **nine** quotes (the full negotiation history, three rounds each from three suppliers) while the quote total reads £3,110,000. Reload the page after the service restart — a stale engine bundle will show the old figures.

- [ ] **Step 9: Run the wider suite for regressions**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/ -q -p no:randomly 2>&1 | tail -20
```

Compare the failure count against the known baseline of 635 at pristine HEAD (8 collectors are known-broken and need `--ignore`). **The count must not have risen.** Do not run this concurrently with any other suite.

- [ ] **Step 10: Final commit if anything was adjusted**

If steps 1–9 required no code changes, there is nothing to commit and the branch is already complete. If they did, commit through the private index as in earlier tasks and re-run the affected task's tests first.

---

## Self-Review

**Spec coverage** — every section of `specs/2026-09-30-deal-less-documents-design.md` maps to a task:

| Spec section | Task |
|---|---|
| Part 1 — attach quote revisions | Task 2 (rule and SQL in Task 1) |
| Part 2 — latest round counts, money only | Task 3 |
| Part 2 — opening-bid ruling on dates | Task 3, Step 1 `test_the_cycle_runs_from_the_opening_bid` |
| Part 3 — report the unattached | Task 4 |
| Testing 1, 1b, 2, 3, 4, 5 | Task 2, Step 1 |
| Testing 6, 6b, 7, 8 | Task 3, Step 1 |
| Testing 9 (SQL/Python parity) | Task 1, Step 1 |
| Testing 9b (opening bid) | Task 3, Step 1 |
| Testing 10 (endpoint) | Task 4, Step 1 |
| Live verification 1–7 | Task 5 |

**Placeholder scan** — no TBD/TODO; every code step carries the literal code; no step says "similar to Task N".

**Type consistency** — `REVISION_CANDIDATES_SQL` returns `quote_id, supplier_id, total_amount, currency, candidate_deal_id, candidate_deal_name, candidate_award_status`; Task 2 reads `candidate_deal_id`, `candidate_deal_name`, `candidate_award_status` and Task 4 reads `quote_id`, `supplier_id`, `total_amount`, `currency`, `candidate_deal_id`. `_attach_quote_revisions(cur) -> int` is called with a cursor in both the test and `_run`. `unattached_documents(conn=None) -> dict` is called with a connection in tests and without one in the route.

**One addition beyond the spec**, made because writing Task 2's tests surfaced it: the attached round **inherits the anchor's `award_status`**. Without it, a base whose deal-bearing rows are all `not_awarded` rivals would contribute a *counted* quote to that deal through one of its other rounds — entering the money through a door its own siblings are deliberately shut out of. This is Review Focus 1 and is covered by `test_a_round_of_a_losing_bid_does_not_become_a_counted_quote`. The spec should be amended to match.
