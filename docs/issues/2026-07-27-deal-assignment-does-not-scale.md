# Deal assignment does not survive a realistic corpus

**Raised:** 2026-07-27
**Severity:** High — blocks deal-scoped features on any full-size dataset
**Component:** `src/services/deal_assignment_service.py`, `src/services/linking_engine.py`
**Found by:** the benchmark price-pool work, running against the newly seeded `bp_testdb`

## In plain English

Documents are grouped into "deals" by a background service. That service works fine
today because the live database is small — 34 purchase orders. Against a realistic
volume it does not finish: it issues tens of thousands of unnecessary database
questions and the connection drops before it completes. Nothing gets grouped, so any
screen or endpoint that works per-deal has nothing to show.

This has never been visible before because we have never had enough data to expose it.

## What happens

Running the linking passes against `bp_testdb` (232,023 rows: 5,037 purchase orders,
21,020 quotes, 11,908 invoices, 115,610 quote lines):

```
psycopg2.OperationalError: SSL connection has been closed unexpectedly
  File "src/services/deal_assignment_service.py", line 472, in _look_back
    _persist_deal(cur, dt, dpk, deal_id=deal_id, ...)
  File "src/services/deal_assignment_service.py", line 146, in _persist_deal
    cols = _table_columns(cur, table)
  File "src/services/linking_engine.py", line 458, in _table_columns
    cur.execute(...)
```

The run does not complete and nothing is committed. Afterwards `deal_id` is NULL on
all 115,610 quote lines and all 5,037 purchase orders — no partial state, but no
result either.

## Root cause

`linking_engine._table_columns` (line 456) queries `information_schema.columns` and
**caches nothing**. `deal_assignment_service._persist_deal` (line 145) calls it inside
a loop over four tables — staging header, target header, staging lines, target lines —
for **every document it persists**.

Each deal persists its purchase order, its anchoring quote, and its invoices — roughly
4.4 documents per deal across this corpus. So:

> 5,037 deals × ~4.4 documents × 4 tables ≈ **88,000 `information_schema` queries**,
> all inside a single long-lived transaction.

Every one of those returns the same answer. The schema cannot change during a run.

Against 34 live purchase orders this is roughly 600 queries and finishes instantly,
which is why it has never been noticed.

## Impact

- `GET /benchmark/by-deal/{deal_id}` returns nothing on a full-size dataset.
- Any deal-scoped feature — deal summaries, the pipeline view, supplier ranking, which
  is deal-scoped by design — has no deals to work with.
- The seeded test dataset cannot exercise deal-scoped behaviour at all until this is
  fixed, so a whole class of tests stays unwritten.

## Suggested fix

Cache the column list per table for the lifetime of a run. The schema is fixed during
execution, so a dictionary keyed by table name, populated on first use, removes ~99.99%
of these queries. That alone may be sufficient.

If it is not, the second thing to look at is transaction length: the passes run as one
long transaction, so a mid-run connection drop discards everything. Committing per deal,
or in batches, would make the work resumable rather than all-or-nothing.

Both are contained changes to existing code. Neither requires redesigning the grouping
logic, which is not implicated — the algorithm is fine, the plumbing around it is not.

## Reproducing

```bash
.venv/bin/python -c "
from scripts.testdata.db import connect
import src.services.deal_assignment_service as das
conn = connect('bp_testdb'); conn.autocommit = False
das._run(conn.cursor())"
```

Do **not** call the public `assign_deals()` for this: it finishes by calling
`sync_deal_summaries()`, which opens its own connection from the environment settings —
pointing at the **live** database regardless of the connection passed in. It would
generate summaries over live deals and write them to live `proc.bp_summary`.

## Also worth noting

`_look_forward` returns 0 immediately. It reads `proc.process_monitor`, which the test
seeder leaves empty (0 rows) even after the raw and staging tiers were loaded. That is
correct behaviour for synthetic data — those rows normally come from real uploads — but
it means only the look-back path is exercised by the test dataset. Worth deciding
whether the seeder should populate `process_monitor` so both paths get covered.

## Evidence

Measured on 2026-07-27 against `bp_testdb` (232,023 rows). `_quote_anchor_for_po`
alone costs 0.022s per purchase order — about 1.8 minutes across the corpus — and
that measurement excludes the persist work where the failure occurs, so it is a floor
rather than an estimate of the total.

Full context: `docs/superpowers/specs/2026-07-27-benchmark-price-pool-RESULTS.md`.
