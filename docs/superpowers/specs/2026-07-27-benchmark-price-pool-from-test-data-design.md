# Benchmark price pool from test data

**Date:** 2026-07-27
**Status:** design, awaiting approval
**Depends on:** `2026-07-24-full-scope-test-dataset-design.md` (Plan 1 complete at `1db0767`)

## 1. Why

The deterministic benchmark pricing engine (`src/services/benchmark/`, shipped
2026-07-16) is complete and penny-exact against the Excel prototype — 58 tests
pass, including a golden-fixture parity suite over 54 data points. It is wired
to two live endpoints, `GET /benchmark/by-deal/{deal_id}` and
`POST /benchmark/preview`.

It has nothing to work on. The engine needs several comparable historical prices
for the same item before it will produce a number; below that threshold it fails
closed by design. In `bp_sqldb` only one deal has any deal-linked quote line
with matching history, and that history is a single observation of the deal's
own purchase order. Every line gates. The engine has never been exercised at
realistic evidence depth.

The test-data generator already produces exactly the missing raw material, as a
side effect of how it was built: a fixed 5,000-item catalogue
(`scripts/testdata/catalogue.py`) whose items are drawn repeatedly across 6,000
document chains, priced by `price_on()` at 3.8% annual drift plus a
deterministic per-item-per-date noise band of ±4%. Regenerating at seed 42
against the real taxonomy yields 21,020 quotes (115,610 lines), 5,037 purchase
orders (22,537 lines) and 11,908 invoices (53,316 lines).

The price pool the engine reads is the purchase-order and invoice lines:
**75,853 priced observations across 5,000 catalogue items, averaging about 15
per item** — comfortably above the three-point minimum the engine requires, with
three and a half years of drift between them. The 115,610 quote lines are the
things being benchmarked, not pool members.

## 2. The gap

Those documents are never written to a database. `scripts/testdata/build.py`
generates the chains, plants the defects, writes the answer key, and then
persists **only** suppliers and the supplier crosswalk (`build.py:132-133`).
All 37,965 documents and their 191,463 lines are checksummed in memory and then
discarded. `docs/testdata/BUILD_LOG.md` records this as known gap 1 and defers
persistence to a Plan 3 that does not exist yet. (The log's own counts —
44,455 documents, 193,568 lines — predate the D05 fix in `1db0767` and no longer
reproduce; the figures used throughout this spec were regenerated at that
commit.)

So the work here is not "generate price points" — the price points already
exist. It is "persist the documents that already contain them".

**Current database state (verified 2026-07-27 10:22 UTC):** `bp_testdb` holds
111 tables and zero rows; `proc.bp_supplier_id_crosswalk` is absent.
`uicanvas_test` holds `proc.supplier` at 5,000 rows plus the reference tables.
`BUILD_LOG.md` describes a fuller state that no longer exists — `bp_testdb` was
re-cloned after the recorded run and not repopulated. A clean rebuild is a
prerequisite for every acceptance criterion below.

## 3. Scope

**In scope:** persist quote, purchase-order and invoice headers and their line
items into the target tables; assign deal identifiers using the real service;
correct two data defects that would otherwise corrupt the pool; extend
verification to cover the new data.

**Out of scope, deliberately:** the four adjustment factors the engine supports
but the corpus cannot feed — specification score, service-level score, location
cost index, and a dated price index. `src/services/benchmark_live.py:32`
neutralises all four today by supplying identical values to both sides of each
comparison, and discloses that it has done so. That behaviour stays exactly as
it is, so the test database exercises the engine the same way live data does.
Volume is the only live adjustment, and quantities in the generated data are
real. Generating the other four dimensions is a separate decision with its own
spec; it would make the test database diverge from live behaviour, which is a
cost worth paying only once someone needs those factors demonstrated.

**Also out of scope:** the organisation tables, downstream rankings and
decisions, golden document files, and scenario harnesses. Those belong to Plans
2 and 3.

## 4. Design

### 4.1 `scripts/testdata/persist.py` — new module

Maps the existing `Document` and `LineItem` dataclasses from `documents.py`
onto six existing tables, and bulk-loads them through the `copy_rows` helper
already in `scripts/testdata/db.py`.

| Generated | Target table |
|---|---|
| `Document(doc_type="Quote")` | `proc.bp_quote_trgt` |
| its `lines` | `proc.bp_quote_line_items_trgt` |
| `Document(doc_type="Purchase_Order")` | `proc.bp_purchase_order_trgt` |
| its `lines` | `proc.bp_po_line_items_trgt` |
| `Document(doc_type="Invoice")` | `proc.bp_invoice_trgt` |
| its `lines` | `proc.bp_invoice_line_items_trgt` |

Note the purchase-order header table is `bp_purchase_order_trgt`, not
`bp_po_trgt` — the latter does not exist. Only `bp_po_line_items_trgt` follows
the `po` abbreviation.

Interface:

```python
COLUMNS: dict[str, tuple[str, ...]]          # table -> ordered column list
def write_chains(conn, chains: Sequence[Chain]) -> dict[str, int]
```

`write_chains` truncates the six tables before loading, so a rebuild is
idempotent. It returns rows written per table for the build log.

Every column written is populated from a field the generator already produces.
Columns with no generated source are left NULL rather than invented. The one
judgement call worth naming: organisation attribution. The header tables carry
`buyer_id`, `country` and `region` but no business-unit or cost-centre column,
so `buyer_id` takes the cost-centre identifier and `country`/`region` come from
the owning entity. The fuller organisation model has nowhere to land until Plan
3 creates it, and the seeder discards it today regardless.

`created_by` and `last_modified_by` are set to a constant `testdata` marker so
generated rows are distinguishable from anything else by query alone.

Each of the three header tables carries an `AFTER INSERT` trigger
(`trg_quote_trgt_outcome`, `trg_po_trgt_outcome`, `trg_invoice_trgt_outcome`)
that looks the document up in the corresponding `_raw` table and records an
outcome when it finds one. Loading 37,965 headers fires it 37,965 times. The
`_raw` tables stay empty in the test database, so each call finds nothing and
returns immediately, but the cost is measured during implementation rather than
assumed. The triggers are left enabled: they are part of the structure the clone
is supposed to reproduce.

### 4.2 `build.py` wiring

One call to `write_chains` after the supplier write, and its row counts added to
the printed summary.

### 4.3 Deal identifiers

`GET /benchmark/by-deal/{deal_id}` selects quote lines by `deal_id`, so without
deal assignment the endpoint returns nothing even with a full pool. (The pool
query itself does not filter by deal and needs no deal work.)

Deal assignment runs through the existing production service,
`src/services/deal_assignment_service.assign_deals()`, pointed at the test
database after loading. The seeder does not stamp `deal_id` itself. Running the
real grouping code over 6,000 synthetic chains is worth having on its own
merits.

Worth recording, because it contradicts a standing assumption: **the cloned
schema contains no deal-assignment routine.** All ten functions in `bp_testdb`
are outcome, discrepancy or process-monitor related, and none of the nine
triggers touches `deal_id`. If a deal-assigning stored procedure exists
somewhere, it is not part of what `pg_dump --schema-only` brought across from
`bp_sqldb`, so a freshly built test database will not assign deal identifiers on
its own. The Python service is the only mechanism available here.

This is the one step with genuine uncertainty. `assign_deals` derives look-back
deal identifiers from a canonical purchase order using `linking_engine` scores,
and it has never been run at this volume or against data it did not extract
itself. If it groups the chains poorly, that is a product finding about the
grouping service, reported as such — not something the seeder works around by
writing deal identifiers directly. The design document for the test dataset
takes the same position in its §13 note: a red result from a correctly built
test is a finding, not a defect in the test.

### 4.4 Two data corrections

Both are defects in already-committed generator code that would corrupt the
pool. Both are small.

**Description collisions.** The engine pools on item description, unit of
measure and currency. Run against the real 246-leaf taxonomy, `build_catalogue`
produces 50 key combinations shared by 102 of the 5,000 catalogue items: the
description is assembled from a 10-word qualifier list, the leaf name and a
10-word noun list, giving only 100 distinct shapes per leaf for roughly 20 items
per leaf. The worst colliding pair carries base prices of £4.81 and £8,865.99 —
a 1,843-fold spread that would pool as though the two were the same product,
and would drag any benchmark built from that pool into nonsense. Fix: make the
description unique per item. This changes the generated corpus and therefore the
determinism checksum and the answer key; both are regenerated and the new
checksum recorded.

**Currency labels without conversion.** `documents.py:88` stamps the cost
centre's currency on each line, while `unit_price` comes from `price_on()`,
which returns the catalogue's untranslated sterling figure. A US cost centre
therefore records a sterling magnitude labelled USD. Benchmarking survives this,
because currency is a match key and the pools stay separate — but spend roll-up
does not, and check V05 ("FX conversions re-derive") exists to catch exactly
this. Fix: convert through `proc.bp_fx_rates`, which `reference.py` already
copies verbatim. That table is currently empty in `bp_testdb`, which the
prerequisite rebuild resolves.

### 4.5 Verification

- Extend V01 to assert document and line-item counts, not just suppliers.
- Implement V04: line totals sum to their header totals, per document type.
- Implement V05: a converted amount re-derives from the copied FX rate.
- Add one benchmark check: take the most frequently purchased catalogue item,
  run `compute_benchmark` against the loaded pool, and assert the result is not
  gated and reports HIGH confidence.

The last is the acceptance test for this work in a single line: the engine
computes where it previously gated.

### 4.6 What does not change

No file under `src/services/benchmark/` is touched. `benchmark_live.py` already
reads precisely these tables; pointing it at `bp_testdb` is a connection
setting. The neutralised adjustments stay neutralised and stay disclosed.

Pool size was measured rather than assumed: constructing 120,000
`BenchmarkPoint` objects takes 0.41 s, and one `compute_benchmark` call over the
full pool takes 8 ms, so a twenty-line deal costs roughly 0.6 s of engine time
plus the row fetch. No optimisation is warranted, and none is proposed —
narrowing the pool query would risk changing which points match, and the engine's
accuracy is not negotiable for a saving this size.

## 5. Testing

Unit tests over a small chain fixture, no database: `write_chains` produces one
row per document and one per line; column lists match the live schema; totals
survive the mapping; the marker fields are set. Database writes follow the
integration-marked pattern already used in `tests/testdata/test_db.py`.

Regression tests for the two corrections: descriptions are unique across the
full 5,000-item catalogue, and a converted line re-derives from its FX rate.

## 6. Acceptance criteria

1. `python -m scripts.testdata.build --target bp_testdb --seed 42 --drop-first`
   exits 0 and `bp_testdb` holds the suppliers, reference data, documents and
   line items — verified by query, not by the build log.
2. `proc.bp_po_line_items_trgt` and `proc.bp_invoice_line_items_trgt` together
   hold at least 70,000 priced lines (75,853 at seed 42 before defect planting),
   and the median catalogue item has at least three observations in that pool.
3. No two catalogue items share description, unit of measure and currency.
4. V01, V02, V03, V04, V05 and V14 pass; the benchmark check passes.
5. `GET /benchmark/by-deal/{deal_id}` returns non-gated lines at HIGH confidence
   for at least one deal — or, if `assign_deals` fails to group the synthetic
   chains, that failure is documented with its cause and the pool is
   demonstrated directly instead.
6. Two builds at seed 42 in separate processes reproduce one checksum, and
   `BUILD_LOG.md` records the new value.
7. `bp_sqldb` and `uicanvas` are provably unchanged (V14).

## 7. Risks

**`assign_deals` at volume.** Covered in §4.3. The fallback is explicit and does
not involve fabricating deal identifiers.

**Checksum and answer key churn.** The two corrections change the generated
corpus, so the recorded determinism checksum and the 7,070-instance answer key
both change. This is expected; the criterion is that the new values are
reproducible, not that they match the old ones.

**Planted defects interacting with unique descriptions.** The defect planter
mutates chains after generation. Making descriptions unique should not affect
it, but the existing defect regression tests must still pass, and the per-type
instance counts must still hit their declared targets.
