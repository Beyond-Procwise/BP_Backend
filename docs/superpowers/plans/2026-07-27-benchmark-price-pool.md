# Benchmark Price Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the benchmark pricing engine a real price pool by persisting the generated test documents, correct four defects in how it reads the database, and raise extreme prices as review checkpoints in the Action Centre.

**Architecture:** Three strands. The seeder gains a persistence module that maps the existing `Document`/`LineItem` dataclasses onto the six `_trgt` tables and bulk-loads them with the existing `copy_rows` helper. `benchmark_live.py` gets four corrections to its SQL and its point construction. A new `price_outlier` service compares each priced line against the median of comparable purchases and writes findings into `proc.bp_extraction_discrepancy`, which is already what the Action Centre reads.

**Tech Stack:** Python 3.12, psycopg2, pydantic v2, pytest, PostgreSQL. Run everything with `.venv/bin/python` — there is no bare `python` on this machine.

**Spec:** `docs/superpowers/specs/2026-07-27-benchmark-price-pool-from-test-data-design.md`

## Global Constraints

- Run Python as `.venv/bin/python`. `python` does not exist on PATH.
- Tests that import from `src/` need `PYTHONPATH=src`. Tests importing `scripts.testdata.*` run from the repo root with no PYTHONPATH.
- Never target `bp_sqldb`, `uicanvas`, `ses`, `postgres` or `rdsadmin` as a write target. `scripts/testdata/guards.assert_safe_target` enforces this and there is no override.
- Never use Python's built-in `round()` anywhere under `src/services/benchmark/`. It is half-to-even and provably breaks Excel parity. Use `excel_round`.
- New database tables use the `bp_` prefix. This plan creates no new tables.
- Do not put `Co-Authored-By` lines in commit messages.
- All work stays on the `Development` branch.
- Money is `Decimal` quantised to `0.01` with `ROUND_HALF_UP` in the seeder (`documents._money`).
- The engine stays pure: no I/O, no logging of values, no database access inside `src/services/benchmark/engine.py`.
- A GET endpoint must never write findings. Detection runs as a job, not on read.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/testdata/catalogue.py` (modify) | Item descriptions become unique |
| `scripts/testdata/reference.py` (modify) | Load the FX snapshot for price conversion |
| `scripts/testdata/documents.py` (modify) | Convert line prices into the cost centre's currency |
| `scripts/testdata/persist.py` (create) | Map documents onto the six `_trgt` tables and bulk-load |
| `scripts/testdata/build.py` (modify) | Call the persistence step, then deal assignment |
| `scripts/testdata/verify.py` (modify) | V01 document counts, V04 line sums, V05 FX re-derive, benchmark check |
| `src/services/benchmark/models.py` (modify) | `historical_quantity` becomes optional |
| `src/services/benchmark/engine.py` (modify) | Reference-quantity average skips missing quantities |
| `src/services/benchmark_live.py` (modify) | Currency join, own-document exclusion, disclosure counts |
| `src/services/price_outlier/rule.py` (create) | Pure outlier decision rule |
| `src/services/price_outlier/detector.py` (create) | Peer sets from the database, findings out |
| `src/services/price_outlier/__init__.py` (create) | Public surface |
| `src/services/backend_scheduler.py` (modify) | Register the detector job |
| `src/services/benchmark/README.md` (modify) | Record the add-once rule |

---

### Task 1: Unique catalogue item descriptions

Two catalogue items sharing a description, unit and currency pool as if they were the same product. Against the real 246-leaf taxonomy that happens 50 times, and the worst pair spans £4.81 to £8,865.99.

**Files:**
- Modify: `scripts/testdata/catalogue.py:55-70`
- Test: `tests/testdata/test_catalogue.py`

**Interfaces:**
- Consumes: `build_catalogue(seed, leaves, supplier_ids) -> list[CatalogueItem]` (unchanged signature)
- Produces: `CatalogueItem.description` is unique across the returned list

- [ ] **Step 1: Write the failing test**

Add to `tests/testdata/test_catalogue.py`:

```python
def test_descriptions_are_unique_across_the_catalogue():
    """Two items sharing description+uom+currency would pool as one product in
    the benchmark engine, averaging unrelated prices together."""
    items = build_catalogue(42, _leaves(), _supplier_ids())
    keys = [(i.description, i.unit_of_measure, i.currency) for i in items]
    assert len(set(keys)) == len(keys)


def test_description_still_names_its_leaf():
    items = build_catalogue(42, _leaves(), _supplier_ids())
    for item in items[:100]:
        assert item.leaf.l5 in item.description
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_catalogue.py::test_descriptions_are_unique_across_the_catalogue -v
```

Expected: FAIL — 4948 unique keys against 5000 items.

- [ ] **Step 3: Write minimal implementation**

In `scripts/testdata/catalogue.py`, replace the body of the inner loop in `build_catalogue`:

```python
    for leaf, count in zip(leaves, per_leaf):
        for _ in range(max(count, 1) if count == 0 else count):
            counter += 1
            item_id = f"ITM{counter:06d}"
            qualifier = rng.choice(_QUALIFIERS)
            noun = rng.choice(_NOUNS)
            magnitude = rng.choice([1, 1, 1, 10, 10, 100, 1000])
            base = Decimal(str(round(rng.uniform(0.8, 9.9) * magnitude, 2)))
            items.append(
                CatalogueItem(
                    item_id=item_id,
                    # The part code is what makes the description unique. Without
                    # it, 10 qualifiers x 10 nouns gives only 100 shapes per leaf
                    # for ~20 items, and colliding items pool as one product.
                    description=f"{qualifier} {leaf.l5} {noun} {item_id}",
                    leaf=leaf,
                    unit_of_measure=rng.choice(UNITS),
                    base_price=base,
                    currency="GBP",
                    preferred_supplier_id=supplier_ids[counter % len(supplier_ids)],
                )
            )
```

- [ ] **Step 4: Run the catalogue tests**

```bash
.venv/bin/python -m pytest tests/testdata/test_catalogue.py -v
```

Expected: PASS — all tests including the two new ones.

- [ ] **Step 5: Run the whole testdata suite to catch knock-on breakage**

```bash
.venv/bin/python -m pytest tests/testdata/ -q
```

Expected: PASS. The defect planter mutates chains by `requirement_id`, not by description, so nothing should break. If `tests/testdata/test_defects.py` fails, stop and report — do not adjust defect counts to fit.

- [ ] **Step 6: Commit**

```bash
git add scripts/testdata/catalogue.py tests/testdata/test_catalogue.py
git commit -m "fix(testdata): unique item descriptions so unrelated items cannot pool"
```

---

### Task 2: Convert line prices into the cost centre's currency

`documents._build_lines` stamps the cost centre's currency on each line while the price stays in untranslated sterling, so a US cost centre records a sterling magnitude labelled USD.

**Files:**
- Modify: `scripts/testdata/reference.py`, `scripts/testdata/documents.py:75-97,128-152`
- Test: `tests/testdata/test_reference.py`, `tests/testdata/test_documents.py`

**Interfaces:**
- Consumes: `proc.bp_fx_rates` (columns `base_currency`, `currency`, `rate`, `fetched_at`; base is `USD`)
- Produces:
  - `reference.load_fx_rates(source_db: str) -> dict[str, float]` — currency code to USD-based rate, from the single most recent `fetched_at` snapshot
  - `documents.convert(amount: Decimal, to_currency: str, fx: Mapping[str, float]) -> Decimal`
  - `documents.build_chains(seed, suppliers, catalogue_items, cost_centres, *, fx: Mapping[str, float], count: int = 6000) -> list[Chain]` — `fx` is now a required keyword argument

- [ ] **Step 1: Write the failing tests**

Add to `tests/testdata/test_reference.py`:

```python
import pytest
from scripts.testdata.reference import load_fx_rates


@pytest.mark.integration
def test_fx_snapshot_is_one_moment_and_includes_the_majors():
    rates = load_fx_rates("bp_sqldb")
    for code in ("USD", "GBP", "EUR", "INR", "AED"):
        assert code in rates, code
        assert rates[code] > 0
    assert rates["USD"] == pytest.approx(1.0)
```

Add to `tests/testdata/test_documents.py`:

```python
from decimal import Decimal

from scripts.testdata.documents import convert

FX = {"USD": 1.0, "GBP": 0.8, "EUR": 0.92}


def test_convert_goes_through_the_usd_base():
    # 100 GBP -> USD is 100 / 0.8 = 125; -> EUR is 125 * 0.92 = 115.00
    assert convert(Decimal("100.00"), "EUR", FX) == Decimal("115.00")


def test_convert_is_identity_for_sterling():
    assert convert(Decimal("100.00"), "GBP", FX) == Decimal("100.00")


def test_convert_rounds_half_up_to_the_penny():
    assert convert(Decimal("10.00"), "EUR", FX) == Decimal("11.50")


def test_unknown_currency_is_an_error_not_a_silent_passthrough():
    import pytest
    with pytest.raises(KeyError):
        convert(Decimal("100.00"), "ZZZ", FX)


def test_line_prices_are_denominated_in_the_line_currency():
    chains = _fixture(50)
    for chain in chains:
        for quote in chain.quotes:
            for line in quote.lines:
                assert line.currency == quote.currency
```

Update `_fixture` in `tests/testdata/test_documents.py` to pass the new argument:

```python
FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "AED": 3.6725}


def _fixture(chain_count: int = 200):
    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(40000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(60)
    ]
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    return build_chains(42, suppliers, items, centres, fx=FX, count=chain_count)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/testdata/test_documents.py -v -m "not integration"
```

Expected: FAIL — `ImportError: cannot import name 'convert'`.

- [ ] **Step 3: Implement `load_fx_rates`**

Append to `scripts/testdata/reference.py`:

```python
def load_fx_rates(source_db: str) -> dict[str, float]:
    """Currency code -> USD-based rate, from the newest snapshot only.

    bp_fx_rates accumulates snapshots; mixing two of them would make the same
    currency convert two ways in one build. The build log records which
    snapshot was used, because a different snapshot means a different checksum.
    """
    conn = connect(source_db)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select currency, rate from proc.bp_fx_rates
                 where fetched_at = (select max(fetched_at) from proc.bp_fx_rates)
                   and base_currency = 'USD'
                """
            )
            return {code: float(rate) for code, rate in cur.fetchall()}
    finally:
        conn.close()
```

- [ ] **Step 4: Implement `convert` and thread `fx` through**

In `scripts/testdata/documents.py`, add after `_money`:

```python
def convert(amount: Decimal, to_currency: str, fx: Mapping[str, float]) -> Decimal:
    """Sterling amount -> to_currency, via the USD base the FX table uses.

    Raises KeyError on an unknown currency: a missing rate must fail loudly
    rather than silently label a sterling figure as something else, which is
    the bug this function exists to fix.
    """
    if to_currency == "GBP":
        return _money(amount)
    usd = Decimal(str(amount)) / Decimal(str(fx["GBP"]))
    return _money(usd * Decimal(str(fx[to_currency])))
```

Add `Mapping` to the `typing` import. Change `_build_lines` to take and use `fx`:

```python
def _build_lines(
    rng, items: Sequence[CatalogueItem], when: date, seed: int, currency: str,
    fx: Mapping[str, float],
) -> tuple[LineItem, ...]:
    count = rng.randint(2, 9)
    lines: list[LineItem] = []
    for number in range(1, count + 1):
        item = items[rng.randrange(len(items))]
        unit_price = convert(price_on(item, when, seed=seed), currency, fx)
        quantity = Decimal(str(rng.choice([1, 1, 2, 3, 4, 5, 8, 10, 12, 25, 40])))
        line_total = _money(quantity * unit_price)
        lines.append(
            LineItem(
                line_number=number,
                item_id=item.item_id,
                description=item.description,
                quantity=quantity,
                unit_price=unit_price,
                line_total=line_total,
                currency=currency,
                leaf_path=item.leaf.path,
            )
        )
    return tuple(lines)
```

Change `build_chains` to require `fx` and pass it down:

```python
def build_chains(
    seed: int,
    suppliers: Sequence[Supplier],
    catalogue_items: Sequence[CatalogueItem],
    cost_centres: Sequence[CostCentre],
    *,
    fx: Mapping[str, float],
    count: int = 6000,
) -> list[Chain]:
```

and inside, the single `_build_lines` call becomes:

```python
            lines = _build_lines(rng, catalogue_items, quote_date, seed, centre.currency, fx)
```

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/testdata/test_documents.py tests/testdata/test_reference.py -v -m "not integration"
```

Expected: PASS.

- [ ] **Step 6: Fix the other callers**

`scripts/testdata/build.py` and `tests/testdata/test_defects.py` both call `build_chains`. In `build.py`, add the import and the load:

```python
from scripts.testdata.reference import copy_reference, load_fx_rates, load_taxonomy
```

and in `main`, immediately before the `build_chains` call:

```python
    fx = load_fx_rates("bp_sqldb")
    print(f"  FX snapshot: {len(fx)} currencies")
    chains = build_chains(args.seed, suppliers, items, centres, fx=fx, count=6000)
```

In `tests/testdata/test_defects.py`, add the same literal `fx` dict used in `test_documents.py` to its fixture builder.

```bash
.venv/bin/python -m pytest tests/testdata/ -q -m "not integration"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/testdata/reference.py scripts/testdata/documents.py scripts/testdata/build.py tests/testdata/
git commit -m "fix(testdata): convert line prices into the cost centre currency"
```

---

### Task 3: Map documents onto the target tables

Pure mapping, no database. Splitting the mapping from the loading means the column lists can be tested against the live schema without writing anything.

**Files:**
- Create: `scripts/testdata/persist.py`
- Test: `tests/testdata/test_persist.py`

**Interfaces:**
- Consumes: `documents.Chain`, `documents.Document`, `documents.LineItem`, `org.ENTITIES`
- Produces:
  - `COLUMNS: dict[str, tuple[str, ...]]` — keys are the six table names
  - `ENTITY_REGION: dict[str, str]`
  - `rows_for(chains: Sequence[Chain]) -> dict[str, list[list]]` — table name to rows, column order matching `COLUMNS`

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_persist.py`:

```python
from datetime import date
from decimal import Decimal

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.documents import build_chains
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.persist import COLUMNS, rows_for
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers

FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "AED": 3.6725}


def _chains(count: int = 40):
    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(40000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(60)
    ]
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    centres = build_cost_centres(42, build_business_units(42), leaves)
    return build_chains(42, suppliers, items, centres, fx=FX, count=count)


def test_every_table_has_rows():
    rows = rows_for(_chains())
    for table in COLUMNS:
        assert rows[table], table


def test_row_width_matches_the_column_list():
    rows = rows_for(_chains())
    for table, columns in COLUMNS.items():
        for row in rows[table]:
            assert len(row) == len(columns), table


def test_one_row_per_quote_and_one_per_quote_line():
    chains = _chains()
    rows = rows_for(chains)
    assert len(rows["bp_quote_trgt"]) == sum(len(c.quotes) for c in chains)
    assert len(rows["bp_quote_line_items_trgt"]) == sum(
        len(q.lines) for c in chains for q in c.quotes)


def test_line_ids_are_unique_and_derived_from_their_document():
    rows = rows_for(_chains())
    idx = COLUMNS["bp_quote_line_items_trgt"].index("quote_line_id")
    ids = [r[idx] for r in rows["bp_quote_line_items_trgt"]]
    assert len(set(ids)) == len(ids)
    assert all("-" in i for i in ids)


def test_awarded_quote_carries_its_purchase_order_reference():
    chains = _chains()
    rows = rows_for(chains)
    qid = COLUMNS["bp_quote_trgt"].index("quote_id")
    poid = COLUMNS["bp_quote_trgt"].index("po_id")
    by_quote = {r[qid]: r[poid] for r in rows["bp_quote_trgt"]}
    linked = [c for c in chains if c.purchase_order]
    assert linked, "fixture must contain at least one chain that reached a PO"
    for chain in linked:
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        assert by_quote[awarded.doc_id] == chain.purchase_order.doc_id


def test_marker_fields_identify_generated_rows():
    rows = rows_for(_chains())
    idx = COLUMNS["bp_invoice_trgt"].index("created_by")
    assert {r[idx] for r in rows["bp_invoice_trgt"]} == {"testdata"}


def test_totals_survive_the_mapping():
    chains = _chains()
    rows = rows_for(chains)
    amt = COLUMNS["bp_invoice_trgt"].index("invoice_amount")
    total = sum(Decimal(str(r[amt])) for r in rows["bp_invoice_trgt"])
    expected = sum(inv.net_total for c in chains for inv in c.invoices)
    assert total == expected
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.persist'`.

- [ ] **Step 3: Give `LineItem` a unit of measure**

`persist.py` needs it and `LineItem` does not carry it — only the catalogue item
does. Unit of measure is one of the three keys the benchmark engine matches on,
so a wrong or invented one silently splits an item's price pool in two.

In `scripts/testdata/documents.py`, add to `LineItem` after `description`:

```python
    unit_of_measure: str
```

and in `_build_lines`, inside the `LineItem(...)` call after `description=item.description,`:

```python
                unit_of_measure=item.unit_of_measure,
```

- [ ] **Step 4: Write the implementation**

Create `scripts/testdata/persist.py`:

```python
"""Map generated documents onto the six _trgt tables and bulk-load them.

Plan 1 generated 38,000 documents and discarded them. Without these rows the
price-history pool is empty and the benchmark engine gates on every line, so
persistence is what makes the engine testable at all.

Only columns the generator can actually fill are written. Anything else is left
NULL rather than invented. The purchase-order HEADER table is
bp_purchase_order_trgt; only the line table uses the `po` abbreviation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from scripts.testdata.documents import Chain, Document, LineItem
from scripts.testdata.org import ENTITIES

MARKER = "testdata"

# The header tables carry country and region but no business-unit or
# cost-centre column, so entity geography is the most attribution that fits.
ENTITY_COUNTRY: dict[str, str] = {e.org_id: e.country for e in ENTITIES}
ENTITY_REGION: dict[str, str] = {
    "ORG-UK": "Europe",
    "ORG-DE": "Europe",
    "ORG-IE": "Europe",
    "ORG-US": "North America",
    "ORG-IN": "APAC",
    "ORG-AE": "Middle East",
}

COLUMNS: dict[str, tuple[str, ...]] = {
    "bp_quote_trgt": (
        "quote_id", "supplier_id", "buyer_id", "quote_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax", "po_id",
        "country", "region", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "bp_quote_line_items_trgt": (
        "quote_line_id", "quote_id", "line_number", "item_id",
        "item_description", "quantity", "unit_of_measure", "unit_price",
        "line_total", "currency", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "bp_purchase_order_trgt": (
        "po_id", "supplier_id", "buyer_id", "order_date", "currency",
        "total_amount", "tax_amount", "total_amount_incl_tax",
        "quote_reference", "ship_to_country", "delivery_region",
        "created_date", "created_by", "last_modified_by", "last_modified_date",
    ),
    "bp_po_line_items_trgt": (
        "po_line_id", "po_id", "line_number", "item_id", "item_description",
        "quantity", "unit_of_measure", "unit_price", "line_total", "currency",
        "quote_number", "created_date", "created_by", "last_modified_by",
        "last_modified_date",
    ),
    "bp_invoice_trgt": (
        "invoice_id", "po_id", "supplier_id", "buyer_id", "invoice_date",
        "currency", "invoice_amount", "tax_amount", "invoice_total_incl_tax",
        "country", "region", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "bp_invoice_line_items_trgt": (
        "invoice_line_id", "invoice_id", "line_no", "item_id",
        "item_description", "quantity", "unit_of_measure", "unit_price",
        "line_amount", "po_id", "country", "region", "created_date",
        "created_by", "last_modified_by", "last_modified_date",
    ),
}

TABLES: tuple[str, ...] = tuple(COLUMNS)


def _stamp(doc: Document) -> list:
    """created_date, created_by, last_modified_by, last_modified_date.

    Stamped from the document's own date rather than wall-clock time: the build
    must reproduce byte-for-byte from a seed, and now() would break that.
    """
    when = datetime.combine(doc.doc_date, datetime.min.time())
    return [when, MARKER, MARKER, when]


def _quote_rows(doc: Document, po_id: str | None) -> list:
    return [
        doc.doc_id, doc.supplier_id, doc.cc_id, doc.doc_date, doc.currency,
        doc.net_total, doc.tax_amount, doc.gross_total, po_id,
        ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id),
        *_stamp(doc),
    ]


def _quote_line_rows(doc: Document, line: LineItem) -> list:
    return [
        f"{doc.doc_id}-{line.line_number}", doc.doc_id, line.line_number,
        line.item_id, line.description, line.quantity, line.unit_of_measure,
        line.unit_price, line.line_total, line.currency, *_stamp(doc),
    ]


def _po_rows(doc: Document, quote_ref: str | None) -> list:
    return [
        doc.doc_id, doc.supplier_id, doc.cc_id, doc.doc_date, doc.currency,
        doc.net_total, doc.tax_amount, doc.gross_total, quote_ref,
        ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id),
        *_stamp(doc),
    ]


def _po_line_rows(doc: Document, line: LineItem, quote_ref: str | None) -> list:
    return [
        f"{doc.doc_id}-{line.line_number}", doc.doc_id, line.line_number,
        line.item_id, line.description, line.quantity, line.unit_of_measure,
        line.unit_price, line.line_total, line.currency, quote_ref,
        *_stamp(doc),
    ]


def _invoice_rows(doc: Document) -> list:
    return [
        doc.doc_id, doc.parent_doc_id, doc.supplier_id, doc.cc_id,
        doc.doc_date, doc.currency, doc.net_total, doc.tax_amount,
        doc.gross_total, ENTITY_COUNTRY.get(doc.org_id),
        ENTITY_REGION.get(doc.org_id), *_stamp(doc),
    ]


def _invoice_line_rows(doc: Document, line: LineItem) -> list:
    return [
        f"{doc.doc_id}-{line.line_number}", doc.doc_id, line.line_number,
        line.item_id, line.description, line.quantity, line.unit_of_measure,
        line.unit_price, line.line_total, doc.parent_doc_id,
        ENTITY_COUNTRY.get(doc.org_id), ENTITY_REGION.get(doc.org_id),
        *_stamp(doc),
    ]


def rows_for(chains: Sequence[Chain]) -> dict[str, list[list]]:
    """Every table's rows, in COLUMNS order. Pure: no database contact."""
    out: dict[str, list[list]] = {table: [] for table in TABLES}

    for chain in chains:
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        po = chain.purchase_order
        po_id = po.doc_id if po else None

        for quote in chain.quotes:
            linked_po = po_id if quote.doc_id == awarded.doc_id else None
            out["bp_quote_trgt"].append(_quote_rows(quote, linked_po))
            for line in quote.lines:
                out["bp_quote_line_items_trgt"].append(
                    _quote_line_rows(quote, line))

        if po is not None:
            out["bp_purchase_order_trgt"].append(_po_rows(po, awarded.doc_id))
            for line in po.lines:
                out["bp_po_line_items_trgt"].append(
                    _po_line_rows(po, line, awarded.doc_id))

        for invoice in chain.invoices:
            out["bp_invoice_trgt"].append(_invoice_rows(invoice))
            for line in invoice.lines:
                out["bp_invoice_line_items_trgt"].append(
                    _invoice_line_rows(invoice, line))

    return out
```

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py tests/testdata/test_documents.py -v -m "not integration"
```

Expected: PASS.

- [ ] **Step 6: Verify the column lists against the live schema**

```bash
.venv/bin/python -c "
from scripts.testdata.db import connect
from scripts.testdata.persist import COLUMNS
conn=connect('bp_sqldb'); cur=conn.cursor()
bad=0
for t,cols in COLUMNS.items():
    cur.execute(\"select column_name from information_schema.columns where table_schema='proc' and table_name=%s\",(t,))
    actual={r[0] for r in cur.fetchall()}
    missing=[c for c in cols if c not in actual]
    print(f'{t:32s} {\"OK\" if not missing else missing}')
    bad+=len(missing)
conn.close(); raise SystemExit(1 if bad else 0)"
```

Expected: every table `OK`, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/testdata/persist.py scripts/testdata/documents.py tests/testdata/
git commit -m "feat(testdata): map generated documents onto the target tables"
```

---

### Task 4: Load the documents into the database

**Files:**
- Modify: `scripts/testdata/persist.py`, `scripts/testdata/build.py`
- Test: `tests/testdata/test_persist.py`

**Interfaces:**
- Consumes: `db.copy_rows(conn, schema, table, columns, rows) -> int`, `persist.rows_for`
- Produces: `persist.write_chains(conn, chains: Sequence[Chain]) -> dict[str, int]` — table name to rows written

- [ ] **Step 1: Write the failing test**

Add to `tests/testdata/test_persist.py`:

```python
import pytest

from scripts.testdata.db import connect
from scripts.testdata.persist import TABLES, write_chains


@pytest.mark.integration
def test_write_chains_is_idempotent_and_reports_counts():
    chains = _chains(20)
    conn = connect("bp_testdb")
    try:
        first = write_chains(conn, chains)
        second = write_chains(conn, chains)
        assert first == second
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"select count(*) from proc.{table}")
                assert cur.fetchone()[0] == first[table], table
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -v -m integration
```

Expected: FAIL — `ImportError: cannot import name 'write_chains'`.

- [ ] **Step 3: Write the implementation**

Append to `scripts/testdata/persist.py`:

```python
from scripts.testdata.db import copy_rows


def write_chains(conn, chains: Sequence[Chain]) -> dict[str, int]:
    """Truncate the six target tables and bulk-load the generated documents.

    Truncating makes a rebuild idempotent. Safe because the caller has already
    passed guards.assert_safe_target — this never runs against a live database.
    """
    rows = rows_for(chains)
    with conn.cursor() as cur:
        # Reverse order so line tables go before their headers, in case a
        # foreign key is ever added to these tables.
        for table in reversed(TABLES):
            cur.execute(f"truncate proc.{table}")
    conn.commit()

    written: dict[str, int] = {}
    for table in TABLES:
        written[table] = copy_rows(conn, "proc", table, list(COLUMNS[table]),
                                   rows[table])
    conn.commit()
    return written
```

- [ ] **Step 4: Wire it into the build**

In `scripts/testdata/build.py`, add the import:

```python
from scripts.testdata.persist import write_chains
```

and after the `_write_suppliers(...)` call in `main`:

```python
    print("writing documents")
    doc_conn = connect(args.target)
    try:
        written = write_chains(doc_conn, chains)
    finally:
        doc_conn.close()
    for table, count in written.items():
        print(f"  {table:32s} {count:>7,}")
```

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -v
```

Expected: PASS, including the integration test.

- [ ] **Step 6: Commit**

```bash
git add scripts/testdata/persist.py scripts/testdata/build.py tests/testdata/test_persist.py
git commit -m "feat(testdata): bulk-load documents and line items into the target tables"
```

---

### Task 5: Assign deal identifiers with the real service

`GET /benchmark/by-deal/{deal_id}` selects quote lines by `deal_id`. The cloned schema contains no deal-assignment routine — all ten cloned functions are outcome, discrepancy or process-monitor related — so nothing assigns one unless we run the production service.

**Files:**
- Modify: `scripts/testdata/build.py`
- Test: `tests/testdata/test_build.py`

**Interfaces:**
- Consumes: `src.services.deal_assignment_service.assign_deals(conn=None, limit=None) -> dict`
- Produces: `build.assign_deals_on(target_db: str) -> dict` — the raw service result, or `{"error": str}` if it fails

- [ ] **Step 1: Write the failing test**

Add to `tests/testdata/test_build.py`:

```python
def test_deal_assignment_failure_does_not_abort_the_build(monkeypatch):
    """A grouping failure is a product finding to report, not a reason to throw
    away a good 190,000-row build."""
    from scripts.testdata import build

    def boom(*args, **kwargs):
        raise RuntimeError("linking engine exploded")

    monkeypatch.setattr(build, "_assign_deals_impl", boom)
    result = build.assign_deals_on("bp_testdb")
    assert "error" in result
    assert "linking engine exploded" in result["error"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_build.py::test_deal_assignment_failure_does_not_abort_the_build -v
```

Expected: FAIL — `AttributeError: module 'scripts.testdata.build' has no attribute 'assign_deals_on'`.

- [ ] **Step 3: Write the implementation**

Add to `scripts/testdata/build.py`:

```python
def _assign_deals_impl(target_db: str) -> dict:
    """Indirection so the failure path is testable without a linking engine.

    Imported lazily: the seeder must stay runnable when the application's
    dependencies are not importable, and the import is only needed here.
    """
    from scripts.testdata.db import connect as _connect
    from src.services.deal_assignment_service import assign_deals

    conn = _connect(target_db)
    try:
        return assign_deals(conn=conn)
    finally:
        conn.close()


def assign_deals_on(target_db: str) -> dict:
    """Group the seeded documents into deals using the production service.

    The seeder never stamps deal_id itself. If the service cannot group
    synthetic chains that is a finding about the grouping service, reported
    here and in the build log rather than papered over.
    """
    try:
        return _assign_deals_impl(target_db)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
```

and call it in `main`, after the document write:

```python
    print("assigning deals")
    deal_result = assign_deals_on(args.target)
    if "error" in deal_result:
        print(f"  DEAL ASSIGNMENT FAILED: {deal_result['error']}", file=sys.stderr)
        print("  documents are loaded; by-deal benchmark queries will return nothing")
    else:
        print(f"  {deal_result}")
```

- [ ] **Step 4: Run the tests**

```bash
.venv/bin/python -m pytest tests/testdata/test_build.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/build.py tests/testdata/test_build.py
git commit -m "feat(testdata): group seeded documents into deals via the real service"
```

---

### Task 6: Purchase-order currency must come from the header

`load_benchmark_pool` reads `p.currency` from the line table, which is NULL on every live row, so `_norm_currency` labels all of it sterling. 77 of the 136 live pool lines are actually US or New Zealand dollars.

**Files:**
- Modify: `src/services/benchmark_live.py:75-92`
- Test: `tests/test_benchmark_live_reads.py` (create)

**Interfaces:**
- Consumes: `proc.bp_po_line_items_trgt`, `proc.bp_purchase_order_trgt`
- Produces: `load_benchmark_pool(cur)` rows gain a `doc_id` key alongside the existing ones

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_live_reads.py`:

```python
"""The pool query's reads, exercised against a fake cursor.

A fake rather than a database: these tests pin which COLUMNS the SQL selects
and how NULLs are treated, which is exactly where the currency defect lived.
"""
from services.benchmark_live import _to_points, load_benchmark_pool


class FakeCursor:
    """Records the SQL it was given and replays a canned result."""

    def __init__(self, rows, description):
        self._rows = rows
        self.description = description
        self.sql = ""

    def execute(self, sql, params=None):
        self.sql = sql

    def fetchall(self):
        return self._rows


def test_pool_query_joins_the_purchase_order_header_for_currency():
    cur = FakeCursor([], [("point_id",), ("item_description",),
                          ("unit_of_measure",), ("currency",), ("unit_price",),
                          ("quantity",), ("doc_id",)])
    load_benchmark_pool(cur)
    sql = " ".join(cur.sql.split()).lower()
    assert "bp_purchase_order_trgt" in sql, "PO header is never joined"
    assert "coalesce(p.currency, h.currency)" in sql


def test_header_currency_survives_into_the_point():
    rows = [{"point_id": "po:1", "item_description": "Widget",
             "unit_of_measure": "each", "currency": "USD",
             "unit_price": 10.0, "quantity": 2, "doc_id": "PO1"}]
    assert _to_points(rows)[0].currency == "USD"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_live_reads.py -v
```

Expected: FAIL — `bp_purchase_order_trgt` is not in the SQL.

- [ ] **Step 3: Write the implementation**

Replace the query body in `load_benchmark_pool`:

```python
def load_benchmark_pool(cur, exclude_deal_id: Optional[str] = None) -> list[dict[str, Any]]:
    """All PO + invoice lines with a price — the price-history pool.

    Currency is taken from the document header when the line does not carry it.
    Every live PO line has a NULL currency while 57% of PO headers are not
    sterling, so reading the line alone silently relabels dollars as pounds.
    """
    cur.execute(
        """
        SELECT 'po:' || p.po_line_id AS point_id, p.item_description,
               p.unit_of_measure, COALESCE(p.currency, ph.currency) AS currency,
               p.unit_price, p.quantity, p.po_id AS doc_id
        FROM proc.bp_po_line_items_trgt p
        LEFT JOIN proc.bp_purchase_order_trgt ph ON ph.po_id = p.po_id
        WHERE p.unit_price IS NOT NULL AND p.item_description IS NOT NULL
          AND (%(deal)s::text IS NULL OR p.deal_id IS DISTINCT FROM %(deal)s)
        UNION ALL
        SELECT 'inv:' || i.invoice_line_id, i.item_description,
               i.unit_of_measure, h.currency, i.unit_price, i.quantity,
               i.invoice_id AS doc_id
        FROM proc.bp_invoice_line_items_trgt i
        LEFT JOIN proc.bp_invoice_trgt h ON h.invoice_id = i.invoice_id
        WHERE i.unit_price IS NOT NULL AND i.item_description IS NOT NULL
          AND (%(deal)s::text IS NULL OR i.deal_id IS DISTINCT FROM %(deal)s)
        """,
        {"deal": exclude_deal_id},
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
```

The `exclude_deal_id` parameter is wired up in Task 8; it defaults to `None` here so this task's change is self-contained.

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_live_reads.py tests/test_benchmark_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Prove it against live data**

```bash
PYTHONPATH=src .venv/bin/python -c "
from scripts.testdata.db import connect
from services.benchmark_live import load_benchmark_pool, _to_points
conn=connect('bp_sqldb')
pts=_to_points(load_benchmark_pool(conn.cursor()))
from collections import Counter
print(Counter(p.currency for p in pts if p.benchmark_point_id.startswith('po:')))
conn.close()"
```

Expected: a mix including `USD` and `NZD`, not `{'GBP': 136}`.

- [ ] **Step 6: Commit**

```bash
git add src/services/benchmark_live.py tests/test_benchmark_live_reads.py
git commit -m "fix(benchmark): take purchase-order currency from the header, not the empty line column"
```

---

### Task 7: A missing quantity is unknown, not zero

`benchmark_live.py:111` turns a NULL quantity into `0.0`, which drags down the weighted reference quantity and tilts the volume adjustment. Services lines legitimately have no quantity.

**Files:**
- Modify: `src/services/benchmark/models.py:103`, `src/services/benchmark/engine.py:49-57,172`, `src/services/benchmark_live.py:111`
- Test: `tests/test_benchmark_engine.py`, `tests/test_benchmark_live_reads.py`

**Interfaces:**
- Produces: `BenchmarkPoint.historical_quantity: Optional[float]`; `engine._weighted_avg` skips `None` values and their weights

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark_engine.py`:

```python
def test_missing_historical_quantity_is_excluded_not_zeroed():
    """A services line with no quantity must not pull the reference quantity
    toward zero — that would fake a volume premium out of missing data."""
    from services.benchmark.engine import compute_benchmark
    from services.benchmark.models import BenchmarkPoint, QuoteLine

    def point(pid, qty):
        return BenchmarkPoint(
            benchmark_point_id=pid, source="internal", item_name="widget",
            uom="each", currency="GBP", include=True, raw_unit_price=100.0,
            source_weight=1.0, specification_score=5.0, location_cost_index=1.0,
            sla_score=5.0, historical_quantity=qty,
            index_value_at_price_date=1.0,
        )

    quote = QuoteLine(
        deal_id="D", item_name="widget", quantity=100, uom="each",
        currency="GBP", location="UK", requested_spec_score=5,
        requested_sla_score=5, index_id="", quoted_unit_price=100.0,
    )
    known = [point("a", 100.0), point("b", 100.0), point("c", 100.0)]
    with_gap = known + [point("d", None)]

    assert compute_benchmark(quote, known, {}, {}).ref_quantity == 100.0
    assert compute_benchmark(quote, with_gap, {}, {}).ref_quantity == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_engine.py::test_missing_historical_quantity_is_excluded_not_zeroed -v
```

Expected: FAIL — a pydantic validation error, `historical_quantity` is not optional.

- [ ] **Step 3: Write the implementation**

In `src/services/benchmark/models.py`, change the field:

```python
    # None means "no quantity recorded" (services lines legitimately have none).
    # Excluded from the reference-quantity average rather than counted as zero.
    historical_quantity: Optional[float] = None
```

In `src/services/benchmark/engine.py`, add a sibling of `_weighted_avg`:

```python
def _weighted_avg_optional(
    values: Sequence[Optional[float]], weights: Sequence[float], digits: int
) -> Optional[float]:
    """Weighted average over the values that exist, ignoring the rest.

    A missing value drops its weight too, so the remaining points keep their
    relative influence. Identical to _weighted_avg when nothing is missing,
    which is why every golden fixture is unaffected.
    """
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total == 0:
        return None
    return excel_round(sum(v * w for v, w in pairs) / total, digits)
```

and change the reference-quantity line:

```python
    ref_quantity = _weighted_avg_optional(
        [p.historical_quantity for p in matched], weights, 2)
```

In `src/services/benchmark_live.py`, stop coercing:

```python
                historical_quantity=(
                    float(row["quantity"]) if row["quantity"] is not None else None
                ),
```

- [ ] **Step 4: Run the full benchmark suite**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_engine.py tests/test_benchmark_models.py tests/test_benchmark_parity.py tests/test_benchmark_api.py tests/test_benchmark_live_reads.py -q
```

Expected: PASS — 59+ tests, with every golden-fixture value unchanged. If a parity test fails, stop: the change was supposed to be inert on complete data.

- [ ] **Step 5: Commit**

```bash
git add src/services/benchmark/models.py src/services/benchmark/engine.py src/services/benchmark_live.py tests/test_benchmark_engine.py
git commit -m "fix(benchmark): treat a missing historical quantity as unknown, not zero"
```

---

### Task 8: A deal is not its own benchmark

The pool has no deal filter, so a deal's own purchase order and invoices judge its quotes — and in the seeded data every purchase order copies the awarded quote verbatim.

**Files:**
- Modify: `src/services/benchmark_live.py:118-164`
- Test: `tests/test_benchmark_live_reads.py`

**Interfaces:**
- Produces: `benchmark_deal(...)` result gains `own_documents_excluded: int` at the top level

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark_live_reads.py`:

```python
def test_pool_query_excludes_the_deal_under_analysis():
    cur = FakeCursor([], [("point_id",), ("item_description",),
                          ("unit_of_measure",), ("currency",), ("unit_price",),
                          ("quantity",), ("doc_id",)])
    load_benchmark_pool(cur, exclude_deal_id="DEAL-1")
    sql = " ".join(cur.sql.split()).lower()
    assert "p.deal_id is distinct from" in sql
    assert "i.deal_id is distinct from" in sql
```

and a counting test:

```python
def test_excluded_own_document_count_is_reported():
    """Read the two pool sizes and report the difference, so a line that gates
    because its own documents were removed can be explained."""
    from services.benchmark_live import _pool_delta
    assert _pool_delta(120, 100) == 20
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_live_reads.py -v
```

Expected: FAIL — `cannot import name '_pool_delta'`.

- [ ] **Step 3: Write the implementation**

In `src/services/benchmark_live.py`, add:

```python
def _pool_delta(full: int, scoped: int) -> int:
    """How many points the deal's own documents contributed."""
    return max(0, full - scoped)
```

and change `benchmark_deal` to load the pool twice — once unscoped for the count, once scoped for the calculation:

```python
    quote_rows = load_quote_lines(cur, deal_id)
    full_pool = load_benchmark_pool(cur)
    scoped_pool = load_benchmark_pool(cur, exclude_deal_id=deal_id)
    own_excluded = _pool_delta(len(full_pool), len(scoped_pool))
    points = _to_points(scoped_pool)
```

and add to the returned dictionary:

```python
        "own_documents_excluded": own_excluded,
```

Replace the third disclosure string, which now describes behaviour that no longer happens:

```python
    "price history excludes this deal's own purchase orders and invoices, so a supplier is never compared against its own price",
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_live_reads.py tests/test_benchmark_api.py -v
```

Expected: PASS. `tests/test_benchmark_api.py` pins the disclosures against the output-safety gate — if it fails on the reworded string, the wording tripped the gate's route heuristic and needs rephrasing without slashes, not a weakened assertion.

- [ ] **Step 5: Commit**

```bash
git add src/services/benchmark_live.py tests/test_benchmark_live_reads.py
git commit -m "fix(benchmark): exclude a deal's own documents from its own price history"
```

---

### Task 9: Report how many comparison prices are already suspect

For 22% of live quote lines, unit price times quantity does not match the line total. Those rows stay in the pool — we do not know which of the three numbers is wrong — but the response must say how many carry an open finding.

**Files:**
- Modify: `src/services/benchmark_live.py`
- Test: `tests/test_benchmark_live_reads.py`

**Interfaces:**
- Produces: `load_flagged_documents(cur) -> set[str]`; each line in the result gains `suspect_points: int`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmark_live_reads.py`:

```python
def test_flagged_documents_query_reads_open_findings_only():
    cur = FakeCursor([("PO1",), ("INV2",)], [("doc_pk_candidate",)])
    from services.benchmark_live import load_flagged_documents
    assert load_flagged_documents(cur) == {"PO1", "INV2"}
    sql = " ".join(cur.sql.split()).lower()
    assert "bp_extraction_discrepancy" in sql
    assert "status = 'open'" in sql


def test_suspect_points_counts_matched_points_from_flagged_documents():
    from services.benchmark_live import _count_suspect
    doc_by_point = {"po:1": "PO1", "po:2": "PO2", "inv:3": "INV3"}
    assert _count_suspect(["po:1", "inv:3"], doc_by_point, {"PO1"}) == 1
    assert _count_suspect(["po:2"], doc_by_point, {"PO1"}) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_live_reads.py -v
```

Expected: FAIL — `cannot import name 'load_flagged_documents'`.

- [ ] **Step 3: Write the implementation**

Add to `src/services/benchmark_live.py`:

```python
def load_flagged_documents(cur) -> set[str]:
    """Document ids carrying at least one open finding.

    Their prices stay in the pool: when unit price, quantity and line total
    disagree we do not know which is wrong, and dropping the row would be a
    guess presented as a correction. The count is disclosed instead.
    """
    cur.execute(
        """
        SELECT DISTINCT doc_pk_candidate FROM proc.bp_extraction_discrepancy
         WHERE status = 'open' AND doc_pk_candidate IS NOT NULL
        """
    )
    return {row[0] for row in cur.fetchall()}


def _count_suspect(
    point_ids: list[str], doc_by_point: dict[str, str], flagged: set[str]
) -> int:
    return sum(1 for pid in point_ids if doc_by_point.get(pid) in flagged)
```

In `benchmark_deal`, after building `points`:

```python
    doc_by_point = {row["point_id"]: row["doc_id"] for row in scoped_pool}
    flagged = load_flagged_documents(cur)
```

and inside the per-line loop, after `payload = result.model_dump()`:

```python
        payload["suspect_points"] = _count_suspect(
            result.matched_point_ids, doc_by_point, flagged)
```

Add a fifth disclosure:

```python
    "some comparison prices come from documents with an unresolved data-quality finding; each line reports how many",
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_live_reads.py tests/test_benchmark_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/benchmark_live.py tests/test_benchmark_live_reads.py
git commit -m "feat(benchmark): report how many comparison prices carry an open finding"
```

---

### Task 10: The outlier decision rule

Pure function, no database. Compare against the median rather than the mean, because the mean is dragged by the very outliers being hunted.

**Files:**
- Create: `src/services/price_outlier/__init__.py`, `src/services/price_outlier/rule.py`
- Test: `tests/test_price_outlier_rule.py`

**Interfaces:**
- Produces:
  - `OutlierSettings(min_peers=5, robust_threshold=5.0, material_ratio=3.0, critical_ratio=10.0)` — frozen pydantic model
  - `Verdict` — frozen dataclass with `flagged: bool`, `peer_count: int`, `median: Optional[float]`, `ratio: Optional[float]`, `robust_score: Optional[float]`, `severity: Optional[str]`
  - `assess(price: float, peers: Sequence[float], settings: OutlierSettings) -> Verdict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_outlier_rule.py`:

```python
import pytest

from services.price_outlier.rule import OutlierSettings, assess

S = OutlierSettings()
TIGHT = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2]


def test_ten_times_the_median_is_critical():
    v = assess(1000.0, TIGHT, S)
    assert v.flagged and v.severity == "critical"
    assert v.ratio == pytest.approx(10.0, rel=0.02)


def test_a_tenth_of_the_median_is_critical():
    assert assess(10.0, TIGHT, S).severity == "critical"


def test_three_times_the_median_is_a_warning():
    v = assess(300.0, TIGHT, S)
    assert v.flagged and v.severity == "warning"


def test_a_statistically_huge_but_trivial_difference_does_not_flag():
    """2% away from identical peers is arithmetically enormous and
    commercially meaningless. Both conditions must hold."""
    identical = [100.0] * 8
    v = assess(102.0, identical, S)
    assert not v.flagged


def test_identical_peers_fall_back_to_the_ratio_test():
    identical = [100.0] * 8
    assert assess(1000.0, identical, S).flagged


def test_genuine_price_spread_does_not_flag():
    spread = [50.0, 80.0, 100.0, 130.0, 160.0, 200.0]
    assert not assess(210.0, spread, S).flagged


def test_too_few_peers_never_flags():
    v = assess(1000.0, [100.0, 100.0, 100.0, 100.0], S)
    assert not v.flagged
    assert v.peer_count == 4


def test_a_zero_or_negative_median_cannot_form_a_ratio():
    assert not assess(100.0, [0.0] * 8, S).flagged


def test_a_normal_price_does_not_flag():
    assert not assess(100.3, TIGHT, S).flagged
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_price_outlier_rule.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'services.price_outlier'`.

- [ ] **Step 3: Write the implementation**

Create `src/services/price_outlier/__init__.py`:

```python
"""Flag extreme prices for human review. See rule.py for the decision,
detector.py for the database pass."""
from services.price_outlier.rule import OutlierSettings, Verdict, assess

__all__ = ["OutlierSettings", "Verdict", "assess"]
```

Create `src/services/price_outlier/rule.py`:

```python
"""The outlier decision, as a pure function.

Median and median-absolute-deviation, not mean and standard deviation. The mean
and the standard deviation are both dragged by the outliers being hunted, so a
single ten-times-wrong price raises the bar enough to hide itself. The median
does not move.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

# Scales the median absolute deviation so that, for normally distributed data,
# it estimates the same spread as the standard deviation.
_MAD_TO_SIGMA = 1.4826


class OutlierSettings(BaseModel):
    """Every threshold injectable; no magic numbers in the rule."""

    model_config = ConfigDict(frozen=True)

    min_peers: int = Field(default=5, ge=2)
    robust_threshold: float = 5.0
    material_ratio: float = Field(default=3.0, gt=1.0)
    critical_ratio: float = Field(default=10.0, gt=1.0)


@dataclass(frozen=True)
class Verdict:
    flagged: bool
    peer_count: int
    median: Optional[float] = None
    ratio: Optional[float] = None
    robust_score: Optional[float] = None
    severity: Optional[str] = None


def assess(
    price: float, peers: Sequence[float], settings: OutlierSettings
) -> Verdict:
    """Is `price` extreme against `peers`?

    Flags only when the price is BOTH statistically extreme and commercially
    material. Either test alone is useless: the first flags trivia when peers
    are nearly identical, the second flags ordinary price variety.
    """
    n = len(peers)
    if n < settings.min_peers:
        return Verdict(flagged=False, peer_count=n)

    mid = float(median(peers))
    if mid <= 0:
        # No ratio can be formed against a zero or negative median.
        return Verdict(flagged=False, peer_count=n, median=mid)

    ratio = price / mid
    material = ratio >= settings.material_ratio or ratio <= 1.0 / settings.material_ratio

    mad = float(median([abs(p - mid) for p in peers])) * _MAD_TO_SIGMA
    if mad == 0:
        # More than half the peers share one price, so deviation is undefined.
        # The ratio test stands alone rather than flagging everything.
        robust = None
        extreme = material
    else:
        robust = abs(price - mid) / mad
        extreme = robust >= settings.robust_threshold

    flagged = material and extreme
    severity = None
    if flagged:
        severity = (
            "critical"
            if ratio >= settings.critical_ratio or ratio <= 1.0 / settings.critical_ratio
            else "warning"
        )

    return Verdict(
        flagged=flagged, peer_count=n, median=mid, ratio=ratio,
        robust_score=robust, severity=severity,
    )
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_price_outlier_rule.py -v
```

Expected: PASS — 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/services/price_outlier/ tests/test_price_outlier_rule.py
git commit -m "feat(price-outlier): median-based rule for flagging extreme prices"
```

---

### Task 11: Detect outliers across the database

**Files:**
- Create: `src/services/price_outlier/detector.py`
- Test: `tests/test_price_outlier_detector.py`

**Interfaces:**
- Consumes: `rule.assess`, `benchmark_live._norm_item`, `_norm_uom`, `_norm_currency`
- Produces:
  - `Finding` — frozen dataclass: `doc_type`, `doc_pk`, `line_number`, `field_name`, `item_description`, `price`, `verdict`, `note`
  - `build_peer_index(pool_rows) -> dict[tuple[str, str, str], list[float]]`
  - `find_outliers(cur, settings=None) -> list[Finding]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_outlier_detector.py`:

```python
from services.price_outlier.detector import (
    Finding, build_peer_index, describe, peers_for,
)
from services.price_outlier.rule import OutlierSettings, assess


def _row(item, uom, ccy, price, doc="PO1"):
    return {"point_id": f"po:{price}", "item_description": item,
            "unit_of_measure": uom, "currency": ccy, "unit_price": price,
            "quantity": 1, "doc_id": doc}


def test_peer_index_groups_on_the_same_key_the_engine_matches_on():
    rows = [_row("  Widget  ", "Each", "gbp", 100.0, doc="PO1"),
            _row("widget", "each", "GBP", 102.0, doc="PO2"),
            _row("Widget", "box", "GBP", 500.0, doc="PO3")]
    index = build_peer_index(rows)
    assert index[("widget", "each", "GBP")] == [("PO1", 100.0), ("PO2", 102.0)]
    assert index[("widget", "box", "GBP")] == [("PO3", 500.0)]


def test_peers_exclude_the_line_s_own_document():
    rows = [_row("Widget", "each", "GBP", 100.0, doc="PO1"),
            _row("Widget", "each", "GBP", 900.0, doc="PO2")]
    index = build_peer_index(rows)
    assert peers_for(index, ("widget", "each", "GBP"), own_doc="PO1") == [900.0]


def test_peers_for_an_unknown_key_is_empty_not_an_error():
    assert peers_for({}, ("nothing", "each", "GBP"), own_doc=None) == []


def test_note_names_the_comparison_in_plain_english():
    verdict = assess(11.69, [1.17] * 18, OutlierSettings())
    note = describe(3, "A4 Ruled Notebook", 11.69, verdict)
    assert "line 3" in note
    assert "A4 Ruled Notebook" in note
    assert "11.69" in note
    assert "1.17" in note
    assert "18 comparable" in note
    assert "check the unit price and quantity" in note
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_price_outlier_detector.py -v
```

Expected: FAIL — no module `services.price_outlier.detector`.

- [ ] **Step 3: Write the implementation**

Create `src/services/price_outlier/detector.py`:

```python
"""Find extreme prices across the target tables and describe them.

Detection is separated from the benchmark API deliberately: a GET request must
never write findings. This runs as a scheduled job.

The peer set uses exactly the match rule the benchmark engine uses — same item,
unit and currency, normalised the same way — so a flag and a benchmark can never
disagree about what counts as comparable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from services.benchmark_live import _norm_currency, _norm_item, _norm_uom
from services.price_outlier.rule import OutlierSettings, Verdict, assess

logger = logging.getLogger(__name__)

Key = tuple[str, str, str]

# Which tables are scanned, and how each names its columns. Invoice lines use
# line_no and line_amount where the others use line_number and line_total.
_SOURCES: tuple[dict[str, str], ...] = (
    {"doc_type": "quote", "table": "proc.bp_quote_line_items_trgt",
     "doc_pk": "quote_id", "line_no": "line_number"},
    {"doc_type": "purchase_order", "table": "proc.bp_po_line_items_trgt",
     "doc_pk": "po_id", "line_no": "line_number"},
    {"doc_type": "invoice", "table": "proc.bp_invoice_line_items_trgt",
     "doc_pk": "invoice_id", "line_no": "line_no"},
)


@dataclass(frozen=True)
class Finding:
    doc_type: str
    doc_pk: str
    line_number: int
    field_name: str
    item_description: str
    price: float
    verdict: Verdict
    note: str


def build_peer_index(
    pool_rows: Sequence[dict[str, Any]],
) -> dict[Key, list[tuple[Optional[str], float]]]:
    """Group the pool once, by the engine's match key.

    Indexed rather than rescanned: ~190,000 lines against a ~76,000-row pool is
    14 billion comparisons if every line filters the whole list, and about
    76,000 if the pool is grouped once up front.
    """
    index: dict[Key, list[tuple[Optional[str], float]]] = {}
    for row in pool_rows:
        key = (
            _norm_item(row["item_description"]),
            _norm_uom(row["unit_of_measure"]),
            _norm_currency(row["currency"]),
        )
        index.setdefault(key, []).append(
            (row.get("doc_id"), float(row["unit_price"])))
    return index


def peers_for(
    index: dict[Key, list[tuple[Optional[str], float]]], key: Key,
    own_doc: Optional[str],
) -> list[float]:
    """Comparable prices for one match key, excluding the line's own document."""
    return [price for doc, price in index.get(key, ()) if doc != own_doc]


def describe(line_number: int, item: str, price: float, verdict: Verdict) -> str:
    """The sentence a reviewer reads. Names the comparison, not the statistics."""
    direction = "×" if verdict.ratio >= 1 else "÷"
    factor = verdict.ratio if verdict.ratio >= 1 else 1.0 / verdict.ratio
    return (
        f"line {line_number}: '{item}' at {price:,.2f} each is "
        f"{factor:,.1f}{direction} the usual {verdict.median:,.2f} across "
        f"{verdict.peer_count} comparable purchases — "
        f"check the unit price and quantity"
    )


def _load_pool(cur) -> list[dict[str, Any]]:
    from services.benchmark_live import load_benchmark_pool

    return load_benchmark_pool(cur)


def _load_lines(cur, source: dict[str, str]) -> list[dict[str, Any]]:
    cur.execute(
        f"""
        SELECT {source['doc_pk']} AS doc_pk, {source['line_no']} AS line_number,
               item_description, unit_of_measure, unit_price
          FROM {source['table']}
         WHERE unit_price IS NOT NULL AND item_description IS NOT NULL
           AND {source['doc_pk']} IS NOT NULL
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _line_currency(cur, source: dict[str, str]) -> dict[str, str]:
    """Header currency per document, since line currency is unreliable."""
    header = {
        "quote": ("proc.bp_quote_trgt", "quote_id"),
        "purchase_order": ("proc.bp_purchase_order_trgt", "po_id"),
        "invoice": ("proc.bp_invoice_trgt", "invoice_id"),
    }[source["doc_type"]]
    cur.execute(f"SELECT {header[1]}, currency FROM {header[0]}")
    return {row[0]: row[1] for row in cur.fetchall()}


def find_outliers(cur, settings: Optional[OutlierSettings] = None) -> list[Finding]:
    """Every line whose price is extreme against its comparable purchases."""
    settings = settings if settings is not None else OutlierSettings()
    index = build_peer_index(_load_pool(cur))
    findings: list[Finding] = []

    for source in _SOURCES:
        currency_by_doc = _line_currency(cur, source)
        for line in _load_lines(cur, source):
            key = (
                _norm_item(line["item_description"]),
                _norm_uom(line["unit_of_measure"]),
                _norm_currency(currency_by_doc.get(line["doc_pk"])),
            )
            peers = peers_for(index, key, own_doc=line["doc_pk"])
            verdict = assess(float(line["unit_price"]), peers, settings)
            if not verdict.flagged:
                continue
            findings.append(
                Finding(
                    doc_type=source["doc_type"],
                    doc_pk=str(line["doc_pk"]),
                    line_number=int(line["line_number"] or 0),
                    field_name=f"line_items[{line['line_number']}].unit_price",
                    item_description=line["item_description"],
                    price=float(line["unit_price"]),
                    verdict=verdict,
                    note=describe(
                        int(line["line_number"] or 0), line["item_description"],
                        float(line["unit_price"]), verdict),
                )
            )

    logger.info("price outlier scan produced %d findings", len(findings))
    return findings
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_price_outlier_detector.py -v
```

Expected: PASS.

- [ ] **Step 5: Dry-run against live data and report the count**

```bash
PYTHONPATH=src .venv/bin/python -c "
from scripts.testdata.db import connect
from services.price_outlier.detector import find_outliers
conn=connect('bp_sqldb')
f=find_outliers(conn.cursor())
print(f'{len(f)} findings')
for x in f[:5]: print(' ', x.verdict.severity, x.note)
conn.close()"
```

Expected: a small number on the sparse live corpus. Record the count; nothing is written yet.

- [ ] **Step 6: Commit**

```bash
git add src/services/price_outlier/detector.py tests/test_price_outlier_detector.py
git commit -m "feat(price-outlier): scan the target tables for extreme prices"
```

---

### Task 12: Write findings into the Action Centre

The Action Centre reads `proc.bp_extraction_discrepancy` and filters on status alone, so a row with `status='open'` is the whole integration.

**Files:**
- Modify: `src/services/price_outlier/detector.py`, `src/services/price_outlier/__init__.py`
- Test: `tests/test_price_outlier_persist.py`

**Interfaces:**
- Produces: `persist_findings(cur, findings: Sequence[Finding]) -> int` — number of rows written; `ISSUE_TYPE = "price_outlier"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_outlier_persist.py`:

```python
from services.price_outlier.detector import (
    ISSUE_TYPE, Finding, persist_findings,
)
from services.price_outlier.rule import OutlierSettings, assess


class RecordingCursor:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.inserts = []
        self._last = None

    def execute(self, sql, params=None):
        if "SELECT" in sql.upper() and "bp_extraction_discrepancy" in sql:
            self._last = [(k,) for k in self.existing]
        else:
            self.inserts.append(params)
            self._last = []

    def fetchall(self):
        return self._last


def _finding():
    verdict = assess(11.69, [1.17] * 18, OutlierSettings())
    return Finding(
        doc_type="invoice", doc_pk="INV1", line_number=3,
        field_name="line_items[3].unit_price",
        item_description="A4 Ruled Notebook", price=11.69,
        verdict=verdict, note="line 3: ...",
    )


def test_row_uses_expected_value_and_leaves_computed_value_null():
    cur = RecordingCursor()
    persist_findings(cur, [_finding()])
    (params,) = cur.inserts
    assert params[4] == ISSUE_TYPE
    assert float(params[3]) == 11.69          # raw_value: the observed price
    assert float(params[5]) == 1.17           # expected_value: the peer median
    assert params[6] is None                  # computed_value stays empty


def test_severity_and_status_follow_the_verdict():
    cur = RecordingCursor()
    persist_findings(cur, [_finding()])
    (params,) = cur.inserts
    assert params[7] == "critical"
    assert params[8] == "open"
    assert params[9] is False                 # blocks_promotion


def test_an_existing_open_finding_is_not_duplicated():
    cur = RecordingCursor(existing={("invoice", "INV1", "line_items[3].unit_price")})
    assert persist_findings(cur, [_finding()]) == 0
    assert cur.inserts == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_price_outlier_persist.py -v
```

Expected: FAIL — `cannot import name 'ISSUE_TYPE'`.

- [ ] **Step 3: Write the implementation**

Append to `src/services/price_outlier/detector.py`:

```python
ISSUE_TYPE = "price_outlier"

# The Action Centre reads this table and filters on status alone, with no
# issue-type allowlist, so an open row is the whole integration.
_INSERT = """
    INSERT INTO proc.bp_extraction_discrepancy
        (doc_type, source_file, doc_pk_candidate, raw_value, issue_type,
         expected_value, computed_value, severity, status, blocks_promotion,
         field_name, notes)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_EXISTING = f"""
    SELECT doc_type, doc_pk_candidate, field_name
      FROM proc.bp_extraction_discrepancy
     WHERE issue_type = '{ISSUE_TYPE}' AND status = 'open'
"""


def persist_findings(cur, findings: Sequence[Finding]) -> int:
    """Write findings as open discrepancies. Returns rows written.

    computed_value is deliberately left NULL. That column carries two different
    conventions across the codebase — sometimes an expected value, sometimes a
    signed delta — and the gateway reads `expected_value ?? computed_value` as
    the expected figure. Populating only expected_value removes the ambiguity.
    """
    cur.execute(_EXISTING)
    already = {tuple(row) for row in cur.fetchall()}

    written = 0
    for finding in findings:
        key = (finding.doc_type, finding.doc_pk, finding.field_name)
        if key in already:
            continue
        cur.execute(
            _INSERT,
            (
                finding.doc_type,
                f"agent:{ISSUE_TYPE}",
                finding.doc_pk,
                f"{finding.price:.2f}",
                ISSUE_TYPE,
                f"{finding.verdict.median:.2f}",
                None,
                finding.verdict.severity,
                "open",
                False,
                finding.field_name,
                finding.note,
            ),
        )
        already.add(key)
        written += 1

    logger.info("price outlier findings written: %d", written)
    return written
```

Update `src/services/price_outlier/__init__.py`:

```python
"""Flag extreme prices for human review. See rule.py for the decision,
detector.py for the database pass."""
from services.price_outlier.detector import (
    ISSUE_TYPE, Finding, find_outliers, persist_findings,
)
from services.price_outlier.rule import OutlierSettings, Verdict, assess

__all__ = [
    "ISSUE_TYPE", "Finding", "OutlierSettings", "Verdict", "assess",
    "find_outliers", "persist_findings",
]
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_price_outlier_persist.py tests/test_price_outlier_detector.py tests/test_price_outlier_rule.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/price_outlier/ tests/test_price_outlier_persist.py
git commit -m "feat(price-outlier): raise extreme prices as open findings in the Action Centre"
```

---

### Task 13: Run the detector on a schedule, dry-run first

**Files:**
- Modify: `src/services/backend_scheduler.py:334-339` (constants) and near `_register_deal_assignment_job`
- Create: `scripts/run_price_outlier_scan.py`
- Test: `tests/test_price_outlier_job.py`

**Interfaces:**
- Produces:
  - `BackendScheduler.PRICE_OUTLIER_JOB_NAME = "price-outlier-scan"`
  - `BackendScheduler._register_price_outlier_job()`, `BackendScheduler._run_price_outlier_scan()`
  - `scripts/run_price_outlier_scan.py` — `--target`, `--write` (default is a dry run)

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_outlier_job.py`:

Do NOT construct a `BackendScheduler` in these tests. Its `__init__` requires an
`agent_nick` argument and then calls `start()` and `_ensure_email_watcher_service()`,
so building one spawns a real background thread and an email watcher. Test the
gate as a pure function instead.

```python
from src.services.backend_scheduler import (
    BackendScheduler, price_outlier_enabled, price_outlier_interval_minutes,
)


def test_job_is_off_by_default_until_the_first_run_is_reviewed(monkeypatch):
    """190,000 seeded lines could bury the queue. The job stays off until a
    human has looked at a dry run."""
    monkeypatch.delenv("PRICE_OUTLIER_ENABLED", raising=False)
    assert price_outlier_enabled() is False


def test_job_turns_on_only_when_explicitly_enabled(monkeypatch):
    for value, expected in (("1", True), ("true", True), ("True", True),
                            ("0", False), ("", False), ("yes", False)):
        monkeypatch.setenv("PRICE_OUTLIER_ENABLED", value)
        assert price_outlier_enabled() is expected, value


def test_interval_defaults_to_an_hour_and_survives_rubbish(monkeypatch):
    monkeypatch.delenv("PRICE_OUTLIER_INTERVAL_MINUTES", raising=False)
    assert price_outlier_interval_minutes() == 60
    monkeypatch.setenv("PRICE_OUTLIER_INTERVAL_MINUTES", "not-a-number")
    assert price_outlier_interval_minutes() == 60
    monkeypatch.setenv("PRICE_OUTLIER_INTERVAL_MINUTES", "0")
    assert price_outlier_interval_minutes() == 1


def test_the_job_has_a_name():
    assert BackendScheduler.PRICE_OUTLIER_JOB_NAME == "price-outlier-scan"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_price_outlier_job.py -v
```

Expected: FAIL — `ImportError: cannot import name 'price_outlier_enabled'`.

- [ ] **Step 3: Write the implementation**

In `src/services/backend_scheduler.py`, add beside the other job-name constants:

```python
    PRICE_OUTLIER_JOB_NAME = "price-outlier-scan"
```

and add these two module-level helpers near the top of the file, so the gate is
testable without constructing a scheduler:

```python
def price_outlier_enabled() -> bool:
    """OFF by default, unlike the other jobs.

    The seeded corpus is ~190,000 lines against ~1,000 findings already open,
    so a loose threshold could bury the Action Centre. Enable only after
    reviewing a dry run.
    """
    import os
    return os.environ.get("PRICE_OUTLIER_ENABLED", "0").strip() in ("1", "true", "True")


def price_outlier_interval_minutes() -> int:
    import os
    try:
        minutes = int(os.environ.get("PRICE_OUTLIER_INTERVAL_MINUTES", "60"))
    except ValueError:
        return 60
    return max(1, minutes)
```

and the registration method itself:

```python
    def _register_price_outlier_job(self) -> None:
        """Scan for extreme prices and raise them for review."""
        if not price_outlier_enabled():
            logger.info("price outlier job disabled by PRICE_OUTLIER_ENABLED")
            return
        if self.PRICE_OUTLIER_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.PRICE_OUTLIER_JOB_NAME,
            self._run_price_outlier_scan,
            interval=timedelta(minutes=price_outlier_interval_minutes()),
            initial_delay=timedelta(minutes=10),
        )

    def _run_price_outlier_scan(self) -> None:
        try:
            from src.services.db import get_conn
            from src.services.price_outlier import find_outliers, persist_findings

            with get_conn() as conn:
                with conn.cursor() as cur:
                    findings = find_outliers(cur)
                    written = persist_findings(cur, findings)
                conn.commit()
            logger.info(
                "price outlier scan: %d findings, %d new", len(findings), written)
        except Exception:
            logger.exception("price outlier scan failed")
```

Call `self._register_price_outlier_job()` wherever `_register_deal_assignment_job()` is called.

Create `scripts/run_price_outlier_scan.py`:

```python
"""Scan for extreme prices. Dry run by default.

    .venv/bin/python scripts/run_price_outlier_scan.py --target bp_testdb
    .venv/bin/python scripts/run_price_outlier_scan.py --target bp_testdb --write

Reports how many findings it WOULD raise before writing any, so the volume can
be judged before the Action Centre fills up.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

sys.path.insert(0, "src")

from scripts.testdata.db import connect  # noqa: E402
from services.price_outlier import find_outliers, persist_findings  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="bp_testdb")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    conn = connect(args.target)
    try:
        with conn.cursor() as cur:
            findings = find_outliers(cur)
            by_severity = Counter(f.verdict.severity for f in findings)
            print(f"{len(findings)} findings: {dict(by_severity)}")
            for finding in findings[:20]:
                print(f"  [{finding.verdict.severity}] {finding.note}")
            if not args.write:
                print("\ndry run — nothing written. Pass --write to persist.")
                return 0
            written = persist_findings(cur, findings)
        conn.commit()
        print(f"wrote {written} new findings")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_price_outlier_job.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/backend_scheduler.py scripts/run_price_outlier_scan.py tests/test_price_outlier_job.py
git commit -m "feat(price-outlier): scheduled scan, off by default, with a dry-run script"
```

---

### Task 14: Verification checks over the new data

**Files:**
- Modify: `scripts/testdata/verify.py`
- Test: `tests/testdata/test_verify.py`

**Interfaces:**
- Produces: `check_document_counts`, `check_line_sums`, `check_fx_rederives`, `check_benchmark_computes` — each returning `CheckResult`

- [ ] **Step 1: Write the failing test**

Add to `tests/testdata/test_verify.py`:

```python
from scripts.testdata import verify


def test_new_checks_are_registered_and_no_longer_skip():
    refs = {c.ref for c in verify.CHECKS}
    assert {"V01", "V04", "V05"} <= refs
    assert "V15" in refs, "the benchmark check needs its own ref"


def test_benchmark_check_is_blocking():
    assert verify.CHECK_BY_REF["V15"].blocking
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_verify.py -v
```

Expected: FAIL — `V15` is not in `CHECKS`.

- [ ] **Step 3: Write the implementation**

In `scripts/testdata/verify.py`, add to `CHECKS`:

```python
    Check("V15", "Benchmark computes on the busiest item", True),
```

Replace `check_row_counts` and add the three new checks:

```python
def check_row_counts(target_db: str) -> CheckResult:
    conn = connect(target_db)
    try:
        suppliers = _scalar(conn, "select count(*) from proc.bp_supplier")
        quotes = _scalar(conn, "select count(*) from proc.bp_quote_trgt")
        pool = _scalar(conn, """
            select (select count(*) from proc.bp_po_line_items_trgt)
                 + (select count(*) from proc.bp_invoice_line_items_trgt)""")
        passed = suppliers == 5000 and quotes > 0 and pool >= 70000
        return CheckResult(
            ref="V01", passed=passed,
            detail=(f"suppliers {suppliers} (expect 5000), quotes {quotes}, "
                    f"price-history lines {pool} (expect >=70000)"),
        )
    finally:
        conn.close()


def check_line_sums(target_db: str) -> CheckResult:
    """Line totals must reconcile to their header totals."""
    conn = connect(target_db)
    try:
        bad = _scalar(conn, """
            select count(*) from (
              select i.invoice_id
                from proc.bp_invoice_trgt i
                join proc.bp_invoice_line_items_trgt l on l.invoice_id = i.invoice_id
               group by i.invoice_id, i.invoice_amount
              having abs(sum(l.line_amount) - i.invoice_amount) > 0.01
            ) x""")
        return CheckResult(ref="V04", passed=bad == 0,
                           detail=f"{bad} invoices whose lines do not sum to the header")
    finally:
        conn.close()


def check_fx_rederives(target_db: str) -> CheckResult:
    """A non-sterling line must be denominated in its own currency, not
    sterling wearing a different label."""
    conn = connect(target_db)
    try:
        currencies = _scalar(conn, """
            select count(distinct currency) from proc.bp_quote_trgt
             where currency is not null""")
        return CheckResult(ref="V05", passed=currencies >= 4,
                           detail=f"{currencies} distinct quote currencies (expect >=4)")
    finally:
        conn.close()


def check_benchmark_computes(target_db: str) -> CheckResult:
    """The whole point of the pool: the engine must produce a number."""
    import sys
    sys.path.insert(0, "src")
    from services.benchmark.engine import compute_benchmark
    from services.benchmark.models import BenchmarkPoint, QuoteLine

    conn = connect(target_db)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select item_description, unit_of_measure, count(*) n
                  from proc.bp_po_line_items_trgt
                 where unit_price is not null
                 group by 1,2 order by n desc limit 1""")
            row = cur.fetchone()
            if not row:
                return CheckResult(ref="V15", passed=False,
                                   detail="no purchase-order lines to benchmark")
            item, uom, n = row
            cur.execute("""
                select po_line_id, unit_price, quantity
                  from proc.bp_po_line_items_trgt
                 where item_description = %s and unit_of_measure = %s
                   and unit_price is not null""", (item, uom))
            rows = cur.fetchall()

        points = [
            BenchmarkPoint(
                benchmark_point_id=str(pid), source="internal",
                item_name=item.strip().lower(), uom=(uom or "each").lower(),
                currency="GBP", include=True, raw_unit_price=float(price),
                source_weight=1.0, specification_score=5.0,
                location_cost_index=1.0, sla_score=5.0,
                historical_quantity=float(qty) if qty is not None else None,
                index_value_at_price_date=1.0,
            )
            for pid, price, qty in rows
        ]
        quote = QuoteLine(
            deal_id="verify", item_name=item.strip().lower(),
            quantity=10, uom=(uom or "each").lower(), currency="GBP",
            location="United Kingdom", requested_spec_score=5.0,
            requested_sla_score=5.0, index_id="",
            quoted_unit_price=float(rows[0][1]),
        )
        result = compute_benchmark(quote, points, {}, {})
        passed = not result.gated and result.confidence == "HIGH"
        return CheckResult(
            ref="V15", passed=passed,
            detail=(f"'{item[:40]}' n={result.n_total} "
                    f"confidence={result.confidence} gated={result.gated}"),
        )
    finally:
        conn.close()
```

and register them in `run_all`:

```python
    results = [
        check_row_counts(target_db),
        check_no_orphans(target_db),
        check_crosswalk(target_db, uicanvas_target_db),
        check_line_sums(target_db),
        check_fx_rederives(target_db),
        check_benchmark_computes(target_db),
    ]
```

- [ ] **Step 4: Run the tests**

```bash
.venv/bin/python -m pytest tests/testdata/test_verify.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/verify.py tests/testdata/test_verify.py
git commit -m "feat(testdata): verify document counts, line sums, currencies and a live benchmark"
```

---

### Task 15: Record the add-once rule

The workbook's formula adds delivery, implementation, support and risk once; the description in that same column says to multiply by quantity. Settled in favour of the formula.

**Files:**
- Modify: `src/services/benchmark/README.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Append the decision to the README**

Add to `src/services/benchmark/README.md`:

```markdown
## Decision: one-off costs are added once (2026-07-27)

Delivery, implementation, support and risk are added once per order, and the
discount is subtracted once. They are NOT multiplied by quantity.

The workbook contradicts itself here: the formula in column AT adds them once,
while the description written beside it says to multiply by quantity. The
formula wins, for three reasons. The field names all denote one-off charges —
multiplying a negotiated £500 rebate by 220 units to reach £110,000 is not what
a rebate means. The formula is what Excel has actually been computing, so it
produced the numbers the author reviewed. And a charge that genuinely scales
with quantity is part of the unit price, not a separate charge.

It also changes less than it appears: the same amount is added to the quoted
total AND the benchmark total, so it cancels in the total cost gap. A 220-unit
line at £120 against a £102 benchmark returns a £3,978 gap with no charges,
with £1,750 added once, and with £1,750 multiplied by quantity. Only the two
displayed absolute totals move.

When these charges are eventually extracted from documents, extraction
normalises to a one-off amount: "delivery £250 per order" contributes £250
once, and "£3 per unit delivery" belongs in the unit price.

**The workbook's column description still needs correcting, by hand, in Excel.**
Do NOT edit the workbook with openpyxl: it does not evaluate formulas and drops
cached values on save, which would destroy the very numbers
`scripts/export_benchmark_fixtures.py` reads with `data_only=True` to build the
golden parity fixtures.
```

- [ ] **Step 2: Confirm the fixtures still load**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_parity.py -q
```

Expected: PASS — the workbook was not touched.

- [ ] **Step 3: Commit**

```bash
git add src/services/benchmark/README.md
git commit -m "docs(benchmark): record that one-off costs are added once, not per unit"
```

---

### Task 16: Full build, measured detector, updated build log

**Files:**
- Modify: `docs/testdata/BUILD_LOG.md`

**Interfaces:** none — this is the end-to-end proof.

- [ ] **Step 1: Build the dataset**

```bash
.venv/bin/python -m scripts.testdata.build --target bp_testdb --uicanvas-target uicanvas_test --seed 42 --drop-first
```

Expected: exit 0. Record every printed row count.

- [ ] **Step 2: Verify by query, not by the log**

```bash
.venv/bin/python -c "
from scripts.testdata.db import connect
conn=connect('bp_testdb'); cur=conn.cursor()
for t in ['bp_supplier','bp_fx_rates','bp_quote_trgt','bp_quote_line_items_trgt',
          'bp_purchase_order_trgt','bp_po_line_items_trgt','bp_invoice_trgt',
          'bp_invoice_line_items_trgt']:
    cur.execute(f'select count(*) from proc.{t}'); print(f'{t:32s}{cur.fetchone()[0]:>9,}')
cur.execute('''select count(*) from proc.bp_quote_line_items_trgt where deal_id is not null''')
print('quote lines with a deal_id:', cur.fetchone()[0])
conn.close()"
```

Expected: every table non-zero; purchase-order plus invoice lines at least 70,000.

- [ ] **Step 3: Confirm determinism**

Run the build twice more in separate processes with the same seed and compare the checksum the build prints. Record the value and the FX snapshot timestamp.

- [ ] **Step 4: Dry-run the detector and measure it against the answer key**

```bash
.venv/bin/python scripts/run_price_outlier_scan.py --target bp_testdb
```

Then compare the flagged documents against `docs/testdata/answer-key.json`:

```bash
.venv/bin/python -c "
import json, sys
sys.path.insert(0,'src')
from scripts.testdata.db import connect
from services.price_outlier import find_outliers
key=json.load(open('docs/testdata/answer-key.json'))
planted={p.get('doc_id') for p in key['planted'] if p.get('doc_id')}
conn=connect('bp_testdb')
found={f.doc_pk for f in find_outliers(conn.cursor())}
hit=found & planted
print(f'flagged {len(found)}, planted {len(planted)}, overlap {len(hit)}')
print(f'precision {len(hit)/max(1,len(found)):.2%}  recall {len(hit)/max(1,len(planted)):.2%}')
conn.close()"
```

Record precision and recall. If the finding count is implausible against the ~1,065 findings already open, tighten `OutlierSettings` and re-run — do not enable the scheduled job on a detector nobody can keep up with.

- [ ] **Step 5: Write the findings and enable the job**

Only once the dry-run volume is judged acceptable:

```bash
.venv/bin/python scripts/run_price_outlier_scan.py --target bp_testdb --write
```

- [ ] **Step 6: Rewrite the build log**

Replace `docs/testdata/BUILD_LOG.md` with the actual numbers from this run. The previous version claimed 5,000 suppliers and passing checks against a database that held zero rows; the new one must be verified by query. Include: row counts per table, the FX snapshot timestamp, the determinism checksum, the deal-assignment result, the outlier dry-run count with precision and recall, and the full verification output.

- [ ] **Step 7: Run every test suite**

```bash
.venv/bin/python -m pytest tests/testdata/ -q -m "not integration"
PYTHONPATH=src .venv/bin/python -m pytest tests/test_benchmark_engine.py tests/test_benchmark_models.py tests/test_benchmark_parity.py tests/test_benchmark_api.py tests/test_benchmark_live_reads.py tests/test_price_outlier_rule.py tests/test_price_outlier_detector.py tests/test_price_outlier_persist.py -q
```

Expected: PASS throughout. The 58 original benchmark tests must all still pass.

- [ ] **Step 8: Commit**

```bash
git add docs/testdata/BUILD_LOG.md
git commit -m "docs(testdata): build log for the seeded corpus with a live price pool"
```
