# Test Dataset S1: Organisation and Catalogue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load 400 business units, 500 cost centres and 5,000 catalogue items into `uicanvas_test`, and verify the cost-centre → business-unit roll-up.

**Architecture:** A new `scripts/testdata/persist.py` holds one declarative `TableSpec` per target table — destination, column list, required set, and a row builder. Loading is one generic function driven by those specs, so adding a table in a later stage is data, not new code. `build.py` calls it after the existing supplier write.

**Tech Stack:** Python 3.12 (`.venv`), psycopg2, pytest, existing `db.copy_rows` COPY loader.

**Spec:** `docs/superpowers/specs/2026-07-27-testdata-persistence-design.md`
**Coverage evidence:** `docs/testdata/BP_Schema_Coverage.xlsx`, tab "Delivery Stages"

## Global Constraints

- **Never write to a live database.** `guards.assert_safe_target` refuses `bp_sqldb`, `uicanvas`, `ses`, `postgres`, `rdsadmin`. No override flag.
- Integration tests target `bp_testdb_it` / `uicanvas_test_it` only. Never open a connection to `bp_testdb` or `uicanvas_test` from a test — `test_scratch_isolation.py` enforces this.
- The seeder writes inputs, never outputs. No `deal_id`, rankings, evaluations, decisions or summaries.
- Deterministic: the same `--seed` produces byte-identical rows.
- New DB tables use the `bp_` prefix; indexes `ix_bp_<table>_<column>`. (S1 creates no new tables.)
- Work on branch `Development`. Do not push to `main`.
- Commit messages carry no AI attribution or `Co-Authored-By` lines.
- Run tests with `.venv/bin/python -m pytest`.

## Two schema facts this stage must respect

1. **There is no entity/organisation table.** `business_unit` (16 columns) and `cost_centre` (26 columns) have no `org_id`. The entity dimension exists only in the generator's `Entity` objects. V07 therefore verifies cost centre → business unit in the database, and reports the entity level as unverifiable-in-schema rather than silently claiming it passed.
2. **`category_level_5_id` is not unique** — 121 distinct values across 246 leaves (`C-5101` is both "GL & Consolidation" and "Requisitioning"). It identifies a level-5 node within its branch, not a leaf globally. Nothing may assume uniqueness.

---

### Task 1: Taxonomy leaves carry their level identifiers

`cost_centre.linked_category_level_5_id` and `item.category_id` both want
`bp_category.category_level_5_id`. `TaxonomyLeaf` does not currently load it, and
`org.py` puts the UNSPSC code in that column instead — a real bug shipped in Plan 1.

**Files:**
- Modify: `scripts/testdata/reference.py` (`TaxonomyLeaf`, `load_taxonomy`)
- Modify: `scripts/testdata/org.py:199`
- Test: `tests/testdata/test_reference.py`, `tests/testdata/test_org.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `TaxonomyLeaf.l1_id … l5_id: str | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/testdata/test_reference.py`:

```python
@pytest.mark.integration
def test_taxonomy_leaves_carry_their_level_identifiers():
    leaves = load_taxonomy("uicanvas")
    for leaf in leaves:
        assert leaf.l5_id, leaf.l5
        assert leaf.l1_id
    # Not unique: the same level-5 id appears under different branches.
    ids = {leaf.l5_id for leaf in leaves}
    assert 0 < len(ids) < len(leaves)
```

Append to `tests/testdata/test_org.py`:

```python
def test_cost_centre_links_the_real_category_level_5_id():
    """Plan 1 put the UNSPSC code in this column. It wants the L5 id."""
    units = build_business_units(42)
    leaves = _leaves(50)
    centres = build_cost_centres(42, units, leaves)
    valid = {leaf.l5_id for leaf in leaves}
    for centre in centres:
        assert centre.linked_category_level_5_id in valid
```

Update `_leaves` in `tests/testdata/test_org.py` to pass the new fields:

```python
def _leaves(count: int) -> list[TaxonomyLeaf]:
    return [
        TaxonomyLeaf(
            l1="IT & Technology", l2="Software", l3="ERP", l4=f"Sub{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000", l4_id=f"C-45{i:02d}",
            l5_id=f"C-51{i:02d}",
            unspsc_code=str(10000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(count)
    ]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_org.py -q -k category_level_5
```

Expected: FAIL — `TypeError: TaxonomyLeaf.__init__() got an unexpected keyword argument 'l1_id'`

- [ ] **Step 3: Write the implementation**

In `scripts/testdata/reference.py`, add the id fields to the dataclass immediately
after `l5` so positional construction from the query still works:

```python
@dataclass(frozen=True)
class TaxonomyLeaf:
    l1: str | None
    l2: str | None
    l3: str | None
    l4: str | None
    l5: str | None
    l1_id: str | None
    l2_id: str | None
    l3_id: str | None
    l4_id: str | None
    l5_id: str | None
    unspsc_code: str | None
    esg_impact: str | None
    category_status: str | None
    spend_classification: str | None
    category_risk_rating: str | None
    audit_frequency: str | None
    policy_coverage: str | None
```

and change the query in `load_taxonomy` to select them in that order:

```python
            cur.execute(
                """
                select category_level_1, category_level_2, category_level_3,
                       category_level_4, category_level_5,
                       category_level_1_id, category_level_2_id, category_level_3_id,
                       category_level_4_id, category_level_5_id,
                       unspsc_code, esg_impact,
                       category_status, spend_classification, category_risk_rating,
                       audit_frequency, policy_coverage
                from proc.bp_category
                order by 1, 2, 3, 4, 5
                """
            )
```

In `scripts/testdata/org.py`, replace line 199:

```python
                    linked_category_level_5_id=leaf.l5_id or "UNKNOWN",
```

- [ ] **Step 4: Fix the other TaxonomyLeaf constructions**

Every test that builds a `TaxonomyLeaf` by keyword needs the five new fields.
Apply the same five lines (`l1_id="C-2000", l2_id="C-3000", l3_id="C-4000",
l4_id=f"C-45{i:02d}", l5_id=f"C-51{i:02d}",`) to the `_leaves` helper in
`tests/testdata/test_suppliers.py`, `tests/testdata/test_catalogue.py`,
`tests/testdata/test_documents.py` and `tests/testdata/test_defects.py`.

- [ ] **Step 5: Run the whole suite**

```bash
.venv/bin/python -m pytest tests/testdata/ -q
```

Expected: PASS — all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/testdata/reference.py scripts/testdata/org.py tests/testdata/
git commit -m "fix(testdata): cost centres link the real category level-5 id

The column wants bp_category.category_level_5_id; Plan 1 wrote the UNSPSC code
into it. TaxonomyLeaf now carries all five level identifiers, which the
catalogue also needs to resolve a line to a category path."
```

---

### Task 2: The table specification framework

**Files:**
- Create: `scripts/testdata/persist.py`
- Test: `tests/testdata/test_persist.py`

**Interfaces:**
- Consumes: `db.connect`, `db.copy_rows`, `guards.assert_safe_target`
- Produces:
  - `@dataclass(frozen=True) TableSpec: database, table, columns: tuple[str, ...], required: tuple[str, ...], build: Callable[[Any], list]`
  - `load_table(target_db: str, spec: TableSpec, rows: Iterable[Any]) -> int`
  - `MissingRequiredValue(RuntimeError)`

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_persist.py`:

```python
import pytest

from scripts.testdata.guards import UnsafeTargetError
from scripts.testdata.persist import MissingRequiredValue, TableSpec, load_table


def _spec(**overrides) -> TableSpec:
    base = dict(
        database="uicanvas",
        table="probe",
        columns=("a", "b"),
        required=("a",),
        build=lambda item: [item, None],
    )
    base.update(overrides)
    return TableSpec(**base)


def test_spec_declares_its_columns_and_required_set():
    spec = _spec()
    assert spec.columns == ("a", "b")
    assert spec.required == ("a",)


def test_required_columns_must_be_a_subset_of_columns():
    with pytest.raises(ValueError, match="not a column"):
        _spec(required=("nope",))


def test_load_refuses_a_live_target():
    with pytest.raises(UnsafeTargetError):
        load_table("uicanvas", _spec(), [1, 2])


def test_a_missing_required_value_is_an_error_not_a_silent_null():
    """These tables have almost no NOT NULL constraints; the DB will not catch it."""
    spec = _spec(build=lambda item: [None, item])
    with pytest.raises(MissingRequiredValue, match="probe.a"):
        load_table("scratch_db", spec, [1], _dry_run=True)


def test_dry_run_returns_the_row_count_without_a_database():
    assert load_table("scratch_db", _spec(), [1, 2, 3], _dry_run=True) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.persist'`

- [ ] **Step 3: Write the implementation**

Create `scripts/testdata/persist.py`:

```python
"""Declarative loading of generated data into the test databases.

One TableSpec per target table: where it goes, which columns, which of those
must carry a value, and how to build a row from a domain object. Loading is one
generic function driven by those specs, so a new table is data rather than code.

The required set matters more than it looks. Of the tables this package writes,
almost none carries a NOT NULL constraint -- a load that put NULL in every
invoice amount would be accepted in silence. The declared required set is the
only thing that catches it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from scripts.testdata.db import connect, copy_rows
from scripts.testdata.guards import assert_safe_target


class MissingRequiredValue(RuntimeError):
    """A column declared required came out NULL."""


@dataclass(frozen=True)
class TableSpec:
    database: str          # "bp_sqldb" or "uicanvas": which side it belongs to
    table: str
    columns: tuple[str, ...]
    required: tuple[str, ...]
    build: Callable[[Any], Sequence[Any]]

    def __post_init__(self) -> None:
        unknown = [c for c in self.required if c not in self.columns]
        if unknown:
            raise ValueError(f"required entries are not a column of {self.table}: {unknown}")


def load_table(
    target_db: str,
    spec: TableSpec,
    rows: Iterable[Any],
    *,
    _dry_run: bool = False,
) -> int:
    """Truncate and bulk-load one table. Returns the number of rows written."""
    assert_safe_target(target_db)

    required_positions = [(spec.columns.index(c), c) for c in spec.required]
    built: list[Sequence[Any]] = []
    for item in rows:
        values = spec.build(item)
        if len(values) != len(spec.columns):
            raise ValueError(
                f"{spec.table}: builder returned {len(values)} values for "
                f"{len(spec.columns)} columns"
            )
        for position, name in required_positions:
            if values[position] is None or values[position] == "":
                raise MissingRequiredValue(f"{spec.table}.{name} is required but came out empty")
        built.append(values)

    if _dry_run:
        return len(built)

    conn = connect(target_db)
    try:
        with conn.cursor() as cur:
            cur.execute(f'truncate proc."{spec.table}"')
        conn.commit()
        written = copy_rows(conn, "proc", spec.table, list(spec.columns), built)
        conn.commit()
    finally:
        conn.close()
    return written
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q
```

Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/persist.py tests/testdata/test_persist.py
git commit -m "feat(testdata): declarative table specifications for loading"
```

---

### Task 3: Business unit and cost centre specifications

**Files:**
- Modify: `scripts/testdata/persist.py`
- Test: `tests/testdata/test_persist.py`

**Interfaces:**
- Consumes: `org.BusinessUnit`, `org.CostCentre`, `TableSpec`
- Produces: `BUSINESS_UNIT_SPEC: TableSpec`, `COST_CENTRE_SPEC: TableSpec`, `AUDIT_USER: str = "testdata"`

- [ ] **Step 1: Write the failing test**

Append to `tests/testdata/test_persist.py`:

```python
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.persist import BUSINESS_UNIT_SPEC, COST_CENTRE_SPEC


def _leaves(count: int = 50):
    from scripts.testdata.reference import TaxonomyLeaf

    return [
        TaxonomyLeaf(
            l1="IT & Technology", l2="Software", l3="ERP", l4=f"Sub{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000", l4_id=f"C-45{i:02d}",
            l5_id=f"C-51{i:02d}",
            unspsc_code=str(10000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(count)
    ]


def test_business_unit_spec_has_all_sixteen_columns():
    assert len(BUSINESS_UNIT_SPEC.columns) == 16
    assert BUSINESS_UNIT_SPEC.table == "business_unit"
    assert "business_unit_id" in BUSINESS_UNIT_SPEC.required


def test_cost_centre_spec_has_all_twenty_six_columns():
    assert len(COST_CENTRE_SPEC.columns) == 26
    assert COST_CENTRE_SPEC.table == "cost_centre"
    assert "cost_centre_level_id" in COST_CENTRE_SPEC.required


def test_every_business_unit_builds_a_full_row():
    units = build_business_units(42)
    for unit in units[:20]:
        row = BUSINESS_UNIT_SPEC.build(unit)
        assert len(row) == 16
        assert row[BUSINESS_UNIT_SPEC.columns.index("business_unit_id")] == unit.bu_id
        assert row[BUSINESS_UNIT_SPEC.columns.index("business_unit_level_5")] == unit.l5


def test_every_cost_centre_builds_a_full_row_with_six_levels():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves())
    for centre in centres[:20]:
        row = COST_CENTRE_SPEC.build(centre)
        assert len(row) == 26
        assert row[COST_CENTRE_SPEC.columns.index("cost_centre_level_id")] == centre.cc_id
        assert row[COST_CENTRE_SPEC.columns.index("cost_centre_level_6")]
        assert row[COST_CENTRE_SPEC.columns.index("business_unit_id")] == centre.bu_id


def test_cost_centre_document_links_are_left_null():
    """po_id and invoice_id are filled by stage S3, not invented here."""
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves())
    row = COST_CENTRE_SPEC.build(centres[0])
    assert row[COST_CENTRE_SPEC.columns.index("po_id")] is None
    assert row[COST_CENTRE_SPEC.columns.index("invoice_id")] is None


def test_org_rows_pass_their_own_required_check():
    from scripts.testdata.persist import load_table

    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves())
    assert load_table("scratch_db", BUSINESS_UNIT_SPEC, units, _dry_run=True) == 400
    assert load_table("scratch_db", COST_CENTRE_SPEC, centres, _dry_run=True) == 500
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q -k "business_unit_spec or cost_centre_spec"
```

Expected: FAIL — `ImportError: cannot import name 'BUSINESS_UNIT_SPEC'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/testdata/persist.py`:

```python
from datetime import datetime

from scripts.testdata.org import BusinessUnit, CostCentre

AUDIT_USER = "testdata"
AUDIT_STAMP = datetime(2026, 7, 27, 0, 0, 0)

BUSINESS_UNIT_SPEC = TableSpec(
    database="uicanvas",
    table="business_unit",
    columns=(
        "business_unit_level_1_id", "business_unit_level_1", "business_unit_level_2",
        "business_unit_level_3", "business_unit_level_4", "business_unit_level_5",
        "business_unit_id", "business_unit_head_name", "business_unit_head_email",
        "region", "bu_status", "notes", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    required=(
        "business_unit_id", "business_unit_level_1", "business_unit_level_5",
        "business_unit_head_email", "bu_status",
    ),
    build=lambda unit: [
        # The schema carries an id for level 1 only; derive it from the name so
        # the same function always maps to the same identifier.
        _bu_level1_id(unit.l1),
        unit.l1, unit.l2, unit.l3, unit.l4, unit.l5,
        unit.bu_id, unit.head_name, unit.head_email,
        unit.region, unit.status, None,
        AUDIT_STAMP, AUDIT_USER, AUDIT_USER, AUDIT_STAMP,
    ],
)

COST_CENTRE_SPEC = TableSpec(
    database="uicanvas",
    table="cost_centre",
    columns=(
        "cost_centre_level_id", "cost_centre_level_1", "cost_centre_level_2",
        "cost_centre_level_3", "cost_centre_level_4", "cost_centre_level_5",
        "cost_centre_level_6", "business_unit_id", "finance_account_code",
        "cost_centre_manager_name", "cost_centre_manager_email", "is_active",
        "spend_threshold_limit", "currency", "po_id", "invoice_id",
        "budget_allocated_annual", "actual_spend_ytd", "forecast_spend_annual",
        "cost_centre_type", "linked_category_level_5_id", "notes",
        "created_date", "created_by", "last_modified_by", "last_modified_date",
    ),
    required=(
        "cost_centre_level_id", "cost_centre_level_1", "cost_centre_level_6",
        "business_unit_id", "currency", "cost_centre_type",
        "linked_category_level_5_id",
    ),
    build=lambda centre: [
        centre.cc_id, *centre.levels,
        centre.bu_id, centre.finance_account_code,
        centre.manager_name, centre.manager_email, centre.is_active,
        centre.spend_threshold_limit, centre.currency,
        # Document links belong to stage S3; inventing them here would assert a
        # relationship that no document backs.
        None, None,
        centre.budget_allocated_annual, centre.actual_spend_ytd,
        centre.forecast_spend_annual, centre.cost_centre_type,
        centre.linked_category_level_5_id, None,
        AUDIT_STAMP, AUDIT_USER, AUDIT_USER, AUDIT_STAMP,
    ],
)
```

and the level-1 id helper above the specs:

```python
_BU_LEVEL1_IDS: dict[str, str] = {}


def _bu_level1_id(name: str) -> str:
    """A stable id per level-1 function. The schema has an id column for level 1
    only, and the generator's tree carries names, so one is assigned in first-seen
    order -- deterministic because the tree itself is."""
    if name not in _BU_LEVEL1_IDS:
        _BU_LEVEL1_IDS[name] = f"BU-L1-{len(_BU_LEVEL1_IDS) + 1:03d}"
    return _BU_LEVEL1_IDS[name]
```

`_bu_level1_id` must be defined above `BUSINESS_UNIT_SPEC`: the spec's `build`
lambda closes over it, and the module-level constant is evaluated at import.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q
```

Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/persist.py tests/testdata/test_persist.py
git commit -m "feat(testdata): business unit and cost centre table specifications"
```

---

### Task 4: Catalogue item specification

**Files:**
- Modify: `scripts/testdata/persist.py`
- Test: `tests/testdata/test_persist.py`

**Interfaces:**
- Consumes: `catalogue.CatalogueItem`, `TableSpec`
- Produces: `ITEM_SPEC: TableSpec`

- [ ] **Step 1: Write the failing test**

Append to `tests/testdata/test_persist.py`:

```python
def test_item_spec_has_all_fifteen_columns():
    from scripts.testdata.persist import ITEM_SPEC

    assert len(ITEM_SPEC.columns) == 15
    assert ITEM_SPEC.table == "item"
    assert "item_id" in ITEM_SPEC.required
    assert "category_id" in ITEM_SPEC.required


def test_item_rows_carry_the_real_category_level_5_id():
    from scripts.testdata.catalogue import build_catalogue
    from scripts.testdata.persist import ITEM_SPEC

    leaves = _leaves()
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(50)])
    valid = {leaf.l5_id for leaf in leaves}
    for item in items[:50]:
        row = ITEM_SPEC.build(item)
        assert len(row) == 15
        assert row[ITEM_SPEC.columns.index("category_id")] in valid
        assert row[ITEM_SPEC.columns.index("standard_price")] > 0
        assert row[ITEM_SPEC.columns.index("item_name")] == item.description


def test_all_five_thousand_items_pass_the_required_check():
    from scripts.testdata.catalogue import build_catalogue
    from scripts.testdata.persist import ITEM_SPEC, load_table

    items = build_catalogue(42, _leaves(246), [f"SUP-S{i}" for i in range(500)])
    assert load_table("scratch_db", ITEM_SPEC, items, _dry_run=True) == 5000
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q -k item_spec
```

Expected: FAIL — `ImportError: cannot import name 'ITEM_SPEC'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/testdata/persist.py`:

```python
from scripts.testdata.catalogue import CatalogueItem

ITEM_SPEC = TableSpec(
    database="uicanvas",
    table="item",
    columns=(
        "item_id", "item_name", "category_id", "unit", "standard_price",
        "currency", "preferred_supplier_id", "manufacturer", "brand",
        "spec_sheet_url", "uom_conversion", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    required=(
        "item_id", "item_name", "category_id", "unit", "standard_price",
        "currency", "preferred_supplier_id",
    ),
    build=lambda item: [
        item.item_id, item.description, item.leaf.l5_id or "UNKNOWN",
        item.unit_of_measure, item.base_price, item.currency,
        item.preferred_supplier_id,
        # Manufacturer, brand, spec sheet and UOM conversion are not modelled by
        # the generator. Inventing them would put unverifiable strings in columns
        # nothing reads.
        None, None, None, None,
        AUDIT_STAMP, AUDIT_USER, AUDIT_USER, AUDIT_STAMP,
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q
```

Expected: PASS — 14 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/persist.py tests/testdata/test_persist.py
git commit -m "feat(testdata): catalogue item specification on the real L5 identifier"
```

---

### Task 5: Load the three tables in the build

**Files:**
- Modify: `scripts/testdata/persist.py` (add `load_stage_one`)
- Modify: `scripts/testdata/build.py`
- Test: `tests/testdata/test_persist.py`

**Interfaces:**
- Consumes: the three specs
- Produces: `load_stage_one(uicanvas_target_db, units, centres, items) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

Append to `tests/testdata/test_persist.py`:

```python
@pytest.mark.integration
def test_stage_one_loads_all_three_tables(scratch_uicanvas_schema):
    from scripts.testdata.catalogue import build_catalogue
    from scripts.testdata.db import connect
    from scripts.testdata.persist import load_stage_one
    from tests.testdata import SCRATCH_UICANVAS_DB

    leaves = _leaves(246)
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(500)])

    written = load_stage_one(SCRATCH_UICANVAS_DB, units, centres, items)
    assert written == {"business_unit": 400, "cost_centre": 500, "item": 5000}

    conn = connect(SCRATCH_UICANVAS_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("select count(*) from proc.business_unit")
            assert cur.fetchone()[0] == 400
            cur.execute("select count(*) from proc.cost_centre where business_unit_id is not null")
            assert cur.fetchone()[0] == 500
            cur.execute("select count(*) from proc.item where category_id is null")
            assert cur.fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.integration
def test_stage_one_is_idempotent(scratch_uicanvas_schema):
    from scripts.testdata.catalogue import build_catalogue
    from scripts.testdata.persist import load_stage_one
    from tests.testdata import SCRATCH_UICANVAS_DB

    leaves = _leaves(246)
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(500)])

    first = load_stage_one(SCRATCH_UICANVAS_DB, units, centres, items)
    second = load_stage_one(SCRATCH_UICANVAS_DB, units, centres, items)
    assert first == second
```

Add the fixture to `tests/testdata/conftest.py`:

```python
@pytest.fixture(scope="session")
def scratch_uicanvas_schema():
    """Clone uicanvas's structure into the scratch database, once per session."""
    from scripts.testdata.schema import clone_schema
    from tests.testdata import SCRATCH_UICANVAS_DB

    clone_schema("uicanvas", SCRATCH_UICANVAS_DB, drop_first=True)
    return SCRATCH_UICANVAS_DB
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_persist.py -q -m integration
```

Expected: FAIL — `ImportError: cannot import name 'load_stage_one'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/testdata/persist.py`:

```python
def load_stage_one(
    uicanvas_target_db: str,
    units: Sequence[BusinessUnit],
    centres: Sequence[CostCentre],
    items: Sequence[CatalogueItem],
) -> dict[str, int]:
    """Stage S1: organisation and catalogue. Order matters -- cost centres
    reference business units."""
    return {
        "business_unit": load_table(uicanvas_target_db, BUSINESS_UNIT_SPEC, units),
        "cost_centre": load_table(uicanvas_target_db, COST_CENTRE_SPEC, centres),
        "item": load_table(uicanvas_target_db, ITEM_SPEC, items),
    }
```

In `scripts/testdata/build.py`, import it:

```python
from scripts.testdata.persist import load_stage_one
```

and insert after the `_write_suppliers(...)` call:

```python
    print("loading organisation and catalogue")
    loaded = load_stage_one(args.uicanvas_target, units, centres, items)
    for table, count in loaded.items():
        print(f"  proc.{table}: {count} rows")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/testdata/ -q
```

Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/persist.py scripts/testdata/build.py tests/testdata/
git commit -m "feat(testdata): load organisation and catalogue in the build"
```

---

### Task 6: Verification check V07

**Files:**
- Modify: `scripts/testdata/verify.py`
- Test: `tests/testdata/test_verify.py`

**Interfaces:**
- Consumes: `db.connect`
- Produces: `check_rollup(uicanvas_target_db: str) -> CheckResult`

The schema has no entity column, so the check verifies what the schema can
express — every cost centre resolves to a business unit that exists — and names
the entity level as unverifiable rather than claiming it passed.

- [ ] **Step 1: Write the failing test**

Append to `tests/testdata/test_verify.py`:

```python
def test_v07_is_declared_blocking():
    from scripts.testdata.verify import CHECK_BY_REF

    assert CHECK_BY_REF["V07"].blocking is True


@pytest.mark.integration
def test_rollup_check_passes_on_a_loaded_scratch_database(scratch_uicanvas_schema):
    from scripts.testdata.catalogue import build_catalogue
    from scripts.testdata.org import build_business_units, build_cost_centres
    from scripts.testdata.persist import load_stage_one
    from scripts.testdata.reference import TaxonomyLeaf
    from scripts.testdata.verify import check_rollup
    from tests.testdata import SCRATCH_UICANVAS_DB

    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2="Software", l3="ERP", l4=f"Sub{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000", l4_id=f"C-45{i:02d}",
            l5_id=f"C-51{i:02d}",
            unspsc_code=str(10000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(246)
    ]
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(500)])
    load_stage_one(SCRATCH_UICANVAS_DB, units, centres, items)

    result = check_rollup(SCRATCH_UICANVAS_DB)
    assert result.ref == "V07"
    assert result.passed, result.detail
    assert "entity" in result.detail.lower()


@pytest.mark.integration
def test_rollup_check_fails_when_a_cost_centre_points_nowhere(scratch_uicanvas_schema):
    from scripts.testdata.db import connect
    from scripts.testdata.verify import check_rollup
    from tests.testdata import SCRATCH_UICANVAS_DB

    conn = connect(SCRATCH_UICANVAS_DB)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "insert into proc.cost_centre (cost_centre_level_id, business_unit_id) "
                "values ('CC-ORPHAN', 'BU-DOES-NOT-EXIST')"
            )
        conn.commit()
        result = check_rollup(SCRATCH_UICANVAS_DB)
        assert not result.passed
        assert "1" in result.detail
    finally:
        with conn.cursor() as cur:
            cur.execute("delete from proc.cost_centre where cost_centre_level_id = 'CC-ORPHAN'")
        conn.commit()
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_verify.py -q -k rollup
```

Expected: FAIL — `ImportError: cannot import name 'check_rollup'`

- [ ] **Step 3: Write the implementation**

Add to `scripts/testdata/verify.py`:

```python
def check_rollup(uicanvas_target_db: str) -> CheckResult:
    """V07: every cost centre resolves to a business unit that exists.

    The schema carries no entity column on either table -- there is no
    organisation table at all -- so the entity and group levels of the roll-up
    cannot be verified here. That is stated in the detail rather than passed
    over, because a check that quietly narrows its own scope is worse than one
    that fails.
    """
    conn = connect(uicanvas_target_db)
    try:
        centres = _scalar(conn, "select count(*) from proc.cost_centre")
        units = _scalar(conn, "select count(*) from proc.business_unit")
        orphans = _scalar(
            conn,
            """
            select count(*) from proc.cost_centre c
            where c.business_unit_id is null
               or not exists (
                   select 1 from proc.business_unit b
                   where b.business_unit_id = c.business_unit_id
               )
            """,
        )
        return CheckResult(
            ref="V07",
            passed=orphans == 0 and centres > 0 and units > 0,
            detail=(
                f"{centres} cost centres over {units} business units, "
                f"{orphans} unresolved; entity and group levels not verifiable "
                f"(no entity column in the schema)"
            ),
        )
    finally:
        conn.close()
```

and wire it into `run_all`, replacing the `NOT_YET_IMPLEMENTED` entry:

```python
    results = [
        check_row_counts(target_db),
        check_no_orphans(target_db),
        check_crosswalk(target_db, uicanvas_target_db),
        check_rollup(uicanvas_target_db),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_verify.py -q
```

Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/verify.py tests/testdata/test_verify.py
git commit -m "feat(testdata): V07 verifies the cost centre to business unit roll-up"
```

---

### Task 7: Full build and stage sign-off

**Files:**
- Modify: `docs/testdata/BUILD_LOG.md`

- [ ] **Step 1: Capture live row counts**

```bash
.venv/bin/python -c "
from scripts.testdata.guards import snapshot_counts
import json, pathlib
counts = snapshot_counts(['bp_sqldb', 'uicanvas'])
pathlib.Path('/tmp/live_before_s1.json').write_text(json.dumps(counts, indent=2, sort_keys=True))
print(f'{len(counts)} tables, {sum(counts.values())} rows total')
"
```

- [ ] **Step 2: Run the full build**

```bash
.venv/bin/python -m scripts.testdata.build --target bp_testdb --uicanvas-target uicanvas_test --seed 42 --drop-first
```

Expected: exit 0, `business_unit: 400`, `cost_centre: 500`, `item: 5000`, and V07 PASS.

- [ ] **Step 3: Prove the live databases are unchanged**

```bash
.venv/bin/python -c "
from scripts.testdata.guards import assert_live_unchanged, snapshot_counts
import json, pathlib
before = json.loads(pathlib.Path('/tmp/live_before_s1.json').read_text())
assert_live_unchanged(before, snapshot_counts(['bp_sqldb', 'uicanvas']))
print('ISOLATION PROVEN')
"
```

- [ ] **Step 4: Regenerate the coverage workbook**

```bash
.venv/bin/python -m scripts.testdata.schema_workbook
```

The three S1 tables must now show non-zero Test-DB rows.

- [ ] **Step 5: Record the result and commit**

Add an S1 section to `docs/testdata/BUILD_LOG.md` with the actual row counts, the
V07 detail string, and the isolation result. Then:

```bash
git add -f docs/testdata/BUILD_LOG.md docs/testdata/BP_Schema_Coverage.xlsx
git commit -m "docs(testdata): record stage S1 load"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §3.1 organisation (business_unit, cost_centre) | 3, 5 |
| §3.1 catalogue (item) | 4, 5 |
| §4.1 declarative mappings and required sets | 2 |
| §4.1 the database will not catch mistakes | 2 (`MissingRequiredValue`) |
| §4.2 load order | 5 (`load_stage_one` orders BU before CC) |
| §5 roll-up and currency | 6 |
| §7 testing, scratch databases only | 5, 6 (fixtures) |
| §8.1 build reports row counts | 5 |
| §8.4 V07 passes | 6 |
| §8.6 idempotent re-run | 5 |
| §8.7 live unchanged | 7 |
| §8.9 workbook shows no UNCLEAR | 7 |

**Deferred by design:** `cost_centre.po_id` / `invoice_id` are left NULL until
stage S3 has documents to point at. `item.manufacturer`, `brand`,
`spec_sheet_url` and `uom_conversion` stay NULL — the generator does not model
them, and inventing values would put unverifiable strings in columns nothing reads.

**Type consistency:** `TaxonomyLeaf.l5_id` (Task 1) is consumed by Task 3
(`linked_category_level_5_id`) and Task 4 (`category_id`). `TableSpec` (Task 2)
is used unchanged by Tasks 3, 4 and 5. `load_table`'s `_dry_run` keyword is used
by Tasks 3 and 4 and not by Task 5. `CheckResult` in Task 6 matches the existing
dataclass in `verify.py`.

**Known rough edge:** `_bu_level1_id` assigns identifiers in first-seen order via
module-level state. That is deterministic within a process because the business
unit tree is deterministic, but it means the mapping is not stable if the set of
level-1 function names ever changes. The six names are a fixed tuple in `org.py`,
so this holds today; if that tuple becomes dynamic, the id must move into `org.py`
and be generated with the tree.
