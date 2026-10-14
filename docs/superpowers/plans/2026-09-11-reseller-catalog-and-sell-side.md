# Reseller Catalog, Cost and Sell-Side Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish spec build steps 1–6: harden the committed catalog importer, ship the DDL as real migrations, then add SKU↔history matching, accounts/opportunities, outbound quotes with a customer-facing serialiser allowlist, outcome capture and win-probability calibration — with HTTP endpoints and live acceptance tests.

**Architecture:** Deterministic services, no model calls anywhere. `src/services/catalog_import.py` (exists) and a new `src/services/catalog_match.py` own the catalog; a new package `src/services/sell_side/` owns accounts, opportunities, quotes, rendering, outcomes and calibration. Pure arithmetic/logic lives in small modules tested without a database; SQL paths are tested against the live `bp_testdb` behind `PROCWISE_TEST_LIVE_DB=1` with sentinel-prefixed rows cleaned up. Two routers (`/catalog`, `/sales`) expose it behind the existing gate.

**Tech Stack:** Python 3, FastAPI, psycopg2 (`RealDictCursor`), rapidfuzz 3.x, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md` — read §4 (decisions), §7 (build order), §9 (acceptance criteria) before any task.

## Global Constraints

- New tables use the `bp_` prefix; indexes `ix_bp_<table>_<cols>`. (Tables already exist in the DDL — do not rename.)
- **No model call** anywhere in this feature. Catalog data is asserted, not extracted.
- **Absence stays absent:** a missing cost/price/quantity yields `NULL`, never `0`, never a guess. A margin over only some lines is not the quote's margin → header margin is `NULL`.
- **No FX conversion.** A catalog item priced in one currency on a quote in another is refused, not converted.
- Money: `Decimal` only, never float. 2dp amounts / 4dp unit prices and ratios, `ROUND_HALF_UP`.
- Identity on any write (`created_by`, `approved_by`, `recorded_by`, `confirmed_by`, `imported_by`) comes from the token: `getattr(principal, "subject", None)`. **Never** from a request body.
- Every write endpoint calls `gate(<action>, principal, agent=..., context=...)` from `api.endpoint_gate` before touching data.
- Cost and margin fields (`unit_cost`, `cost_tier_applied`, `line_margin`, `line_margin_pct`, `total_cost`, `total_margin`, `margin_pct`, `expected_cost`, `expected_margin`, `cost_price`, `cost_basis`, `customer_safe`) never reach a customer-facing payload.
- `win_probability` is written **only** by the calibration job, and only for groups with ≥ `calibration_min_closed` closed outcomes.
- Governed numbers come from `src.services.governed_limits.limit(...)`; a missing policy value raises.
- Tests: `./venv/bin/python -m pytest`, after `set -a; . /home/muthu/PycharmProjects/BP_Backend/.env; set +a`. Live DB tests need `PROCWISE_TEST_LIVE_DB=1` or they silently SKIP — a skip is not a pass.
- Never bare `git stash`; never push to `main`. Commits end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QqGRjqTTacQQ3FjY4sfmq2
  ```
- Every guard gets a mutation proof: break it on purpose, watch a test go red, restore. Record the red test name in the commit body.

## File Map

| File | Responsibility |
|---|---|
| `src/services/catalog_import.py` (modify) | Feed → versioned `bp_catalog_item`; + `save_mapping`/`get_mapping` |
| `src/services/catalog_match.py` (create) | Propose / confirm / reject SKU ↔ `item_id` matches |
| `src/services/sell_side/__init__.py` (create) | empty |
| `src/services/sell_side/_db.py` (create) | `dict_cursor(conn)`, error classes |
| `src/services/sell_side/ladder.py` (create) | Verbatim sales-ladder ids + validation |
| `src/services/sell_side/money.py` (create) | Line/quote arithmetic, pure |
| `src/services/sell_side/costing.py` (create) | Cost at a quantity (volume tiers), catalog snapshot |
| `src/services/sell_side/accounts.py` (create) | Account, contacts, history scope |
| `src/services/sell_side/opportunities.py` (create) | Opportunity + justification writes |
| `src/services/sell_side/quotes.py` (create) | Draft, submit, approve, issue, supersede, read |
| `src/services/sell_side/quote_render.py` (create) | Customer view via allowlist + leak guard, HTML |
| `src/services/sell_side/outcomes.py` (create) | Won/lost capture |
| `src/services/sell_side/calibration.py` (create) | Win-probability calibration |
| `src/services/backend_scheduler.py` (modify) | Daily calibration job |
| `src/services/actions.py` (modify) | Six new action names |
| `src/api/routers/catalog.py`, `src/api/routers/sales.py` (create) | HTTP |
| `src/api/main.py` (modify) | Register both routers as authenticated |
| `deploy/sql/2026-09-11_bp_catalog.sql` (+`_rollback`) | moved from `sql/bp_catalog.sql` |
| `deploy/sql/2026-09-11_bp_sell_side.sql` (+`_rollback`) | moved from `sql/bp_sell_side.sql` |
| `deploy/sql/2026-09-11_reseller_governance.sql` (+`_rollback`) | limit row + two permit rows |
| `tests/conftest.py`, `tests/governance/test_governed_limits.py` (modify) | seed + live counts |
| `tests/sell_side/conftest.py` (create) | live fixture + sentinel cleanup |
| `tests/sell_side/test_*.py`, `tests/services/test_catalog_match.py`, `tests/api/test_catalog_router.py`, `tests/api/test_sales_router.py` (create) | tests |

---

### Task 1: Harden the catalog importer

Four defects found reviewing the committed draft, each of which either corrupts versioning or turns one bad cell into a whole-file rollback:

1. **Dates never compare equal.** `_apply_transform` returns an ISO *string* for date columns; the current row from Postgres holds a `datetime.date`. `_same_as_current` therefore sees every dated row as changed and re-versions it on every import — churning history and breaking spec criterion 3's "unchanged row makes no version".
2. **A SKU twice in one file** versions itself within the same import (row 2 is inserted, row 9 closes it and inserts again).
3. **`lead_time_days` "5.5"** silently becomes 5.
4. **A non-ISO currency** ("Pounds") passes the row check and then fails the `char(3)` insert — which aborts and rolls back the *entire* file instead of rejecting one row. Same shape for vocabulary columns (`lifecycle_status`, `availability_status`, `cost_basis`), which the DDL comments enumerate but nothing enforces.
5. **The mapping is read by profile only**, ignoring `bp_catalog_mapping.distributor_id` — distributor B's feed can be read with distributor A's column map.

**Files:**
- Modify: `src/services/catalog_import.py`
- Test: `tests/services/test_catalog_import.py`

**Interfaces:**
- Produces (used by Task 9): `save_mapping(conn, *, mapping_profile: str, distributor_id: str, entries: list[dict]) -> list[dict]` and `get_mapping(conn, mapping_profile: str) -> list[dict]`; `TRANSFORMS: frozenset[str]`; `VOCABULARIES: dict[str, frozenset[str]]`.

- [ ] **Step 1: Write the failing tests** — append to `tests/services/test_catalog_import.py`:

```python
import datetime as _dt


# --- review fixes (2026-09-11) ---------------------------------------------

def test_an_unchanged_dated_row_creates_no_new_version():
    """Postgres hands back a date; the feed carries text. Comparing the two as
    strings re-versioned every dated row on every import."""
    conn = FakeConn(
        mapping=_map(extra=[{"target_column": "end_of_sale_date",
                             "source_header": "EOS", "transform": None,
                             "is_required": False}]),
        current={"A1": {"catalog_item_id": 900, "distributor_sku": "A1",
                        "item_description": "Widget", "currency": "GBP",
                        "end_of_sale_date": _dt.date(2026, 6, 30)}},
    )
    res = _run(conn, [[["SKU", "Description", "Ccy", "EOS"],
                       ["A1", "Widget", "GBP", "2026-06-30"]]])

    assert _inserted_items(conn) == []
    assert res.rows_unchanged == 1


def test_a_sku_twice_in_one_file_rejects_the_second_occurrence():
    conn = FakeConn(mapping=_map())
    res = _run(conn, [[["SKU", "Description", "Ccy"],
                       ["A1", "Widget", "GBP"],
                       ["A1", "Widget v2", "GBP"]]])

    assert res.rows_loaded == 1
    assert res.rows_rejected == 1
    assert any("twice" in r.reason for r in res.rejects)


def test_a_fractional_lead_time_is_rejected_not_truncated():
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "lead_time_days", "source_header": "Lead",
         "transform": None, "is_required": False}]))
    res = _run(conn, [[["SKU", "Description", "Ccy", "Lead"],
                       ["A1", "Widget", "GBP", "5.5"]]])

    assert res.rows_loaded == 0
    assert any("lead_time_days" in r.reason for r in res.rejects)


def test_currency_is_upper_cased_and_a_non_iso_value_rejects_the_row():
    conn = FakeConn(mapping=_map())
    res = _run(conn, [[["SKU", "Description", "Ccy"],
                       ["A1", "Widget", "gbp"],
                       ["A2", "Gadget", "Pounds"]]])

    (params,) = _inserted_items(conn)
    assert dict(zip(catalog_import.ITEM_COLUMNS, params))["currency"] == "GBP"
    assert res.rows_rejected == 1
    assert any("currency" in r.reason for r in res.rejects)


def test_lifecycle_is_normalised_and_an_unknown_value_is_rejected():
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "lifecycle_status", "source_header": "Life",
         "transform": None, "is_required": False}]))
    res = _run(conn, [[["SKU", "Description", "Ccy", "Life"],
                       ["A1", "Widget", "GBP", "End of Sale"],
                       ["A2", "Gadget", "GBP", "EOL"]]])

    (params,) = _inserted_items(conn)
    assert dict(zip(catalog_import.ITEM_COLUMNS, params))["lifecycle_status"] == "end_of_sale"
    assert any("lifecycle_status" in r.reason for r in res.rejects)


def test_the_mapping_is_read_for_this_distributor_only():
    conn = FakeConn(mapping=_map())
    _run(conn, [[["SKU", "Description", "Ccy"], ["A1", "Widget", "GBP"]]])

    (params,) = [p for sql, p in conn.calls if "FROM proc.bp_catalog_mapping" in sql]
    assert params == ("ingram_v1", "SUP-001")


def test_save_mapping_refuses_a_column_a_feed_may_not_set():
    conn = FakeConn(mapping=[])
    with pytest.raises(ValueError, match="distributor_id"):
        catalog_import.save_mapping(conn, mapping_profile="p", distributor_id="SUP-001",
                                    entries=[{"target_column": "distributor_id",
                                              "source_header": "X"}])


def test_save_mapping_refuses_an_unknown_transform():
    conn = FakeConn(mapping=[])
    with pytest.raises(ValueError, match="transform"):
        catalog_import.save_mapping(conn, mapping_profile="p", distributor_id="SUP-001",
                                    entries=[{"target_column": "cost_price",
                                              "source_header": "Cost",
                                              "transform": "guess"}])
```

- [ ] **Step 2: Run to verify they fail**

Run: `./venv/bin/python -m pytest tests/services/test_catalog_import.py -q`
Expected: the 8 new tests FAIL (the 19 existing still pass).

- [ ] **Step 3: Implement.** In `src/services/catalog_import.py`:

Add `import datetime as _dt` to the imports. Below `_DATE_COLUMNS` add:

```python
TRANSFORMS = frozenset({"trim_currency", "pence_to_major", "pack_split"})

# The DDL comments enumerate these and nothing in the schema enforces them, so
# the importer does. A value outside the vocabulary is a mapping problem the
# operator must see, not a new status the product silently learns.
VOCABULARIES: Dict[str, frozenset] = {
    "cost_basis": frozenset({"contract", "spot", "promotion", "unknown"}),
    "availability_status": frozenset(
        {"in_stock", "backorder", "special_order", "discontinued"}),
    "lifecycle_status": frozenset(
        {"active", "end_of_sale", "end_of_life", "superseded"}),
}

_ISO_CURRENCY = re.compile(r"[A-Z]{3}")
```

Replace the tail of `_apply_transform` (from `if target in _DECIMAL_COLUMNS:` to the final `return text`) with:

```python
    if target in _DECIMAL_COLUMNS:
        try:
            return _decimal(text)
        except InvalidOperation:
            raise ValueError(f"{target}: {text!r} is not a number") from None

    if target in _INT_COLUMNS:
        try:
            number = _decimal(text)
        except InvalidOperation:
            raise ValueError(f"{target}: {text!r} is not a whole number") from None
        if number != number.to_integral_value():
            # Truncating 5.5 days to 5 is a guess that reaches a delivery promise.
            raise ValueError(f"{target}: {text!r} is not a whole number")
        return int(number)

    if target in _DATE_COLUMNS:
        # A date, not the string: Postgres returns a date for the current row,
        # and a string never compares equal to one -- every dated row would
        # re-version on every import.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            raise ValueError(f"{target}: {text!r} is not an ISO date")
        try:
            return _dt.date.fromisoformat(text)
        except ValueError:
            raise ValueError(f"{target}: {text!r} is not a real date") from None

    if target == "currency":
        code = text.upper()
        if not _ISO_CURRENCY.fullmatch(code):
            raise ValueError(f"currency: {text!r} is not a three-letter ISO code")
        return code

    if target in VOCABULARIES:
        value = "_".join(text.casefold().replace("-", " ").split())
        if value not in VOCABULARIES[target]:
            raise ValueError(
                f"{target}: {text!r} is not one of {sorted(VOCABULARIES[target])}")
        return value

    return text
```

Also change the unknown-transform check near the top of `_apply_transform` to use the constant: `if transform and transform not in TRANSFORMS: raise ValueError(f"{target}: unknown transform {transform!r}")` — place it immediately after the blank-cell `return None`, and delete the old `if transform: raise ...` line.

Scope the mapping read to the distributor — change `_load_mapping`'s signature and query:

```python
def _load_mapping(cur, mapping_profile: str, distributor_id: str) -> List[_MappingEntry]:
    cur.execute(
        "SELECT target_column, source_header, transform, is_required "
        "FROM proc.bp_catalog_mapping WHERE mapping_profile = %s AND distributor_id = %s",
        (mapping_profile, distributor_id),
    )
```

and the call in `import_catalog`: `mapping = _load_mapping(cur, mapping_profile, distributor_id)`. Update the "no mapping profile" error text to `f"no mapping profile {mapping_profile!r} for distributor {distributor_id!r} in proc.bp_catalog_mapping; ..."`.

Duplicate SKUs: in `_read_sheets`, create `seen_skus: Dict[str, Tuple[int, int]] = {}` before the page loop and pass `seen_skus=seen_skus` to `_read_row`. Add the keyword parameter `seen_skus: Dict[str, Tuple[int, int]]` to `_read_row`, and after the `_REQUIRED_COLUMNS` loop insert:

```python
    sku = values["distributor_sku"]
    first = seen_skus.get(sku)
    if first is not None:
        # Loading both would version the SKU against itself inside one import.
        reject(f"distributor_sku {sku!r} appears twice in this file "
               f"(first at sheet {first[0]} row {first[1]})")
        return
    seen_skus[sku] = (sheet, row_no)
```

Mapping write path, appended at the end of the module:

```python
# --- mapping administration -------------------------------------------------

def get_mapping(conn: Any, mapping_profile: str) -> List[Dict[str, Any]]:
    cur = _dict_cursor(conn)
    cur.execute(
        "SELECT mapping_profile, distributor_id, target_column, source_header, "
        "transform, is_required FROM proc.bp_catalog_mapping "
        "WHERE mapping_profile = %s ORDER BY target_column",
        (mapping_profile,),
    )
    return [dict(r) for r in (cur.fetchall() or [])]


def save_mapping(
    conn: Any, *, mapping_profile: str, distributor_id: str,
    entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Replace a profile's column map. The map is a row a human owns, so this is
    the only way one is written, and it validates what the importer would
    otherwise discover row by row."""
    if not entries:
        raise ValueError("a mapping profile needs at least one entry")
    seen = set()
    for e in entries:
        target = e.get("target_column")
        if target not in MAPPABLE_COLUMNS:
            raise ValueError(f"{target!r} is not a column a feed may set")
        if target in seen:
            raise ValueError(f"{target!r} is mapped twice")
        seen.add(target)
        if not (e.get("source_header") or "").strip():
            raise ValueError(f"{target}: source_header is empty")
        if e.get("transform") and e["transform"] not in TRANSFORMS:
            raise ValueError(f"{target}: unknown transform {e['transform']!r}")
    missing = [c for c in _REQUIRED_COLUMNS if c not in seen]
    if missing:
        raise ValueError(f"mapping must cover the NOT NULL columns: {missing}")

    cur = _dict_cursor(conn)
    cur.execute(
        "SELECT DISTINCT distributor_id FROM proc.bp_catalog_mapping "
        "WHERE mapping_profile = %s", (mapping_profile,))
    owners = {r["distributor_id"] for r in (cur.fetchall() or [])}
    if owners and owners != {distributor_id}:
        raise ValueError(
            f"mapping profile {mapping_profile!r} belongs to {sorted(owners)}")
    cur.execute("DELETE FROM proc.bp_catalog_mapping WHERE mapping_profile = %s",
                (mapping_profile,))
    for e in entries:
        cur.execute(
            "INSERT INTO proc.bp_catalog_mapping (mapping_profile, distributor_id, "
            "target_column, source_header, transform, is_required) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (mapping_profile, distributor_id, e["target_column"],
             e["source_header"].strip(), e.get("transform") or None,
             bool(e.get("is_required")) or e["target_column"] in _REQUIRED_COLUMNS),
        )
    conn.commit()
    return get_mapping(conn, mapping_profile)
```

- [ ] **Step 4: Run to verify all pass**

Run: `./venv/bin/python -m pytest tests/services/test_catalog_import.py -q`
Expected: 27 passed.

- [ ] **Step 5: Mutation proof.** Temporarily make the date branch `return text` again → `test_an_unchanged_dated_row_creates_no_new_version` must go red. Temporarily delete the `seen_skus` check → the duplicate test must go red. Restore both; rerun green.

- [ ] **Step 6: Commit**

```bash
git add src/services/catalog_import.py tests/services/test_catalog_import.py
git commit -m "fix(catalog): a date that never compares equal re-versioned every row

<one paragraph per defect fixed; name the two mutation-proved tests>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QqGRjqTTacQQ3FjY4sfmq2"
```

### Task 2: Migrations, action vocabulary and governance rows

The DDL sits in `sql/` as reference files; this repo's migrations are dated, transactional, idempotent files in `deploy/sql/` with a matching `_rollback.sql`. Irreversible action classes (`transact`, `communicate`) are refused by the gate unless a policy row states `effect: allow` — so approving and issuing a quote need permit rows, or every call returns 403 "raised for review".

**Files:**
- Move: `sql/bp_catalog.sql` → `deploy/sql/2026-09-11_bp_catalog.sql`
- Move: `sql/bp_sell_side.sql` → `deploy/sql/2026-09-11_bp_sell_side.sql`
- Create: `deploy/sql/2026-09-11_bp_catalog_rollback.sql`, `deploy/sql/2026-09-11_bp_sell_side_rollback.sql`
- Create: `deploy/sql/2026-09-11_reseller_governance.sql`, `deploy/sql/2026-09-11_reseller_governance_rollback.sql`
- Modify: `src/services/actions.py`, `tests/conftest.py`, `tests/governance/test_governed_limits.py`
- Modify: `docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md` (DDL paths on lines 7, 102, 113; add `-f` when staging — `/docs/*` is gitignored)

**Interfaces:**
- Produces: action names `catalog.write` (write), `account.write` (write), `sales.write` (write), `sales.calibrate` (compute), `sales_quote.issue` (communicate), `sales_quote.approve` (transact). Governed limits `limit("reseller_catalog", "fuzzy_propose_min")` → 88 and `limit("reseller_catalog", "calibration_min_closed", cast=int)` → 30.

- [ ] **Step 1: Move the DDL and make it transactional**

```bash
git mv sql/bp_catalog.sql deploy/sql/2026-09-11_bp_catalog.sql
git mv sql/bp_sell_side.sql deploy/sql/2026-09-11_bp_sell_side.sql
```

In each moved file insert a line `BEGIN;` immediately after the header comment block (before the first SQL statement) and append `COMMIT;` as the last line. Change the `Spec:` comment lines to keep pointing at the spec. At the top of `2026-09-11_bp_sell_side.sql` add the comment `-- Apply AFTER 2026-09-11_bp_catalog.sql: bp_sales_opportunity and bp_sales_quote_line reference bp_catalog_item.`

- [ ] **Step 2: Write the rollbacks**

`deploy/sql/2026-09-11_bp_sell_side_rollback.sql`:

```sql
-- Rollback of 2026-09-11_bp_sell_side.sql. Children first.
BEGIN;
DROP TABLE IF EXISTS proc.bp_sales_quote_outcome;
DROP TABLE IF EXISTS proc.bp_sales_quote_line;
DROP TABLE IF EXISTS proc.bp_sales_quote;
DROP TABLE IF EXISTS proc.bp_sales_justification;
DROP TABLE IF EXISTS proc.bp_sales_opportunity;
DROP TABLE IF EXISTS proc.bp_account_history_scope;
DROP TABLE IF EXISTS proc.bp_account_contact;
DROP TABLE IF EXISTS proc.bp_account;
COMMIT;
```

`deploy/sql/2026-09-11_bp_catalog_rollback.sql`:

```sql
-- Rollback of 2026-09-11_bp_catalog.sql. Roll back bp_sell_side first: it references bp_catalog_item.
BEGIN;
DROP TABLE IF EXISTS proc.bp_catalog_item_match;
DROP TABLE IF EXISTS proc.bp_catalog_item_relation;
DROP TABLE IF EXISTS proc.bp_catalog_cost_tier;
DROP TABLE IF EXISTS proc.bp_catalog_item;
DROP TABLE IF EXISTS proc.bp_catalog_mapping;
DROP TABLE IF EXISTS proc.bp_catalog_source;
-- The key this migration added. Safe only once nothing references it.
ALTER TABLE proc.bp_supplier DROP CONSTRAINT IF EXISTS pk_bp_supplier;
COMMIT;
```

- [ ] **Step 3: Write the governance migration** — `deploy/sql/2026-09-11_reseller_governance.sql`:

```sql
-- Reseller catalog + sell side: the two numbers it is governed by, and the two
-- irreversible actions it needs a stated permit for.
--
-- LIMITS (read by name through governed_limits.limit, deliberately no applies_to):
--   fuzzy_propose_min      rapidfuzz token_sort_ratio at or above which a catalog
--                          description is PROPOSED as matching a history item.
--                          Proposed, never applied: a person confirms.
--   calibration_min_closed closed (won + lost) opportunities of one type needed
--                          before win_probability is set for that type at all.
--                          Below it the column stays NULL -- an uncalibrated
--                          number is indistinguishable from a measured one.
--
-- PERMITS. The gate refuses transact/communicate unless a policy states
-- effect=allow. Approving an outbound quote commits us to a price; issuing one
-- sends it to a customer. Both need Approver. That nobody approves their own
-- quote is enforced in sell_side.quotes.approve, not stated here: a policy key
-- nothing reads is how this project has shipped rules that governed nothing.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'ResellerCatalogLimitPolicy', 'limit',
       'How close a catalog description must be before it is offered as a match '
       'to purchase history, and how much closed history a win probability needs.',
       jsonb_build_object(
         'policy_identifier', 'reseller_catalog',
         'rules', jsonb_build_object(
             'fuzzy_propose_min',      88,
             'calibration_min_closed', 30)),
       '', 1, 1, now(), 'reseller_catalog', now(), 'reseller_catalog'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'reseller_catalog'
       AND p.policy_status = 1);

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT v.name, 'authority', v.descr,
       jsonb_build_object(
         'policy_identifier', v.slug,
         'applies_to', jsonb_build_array(v.action),
         'required_role', 'Approver',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'reseller_catalog', now(), 'reseller_catalog'
  FROM (VALUES
    ('SalesQuoteApprovalAuthorityPolicy', 'sales_quote_approval_authority',
     'sales_quote.approve',
     'Who may approve an outbound sales quote. The approver is the authenticated '
     'caller and may not be the quote''s author.'),
    ('SalesQuoteIssueAuthorityPolicy', 'sales_quote_issue_authority',
     'sales_quote.issue',
     'Who may issue an approved sales quote to a customer.')
  ) AS v(name, slug, action, descr)
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = v.slug
       AND p.policy_status = 1);

COMMIT;
```

`deploy/sql/2026-09-11_reseller_governance_rollback.sql`:

```sql
BEGIN;
DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' IN (
    'reseller_catalog', 'sales_quote_approval_authority', 'sales_quote_issue_authority')
   AND created_by = 'reseller_catalog';
COMMIT;
```

- [ ] **Step 4: Add the actions** — in `src/services/actions.py` `ACTIONS`:
  - under `# --- computing`: `"sales.calibrate": "compute",`
  - under `# --- writing`: `"catalog.write": "write",`, `"account.write": "write",`, `"sales.write": "write",`
  - under `# --- communicating`: `"sales_quote.issue": "communicate",`
  - under `# --- transacting`: `"sales_quote.approve": "transact",`

- [ ] **Step 5: Seed copy + live counts.** In `tests/conftest.py` `GOVERNED_LIMIT_SEED` add:

```python
    "reseller_catalog": {"fuzzy_propose_min": 88, "calibration_min_closed": 30},
```

In `tests/governance/test_governed_limits.py::test_every_governed_limit_is_present_in_the_live_policy_set` change `len(live) == 7` → `8` (message "eight") and `== 33` → `== 35`.

- [ ] **Step 6: Run the static tests**

Run: `./venv/bin/python -m pytest tests/guardrails/test_policy_action_vocabulary.py tests/governance/test_governed_limits.py -q`
Expected: the two live-row tests in `test_governed_limits.py` FAIL (DB not migrated yet: "expected eight limit rows, found 7" / seed and live disagree). Everything else passes.

- [ ] **Step 7: Apply all three migrations to `bp_testdb`, twice** (idempotency is acceptance criterion 1). No psql dependency:

```bash
set -a; . /home/muthu/PycharmProjects/BP_Backend/.env; set +a
for pass in 1 2; do for f in bp_catalog bp_sell_side reseller_governance; do
./venv/bin/python - "$f" <<'EOF'
import os, sys, psycopg2
conn = psycopg2.connect(host=os.environ["DB_HOST"], port=os.getenv("DB_PORT", 5432),
    dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"])
conn.autocommit = True
assert conn.get_dsn_parameters()["dbname"] == "bp_testdb", "only bp_testdb in this task"
conn.cursor().execute(open(f"deploy/sql/2026-09-11_{sys.argv[1]}.sql").read())
print("applied", sys.argv[1])
EOF
done; done
```

Expected: six "applied" lines, no error. Then verify: 14 tables (`SELECT count(*) FROM information_schema.tables WHERE table_schema='proc' AND (table_name LIKE 'bp_catalog%' OR table_name LIKE 'bp_sales%' OR table_name LIKE 'bp_account%')` → 14), `pk_bp_supplier` present, exactly one active row per new `policy_identifier`.

**Do not apply to `bp_sqldb`** — that is the controller's call after asking the user.

- [ ] **Step 8: Run the tests again, live**

Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/guardrails/test_policy_action_vocabulary.py tests/governance/test_governed_limits.py -q`
Expected: all pass, none skipped (check the summary line for `skipped`).

- [ ] **Step 9: Mutation proof.** Change the seed's `calibration_min_closed` to 31 → `test_the_in_memory_seed_matches_the_live_rows` goes red. Restore.

- [ ] **Step 10: Update the spec's DDL paths**, then commit:

```bash
git add -A deploy/sql/2026-09-11_* src/services/actions.py tests/conftest.py tests/governance/test_governed_limits.py
git add -f docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md
git commit -m "feat(catalog): the catalog and sell-side tables become migrations, with their permits

<body: what moved, the two permits and why they are needed, applied twice to bp_testdb, mutation proof>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QqGRjqTTacQQ3FjY4sfmq2"
```

### Task 3: Sell-side foundations — ladder, money, costing, live fixture

**Files:**
- Create: `src/services/sell_side/__init__.py` (empty), `src/services/sell_side/_db.py`, `src/services/sell_side/ladder.py`, `src/services/sell_side/money.py`, `src/services/sell_side/costing.py`
- Create: `tests/sell_side/__init__.py` (empty), `tests/sell_side/conftest.py`, `tests/sell_side/test_money.py`, `tests/sell_side/test_ladder.py`, `tests/sell_side/test_costing_live.py`

**Interfaces (Produces — every later task uses these exact names):**
- `_db.dict_cursor(conn)`; `_db.NotFound(LookupError)` → HTTP 404; `_db.StateConflict(ValueError)` → HTTP 409. Plain `ValueError` → HTTP 422.
- `ladder.PHASES: tuple[str, ...]`, `ladder.SUBPROCESSES: dict[str, str]` (subprocess → phase), `ladder.check(phase_id, subprocess_id) -> None` (raises `ValueError`), `ladder.QUOTE_RUNG: dict[str, tuple[str, str]]` (quote status → (phase, subprocess)).
- `money.q2(Decimal) -> Decimal`, `money.q4(Decimal) -> Decimal`, `money.LinePrice` (fields `line_total, line_cost, line_margin, line_margin_pct, discount_pct`), `money.price_line(quantity, unit_price, unit_cost, list_price) -> LinePrice`, `money.QuoteTotals` (fields `total_ex_tax, total_cost, total_margin, margin_pct`), `money.total_quote(lines: Sequence[LinePrice]) -> QuoteTotals`, `money.iso_currency(value) -> str` (raises `ValueError`).
- `costing.CostAt` (frozen dataclass: `catalog_item_id, distributor_id, distributor_sku, mpn, item_description, unit_of_measure, currency, list_price, unit_cost, cost_tier_applied, is_current`), `costing.pick_tier(tiers, quantity) -> tuple[Decimal, Decimal] | None`, `costing.cost_at(cur, catalog_item_id: int, quantity: Decimal) -> CostAt` (raises `NotFound`, `ValueError`).
- Test fixture `live_db` → `(conn, distributor_id)`; marker `live`; helper `clean(cur)`; constant `SENTINEL = "LIVETEST"`.

- [ ] **Step 1: Write `tests/sell_side/conftest.py`**

```python
"""Live-database fixture for the sell side.

Every row these tests write carries the LIVETEST sentinel (account ids, SKUs,
feed names, mapping profiles), and `clean` removes exactly those rows, children
first -- before the test, so a crashed earlier run cannot poison this one, and
after it. The services commit on their own, so a rolled-back transaction would
prove nothing; cleanup is by sentinel instead.

Run: set -a && . ./.env && set +a; PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side
"""
import os

import pytest

SENTINEL = "LIVETEST"
_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_QUOTES = "SELECT sales_quote_id FROM proc.bp_sales_quote WHERE account_id LIKE 'LIVETEST-%%'"


def clean(cur):
    cur.execute(f"DELETE FROM proc.bp_sales_quote_outcome WHERE sales_quote_id IN ({_QUOTES})")
    cur.execute(f"DELETE FROM proc.bp_sales_quote_line WHERE sales_quote_id IN ({_QUOTES})")
    cur.execute("UPDATE proc.bp_sales_quote SET supersedes_id = NULL WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_sales_quote WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_sales_opportunity WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_account WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_item_match WHERE distributor_sku LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_item_relation WHERE from_sku LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_item WHERE distributor_sku LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_source WHERE feed_name LIKE 'LIVETEST%%'")
    cur.execute("DELETE FROM proc.bp_catalog_mapping WHERE mapping_profile LIKE 'livetest%%'")


@pytest.fixture
def live_db():
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ["DB_HOST"], port=os.getenv("DB_PORT", 5432),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=10)
    try:
        with conn.cursor() as cur:
            clean(cur)
            cur.execute("SELECT supplier_id FROM proc.bp_supplier ORDER BY supplier_id LIMIT 1")
            distributor_id = cur.fetchone()[0]
        conn.commit()
        yield conn, distributor_id
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            clean(cur)
        conn.commit()
        conn.close()


def seed_item(conn, distributor_id, sku, *, cost="10.0000", list_price="15.0000",
              currency="GBP", mpn=None, description="LIVETEST widget", tiers=()):
    """One current catalog version (+ optional tiers). Returns catalog_item_id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.bp_catalog_source (distributor_id, feed_name, content_sha256, "
            "mapping_profile, price_effective, status) VALUES (%s, %s, %s, 'livetest_v1', "
            "CURRENT_DATE, 'imported') ON CONFLICT (distributor_id, content_sha256) "
            "DO UPDATE SET status = 'imported' RETURNING source_id",
            (distributor_id, f"{SENTINEL} seed", f"{SENTINEL}-seed-{sku}"))
        source_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO proc.bp_catalog_item (source_id, distributor_id, distributor_sku, mpn, "
            "item_description, currency, list_price, cost_price) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING catalog_item_id",
            (source_id, distributor_id, sku, mpn, description, currency, list_price, cost))
        item_id = cur.fetchone()[0]
        for min_q, tier_cost in tiers:
            cur.execute(
                "INSERT INTO proc.bp_catalog_cost_tier VALUES (%s, %s, %s, %s)",
                (item_id, min_q, tier_cost, currency))
    conn.commit()
    return item_id
```

- [ ] **Step 2: Write the failing pure tests.** `tests/sell_side/test_money.py`:

```python
from decimal import Decimal as D

import pytest

from src.services.sell_side import money


def test_a_line_prices_to_two_places_half_up():
    p = money.price_line(D("3"), D("10.005"), D("7.0000"), D("12.0000"))
    assert p.line_total == D("30.02")        # 30.015 -> 30.02
    assert p.line_cost == D("21.00")
    assert p.line_margin == D("9.02")
    assert p.line_margin_pct == D("0.3005")
    assert p.discount_pct == D("0.1663")     # 1 - 10.005/12


def test_an_unknown_cost_gives_no_margin_not_a_zero_one():
    p = money.price_line(D("2"), D("5"), None, None)
    assert p.line_total == D("10.00")
    assert (p.line_cost, p.line_margin, p.line_margin_pct, p.discount_pct) == (None,) * 4


def test_a_free_line_has_no_margin_percentage():
    p = money.price_line(D("1"), D("0"), D("3"), D("5"))
    assert p.line_margin == D("-3.00")
    assert p.line_margin_pct is None


def test_quote_totals_sum_lines():
    a = money.price_line(D("1"), D("10"), D("6"), None)
    b = money.price_line(D("2"), D("5"), D("4"), None)
    t = money.total_quote([a, b])
    assert (t.total_ex_tax, t.total_cost, t.total_margin, t.margin_pct) == \
        (D("20.00"), D("14.00"), D("6.00"), D("0.3000"))


def test_one_uncosted_line_makes_the_quote_margin_unknown():
    """A margin over some lines is not the quote's margin."""
    a = money.price_line(D("1"), D("10"), D("6"), None)
    b = money.price_line(D("1"), D("10"), None, None)
    t = money.total_quote([a, b])
    assert t.total_ex_tax == D("20.00")
    assert (t.total_cost, t.total_margin, t.margin_pct) == (None, None, None)


@pytest.mark.parametrize("raw,ok", [("gbp", "GBP"), (" EUR ", "EUR")])
def test_iso_currency_normalises(raw, ok):
    assert money.iso_currency(raw) == ok


@pytest.mark.parametrize("raw", ["", None, "£", "Pounds", "GB"])
def test_iso_currency_refuses(raw):
    with pytest.raises(ValueError):
        money.iso_currency(raw)
```

`tests/sell_side/test_ladder.py`:

```python
import pytest

from src.services.sell_side import ladder


def test_the_ids_are_the_ui_seed_verbatim():
    # beyond_procwise_ui/src/lib/processTaxonomy/salesLifecycle.js, v1.0.0
    assert ladder.PHASES == ("sales.opportunity", "sales.margin", "sales.approval")
    assert set(ladder.SUBPROCESSES) == {
        "sales.opportunity.qualified", "sales.opportunity.quote-drafted",
        "sales.opportunity.quote-reviewed", "sales.margin.cost-to-serve-modelled",
        "sales.margin.discount-checked", "sales.margin.margin-floor-tested",
        "sales.approval.deal-desk-review", "sales.approval.pricing-approval",
        "sales.approval.conditions-attached"}


def test_a_matching_pair_and_an_empty_pair_pass():
    ladder.check("sales.margin", "sales.margin.discount-checked")
    ladder.check(None, None)
    ladder.check("sales.margin", None)


@pytest.mark.parametrize("phase,sub", [
    ("sales.nope", None),
    ("sales.margin", "sales.approval.pricing-approval"),
    (None, "sales.margin.discount-checked"),
])
def test_an_unknown_or_mismatched_pair_is_refused(phase, sub):
    with pytest.raises(ValueError):
        ladder.check(phase, sub)


def test_every_quote_rung_is_on_the_ladder():
    for phase, sub in ladder.QUOTE_RUNG.values():
        ladder.check(phase, sub)
```

- [ ] **Step 3: Run to verify they fail** — `./venv/bin/python -m pytest tests/sell_side -q` → ImportError.

- [ ] **Step 4: Implement.** `src/services/sell_side/_db.py`:

```python
"""Shared plumbing for the sell-side services."""
from __future__ import annotations

from typing import Any

import psycopg2.extras


class NotFound(LookupError):
    """The row asked for does not exist. HTTP 404."""


class StateConflict(ValueError):
    """The row exists but is in a state that forbids this. HTTP 409."""


def dict_cursor(conn: Any):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
```

`src/services/sell_side/ladder.py`:

```python
"""The sales ladder, verbatim from the UI seed.

Source of truth: beyond_procwise_ui/src/lib/processTaxonomy/salesLifecycle.js
(generated from seeds/sales_lifecycle/v1.0.0.csv). These ids are not invented
here; if the seed changes, this changes with it and the test fails first.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PHASES: Tuple[str, ...] = ("sales.opportunity", "sales.margin", "sales.approval")

SUBPROCESSES: Dict[str, str] = {
    "sales.opportunity.qualified": "sales.opportunity",
    "sales.opportunity.quote-drafted": "sales.opportunity",
    "sales.opportunity.quote-reviewed": "sales.opportunity",
    "sales.margin.cost-to-serve-modelled": "sales.margin",
    "sales.margin.discount-checked": "sales.margin",
    "sales.margin.margin-floor-tested": "sales.margin",
    "sales.approval.deal-desk-review": "sales.approval",
    "sales.approval.pricing-approval": "sales.approval",
    "sales.approval.conditions-attached": "sales.approval",
}

# Where a quote sits on the ladder in each status. Issued stays on the last
# rung reached: the ladder ends at approval and has no "sent" rung to claim.
QUOTE_RUNG: Dict[str, Tuple[str, str]] = {
    "draft": ("sales.opportunity", "sales.opportunity.quote-drafted"),
    "in_review": ("sales.approval", "sales.approval.deal-desk-review"),
    "approved": ("sales.approval", "sales.approval.pricing-approval"),
    "issued": ("sales.approval", "sales.approval.pricing-approval"),
}


def check(phase_id: Optional[str], subprocess_id: Optional[str]) -> None:
    if phase_id is not None and phase_id not in PHASES:
        raise ValueError(f"{phase_id!r} is not a phase on the sales ladder")
    if subprocess_id is None:
        return
    owner = SUBPROCESSES.get(subprocess_id)
    if owner is None:
        raise ValueError(f"{subprocess_id!r} is not a sub-process on the sales ladder")
    if phase_id != owner:
        raise ValueError(f"{subprocess_id!r} belongs to {owner!r}, not {phase_id!r}")
```

`src/services/sell_side/money.py`:

```python
"""Quote arithmetic. Pure, Decimal-only, and NULL-honest: an unknown cost is an
unknown margin, never a zero one."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, Sequence

_TWO = Decimal("0.01")
_FOUR = Decimal("0.0001")
_ISO = re.compile(r"[A-Z]{3}")


def q2(x: Decimal) -> Decimal:
    return x.quantize(_TWO, rounding=ROUND_HALF_UP)


def q4(x: Decimal) -> Decimal:
    return x.quantize(_FOUR, rounding=ROUND_HALF_UP)


def iso_currency(value: Optional[str]) -> str:
    code = (value or "").strip().upper()
    if not _ISO.fullmatch(code):
        raise ValueError(f"{value!r} is not a three-letter ISO currency code")
    return code


@dataclass(frozen=True)
class LinePrice:
    line_total: Decimal
    line_cost: Optional[Decimal]
    line_margin: Optional[Decimal]
    line_margin_pct: Optional[Decimal]
    discount_pct: Optional[Decimal]


@dataclass(frozen=True)
class QuoteTotals:
    total_ex_tax: Decimal
    total_cost: Optional[Decimal]
    total_margin: Optional[Decimal]
    margin_pct: Optional[Decimal]


def price_line(quantity: Decimal, unit_price: Decimal,
               unit_cost: Optional[Decimal], list_price: Optional[Decimal]) -> LinePrice:
    line_total = q2(quantity * unit_price)
    line_cost = q2(quantity * unit_cost) if unit_cost is not None else None
    line_margin = q2(line_total - line_cost) if line_cost is not None else None
    line_margin_pct = (q4(line_margin / line_total)
                       if line_margin is not None and line_total != 0 else None)
    discount_pct = (q4(Decimal(1) - unit_price / list_price)
                    if list_price is not None and list_price > 0 else None)
    return LinePrice(line_total, line_cost, line_margin, line_margin_pct, discount_pct)


def total_quote(lines: Sequence[LinePrice]) -> QuoteTotals:
    total_ex_tax = q2(sum((l.line_total for l in lines), Decimal(0)))
    if any(l.line_cost is None for l in lines):
        return QuoteTotals(total_ex_tax, None, None, None)
    total_cost = q2(sum((l.line_cost for l in lines), Decimal(0)))
    total_margin = q2(total_ex_tax - total_cost)
    margin_pct = q4(total_margin / total_ex_tax) if total_ex_tax != 0 else None
    return QuoteTotals(total_ex_tax, total_cost, total_margin, margin_pct)
```

`src/services/sell_side/costing.py`:

```python
"""What one catalog item costs us at one quantity.

A volume break changes the margin exactly at the quantities a large quote turns
on, so the flat cost_price is only the answer when no break applies. The tier
used is returned, so the margin can be re-derived by hand later (spec §4.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from src.services.sell_side._db import NotFound


@dataclass(frozen=True)
class CostAt:
    catalog_item_id: int
    distributor_id: str
    distributor_sku: str
    mpn: Optional[str]
    item_description: str
    unit_of_measure: Optional[str]
    currency: str
    list_price: Optional[Decimal]
    unit_cost: Optional[Decimal]
    cost_tier_applied: Optional[Decimal]
    is_current: bool


def pick_tier(tiers: Sequence[Tuple[Decimal, Decimal]],
              quantity: Decimal) -> Optional[Tuple[Decimal, Decimal]]:
    """The (min_quantity, cost) break with the highest floor at or below quantity."""
    eligible = [t for t in tiers if t[0] <= quantity]
    return max(eligible, key=lambda t: t[0]) if eligible else None


def cost_at(cur, catalog_item_id: int, quantity: Decimal) -> CostAt:
    """``cur`` must be a RealDictCursor."""
    cur.execute(
        "SELECT catalog_item_id, distributor_id, distributor_sku, mpn, item_description, "
        "unit_of_measure, currency, list_price, cost_price, valid_to IS NULL AS is_current "
        "FROM proc.bp_catalog_item WHERE catalog_item_id = %s",
        (catalog_item_id,))
    item = cur.fetchone()
    if item is None:
        raise NotFound(f"catalog item {catalog_item_id} does not exist")
    cur.execute(
        "SELECT min_quantity, cost_price, currency FROM proc.bp_catalog_cost_tier "
        "WHERE catalog_item_id = %s ORDER BY min_quantity", (catalog_item_id,))
    tiers = cur.fetchall() or []
    for t in tiers:
        if t["currency"] != item["currency"]:
            raise ValueError(
                f"catalog item {catalog_item_id} is priced in {item['currency']} but a "
                f"cost tier is in {t['currency']}; refusing to mix them")
    tier = pick_tier([(t["min_quantity"], t["cost_price"]) for t in tiers], quantity)
    return CostAt(
        catalog_item_id=item["catalog_item_id"], distributor_id=item["distributor_id"],
        distributor_sku=item["distributor_sku"], mpn=item["mpn"],
        item_description=item["item_description"],
        unit_of_measure=item["unit_of_measure"], currency=item["currency"].strip(),
        list_price=item["list_price"],
        unit_cost=tier[1] if tier else item["cost_price"],
        cost_tier_applied=tier[0] if tier else None,
        is_current=bool(item["is_current"]),
    )
```

- [ ] **Step 5: Write the live costing test** — `tests/sell_side/test_costing_live.py`:

```python
from decimal import Decimal as D

import pytest

from src.services.sell_side import costing
from src.services.sell_side._db import NotFound, dict_cursor
from tests.sell_side.conftest import live, seed_item

pytestmark = live


def test_the_highest_break_at_or_below_the_quantity_applies(live_db):
    conn, dist = live_db
    item = seed_item(conn, dist, "LIVETEST-T1", cost="10.0000",
                     tiers=[(D("10"), D("9.0000")), (D("50"), D("8.0000"))])
    cur = dict_cursor(conn)
    assert costing.cost_at(cur, item, D("5")).unit_cost == D("10.0000")
    at_10 = costing.cost_at(cur, item, D("10"))
    assert (at_10.unit_cost, at_10.cost_tier_applied) == (D("9.0000"), D("10"))
    assert costing.cost_at(cur, item, D("75")).unit_cost == D("8.0000")


def test_no_cost_anywhere_is_none_not_zero(live_db):
    conn, dist = live_db
    item = seed_item(conn, dist, "LIVETEST-T2", cost=None)
    assert costing.cost_at(dict_cursor(conn), item, D("1")).unit_cost is None


def test_a_missing_item_is_not_found(live_db):
    conn, _ = live_db
    with pytest.raises(NotFound):
        costing.cost_at(dict_cursor(conn), -1, D("1"))
```

- [ ] **Step 6: Run everything**

Run: `./venv/bin/python -m pytest tests/sell_side -q` → pure tests pass, live tests SKIPPED.
Run: `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side -q` → all pass, 0 skipped.

- [ ] **Step 7: Mutation proof.** In `money.total_quote` delete the `any(... is None)` early return → `test_one_uncosted_line_makes_the_quote_margin_unknown` goes red (TypeError or wrong value). In `pick_tier` change `<=` to `<` → `test_the_highest_break_at_or_below_the_quantity_applies` goes red. Restore.

- [ ] **Step 8: Commit** — `git add src/services/sell_side tests/sell_side` then commit `feat(sell-side): what an item costs at a quantity, and a margin that admits when it is unknown` with the trailer.

### Task 4: Catalog ↔ purchase-history matching (spec §4.6, build step 3)

Purchase-history lines carry one identifier, `item_id` (the extractor maps a document's "sku" into it), and `item_description`. There is **no MPN column** on any `_trgt` line table. So `mpn_exact` compares the catalog `mpn` to `item_id`, and `sku_exact` compares `distributor_sku` to `item_id`. On the current seeded corpus (`item_id` like `ITM003225`) exact methods will honestly find nothing; that is a property of the data, not a bug to paper over.

Every match is written `proposed`. A person confirms or rejects — the supplier entity-resolution pattern (`src/services/extraction_v3/supplier_resolver.py` `confirm_review` / `reject_review`): lock the row `FOR UPDATE`, require `status = 'proposed'`, record who from the token. `confirmed_by`/`confirmed_at` record whoever **decided**, confirm or reject — the column name predates the reject path; say so in the docstring.

**Files:**
- Create: `src/services/catalog_match.py`
- Test: `tests/services/test_catalog_match.py` (pure), `tests/sell_side/test_catalog_match_live.py`

**Interfaces:**
- Consumes: `sell_side._db.dict_cursor`, `NotFound`, `StateConflict`; `limit("reseller_catalog", "fuzzy_propose_min")`.
- Produces: `Proposal` (frozen: `distributor_sku, item_id, match_method, confidence: Decimal|None`); `propose(catalog: Sequence[dict], history: Sequence[dict], *, fuzzy_min: float) -> list[Proposal]`; `propose_matches(conn, distributor_id: str) -> dict` (keys `proposed, exact, fuzzy, already_known`); `list_matches(conn, *, distributor_id=None, status="proposed", limit=100) -> list[dict]`; `confirm_match(conn, match_id: int, reviewer: str|None) -> dict`; `reject_match(conn, match_id: int, reviewer: str|None) -> dict`; `record_human_match(conn, *, distributor_id, distributor_sku, item_id, reviewer) -> dict`.

- [ ] **Step 1: Write the failing pure tests** — `tests/services/test_catalog_match.py`:

```python
from decimal import Decimal

from src.services import catalog_match as cm

HISTORY = [
    {"item_id": "CISCO-C9200-24T", "item_description": "Cisco Catalyst 9200 24 port switch"},
    {"item_id": "IN-1002", "item_description": "HP LaserJet toner black"},
    {"item_id": "ITM000160", "item_description": "Premium Furniture Unit ITM000160"},
]


def _item(sku, desc, mpn=None):
    return {"distributor_sku": sku, "item_description": desc, "mpn": mpn}


def test_an_mpn_equal_to_a_history_item_id_is_an_exact_match_with_no_confidence():
    (p,) = cm.propose([_item("X1", "whatever", mpn=" cisco-c9200-24t ")], HISTORY, fuzzy_min=88)
    assert (p.item_id, p.match_method, p.confidence) == ("CISCO-C9200-24T", "mpn_exact", None)


def test_a_sku_equal_to_a_history_item_id_is_sku_exact():
    (p,) = cm.propose([_item("in-1002", "whatever")], HISTORY, fuzzy_min=88)
    assert (p.item_id, p.match_method) == ("IN-1002", "sku_exact")


def test_mpn_and_sku_naming_the_same_item_propose_it_once():
    got = cm.propose([_item("IN-1002", "x", mpn="IN-1002")], HISTORY, fuzzy_min=88)
    assert [(p.item_id, p.match_method) for p in got] == [("IN-1002", "mpn_exact")]


def test_a_close_description_is_proposed_fuzzy_with_its_score():
    # Word order differs; token_sort_ratio is chosen to absorb exactly that.
    # (Measured: "24-port" vs "24 port" scores 85.3 -- below 88, so NOT used here.)
    (p,) = cm.propose([_item("Z9", "Catalyst 9200 Cisco 24 port switch")], HISTORY, fuzzy_min=88)
    assert p.match_method == "description_fuzzy"
    assert p.item_id == "CISCO-C9200-24T"
    assert Decimal("0.88") <= p.confidence <= Decimal("1")


def test_below_the_threshold_nothing_is_proposed():
    assert cm.propose([_item("Z9", "Office chair mesh")], HISTORY, fuzzy_min=88) == []


def test_an_exact_match_suppresses_fuzzy_guessing_for_that_sku():
    got = cm.propose([_item("IN-1002", "Cisco Catalyst 9200 24 port switch")], HISTORY, fuzzy_min=88)
    assert [p.match_method for p in got] == ["sku_exact"]


def test_no_history_proposes_nothing():
    assert cm.propose([_item("A", "b")], [], fuzzy_min=88) == []
```

- [ ] **Step 2: Run** `./venv/bin/python -m pytest tests/services/test_catalog_match.py -q` → ImportError.

- [ ] **Step 3: Implement `src/services/catalog_match.py`:**

```python
"""Which catalog SKU is which item in purchase history (spec §4.6).

A match is a claim about two datasets, so it is proposed and a person decides.
Exact methods first -- they carry no confidence, because a number there would
imply a judgement nobody made -- then one best fuzzy candidate per SKU that no
exact method resolved.

History carries one identifier, item_id, and no MPN column exists on any _trgt
line table. mpn_exact and sku_exact both compare against item_id for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from rapidfuzz import fuzz, process

from src.services.governed_limits import limit as _governed_limit
from src.services.sell_side._db import NotFound, StateConflict, dict_cursor


def _FUZZY_MIN() -> float:
    return _governed_limit("reseller_catalog", "fuzzy_propose_min")


@dataclass(frozen=True)
class Proposal:
    distributor_sku: str
    item_id: str
    match_method: str
    confidence: Optional[Decimal]


def _key(value: Optional[str]) -> str:
    return " ".join((value or "").split()).casefold()


def propose(catalog: Sequence[Dict[str, Any]], history: Sequence[Dict[str, Any]],
            *, fuzzy_min: float) -> List[Proposal]:
    by_id = {_key(h["item_id"]): h["item_id"] for h in history if h.get("item_id")}
    out: List[Proposal] = []
    unresolved: List[Dict[str, Any]] = []
    for item in catalog:
        sku = item["distributor_sku"]
        found = set()
        mpn = _key(item.get("mpn"))
        if mpn and mpn in by_id:
            out.append(Proposal(sku, by_id[mpn], "mpn_exact", None))
            found.add(by_id[mpn])
        sku_hit = by_id.get(_key(sku))
        if sku_hit and sku_hit not in found:
            out.append(Proposal(sku, sku_hit, "sku_exact", None))
            found.add(sku_hit)
        if not found:
            unresolved.append(item)

    if unresolved and history:
        choices = [_key(h["item_description"]) for h in history]
        for item in unresolved:
            best = process.extractOne(_key(item["item_description"]), choices,
                                      scorer=fuzz.token_sort_ratio, score_cutoff=fuzzy_min)
            if best:
                _, score, idx = best
                out.append(Proposal(item["distributor_sku"], history[idx]["item_id"],
                                    "description_fuzzy",
                                    Decimal(str(round(score / 100.0, 4)))))
    return out


_HISTORY_SQL = """
SELECT item_id, MIN(item_description) AS item_description FROM (
    SELECT item_id, item_description FROM proc.bp_invoice_line_items_trgt WHERE item_id IS NOT NULL
    UNION ALL
    SELECT item_id, item_description FROM proc.bp_po_line_items_trgt WHERE item_id IS NOT NULL
) h GROUP BY item_id
"""


def propose_matches(conn: Any, distributor_id: str) -> Dict[str, int]:
    """Propose matches for this distributor's current catalog. Re-running is safe:
    a pair already proposed, confirmed or rejected is never re-proposed."""
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT distributor_sku, mpn, item_description FROM proc.bp_catalog_item "
        "WHERE distributor_id = %s AND valid_to IS NULL", (distributor_id,))
    catalog = cur.fetchall() or []
    cur.execute(_HISTORY_SQL)
    history = cur.fetchall() or []
    proposals = propose(catalog, history, fuzzy_min=_FUZZY_MIN())

    counts = {"proposed": 0, "exact": 0, "fuzzy": 0, "already_known": 0}
    for p in proposals:
        cur.execute(
            "INSERT INTO proc.bp_catalog_item_match (distributor_id, distributor_sku, item_id, "
            "match_method, confidence) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (distributor_id, distributor_sku, item_id) DO NOTHING",
            (distributor_id, p.distributor_sku, p.item_id, p.match_method, p.confidence))
        if cur.rowcount:
            counts["proposed"] += 1
            counts["fuzzy" if p.match_method == "description_fuzzy" else "exact"] += 1
        else:
            counts["already_known"] += 1
    conn.commit()
    return counts


def list_matches(conn: Any, *, distributor_id: Optional[str] = None,
                 status: str = "proposed", limit: int = 100) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT m.*, c.item_description AS catalog_description "
        "FROM proc.bp_catalog_item_match m "
        "LEFT JOIN proc.bp_catalog_item c ON c.distributor_id = m.distributor_id "
        " AND c.distributor_sku = m.distributor_sku AND c.valid_to IS NULL "
        "WHERE (%s IS NULL OR m.distributor_id = %s) AND (%s = 'all' OR m.status = %s) "
        "ORDER BY m.confidence DESC NULLS FIRST, m.match_id LIMIT %s",
        (distributor_id, distributor_id, status, status, max(1, min(limit, 500))))
    return [dict(r) for r in (cur.fetchall() or [])]


def _decide(conn: Any, match_id: int, reviewer: Optional[str], status: str) -> Dict[str, Any]:
    """confirmed_by / confirmed_at record whoever DECIDED -- a rejection too."""
    cur = dict_cursor(conn)
    cur.execute("SELECT status FROM proc.bp_catalog_item_match WHERE match_id = %s FOR UPDATE",
                (match_id,))
    row = cur.fetchone()
    if row is None:
        conn.rollback()
        raise NotFound(f"match {match_id} does not exist")
    if row["status"] != "proposed":
        conn.rollback()
        raise StateConflict(f"match {match_id} is already {row['status']}")
    cur.execute(
        "UPDATE proc.bp_catalog_item_match SET status = %s, confirmed_by = %s, "
        "confirmed_at = now() WHERE match_id = %s RETURNING *",
        (status, reviewer, match_id))
    out = dict(cur.fetchone())
    conn.commit()
    return out


def confirm_match(conn: Any, match_id: int, reviewer: Optional[str]) -> Dict[str, Any]:
    return _decide(conn, match_id, reviewer, "confirmed")


def reject_match(conn: Any, match_id: int, reviewer: Optional[str]) -> Dict[str, Any]:
    return _decide(conn, match_id, reviewer, "rejected")


def record_human_match(conn: Any, *, distributor_id: str, distributor_sku: str,
                       item_id: str, reviewer: Optional[str]) -> Dict[str, Any]:
    """A person asserts a match no method found. Confirmed on write; replaces any
    earlier machine proposal or rejection for the same pair."""
    cur = dict_cursor(conn)
    cur.execute("SELECT 1 FROM proc.bp_catalog_item WHERE distributor_id = %s "
                "AND distributor_sku = %s AND valid_to IS NULL", (distributor_id, distributor_sku))
    if cur.fetchone() is None:
        conn.rollback()
        raise NotFound(f"{distributor_id}/{distributor_sku} is not a current catalog SKU")
    cur.execute(
        "INSERT INTO proc.bp_catalog_item_match (distributor_id, distributor_sku, item_id, "
        "match_method, confidence, status, confirmed_by, confirmed_at) "
        "VALUES (%s, %s, %s, 'human', NULL, 'confirmed', %s, now()) "
        "ON CONFLICT (distributor_id, distributor_sku, item_id) DO UPDATE SET "
        "match_method = 'human', confidence = NULL, status = 'confirmed', "
        "confirmed_by = EXCLUDED.confirmed_by, confirmed_at = now() RETURNING *",
        (distributor_id, distributor_sku, item_id, reviewer))
    out = dict(cur.fetchone())
    conn.commit()
    return out
```

- [ ] **Step 4: Run pure tests** → 7 passed.

- [ ] **Step 5: Live test** — `tests/sell_side/test_catalog_match_live.py`. It needs a real history `item_id`; take one from the corpus so the exact path is exercised on real rows:

```python
import pytest

from src.services import catalog_match as cm
from src.services.sell_side._db import StateConflict, dict_cursor
from tests.sell_side.conftest import live, seed_item

pytestmark = live


def _a_real_history_item(conn):
    cur = dict_cursor(conn)
    cur.execute("SELECT item_id, item_description FROM proc.bp_invoice_line_items_trgt "
                "WHERE item_id IS NOT NULL ORDER BY item_id LIMIT 1")
    return cur.fetchone()


def test_proposing_twice_proposes_nothing_the_second_time(live_db):
    conn, dist = live_db
    real = _a_real_history_item(conn)
    seed_item(conn, dist, "LIVETEST-M1", mpn=real["item_id"])
    first = cm.propose_matches(conn, dist)
    second = cm.propose_matches(conn, dist)
    assert first["exact"] >= 1
    assert second["proposed"] == 0 and second["already_known"] >= 1


def test_a_decision_is_final_and_attributed(live_db):
    conn, dist = live_db
    real = _a_real_history_item(conn)
    seed_item(conn, dist, "LIVETEST-M2", mpn=real["item_id"])
    cm.propose_matches(conn, dist)
    (m,) = [r for r in cm.list_matches(conn, distributor_id=dist)
            if r["distributor_sku"] == "LIVETEST-M2"]
    done = cm.confirm_match(conn, m["match_id"], "sub-reviewer")
    assert (done["status"], done["confirmed_by"]) == ("confirmed", "sub-reviewer")
    with pytest.raises(StateConflict):
        cm.reject_match(conn, m["match_id"], "sub-other")


def test_a_human_match_is_confirmed_with_no_confidence(live_db):
    conn, dist = live_db
    seed_item(conn, dist, "LIVETEST-M3")
    row = cm.record_human_match(conn, distributor_id=dist, distributor_sku="LIVETEST-M3",
                                item_id="ANY-ID", reviewer="sub-reviewer")
    assert (row["match_method"], row["status"], row["confidence"]) == ("human", "confirmed", None)
```

Run with `PROCWISE_TEST_LIVE_DB=1` → 3 passed, 0 skipped. Also time the real run: `propose_matches` against the full corpus with one seeded SKU must finish in under 10 s (print `time.monotonic()` delta in a scratch run, not in the test).

- [ ] **Step 6: Mutation proof.** Remove `if not found:` gating (always append to `unresolved`) → `test_an_exact_match_suppresses_fuzzy_guessing_for_that_sku` red. Remove the `status != 'proposed'` check → `test_a_decision_is_final_and_attributed` red. Restore.

- [ ] **Step 7: Commit** — `feat(catalog): a SKU is proposed as a history item, and a person decides` + trailer.

### Task 5: Accounts and opportunities (build step 4, first half)

**Files:**
- Create: `src/services/sell_side/accounts.py`, `src/services/sell_side/opportunities.py`
- Test: `tests/sell_side/test_accounts_opportunities_live.py`

**Interfaces:**
- Consumes: `_db.*`, `ladder.check`, `money.iso_currency`, `money.price_line`, `costing.cost_at`.
- Produces:
  - `accounts.create_account(conn, *, account_name: str, account_id: str|None = None, **fields) -> dict` — allowed `fields`: `trading_name, also_supplier_id, registration_number, vat_number, country, default_currency, payment_terms, credit_limit_amount, account_owner_email`. Duplicate id → `StateConflict`.
  - `accounts.get_account(conn, account_id) -> dict` (with `contacts` and `history_scope` lists; raises `NotFound`).
  - `accounts.add_contact(conn, account_id, *, contact_name, contact_role=None, contact_email=None, contact_phone=None, is_primary=False) -> dict`.
  - `accounts.set_history_scope(conn, account_id, *, source_kind, completeness, covers_from=None, covers_to=None, note=None) -> dict`.
  - `opportunities.OPPORTUNITY_TYPES`, `opportunities.JUSTIFICATION_KINDS` (frozensets).
  - `opportunities.create_opportunity(conn, *, account_id, opportunity_type, catalog_item_id=None, currency=None, expected_quantity=None, expected_unit_price=None, phase_id="sales.opportunity", subprocess_id="sales.opportunity.qualified", detector_type=None, reason_codes=None) -> dict`.
  - `opportunities.get_opportunity(conn, sales_opportunity_id) -> dict` (with `justifications`), `opportunities.list_opportunities(conn, *, account_id=None, outcome=None, limit=100) -> list[dict]`.
  - `opportunities.add_justification(conn, sales_opportunity_id, *, kind, claim, evidence_ref=None, evidence_value=None, customer_safe=True) -> dict`.
  - `opportunities.set_stage(conn, sales_opportunity_id, *, phase_id, subprocess_id) -> dict`.

Rules the code enforces (each has a test):
- `win_probability` / `win_probability_basis` are **not parameters** of any function here. They are written only by calibration (Task 8).
- Opportunity currency: taken from the catalog item; a caller currency that differs → `ValueError` ("no FX conversion"). No catalog item → caller currency required.
- Expected values: `expected_revenue = q2(qty × expected_unit_price)` only when both given; `expected_cost` / `expected_margin` / `margin_pct` from `money.price_line` using `costing.cost_at(..., qty)`. Anything unknown is `NULL`.
- Vocabularies: `opportunity_type ∈ {upsell, cross_sell, upgrade, refill, switch_supplier}`; `kind ∈ {price_gap, benchmark, end_of_life, usage_cadence, coverage_gap}`; `source_kind ∈ {our_invoices, customer_shared, third_party}`; `completeness ∈ {complete, partial, unknown}`; `default_currency` via `iso_currency`. Unknown → `ValueError`.
- `claim` must be non-blank; `expected_quantity` if given must be > 0; `expected_unit_price` if given must be ≥ 0.
- `add_contact(is_primary=True)` clears `is_primary` on the account's other contacts in the same transaction.
- Generated `account_id` = `"ACC-" + uuid4().hex[:12].upper()`.
- Every function commits on success and rolls back before raising.

- [ ] **Step 1: Write the failing live tests** — `tests/sell_side/test_accounts_opportunities_live.py`:

```python
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, opportunities as opp
from src.services.sell_side._db import NotFound, StateConflict
from tests.sell_side.conftest import live, seed_item

pytestmark = live


def _acct(conn, suffix="A"):
    return accounts.create_account(conn, account_id=f"LIVETEST-{suffix}",
                                   account_name="Livetest Ltd", default_currency="gbp")


def test_an_account_round_trips_with_contacts_and_scope(live_db):
    conn, _ = live_db
    a = _acct(conn)
    assert a["default_currency"] == "GBP"
    accounts.add_contact(conn, a["account_id"], contact_name="Ann", is_primary=True)
    accounts.add_contact(conn, a["account_id"], contact_name="Bob", is_primary=True)
    accounts.set_history_scope(conn, a["account_id"], source_kind="our_invoices",
                               completeness="complete")
    got = accounts.get_account(conn, a["account_id"])
    assert [c["contact_name"] for c in got["contacts"] if c["is_primary"]] == ["Bob"]
    assert got["history_scope"][0]["source_kind"] == "our_invoices"


def test_a_duplicate_account_id_conflicts(live_db):
    conn, _ = live_db
    _acct(conn)
    with pytest.raises(StateConflict):
        _acct(conn)


def test_an_unknown_scope_kind_is_refused(live_db):
    conn, _ = live_db
    a = _acct(conn)
    with pytest.raises(ValueError):
        accounts.set_history_scope(conn, a["account_id"], source_kind="gossip",
                                   completeness="complete")


def test_an_opportunity_prices_off_the_tier_and_leaves_probability_null(live_db):
    conn, dist = live_db
    a = _acct(conn)
    item = seed_item(conn, dist, "LIVETEST-O1", cost="10.0000", tiers=[(D("10"), D("9.0000"))])
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upsell",
                               catalog_item_id=item, expected_quantity=D("10"),
                               expected_unit_price=D("12.50"))
    assert o["currency"] == "GBP"
    assert (o["expected_revenue"], o["expected_cost"], o["expected_margin"]) == \
        (D("125.00"), D("90.00"), D("35.00"))
    assert o["win_probability"] is None and o["win_probability_basis"] is None
    assert (o["phase_id"], o["subprocess_id"]) == ("sales.opportunity", "sales.opportunity.qualified")


def test_no_price_means_no_revenue_and_no_margin(live_db):
    conn, dist = live_db
    a = _acct(conn)
    item = seed_item(conn, dist, "LIVETEST-O2")
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="refill",
                               catalog_item_id=item, expected_quantity=D("3"))
    assert (o["expected_revenue"], o["expected_margin"]) == (None, None)
    assert o["expected_cost"] == D("30.00")


def test_a_currency_other_than_the_catalogs_is_refused_not_converted(live_db):
    conn, dist = live_db
    a = _acct(conn)
    item = seed_item(conn, dist, "LIVETEST-O3", currency="GBP")
    with pytest.raises(ValueError, match="FX"):
        opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upsell",
                               catalog_item_id=item, currency="EUR")


def test_a_justification_attaches_and_a_blank_claim_is_refused(live_db):
    conn, dist = live_db
    a = _acct(conn)
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upgrade",
                               currency="GBP")
    opp.add_justification(conn, o["sales_opportunity_id"], kind="end_of_life",
                          claim="The 2960X reached end of sale on 2026-06-30.",
                          customer_safe=True)
    with pytest.raises(ValueError):
        opp.add_justification(conn, o["sales_opportunity_id"], kind="benchmark", claim="  ")
    got = opp.get_opportunity(conn, o["sales_opportunity_id"])
    assert [j["kind"] for j in got["justifications"]] == ["end_of_life"]


def test_a_stage_off_the_ladder_is_refused(live_db):
    conn, _ = live_db
    a = _acct(conn)
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upsell",
                               currency="GBP")
    opp.set_stage(conn, o["sales_opportunity_id"], phase_id="sales.margin",
                  subprocess_id="sales.margin.discount-checked")
    with pytest.raises(ValueError):
        opp.set_stage(conn, o["sales_opportunity_id"], phase_id="sales.margin",
                      subprocess_id="sales.approval.pricing-approval")


def test_a_missing_opportunity_is_not_found(live_db):
    conn, _ = live_db
    with pytest.raises(NotFound):
        opp.get_opportunity(conn, -1)


def test_win_probability_cannot_be_passed_in():
    import inspect
    assert "win_probability" not in inspect.signature(opp.create_opportunity).parameters
```

- [ ] **Step 2: Run** with `PROCWISE_TEST_LIVE_DB=1` → ImportError.

- [ ] **Step 3: Implement `accounts.py`:**

```python
"""The customer we sell TO. Deliberately not bp_supplier (spec §4.4)."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

import psycopg2.errors

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor
from src.services.sell_side.money import iso_currency

_ACCOUNT_FIELDS = ("trading_name", "also_supplier_id", "registration_number", "vat_number",
                   "country", "default_currency", "payment_terms", "credit_limit_amount",
                   "account_owner_email")
SOURCE_KINDS = frozenset({"our_invoices", "customer_shared", "third_party"})
COMPLETENESS = frozenset({"complete", "partial", "unknown"})


def _require_account(cur, account_id: str) -> None:
    cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
    if cur.fetchone() is None:
        raise NotFound(f"account {account_id!r} does not exist")


def create_account(conn: Any, *, account_name: str, account_id: Optional[str] = None,
                   **fields: Any) -> Dict[str, Any]:
    unknown = set(fields) - set(_ACCOUNT_FIELDS)
    if unknown:
        raise ValueError(f"not account fields: {sorted(unknown)}")
    if not (account_name or "").strip():
        raise ValueError("account_name is empty")
    if fields.get("default_currency") is not None:
        fields["default_currency"] = iso_currency(fields["default_currency"])
    account_id = account_id or f"ACC-{uuid.uuid4().hex[:12].upper()}"
    cols = ["account_id", "account_name", *fields]
    cur = dict_cursor(conn)
    try:
        cur.execute(
            f"INSERT INTO proc.bp_account ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))}) RETURNING *",
            (account_id, account_name.strip(), *fields.values()))
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise StateConflict(f"account {account_id!r} already exists") from None
    except psycopg2.errors.ForeignKeyViolation:
        conn.rollback()
        raise ValueError(f"also_supplier_id {fields.get('also_supplier_id')!r} is not a supplier") from None
    row = dict(cur.fetchone())
    conn.commit()
    return row


def get_account(conn: Any, account_id: str) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute("SELECT * FROM proc.bp_account WHERE account_id = %s", (account_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"account {account_id!r} does not exist")
    out = dict(row)
    cur.execute("SELECT * FROM proc.bp_account_contact WHERE account_id = %s "
                "ORDER BY is_primary DESC, contact_id", (account_id,))
    out["contacts"] = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT * FROM proc.bp_account_history_scope WHERE account_id = %s "
                "ORDER BY source_kind", (account_id,))
    out["history_scope"] = [dict(r) for r in cur.fetchall()]
    return out


def add_contact(conn: Any, account_id: str, *, contact_name: str,
                contact_role: Optional[str] = None, contact_email: Optional[str] = None,
                contact_phone: Optional[str] = None, is_primary: bool = False) -> Dict[str, Any]:
    if not (contact_name or "").strip():
        raise ValueError("contact_name is empty")
    cur = dict_cursor(conn)
    try:
        _require_account(cur, account_id)
    except NotFound:
        conn.rollback()
        raise
    if is_primary:
        cur.execute("UPDATE proc.bp_account_contact SET is_primary = FALSE "
                    "WHERE account_id = %s", (account_id,))
    cur.execute(
        "INSERT INTO proc.bp_account_contact (account_id, contact_name, contact_role, "
        "contact_email, contact_phone, is_primary) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
        (account_id, contact_name.strip(), contact_role, contact_email, contact_phone,
         bool(is_primary)))
    row = dict(cur.fetchone())
    conn.commit()
    return row


def set_history_scope(conn: Any, account_id: str, *, source_kind: str, completeness: str,
                      covers_from: Any = None, covers_to: Any = None,
                      note: Optional[str] = None) -> Dict[str, Any]:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")
    if completeness not in COMPLETENESS:
        raise ValueError(f"completeness must be one of {sorted(COMPLETENESS)}")
    cur = dict_cursor(conn)
    try:
        _require_account(cur, account_id)
    except NotFound:
        conn.rollback()
        raise
    cur.execute(
        "INSERT INTO proc.bp_account_history_scope (account_id, source_kind, covers_from, "
        "covers_to, completeness, note) VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (account_id, source_kind) DO UPDATE SET covers_from = EXCLUDED.covers_from, "
        "covers_to = EXCLUDED.covers_to, completeness = EXCLUDED.completeness, "
        "note = EXCLUDED.note RETURNING *",
        (account_id, source_kind, covers_from, covers_to, completeness, note))
    row = dict(cur.fetchone())
    conn.commit()
    return row
```

- [ ] **Step 4: Implement `opportunities.py`:**

```python
"""A sell-side opportunity: one catalog SKU one account should be buying.

win_probability is not a parameter of anything here. Calibration writes it,
from closed outcomes, or nothing does (spec §4.5).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from src.services.sell_side import ladder
from src.services.sell_side._db import NotFound, dict_cursor
from src.services.sell_side.costing import cost_at
from src.services.sell_side.money import iso_currency, price_line, q2

OPPORTUNITY_TYPES = frozenset({"upsell", "cross_sell", "upgrade", "refill", "switch_supplier"})
JUSTIFICATION_KINDS = frozenset(
    {"price_gap", "benchmark", "end_of_life", "usage_cadence", "coverage_gap"})


def create_opportunity(
    conn: Any, *, account_id: str, opportunity_type: str,
    catalog_item_id: Optional[int] = None, currency: Optional[str] = None,
    expected_quantity: Optional[Decimal] = None,
    expected_unit_price: Optional[Decimal] = None,
    phase_id: Optional[str] = "sales.opportunity",
    subprocess_id: Optional[str] = "sales.opportunity.qualified",
    detector_type: Optional[str] = None, reason_codes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if opportunity_type not in OPPORTUNITY_TYPES:
        raise ValueError(f"opportunity_type must be one of {sorted(OPPORTUNITY_TYPES)}")
    ladder.check(phase_id, subprocess_id)
    if expected_quantity is not None and expected_quantity <= 0:
        raise ValueError("expected_quantity must be greater than zero")
    if expected_unit_price is not None and expected_unit_price < 0:
        raise ValueError("expected_unit_price cannot be negative")
    wanted = iso_currency(currency) if currency is not None else None

    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
        if cur.fetchone() is None:
            raise NotFound(f"account {account_id!r} does not exist")

        revenue = cost = margin = margin_pct = None
        if catalog_item_id is not None:
            c = cost_at(cur, catalog_item_id, expected_quantity or Decimal(1))
            if wanted is not None and wanted != c.currency:
                raise ValueError(f"catalog item is priced in {c.currency}, not {wanted}; "
                                 "no FX conversion is performed")
            wanted = c.currency
            if expected_quantity is not None:
                if expected_unit_price is not None:
                    p = price_line(expected_quantity, expected_unit_price, c.unit_cost, c.list_price)
                    revenue, cost, margin, margin_pct = (
                        p.line_total, p.line_cost, p.line_margin, p.line_margin_pct)
                elif c.unit_cost is not None:
                    cost = q2(expected_quantity * c.unit_cost)
        elif expected_quantity is not None and expected_unit_price is not None:
            revenue = q2(expected_quantity * expected_unit_price)
        if wanted is None:
            raise ValueError("currency is required when no catalog item is named")

        cur.execute(
            "INSERT INTO proc.bp_sales_opportunity (account_id, catalog_item_id, opportunity_type, "
            "currency, expected_quantity, expected_revenue, expected_cost, expected_margin, "
            "margin_pct, phase_id, subprocess_id, detector_type, reason_codes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (account_id, catalog_item_id, opportunity_type, wanted, expected_quantity,
             revenue, cost, margin, margin_pct, phase_id, subprocess_id, detector_type,
             list(reason_codes) if reason_codes else None))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def get_opportunity(conn: Any, sales_opportunity_id: int) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute("SELECT * FROM proc.bp_sales_opportunity WHERE sales_opportunity_id = %s",
                (sales_opportunity_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"opportunity {sales_opportunity_id} does not exist")
    out = dict(row)
    cur.execute("SELECT * FROM proc.bp_sales_justification WHERE sales_opportunity_id = %s "
                "ORDER BY justification_id", (sales_opportunity_id,))
    out["justifications"] = [dict(r) for r in cur.fetchall()]
    return out


def list_opportunities(conn: Any, *, account_id: Optional[str] = None,
                       outcome: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT * FROM proc.bp_sales_opportunity WHERE (%s IS NULL OR account_id = %s) "
        "AND (%s IS NULL OR outcome = %s) ORDER BY sales_opportunity_id DESC LIMIT %s",
        (account_id, account_id, outcome, outcome, max(1, min(limit, 500))))
    return [dict(r) for r in cur.fetchall()]


def add_justification(conn: Any, sales_opportunity_id: int, *, kind: str, claim: str,
                      evidence_ref: Optional[str] = None,
                      evidence_value: Optional[Decimal] = None,
                      customer_safe: bool = True) -> Dict[str, Any]:
    if kind not in JUSTIFICATION_KINDS:
        raise ValueError(f"kind must be one of {sorted(JUSTIFICATION_KINDS)}")
    if not (claim or "").strip():
        raise ValueError("claim is empty")
    cur = dict_cursor(conn)
    cur.execute("SELECT 1 FROM proc.bp_sales_opportunity WHERE sales_opportunity_id = %s",
                (sales_opportunity_id,))
    if cur.fetchone() is None:
        conn.rollback()
        raise NotFound(f"opportunity {sales_opportunity_id} does not exist")
    cur.execute(
        "INSERT INTO proc.bp_sales_justification (sales_opportunity_id, kind, claim, "
        "evidence_ref, evidence_value, customer_safe) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
        (sales_opportunity_id, kind, claim.strip(), evidence_ref, evidence_value,
         bool(customer_safe)))
    row = dict(cur.fetchone())
    conn.commit()
    return row


def set_stage(conn: Any, sales_opportunity_id: int, *, phase_id: str,
              subprocess_id: Optional[str]) -> Dict[str, Any]:
    ladder.check(phase_id, subprocess_id)
    cur = dict_cursor(conn)
    cur.execute(
        "UPDATE proc.bp_sales_opportunity SET phase_id = %s, subprocess_id = %s, "
        "last_modified_date = now() WHERE sales_opportunity_id = %s RETURNING *",
        (phase_id, subprocess_id, sales_opportunity_id))
    row = cur.fetchone()
    if row is None:
        conn.rollback()
        raise NotFound(f"opportunity {sales_opportunity_id} does not exist")
    conn.commit()
    return dict(row)
```

- [ ] **Step 5: Run** with `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side -q` → all pass, 0 skipped.

- [ ] **Step 6: Mutation proof.** Remove the FX `raise` in `create_opportunity` → `test_a_currency_other_than_the_catalogs_is_refused_not_converted` red. Restore.

- [ ] **Step 7: Commit** — `feat(sell-side): accounts and opportunities, with no probability nobody measured` + trailer.

### Task 6: Outbound quotes — draft, submit, approve, issue, supersede (build step 4, second half)

**Files:**
- Create: `src/services/sell_side/quotes.py`
- Test: `tests/sell_side/test_quotes_live.py`

**Interfaces:**
- Consumes: `_db.*`, `ladder.QUOTE_RUNG`, `money.*`, `costing.cost_at`.
- Produces:
  - `create_draft(conn, *, account_id: str, currency: str, valid_until: date, lines: Sequence[dict], created_by: str|None, contact_id: int|None = None, quote_date: date|None = None, supersedes_id: int|None = None) -> dict` — each line dict: `catalog_item_id: int` (required), `quantity: Decimal`, `unit_price: Decimal`, optional `sales_opportunity_id`, `justification_id`.
  - `get_quote(conn, sales_quote_id) -> dict` — header + `account_name`, `contact_name`, `lines` (ordered by `line_no`), `justifications` (full rows referenced by lines, including `customer_safe`). **Internal**: contains cost and margin.
  - `submit(conn, sales_quote_id, *, actor) -> dict`, `approve(conn, sales_quote_id, *, approver) -> dict`, `issue(conn, sales_quote_id, *, actor) -> dict`.
  - `QUOTE_STATUSES` frozenset.

Rules (each has a test):
- A draft snapshots, per line, from the **current** catalog version: `distributor_sku, mpn, item_description, unit_of_measure, list_price_at_quote, unit_cost, cost_tier_applied`. A closed version → `ValueError`. Nothing on a line is ever re-read from the catalog afterwards (criterion 5).
- Line currency = quote currency = catalog currency, or `ValueError` (no FX).
- `quantity > 0`, `unit_price >= 0`, at least one line, `valid_until >= quote_date` (default today).
- `quote_ref = f"SQ-{quote_date:%Y%m%d}-{id:06d}"`, the id taken from the serial sequence **before** insert.
- No function edits lines after creation. To change a quote, draft a new one with `supersedes_id`; the old one (same account, status `draft|in_review|approved|issued`) becomes `superseded`.
- Transitions, each under `SELECT … FOR UPDATE`, anything else → `StateConflict`: `draft → in_review` (submit), `in_review → approved` (approve), `approved → issued` (issue). Each sets `phase_id/subprocess_id` from `QUOTE_RUNG`.
- `approve`: `approver` must be non-blank (`StateConflict`), and **must differ from `created_by`** (`StateConflict("nobody approves their own quote")`). Sets `approved_by`, `approved_at`.
- `issue`: refused when `valid_until < CURRENT_DATE` (`StateConflict`, "has expired"). Sets `issued_at`.

- [ ] **Step 1: Write the failing live tests** — `tests/sell_side/test_quotes_live.py`:

```python
import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, quotes
from src.services.sell_side._db import StateConflict
from tests.sell_side.conftest import live, seed_item

pytestmark = live
TODAY = dt.date.today()


def _setup(conn, dist, **item_kw):
    accounts.create_account(conn, account_id="LIVETEST-Q", account_name="Livetest Ltd")
    return seed_item(conn, dist, item_kw.pop("sku", "LIVETEST-Q1"), **item_kw)


def _draft(conn, item, **kw):
    kw.setdefault("created_by", "sub-author")
    return quotes.create_draft(
        conn, account_id="LIVETEST-Q", currency="GBP",
        valid_until=TODAY + dt.timedelta(days=30),
        lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00")}], **kw)


def test_a_draft_snapshots_cost_and_totals(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000", list_price="15.0000")
    q = _draft(conn, item)
    (line,) = q["lines"]
    assert q["quote_ref"].startswith(f"SQ-{TODAY:%Y%m%d}-")
    assert (q["status"], q["phase_id"], q["subprocess_id"]) == \
        ("draft", "sales.opportunity", "sales.opportunity.quote-drafted")
    assert (line["unit_cost"], line["list_price_at_quote"], line["line_total"],
            line["line_margin"]) == (D("10.0000"), D("15.0000"), D("56.00"), D("16.00"))
    assert (q["total_ex_tax"], q["total_cost"], q["total_margin"]) == \
        (D("56.00"), D("40.00"), D("16.00"))


def test_a_repricing_after_drafting_does_not_move_the_quote(live_db):
    """Acceptance criterion 5."""
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    q = _draft(conn, item)
    with conn.cursor() as cur:  # the catalog reprices: close this version, open a dearer one
        cur.execute("UPDATE proc.bp_catalog_item SET valid_to = now() WHERE catalog_item_id = %s", (item,))
        cur.execute(
            "INSERT INTO proc.bp_catalog_item (source_id, distributor_id, distributor_sku, "
            "item_description, currency, cost_price) SELECT source_id, distributor_id, "
            "distributor_sku, item_description, currency, 12.0000 FROM proc.bp_catalog_item "
            "WHERE catalog_item_id = %s", (item,))
    conn.commit()
    (line,) = quotes.get_quote(conn, q["sales_quote_id"])["lines"]
    assert (line["unit_cost"], line["line_margin"]) == (D("10.0000"), D("16.00"))


def test_a_closed_catalog_version_cannot_be_quoted(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET valid_to = now() WHERE catalog_item_id = %s", (item,))
    conn.commit()
    with pytest.raises(ValueError, match="closed"):
        _draft(conn, item)


def test_a_foreign_currency_line_is_refused(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, currency="EUR")
    with pytest.raises(ValueError, match="FX"):
        _draft(conn, item)


def test_an_already_expired_validity_is_refused(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with pytest.raises(ValueError):
        quotes.create_draft(conn, account_id="LIVETEST-Q", currency="GBP",
                            valid_until=TODAY - dt.timedelta(days=1), created_by="x",
                            lines=[{"catalog_item_id": item, "quantity": D("1"),
                                    "unit_price": D("1")}])


def test_the_happy_path_and_its_ladder(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    qid = q["sales_quote_id"]
    assert quotes.submit(conn, qid, actor="sub-author")["status"] == "in_review"
    a = quotes.approve(conn, qid, approver="sub-approver")
    assert (a["status"], a["approved_by"], a["subprocess_id"]) == \
        ("approved", "sub-approver", "sales.approval.pricing-approval")
    i = quotes.issue(conn, qid, actor="sub-approver")
    assert i["status"] == "issued" and i["issued_at"] is not None


def test_nobody_approves_their_own_quote(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    with pytest.raises(StateConflict, match="own"):
        quotes.approve(conn, q["sales_quote_id"], approver="sub-author")


def test_an_anonymous_approval_is_refused(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    with pytest.raises(StateConflict):
        quotes.approve(conn, q["sales_quote_id"], approver=None)


def test_a_draft_cannot_be_issued(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    with pytest.raises(StateConflict):
        quotes.issue(conn, q["sales_quote_id"], actor="sub-approver")


def test_superseding_retires_the_old_quote(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    old = _draft(conn, item)
    new = _draft(conn, item, supersedes_id=old["sales_quote_id"])
    assert new["supersedes_id"] == old["sales_quote_id"]
    assert quotes.get_quote(conn, old["sales_quote_id"])["status"] == "superseded"
    with pytest.raises(StateConflict):
        _draft(conn, item, supersedes_id=old["sales_quote_id"])
```

- [ ] **Step 2: Run** with `PROCWISE_TEST_LIVE_DB=1` → ImportError.

- [ ] **Step 3: Implement `src/services/sell_side/quotes.py`:**

```python
"""The outbound quote -- the artifact this platform never had (spec §4.2).

Cost and list price are SNAPSHOTS copied onto the line at draft time and never
re-read. Lines are never edited: a changed quote is a new quote that supersedes
the old one, so what the customer was sent stays what they were sent.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor
from src.services.sell_side.costing import cost_at
from src.services.sell_side.ladder import QUOTE_RUNG
from src.services.sell_side.money import iso_currency, price_line, total_quote

QUOTE_STATUSES = frozenset(
    {"draft", "in_review", "approved", "issued", "expired", "superseded"})
_SUPERSEDABLE = ("draft", "in_review", "approved", "issued")

_LINE_COLS = ("sales_quote_id", "line_no", "sales_opportunity_id", "catalog_item_id",
              "distributor_sku", "mpn", "item_description", "quantity", "unit_of_measure",
              "currency", "list_price_at_quote", "unit_price", "discount_pct", "line_total",
              "unit_cost", "cost_tier_applied", "line_margin", "line_margin_pct",
              "justification_id")


def create_draft(conn: Any, *, account_id: str, currency: str, valid_until: dt.date,
                 lines: Sequence[Dict[str, Any]], created_by: Optional[str],
                 contact_id: Optional[int] = None, quote_date: Optional[dt.date] = None,
                 supersedes_id: Optional[int] = None) -> Dict[str, Any]:
    currency = iso_currency(currency)
    quote_date = quote_date or dt.date.today()
    if valid_until < quote_date:
        raise ValueError("valid_until is before the quote date")
    if not lines:
        raise ValueError("a quote needs at least one line")

    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
        if cur.fetchone() is None:
            raise NotFound(f"account {account_id!r} does not exist")

        snapshots, priced = [], []
        for n, line in enumerate(lines, start=1):
            qty, price = Decimal(line["quantity"]), Decimal(line["unit_price"])
            if qty <= 0:
                raise ValueError(f"line {n}: quantity must be greater than zero")
            if price < 0:
                raise ValueError(f"line {n}: unit_price cannot be negative")
            c = cost_at(cur, int(line["catalog_item_id"]), qty)
            if not c.is_current:
                raise ValueError(f"line {n}: catalog item {c.catalog_item_id} is a closed "
                                 "version; quote the current one")
            if c.currency != currency:
                raise ValueError(f"line {n}: catalog item is priced in {c.currency}, the "
                                 f"quote is in {currency}; no FX conversion is performed")
            p = price_line(qty, price, c.unit_cost, c.list_price)
            priced.append(p)
            snapshots.append((line, n, qty, price, c, p))
        totals = total_quote(priced)

        if supersedes_id is not None:
            cur.execute("SELECT account_id, status FROM proc.bp_sales_quote "
                        "WHERE sales_quote_id = %s FOR UPDATE", (supersedes_id,))
            old = cur.fetchone()
            if old is None:
                raise NotFound(f"quote {supersedes_id} does not exist")
            if old["account_id"] != account_id:
                raise ValueError("a quote can only supersede one for the same account")
            if old["status"] not in _SUPERSEDABLE:
                raise StateConflict(f"quote {supersedes_id} is {old['status']} and cannot be superseded")
            cur.execute("UPDATE proc.bp_sales_quote SET status = 'superseded', "
                        "last_modified_date = now() WHERE sales_quote_id = %s", (supersedes_id,))

        cur.execute("SELECT nextval(pg_get_serial_sequence('proc.bp_sales_quote', "
                    "'sales_quote_id')) AS id")
        quote_id = cur.fetchone()["id"]
        phase, sub = QUOTE_RUNG["draft"]
        cur.execute(
            "INSERT INTO proc.bp_sales_quote (sales_quote_id, quote_ref, account_id, contact_id, "
            "currency, quote_date, valid_until, total_ex_tax, total_cost, total_margin, "
            "margin_pct, phase_id, subprocess_id, status, supersedes_id, created_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', %s, %s)",
            (quote_id, f"SQ-{quote_date:%Y%m%d}-{quote_id:06d}", account_id, contact_id,
             currency, quote_date, valid_until, totals.total_ex_tax, totals.total_cost,
             totals.total_margin, totals.margin_pct, phase, sub, supersedes_id, created_by))
        for line, n, qty, price, c, p in snapshots:
            cur.execute(
                f"INSERT INTO proc.bp_sales_quote_line ({', '.join(_LINE_COLS)}) "
                f"VALUES ({', '.join(['%s'] * len(_LINE_COLS))})",
                (quote_id, n, line.get("sales_opportunity_id"), c.catalog_item_id,
                 c.distributor_sku, c.mpn, c.item_description, qty, c.unit_of_measure,
                 currency, c.list_price, price, p.discount_pct, p.line_total,
                 c.unit_cost, c.cost_tier_applied, p.line_margin, p.line_margin_pct,
                 line.get("justification_id")))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get_quote(conn, quote_id)


def get_quote(conn: Any, sales_quote_id: int) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT q.*, a.account_name, c.contact_name FROM proc.bp_sales_quote q "
        "JOIN proc.bp_account a ON a.account_id = q.account_id "
        "LEFT JOIN proc.bp_account_contact c ON c.contact_id = q.contact_id "
        "WHERE q.sales_quote_id = %s", (sales_quote_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"quote {sales_quote_id} does not exist")
    out = dict(row)
    cur.execute("SELECT * FROM proc.bp_sales_quote_line WHERE sales_quote_id = %s "
                "ORDER BY line_no", (sales_quote_id,))
    out["lines"] = [dict(r) for r in cur.fetchall()]
    ids = [l["justification_id"] for l in out["lines"] if l["justification_id"]]
    if ids:
        cur.execute("SELECT * FROM proc.bp_sales_justification WHERE justification_id = ANY(%s) "
                    "ORDER BY justification_id", (ids,))
        out["justifications"] = [dict(r) for r in cur.fetchall()]
    else:
        out["justifications"] = []
    return out


def _transition(conn: Any, sales_quote_id: int, *, frm: str, to: str,
                check=None, sets: str = "", params: tuple = ()) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT * FROM proc.bp_sales_quote WHERE sales_quote_id = %s FOR UPDATE",
                    (sales_quote_id,))
        row = cur.fetchone()
        if row is None:
            raise NotFound(f"quote {sales_quote_id} does not exist")
        if row["status"] != frm:
            raise StateConflict(f"quote {sales_quote_id} is {row['status']}, not {frm}")
        if check:
            check(row)
        phase, sub = QUOTE_RUNG[to]
        cur.execute(
            f"UPDATE proc.bp_sales_quote SET status = %s, phase_id = %s, subprocess_id = %s, "
            f"last_modified_date = now(){sets} WHERE sales_quote_id = %s",
            (to, phase, sub, *params, sales_quote_id))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get_quote(conn, sales_quote_id)


def submit(conn: Any, sales_quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    return _transition(conn, sales_quote_id, frm="draft", to="in_review")


def approve(conn: Any, sales_quote_id: int, *, approver: Optional[str]) -> Dict[str, Any]:
    def _check(row):
        if not (approver or "").strip():
            raise StateConflict("an approval needs an authenticated approver")
        if row["created_by"] and row["created_by"] == approver:
            raise StateConflict("nobody approves their own quote")
    return _transition(conn, sales_quote_id, frm="in_review", to="approved", check=_check,
                       sets=", approved_by = %s, approved_at = now()", params=(approver,))


def issue(conn: Any, sales_quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    def _check(row):
        if row["valid_until"] < dt.date.today():
            raise StateConflict(f"quote {sales_quote_id} has expired ({row['valid_until']})")
    return _transition(conn, sales_quote_id, frm="approved", to="issued", check=_check,
                       sets=", issued_at = now()")
```

- [ ] **Step 4: Run** `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side -q` → all pass, 0 skipped.

- [ ] **Step 5: Mutation proof.** In `get_quote`, replace `SELECT * FROM proc.bp_sales_quote_line` with a join that reads `unit_cost` from `bp_catalog_item` (the live-join the spec forbids — e.g. `SELECT l.*, ci.cost_price AS unit_cost ... JOIN proc.bp_catalog_item ci ON ci.distributor_sku = l.distributor_sku AND ci.valid_to IS NULL`) → criterion-5 test red. Remove the `created_by == approver` check → self-approval test red. Restore both.

- [ ] **Step 6: Commit** — `feat(sell-side): an outbound quote that stays what the customer was sent` + trailer.

### Task 7: The customer view — serialiser allowlist and leak guard (build step 5, spec §4.3)

The DDL's `INTERNAL` comments are convention. This is the control. Two layers: an **allowlist** projection (only named fields are copied), and a **leak guard** that walks the finished payload and raises if any internal field name appears anywhere — so a future edit that widens the allowlist fails loudly instead of shipping margin to a customer. Acceptance criterion 6 is proved by rendering with a mutated allowlist and watching it raise.

**Files:**
- Create: `src/services/sell_side/quote_render.py`
- Test: `tests/sell_side/test_quote_render.py` (pure), plus one case in `tests/sell_side/test_quotes_live.py`

**Interfaces:**
- Consumes: the dict shape returned by `quotes.get_quote`.
- Produces: `INTERNAL_FIELDS`, `CUSTOMER_HEADER_FIELDS`, `CUSTOMER_LINE_FIELDS`, `CUSTOMER_JUSTIFICATION_FIELDS` (frozensets); `InternalFieldLeak(RuntimeError)`; `NotCustomerReady(ValueError)`; `customer_view(quote: dict) -> dict`; `render_html(view: dict) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/sell_side/test_quote_render.py`:

```python
import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import quote_render as qr


def _quote(status="approved", safe=True):
    return {
        "sales_quote_id": 1, "quote_ref": "SQ-20260911-000001", "status": status,
        "account_name": "Acme <Ltd>", "contact_name": "Ann", "currency": "GBP",
        "quote_date": dt.date(2026, 9, 11), "valid_until": dt.date(2026, 10, 11),
        "total_ex_tax": D("56.00"), "total_cost": D("40.00"), "total_margin": D("16.00"),
        "margin_pct": D("0.2857"), "created_by": "sub-author",
        "lines": [{"line_no": 1, "distributor_sku": "IN-1", "mpn": None,
                   "item_description": "Widget & co", "quantity": D("4"),
                   "unit_of_measure": "EA", "currency": "GBP",
                   "list_price_at_quote": D("15.0000"), "unit_price": D("14.0000"),
                   "discount_pct": D("0.0667"), "line_total": D("56.00"),
                   "unit_cost": D("10.0000"), "cost_tier_applied": None,
                   "line_margin": D("16.00"), "line_margin_pct": D("0.2857")}],
        "justifications": [
            {"kind": "end_of_life", "claim": "EOS 2026-06-30", "customer_safe": True},
            {"kind": "price_gap", "claim": "We make 29% here", "customer_safe": safe},
        ],
    }


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_the_customer_view_carries_no_internal_field_anywhere():
    view = qr.customer_view(_quote())
    assert not set(_keys(view)) & qr.INTERNAL_FIELDS
    assert view["lines"][0]["unit_price"] == D("14.0000")


def test_an_internal_only_justification_never_leaves():
    view = qr.customer_view(_quote(safe=False))
    assert [j["claim"] for j in view["justifications"]] == ["EOS 2026-06-30"]


def test_widening_the_allowlist_to_a_cost_field_fails_the_render(monkeypatch):
    """Acceptance criterion 6: attempt to render cost and watch it fail."""
    monkeypatch.setattr(qr, "CUSTOMER_LINE_FIELDS", qr.CUSTOMER_LINE_FIELDS | {"unit_cost"})
    with pytest.raises(qr.InternalFieldLeak, match="unit_cost"):
        qr.customer_view(_quote())


def test_a_hand_built_view_with_margin_cannot_be_rendered_to_html():
    with pytest.raises(qr.InternalFieldLeak, match="total_margin"):
        qr.render_html({"quote_ref": "x", "total_margin": D("1"), "lines": []})


def test_an_unapproved_quote_has_no_customer_view():
    with pytest.raises(qr.NotCustomerReady):
        qr.customer_view(_quote(status="draft"))


def test_html_escapes_everything_and_shows_no_cost():
    html = qr.render_html(qr.customer_view(_quote()))
    assert "Acme &lt;Ltd&gt;" in html and "Widget &amp; co" in html
    assert "10.0000" not in html and "16.00" not in html


def test_the_allowlists_and_the_internal_set_are_disjoint():
    for allowed in (qr.CUSTOMER_HEADER_FIELDS, qr.CUSTOMER_LINE_FIELDS,
                    qr.CUSTOMER_JUSTIFICATION_FIELDS):
        assert not allowed & qr.INTERNAL_FIELDS
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement `src/services/sell_side/quote_render.py`:**

```python
"""What a customer may see of a quote (spec §4.3). THE control on cost and margin.

Two layers, deliberately redundant:
  1. an allowlist projection -- only named fields are copied, so a column added
     to a table later is invisible here until someone names it;
  2. a leak guard that walks the finished payload and raises on any internal
     field name, so naming one fails the render instead of shipping it.
render_html re-runs the guard, so a view built by hand cannot bypass it.
"""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable

INTERNAL_FIELDS = frozenset({
    "unit_cost", "cost_tier_applied", "line_margin", "line_margin_pct",
    "total_cost", "total_margin", "margin_pct", "expected_cost", "expected_margin",
    "cost_price", "cost_basis", "customer_safe",
})
CUSTOMER_HEADER_FIELDS = frozenset({
    "quote_ref", "account_name", "contact_name", "currency", "quote_date",
    "valid_until", "total_ex_tax",
})
CUSTOMER_LINE_FIELDS = frozenset({
    "line_no", "distributor_sku", "mpn", "item_description", "quantity",
    "unit_of_measure", "currency", "list_price_at_quote", "unit_price",
    "discount_pct", "line_total",
})
CUSTOMER_JUSTIFICATION_FIELDS = frozenset({"kind", "claim"})

_READY = ("approved", "issued")


class InternalFieldLeak(RuntimeError):
    """An internal field reached a customer-facing payload. Never caught to recover."""


class NotCustomerReady(ValueError):
    """Only an approved or issued quote has a customer view."""


def _project(row: Dict[str, Any], allowed: Iterable[str]) -> Dict[str, Any]:
    return {k: row.get(k) for k in sorted(allowed)}


def _assert_no_internal(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in INTERNAL_FIELDS:
                raise InternalFieldLeak(f"{path}.{k} is internal and may not reach a customer")
            _assert_no_internal(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_internal(v, f"{path}[{i}]")


def customer_view(quote: Dict[str, Any]) -> Dict[str, Any]:
    if quote.get("status") not in _READY:
        raise NotCustomerReady(f"quote is {quote.get('status')}; only approved or issued "
                               "quotes have a customer view")
    view = _project(quote, CUSTOMER_HEADER_FIELDS)
    view["lines"] = [_project(l, CUSTOMER_LINE_FIELDS) for l in quote.get("lines") or []]
    view["justifications"] = [
        _project(j, CUSTOMER_JUSTIFICATION_FIELDS)
        for j in quote.get("justifications") or [] if j.get("customer_safe") is True]
    _assert_no_internal(view)
    return view


def _e(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


def render_html(view: Dict[str, Any]) -> str:
    _assert_no_internal(view)
    rows = "".join(
        f"<tr><td>{_e(l.get('line_no'))}</td><td>{_e(l.get('distributor_sku'))}</td>"
        f"<td>{_e(l.get('item_description'))}</td><td>{_e(l.get('quantity'))}</td>"
        f"<td>{_e(l.get('unit_of_measure'))}</td><td>{_e(l.get('unit_price'))}</td>"
        f"<td>{_e(l.get('line_total'))}</td></tr>"
        for l in view.get("lines") or [])
    notes = "".join(f"<li>{_e(j.get('claim'))}</li>" for j in view.get("justifications") or [])
    return (
        f"<article class=\"sales-quote\"><h1>Quote {_e(view.get('quote_ref'))}</h1>"
        f"<p>For {_e(view.get('account_name'))}"
        f"{(' — ' + _e(view.get('contact_name'))) if view.get('contact_name') else ''}</p>"
        f"<p>Dated {_e(view.get('quote_date'))}, valid until {_e(view.get('valid_until'))}. "
        f"Prices in {_e(view.get('currency'))}, excluding tax.</p>"
        "<table><thead><tr><th>#</th><th>SKU</th><th>Description</th><th>Qty</th>"
        "<th>Unit</th><th>Unit price</th><th>Line total</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<p><strong>Total ex tax: {_e(view.get('total_ex_tax'))} {_e(view.get('currency'))}</strong></p>"
        f"{('<ul>' + notes + '</ul>') if notes else ''}</article>"
    )
```

- [ ] **Step 4: Add one live case** to `tests/sell_side/test_quotes_live.py` — a real quote from the database through the real renderer:

```python
def test_a_real_approved_quote_renders_with_no_internal_field(live_db):
    from src.services.sell_side import quote_render as qr

    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    quotes.approve(conn, q["sales_quote_id"], approver="sub-approver")
    view = qr.customer_view(quotes.get_quote(conn, q["sales_quote_id"]))
    flat = repr(view)
    assert not any(f"'{f}'" in flat for f in qr.INTERNAL_FIELDS)
    assert "10.0000" not in qr.render_html(view)
```

- [ ] **Step 5: Run** pure + live → all pass, 0 skipped.

- [ ] **Step 6: Mutation proof.** Delete the `_assert_no_internal(view)` call in `customer_view` → `test_widening_the_allowlist_to_a_cost_field_fails_the_render` red. Restore.

- [ ] **Step 7: Commit** — `feat(sell-side): a quote a customer can see, and cost that cannot reach it` + trailer (name the mutation-proved test).

---

### Task 8: Outcomes, calibration, and the daily job (build step 6, spec §4.5)

**Files:**
- Create: `src/services/sell_side/outcomes.py`, `src/services/sell_side/calibration.py`
- Modify: `src/services/backend_scheduler.py` (register a daily job next to `self._register_capture_retention_job()` at ~line 422)
- Test: `tests/sell_side/test_calibration.py` (pure), `tests/sell_side/test_outcomes_live.py`

**Interfaces:**
- Consumes: `_db.*`, `quotes.get_quote`, `money.q4`, `limit("reseller_catalog", "calibration_min_closed", cast=int)`.
- Produces: `outcomes.QUOTE_OUTCOMES`, `outcomes.LOST_REASONS`, `outcomes.record_outcome(conn, *, sales_quote_id, outcome, outcome_date, recorded_by, lost_reason=None, competitor_name=None) -> dict`; `calibration.Calibration` (frozen: `opportunity_type, won, lost, rate: Decimal|None, applied: bool`), `calibration.rates(counts: Mapping[str, tuple[int, int]], min_closed: int) -> list[Calibration]`, `calibration.calibrate(conn) -> list[Calibration]`.

Rules:
- An outcome is recorded once per quote (PK) — a second → `StateConflict`. Only an `issued` quote takes an outcome. `lost_reason` only with `lost`, vocab `{price, lead_time, incumbent, no_budget, spec, other}`.
- `won_value = total_ex_tax`, `won_margin = total_margin` (may be NULL) for `won`; both NULL otherwise.
- `expired` → quote status `expired`; opportunities on its lines stay `open` (an expired quote is not a lost sale). `won|lost|withdrawn` → each still-`open` opportunity on the quote's lines takes that outcome.
- Calibration: per `opportunity_type`, `closed = won + lost` (withdrawn excluded: we withdrew, the customer did not decide). Only when `closed >= calibration_min_closed`: set `win_probability = q4(won / closed)`, `basis = 'calibrated'` on that type's `open` opportunities whose basis is NULL or `'calibrated'` (never overwrite `'manual'`). Below the threshold: write nothing.

- [ ] **Step 1: Pure tests** — `tests/sell_side/test_calibration.py`:

```python
from decimal import Decimal as D

from src.services.sell_side import calibration as cal


def test_a_type_with_enough_history_gets_its_win_rate():
    (c,) = cal.rates({"upsell": (9, 21)}, min_closed=30)
    assert (c.rate, c.applied) == (D("0.3000"), True)


def test_one_short_of_the_threshold_gets_nothing():
    (c,) = cal.rates({"upsell": (9, 20)}, min_closed=30)
    assert (c.rate, c.applied) == (None, False)


def test_no_history_at_all_gets_nothing():
    assert cal.rates({}, min_closed=30) == []
```

- [ ] **Step 2: Live tests** — `tests/sell_side/test_outcomes_live.py`:

```python
import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, calibration, opportunities as opp, outcomes, quotes
from src.services.sell_side._db import StateConflict
from tests.sell_side.conftest import live, seed_item

pytestmark = live
TODAY = dt.date.today()


def _issued_quote(conn, dist, n=1):
    accounts.create_account(conn, account_id=f"LIVETEST-W{n}", account_name="Livetest")
    item = seed_item(conn, dist, f"LIVETEST-W{n}")
    o = opp.create_opportunity(conn, account_id=f"LIVETEST-W{n}", opportunity_type="upsell",
                               catalog_item_id=item)
    q = quotes.create_draft(conn, account_id=f"LIVETEST-W{n}", currency="GBP",
                            valid_until=TODAY + dt.timedelta(days=5), created_by="sub-a",
                            lines=[{"catalog_item_id": item, "quantity": D("2"),
                                    "unit_price": D("20"),
                                    "sales_opportunity_id": o["sales_opportunity_id"]}])
    quotes.submit(conn, q["sales_quote_id"], actor="sub-a")
    quotes.approve(conn, q["sales_quote_id"], approver="sub-b")
    quotes.issue(conn, q["sales_quote_id"], actor="sub-b")
    return q["sales_quote_id"], o["sales_opportunity_id"]


def test_a_win_is_recorded_once_and_closes_the_opportunity(live_db):
    conn, dist = live_db
    qid, oid = _issued_quote(conn, dist)
    row = outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won",
                                  outcome_date=TODAY, recorded_by="sub-b")
    assert (row["won_value"], row["won_margin"]) == (D("40.00"), D("20.00"))
    assert opp.get_opportunity(conn, oid)["outcome"] == "won"
    with pytest.raises(StateConflict):
        outcomes.record_outcome(conn, sales_quote_id=qid, outcome="lost",
                                outcome_date=TODAY, recorded_by="sub-b")


def test_an_expired_quote_leaves_the_opportunity_open(live_db):
    conn, dist = live_db
    qid, oid = _issued_quote(conn, dist)
    outcomes.record_outcome(conn, sales_quote_id=qid, outcome="expired",
                            outcome_date=TODAY, recorded_by="sub-b")
    assert quotes.get_quote(conn, qid)["status"] == "expired"
    assert opp.get_opportunity(conn, oid)["outcome"] == "open"


def test_a_lost_reason_on_a_win_is_refused(live_db):
    conn, dist = live_db
    qid, _ = _issued_quote(conn, dist)
    with pytest.raises(ValueError):
        outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won", lost_reason="price",
                                outcome_date=TODAY, recorded_by="sub-b")


def test_below_the_threshold_win_probability_stays_null(live_db):
    """Acceptance criterion 7."""
    conn, dist = live_db
    qid, _ = _issued_quote(conn, dist, n=1)
    outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won",
                            outcome_date=TODAY, recorded_by="sub-b")
    _, open_oid = _issued_quote(conn, dist, n=2)
    calibration.calibrate(conn)  # threshold 30; one closed upsell in the test data
    got = opp.get_opportunity(conn, open_oid)
    assert (got["win_probability"], got["win_probability_basis"]) == (None, None)


def test_at_the_threshold_open_opportunities_are_calibrated(live_db, monkeypatch):
    conn, dist = live_db
    monkeypatch.setattr(calibration, "_MIN_CLOSED", lambda: 1)
    qid, _ = _issued_quote(conn, dist, n=1)
    outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won",
                            outcome_date=TODAY, recorded_by="sub-b")
    _, open_oid = _issued_quote(conn, dist, n=2)
    calibration.calibrate(conn)
    got = opp.get_opportunity(conn, open_oid)
    assert got["win_probability_basis"] == "calibrated"
    assert got["win_probability"] is not None
```

Note: calibration runs over the whole table. On `bp_testdb` these tables hold only LIVETEST rows, so the assertions are exact; if another session's rows ever appear, the threshold test still holds (it asserts NULL on a row only calibration could write).

- [ ] **Step 3: Implement `outcomes.py`:**

```python
"""Won or lost, and why -- the only thing that can ever calibrate win_probability."""
from __future__ import annotations

from typing import Any, Dict, Optional

import psycopg2.errors

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor

QUOTE_OUTCOMES = frozenset({"won", "lost", "expired", "withdrawn"})
LOST_REASONS = frozenset({"price", "lead_time", "incumbent", "no_budget", "spec", "other"})
_OPPORTUNITY_OUTCOME = {"won": "won", "lost": "lost", "withdrawn": "withdrawn"}


def record_outcome(conn: Any, *, sales_quote_id: int, outcome: str, outcome_date: Any,
                   recorded_by: Optional[str], lost_reason: Optional[str] = None,
                   competitor_name: Optional[str] = None) -> Dict[str, Any]:
    if outcome not in QUOTE_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(QUOTE_OUTCOMES)}")
    if lost_reason is not None:
        if outcome != "lost":
            raise ValueError("lost_reason is only recorded for a lost quote")
        if lost_reason not in LOST_REASONS:
            raise ValueError(f"lost_reason must be one of {sorted(LOST_REASONS)}")
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT status, total_ex_tax, total_margin FROM proc.bp_sales_quote "
                    "WHERE sales_quote_id = %s FOR UPDATE", (sales_quote_id,))
        q = cur.fetchone()
        if q is None:
            raise NotFound(f"quote {sales_quote_id} does not exist")
        if q["status"] != "issued":
            raise StateConflict(f"quote {sales_quote_id} is {q['status']}; only an issued "
                                "quote has an outcome")
        won = outcome == "won"
        try:
            cur.execute(
                "INSERT INTO proc.bp_sales_quote_outcome (sales_quote_id, outcome, outcome_date, "
                "lost_reason, competitor_name, won_value, won_margin, recorded_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (sales_quote_id, outcome, outcome_date, lost_reason, competitor_name,
                 q["total_ex_tax"] if won else None, q["total_margin"] if won else None,
                 recorded_by))
        except psycopg2.errors.UniqueViolation:
            raise StateConflict(f"quote {sales_quote_id} already has an outcome") from None
        row = dict(cur.fetchone())
        if outcome == "expired":
            cur.execute("UPDATE proc.bp_sales_quote SET status = 'expired', "
                        "last_modified_date = now() WHERE sales_quote_id = %s", (sales_quote_id,))
        if outcome in _OPPORTUNITY_OUTCOME:
            cur.execute(
                "UPDATE proc.bp_sales_opportunity SET outcome = %s, last_modified_date = now() "
                "WHERE outcome = 'open' AND sales_opportunity_id IN (SELECT sales_opportunity_id "
                "FROM proc.bp_sales_quote_line WHERE sales_quote_id = %s "
                "AND sales_opportunity_id IS NOT NULL)",
                (_OPPORTUNITY_OUTCOME[outcome], sales_quote_id))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row
```

The outcome `StateConflict` from the unique violation is raised inside the outer `try`, so the outer `except` rolls back before it propagates — keep that structure.

- [ ] **Step 4: Implement `calibration.py`:**

```python
"""Win probability from closed outcomes, or no win probability at all (spec §4.5).

Measured per opportunity_type as won / (won + lost). Withdrawn is excluded: we
withdrew, the customer decided nothing. Below the governed minimum of closed
outcomes a type gets NOTHING -- an uncalibrated 0.5 in the column would be
indistinguishable from a measured one.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List, Mapping, Optional, Tuple

from src.services.governed_limits import limit as _governed_limit
from src.services.sell_side._db import dict_cursor
from src.services.sell_side.money import q4


def _MIN_CLOSED() -> int:
    return _governed_limit("reseller_catalog", "calibration_min_closed", cast=int)


@dataclass(frozen=True)
class Calibration:
    opportunity_type: str
    won: int
    lost: int
    rate: Optional[Decimal]
    applied: bool


def rates(counts: Mapping[str, Tuple[int, int]], min_closed: int) -> List[Calibration]:
    out = []
    for kind in sorted(counts):
        won, lost = counts[kind]
        closed = won + lost
        if closed >= min_closed and closed > 0:
            out.append(Calibration(kind, won, lost, q4(Decimal(won) / Decimal(closed)), True))
        else:
            out.append(Calibration(kind, won, lost, None, False))
    return out


def calibrate(conn: Any) -> List[Calibration]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT opportunity_type, count(*) FILTER (WHERE outcome = 'won') AS won, "
        "count(*) FILTER (WHERE outcome = 'lost') AS lost "
        "FROM proc.bp_sales_opportunity GROUP BY opportunity_type")
    counts = {r["opportunity_type"]: (r["won"], r["lost"]) for r in cur.fetchall()}
    result = rates(counts, _MIN_CLOSED())
    for c in result:
        if c.applied:
            cur.execute(
                "UPDATE proc.bp_sales_opportunity SET win_probability = %s, "
                "win_probability_basis = 'calibrated', last_modified_date = now() "
                "WHERE opportunity_type = %s AND outcome = 'open' "
                "AND (win_probability_basis IS NULL OR win_probability_basis = 'calibrated')",
                (c.rate, c.opportunity_type))
    conn.commit()
    return result
```

- [ ] **Step 5: The daily job.** In `src/services/backend_scheduler.py`, beside `CAPTURE_RETENTION_JOB_NAME` add `SALES_CALIBRATION_JOB_NAME = "sales_win_probability_calibration"`; add the call `self._register_sales_calibration_job()` on the line after `self._register_capture_retention_job()` (~422); and add, after `_run_capture_retention`:

```python
    def _register_sales_calibration_job(self) -> None:
        """Daily: set win_probability from closed quote outcomes (spec §4.5).
        Writes nothing until a type has the governed minimum of closed outcomes."""
        if self.SALES_CALIBRATION_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.SALES_CALIBRATION_JOB_NAME,
            self._run_sales_calibration,
            interval=timedelta(days=1),
        )

    def _run_sales_calibration(self) -> None:
        try:
            from src.services.db import get_conn
            from src.services.sell_side.calibration import calibrate
            with get_conn() as conn:
                applied = [c for c in calibrate(conn) if c.applied]
            if applied:
                logger.info("sales calibration: %s", ", ".join(
                    f"{c.opportunity_type}={c.rate} ({c.won}/{c.won + c.lost})" for c in applied))
        except Exception:
            logger.exception("sales win-probability calibration failed")
```

Add to `tests/sell_side/test_calibration.py`:

```python
def test_the_scheduler_registers_the_daily_calibration():
    from src.services import backend_scheduler as bs
    assert "_register_sales_calibration_job()" in open(bs.__file__).read()
    assert bs.BackendScheduler.SALES_CALIBRATION_JOB_NAME == "sales_win_probability_calibration"
```

(First confirm the class name with `grep -n "^class " src/services/backend_scheduler.py` and use the one that defines `register_job`.)

- [ ] **Step 6: Run** pure + live (`PROCWISE_TEST_LIVE_DB=1`) → all pass, 0 skipped. Also `./venv/bin/python -m pytest tests/ -q -k scheduler` → no new failures.

- [ ] **Step 7: Mutation proof.** In `rates`, change `closed >= min_closed` to `closed >= 0` → `test_one_short_of_the_threshold_gets_nothing` and `test_below_the_threshold_win_probability_stays_null` red. Restore.

- [ ] **Step 8: Commit** — `feat(sell-side): won or lost, and a win probability only once it is measured` + trailer.

### Task 9: HTTP — `/catalog` and `/sales` routers

**Files:**
- Create: `src/api/sell_side_http.py` (error mapping; NOT under `routers/`)
- Create: `src/api/routers/catalog.py`, `src/api/routers/sales.py`
- Modify: `src/api/main.py` — add `from api.routers import catalog as catalog_router` and `from api.routers import sales as sales_router` beside the other router imports (~line 71), and `catalog_router.router,` + `sales_router.router,` at the end of `_AUTHENTICATED_ROUTERS` (~line 577)
- Test: `tests/api/test_catalog_router.py`, `tests/api/test_sales_router.py`

**Interfaces:**
- Consumes: everything from Tasks 1 and 3–8, with the exact names given there.
- Produces (endpoints, all mounted behind `require_user` by `main.py`):

| Method + path | Gate action | Service call | Identity from token → |
|---|---|---|---|
| `GET /catalog/mappings/{profile}` | — | `catalog_import.get_mapping` | |
| `PUT /catalog/mappings/{profile}` | `catalog.write` | `catalog_import.save_mapping` | |
| `POST /catalog/import` (multipart) | `catalog.write` | `catalog_import.import_catalog` | `imported_by` |
| `POST /catalog/matches/propose` | `catalog.write` | `catalog_match.propose_matches` | |
| `GET /catalog/matches` | — | `catalog_match.list_matches` | |
| `POST /catalog/matches/{id}/confirm` · `/reject` | `catalog.write` | `confirm_match` / `reject_match` | `reviewer` |
| `POST /catalog/matches/human` | `catalog.write` | `record_human_match` | `reviewer` |
| `POST /sales/accounts` | `account.write` | `accounts.create_account` | |
| `GET /sales/accounts/{id}` | — | `accounts.get_account` | |
| `POST /sales/accounts/{id}/contacts` | `account.write` | `accounts.add_contact` | |
| `PUT /sales/accounts/{id}/history-scope/{source_kind}` | `account.write` | `accounts.set_history_scope` | |
| `POST /sales/opportunities` | `sales.write` | `opportunities.create_opportunity` | |
| `GET /sales/opportunities` · `GET /sales/opportunities/{id}` | — | `list_opportunities` / `get_opportunity` | |
| `POST /sales/opportunities/{id}/justifications` | `sales.write` | `add_justification` | |
| `PUT /sales/opportunities/{id}/stage` | `sales.write` | `set_stage` | |
| `POST /sales/quotes` | `sales.write` | `quotes.create_draft` | `created_by` |
| `GET /sales/quotes/{id}` | — | `quotes.get_quote` + `margin_note` | |
| `POST /sales/quotes/{id}/submit` | `sales.write` | `quotes.submit` | `actor` |
| `POST /sales/quotes/{id}/approve` | `sales_quote.approve` | `quotes.approve` | `approver` |
| `POST /sales/quotes/{id}/issue` | `sales_quote.issue` | `quotes.issue` | `actor` |
| `GET /sales/quotes/{id}/customer` | — | `quote_render.customer_view` | |
| `GET /sales/quotes/{id}/customer.html` | — | `quote_render.render_html` | |
| `POST /sales/quotes/{id}/outcome` | `sales.write` | `outcomes.record_outcome` | `recorded_by` |
| `POST /sales/calibrate` | `sales.calibrate` | `calibration.calibrate` | |

Errors: `NotFound` → 404, `StateConflict` and `NotCustomerReady` → 409, other `ValueError` → 422. The gate is called **before** any service call. No request body has a field that names who acted.

- [ ] **Step 1: Write `tests/api/test_catalog_router.py`** (failing):

```python
import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoint_gate import NotPermitted
from api.routers import catalog as cr
from src.services.catalog_import import ImportResult
from src.services.sell_side._db import NotFound, StateConflict

CALLER, OTHER = "sub-real-caller", "sub-someone-else"


class _P:
    subject = CALLER


@pytest.fixture
def client(monkeypatch):
    gates = []
    monkeypatch.setattr(cr, "gate", lambda action, *a, **k: gates.append(action))
    monkeypatch.setattr(cr, "get_conn", lambda: contextlib.nullcontext("CONN"))
    monkeypatch.setattr(cr, "_max_upload_bytes", lambda: 1000)
    app = FastAPI()
    app.include_router(cr.router)
    app.dependency_overrides[cr.require_user] = lambda: _P()
    c = TestClient(app)
    c.gates = gates
    return c


def _upload(client, name="feed.csv", body=b"SKU,Description,Ccy\nA1,W,GBP\n"):
    return client.post("/catalog/import", files={"file": (name, body, "text/csv")},
                       data={"distributor_id": "SUP-1", "feed_name": "March",
                             "mapping_profile": "p1", "price_effective": "2026-03-01"})


def test_an_import_is_attributed_to_the_token(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: seen.update(k) or ImportResult(status="imported"))
    r = _upload(client)
    assert r.status_code == 200, r.text
    assert seen["imported_by"] == CALLER
    assert client.gates == ["catalog.write"]


def test_an_oversize_feed_is_refused_before_import(client, monkeypatch):
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: pytest.fail("import must not run"))
    assert _upload(client, body=b"x" * 1001).status_code == 413


def test_a_file_that_is_not_a_spreadsheet_is_refused(client, monkeypatch):
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: pytest.fail("import must not run"))
    assert _upload(client, name="feed.pdf").status_code == 415


def test_a_refused_gate_stops_the_import(client, monkeypatch):
    def _refuse(*a, **k):
        raise NotPermitted("no")
    monkeypatch.setattr(cr, "gate", _refuse)
    monkeypatch.setattr(cr.catalog_import, "import_catalog",
                        lambda **k: pytest.fail("import must not run"))
    assert _upload(client).status_code == 403


def test_a_match_decision_is_the_callers_not_the_bodys(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(cr.catalog_match, "confirm_match",
                        lambda conn, mid, reviewer: seen.update(reviewer=reviewer) or {"match_id": mid})
    r = client.post("/catalog/matches/5/confirm", json={"reviewer": OTHER})
    assert r.status_code == 200 and seen["reviewer"] == CALLER


@pytest.mark.parametrize("exc,code", [(NotFound("x"), 404), (StateConflict("x"), 409),
                                      (ValueError("x"), 422)])
def test_service_errors_map_to_status_codes(client, monkeypatch, exc, code):
    def _raise(*a, **k):
        raise exc
    monkeypatch.setattr(cr.catalog_match, "reject_match", _raise)
    assert client.post("/catalog/matches/5/reject").status_code == code
```

- [ ] **Step 2: Write `tests/api/test_sales_router.py`** (failing):

```python
import contextlib
import datetime as dt
from decimal import Decimal as D

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import sales as sr
from src.services.sell_side import quote_render as qr

CALLER, OTHER = "sub-real-caller", "sub-someone-else"


class _P:
    subject = CALLER


@pytest.fixture
def client(monkeypatch):
    gates = []
    monkeypatch.setattr(sr, "gate", lambda action, *a, **k: gates.append(action))
    monkeypatch.setattr(sr, "get_conn", lambda: contextlib.nullcontext("CONN"))
    app = FastAPI()
    app.include_router(sr.router)
    app.dependency_overrides[sr.require_user] = lambda: _P()
    c = TestClient(app)
    c.gates = gates
    return c


def _quote(status="approved"):
    return {"sales_quote_id": 9, "quote_ref": "SQ-1", "status": status, "account_name": "A",
            "contact_name": None, "currency": "GBP", "quote_date": dt.date(2026, 9, 11),
            "valid_until": dt.date(2026, 10, 11), "total_ex_tax": D("56.00"),
            "total_cost": D("40.00"), "total_margin": D("16.00"), "margin_pct": D("0.2857"),
            "lines": [{"line_no": 1, "item_description": "W", "quantity": D("4"),
                       "unit_price": D("14"), "line_total": D("56.00"),
                       "unit_cost": D("10"), "line_margin": D("16.00")}],
            "justifications": []}


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_a_quote_is_created_by_the_token_holder(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr.quotes, "create_draft", lambda conn, **k: seen.update(k) or _quote("draft"))
    r = client.post("/sales/quotes", json={
        "account_id": "ACC-1", "currency": "GBP", "valid_until": "2026-10-11",
        "created_by": OTHER,
        "lines": [{"catalog_item_id": 1, "quantity": "4", "unit_price": "14"}]})
    assert r.status_code == 200, r.text
    assert seen["created_by"] == CALLER
    assert client.gates == ["sales.write"]


def test_approval_asks_the_transact_gate_and_uses_the_token(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr.quotes, "approve",
                        lambda conn, qid, approver: seen.update(approver=approver) or _quote())
    assert client.post("/sales/quotes/9/approve").status_code == 200
    assert seen["approver"] == CALLER
    assert client.gates == ["sales_quote.approve"]


def test_issuing_asks_the_communicate_gate(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "issue", lambda conn, qid, actor: _quote("issued"))
    client.post("/sales/quotes/9/issue")
    assert client.gates == ["sales_quote.issue"]


def test_the_customer_endpoint_carries_no_internal_field(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote())
    r = client.get("/sales/quotes/9/customer")
    assert r.status_code == 200
    assert not set(_walk_keys(r.json())) & qr.INTERNAL_FIELDS


def test_the_customer_html_carries_no_cost(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote())
    r = client.get("/sales/quotes/9/customer.html")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "16.00" not in r.text          # line and total margin
    assert ">10<" not in r.text           # unit cost would only ever appear as a cell


def test_a_draft_has_no_customer_view(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote("draft"))
    assert client.get("/sales/quotes/9/customer").status_code == 409


def test_the_internal_view_says_margin_is_front_end_only(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote())
    body = client.get("/sales/quotes/9").json()
    assert body["total_margin"] == "16.00"
    assert "front-end" in body["margin_note"]


def test_an_outcome_is_recorded_by_the_token_holder(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr.outcomes, "record_outcome", lambda conn, **k: seen.update(k) or {"outcome": "won"})
    client.post("/sales/quotes/9/outcome", json={"outcome": "won", "outcome_date": "2026-09-11",
                                                 "recorded_by": OTHER})
    assert seen["recorded_by"] == CALLER


def test_calibration_asks_the_compute_gate(client, monkeypatch):
    monkeypatch.setattr(sr.calibration, "calibrate", lambda conn: [])
    assert client.post("/sales/calibrate").status_code == 200
    assert client.gates == ["sales.calibrate"]
```

(The HTML assertion in `test_the_customer_html_carries_no_cost` is fiddly — simplify it to: `assert "16.00" not in r.text` and `assert ">10<" not in r.text`. The unit cost `10` would only appear as a cell value.)

- [ ] **Step 3: Run** `./venv/bin/python -m pytest tests/api/test_catalog_router.py tests/api/test_sales_router.py -q` → ImportError.

- [ ] **Step 4: Write `src/api/sell_side_http.py`:**

```python
"""One mapping from sell-side service errors to HTTP, shared by /catalog and /sales."""
from __future__ import annotations

from contextlib import contextmanager

from fastapi import HTTPException

from src.services.sell_side._db import NotFound, StateConflict
from src.services.sell_side.quote_render import NotCustomerReady


@contextmanager
def http_errors():
    try:
        yield
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (StateConflict, NotCustomerReady) as exc:   # before ValueError: both subclass it
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def money_json(obj):
    """Money leaves as a string. FastAPI's default encoder turns Decimal('16.00')
    into the float 16.0 (verified: fastapi 0.140.5), which drops the scale and
    invites float arithmetic on a price downstream."""
    from decimal import Decimal

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    return JSONResponse(content=jsonable_encoder(obj, custom_encoder={Decimal: str}))
```

**Every JSON-returning endpoint in both routers returns `money_json(<value>)`** instead of the bare value — e.g. `return money_json(quotes.approve(c, quote_id, approver=_subject(principal)))`, `return money_json(asdict(result))`, `return money_json({"count": len(rows), "matches": rows})`. The only exception is `customer_html`, which returns `HTMLResponse`. Import it with `from api.sell_side_http import http_errors, money_json`. The router tests above assert string money (`body["total_margin"] == "16.00"`) and so fail if any endpoint is missed on a money-bearing path; add to `tests/api/test_sales_router.py`:

```python
def test_money_leaves_as_a_string_not_a_float(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote())
    body = client.get("/sales/quotes/9/customer").json()
    assert body["total_ex_tax"] == "56.00"
    assert body["lines"][0]["unit_price"] == "14"
```

- [ ] **Step 5: Write `src/api/routers/catalog.py`:**

```python
"""Distributor catalog API: column maps, feed import, SKU <-> history matches.

A feed is asserted data read by a column map -- no model runs here. A match is a
claim, so it is proposed and a person decides; who decided comes from the token.
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.services import catalog_import, catalog_match
from src.services.db import get_conn

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.sell_side_http import http_errors

router = APIRouter(prefix="/catalog", tags=["Catalog"])
_AGENT = "CatalogRouter"
_FEED_SUFFIXES = (".csv", ".xlsx")


def _subject(principal) -> Optional[str]:
    return getattr(principal, "subject", None) or None


def _max_upload_bytes() -> int:
    """The same per-file cap document intake enforces, from the same policy row.
    Imported lazily: the documents router pulls in the RAG stack."""
    from api.routers.documents import _intake_limits

    return _intake_limits()[1]


class MappingEntry(BaseModel):
    target_column: str
    source_header: str
    transform: Optional[str] = None
    is_required: bool = False


class MappingBody(BaseModel):
    distributor_id: str
    entries: List[MappingEntry]


class ProposeBody(BaseModel):
    distributor_id: str


class HumanMatchBody(BaseModel):
    distributor_id: str
    distributor_sku: str
    item_id: str


@router.get("/mappings/{mapping_profile}")
def get_mapping(mapping_profile: str):
    with get_conn() as c:
        return {"mapping_profile": mapping_profile,
                "entries": catalog_import.get_mapping(c, mapping_profile)}


@router.put("/mappings/{mapping_profile}")
def put_mapping(mapping_profile: str, body: MappingBody, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT,
         context={"mapping_profile": mapping_profile, "distributor_id": body.distributor_id})
    with http_errors(), get_conn() as c:
        return {"mapping_profile": mapping_profile, "entries": catalog_import.save_mapping(
            c, mapping_profile=mapping_profile, distributor_id=body.distributor_id,
            entries=[e.model_dump() for e in body.entries])}


@router.post("/import")
async def import_feed(
    file: UploadFile = File(...), distributor_id: str = Form(...),
    feed_name: str = Form(...), mapping_profile: str = Form(...),
    price_effective: dt.date = Form(...), principal=Depends(require_user),
):
    gate("catalog.write", principal, agent=_AGENT,
         context={"distributor_id": distributor_id, "feed_name": feed_name})
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in _FEED_SUFFIXES:
        raise HTTPException(status_code=415,
                            detail=f"a catalog feed is one of {_FEED_SUFFIXES}, not {suffix or 'unnamed'}")
    max_bytes = _max_upload_bytes()
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"feed exceeds {max_bytes} bytes")
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        result = await run_in_threadpool(
            catalog_import.import_catalog, distributor_id=distributor_id,
            feed_name=feed_name, mapping_profile=mapping_profile,
            price_effective=price_effective, imported_by=_subject(principal),
            file_bytes=data, file_name=file.filename, file_path=tmp.name)
    return asdict(result)


@router.post("/matches/propose")
def propose(body: ProposeBody, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context={"distributor_id": body.distributor_id})
    with http_errors(), get_conn() as c:
        return catalog_match.propose_matches(c, body.distributor_id)


@router.get("/matches")
def list_matches(distributor_id: Optional[str] = None, status: str = "proposed", limit: int = 100):
    with get_conn() as c:
        rows = catalog_match.list_matches(c, distributor_id=distributor_id, status=status, limit=limit)
    return {"count": len(rows), "matches": rows}


@router.post("/matches/{match_id}/confirm")
def confirm(match_id: int, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context={"match_id": match_id})
    with http_errors(), get_conn() as c:
        return catalog_match.confirm_match(c, match_id, _subject(principal))


@router.post("/matches/{match_id}/reject")
def reject(match_id: int, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context={"match_id": match_id})
    with http_errors(), get_conn() as c:
        return catalog_match.reject_match(c, match_id, _subject(principal))


@router.post("/matches/human")
def human_match(body: HumanMatchBody, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context=body.model_dump())
    with http_errors(), get_conn() as c:
        return catalog_match.record_human_match(
            c, distributor_id=body.distributor_id, distributor_sku=body.distributor_sku,
            item_id=body.item_id, reviewer=_subject(principal))
```

`_FEED_SUFFIXES` must be `(".csv", ".xlsx", ".xls")` — exactly the spreadsheet suffixes `src/services/extraction_v3/parsers/router.py:70-76` dispatches (verified). Change the constant in the code block above accordingly.

The confirm/reject tests post with a JSON body containing `reviewer`; the endpoints declare no body, so FastAPI ignores it — that is the point.

- [ ] **Step 6: Write `src/api/routers/sales.py`:**

```python
"""Sell-side API: accounts, opportunities, outbound quotes, outcomes.

Who acted -- created_by, approver, recorded_by -- is the token's subject and
nothing else; no body here has a field for it. Approving a quote is a transact
action and issuing one is communicate, so both need a stated permit
(deploy/sql/2026-09-11_reseller_governance.sql). The customer endpoints go
through quote_render, which is the control on cost and margin.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.services.db import get_conn
from src.services.sell_side import (accounts, calibration, opportunities, outcomes,
                                    quote_render, quotes)

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.sell_side_http import http_errors

router = APIRouter(prefix="/sales", tags=["Sales"])
_AGENT = "SalesRouter"
MARGIN_NOTE = ("Front-end margin only: distributor back-end rebates are not modelled, "
               "so this understates what a line may finally earn.")


def _subject(principal) -> Optional[str]:
    return getattr(principal, "subject", None) or None


class AccountBody(BaseModel):
    account_name: str
    account_id: Optional[str] = None
    trading_name: Optional[str] = None
    also_supplier_id: Optional[str] = None
    registration_number: Optional[str] = None
    vat_number: Optional[str] = None
    country: Optional[str] = None
    default_currency: Optional[str] = None
    payment_terms: Optional[str] = None
    credit_limit_amount: Optional[Decimal] = None
    account_owner_email: Optional[str] = None


class ContactBody(BaseModel):
    contact_name: str
    contact_role: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    is_primary: bool = False


class ScopeBody(BaseModel):
    completeness: str
    covers_from: Optional[dt.date] = None
    covers_to: Optional[dt.date] = None
    note: Optional[str] = None


class OpportunityBody(BaseModel):
    account_id: str
    opportunity_type: str
    catalog_item_id: Optional[int] = None
    currency: Optional[str] = None
    expected_quantity: Optional[Decimal] = None
    expected_unit_price: Optional[Decimal] = None
    phase_id: Optional[str] = "sales.opportunity"
    subprocess_id: Optional[str] = "sales.opportunity.qualified"
    detector_type: Optional[str] = None
    reason_codes: Optional[List[str]] = None


class JustificationBody(BaseModel):
    kind: str
    claim: str
    evidence_ref: Optional[str] = None
    evidence_value: Optional[Decimal] = None
    customer_safe: bool = True


class StageBody(BaseModel):
    phase_id: str
    subprocess_id: Optional[str] = None


class QuoteLine(BaseModel):
    catalog_item_id: int
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    sales_opportunity_id: Optional[int] = None
    justification_id: Optional[int] = None


class QuoteBody(BaseModel):
    account_id: str
    currency: str
    valid_until: dt.date
    lines: List[QuoteLine]
    contact_id: Optional[int] = None
    quote_date: Optional[dt.date] = None
    supersedes_id: Optional[int] = None


class OutcomeBody(BaseModel):
    outcome: str
    outcome_date: dt.date
    lost_reason: Optional[str] = None
    competitor_name: Optional[str] = None


# --- accounts ---------------------------------------------------------------

@router.post("/accounts")
def create_account(body: AccountBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    fields = body.model_dump(exclude_none=True)
    name = fields.pop("account_name")
    with http_errors(), get_conn() as c:
        return accounts.create_account(c, account_name=name, **fields)


@router.get("/accounts/{account_id}")
def get_account(account_id: str):
    with http_errors(), get_conn() as c:
        return accounts.get_account(c, account_id)


@router.post("/accounts/{account_id}/contacts")
def add_contact(account_id: str, body: ContactBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return accounts.add_contact(c, account_id, **body.model_dump())


@router.put("/accounts/{account_id}/history-scope/{source_kind}")
def set_scope(account_id: str, source_kind: str, body: ScopeBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return accounts.set_history_scope(c, account_id, source_kind=source_kind, **body.model_dump())


# --- opportunities ----------------------------------------------------------

@router.post("/opportunities")
def create_opportunity(body: OpportunityBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    with http_errors(), get_conn() as c:
        return opportunities.create_opportunity(c, **body.model_dump())


@router.get("/opportunities")
def list_opportunities(account_id: Optional[str] = None, outcome: Optional[str] = None,
                       limit: int = 100):
    with get_conn() as c:
        rows = opportunities.list_opportunities(c, account_id=account_id, outcome=outcome, limit=limit)
    return {"count": len(rows), "opportunities": rows}


@router.get("/opportunities/{opportunity_id}")
def get_opportunity(opportunity_id: int):
    with http_errors(), get_conn() as c:
        return opportunities.get_opportunity(c, opportunity_id)


@router.post("/opportunities/{opportunity_id}/justifications")
def add_justification(opportunity_id: int, body: JustificationBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_opportunity_id": opportunity_id})
    with http_errors(), get_conn() as c:
        return opportunities.add_justification(c, opportunity_id, **body.model_dump())


@router.put("/opportunities/{opportunity_id}/stage")
def set_stage(opportunity_id: int, body: StageBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_opportunity_id": opportunity_id})
    with http_errors(), get_conn() as c:
        return opportunities.set_stage(c, opportunity_id, **body.model_dump())


# --- quotes -----------------------------------------------------------------

@router.post("/quotes")
def create_quote(body: QuoteBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    data = body.model_dump()
    data["lines"] = [l.model_dump() for l in body.lines]
    with http_errors(), get_conn() as c:
        return quotes.create_draft(c, created_by=_subject(principal), **data)


@router.get("/quotes/{quote_id}")
def get_quote(quote_id: int):
    """INTERNAL view: carries cost and margin. Never hand this to a customer."""
    with http_errors(), get_conn() as c:
        return {**quotes.get_quote(c, quote_id), "margin_note": MARGIN_NOTE}


@router.post("/quotes/{quote_id}/submit")
def submit(quote_id: int, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return quotes.submit(c, quote_id, actor=_subject(principal))


@router.post("/quotes/{quote_id}/approve")
def approve(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.approve", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return quotes.approve(c, quote_id, approver=_subject(principal))


@router.post("/quotes/{quote_id}/issue")
def issue(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.issue", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return quotes.issue(c, quote_id, actor=_subject(principal))


@router.get("/quotes/{quote_id}/customer")
def customer_view(quote_id: int):
    with http_errors(), get_conn() as c:
        return quote_render.customer_view(quotes.get_quote(c, quote_id))


@router.get("/quotes/{quote_id}/customer.html", response_class=HTMLResponse)
def customer_html(quote_id: int):
    with http_errors(), get_conn() as c:
        return HTMLResponse(quote_render.render_html(
            quote_render.customer_view(quotes.get_quote(c, quote_id))))


@router.post("/quotes/{quote_id}/outcome")
def record_outcome(quote_id: int, body: OutcomeBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return outcomes.record_outcome(c, sales_quote_id=quote_id,
                                       recorded_by=_subject(principal), **body.model_dump())


@router.post("/calibrate")
def calibrate(principal=Depends(require_user)):
    gate("sales.calibrate", principal, agent=_AGENT)
    with get_conn() as c:
        return {"calibrations": [c_.__dict__ for c_ in calibration.calibrate(c)]}
```

Note: the service test stubs in Step 2 use positional/keyword shapes matching these calls exactly (`quotes.approve(conn, qid, approver=...)` is called as `quotes.approve(c, quote_id, approver=...)` — stubs take `(conn, qid, approver)`). If a stub signature and a call disagree, fix the stub, not the router.

- [ ] **Step 7: Register** both routers in `src/api/main.py` (imports + `_AUTHENTICATED_ROUTERS`).

- [ ] **Step 8: Run**

```bash
./venv/bin/python -m pytest tests/api/test_catalog_router.py tests/api/test_sales_router.py tests/api/test_every_router_is_authenticated.py -q
```
Expected: all pass. Then `./venv/bin/python -m pytest tests/api -q` — no new failures versus a baseline run in a detached worktree (never `git stash`): `git worktree add --detach /tmp/claude-1001/.../scratchpad/base HEAD~1` and compare failure lists.

- [ ] **Step 9: Mutation proof.** In `sales.approve` pass `approver="api"` instead of the token subject → `test_approval_asks_the_transact_gate_and_uses_the_token` red. Swap the `except` order in `http_errors` (ValueError first) → the 409 parametrised case red. Restore.

- [ ] **Step 10: Commit** — `feat(api): /catalog and /sales, where the token says who acted` + trailer.

### Task 10: Acceptance on live data, and an end-to-end HTTP walk (controller-run)

Spec §9 criteria 2–4 were only ever covered by a fake connection. This proves them through the real parser into the real `bp_testdb`, then walks the whole flow over HTTP.

**Why not the product server:** `src/api/main.py` lifespan starts the full backend scheduler (promotion, deal assignment, email watcher, …) and there is no master switch. A second full server against the same database would duplicate every scheduled job. The walk therefore runs a side-port app that mounts **only** the two new routers, behind the real `require_user` dependency and the real gate — the same mount `main.py` uses. `procwise.service` is not restarted by this plan.

**Files:**
- Create: `tests/sell_side/test_acceptance_live.py`
- Create: `scripts/serve_sell_side_demo.py`

- [ ] **Step 1: Write `tests/sell_side/test_acceptance_live.py`:**

```python
"""Spec §9 criteria 2-4 through the REAL parser into the REAL database.
The unit suite proves them against a fake connection; this proves them where
they matter."""
import datetime as dt
from decimal import Decimal as D

from src.services import catalog_import
from tests.sell_side.conftest import SENTINEL, live

pytestmark = live
PROFILE = "livetest_v1"
NOCOST = "livetest_nocost_v1"
BASE = [{"target_column": "distributor_sku", "source_header": "SKU"},
        {"target_column": "item_description", "source_header": "Description"},
        {"target_column": "currency", "source_header": "Ccy"}]


def _profiles(conn, dist):
    catalog_import.save_mapping(conn, mapping_profile=PROFILE, distributor_id=dist,
                                entries=BASE + [{"target_column": "cost_price",
                                                 "source_header": "Cost"}])
    catalog_import.save_mapping(conn, mapping_profile=NOCOST, distributor_id=dist, entries=BASE)


def _feed(tmp_path, name, rows, header="SKU,Description,Ccy,Cost"):
    path = tmp_path / name
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def _import(conn, dist, path, profile=PROFILE):
    return catalog_import.import_catalog(
        distributor_id=dist, feed_name=f"{SENTINEL} {path.name}", mapping_profile=profile,
        price_effective=dt.date.today(), imported_by="sub-livetest",
        file_path=str(path), conn=conn)


def _rows(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def test_criterion_2_the_same_file_twice_is_one_receipt_and_no_duplicates(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    feed = _feed(tmp_path, "a.csv", ["LIVETEST-A1,Widget,GBP,10.00",
                                     "LIVETEST-A2,Gadget,GBP,20.00"])
    first, second = _import(conn, dist, feed), _import(conn, dist, feed)
    assert (first.status, first.rows_loaded) == ("imported", 2)
    assert second.status == "duplicate" and second.source_id == first.source_id
    assert _rows(conn, "SELECT count(*) FROM proc.bp_catalog_source WHERE source_id = %s",
                 (first.source_id,)) == [(1,)]
    assert _rows(conn, "SELECT count(*) FROM proc.bp_catalog_item WHERE distributor_sku "
                 "LIKE 'LIVETEST-A%%'", ()) == [(2,)]


def test_criterion_3_a_repriced_feed_keeps_one_current_row_and_the_old_price(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    _import(conn, dist, _feed(tmp_path, "march.csv", ["LIVETEST-R1,Widget,GBP,10.00"]))
    res = _import(conn, dist, _feed(tmp_path, "april.csv", ["LIVETEST-R1,Widget,GBP,11.00"]))
    assert res.rows_versioned == 1
    rows = _rows(conn, "SELECT cost_price, valid_to IS NULL FROM proc.bp_catalog_item "
                 "WHERE distributor_sku = 'LIVETEST-R1' ORDER BY catalog_item_id", ())
    assert rows == [(D("10.0000"), False), (D("11.0000"), True)]


def test_criterion_4_a_feed_with_no_cost_column_loads_null_cost(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    res = _import(conn, dist, _feed(tmp_path, "nocost.csv",
                                    ["LIVETEST-N1,Widget,GBP", "LIVETEST-N2,Gadget,GBP"],
                                    header="SKU,Description,Ccy"), profile=NOCOST)
    assert res.rows_loaded == 2
    assert _rows(conn, "SELECT cost_price FROM proc.bp_catalog_item WHERE distributor_sku "
                 "LIKE 'LIVETEST-N%%'", ()) == [(None,), (None,)]


def test_a_bad_row_is_reported_and_the_rest_load(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    res = _import(conn, dist, _feed(tmp_path, "bad.csv", ["LIVETEST-B1,Widget,GBP,10.00",
                                                          "LIVETEST-B2,Gadget,Pounds,5.00"]))
    assert (res.status, res.rows_loaded, res.rows_rejected) == ("partial", 1, 1)
    assert _rows(conn, "SELECT status, rows_rejected FROM proc.bp_catalog_source "
                 "WHERE source_id = %s", (res.source_id,)) == [("partial", 1)]
```

- [ ] **Step 2: Run** `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side/test_acceptance_live.py -v` → 4 passed, 0 skipped.

- [ ] **Step 3: Mutation proof.** In `catalog_import.import_catalog`, delete the early `return ImportResult(status=_STATUS_DUPLICATE, ...)` → criterion 2 red. Change `_CLOSE_CURRENT` to `... WHERE false AND catalog_item_id = %s` → criterion 3 red (the partial unique index refuses the second current row; the import fails). Restore both.

- [ ] **Step 4: Write `scripts/serve_sell_side_demo.py`:**

```python
"""Serve ONLY /catalog and /sales, on the real auth dependency and the real gate,
for a local end-to-end walk.

Not the product server, on purpose: api.main's lifespan starts the full backend
scheduler, and a second one against the same database would run every scheduled
job twice. This mounts the two routers exactly the way main.py does.

Run:
  set -a && . ./.env && set +a
  PYTHONPATH=.:src ./.venv/bin/uvicorn scripts.serve_sell_side_demo:app --port 8765
"""
from fastapi import Depends, FastAPI

from api import auth as _auth
from api.routers import catalog, sales

app = FastAPI(title="Sell-side walk (catalog + sales routers only)")
for _router in (catalog.router, sales.router):
    app.include_router(_router, dependencies=[Depends(_auth.require_user)])
```

- [ ] **Step 5: The HTTP walk.** Start the server in the background with the runtime venv (`.venv`, not `venv` — they differ). Against `http://127.0.0.1:8765`, with `DIST` = the first `supplier_id` in `proc.bp_supplier`, all ids/SKUs prefixed `LIVETEST`:
  1. `PUT /catalog/mappings/livetest_walk` (SKU/Description/Ccy/Cost/RRP/Lifecycle) → 200, entries echoed.
  2. `POST /catalog/import` with a 3-row CSV (one row with currency `Pounds`) → `status: partial`, `rows_loaded: 2`, `rows_rejected: 1`, reject reason names `currency`.
  3. Same file again → `status: duplicate`.
  4. `POST /catalog/matches/propose` → counts returned; report the real numbers (expect 0 exact on this corpus — `item_id`s are synthetic `ITM…`).
  5. `POST /sales/accounts` (`account_id: LIVETEST-WALK`) → 200. Then `POST /sales/opportunities` → `win_probability: null`, money as strings.
  6. `POST /sales/quotes` → draft with `total_margin` as a string; `GET /sales/quotes/{id}` shows `margin_note`.
  7. `POST /sales/quotes/{id}/submit` → `in_review`.
  8. `POST /sales/quotes/{id}/approve` → with `ASK_AUTH_MODE=off` there is no principal, so expect **403** "no authenticated principal: irreversible actions are refused" — the permit and the gate working live. Record the response.
  9. Approve and issue through the service with named people (`quotes.approve(conn, id, approver="sub-approver")`, `quotes.issue(...)`), since the walk has no token.
  10. `GET /sales/quotes/{id}/customer` and `/customer.html` → no internal field name in the JSON (check with `jq 'paths | map(tostring) | last'` against the internal list), no cost figure in the HTML.
  11. `POST /sales/quotes/{id}/outcome` `{"outcome":"won",...}` → `won_value` set. `POST /sales/calibrate` → the type shows `applied: false` (below 30).
  12. Stop the server. Delete every `LIVETEST` row with `tests/sell_side/conftest.clean` and confirm the 14 tables hold no sentinel rows.

Save the transcript (commands + responses) to the scratchpad; the final report quotes it.

- [ ] **Step 6: Full regression.** `./venv/bin/python -m pytest tests -q -p no:cacheprovider` on this branch and on a detached baseline worktree of `Development` (never `git stash`); the only differences allowed are new passing tests. Then `PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side tests/services/test_catalog_import.py tests/services/test_catalog_match.py tests/governance/test_governed_limits.py tests/guardrails/test_policy_action_vocabulary.py -q` → 0 skipped.

- [ ] **Step 7: Commit** — `test(sell-side): the acceptance criteria, on live data, through the real parser` + trailer.

---

## Out of scope (named so nobody reads the absence as an oversight)

- **Cost-tier and relation feeds.** `bp_catalog_cost_tier` / `bp_catalog_item_relation` have tables, and `costing.cost_at` reads tiers, but no importer writes them: the spec says they arrive as separate files with their own shapes, and no such file exists to design against.
- **Opportunity detection.** Nothing here *finds* an upsell; this plan builds the write paths a detector would use. No account→purchase-history link exists in the corpus (`buyer_id` is an internal cost centre), which a detector would need first.
- **Applying to `bp_sqldb`.** Migrations go to `bp_testdb` only; the other database waits for the user.
- **UI.** The Sales mode in `beyond_procwise_ui` is not wired to these endpoints here.
- Everything in spec §8.

