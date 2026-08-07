# Phase 1b — Carry the Fact Model Across the Seam: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a typed, provenanced `CommercialFact` the atomic unit that survives from extraction to the opportunity record, so that for any opportunity you can reconstruct the unit price, quantity, unit of measure, currency, term and source document of every number contributing to it — without parsing free text.

**Architecture:** Four new tables (`bp_commercial_fact`, `bp_fact_provenance`, `bp_constraint`, `bp_finding_fact`) plus structured columns on `bp_opportunity`. A deterministic **Fact Assembler** service reads `_trgt` rows, resolves each field's evidence from `bp_extraction_provenance_v3`, normalises unit of measure and currency, and persists a fact whose provenance is non-empty by construction. `calculation_details` becomes derived and optional; a logged deprecation shim covers the one release. No LLM anywhere in this phase.

**Tech Stack:** Python 3.12, Pydantic v2 (following `src/services/benchmark/models.py` as the house template), psycopg2, PostgreSQL `proc` schema, pytest.

---

## Global Constraints

- **Provenance is mandatory and non-empty.** A `CommercialFact` with no provenance fails construction — enforced in the Pydantic validator, not by convention, and backed by a database constraint.
- **No LLM in this phase.** Everything here is deterministic. The Fact Assembler, the UoM normaliser and the FX resolver are pure functions or plain SQL.
- **Fail closed.** An unmappable unit of measure carries forward un-normalised with reason code `UOM_UNMAPPED`. An unavailable FX rate yields `FX_UNAVAILABLE`. Neither is ever coerced into a guess.
- **No fabrication.** Absent data stays NULL. `calculation_details` values that cannot be parsed into a structured column are marked `INDETERMINATE`, never inferred.
- **`tenant_id` on every new table**, per the B2 decision, defaulted to a single tenant constant. RLS is not enabled — there is no second tenant and no tenant dimension anywhere in `proc` — and that limitation is recorded rather than faked.
- **Bitemporal columns on every new table:** `valid_from`, `valid_to`, `recorded_at`.
- Migrations ship forward **and** reversible, additive, idempotent, and applied to **both** `bp_sqldb` and `bp_testdb`.
- Run tests with the environment loaded: `set -a && . ./.env && set +a && ./venv/bin/python -m pytest …`.
- Commit style: no `Co-Authored-By` lines, no AI-attribution trailers. Work stays on `Development`.
- **Shared checkout.** Another session owns uncommitted edits to `src/api/routers/stream.py`, `src/services/deal_assignment_service.py`, `tests/services/test_deal_assignment_service.py`. Never `git add -A`, `git add .`, or `git commit -a`.
- **Never launch a background job or Monitor.** Three implementers stalled that way in Phase 1a. Foreground only.

---

## What the pre-plan investigation found — read this before designing anything

Four measured facts changed this plan's design. Do not re-derive them; do verify any you depend on.

### F1. Provenance coverage is ~100%, but only if you join in the right direction

Joining *provenance → `_trgt`* looks catastrophic: of 125 quote `doc_pk`s in `bp_extraction_provenance_v3`, only 45 match a `bp_quote_trgt.quote_id`. Invoice 51 of 208. Purchase order 39 of 81.

That is not data loss. The unmatched `doc_pk`s are values like `'10'`, `'048597'`, `'005-022'` — **failed extraction attempts whose primary key was garbage and which never promoted.** Provenance records every attempt.

Joining the other way is near-perfect: `bp_quote_stg` has 45 rows and **all 45** have provenance; `bp_invoice_stg` has 112 rows and **111** do.

**Design consequence:** the Fact Assembler drives from the `_trgt` row and looks provenance up by `(doc_type, doc_pk, field_path)`. It never enumerates provenance and joins forward. Mandatory provenance is achievable precisely because of this direction.

### F2. Mandatory provenance means the seeded corpus produces almost no facts

`bp_testdb` holds 21,049 quotes and 115,814 quote lines, but only **722 provenance rows** — that corpus was seeded, not extracted. `bp_sqldb` has 64,118 provenance rows against 45 quotes.

So on `bp_testdb`, nearly every row will fail the mandatory-provenance rule and produce **no fact**. That is correct fail-closed behaviour, not a bug — but anyone testing there will conclude the assembler is broken. Task 9's acceptance must run against `bp_sqldb`, and the decision record must state the split plainly.

### F3. A finding draws on multiple documents — so facts need a join table, not flat columns

Measured on `bp_opportunity`: **301 of 308 rows have 3 source documents**, one has 4, six have 1. Flattening `unit_price` onto the opportunity row would force choosing one of three documents arbitrarily.

**Design consequence:** `bp_finding_fact` links opportunity → fact many-to-many with a `role`. Facts are independent of findings — one invoice line can support both an overbilling finding and a duplicate finding. This is the justification the brief asks for in the PR description.

### F4. `bp_fx_rates` has no historical dimension

Its columns are `base_currency, currency, rate, fetched_at` — a single snapshot (all rows `2026-07-16`), not a dated series. It cannot answer "what was GBP/USD on 2025-04-01".

**Design consequence:** the fact stores `fx_rate`, `fx_rate_date` (the `fetched_at` of the row actually used) and `fx_rate_source`. This satisfies the brief's reproducibility requirement — the fact carries its own rate, so re-rendering tomorrow cannot change yesterday's number. It does **not** deliver historical accuracy, and the plan does not pretend otherwise. Recorded as a known limitation; a dated rate corpus is separate work.

### F5. The real unit-of-measure value set is 28 strings, half of them junk

Across all three `_trgt` line tables: ten high-volume units (`case`, `tonne`, `each`, `month`, `pack`, `hour`, `box`, `day`, `licence`, `metre`), four rare but real (`shipment`, `year`, `week`, `seat`), and fourteen that are not units at all — `'30 days from quote date'`, `'45 days from invoice'`, `'annual in advance'`, `'included'`, `'transition 7 weeks'`, `'onboarding 10 weeks'`, `'implementation (one-off, fixed) — £72,000.00'`.

Those last are payment terms and scope descriptions that landed in the UoM column. They must yield `UOM_UNMAPPED` and carry forward un-normalised. This gives the normaliser a concrete, closed acceptance set: 14 map, 14 must refuse.

---

## Blocker B3 — resolved 2026-08-07: no data dictionary is needed

The brief names GPSS as "the vocabulary" and instructs us to extend it rather than shadow it. **There is no GPSS dictionary in this project, and on inspection there does not need to be one.** Everything the brief actually asks a vocabulary to do is already satisfied by artefacts this repo owns:

| What the brief wants from a vocabulary | What already provides it |
|---|---|
| Stable field names, so nobody invents a second name for an existing concept | `extraction_schemas/*.yaml` — a versioned, declarative registry of ~90 field definitions across four document types. This *is* a data dictionary; it simply isn't called one. |
| A category key for per-category behaviour | `proc.bp_category` already holds a three-level hierarchy in `L1~L2~L3` form (`Information Technology~Hardware~Desktop Computers`, `Professional Services~IT Consulting~IT Consulting`) — 49 rows, 22 distinct on `bp_sqldb`. `bp_requirement.category` is populated on 6,009 of 6,010 rows, 247 distinct. |
| Cross-document concept identity — that `unit_price` on a quote is the same concept as on an invoice | Implicit today (same name, nothing asserts it). Closed later by one `concept:` key per schema field, when Phase 5 needs it. |
| Detecting a field no renderer knows about, so it is never silently dropped | The schema registry serves as the checklist. |

A formal external dictionary is only required for **interoperability** — exchanging facts with another system, or mapping onto a customer's own taxonomy. Nothing in Phases 1–5 requires that.

**Decision:** the field is `concept_code`, not `gpss_code`, and it is sourced from the extraction-schema field name. Naming a column after a standard that does not exist here would invite the next reader to assume there is authority behind it — which is precisely the shadow-vocabulary failure the brief's constraint exists to prevent. If GPSS is later adopted, `concept_code` is the column it maps into.

> ⚠️ **Correction to the Phase 0 seam map.** That document recorded `bp_category` as empty. It is empty on `bp_testdb` but populated on `bp_sqldb`, with the hierarchy shown above. The seam map has been amended.

The UoM normaliser in Task 1 remains **category-independent** — it maps a UoM string to a canonical unit and a dimension (count / time / mass / length), not to a per-category basis. That is not because of a missing dictionary but because the per-category basis is a *rendering* concern (software → per user per month; services → per day per grade), and it belongs with the Category Profile Resolver in Phase 5, where the category hierarchy above becomes its key.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/services/facts/uom.py` | Deterministic UoM normaliser; closed map; `UOM_UNMAPPED` |
| `src/services/facts/fx.py` | Dated FX resolution against `bp_fx_rates`; `FX_UNAVAILABLE` |
| `src/services/facts/models.py` | `CommercialFact`, `FactProvenance`, `Constraint`, enums, validators |
| `src/services/facts/assembler.py` | Builds and persists facts from `_trgt` rows + provenance |
| `src/services/facts/store.py` | Persistence for facts, provenance, constraints, finding links |
| `src/services/facts/concept_codes.py` | Concept vocabulary derived from the extraction-schema field names (B3 resolution) |
| `deploy/sql/2026-08-07_commercial_fact.sql` (+ `_rollback`) | Four new tables |
| `deploy/sql/2026-08-07_opportunity_structured_columns.sql` (+ `_rollback`) | Structured columns on `bp_opportunity` + back-population |
| `tests/services/facts/test_uom.py` | 28-value acceptance set |
| `tests/services/facts/test_fx.py` | Rate resolution and `FX_UNAVAILABLE` |
| `tests/services/facts/test_models.py` | Provenance-mandatory enforcement |
| `tests/services/facts/test_assembler.py` | Fact construction from real rows |
| `tests/services/facts/test_acceptance_reconstruct.py` | The phase's acceptance query |
| `docs/remediation/01b_fact_model_seam.md` | Decision record |

---

## Task 1: The unit-of-measure normaliser

Pure functions, no I/O, no database. Built first because the fact model depends on its output type.

**Files:**
- Create: `src/services/facts/__init__.py`, `src/services/facts/uom.py`
- Create: `tests/services/facts/__init__.py`, `tests/services/facts/test_uom.py`

**Interfaces:**
- Produces: `normalise_uom(raw: str | None) -> UomResult`, where `UomResult` is a frozen dataclass `(canonical: str | None, dimension: str | None, factor: Decimal | None, reason_codes: tuple[str, ...])`. `canonical is None` ⟺ `"UOM_UNMAPPED" in reason_codes`.

- [ ] **Step 1: Write the failing test**

```python
"""The UoM normaliser must map the corpus's real units and refuse its junk.

The 28 values below are every distinct unit_of_measure across
bp_quote_line_items_trgt, bp_po_line_items_trgt and bp_invoice_line_items_trgt.
Fourteen are units. Fourteen are payment terms, scope descriptions or prices
that landed in the UoM column, and coercing any of them into a unit would be
fabrication.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.uom import UOM_UNMAPPED, normalise_uom  # noqa: E402

MAPPED = {
    "each": ("each", "count"),
    "case": ("case", "count"),
    "pack": ("pack", "count"),
    "box": ("box", "count"),
    "seat": ("seat", "count"),
    "licence": ("licence", "count"),
    "shipment": ("shipment", "count"),
    "hour": ("hour", "time"),
    "day": ("day", "time"),
    "week": ("week", "time"),
    "month": ("month", "time"),
    "year": ("year", "time"),
    "tonne": ("tonne", "mass"),
    "metre": ("metre", "length"),
}

UNMAPPED = [
    "included",
    "annual in advance",
    "30 days from quote date",
    "21 days from quote date",
    "45 days from quote date",
    "30 days from invoice",
    "45 days from invoice",
    "transition 5 weeks",
    "transition 6 weeks",
    "transition 7 weeks",
    "onboarding 7 weeks",
    "onboarding 10 weeks",
    "implementation (one-off, fixed) — £72,000.00",
    "implementation (one-off, fixed) — £58,000.00",
]


@pytest.mark.parametrize("raw,expected", sorted(MAPPED.items()))
def test_real_units_normalise(raw, expected):
    r = normalise_uom(raw)
    assert (r.canonical, r.dimension) == expected
    assert UOM_UNMAPPED not in r.reason_codes


@pytest.mark.parametrize("raw", UNMAPPED)
def test_junk_is_refused_not_coerced(raw):
    r = normalise_uom(raw)
    assert r.canonical is None, f"{raw!r} was coerced to {r.canonical!r}"
    assert r.dimension is None
    assert UOM_UNMAPPED in r.reason_codes


@pytest.mark.parametrize("raw", ["EACH", " Each ", "eaches", "EA", "hrs", "Hours"])
def test_case_whitespace_and_common_abbreviations(raw):
    """Real documents do not spell units the way the seeder did."""
    assert normalise_uom(raw).canonical is not None


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_absent_uom_is_unmapped_not_defaulted(raw):
    """A missing unit must never silently become 'each' — benchmark_live already
    does that at its own layer, and doing it here would bake the guess into the
    fact base."""
    r = normalise_uom(raw)
    assert r.canonical is None
    assert UOM_UNMAPPED in r.reason_codes


def test_result_is_hashable_and_frozen():
    r = normalise_uom("each")
    with pytest.raises(Exception):
        r.canonical = "box"  # type: ignore[misc]


def test_time_units_carry_a_factor_to_a_common_basis():
    """Cross-document comparison needs hour/day/week/month/year on one basis.
    Months and years are calendar-ambiguous, so the factor is stated in days
    with the convention recorded in reason_codes, not silently assumed."""
    assert normalise_uom("day").factor == Decimal("1")
    assert normalise_uom("week").factor == Decimal("7")
    assert normalise_uom("hour").factor is not None
    r = normalise_uom("month")
    assert r.factor is not None
    assert any("CALENDAR_CONVENTION" in c for c in r.reason_codes)
```

- [ ] **Step 2: Run to verify it fails**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/services/facts/test_uom.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.services.facts'`.

- [ ] **Step 3: Implement**

Write `src/services/facts/uom.py` with:
- A frozen `UomResult` dataclass as specified in Interfaces.
- `UOM_UNMAPPED = "UOM_UNMAPPED"` and `CALENDAR_CONVENTION = "CALENDAR_CONVENTION_30D_365D"`.
- A single `_CANONICAL` dict mapping a normalised key (lowercased, whitespace-collapsed) to `(canonical, dimension, factor)`, plus an `_ALIASES` dict for `ea`/`eaches`→`each`, `hr`/`hrs`/`hours`→`hour`, `mo`/`months`→`month`, `yr`/`years`/`annum`→`year`, `mtr`/`m`/`metres`→`metre`, `t`/`tonnes`/`mt`→`tonne`, and the plural of every canonical unit.
- Matching is **exact on the normalised key only**. Do not substring-match: `'transition 7 weeks'` contains `week`, and a substring match would turn a scope description into a time unit. This is the single most important rule in the module — state it in a comment.
- Time factors in days: hour `Decimal("1")/Decimal("24")`, day `1`, week `7`, month `30` (+ `CALENDAR_CONVENTION`), year `365` (+ `CALENDAR_CONVENTION`). Count/mass/length units get `factor = None`.
- Anything not in the map returns `UomResult(None, None, None, (UOM_UNMAPPED,))`.

- [ ] **Step 4: Run to verify it passes**

```bash
set -a && . ./.env && set +a && ./venv/bin/python -m pytest tests/services/facts/test_uom.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/services/facts/__init__.py src/services/facts/uom.py \
        tests/services/facts/__init__.py tests/services/facts/test_uom.py
git commit -m "feat(facts): deterministic unit-of-measure normaliser that refuses what it cannot map"
```

---

## Task 2: FX resolution

**Files:**
- Create: `src/services/facts/fx.py`, `tests/services/facts/test_fx.py`

**Interfaces:**
- Produces: `resolve_fx(cur, from_ccy: str, to_ccy: str) -> FxResult` — frozen `(rate: Decimal | None, rate_date: datetime | None, source: str | None, reason_codes: tuple[str, ...])`. Same-currency returns rate `1`. Unavailable returns all-None with `"FX_UNAVAILABLE"`.

- [ ] **Step 1: Write the failing test**

```python
"""FX must be resolved once, stamped onto the fact, and never looked up again
at render time — otherwise the same report renders differently tomorrow.

bp_fx_rates has no historical dimension: its columns are
(base_currency, currency, rate, fetched_at) and every row shares one
fetched_at. So rate_date is the snapshot's fetched_at, NOT the transaction
date. These tests pin that behaviour so the limitation stays visible.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.fx import FX_UNAVAILABLE, resolve_fx  # noqa: E402

_T = datetime(2026, 7, 16, 15, 35, 57, tzinfo=timezone.utc)


class _Cur:
    def __init__(self, rows):
        self._rows = rows
        self.description = [("rate",), ("fetched_at",), ("base_currency",)]
    def execute(self, sql, params=None):
        self.sql, self.params = sql, params
    def fetchone(self):
        return self._rows[0] if self._rows else None


def test_same_currency_is_rate_one_without_a_lookup():
    r = resolve_fx(_Cur([]), "GBP", "GBP")
    assert r.rate == Decimal("1")
    assert FX_UNAVAILABLE not in r.reason_codes


def test_resolved_rate_carries_its_date_and_source():
    r = resolve_fx(_Cur([(Decimal("1.2734"), _T, "USD")]), "USD", "GBP")
    assert r.rate == Decimal("1.2734")
    assert r.rate_date == _T
    assert r.source and "bp_fx_rates" in r.source


def test_missing_pair_fails_closed():
    r = resolve_fx(_Cur([]), "XYZ", "GBP")
    assert r.rate is None and r.rate_date is None
    assert FX_UNAVAILABLE in r.reason_codes


def test_rate_is_decimal_not_float():
    """Float FX rates reintroduce the rounding drift the benchmark engine went
    to some trouble to eliminate."""
    r = resolve_fx(_Cur([(Decimal("1.2734"), _T, "USD")]), "USD", "GBP")
    assert isinstance(r.rate, Decimal)


def test_unknown_currency_codes_do_not_raise():
    assert resolve_fx(_Cur([]), "", "GBP").rate is None
    assert resolve_fx(_Cur([]), None, "GBP").rate is None  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify it fails.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/services/facts/fx.py`.**

Query `proc.bp_fx_rates` for the pair, preferring a direct `(base_currency, currency)` row and falling back to a cross-rate through the table's base currency if both legs exist. Return `Decimal`, never `float`. Set `source` to a string naming the table and the row's `fetched_at`, e.g. `"bp_fx_rates@2026-07-16T15:35:57Z"`. Put the F4 limitation in the module docstring: `rate_date` is the snapshot timestamp, not the transaction date, and a dated rate corpus is separate work.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Live sanity check** (read-only, does not gate the tests):

```bash
set -a && . ./.env && set +a && ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src.services.db import get_conn
from src.services.facts.fx import resolve_fx
with get_conn() as c:
    cur=c.cursor()
    for pair in [('USD','GBP'),('EUR','GBP'),('GBP','GBP'),('XYZ','GBP')]:
        print(pair, resolve_fx(cur,*pair))
"
```

- [ ] **Step 6: Commit**

```bash
git add src/services/facts/fx.py tests/services/facts/test_fx.py
git commit -m "feat(facts): FX resolution that stamps rate, date and source onto the fact"
```

---

## Task 3: `CommercialFact` and `FactProvenance` models

**Files:**
- Create: `src/services/facts/concept_codes.py`, `src/services/facts/models.py`
- Create: `tests/services/facts/test_models.py`

**Interfaces:**
- Produces: `FactProvenance`, `CommercialFact`, enums `ValueBasis`, `ValidationState`, and `concept_codes.CONCEPT_CODES: frozenset[str]` / `concept_codes.is_known(code) -> bool`. `CommercialFact.provenance: list[FactProvenance]` with a validator rejecting an empty list.

- [ ] **Step 1: Write the failing test**

```python
"""The load-bearing guarantee of this phase: a CommercialFact cannot exist
without provenance. Enforced in the type, not by convention — so every number
downstream provably originated in a document span."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.models import (  # noqa: E402
    CommercialFact,
    FactProvenance,
    ValidationState,
    ValueBasis,
)


def _prov(**over):
    base = dict(document_id="INV000109-1", doc_type="invoice",
                extraction_id="prov-1", field_path="line_items[0].unit_price",
                page=1, locator="bbox:10,20,30,40", verbatim_snippet="86.94")
    base.update(over)
    return FactProvenance(**base)


def _fact(**over):
    base = dict(fact_id="f-1", tenant_id="default", fact_type="line_unit_price",
                unit_price=Decimal("86.94"), currency="GBP", quantity=Decimal("2"),
                value_basis=ValueBasis.AS_SUPPLIED,
                validation_state=ValidationState.VALID,
                provenance=[_prov()])
    base.update(over)
    return CommercialFact(**base)


def test_a_fact_with_provenance_constructs():
    f = _fact()
    assert f.provenance and f.provenance[0].document_id == "INV000109-1"


def test_a_fact_with_no_provenance_cannot_be_constructed():
    with pytest.raises(ValidationError) as e:
        _fact(provenance=[])
    assert "provenance" in str(e.value).lower()


def test_a_fact_with_provenance_omitted_cannot_be_constructed():
    with pytest.raises(ValidationError):
        CommercialFact(fact_id="f-2", tenant_id="default", fact_type="x")


def test_provenance_requires_a_document_and_a_locator():
    with pytest.raises(ValidationError):
        _prov(document_id="")
    with pytest.raises(ValidationError):
        _prov(locator="")


def test_money_fields_are_decimal_not_float():
    """float unit prices reintroduce the rounding drift this whole programme
    exists to eliminate."""
    f = _fact()
    assert isinstance(f.unit_price, Decimal)
    assert isinstance(f.quantity, Decimal)


def test_extended_value_is_not_computed_by_the_model():
    """The model is a record, not a calculator. Anything derived is computed by
    a named deterministic engine that can be pointed at, not silently by a
    property nobody reviews."""
    f = _fact()
    assert not hasattr(f, "compute_extended_value")


def test_reason_codes_survive_onto_the_fact():
    f = _fact(uom="30 days from quote date", uom_normalised=None,
              reason_codes=["UOM_UNMAPPED"])
    assert f.uom == "30 days from quote date"
    assert f.uom_normalised is None
    assert "UOM_UNMAPPED" in f.reason_codes


def test_value_basis_is_a_closed_enum():
    assert {b.value for b in ValueBasis} == {
        "as_supplied", "baseline_corrected", "normalised"}
    with pytest.raises(ValidationError):
        _fact(value_basis="whatever")


def test_concept_code_comes_from_the_extraction_schemas_not_an_invented_list():
    """B3 resolution: the extraction schemas ARE the field registry. Every
    concept_code must be a field name that actually exists in one of them —
    otherwise we have quietly created a second, shadow vocabulary, which is the
    failure the brief's 'do not shadow the dictionary' rule exists to prevent."""
    import yaml
    from pathlib import Path

    from src.services.facts.concept_codes import CONCEPT_CODES

    root = Path(__file__).resolve().parents[3] / "extraction_schemas"
    declared = set()
    for p in root.glob("*.yaml"):
        d = yaml.safe_load(p.read_text())
        declared |= {f["name"] for f in (d.get("fields") or [])}
        li = d.get("line_items") or {}
        declared |= {f["name"] for f in (li.get("fields") or [])}

    assert CONCEPT_CODES, "concept vocabulary is empty"
    orphans = CONCEPT_CODES - declared
    assert not orphans, f"concept codes with no schema field behind them: {sorted(orphans)}"


def test_concept_code_is_optional_on_a_fact():
    """A fact whose concept is not yet classified is still a valid fact. The
    code is an index into the registry, not a precondition for existing."""
    assert _fact().concept_code is None


def test_an_unknown_concept_code_is_rejected_rather_than_stored():
    """Silently accepting an unknown code is how a shadow vocabulary starts."""
    with pytest.raises(ValidationError):
        _fact(concept_code="not_a_real_field_name")
```

- [ ] **Step 2: Run to verify it fails.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**

`concept_codes.py` **derives** its vocabulary by reading `extraction_schemas/*.yaml` at import time and collecting every declared field name into `CONCEPT_CODES: frozenset[str]`, with `is_known(code) -> bool`. Do not hand-type a list — a hand-typed list is a second vocabulary that drifts from the schemas the moment either changes, which is exactly the shadowing failure this is meant to avoid. Cache the parse at module level; the loader already reads these files, so cost is negligible. The module docstring should state that the extraction schemas are the field registry, and that `concept_code` is the column a formal dictionary would map into if one is ever adopted.

`models.py` follows `src/services/benchmark/models.py`'s conventions (`from __future__ import annotations`, `BaseModel`, `ConfigDict`, `Field`, `field_validator`, `str`-valued `Enum`s). Field groups per the brief: identity, economics, commercial identity, term, allocation, grouping, integrity, provenance, bitemporal. Every monetary and quantity field is `Decimal`. `concept_code` is `Optional[str]` with a `field_validator` rejecting any value not in `CONCEPT_CODES`. Every field the corpus cannot supply today (`category_l1..l4`, `contract_id`, `term_*`, `escalator_*`, `cost_centre`, `bundle_group_id`, …) is `Optional` with default `None`.

`category_l1..l4` stay nullable in this phase but are **not** blocked: `proc.bp_category` supplies a `L1~L2~L3` hierarchy keyed on `item_description`. Populating them is a join the Fact Assembler could do later; it is deliberately out of scope here so that Task 6 stays a single-responsibility service. Note the split — `bp_sqldb` has 49 rows, `bp_testdb` has 0 — so any future population must fail closed, not default.

The provenance validator must reject both an empty list and an omitted field — use `Field(...)` (required) plus a `field_validator` asserting non-empty.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add src/services/facts/models.py src/services/facts/concept_codes.py \
        tests/services/facts/test_models.py
git commit -m "feat(facts): CommercialFact with provenance mandatory in the type system"
```

---

## Task 4: The `Constraint` model

Sibling of `CommercialFact` per brief §1.4. Not every commercially material fact is a price.

**Files:**
- Modify: `src/services/facts/models.py`
- Modify: `tests/services/facts/test_models.py`

**Interfaces:**
- Produces: `Constraint`, `BoundDirection`, `BoundBasis`, `TestabilityState`.

- [ ] **Step 1: Write the failing test**

```python
def test_constraint_requires_provenance_like_a_fact():
    from src.services.facts.models import Constraint
    with pytest.raises(ValidationError):
        Constraint(constraint_id="c-1", tenant_id="default",
                   constraint_type="usage_limit", bound_value=Decimal("500"),
                   provenance=[])


def test_the_fields_that_cannot_be_read_off_the_page_may_be_null():
    """bound_basis, measurement_period and applies_to_entities are exactly the
    'limit that needs context' case. They must be null-able with
    PENDING_CONTEXT — never filled by the extractor's guess. Resolving them is
    Phase 4's job and the resolution is stored separately."""
    from src.services.facts.models import Constraint, TestabilityState
    c = Constraint(constraint_id="c-1", tenant_id="default",
                   constraint_type="usage_limit", bound_value=Decimal("500"),
                   bound_uom="named_user", provenance=[_prov()])
    assert c.bound_basis is None
    assert c.measurement_period is None
    assert c.applies_to_entities == []
    assert c.testability_state == TestabilityState.PENDING_CONTEXT


def test_bound_basis_is_a_closed_enum_when_supplied():
    from src.services.facts.models import BoundBasis, Constraint
    assert {b.value for b in BoundBasis} == {
        "named", "concurrent", "peak", "average", "cumulative"}
    with pytest.raises(ValidationError):
        Constraint(constraint_id="c-1", tenant_id="default",
                   constraint_type="usage_limit", bound_value=Decimal("500"),
                   bound_basis="guessed", provenance=[_prov()])


def test_constraint_carries_no_interpretation_field():
    """Facts and reasoning are separate objects. A rationale field here would
    be the seam through which an LLM's reading contaminates the fact base."""
    from src.services.facts.models import Constraint
    for banned in ("rationale", "interpretation", "reasoning", "justification",
                   "explanation", "notes"):
        assert banned not in Constraint.model_fields, (
            f"Constraint must not carry a {banned!r} field")
```

- [ ] **Step 2: Run to verify it fails. Step 3: Implement. Step 4: Run to verify it passes.**

`TestabilityState` defaults to `PENDING_CONTEXT` whenever `bound_basis`, `measurement_period` or `applies_to_entities` is unresolved. Scope list fields default to `[]` via `Field(default_factory=list)`.

- [ ] **Step 5: Commit**

```bash
git add src/services/facts/models.py tests/services/facts/test_models.py
git commit -m "feat(facts): Constraint model with context-pending bounds left null"
```

---

## Task 5: The four new tables

**Files:**
- Create: `deploy/sql/2026-08-07_commercial_fact.sql` and `..._rollback.sql`

**Interfaces:**
- Produces: `proc.bp_commercial_fact`, `proc.bp_fact_provenance`, `proc.bp_constraint`, `proc.bp_finding_fact`.

- [ ] **Step 1: Write the forward migration.**

Requirements, all load-bearing:

- Every table carries `tenant_id TEXT NOT NULL DEFAULT 'default'` (B2 decision — new tables only) and the bitemporal trio `valid_from TIMESTAMPTZ NOT NULL DEFAULT now()`, `valid_to TIMESTAMPTZ`, `recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()`.
- Money and quantity columns are `NUMERIC`, never `double precision`.
- `bp_fact_provenance` has `fact_id` FK to `bp_commercial_fact` with `ON DELETE CASCADE`, plus `document_id`, `doc_type`, `extraction_id`, `field_path`, `page`, `locator`, `verbatim_snippet`, `extracted_at`.
- **Enforce the mandatory-provenance rule at the database level, not only in Pydantic.** A deferred constraint trigger is the only way to express "at least one child row" in Postgres; implement it as a `CONSTRAINT TRIGGER … DEFERRABLE INITIALLY DEFERRED` on `bp_commercial_fact` that raises if no `bp_fact_provenance` row exists for the fact at commit. State in a comment why a `CHECK` cannot express this.
- `bp_finding_fact`: `(opportunity_ref_id, fact_id, role)` with a primary key over all three, FK to `bp_commercial_fact`, and an index on `opportunity_ref_id` — the acceptance query in Task 9 drives from it.
- Indexes: `bp_commercial_fact (tenant_id, fact_type)`, `(supplier_id)`, `(contract_id)`, `(document_version)`; `bp_fact_provenance (fact_id)`, `(document_id)`.
- `bp_constraint` mirrors the Pydantic model, with scope lists as `TEXT[]`.

- [ ] **Step 2: Write the rollback.** Drop in FK-safe order: trigger, `bp_finding_fact`, `bp_fact_provenance`, `bp_constraint`, `bp_commercial_fact`. Add the ordering warning Phase 1a's rollbacks carry.

- [ ] **Step 3: Apply to both databases.**

```bash
set -a && . ./.env && set +a
for DB in bp_sqldb bp_testdb; do
  echo "=== $DB ==="
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-08-07_commercial_fact.sql
done
```

- [ ] **Step 4: Prove the deferred trigger actually fires.** In a transaction on `bp_testdb`: insert a fact with no provenance, commit, and assert the commit **fails**; then insert a fact plus a provenance row in one transaction and assert it succeeds. A trigger nobody has seen reject anything is not a constraint. Capture both outputs.

- [ ] **Step 5: Prove the rollback reverses**, on `bp_testdb` only: run rollback, assert all four tables absent, re-apply, assert all four present.

- [ ] **Step 6: Commit.**

---

## Task 6: The Fact Assembler

The heart of the phase. Deterministic service, no LLM.

**Files:**
- Create: `src/services/facts/store.py`, `src/services/facts/assembler.py`
- Create: `tests/services/facts/test_assembler.py`

**Interfaces:**
- Consumes: `normalise_uom`, `resolve_fx`, `CommercialFact`, `FactProvenance`
- Produces: `assemble_line_facts(cur, doc_type: str, doc_pk: str) -> list[CommercialFact]` and `persist_facts(cur, facts) -> int`.

- [ ] **Step 1: Write the failing test.**

Cover, with fake cursors returning shaped rows:
- A line row with matching provenance produces a fact whose `provenance` is non-empty and whose `unit_price`, `quantity`, `currency` come from the row.
- **A line row with no matching provenance produces NO fact** — assert the returned list is empty and a reason is logged. This is F2's fail-closed behaviour and is the test most likely to be quietly weakened; it must assert absence, not a fact with empty provenance.
- An unmappable UoM yields a fact carrying the raw `uom`, `uom_normalised=None` and `UOM_UNMAPPED` in `reason_codes`.
- A non-GBP line yields `fx_rate`, `fx_rate_date`, `fx_rate_source` populated; a currency absent from `bp_fx_rates` yields `FX_UNAVAILABLE` in `reason_codes` and a NULL rate, not a guessed 1.0.
- `field_path` lookup uses the line's index, and **the index convention is asserted explicitly** — provenance writes `line_items[0]` while the `_trgt` line tables use `line_number`/`line_no`. Verify against real data which is 0-based and which is 1-based before writing the mapping; an off-by-one here silently attaches the wrong line's evidence to a price, which is worse than no evidence.

- [ ] **Step 2: Run to verify it fails. Step 3: Implement. Step 4: Run to verify it passes.**

Per F1, the assembler drives from the `_trgt` line row and looks provenance up by `(doc_type, doc_pk, field_path)`. It never enumerates provenance and joins forward.

- [ ] **Step 5: Live smoke run against `bp_sqldb`** (the only database with real provenance, per F2). Assemble facts for a handful of invoices and print how many facts were produced versus lines read, with the skip reasons. Record the ratio — it is the honest measure of what mandatory provenance costs on this corpus, and Task 10 needs it.

- [ ] **Step 6: Commit.**

---

## Task 7: Structured columns on `bp_opportunity`, and back-population

**Files:**
- Create: `deploy/sql/2026-08-07_opportunity_structured_columns.sql` and `..._rollback.sql`
- Create: `scripts/backfill_opportunity_structured.py`, `tests/services/test_backfill_opportunity_structured.py`

- [ ] **Step 1: Migration** adding to `proc.bp_opportunity`: `currency TEXT`, `amount_native NUMERIC`, `unit_price NUMERIC`, `quantity NUMERIC`, `uom TEXT`, `uom_normalised TEXT`, `fx_rate NUMERIC`, `fx_rate_date TIMESTAMPTZ`, `value_basis TEXT`, `reason_codes TEXT[]`, `facts_state TEXT` (`RESOLVED` | `INDETERMINATE`).

- [ ] **Step 2: Write the failing test for the backfill.**

The measured shapes — do not guess them:

| Detector | Rows | Keys present |
|---|---|---|
| Duplicate Invoice Recovery | 300 | `currency`, `amount_gbp`, `amount_native`, `duplicate_of`, `band`, `signals`, `payment_note`, `payment_confirmed`, `relationship_score` |
| Invoice Overbilling | 6 | `deal_id`, `po_total`, `quote_total`, `invoice_total`, `variance_pct`, `auto_detect` |
| Price Benchmark Variance | 2 | `quantity`, `actual_price`, `benchmark_price`, `item_reference`, `item_description`, `variance_pct`, `flow_coverage`, `risk_score_normalised`, `auto_detect` |

So the honest harvest is: `currency` and `amount_native` for 300 rows; `quantity` and `unit_price` (from `actual_price`) for 2 rows; nothing structured for the other 6. **Everything else must be marked `INDETERMINATE`, not inferred.** Assert that explicitly — including that a row with no parseable key gets `facts_state='INDETERMINATE'` and leaves every new column NULL.

- [ ] **Step 3: Implement. Step 4: Verify. Step 5: Run the backfill on both databases and record the counts.**

- [ ] **Step 6: Commit.**

---

## Task 8: Migrate the consumers, with a logged deprecation shim

**Files:**
- Modify: `src/agents/opportunity_miner_agent.py` (the reader groups at `:2007`, `:2183`, `:5191`, `:5361-5402`, `:5649`)
- Modify: `src/services/opportunity_store.py`
- Create: `src/services/facts/deprecation.py`, `tests/services/facts/test_deprecation_shim.py`

Phase 0 measured the surface: **2 producers, 4 reader groups, 1 store, and no API or render path reads `calculation_details` at all.** That is the whole migration.

- [ ] **Step 1: Write the failing test.** `read_calculation_detail(rec, key)` returns the structured column when present, falls back to the JSONB when not, and **logs every fallback hit** with the key and the opportunity id. Assert the log fires — a shim nobody can measure cannot be retired.

- [ ] **Step 2–4: Implement, verify.** Each reader group calls the shim instead of touching `calculation_details` directly. `calculation_details` remains written for one release but is no longer the system of record.

- [ ] **Step 5: Commit.**

---

## Task 9: The acceptance query

The brief's bar: *a query over `Finding` can reconstruct, for any opportunity, the unit price, contract version, term, and source document of every number contributing to it — without parsing free text.*

**Files:**
- Create: `tests/services/facts/test_acceptance_reconstruct.py`

- [ ] **Step 1: Write the test.** For a seeded opportunity linked through `bp_finding_fact`, a single SQL statement joining `bp_opportunity → bp_finding_fact → bp_commercial_fact → bp_fact_provenance` must return unit price, quantity, UoM, currency, contract reference, term and `(document_id, page, locator, verbatim_snippet)` for every contributing fact — with **no `->>`, no `jsonb_extract`, no regex, and no reference to `calculation_details`.** Assert that textually against the query string, the way Phase 1a's coverage harness asserts its SQL reads the provenance table.

- [ ] **Step 2: Run it against `bp_sqldb`** — per F2, `bp_testdb` has almost no provenance and will produce almost no facts. Record both results; the difference is the honest measure.

- [ ] **Step 3: Commit.**

---

## Task 10: Decision record

**Files:** Create `docs/remediation/01b_fact_model_seam.md`.

Must cover: the four new tables and why `bp_finding_fact` is a join table rather than flat columns (F3's measured 1:3 cardinality); the provenance-mandatory guarantee and where it is enforced (Pydantic validator **and** deferred constraint trigger); F1's join-direction finding; **F2 prominently** — mandatory provenance means the seeded corpus yields almost no facts, with the measured ratio from Task 6 Step 5; F4's FX limitation stated as a limitation; F5's UoM refusal set; the honest backfill harvest from Task 7 and how much was marked `INDETERMINATE`; and the open blockers carried forward.

It must also record **the B3 resolution as a decision, with its reasoning** — that no data dictionary is needed because the extraction schemas already are the field registry and `bp_category` already carries a three-level hierarchy; that `concept_code` is derived from the schemas rather than hand-listed, so it cannot drift into a shadow vocabulary; and that a formal dictionary would only be required for interoperability with an external system, which nothing in Phases 1–5 needs. Note the correction to the Phase 0 seam map (`bp_category` is populated on `bp_sqldb`, empty on `bp_testdb`), and state what would change if GPSS were later adopted: `concept_code` is the column it maps into.

Remember `docs/` is git-ignored except `docs/remediation/` — this path is negated, so a plain `git add` works.

---

## Acceptance

- [ ] All new tests pass; `tests/extraction/` still shows only its 6 pre-existing failures.
- [ ] A `CommercialFact` cannot be constructed without provenance — proven by test **and** by the database trigger rejecting a commit.
- [ ] The Task 9 query reconstructs every contributing number for an opportunity with no free-text parsing.
- [ ] Both migrations applied to both databases; both rollbacks executed and re-applied.
- [ ] The `UOM_UNMAPPED` set of 14 values is refused, not coerced.
- [ ] No LLM call exists anywhere in `src/services/facts/`.

**Explicitly NOT in scope:** the reference corpus and benchmark resolution (Phase 2); variance decomposition, baseline integrity and the correlation-adjusted rollup (Phase 3); the Interpretation Plane (Phase 4); report blocks (Phase 5). Also out of scope, each deliberately rather than because it is blocked: re-extracting the corpus; populating `category_l1..l4` from `bp_category` (a join the assembler could do, held back to keep Task 6 single-responsibility); and the per-category UoM basis, which is a Phase 5 rendering concern keyed on the category hierarchy, not a Phase 1b one.
