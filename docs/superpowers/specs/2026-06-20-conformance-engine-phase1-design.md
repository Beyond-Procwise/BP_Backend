# Conformance Engine — Phase 1: Detection Foundation (Design)

**Date:** 2026-06-20
**Status:** Approved design direction; Phase 1 of 5 (see roadmap below)
**Author:** Nick + Claude

## Goal

Stand up the foundation of a source-agnostic **conformance / detection engine**:
a **data-driven rule book**, a **findings store**, and a **source adapter seam**,
by renovating the existing opportunity-mining code rather than rewriting it.

After Phase 1: an admin can author/tune what gets detected and at what threshold
**as data** (no code deploy), the same detectors run over any data source, and
every finding lands in one lifecycle-managed store that can back the
Action-Centre UI.

This is **Phase 1 of 5**. Out of scope here (later phases): policy guardrail +
versioning/approval (P2), decision engine + precedence (P3), playbooks (P4),
rule-change-event + external-feed triggers (P5).

## What exists today (the baseline we renovate)

- **11 detectors** in `src/agents/opportunity_miner_agent.py`, registry at
  `:4070-4138`: price_variance, volume_consolidation, contract_expiry,
  supplier_risk, maverick_spend, duplicate_supplier, category_overspend,
  inflation_passthrough, unused_contract_value, supplier_performance,
  esg_opportunity. Each = a Python handler + display name + default conditions.
- Each handler reads its **threshold via** `_get_condition(input_data, "<name>",
  default)` — so thresholds are *already* parameterizable; today they come from
  `input_data` and from `bp_policy.default_conditions` (which are **empty**).
- Data is loaded once from `_trgt` tables into pandas DataFrames via `TABLE_MAP`
  (`:2229-2260`, `_ingest_data`); handlers filter/aggregate locally.
- A `Finding` object is produced (`:4399-4412`) and upserted to
  `proc.bp_opportunity` by `services/opportunity_store.upsert_opportunity`
  (`:23-64`), idempotent on `opportunity_id`, **preserving the `stage`
  lifecycle** (identified → … → realised/closed/rejected).

The structure is sound. What's missing for the target: the **rule registry +
thresholds live in Python**, the store is **opportunity-named only**, and data
loading is **hard-wired to `_trgt`** (no adapter seam).

## Core boundary (the design decision that avoids a DSL)

Split the tunable from the algorithm:

- **Detector primitive** — the Python handler that computes one kind of
  deviation (e.g. the price-variance math). Registered by **slug**. *Stays in
  code.* Arbitrary pandas logic should not be data-driven — that way lies a DSL,
  which is out of scope (YAGNI).
- **Rule (rule-book entry)** — *data*: "run detector `price_variance_check` with
  threshold 5%, scope = PO lines, severity = by_amount, finding_type =
  opportunity, enabled = true, version = 1." Authored/tuned **without code**.

A rule binds to a primitive by slug. The rule book decides *what runs and with
what numbers*; the primitive decides *how to compute it*.

## Components

### 1. Rule book — `proc.bp_rule` (new) + `RuleBook` loader

Table (mirrors the `bp_policy`/`bp_prompt` governance shape; `bp_` prefix per
convention):

```sql
CREATE TABLE IF NOT EXISTS proc.bp_rule (
    rule_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rule_name        TEXT NOT NULL,
    detector_slug    TEXT NOT NULL,          -- binds to a registered primitive
    finding_type     TEXT NOT NULL DEFAULT 'opportunity',  -- opportunity|anomaly|non_conformance
    scope            TEXT,                    -- item type the rule applies to
    conditions       JSONB NOT NULL DEFAULT '{}',  -- thresholds the primitive reads
    severity         TEXT,                    -- info|low|medium|high (advisory in P1)
    rule_status      SMALLINT NOT NULL DEFAULT 1,  -- 1=enabled, 0=disabled
    version          INTEGER NOT NULL DEFAULT 1,   -- audited now; lifecycle in P2
    created_date     TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT NOT NULL DEFAULT 'system',
    last_modified_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by TEXT NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS ix_bp_rule_status   ON proc.bp_rule (rule_status);
CREATE INDEX IF NOT EXISTS ix_bp_rule_detector ON proc.bp_rule (detector_slug);
```

`RuleBook` (new, `src/engines/rule_book.py`) follows the existing
`PolicyEngine`/`PromptEngine` pattern: load enabled rules from `proc.bp_rule`,
cache, expose `active_rules()` / `rules_for(scope)` / `reload()`. Same
connection-factory and test-injection (`rule_rows=...`) shape as those engines.

**Seed** `bp_rule` with the 11 existing detectors and their current default
thresholds, so behaviour is unchanged on day one (see Migration).

### 2. Detector registry (code) — formalize what exists

Extract the implicit registry (`:4070-4138`) into an explicit, importable
`DETECTOR_REGISTRY: dict[slug, DetectorPrimitive]` where each primitive exposes
`run(tables, conditions, ctx) -> list[Finding]`. The existing handlers become
these primitives with a thin, uniform signature. No math changes.

### 3. Findings store — `proc.bp_finding` (generalize `bp_opportunity`)

`bp_opportunity` already has the right bones (detector_type, policy_id,
calculation_details, source_records, **stage lifecycle**, idempotent upsert). We
generalize to a finding superset rather than a rewrite:

```sql
CREATE TABLE IF NOT EXISTS proc.bp_finding (
    finding_id        VARCHAR PRIMARY KEY,        -- = today's deterministic opportunity_id
    finding_ref_id    VARCHAR,
    finding_type      VARCHAR NOT NULL DEFAULT 'opportunity', -- opportunity|anomaly|non_conformance
    rule_id           VARCHAR,                    -- which bp_rule produced it
    rule_version      INTEGER,                    -- snapshot for P5 re-sweep diffing
    detector_slug     VARCHAR,
    severity          VARCHAR,
    supplier_id       VARCHAR,  supplier_name VARCHAR,
    category_id       VARCHAR,  item_id TEXT, item_description TEXT,
    financial_impact_gbp NUMERIC,
    expected          JSONB,                      -- expected / benchmark
    actual            JSONB,                      -- actual / observed
    deviation         JSONB,                      -- magnitude / pct
    calculation_details JSONB,  source_records JSONB,
    stage             VARCHAR NOT NULL DEFAULT 'identified',  -- same lifecycle as bp_opportunity
    deal_id           VARCHAR,
    detected_on       TIMESTAMPTZ,
    stage_updated_at  TIMESTAMPTZ DEFAULT now(),
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT bp_finding_stage_check CHECK (
        stage IN ('identified','negotiation','agreed','realised','closed','rejected'))
);
```

**Dedup / idempotency:** `finding_id` must be a **deterministic** key derived
from detector/source/supplier/item so re-runs **upsert, not duplicate**, and the
stage-preserving upsert logic is carried over verbatim. ⚠️ Build-time check: the
extraction report was inconsistent about whether today's `opportunity_id` is a
per-run sequential integer (`_next_opportunity_id()`) or the deterministic
hash (`opportunity_ref_id`, `:4385-4389`). **First step of the build verifies
this**; if the PK is currently non-deterministic, Phase 1 switches the dedup key
to the deterministic value (using `opportunity_ref_id`'s formula) — otherwise
re-sweeps would duplicate.

**Non-breaking for the existing Opportunities dashboard:** the dashboard reads
`bp_opportunity`. Phase 1 keeps that working by making `bp_opportunity` a
**view** over `bp_finding WHERE finding_type='opportunity'`, projecting the
original columns. (If the dashboard performs writes/stage updates, those are
redirected to `bp_finding`; verified during build.) No dashboard code change.

### 4. Source adapter seam

Today `_ingest_data` hard-loads `_trgt`. Introduce:

```python
class ItemSource(Protocol):
    def load(self) -> dict[str, "DataFrame"]: ...   # returns the TABLE_MAP-keyed frames
```

- `StoreAdapter` — the **first and only Phase-1 impl**: exactly today's
  `_ingest_data` over `_trgt` (the live data). Behaviour-preserving.
- `UploadAdapter` / `FeedAdapter` — **interfaces stubbed only**, implemented in
  later phases. The engine depends on `ItemSource`, not on `_trgt` directly.

### 5. Evaluation core (renovated, not new)

The existing policy-execution loop becomes the core:
1. `tables = source.load()` (StoreAdapter).
2. `rules = rule_book.active_rules()`.
3. For each rule: `primitive = DETECTOR_REGISTRY[rule.detector_slug]`;
   `findings += primitive.run(tables, rule.conditions, ctx)`, tagging each
   finding with `rule_id`, `rule_version`, `finding_type`, `severity`.
4. Upsert findings to `bp_finding`.

Same determinism and explainability as today; the only change is *where the
rule list and thresholds come from* (data, not code) and *where findings land*.

## Migration

1. Create `bp_rule`, `bp_finding`; replace `bp_opportunity` table with a view
   over `bp_finding` (data-preserving: migrate existing rows into `bp_finding`
   first, then swap).
2. Seed `bp_rule` from the 11 detectors with current default thresholds →
   behaviour identical on first run.
3. Point opportunity_miner at `RuleBook` + `DETECTOR_REGISTRY` + `StoreAdapter`
   + `bp_finding`. Delete the in-code registry literal.
4. **Not in Phase 1:** the 3 ranking *scoring configs* in `bp_policy`
   (weights/normalization) — they feed the ranking algorithm, not the sweep;
   left untouched. Genuine hard *policies* are Phase 2.

## Error handling

- A detector that raises is isolated: log loudly, skip that rule, continue the
  sweep (one bad rule never aborts the run). Mirrors today's per-table
  try/except in `_ingest_data`.
- Empty/missing source table → that detector yields no findings (not an error).
- `bp_finding` upsert is transactional per finding; a failed write is logged and
  does not lose other findings.
- No fabrication: a finding is only written when a primitive actually fired on
  real data.

## Testing (no live model needed)

- **RuleBook loader** — load from injected `rule_rows`; enabled/disabled filter;
  reload.
- **Detector registry** — each of the 11 primitives runs on a small fixture
  frame and reproduces today's finding for a known input (regression-locks the
  math during extraction).
- **Engine wiring** — given seeded rules + fixture tables → expected findings
  with correct `finding_type`/`rule_id` tags.
- **Idempotency** — same input twice → one row (upsert), stage preserved.
- **View compatibility** — `bp_opportunity` view returns the original columns
  for `finding_type='opportunity'`.
- **Live proof (acceptance):** run the sweep over live `_trgt` via `StoreAdapter`
  and confirm real rows in `bp_finding`, counts comparable to the current
  opportunity run.

## Roadmap (context only)

P1 detection foundation (this) → P2 policy guardrail + versioning/approval →
P3 decision engine + precedence/escalation → P4 playbooks →
P5 rule-change-event + external-feed triggers.

## Decoupling summary

| Future change | Absorbed by | P1 code change? |
|---|---|---|
| New detector | `DETECTOR_REGISTRY` + a `bp_rule` row | new primitive only |
| Tune a threshold | edit `bp_rule.conditions` | none (no deploy) |
| New data source | new `ItemSource` adapter | none to engine/rules |
| New finding type | `finding_type` value + rule | none to schema |
