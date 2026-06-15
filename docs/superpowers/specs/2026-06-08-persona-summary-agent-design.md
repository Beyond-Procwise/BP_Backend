# Design: Persona Summary Agent + `bp_summary`

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Author:** muthu

## Problem

The backend can summarize a single deal through `src/services/deal_summary.py`
(`GET /deals/{deal_id}/summary`), but only with one hardcoded "procurement
analyst" persona, no caching, and no history. The user wants:

- A summary generated for a **caller-supplied persona** (analysis, negotiation,
  compliance, …).
- Coverage of either a single deal **or** the whole portfolio ("complete data in
  the target tables").
- A `bp_summary` table that stores **precomputed** current summaries (for speed)
  and **historical** summaries, keyed by a unique `summary_id`.
- On-demand refresh: hitting the endpoint regenerates and replaces the current
  summary; a past summary can be regenerated over the data state at a selected
  date/time.

## Goals

1. `proc.bp_summary` table (bp_ convention), `summary_id` UUID PK, storing each
   generated summary plus the exact data snapshot it was built from.
2. A `summary_agent` service that resolves a persona, gathers deal/portfolio data,
   generates a grounded summary via the cloud LLM, and persists/caches it.
3. Personas defined as governed rows in `bp_prompt`
   (`prompt_type='summary_persona'`), with a raw-string fallback for ad-hoc
   personas.
4. Endpoints: `POST /summary` (refresh + store), `GET /summary` (fast cached),
   `GET /summary/history`, `GET /summary/{summary_id}`, `POST /summary/precompute`.
5. A scheduled daily precompute that warms the cache and densifies `as_of`
   history.

## Non-Goals

- No full temporal versioning of the `bp_*_trgt` tables (owned by external SQL
  ingestion; out of this codebase's scope). `as_of` works off snapshots captured
  at generation time, not arbitrary point-in-time table reconstruction.
- No registered `BaseAgent` / `agent_definitions` entry — this is a service, the
  endpoint is its interface (matches the `deal_summary` precedent).
- The existing `GET /deals/{deal_id}/summary` stays untouched (backward compat).
- No history-retention pruning in core scope (noted as future).

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | Both — `deal_id` present → per-deal; omitted → portfolio |
| Persona source | `bp_prompt` rows (`prompt_type='summary_persona'`); raw-string fallback |
| Persona input | Sent via the endpoint as a persona key |
| Refresh model | `GET` serves cache; `POST` regenerates + replaces current |
| `summary_id` | Opaque UUID |
| History / `as_of` | Snapshot-at-generation; regenerate from nearest stored snapshot ≤ `as_of` |
| Precompute | In scope — scheduled daily + manual trigger endpoint |
| Component shape | Service module (not a registered BaseAgent) |
| Precompute cadence | Daily (24h), configurable |

## Schema — `proc.bp_summary`

| column | type | notes |
|---|---|---|
| `summary_id` | `UUID PRIMARY KEY DEFAULT gen_random_uuid()` | opaque id (`gen_random_uuid()` verified available) |
| `persona` | `TEXT NOT NULL` | key sent by caller (e.g. `compliance`) |
| `persona_source` | `TEXT NOT NULL` | `bp_prompt` or `raw` |
| `scope` | `TEXT NOT NULL` | `deal` or `portfolio` |
| `deal_id` | `VARCHAR(25)` | NULL for portfolio |
| `summary` | `TEXT NOT NULL` | generated text |
| `data_snapshot` | `JSONB NOT NULL` | exact facts the summary was built from (enables `as_of` regeneration) |
| `sources` | `JSONB` | counts (invoices/pos/quotes/actions/discrepancies) |
| `model` | `TEXT` | cloud model used |
| `is_current` | `BOOLEAN NOT NULL DEFAULT true` | latest per (persona, scope, deal_id) |
| `generated_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | = content snapshot time |
| `created_by` | `TEXT NOT NULL DEFAULT 'system'` | |

Indexes:
- `ix_bp_summary_current` — partial `WHERE is_current` on `(persona, deal_id)` (fast latest lookup)
- `ix_bp_summary_lookup` — on `(persona, deal_id, generated_at DESC)` (history / `as_of`)
- `ix_bp_summary_deal` — on `(deal_id)`

"Current" uniqueness is maintained by the writer: before insert, set
`is_current=false` for the matching `(persona, scope, deal_id)` group (deal_id
compared with `IS NOT DISTINCT FROM` to handle NULL portfolio rows), then insert
the new row with `is_current=true`.

## Migration + persona seed

New file `deploy/sql/2026-06-08_create_bp_summary.sql`, idempotent:
- `CREATE TABLE IF NOT EXISTS proc.bp_summary (...)` + the three indexes (`IF NOT EXISTS`).
- Seed three persona rows into `proc.bp_prompt` (guarded by `WHERE NOT EXISTS` on
  `prompt_name`, same idiom as the governance migration), `prompt_type='summary_persona'`,
  `prompt_linked_agents='summary_agent'`:
  - **analysis** — "You are a procurement data analyst. Emphasize spend totals,
    price and volume trends, supplier concentration, and quantitative anomalies."
  - **negotiation** — "You are a procurement negotiation strategist. Emphasize
    leverage points, price gaps between quotes/POs/invoices, contract and renewal
    timing, and concession opportunities."
  - **compliance** — "You are a procurement compliance auditor. Emphasize policy
    adherence, discrepancies, missing approvals, tax/currency correctness, and
    audit flags."

Applied to `bp_sqldb` with the same psql invocation pattern as prior migrations.

## Component: `src/services/summary_agent.py`

Mirrors `deal_summary.py` (direct SQL, cloud LLM via `ollama_cloud_generate`,
self-contained). Reuses `gather_deal_context` from `deal_summary.py` for per-deal
scope (DRY).

### Persona resolution
`resolve_persona(persona: str, conn) -> tuple[str, str]` returns
`(framing_text, persona_source)`:
- Query `SELECT prompts_desc FROM proc.bp_prompt WHERE prompt_type='summary_persona'
  AND prompt_name = %s AND COALESCE(prompts_status,1)=1 LIMIT 1`.
- Hit → extract `prompts_desc.prompt_template` → `(template, "bp_prompt")`.
- Miss → `(persona, "raw")` (the persona string is used directly as framing).

### Portfolio gather
`gather_portfolio_context(conn) -> Optional[dict]` aggregates across the target
tables (raw rows are summarized, not dumped, to fit the prompt context):
- Per doc type (`bp_invoice_trgt`, `bp_purchase_order_trgt`, `bp_quote_trgt`):
  row count and `SUM(converted_amount_usd)` (the currency-normalized column; falls
  back to the per-doc amount column when USD is null).
- Top N suppliers by total `converted_amount_usd` (joined to `bp_supplier` for
  names).
- Currency mix (count by `currency`).
- Discrepancy count (`bp_extraction_discrepancy`), recent agent-action count
  (`bp_agent_actions`).
- Category aggregation is best-effort: invoices have no category column, so this
  dimension is included only where a category source exists (else omitted).
- Returns `None` if all target tables are empty.

### Prompt assembly
`_build_persona_prompt(framing: str, facts: dict) -> str` = persona framing line +
the grounded base rules reused from `deal_summary._build_prompt` (use only provided
data, never fabricate, leave absent values out) + `facts` JSON.

### Generate + persist
`generate_summary(persona, deal_id=None, as_of=None, conn=None) -> dict`:
1. `framing, persona_source = resolve_persona(persona, conn)`.
2. Determine scope: `deal` if `deal_id` else `portfolio`.
3. **Data**:
   - `as_of` set → `SELECT ... FROM proc.bp_summary WHERE persona-scope match AND
     generated_at <= %s ORDER BY generated_at DESC LIMIT 1`; use that row's
     `data_snapshot` as the facts (regenerate over the past data state). 404 →
     `SnapshotNotFound` if none.
   - else → `gather_deal_context(deal_id)` (deal) or `gather_portfolio_context()`
     (portfolio). `None` → return `None` (404 upstream).
4. `prompt = _build_persona_prompt(framing, facts)`.
5. `text = ollama_cloud_generate(prompt, model=_SUMMARY_MODEL, temperature=0.0,
   num_predict=1024, timeout=120, retries=2)`. Empty → `SummarizationError`.
6. **Persist** via `_store_summary(...)`: flip prior `is_current=false` for the
   group, insert new row (`as_of` regenerations are stored with `is_current=false`
   so they don't displace the live current). Returns the new row as a dict
   including `summary_id`.

### Precompute
`precompute_summaries(personas=None, deal_ids=None, conn=None) -> dict`:
- `personas` default = all `prompt_name` where `prompt_type='summary_persona'` in
  `bp_prompt`.
- Scopes = portfolio + (`deal_ids` or `SELECT DISTINCT deal_id` UNION across the
  three `_trgt` tables).
- For each `persona × scope`: call `generate_summary` (no `as_of`); on per-item
  exception, log and continue. Logs the planned call count up front. Returns
  `{generated, failed, personas, scopes}`.

## Endpoints — `src/api/routers/summary.py` (`prefix="/summary"`)

- **`POST /summary`** body `{persona: str, deal_id?: str, as_of?: ISO-8601}` →
  `generate_summary(...)`; returns `{summary_id, persona, persona_source, scope,
  deal_id, summary, sources, generated_at}`. 404 if no data / no snapshot ≤ `as_of`;
  502 on empty LLM.
- **`GET /summary`** query `persona` (required), `deal_id?` → latest `is_current`
  row for that group; 404 if none cached (message hints to POST).
- **`GET /summary/history`** query `persona` (required), `deal_id?` → list of
  `{summary_id, generated_at, scope}` ordered `generated_at DESC` (+ a short
  snippet of `summary`).
- **`GET /summary/{summary_id}`** → the full stored row; 404 if unknown.
- **`POST /summary/precompute`** body `{personas?: [str], deal_ids?: [str]}` →
  `precompute_summaries(...)`; returns the counts.

Router registered in `api/main.py` alongside the other routers.

## Scheduled precompute — `src/services/backend_scheduler.py`

- Add `_register_summary_precompute_job()` called from `_register_default_jobs()`,
  gated by `getattr(settings, "enable_summary_precompute", True)`.
- `register_job("summary-precompute", self._run_summary_precompute,
  interval=timedelta(hours=getattr(settings, "summary_precompute_interval_hours", 24)))`.
  Runs in the existing background scheduler thread (does not block startup).
- `_run_summary_precompute()` calls `precompute_summaries()` and logs the returned
  counts; any exception is caught and logged (never kills the scheduler).
- New settings (in `config/settings.py`): `enable_summary_precompute: bool = True`,
  `summary_precompute_interval_hours: int = 24`.

## Data flow

```
POST /summary {persona, deal_id?, as_of?}
  -> resolve_persona (bp_prompt | raw)
  -> facts = as_of? nearest snapshot.data_snapshot
            : deal_id? gather_deal_context : gather_portfolio_context
  -> prompt = persona framing + grounded rules + facts JSON
  -> ollama_cloud_generate (cloud; local GPU untouched)
  -> _store_summary (flip is_current, insert new row)
  -> {summary_id, ...}

GET /summary?persona&deal_id   -> latest is_current row (no LLM call)

scheduler (daily) -> precompute_summaries -> generate_summary x (personas x scopes)
```

## Error handling

| Condition | Result |
|---|---|
| Persona not in `bp_prompt` | raw fallback (`persona_source='raw'`), not an error |
| Unknown deal / empty portfolio | 404 |
| `as_of` with no snapshot ≤ datetime | 404 (`SnapshotNotFound`) |
| Empty LLM output | 502 (`SummarizationError`) |
| Cloud LLM transient failure | retried (`retries=2`) then surfaced |
| Precompute item failure | logged, skipped; run continues |

## Testing

- **Unit** (mock `ollama_cloud_generate` and inject a fake DB connection, like
  `test_deal_summary` / `test_prompt_engine` patterns):
  - `resolve_persona`: bp_prompt hit returns template + `bp_prompt`; miss returns
    raw + `raw`.
  - `_build_persona_prompt`: persona framing present; grounded no-fabrication rules
    present; facts embedded.
  - `_store_summary`: inserts a row, flips the previous group row's `is_current`,
    new row `is_current=true`, returns a `summary_id`.
  - `as_of` selection: picks the nearest snapshot with `generated_at <= as_of`;
    raises `SnapshotNotFound` when none.
  - `gather_portfolio_context`: aggregates counts + USD sums over fixture rows;
    returns `None` when empty.
  - `precompute_summaries`: iterates personas × scopes, continues past a failing
    item, returns counts.
- **Integration** (live, after migration):
  - `POST /summary {persona:'compliance', deal_id:<real>}` → 200 with `summary_id`;
    row persisted with `is_current=true`.
  - `GET /summary?persona=compliance&deal_id=<real>` → returns that cached row, no
    new row created.
  - `GET /summary/history?...` lists ≥1 entry; `GET /summary/{id}` returns it.
  - `POST /summary {persona:'analysis'}` (portfolio) → produces a summary.
  - `POST /summary/precompute {personas:['analysis']}` → counts > 0.

## Rollout / Blast Radius

| Change | Files |
|---|---|
| `bp_summary` table + persona seed | `deploy/sql/2026-06-08_create_bp_summary.sql` |
| Summary service | `src/services/summary_agent.py` (new) |
| Reuse deal gather | `src/services/deal_summary.py` (import only; no change) |
| Endpoints | `src/api/routers/summary.py` (new) + register in `api/main.py` |
| Scheduled precompute | `src/services/backend_scheduler.py` (add one job) |
| Settings | `config/settings.py` (two fields) |
| Tests | `tests/test_summary_agent.py` (new), live integration checks |

## Future (out of scope)

- History-retention pruning for `bp_summary` (e.g. keep last N per group).
- Promote the service to a registered `BaseAgent` if agent-graph integration is
  needed.
- Category-dimension portfolio aggregation once a category source is wired.
