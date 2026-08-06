# Phase 1a — Extraction Schema Extension: Decision Record

**Date:** 2026-08-06
**Branch:** `Development`
**Answers:** `docs/remediation/00_seam_map.md` (Phase 0), specifically the finding that `contract_id`, `unit_of_measure`, `cost_centre`, `parent_contract_id`, `escalator_pct`, `term_months`, `billing_frequency`, `amendment_ref` and `document_version` were either declared in only one of four schemas or did not exist anywhere in the codebase.
**Plan:** `docs/superpowers/plans/2026-08-06-phase-1a-extraction-schema-extension.md`
**Ledger:** `.superpowers/sdd/2026-08-06-phase-1a-extraction-schema-extension/progress.md`

**In plain English:** the extraction pipeline reads procurement documents — invoices, purchase orders, quotes, contracts — and picks facts out of them into the database. Before this phase, it was not even *looking* for several facts that later analysis needs: which contract governs a purchase, what unit goods are measured in, what a contract's cost centre or escalation terms are. This phase taught the pipeline to look for those facts. It did **not** go back and re-read documents that were already processed — so the numbers below, showing how rarely these facts are captured today, will not move until someone deliberately reprocesses the existing corpus. That is a separate decision, described at the end.

---

## 1. What changed

Two database migrations and four extraction schemas (YAML files that tell the pipeline what to look for and how to recognise it).

### Migrations

| Migration | Adds | Applied to |
|---|---|---|
| `deploy/sql/2026-08-06_transaction_contract_link.sql` | `contract_id TEXT` on `bp_quote_{raw,stg,trgt}` and `bp_invoice_{raw,stg,trgt}` (purchase_order already had the column, unused), plus three indexes | `bp_sqldb`, `bp_testdb` |
| `deploy/sql/2026-08-06_contract_commercial_terms.sql` | `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct`, `amendment_ref`, `document_version` on `bp_contracts` and `bp_contract_raw`, plus an index on the pre-existing `parent_contract_id` | `bp_sqldb`, `bp_testdb` |

Both migrations are additive (`ADD COLUMN IF NOT EXISTS`) and were verified with a matching rollback script. Rollback was proven live, not just written: Task 4 ran a real rollback → verify (0 columns) → re-apply → verify (7 columns) cycle against `bp_testdb`, and confirmed the rollback leaves pre-existing columns (`parent_contract_id`, `bp_purchase_order_trgt.contract_id`) untouched.

### Schema field list, per document type

| Document type | Fields added |
|---|---|
| Invoice | `unit_of_measure` (line item), `contract_id` (header) |
| Purchase order | `unit_of_measure` (line item), `contract_id` (header) |
| Quote | `contract_id` (header) — `unit_of_measure` already existed on quote line items |
| Contract | `contract_id` patterns (field existed, had never had a pattern to recognise it), `parent_contract_id`, `cost_centre_id`, `is_amendment`, `amendment_ref`, `document_version`, `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct` |

Every new field carries `required: false`. **Two corrections to what an earlier draft of this record claimed:**

- `confidence_threshold: 0.75` is carried by the eleven fields written from scratch for this phase (`contract_id` *patterns*, `parent_contract_id`, `cost_centre_id`, `is_amendment`, `amendment_ref`, `document_version`, `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct`, plus `contract_id` on the three transaction schemas). It is **not** universal. `unit_of_measure` on `invoice.yaml` and `purchase_order.yaml` declares **no** `confidence_threshold` (so it takes the loader's default of `0.70`) and declares **`grounded_last_resort: true`, not `false`**. That block was copied verbatim from `quote.yaml`, which is exactly what the plan prescribed — the point of the copy was that the three line-item schemas stay identical. It is left as it is.
- `contract.yaml`'s `contract_id` is a **pre-existing** field, `required: true` with `confidence_threshold: 0.70` and `grounded_last_resort: true`. This phase added only its `patterns:` block; it did not create the field and did not change its judge settings.

See §4.1 for why neither setting is doing the work this record originally implied.

---

## 2. Why migration-before-YAML is a hard ordering constraint

**In plain English:** the pipeline checks its own homework every time it starts up. If a YAML file says "this fact lives in database column X" and column X does not exist yet, the pipeline refuses to start rather than run with a broken assumption. So the database column has to exist *before* the YAML file that names it is deployed — otherwise the very next restart takes extraction down entirely.

**Technically:** `src/services/extraction_v3/yaml_schema/loader.py:110–128` (`_verify_db_consistency_with_conn`) walks every field a schema declares and checks its `db_column` against the live database's `information_schema.columns`; if the column is missing it raises `SchemaDriftError`. `load_all_schemas()` is called from the application lifespan in `src/api/main.py:161–162`, i.e. on every process start. Task 1 built a test (`tests/extraction/test_schema_db_consistency.py`) that proves this guard actually fires — it deliberately declares a schema with a nonexistent column and asserts `SchemaDriftError` is raised, both for a main-table column and for a line-items-table column.

This is why every task in this phase applied its migration and verified the column existed live, on both `bp_sqldb` and `bp_testdb`, before the corresponding YAML commit landed. Each of the six feature commits in this phase respects that order.

---

## 3. Why no Python code changed

**In plain English:** the parts of the code that move a fact from "just read off a document" to "saved in the database" already read their instructions from the YAML files and the database's own column list, rather than having field names hard-coded. So teaching the pipeline about a new fact only required adding it to YAML plus a database column — the code that moves and matches data adapted automatically.

**Technically, two mechanisms:**

- **`promote()`** (`src/services/extraction/promotion.py:565`, specifically the intersection at lines 838–845) copies a `_raw` row into `_stg` by querying the live `_stg` table's column list via `information_schema.columns` (`_stg_columns`, line 108) and intersecting it against whatever keys are present in the extracted data. A new field with a matching database column is picked up automatically; nothing in `promotion.py` needed to change.
- **`table_extractor._header_to_field`** (`src/services/extraction/engineered/table_extractor.py:20`) resolves a spreadsheet or table column header to a field name by matching against each field's `canonical_labels` in the schema — not against a hard-coded list. Adding `unit_of_measure` with the label `"Unit"` to a schema's `line_items.fields` made the header-matcher recognise a "Unit" column with no code change.

No file under `src/` was touched in this phase. Every change is a YAML file, a SQL migration, or a test.

---

## 4. Three deliberate refusals

Each of these was a conscious decision to leave a field empty rather than guess, convert, or interpret. They are refusals, not gaps.

### 4.1 `grounded_last_resort: false` — a declaration of intent, **not** the safety mechanism

Every field written from scratch for this phase sets `grounded_last_resort: false` in its `judge:` block, alongside `confidence_threshold: 0.75`. The intent is to forbid the AI judge — the last-resort layer that runs when the deterministic pattern-matching (regex) layer finds nothing — from inventing a value for these fields.

**Why the intent matters:** a guessed `contract_id` does not fail loudly. It silently attaches an invoice or purchase order to a contract that may not actually govern it. Every later comparison — is this rate consistent with what the contract promised, has this line item breached the contract's cap — would then run against the wrong agreement, and nothing downstream would know to distrust the number. An absent value is visibly absent and can be handled as such. A guessed value looks exactly like a real one. Absent is correct; guessed is not.

#### But read this before relying on it

**Neither `grounded_last_resort` nor `confidence_threshold` does anything for these fields in the live path today.** An earlier draft of this record presented them as *the* safety guarantee. They are not. Verified against the code:

- `src/services/extraction/judge_gate.py:78-84` is the only place in the live pipeline that reads either setting:
  ```python
  for f in registry.schema.fields:
      if not f.required:
          continue
      judge_spec = getattr(f, "judge", None)
      if judge_spec is None or not getattr(judge_spec, "grounded_last_resort", False):
          continue
      threshold = float(getattr(f, "confidence_threshold", 0.0) or 0.0)
  ```
  `required: false` short-circuits before `grounded_last_resort` or `confidence_threshold` is ever consulted. Every field this phase *added* is `required: false`, so for all of them both settings are inert.
- The v3 judge orchestrator (`src/services/extraction_v3/judge/orchestrator.py:87`) gates on `field_spec.required` in the same way.
- `PatternRegistry.threshold()` / `FieldMeta.threshold` (`pattern_registry.py:57, 112`) have **no** consumer anywhere in `src/`.
- `unit_of_measure` is a *line-item* field; `judge_gate` iterates `registry.schema.fields`, which is header fields only. Its `grounded_last_resort: true` is inert for a second, independent reason.

The settings are correct and harmless as declarations of intent — they document what the field is allowed to become, and they take effect the moment a field is promoted to `required: true`. But they are not what is protecting the pipeline right now.

**What actually protects it is two things:** `required: false` (which keeps the AI judge out of these fields entirely), and **the specificity of the regexes themselves.** The second one is the load-bearing part, and it is the part that failed review. Four defects were found after every per-task review had passed, all of them the same shape — a regex validated against the single fixture that motivated it, and wrong on the next document:

| Defect | What the regex did |
|---|---|
| `contract_id` matched inside "Parent Contract Number:" and "Master Agreement No:" | An amendment resolved to its **master's** identifier. `contract_id` is the `ON CONFLICT` key for `bp_contracts` (`promotion.py:56-60, 857-860`), so this would not error — it would silently overwrite the master row's dates, value and escalator terms. |
| The identifier value regex accepted `N/A`, `TBC`, `SEE`, and swallowed trailing punctuation | `'MSA-9921.'` is not `'MSA-9921'`, so the later join to `bp_contracts` breaks with no error. `'N/A'` becomes a primary key colliding with every other document that also said "N/A". |
| `escalator_pct` captured the **cap** rather than the rate | "shall not exceed 4%" recorded a 4% uplift the document never granted. The original test passed only because the fixture stated the rate before the cap. |
| `term_months` captured a **renewal** term | A 60-month contract with 12-month renewals recorded as a 12-month contract. |

All four are fixed (see §10). The lesson is not that the judge settings were wrong — it is that they were never the guarantee, and the thing that *was* the guarantee had not been adversarially tested. Phase 1b must not read `grounded_last_resort: false` as a safety net.

### 4.2 No years-to-months conversion on `term_months`

If a contract states its term as "three (3) years" rather than in months, `term_months` is left empty rather than the pipeline computing 36 for it.

**Why:** converting years to months is arithmetic, and interpretation/arithmetic is deliberately kept out of the extraction layer — extraction's job is to find and transcribe what a document literally states, not to derive new facts from it. A derived value (with its own basis — was it years×12, or explicitly stated) belongs to Phase 1b, where it can carry that basis and its own provenance record, rather than being silently folded into a field that looks like it was read directly off the page. Task 6 built a specific guard test (`test_term_stated_in_years_is_not_silently_converted`) to prove this refusal holds, and confirmed the years-pattern *does* fire (proving the test isn't vacuous) while the months-only value pattern correctly declines to produce a number.

### 4.3 `is_amendment` stores the literal marker word, not a normalised boolean

The field captures the document's own word — e.g. `"AMENDMENT"` — rather than converting it to `true`/`false` or `Yes`/`No`.

**Why:** normalising "AMENDMENT" (or whatever a given document actually says) into a boolean is an interpretive step performed inside the extractor, and the database column backing it is `TEXT`, not a boolean type. Storing the literal word is both truthful to the source document and directly storable without an interpretation layer making a judgment call the extractor isn't positioned to make reliably across arbitrary contract phrasing.

---

## 5. The decimal capture-group rule

**In plain English:** when a pattern is written to pull a percentage or a count of months out of a sentence, the number has to be captured on its own — without the `%` sign or the word "months" attached — or the value silently gets thrown away with no visible error.

**Technically:** the codebase's number parser, `parse_amount`, returns `None` for a string like `'3%'` (it doesn't strip the `%` for you), and the naive fallback `float('3%')` raises. If a regex's capture group 1 included the `%` or the unit word alongside the digits, the resulting hit would fail to bind to a number — recorded internally as a `type_bind_error` — and the field would end up empty with no test going red to catch it, because the pattern *matched*; only the downstream type conversion silently failed.

Every percentage field (`escalator_pct`, `escalator_cap_pct`) and every count field (`term_months`) added in this phase therefore keeps the unit token (`%`, `per cent`, `months`, `mos.`, etc.) strictly **outside** capture group 1. Task 6 verified this directly: `test_decimal_fields_capture_a_bindable_number` round-trips each hit through `parse_amount` and asserts the result is not `None`.

---

## 6. Coverage before, and the honest caveat

**These numbers are measured, run live against the databases at the point this phase's schema work landed (commit `c079f58`) — not estimated.**

### `bp_sqldb` (the larger, real corpus)

- Quote `line_items[].unit_of_measure`: **0.8% — 1 of 125 documents.**
- Invoice and purchase-order `unit_of_measure`: **absent entirely — 0 documents.** (The field did not exist in either schema before this phase; there is nothing to have measured.)
- Invoice `line_items[].unit_price`: 41.8% (87/208). `line_items[].quantity`: 43.8% (91/208).
- Purchase order `unit_price` / `quantity`: 7.4% (6/81) each.
- Quote `unit_price` / `quantity`: 3.2% (4/125) each.
- Contract: **only 4 documents have ever been extracted at all.** Of those: `contract_id` 100%, `contract_signatory_name` 100%, `contract_start_date` 50%. **Zero** provenance rows exist for `cost_centre_id`, `parent_contract_id`, `is_amendment`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct`, `term_months`, `billing_frequency`, `amendment_ref`, `document_version` — because, before this phase, none of those fields existed anywhere for the pipeline to have captured a value into.

### `bp_testdb` (the smaller test corpus)

Invoice 21 documents, purchase order 9, quote 54 — **headers only, no line-item provenance at all.**

### The caveat — read this before drawing any conclusion from the numbers above

**These numbers cannot improve until documents are re-extracted.** This phase changed what the extraction pipeline *looks for*. It did not, and could not, retroactively populate fields for documents that were already processed before the schema changed — the pipeline does not re-read a document it has already filed away. The only way any of the figures above move is a deliberate decision to reprocess the corpus (roughly 88,000 documents) through the pipeline again. That is an operational decision with its own cost and risk, and it is **deliberately out of scope for this phase.**

Put plainly: this phase built the capability to capture these facts on the next document that comes in. It did not, and was never going to, retroactively conjure them for documents already on file. Do not read the low coverage percentages above as a failure of this phase, and do not read a future re-run of `scripts/field_coverage.py` against this same un-reprocessed corpus as evidence that this phase itself lifted coverage — it will not have, because nothing new has been read.

---

## 7. The contract-validation limitation

**In plain English:** the new contract fields were tested against a fake contract document written specifically for the test, not against a real one — because there are no real contracts in the system to test against.

**Technically:** `proc.bp_contracts` has **0 rows** in both `bp_sqldb` and `bp_testdb`, and the repository contains no contract documents of any kind. Every pattern for `contract_id`, `parent_contract_id`, `cost_centre_id`, `is_amendment`, `amendment_ref`, `document_version`, `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis` and `escalator_cap_pct` was validated in `tests/extraction/test_contract_l1_parity.py` against a single synthetic plain-text fixture, `tests/extraction/fixtures/contracts/msa_with_amendment.txt` — text written by the implementer to exercise these patterns, not a document a real supplier or buyer ever produced.

The fixture exercises the identical code path a real document would go through (the same regex engine, the same field-resolution logic), so it proves the patterns are *internally consistent* and behave as designed. **It is not evidence that these patterns survive real contract formatting, real layout variance, or real legal boilerplate phrasing.** This is blocked on open blocker **B1** (no contract corpus) and must be re-run against real documents — specifically, `tests/extraction/test_contract_l1_parity.py` should be re-executed against genuine contract text — once such a corpus exists.

---

## 8. What Phase 1b now depends on

Three things are now reachable by later phases without writing any new free-text parsing:

- **`contract_id`** is declared on all three transaction document types (invoice, purchase order, quote), with a physical database column on every layer (`_raw`, `_stg`, `_trgt`). A transaction can be linked to its governing contract without inventing a new mechanism.
- **`unit_of_measure`** is declared on all three line-item schemas (invoice, purchase order, quote) — quote already had it; invoice and purchase order now do too. A quantity can be interpreted consistently across all three document types.
- **Seven commercial-term columns** exist on the contract record (`bp_contracts` and `bp_contract_raw`): `term_months`, `billing_frequency`, `escalator_pct`, `escalator_basis`, `escalator_cap_pct`, `amendment_ref`, `document_version` — plus, from Task 5, `parent_contract_id`, `cost_centre_id` and `is_amendment`. Phase 1b's commercial-fact and conformance work has somewhere to write these values once real contracts exist to extract them from.

None of this required a new database table, a new promotion path, or a new extraction stage — the existing pipeline's field-driven design absorbed all of it (see §3).

---

## 9. Open blockers carried forward

- **B1 — no contract corpus.** `proc.bp_contracts` has 0 rows in both `bp_sqldb` and `bp_testdb`; the repository contains no contract documents. Every contract pattern in this phase is validated against synthetic text only (§7). This blocks real validation of the contract schema and blocks most of the downstream conformance work (Phase 3.2 and later) that depends on contract data actually existing.
- **B3 — no GPSS dictionary.** "GPSS" (the vocabulary the original brief assumed field names should be drawn from) appears nowhere in this codebase. Phase 1a bound every new field to the existing physical column names already in the database rather than inventing or resolving against a GPSS name. Phase 1b's planned `gpss_code` field needs this resolved before it can be built — there is currently no dictionary to bind it to.
- **B2 — resolved, not applicable here.** `tenant_id` scoping applies to new tables only, and is first applied in Phase 1b. Phase 1a created no new tables (only new columns on existing tables), so B2 did not apply to any of this phase's work.

---

## 10. Whole-branch review — defects found after every task had passed its own review

Per-task review could not see these: each task's regexes had been validated only against the one fixture that motivated them, and a fixture cannot show you the document it was not written for. A single fix wave (commits listed at the bottom) closed all of them. **Every fix was written test-first and observed red before the regex changed.**

| # | Fix |
|---|---|
| 1 | `contract.yaml` `contract_id`'s two anchors now carry stacked negative lookbehinds — `parent`, `master`, `principal`, `amends`, `amendment to`, `supplements`, `varies` — one for every qualifier `parent_contract_id`'s own anchors claim. `parent_contract_id` still reads "Parent Contract:" and "Master Agreement No:". |
| 2 | The identifier value regex on all ten contract-reference patterns (both `contract.yaml` fields, and `contract_id` on the three transaction schemas) is now pinned to the start of the post-anchor window (`\A`), must end on an alphanumeric, and refuses `N/A`, `NA`, `TBC`, `TBD`, `NONE`, `SEE` via a negative lookahead. The label connector is mandatory in all four schemas — with `.` accepted as a connector so "Agreement No. AGR-4471" still reads. `under_agreement`'s connector stays optional (its prose form legitimately has none) but it gets the value-regex fixes. |
| 3 | `escalator_pct`'s connector is mandatory and its value pinned to `\A`. The `increase` alternative is **kept** — "Annual Increase" is one of the field's own canonical labels — because the mandatory connector plus `\A` already refuses "increase shall not exceed 4%", "increase shall be capped at 5%" and "shall increase by CPI, capped at 5%". `escalator_cap_pct` still captures all three caps. |
| 4 | `term_months`'s bare `term` alternative is guarded by `(?<!renewal )(?<!extension )(?<!renewed )`, and its value pinned to `\A`. "Initial Term: 36 months" and "for a period of 24 months" still read; a term stated in years still yields nothing. |
| 5 | `tests/extraction/test_schema_db_consistency.py` now asserts every declared `db_column` exists on `_raw` and `_trgt`, not only on the declared `db_table` — `persistence.write_raw` (`persistence.py:170, 188-196`) INSERTs into `_raw` with no column filter, so a `_stg`-only column would pass schema load and then raise `UndefinedColumn` on the next INSERT. Contract's missing `_stg`/`_trgt` layer is skipped rather than failed. The whole file is parametrised over **both** `bp_sqldb` and `bp_testdb`, selected by overriding `DB_NAME` in the environment — no committed config touched. Proven red on both databases by temporarily declaring a `_stg`-only column. |
| 6 | Both rollback scripts now state, at the top, that the corresponding YAML commit must be reverted **before** the script runs — otherwise `load_all_schemas()` raises `SchemaDriftError` in the API lifespan and the API will not start. Comment only; no SQL changed. |
| 8 | `escalator_cap_pct` no longer carries `total_contract_value`'s "Not to Exceed" label, which would have bound a money value into `numeric(9,4)` and overflowed it if the label-driven extractors were re-enabled. |
| 9 | `test_parent_contract_is_not_confused_with_contract_id` asserted only that the two values *differ* — which is true on any fixture where the failure cannot occur, and is why the defect survived eight reviews. It now asserts both exact values, and 46 adversarial cases were added across `test_contract_l1_parity.py` and `test_contract_link_schema.py`. |
| 10 | `.gitignore`'s line-1 `/docs/` hid 54 tracked files from `git add`, which had already cost a real miss. `docs/remediation/` and `docs/superpowers/` are now negated; the other ignored files under `docs/` stay ignored. |

**What is still not covered:** all of this is still synthetic-fixture validation (§7, blocker B1). The lookbehind guard is fixed-width, so phrasings like "amends **the** Contract No: X" are not blocked by it — they fall to `parent_contract_id`'s own higher-prior claim on the same text, but they are not proven by test. And the value regex now requires the identifier to sit immediately after the label; a document that puts anything between them yields nothing rather than something wrong, which is the intended direction of failure.

## Regression status

Controller-run at commit `d825111` (`tests/extraction/`): **6 failed, 233 passed, 18 skipped.** The 6 failures (`test_dispatch` ×2, `test_l1_parity_invoice` ×2, `test_pattern_yaml_loader` ×2) are pre-existing and unrelated to this phase's changes — verified pre-existing by an isolated `git worktree` A/B comparison at the pre-phase commit in Task 2, and re-confirmed identical by name across every subsequent task in this phase. The pass count rose from 221 (before this phase's tests existed) to 233 as this phase's own tests landed — no regressions were introduced.

## Commits

| Commit | Subject |
|---|---|
| `918c161` | test(extraction): schema/DB drift guard |
| `3417d71` | feat(extraction): unit_of_measure on invoice and PO line items |
| `e50cba9` | feat(extraction): contract reference on quotes, POs and invoices (+ migration) |
| `f67a443` | feat(db): commercial-term columns on the contract record |
| `3eff78a` | feat(extraction): cost centre, parent contract and amendment marker |
| `d825111` | feat(extraction): escalator, term, billing frequency and version |
| `c079f58` | feat(scripts): per-field extraction coverage from the provenance record |

## Explicitly not in scope for this phase

Re-extracting the existing ~88k-document corpus (a separate operational decision); any change to `Finding` or `bp_opportunity`; any `CommercialFact` type; any unit-of-measure normalisation. All of these are Phase 1b.
