# Contract structures — deployment and live verification record

Task 11 of `specs/2026-10-02-contract-structures-plan.md`.
Design: `specs/2026-10-02-contract-structures-design.md`.
Written 2026-10-03. Deployment to `bp_sqldb` ran 2026-10-02 11:42–12:31 UTC; the live
verification ran 2026-10-03.

Everything below is a recorded observation. Where something was not proven, §7 says so by name.

---

## 1. What was deployed

Four migrations, applied in dependency order. Applying 4 before 2 would have let `order form`
claim every quote-template workbook on `bp_sqldb`, so the order is part of the record:

| # | Migration | What it does |
|---|---|---|
| 1 | `deploy/sql/2026-10-02_document_type_parent_evidence.sql` | `requires_parent_evidence`, `parent_evidence_phrases` on `proc.bp_document_type` |
| 2 | `deploy/sql/2026-10-02_document_type_order_form_sales_order.sql` | the `order_form` and `sales_order` rows, and the flag/phrases on `order_form` |
| 3 | `deploy/sql/2026-10-02_contract_raw_resolved_type.sql` | `resolved_doc_type`, `resolved_role`, `type_agreement` on `bp_contract_raw` and `bp_contracts`, plus `ix_bp_contracts_resolved_doc_type` |
| 4 | `deploy/sql/2026-10-02_contract_parent_reference_columns.sql` | `framework_ref`, `parent_agreement_ref` on both tables |

Tasks 1, 7, 9 and 10 added no migration.

`bp_sqldb` is reached on the same RDS cluster, user and password as `.env`; only `DB_NAME`
differs. `.env` itself points at `bp_testdb` (see `reference_env_db_is_bp_testdb`), which is why
this task exists: every task before it applied its migration to `bp_testdb` alone.

## 2. bp_sqldb before and after

"It was already there" and "I added it" are different facts, and only the before-state tells
them apart. Captured 2026-10-02 11:42:09 UTC, before anything was applied
(`.superpowers/sdd/2026-10-02-contract-structures-plan/bp_sqldb-before.txt`):

| Fact | Before (11:42) | After (2026-10-03 17:32) |
|---|---|---|
| `proc.bp_document_type` rows / active | 19 / 18 | 21 / 20 |
| `requires_parent_evidence`, `parent_evidence_phrases` columns | absent (0 of 2) | present (2 of 2) |
| `doctype.order_form`, `doctype.sales_order` rows | absent (0) | present (2) |
| `bp_contract_raw` — the five new columns | absent (0 of 5) | present (5 of 5) |
| `bp_contracts` — the five new columns | absent (0 of 5) | present (5 of 5) |
| `ix_bp_contracts_resolved_doc_type` | absent | present |
| `proc.bp_contract_master` | 3,051 | 3,051 |
| `proc.bp_contracts` | 0 | 0 |
| `proc.process_monitor` | 386 | 386 |

Nothing in the corpus moved: the migrations are additive, and the two new vocabulary rows are
the only data they wrote.

**A schema change by another session, so this record is not mistaken for drift.** At ~11:40 on
2026-10-02 a concurrent session applied `deploy/sql/2026-10-02_atb_style_pack.sql` to both
databases, creating `proc.bp_style_pack` and `proc.bp_page_layout`. Those tables are theirs,
not this plan's, and not drift.

## 3. Idempotency, on the database that matters

The whole four-migration loop was run **twice** against `bp_sqldb`. Full output:
`.superpowers/sdd/2026-10-02-contract-structures-plan/task-11-migrate.log`.

Pass 1 — every migration `COMMIT`, `rc=0`. Pass 2 — every migration `COMMIT`, `rc=0`, with
`NOTICE: column … already exists, skipping` for each `ADD COLUMN IF NOT EXISTS`, `NOTICE:
relation "ix_bp_contracts_resolved_doc_type" already exists, skipping`, and the seed migration
reporting `INSERT 0 0 / UPDATE 0 / UPDATE 0 / INSERT 0 0` — it found its rows already correct
and wrote nothing. A re-run is a no-op, proven on `bp_sqldb` rather than argued from the SQL.

## 4. Parity between the two databases

`md5` over every `proc.bp_document_type` row, ordered by `concept_code`, including the two new
columns (2026-10-03):

```
bp_testdb  aee44380d68e4979c70aadf53a36b12f
bp_sqldb   aee44380d68e4979c70aadf53a36b12f
```

Identical. The predecessor held this table at md5 parity and that standard is kept. No row
differed, so nothing had to be reconciled and no human-confirmed row on `bp_sqldb` was
overwritten by a seed.

## 5. The drift test against bp_sqldb

```
DB_NAME=bp_sqldb … PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_concept_table.py \
    tests/services/concepts/test_contract_type_map.py -v
```

**38 passed** (full output: `step4-sqldb-drift.txt`). The seed and the table do not drift on
`bp_sqldb`, and the corpus-coverage figures hold there too: `proc.bp_contract_master` carries
the same 3,051 rows on both databases, so `test_the_corpus_is_the_measured_size` — the test the
brief warned might legitimately fail on its row count — passed unchanged.

## 6. The suite

The brief asked for `pytest tests/ -q` and a comparison against a baseline of "10,553
passed / 1 failed". **That baseline is not reproducible on this machine**, and saying so is
part of the record: the tree carries about 104 failures that have nothing to do with this
work. They are named and attributed below rather than waved at.

**How it was run.** Three parallel shards, not one serial run: this box has 4 cores and ~6GB
free, and a single serial run reached 1% in two minutes (~3h projected). Shards were split by
directory so files sharing live-DB rows stay together — A = `tests/services`, B = `tests/api`
+ the four extraction trees + `structural_extractor`/`migrations`/`sql`/`triage`/`testdata`,
C = the remaining directories plus the 139 root `tests/test_*.py`. Every shard ran with
`PROCWISE_TEST_LIVE_DB=1` against `bp_testdb`, `CUDA_VISIBLE_DEVICES=""` and Ollama pointed at
a dead port, which is this repo's documented isolation for test runs.

`tests/extraction/test_supplier_sweep.py` is **deselected**. `sweep_supplier_duplicates` is an
O(n²) fuzzy comparison over `proc.bp_supplier`, which now holds 5,032 rows (~12.6M
comparisons). Two consecutive full-suite attempts — the predecessor's at 13:29 on 2026-10-02
and mine at 17:25 on 2026-10-03 — stalled on the same test and produced **byte-identical logs
(139,157 bytes)**. It is a pre-existing performance problem in
`extraction_v3/supplier_resolver`, a path this plan never touches, and `pytest-timeout` is not
installed so there is no way to bound it in-run.

**Before the final-review fix pass and after it:**

| | passed | failed | errors |
|---|---|---|---|
| before (17:35) | 15,322 | 114 | 6 |
| after (18:28) | **15,379** | 115 | 6 |

The 57 extra passes are this pass's new tests. The failure count moved by one, and both
candidates were chased down rather than assumed:

- `tests/extraction_v3/test_yaml_schema_loader.py::test_load_all_schemas_uses_single_connection`
  — **not this work.** The test monkeypatches `psycopg2.connect` *globally* (patching
  `mod.psycopg2.connect` patches the module every caller shares) and then counts every
  connection opened anywhere in the process during `load_all_schemas()`, asserting 1. It passes
  when its file runs alone and fails at 3 when the whole `tests/extraction_v3` directory runs,
  so the extra connections are other code's. Measured directly, in one clean process:
  `load_all_schemas()` opens **exactly 1** connection with the current `contract.yaml` and
  **exactly 1** with the pre-fix one. The schema change is not implicated; the test's own
  instrumentation is.
- `tests/testdata/test_coverage.py::test_no_live_table_is_left_unclassified` — **not new.** It
  failed in the pre-fix run too and was dropped from my comparison list for an unrelated
  reason. It reports 49 unclassified `bp_sqldb` tables, among them `bp_account`,
  `bp_account_contact` and nine `bp_catalog_*` tables from the September reseller-catalog work.
  Nothing this plan deployed creates a table.

**The ~104 pre-existing failures, by cause.** Every one of them was re-run in the environment
the product actually runs in (real Ollama, GPU visible): 9 passed there, so those 9 were
artifacts of the test isolation; 104 failed in both. Not one of them is in a file this plan
touched — verified mechanically by intersecting the failure list with this plan's 35 commits'
test files, which returns **nothing**. The identified causes:

| Count | Cause |
|---|---|
| 19 | `spacy.load('en_core_web_sm')` — the model is not installed. Removed from the module on 2026-05-09, five months ago |
| ~25 | Ollama / Qdrant / credentials unavailable, in the root `tests/test_*.py` email, RAG and agent files |
| 12 | `tests/api/test_agent_workflows_*` — another session's **uncommitted** 332-line edit to `src/api/routers/agents.py` |
| 3 | `tests/test_process_monitor_watcher.py` patches `AgentNickOrchestrator`, a symbol deleted from that module on **2026-05-09** |
| 5 | invoice-schema expectations that drifted long ago (`proc.bp_invoice` vs `proc.bp_invoice_stg`; an empty `anchor` on an invoice pattern) |
| 5 | ISO-date parsing in `extraction_v2`/`extraction_v3` |
| 3 | PaddleOCR / Donut model files absent |
| 1 | `tests/services/test_deal_assignment_service.py` — another session's uncommitted file. This is the one failure the brief's own baseline names |
| 1 | `tests/services/test_event_chain.py::test_downstream_chain_throttles_kg_sync` — **diagnosed**: it sets `KG_SYNC_THROTTLE_SECONDS=9999` and relies on `time.monotonic() - 0.0 >= throttle` being true on the first call. This machine had been up 4,077 seconds, so the first call was throttled and the KG sync never ran. The test passes only on a box that has been up longer than the throttle it sets |
| rest | governance/rule seeds, a structural-extractor feature flag, `test_coverage`, supplier-review API |

The three files the brief singled out as running in no CI job all ran, in shard A, and **passed**:
`tests/services/concepts/test_gate_wiring.py`, `tests/services/test_agent_manifest_slices.py`,
`tests/services/extraction/test_type_findings_lifecycle.py`.

The targeted contract suites are green against `bp_testdb` (the database `.env` points at):
**523 passed** across
`tests/services/extraction/`, `tests/services/concepts/`, `tests/services/test_contract_links.py`,
`tests/extraction/test_contract_l1_parity.py`, `test_contract_link_schema.py` and
`test_schema_db_consistency.py`, with the 53-document classification baseline unmoved.

## 7. The live verification

This is the step the work is for, and it is the only place several of these claims could be
tested at all: `proc.bp_contract_raw` held **0 rows on both databases** before today, and
`proc.process_monitor` had **never** recorded a contract-category upload. Every contract fact
in this product rested on fixtures.

Each document below was put through the **real path**: a `proc.process_monitor` row at status
`Completed`, which fires `proc.trg_process_monitor_ready`, which the running server's
`ProcessMonitorWatcher` claims and runs through `dispatch_document` — parse, L1/L2/L3, type
resolution, persistence, promotion. Nothing was called directly except
`propose_parent_links()` and `confirm()`, which nothing in the product calls (§8, item 6).

The server was restarted first, so the process under test was running the committed code:
MainPID 113649, `18:21:32 output_safety: 222 internal routes registered`,
`18:21:32 Application startup complete`. The vocabulary loads lazily and reported
**20 active document types** — 18 before this work, so the two new rows are live in the
process, not merely in the table.

### 7.1 The documents

| Document | Real? | Declared as | Why it is here |
|---|---|---|---|
| `Styled-Pre-filled-Marketing-Agreement_1_complete.pdf` | **REAL** | `contract` | the only real contract PDF that exists locally |
| `framework.pdf` | written for this verification | `contract` | a generic declaration over a page that names its structure (§6) |
| `order_form_with_framework.pdf` | written for this verification | `order form` | the stand-down rule's **positive** half (§4) |
| `order_form_no_parent.pdf` | written for this verification | `order form` | the stand-down rule's **negative** half on a real file |
| `framework2.pdf` | written for this verification | `contract` | a framework with a conventional `Supplier:`/`Buyer:` block |
| `order_form2.pdf` | written for this verification | `order form` | the proposal chain, end to end |

Five of the six are documents I wrote. They are real PDFs through the real pipeline, which no
fixture test exercises, but their wording is mine and they are not corpus documents. Nothing
below should be read as a measurement over a sample.

### 7.2 What the pipeline stored

| Document | `resolved_doc_type` | `resolved_role` | `type_agreement` | `contract_id` | `framework_ref` | promotion |
|---|---|---|---|---|---|---|
| Marketing Agreement (REAL) | `doctype.contract_unspecified` | `role.master` | `agreed` | NULL | NULL | `discrepancy` |
| framework.pdf | `doctype.framework_agreement` | `role.framework` | **`refined`** | `FA-2026-0042` | NULL | `promoted` |
| order_form_with_framework.pdf | `doctype.order_form` | `role.master` | `agreed` | `OF-2026-0117` | **`FA-2026-0042`** | `promoted` |
| order_form_no_parent.pdf | `doctype.order` | `role.transaction` | **`declared_only`** | `OF-2026-0118` | NULL | `discrepancy` |
| framework2.pdf | `doctype.framework_agreement` | `role.framework` | `refined` | `FA-2026-0077` | NULL | `promoted` |
| order_form2.pdf | `doctype.order_form` | `role.master` | `agreed` | `OF-2026-0211` | `FA-2026-0077` | `promoted` |

Read row by row, against what the design promised:

- **§6, a specific structure under a generic declaration is a refinement.** `framework.pdf` was
  uploaded through the Contracts zone as plain `contract`; the page says what it is; the result
  is `refined` and **no `document_type_disagreement` was raised**. Proven on a real file, not a
  constructed string.
- **§4, positive half.** `order_form_with_framework.pdf` names its framework, so `order_form`
  did **not** stand down: it claimed the page and agreed with the uploader. Before today the
  positive half of this rule had never been demonstrated on a file.
- **§4, negative half.** `order_form_no_parent.pdf` names no parent, so `order_form` stood down
  — correctly. What then happened is the finding in §7.4.
- **No fabrication.** The real Marketing Agreement carries no reference number and no
  `framework_ref`, and both columns are NULL. It is held at `promotion_status='discrepancy'`
  with a blocking `missing_required` on `contract_id` rather than promoted with a guess — which
  is the right answer for a document that genuinely does not state one.

### 7.3 The proposal, end to end

On `order_form2.pdf`, which names `FA-2026-0077` and was uploaded after it:

```
propose_parent_links() -> {"proposed": 1, "contested": 0, "no_candidate": 1,
                           "considered": {"children": 4, "with_structure": 2, "with_candidates": 2},
                           "details": [{"contract_id": "OF-2026-0211", "parent": "FA-2026-0077",
                                        "F": 85.847, "routing": "suggested"}]}
```

The row it wrote to `proc.bp_extraction_discrepancy` (id 12232), in full:

```
doc_pk_candidate   OF-2026-0211
expected_value     FA-2026-0077
computed_value     FA-2026-0042
field_name         parent_contract_id
issue_type         contract_parent_proposed
severity           info          blocks_promotion  false        status  open
source_file        home/muthu/Downloads/SpendIQDocs/verification-2026-10-03/order_form2.pdf
notes              this order_form appears to sit under contract FA-2026-0077 (score 85.8,
                   suggested). reference: OK; structure: OK; supplier: OK; term: OK;
                   title: MISSING. Other candidates: FA-2026-0042. Nothing has been linked.
                   Confirm to set parent_contract_id = FA-2026-0077.
```

Then, in order:

| Step | Observed |
|---|---|
| after `propose_parent_links()` | `OF-2026-0211.parent_contract_id` = **NULL** |
| `confirm('OF-2026-0211', 'FA-2026-0077', <source_file>, reviewer='nick')` | returned `True` |
| after `confirm()` | `OF-2026-0211.parent_contract_id` = **`FA-2026-0077`** |
| the finding | `status='resolved'`, `resolved_by='nick'`, `resolved_at=2026-10-03 18:27:18Z` |
| a second `propose_parent_links()` pass | `proposed: 0` — a parented child is not re-proposed |

So §1's success criterion 4 — the maths proposes a parent and a person confirms it — is
demonstrated on live data rather than on fixtures, including that nothing is linked until a
person acts. **One caveat, stated because it matters:** the `supplier: OK` in those notes is
luck. The supplier extractor read the **buyer's** name (`BrightWave Digital Ltd.`) as the
supplier on BOTH documents, so the signal agreed because both were wrong the same way. See
§7.4, finding 3.

### 7.4 What the live run found that no test had

Four faults, all surfaced by putting real files through the real path. Three were fixed in this
pass; the fourth is recorded, not fixed.

**Finding 1 — CRITICAL, fixed (`14a6dae`): one document's promotion overwrote another
contract's row.** `order_form_with_framework.pdf` says "incorporated into and governed by
Framework Agreement No. FA-2026-0042" and prints no other identifier the schema recognised. So:

- `contract_id` read `FA-2026-0042` — its **framework's** id. `contract_id`'s anchors refuse
  `Parent`, `Master`, `Principal`, `amends`, `amendment to`, `supplements` and `varies` — every
  label naming a different contract — but not `Framework`.
- `framework_ref` read **NULL** from the very same sentence: its anchor demanded a colon after
  "Framework Agreement No", which no contract prints, and its prose connectors did not include
  "incorporated into" — a phrase **this plan itself added** to `parent_evidence_phrases` in §4.
  The rule that recognises the sentence and the extractor that reads it disagreed.
- `contract_id` is the upsert key for `proc.bp_contracts` (`ON CONFLICT (contract_id) DO UPDATE
  SET <every other column>`), so promoting the order form **overwrote the framework
  agreement's row**: `resolved_doc_type` on `FA-2026-0042` changed from
  `doctype.framework_agreement` to `doctype.order_form`. Two documents, one row, no error. The
  framework stopped existing as a separate contract.

The fix is in three parts, and one tried-and-reverted part is worth recording: a negative
lookbehind on `contract_id` was tried first and **blinded a framework agreement to its own
number**, because a framework's own first page writes its identifier in exactly the words an
order form writes its pointer. The fact that separates them is the document's own structure,
which is known in `dispatch` and nowhere earlier — so `_drop_self_parent_reference` resolves it
there, and raises a finding that says the identifier belongs to the parent rather than the
generic "could not ground a value". `parent_agreement_ref` carried the identical mandatory-colon
flaw ("Master Agreement No. MSA-4417" is the commonest shape a SOW has) and was fixed with it.
After the fix, re-uploaded: `contract_id=OF-2026-0117`, `framework_ref=FA-2026-0042`, and
`FA-2026-0042` still reads `doctype.framework_agreement`.

**Finding 2 — fixed (`14a6dae`, same commit): the strongest signal was never looked at.** An
order form naming a framework that was sitting in `proc.bp_contracts` under exactly that id got
**no proposal at all**: `candidate_parents` returned `[]` the moment `supplier_id` did not
match, and on that document the supplier extractor had read the words "Framework Agreement No"
as the supplier name. The supplier narrows a search; it is not a precondition for one. The
contracts a document **names** are now candidates in their own right, deduplicated with the
other two sources and still filtered by structure — a reference is evidence, not an override, so
a SOW does not sit under an order form however explicitly it names one.

**Finding 3 — FIXED after this record was first written (see §10). Supplier extraction on
contracts was wrong in two ways.**

| Document | `supplier_id` stored | What the page says |
|---|---|---|
| `framework.pdf`, `framework2.pdf`, `order_form2.pdf` | `BrightWave Digital Ltd.` | that is the **Buyer**; the Supplier is `NexaSpark Marketing Ltd.` |
| `order_form_with_framework.pdf` | `Framework Agreement No` | not a party at all — a fragment of a sentence |

`buyer_org_id` took the same value as `supplier_id` on every document, and the real Marketing
Agreement's candidate list also contained `Services`, `LIABILITY`, `Arbitration`,
`SEVERABILITY` and `Bank Transfer to`. §10 has the root cause and the fix.

**Finding 4 — fixed (`d7c0ad0`): standing down was being reported as a contradiction.** An
order form that names no parent stands down, and the bare `order` alias inside its own heading
then claimed the page: the result asserted the document was a **purchase order** and raised a
`document_type_disagreement` against the exact structure the rule exists to recognise, naming a
type from another pipeline. `routing.pipeline_for_category('order form')` makes
`doctype.order_form` a declarable concept, so a declaration that has stood down could never be
agreed with — every parentless order form a buyer uploaded would have been flagged. A structure
that was never evaluated cannot have been contradicted: the agreement is now `declared_only`,
which raises no review item. The live re-upload reads `type_agreement='declared_only'`, and the
53-document classification baseline did not move.

## 8. What remains unproven

Stated plainly, because a verification record that lists only what passed is a sales
document.

1. **No real order form, framework agreement, call-off or SOW exists in either corpus.**
   `proc.bp_contract_raw` held **0 rows on both databases** before this verification; the only
   real contract document available locally is one Marketing Agreement PDF. Five of the six documents in §7
   are PDFs **I wrote for this verification** — they are real
   files through the real parser, watcher, resolver, promotion and proposal path, which no
   fixture test exercises, but they are not corpus documents and their wording is mine. The
   stand-down rule's positive half is therefore demonstrated, not measured.
2. **`doctype.sales_order` is seeded on a reasoned mapping, not on evidence.** No document in
   either database contains the words "sales order", "sales order acknowledgement" or "order
   acknowledgement". Its precedence over the bare `order` alias is proven by a unit test only.
3. **`contract_hierarchy` has no labelled sample.** It stays in `UNCALIBRATED_PROFILES`, its
   score thresholds are unvalidated, and by design it can never auto-link — every parent it
   finds is a proposal a person must confirm.
4. **`bp_sqldb`'s finding history is meaningful from 2026-10-01 forward only.** Before the
   partial unique index landed (`ix_bp_extraction_discrepancy_open_key`), 65 days of findings
   were silently rejected there. An empty finding history for an older document on that
   database is not evidence that nothing was found.
5. **Multi-word parent-evidence phrases do not match across a line break.** `fold()` collapses
   whitespace on the phrase side only, so an "order of" / "precedence" split across a line
   misses. It fails safe (the
   structure stands down and nothing is mislabelled) but a real order form can be missed.
6. ~~**Nothing in the product calls `src/services/contract_links.propose_parent_links()`.**~~
   **CLOSED 2026-10-04 — see §18.** `extraction.promotion.promote()` now calls it, scoped to
   the contract that just promoted, and a daily `contract-parent-links` sweep backs it up. The
   trap for a future reader stands: the purchase-order sibling
   `link_proposals.propose_parent_links` is wired at `src/api/routers/promotion.py:149`; that
   is a different function on a different table and says nothing about the contract one.

7. **A contract's supplier is read from its party clause, or it is NULL** (§10). Fixed after
   this record was first written. What is still unproven: the clause reader is measured on six
   documents, five of which I wrote, and the two shapes it reads (a labelled block, a role
   definition clause) are the two shapes those six use. A contract that states its parties some
   other way falls to the context layer, which on the two silent documents here answered
   correctly but is not deterministic. `buyer_org_id` is NULL on those two, and
   `supplier_id`/`buyer_org_id` still hold NAMES rather than resolved ids — entity resolution
   to `proc.bp_supplier` is a separate layer and was not touched.
8. **The proposal half is demonstrated on documents I wrote.** §7.3's chain is live and real,
   but both of its documents are mine. No corpus document has ever produced a parent proposal,
   because `proc.bp_contract_master`'s 3,051 rows are almost entirely parent-type structures —
   7 could be children and none of the 7 carries a parent pointer (measured 2026-10-02, §11.1a
   of the design).
9. **Six verification documents now sit in `bp_testdb`** (`proc.bp_contract_raw` rows 371, 372,
   397, 398 and the two from the chain run; `proc.bp_contracts` `FA-2026-0042`, `OF-2026-0117`,
   `FA-2026-0077`, `OF-2026-0211`). They are deliberately stored under ABSOLUTE paths, which
   `tests/services/extraction/test_classification_baseline.py` excludes (`_stored_documents`
   keeps only `source_file LIKE 'documents/%'`), so they cannot move the 53-document baseline.
   They are real rows from a real run, left in place as the evidence for this record. `bp_sqldb`
   has none of them: nothing was uploaded there.

## 9. The findings from the final review

A fresh-context review of all 35 commits (no Critical, five Important, seven Minor) ran
alongside this task. All five Important findings were fixed, each with a test that was red
first:

| # | Finding | Fix |
|---|---|---|
| 1 | a re-read SKIPPED the proposal instead of refreshing it, so the queue kept the first read's parent and `confirm()` would link the superseded contract; the losing row of a shared `contract_id` could never be closed; `SELECT`-then-`INSERT` was a TOCTOU that raised `UniqueViolation` mid-pass | `b8e5eb6` — the `ON CONFLICT DO UPDATE` this table already uses twice; `status` deliberately not in the SET list, so refreshing evidence cannot resurrect a dismissal |
| 2 | standing down reported as a contradiction (§7.4, finding 4) | `d7c0ad0` |
| 3 | a registered contract whose PDF is uploaded was TWO candidates and tied with itself, so one unambiguous parent read as `contested … Other candidates: C02397` | `b8e5eb6` |
| 4 | the design still claimed "the runner is called"; Ruling 57c had never been applied | this commit — §8 now states that nothing imports `contract_links`, and §9 guard 12 break-proofs the runner's body rather than a nonexistent entry point |
| 5 | the Action Centre's generic finding-action path closed a parent proposal and linked nothing; because a resolved row frees the index slot, the next pass re-proposed the same parent — an approve button that never worked | `10ec733` — accepting routes to `contract_links.confirm()`; dismissal keeps the ordinary path, because `contract_links` reads a dismissed row as still occupying the slot |

The seven Minor findings were deferred, not fixed, and are listed in the ledger
(`.superpowers/sdd/2026-10-02-contract-structures-plan/progress.md`). The two worth knowing
here: `no_candidate` conflates "we never looked" with "nothing scored high enough" — it reads
`1` in §7.3 for the Marketing Agreement, which has no supplier and no reference — and the
proposal's notes carry the score and the per-signal detail but not `score_link`'s decision
**band**, which §8 of the design promises.


---

## 10. Supplier extraction, fixed 2026-10-03 (after this record was first written)

§7.4's finding 3 was recorded as an explicit non-fix and then fixed on Nick's instruction. It
is written up here because the root cause is not where anyone would look for it.

**The cause was not the model.** `SpacyNERExtractor.produce_candidates` has party-aware
branches for exactly two field NAMES: `supplier_name` (header position plus a buyer-context
filter) and `buyer_id` (the BILL TO block). Both are shaped for an invoice or a purchase order.
The contract schema names its party fields **`supplier_id`** and **`buyer_org_id`**, so neither
branch matched and both fields fell through to the default path — which emits *every* entity of
the required type for *any* field. Two fields both asking for an ORG therefore received
**identical candidate lists**, in the same order, so whatever was chosen for one was chosen for
the other. Measured on the real Marketing Agreement, the `buyer_org_id` list was:

```
Services · Services · Bank Transfer to · NexaSpark Ltd. Account · the 'Effective Date'
· Services · LIABILITY · Arbitration · SEVERABILITY · BrightWave\nDigital Ltd. · Services
· the "Effective Date · Services
```

and on `order_form2.pdf` both fields got the same three: `BrightWave Digital Ltd.` (the buyer,
first), `NexaSpark Marketing Ltd.` (the actual supplier, second), `Framework Agreement No`.
A contract's first ORG is whichever party its "between A and B" sentence names first, which is
normally the buyer. That is the whole bug.

**Why no test caught it:** `en_core_web_sm` is installed in `.venv` — what the server runs —
and **not** in `venv`, what pytest runs. Under test, `fill_ner_gaps` logs
`[E050] Can't find model` and returns `[]`, so that branch has never executed in CI or locally.
19 of the suite's pre-existing failures are the same missing model (§6).

**The fix is to read the clause, not to guess better.** A contract has no masthead and no BILL
TO block; it has a party clause, and it says in words which party is which.
`src/services/extraction/engineered/contract_parties.py` reads the two shapes these documents
use — a labelled block (`Supplier:` / `Buyer:` / `Vendor:` / `Client:` …, which are the
`canonical_labels` `extraction_schemas/contract.yaml` has always declared and **nothing had
ever read**) and a role definition clause (`X (hereinafter referred to as the "Marketer")`,
`X ("the Supplier")`), mapping the role word to a side through a vocabulary that deliberately
excludes `Company` — it is the buyer in an employment contract and the supplier in a services
one, and guessing which is the failure being fixed.

Three things it will not do: it does not guess (silence yields NULL, for the context layer to
ground or `missing_required` to flag); it never returns the same name for both sides (that is a
read error, not two facts); and its two fields are **barred from the entity sweep even when the
clause says nothing**, so the original bug cannot return on the next silent document.

**One correction the live run forced.** The first version anchored labels to the start of a
line. On a real document that matched nothing: the parser collapses a contract's whole party
block onto **one line** —

```
Framework Agreement No. FA-2026-0077 Buyer: BrightWave Digital Ltd., 123 Innovation Park,
London, UK Supplier: NexaSpark Marketing Ltd., 125 Innovation Park, London, UK Effective
Date: 5 January 2026 End Date: 4 January 2029
```

— so the colon, not the line start, is the label's signature. That change then needed two
guards, both of which are tested: prose about a party (`the Supplier may be asked to provide`)
has no colon and is excluded by that alone; a drafting colon (`If the Supplier: (a) fails to
deliver`) is excluded by the value having to look like a name. A third bug surfaced with them:
`re.IGNORECASE` on the whole pattern made the value's `[A-Z0-9]` match lowercase, so
`the Buyer: may terminate` read `may terminate` as a party — the flag is now scoped to the
label alternation only.

**Live result, all six documents re-uploaded through the real watcher:**

| Document | `supplier_id` | `buyer_org_id` | Source |
|---|---|---|---|
| Marketing Agreement (**REAL**) | `NexaSpark Marketing Ltd.` | `BrightWave Digital Ltd.` | `parties` (role clause) |
| `framework.pdf` | `NexaSpark Marketing Ltd.` | `BrightWave Digital Ltd.` | `parties` (role clause) |
| `framework2.pdf` | `NexaSpark Marketing Ltd.` | `BrightWave Digital Ltd.` | `parties` (labels) |
| `order_form2.pdf` | `NexaSpark Marketing Ltd.` | `BrightWave Digital Ltd.` | `parties` (labels) |
| `order_form_with_framework.pdf` | `NexaSpark Marketing Ltd.` | NULL | context layer — the document names no parties |
| `order_form_no_parent.pdf` | `Helio Print Services Ltd.` | NULL | context layer — "between A and B", no role words |

Before: **six supplier values out of six wrong**, and `supplier_id == buyer_org_id` on all six.
After: **six out of six right** — four read deterministically from the clause, two grounded by
the context layer on documents that state no roles — and the two fields can no longer hold the
same value. 31 tests, every one of them red before its fix.


## 11. The backfill, 2026-10-03

Asked for after §10 landed: correct the contracts already stored.

**There was nothing to correct.** `proc.bp_contract_raw` holds **6 rows on bp_testdb** — the six
documents of this verification, all re-extracted after the fix — and **0 rows on bp_sqldb**. The
product has never ingested a contract other than these six, so no historical row carries the
wrong supplier. `proc.bp_contract_master`, the 3,051-row register, is a **view** whose
`supplier_id` holds proper identifiers (`S3990`, `S4265`, …); it is derived source data, not
extraction output, and nothing here touches it.

The correction was still written, for two reasons: the next corpus of real contracts will need
it, and running it is the cleanest proof that the six stored rows are right.

`scripts/backfill_contract_parties.py` re-reads the party clause from
`parser_snapshot->>'full_text'`, which every raw row already keeps — so it needs **no
re-extraction, no GPU, no model and no watcher**. Its four rules live in
`contract_parties.decide_correction`, each one a test:

1. a human-confirmed value (provenance `hitl`) is never touched;
2. a row with no stored text is left alone — guessing there is worse than skipping;
3. if the document states its parties, they are the answer;
4. if it does not, the stored value is cleared **only** when it came from the entity sweep. The
   context layer reads the whole document and is grounding-checked, so a backfill has no
   standing to overrule it, and an absent provenance is not evidence of the sweep.

**Dry run over the six live rows:** `0 corrected, 0 cleared, 6 unchanged` — four verified
against their own party clause, two left alone as silent documents whose value came from the
context layer rather than the sweep.

**Then it was proven to correct, on live rows rather than fixtures.** Two rows were deliberately
put back into the pre-fix state and the script run with `--apply`:

| Document | Before | After | Rule |
|---|---|---|---|
| `framework2.pdf` | supplier `BrightWave Digital Ltd.` (the buyer), provenance `ner` | supplier **`NexaSpark Marketing Ltd.`**, buyer `BrightWave Digital Ltd.` | 3 — read from the clause |
| `order_form_with_framework.pdf` | supplier `Framework Agreement No`, provenance `ner` | **NULL** both fields | 4 — silent document, sweep value cleared |

Both tiers moved together: `proc.bp_contracts.FA-2026-0077` now reads
`NexaSpark Marketing Ltd.` with `last_modified_by = 'backfill_contract_parties'`, and the raw
rows' provenance was rewritten to `parties` / `parties-cleared` with a `backfilled_at`
timestamp, so no row keeps claiming the sweep's answer. Against `bp_sqldb`:
`proc.bp_contract_raw is empty: nothing to correct.`

**What the backfill left for §12.** `contract_signatory_name` read **`Email Marketing`** on the
real Marketing Agreement — the same default NER path, PERSON instead of ORG, and a single field
so no duplication hid it. It was fixed next, and the backfill now covers it too.
`jurisdiction` reads `United Kingdom` on five of six, which is correct, and is deliberately
still answered by that same path: barring it wholesale would throw away values that are right.


---

## 12. The signatory, fixed 2026-10-03

The last field answered by the broken default path. `contract_signatory_name` held
**`Email Marketing`** — a line from the Marketing Agreement's services list that spaCy tagged as
a PERSON. One field, so nothing as obvious as `supplier_id == buyer_org_id` gave it away.

**The document says it plainly**, and this is the whole fix:

```
SIGNATURE AND DATE
... This agreement is demonstrated by their signatures below:
MARKETER
Name: John Smith      Signature: ____________  Date: June 12, 2025
CLIENT
Name: Sarah Johnson   Signature: ____________  Date: June 12, 2025
```

**A ruling was needed, because a contract has two signatories and `proc.bp_contracts` has one
field.** `contract_signatory_name` holds the **supplier's** signatory: this is a procurement
system, and the question a single slot has to answer is "who bound the counterparty". The
buyer's signatory is *read* — it is what attributes the other one — but **not stored**, because
the schema has nowhere to put it and adding a column is a larger change than this was asked to
be. `read_signatory().buyer_name` exposes it for whoever adds that column. Cost if this ruling
is wrong: one rename and a migration, and the buyer's name is already parsed.

Where a block names only one signatory and does not say which party they signed for, that one is
taken. Where it names several and none can be attributed, **nothing** is stored.

**Three refusals, each a test, each a value the sweep would have taken:** a signature *rule* is
not a name (docling renders the line as escaped underscores); a date is not a name
(`Name: June 12, 2025` is what an unsigned block leaves); a company is not a signatory
(`For and on behalf of NexaSpark Marketing Ltd.`). The role vocabulary is **shared with the
party reader** through `contract_parties.side_for_role`, because a signature block labels its
halves with exactly the words the party clause uses — so the two readers cannot drift apart
about what "Marketer" means.

`contract_signatory_name` joins the fields barred from the entity sweep. `jurisdiction` does
**not**: it comes from the same path and it is correct on five of six documents, so barring the
path wholesale would lose good values. That asymmetry is pinned by two tests.

**Proven three ways on the real document:**

| | `supplier_id` | `buyer_org_id` | `contract_signatory_name` |
|---|---|---|---|
| before any of this work | `BrightWave Digital Ltd.` (the buyer) | `BrightWave Digital Ltd.` | `Email Marketing` |
| the reader, over the stored text | `NexaSpark Marketing Ltd.` | `BrightWave Digital Ltd.` | `John Smith` (+ `Sarah Johnson` as the buyer's) |
| the backfill, applied to the live row | — | — | `Email Marketing` → **`John Smith`**, provenance `parties` |
| a fresh upload through the live watcher | `NexaSpark Marketing Ltd.` | `BrightWave Digital Ltd.` | **`John Smith`**, provenance `parties` |

The five synthetic documents have no signature block, and all five read NULL — which is the
right answer, not a gap.

`scripts/backfill_contract_parties.py` now corrects three fields rather than two, under the same
four rules, and its dry run over the six live rows reads
`0 party corrected, 0 party cleared, 6 party unchanged, 1 signatory changed`.


---

## 13. A column for the buyer's signatory, 2026-10-04

§12 parsed the buyer's signatory and threw it away, because `proc.bp_contracts` had one pair of
signatory columns. Asked for the next morning: give it a home.

**The naming decision, because it is the part a future reader will question.** The existing pair
is **not** renamed:

```
contract_signatory_name / contract_signatory_role   ->  the SUPPLIER's
buyer_signatory_name    / buyer_signatory_role      ->  the BUYER's
```

`contract_signatory_name` is read by the Node gateway, the Obligations screen and every consumer
of `proc.bp_contracts`, so renaming it to `supplier_signatory_name` is a breaking change for a
cosmetic gain — and a rename plus an addition in one migration cannot be rolled back without
deciding what to do with the data in between. The asymmetry is instead documented **on the column
itself**: `COMMENT ON COLUMN proc.bp_contracts.contract_signatory_name` now says it is the
supplier's and why it carries no prefix, so the database can answer the question without anyone
finding the migration. A test asserts that comment contains "SUPPLIER", and another asserts
`supplier_signatory_name` does **not** exist, so a later tidy-up has to read the reasoning first.

**`ner_type_check: "none"` on both new fields, deliberately.** A PERSON-typed field with no
party-aware branch falls straight back into the default entity path — the one that stored
`Email Marketing` as a person. The new columns are filled by the signature-block reader or they
stay NULL, and both are barred from the sweep in `dispatch._contract_party_candidates`. Two tests
pin that.

**No plumbing was needed.** `promotion` derives its column list from
`information_schema.columns` at run time and `dispatch` filters candidates against the schema's
own `db_column` set, so a migration plus a schema field plus a reader candidate is the whole
path from the page to `proc.bp_contracts`. That is why this is four files and not fourteen.

**Deployment.** `deploy/sql/2026-10-04_contract_buyer_signatory.sql` (+ rollback), additive,
`ADD COLUMN IF NOT EXISTS`, no index — nothing looks a contract up by who signed it, and an
unused index on a 0-row table is a liability. Applied **twice to both databases**: every pass
`COMMIT`, `rc=0`, and the second pass of each reports `already exists, skipping` for all four
columns. **The DDL notice did NOT reach the other sessions.** Per the standing rule about
shared databases, a message naming this file and all four objects was sent to the two idle peer
sessions before applying anything. Both were held for their user's approval because those
sessions run in a different permission mode: one was then denied, the other expired unapproved.
Neither peer's Claude saw it. The migration was applied anyway and that is a judgement, not an
oversight: it is `ADD COLUMN IF NOT EXISTS` on two tables, writes no data, creates no index and
has a proven rollback, so the worst case for a peer is two unexpected columns in a schema
inventory — which is exactly what the notice existed to pre-empt, and is why this paragraph
exists instead. A session that diffed `information_schema` on either database between
2026-10-04 06:55 and 07:00 UTC and found `buyer_signatory_name` / `buyer_signatory_role` on
`proc.bp_contract_raw` or `proc.bp_contracts` is looking at this migration, not at drift.

The rollback is proven rather than asserted: `test_the_rollback_removes_exactly_the_two_columns`
applies the migration, drops the columns, checks they are gone AND that
`contract_signatory_name` survived, then puts them back. A rollback nobody has run is a rollback
nobody can rely on.

**Proven on the real document, both paths:**

| | `contract_signatory_name` (supplier) | `buyer_signatory_name` (buyer) |
|---|---|---|
| before any of this | `Email Marketing` | *(no column)* |
| the backfill, applied to the live row | `John Smith` | **`Sarah Johnson`**, provenance `parties` |
| a fresh upload through the live watcher | `John Smith` | **`Sarah Johnson`**, provenance `parties` |

The backfill now corrects five fields under the same four rules, and its dry run showed exactly
the case the new column creates: `signatory: 'John Smith' -> 'John Smith'   buyer's: 'None' ->
'Sarah Johnson'` — the supplier's name was already right, and the buyer's was NULL only because
the column did not exist when that row was extracted. Every contract stored before 2026-10-04
has that same gap, and this is what closes it without re-extracting anything.

**What is still true after this.** A block naming one signatory with no party label fills the
supplier's field and leaves the buyer's NULL — nothing says whose it is, and a column does not
change that. A block naming several with none attributable fills neither.


---

## 14. The real contract on bp_sqldb, 2026-10-04

Everything before this was verified on `bp_testdb`. Asked for: do it on `bp_sqldb`, the
production-lineage database, with the real document.

**The live server was NOT repointed.** It is pointed at `bp_testdb` by `.env` and is shared with
other sessions and the UI. Instead the same `ProcessMonitorWatcher` the server runs was driven
directly — `_claim_record` (the `Completed → Extracting` claim and the content-hash dedup) then
`_process_record` (which calls `dispatch_document`) — in a process whose `DB_NAME` is `bp_sqldb`,
so every read and write landed there. What this skips versus the server is the LISTEN/NOTIFY
trigger and the poll loop; the extraction path itself is the same function the watcher calls.

**Checked before running:** `proc.process_monitor` on `bp_sqldb` had **0** rows in `Completed`
or `Running`, so a watcher instance could not claim anything else by accident. The script
refuses to run if that count is non-zero, or if `DB_NAME` is not `bp_sqldb`.

**Result** (`proc.process_monitor` id 1651, `category='contract'`, claimed to `Extracting`, ended
`Extracted` / `doc_action='needs_review'`):

| column | value | source |
|---|---|---|
| `resolved_doc_type` | `doctype.contract_unspecified` | the page |
| `resolved_role` / `type_agreement` | `role.master` / `agreed` | |
| `supplier_id` | **`NexaSpark Marketing Ltd.`** | `parties` |
| `buyer_org_id` | **`BrightWave Digital Ltd.`** | `parties` |
| `contract_signatory_name` | **`John Smith`** | signature block, supplier side |
| `buyer_signatory_name` | **`Sarah Johnson`** | signature block, buyer side |
| `framework_ref` / `parent_agreement_ref` | NULL | the document names none |
| `promotion_status` | `discrepancy` | |

So the whole of this work holds on `bp_sqldb`: the vocabulary, the structure columns, the party
clause, both signatories and the new `buyer_signatory_name` column. It is the first contract
document that database has ever held (`bp_contract_raw` went 0 → 1).

**It did not promote, and that is the right answer.** Two blocking `missing_required` findings:
`contract_id` and `contract_start_date`. The document carries no contract or agreement number at
all, so `contract_id` is genuinely absent — and the row is held for a person rather than promoted
with a guess. `contract_start_date` was a **real extraction gap**. It was fixed the same day; see §15.

### A bug this upload found, fixed 2026-10-04

Writing a contract that would actually **promote** (the real one cannot — no contract number)
produced a signature block the parser rendered on **one line**:

```
SUPPLIER Name: Priya Raman Title: Managing Director Date: 1 March 2026 CLIENT Name: Tom Okafor Title: Head of Procurement Date: 1 March 2026
```

The reader returned the supplier's signatory and **NULL for the buyer**, on a document naming
both. Two causes, both of which only the real parser output exposes:

* the name label's value was captured as `[^\n]*`, greedy to end of line, so `finditer` consumed
  the second `Name:` inside the first match and never saw it. Each signatory's span is now
  bounded by the **next** label instead;
* `_party_label_before` scanned backwards for capitalised runs, and `SUPPLIER Name` matched as
  one phrase — which is in no role vocabulary, so *neither* side was attributed. It now scans for
  the role **words** themselves, built from `contract_parties.ROLE_WORDS`.

A third fault fell out of the first fix: the job title was searched for across the whole line, so
the supplier's `Title: Managing Director` was read as the buyer's title. The role search is now
bounded by the same span.

The real Marketing Agreement only ever worked because its parser output happens to keep a blank
line after each party label. Three tests pin the one-line shape, and the fix is what makes
`buyer_signatory_role` reachable at all.

### The new column survives promotion — proven, not assumed

Until this upload, `buyer_signatory_name` had only ever been written to `proc.bp_contracts` by
the **backfill**. Promotion derives its column list from `information_schema.columns` at run
time, so it *should* carry a new column without being told — but "should" is not evidence, and
nothing had promoted with a signature block: the real contract cannot promote (no contract
number) and the five synthetic documents have no signatures.

`promoting_signed.pdf` was written to close exactly that gap — a contract with the three
required header fields AND a two-party signature block. Uploaded through the live watcher on
`bp_testdb`, it promoted, and `proc.bp_contracts` reads:

| `contract_id` | `supplier_id` | `buyer_org_id` | supplier signed | title | buyer signed | title |
|---|---|---|---|---|---|---|
| `SA-2026-0310` | NexaSpark Marketing Ltd. | BrightWave Digital Ltd. | **Priya Raman** | Managing Director | **Tom Okafor** | Head of Procurement |

All four party/signatory columns reached the `_trgt` tier through promotion, including both new
ones and both job titles. That is the last unproven link in the path from the page to
`proc.bp_contracts`.


---

## 15. The term, fixed 2026-10-04

The second blocking finding on §14's upload. `contract_start_date` was NULL on a document that
states its start date three times:

```
... is entered into on June 12, 2025 ( ' the Effective Date') by and between ...
... (referred to as the "Effective Date"). It will end on December 12, 2025
Name: John Smith  Signature: ______  Date: June 12, 2025
```

**The same structural hole as `supplier_id`, in a third place.** `contract_start_date` and
`contract_end_date` declare **sixteen `canonical_labels` between them and have no `patterns` at
all**, so nothing deterministic ever read a contract's dates: the only path was the context
layer, which returned nothing. The field carried no provenance entry, which is the tell — no
candidate was ever produced, as against one produced and rejected.

**It was not a date-parsing problem**, and establishing that first mattered: the runtime binder
reads every one of these shapes (`parse_date("June 12, 2025") → 2025-06-12`). The four
`TestIsoDate` failures in §6's list are the two-venv trap — `venv`, which pytest uses, has no
`dateparser` — not a defect in the date code.

`engineered/contract_dates.py` reads a labelled field first (`Effective Date: 5 January 2026`),
then prose (`entered into on …`, `It will end on …`). In both cases the date must be the **first
thing** after the label or connector, inside a 40-character window. That anchoring is what stops
`shall be effective on the date of signing this Agreement` from reaching forward to a later date,
and what stops `Effective Date: 5 January 2026 End Date: 4 January 2029` — one line, as the
parser renders it — from reading the end date as the start.

**Five refusals, each a test:** a label with no date after it (the real document names "Effective
Date" twice and states no date in either place); a bare `Date:` label, which in a signature block
sits beside every field and means the day somebody signed — reading that as the term start is how
a renewal gets the wrong anniversary; a date in prose with no term wording ("The Parties met on 3
February"); a value that is not a calendar date ("31 February", "the first Tuesday after
Michaelmas"); and a term whose start falls after its end, where **both** are dropped, because
that is a misread rather than two facts.

### The two-venv trap caught me, in my own tests

The first version converted dates by calling the pipeline's `parse_date` — `dateparser`
underneath, installed in `.venv` and **not** in `venv`. So the reader produced nothing at all
under pytest while working in production: the exact failure mode that hid the supplier bug for
months, this time in code written to fix it. A module that accepts only four date shapes can
convert those four itself, so it now does, and behaves identically in both environments. One test
asserts the home-grown conversion agrees with `dateparser` wherever `dateparser` exists, and
**skips** where it does not — and that skip is the point: it is why the other conversion tests do
not go through it.

The numeric form is read **day-first** (`12/06/2025 → 2025-06-12`). That is a choice, not a
guess: it matches what `dateparser` already answers for this corpus (GBP, "the laws of England
and Wales"), so the pipeline cannot change its mind about a date depending on which reader saw
it. Nine shapes are pinned by tests, and a month is matched as a **whole word** — prefix matching
read "Februbry 5, 2026" as February and would have read "Octopus 5, 2026" as October.

**Live on bp_sqldb**, the real contract re-run through the watcher:

| | before | after |
|---|---|---|
| `contract_start_date` | NULL | **2025-06-12**, provenance `date` |
| `contract_end_date` | NULL | **2025-12-12**, provenance `date` |
| blocking findings | 2 (`contract_id`, `contract_start_date`) | **1** (`contract_id`) |

The one remaining blocker is correct and will not be "fixed": the document carries no contract or
agreement number anywhere, so `contract_id` is genuinely absent and the row is held for a person
rather than promoted under a number nobody wrote.

The backfill covers the term too, under three rules rather than the parties' four — the sweep
never produced a contract date, so there is no wrong value of its to clear, only an absent one to
fill, and where this reader finds nothing a stored date **stays** (the context layer may have
grounded a shape these four do not cover). Applied on bp_sqldb: `1 term changed`. Its provenance
writer now takes the reader's name, so a backfilled date records `source: date` and not
`source: parties` — a backfill that mislabels which reader answered is an audit trail that lies.


---

## 16. The audit for a fourth group, and the value fix, 2026-10-04

Three readers in, the question was whether the pattern-less hole had any more field groups in it.

**It had one, and it was the money.** Of contract.yaml's 31 fields, **18 are pattern-less**
(labels declared, no `patterns`). Eight are now covered by the three readers. Of the remaining
ten, **nine were NULL on all 7 live contract rows** — only `jurisdiction` was answered, by the
GPE sweep, and correctly. Five of the nine are stated in the documents:

| field | what the documents say | filled before |
|---|---|---|
| `total_contract_value` | "GBP 750,000", "£25,000", "GBP 60,000" — **all 7** state one | 0 of 7 |
| `currency` | the same sentences (GBP / £) | 0 of 7 |
| `payment_terms` | "Payment shall be made within 30 days of receipt of a valid invoice" — 6 of 7 | 0 of 7 |
| `governing_law` | "the laws of England and Wales" — 5 of 7 | 0 of 7 |
| `contract_title` | every document has one | 0 of 7 |

The other four — `spend_category`, `auto_renew_flag`, `renewal_term`, `contract_type` — are NULL
**honestly**: these documents do not state them.

**A measurement that misled, corrected before it was acted on.** The obvious next question is
whether invoice / PO / quote have the same hole, and the obvious query says they do: ~20
pattern-less fields at 0% fill across 38,570 rows. That is **wrong**. Of 38,577 raw rows across
the four tables only **122** came through the renovation pipeline (7 contract, 8 PO, 42 invoice,
65 quote); the other 38,455 predate it and carry **no `parser_snapshot`, so no stored text at
all**. Fill rates over a whole table are dominated by legacy rows and say nothing about today's
pipeline. On the rows that do carry text, where a PO actually stated payment terms (2 documents)
it was read **both times**. `count(parser_snapshot)`, not `count(*)`, is the denominator for any
claim about this pipeline's accuracy.

### `total_contract_value` + `currency`

Fixed first because it is money: a contract whose value is NULL contributes nothing to any spend
or savings figure.

`engineered/contract_value.py` reads a labelled total (`Total Contract Value:`, `Not to Exceed:`
— the `canonical_labels` the schema already declared) or prose in which the word **"total"**
appears. That word is the whole discriminator, because the trap in this field is the *second*
number:

```
The total charges for this Order Form are GBP 48,000, invoiced monthly in arrears at GBP 4,000 per month.
The total cost of the Services will be £25,000.   ... £10,000 at signing, and £15,000 at completion.
   ... if an expense is over £500.
```

A total, a monthly rate, two instalments and a spending threshold. Taking the wrong one is worse
than taking none: it is a plausible figure that silently misreports the contract. So the value is
the **first** money token after a phrase that says "total", and the search stops at the end of
that sentence — `"The total value is stated in Schedule 1. The deposit is GBP 5,000."` yields
nothing.

**A currency marker is required.** "The total charges are 48,000" yields nothing: a bare number
after "total" could be a headcount, and this product has already been burned by a money figure
whose currency nobody stated. Amount and currency come from the same token and are emitted
together or not at all — an amount without its currency is the shape of a figure that later gets
read in the wrong one. Both parsers were checked for the two-venv trap before being relied on,
and a test keeps that true.

Two labelled totals that **disagree** yield neither; the same total stated twice is one fact.

Live, all 7 documents: `12,500 / 750,000 / 48,000 / 750,000 / 60,000 / 25,000 / 60,000`, all GBP
— including both traps read correctly (48,000 not the 4,000 rate; 25,000 not the instalments or
the £500 threshold). On bp_sqldb the real contract now carries `25000.00 GBP`, and the promoted
`SA-2026-0310` carries `60000.00 GBP` in `proc.bp_contracts`.

### A false positive the backfill's own output exposed

`order_form_with_framework.pdf` was given a start date of **2026-01-05** — the date of the
framework it cites ("Framework Agreement No. FA-2026-0042 **dated 5 January 2026**"), not its own
("shall commence on 1 February 2026"). Two faults in the date reader, both fixed:

* a bare `dated` was treated as being about this document, when it attached to **another
  agreement's name**. A cited agreement immediately in front of the connector now refuses the
  match — the date is that document's, not this one's;
* the first prose match won, and the weak connector happened to come first in the text.
  Connectors are now tiered: wording that can only be about this document's own term
  (`shall commence on`, `effective from`, `with effect from`) is tried before `dated` / `made on`
  / `entered into on`.

A start date belonging to a different contract is worse than none — it is plausible, and it dated
that order form a month early. The row was corrected by re-running the backfill, which is what
rule 3 is for: the document's own words outrank whatever an earlier read stored.


---

## 17. Title, governing law and payment terms, fixed 2026-10-04

The last three of the pattern-less group. Each one stated in the documents, each one NULL in all
7 rows. One reader, `engineered/contract_header.py`, and **three traps that were all live in
those seven files** — which is the argument for reading the corpus before writing the patterns
rather than after.

**Trap 1: "governed by" is not a choice of law.** Two of the seven say *"incorporated into and
governed by Framework Agreement No. FA-2026-0042"*. That is an incorporation clause — the same
cited-agreement trap that gave an order form its framework's start date, now in a third field. So
`the laws of …` (or `<Adjective> law`) is mandatory, and `governed by` alone never answers this
field. Both of those documents correctly read **NULL**: they state no governing law at all.

**Trap 2: a cadence and a cure period are not payment terms.** The real Marketing Agreement says
*"will provide an invoice to the Client every 30 days"* (how often it invoices) and *"without
amending it within a period of 10 business days"* (a cure period). Neither is when payment falls
due, and both contain "30 days"-shaped text. A day count only counts when it hangs off payment
wording, so that document reads **NULL** for payment terms while the other six read `30 days`.

**Trap 3: the parser dropped the real document's title.** Its parsed text begins `## PARTIES` —
a section heading, not a name. So a heading must name a contract-family thing to be a title, and
the document's own opening sentence (*"This Marketing Agreement (hereinafter …)"*) is the second
source. It is the only one that works for that file, and it yields **Marketing Agreement**.

**And a fourth, found on a live document after the first version was written:**
`promoting_signed.pdf` took its title from its SIGNATURE BLOCK — `Title: Managing Director Date:
1 March 2026 CLIENT Name: Tom Okafor …` — because `contract.yaml` lists a bare **"Title"** among
`contract_title`'s labels, and in a signature block that is the job title. The bare label is now
dropped (the unambiguous ones — `Contract Title`, `Agreement Name`, `Subject` — are kept), and a
labelled value is bounded by the next field on the line like every other reader's. The document's
own heading already answers what the bare label was for.

**Conventions taken from the data, not invented.** The title is stored in Title Case because ALL
CAPS in a heading is typography and `proc.bp_contract_master` holds Title Case. Payment terms are
stored as the document's phrasing lightly normalised, which is what is already there:
`proc.bp_purchase_order_raw` holds *"Annual in advance, 30 days"* and `proc.bp_invoice_stg` holds
*"30 days — due 30 Jul 2025"*, so `30 days` is the comparable core of both. A **labelled** value
is kept verbatim (`Net 30`), because there the document is filling in a field rather than writing
a sentence.

**All 7 live documents:**

| document | title | governing law | payment terms |
|---|---|---|---|
| Marketing Agreement (**REAL**) | Marketing Agreement | England and Wales | — (states a cadence only) |
| framework.pdf | Framework Agreement | England and Wales | 30 days |
| framework2.pdf | Framework Agreement | England and Wales | 30 days |
| order_form_with_framework.pdf | Order Form | — (cites an agreement, not a law) | 30 days |
| order_form2.pdf | Order Form | — (same) | 30 days |
| order_form_no_parent.pdf | Order Form | England and Wales | 30 days |
| promoting_signed.pdf | Service Agreement | England and Wales | 30 days |

Every dash is a document that does not state the field, not a field that was missed.

**Both tiers, both databases.** `proc.bp_contracts.SA-2026-0310` now reads
`Service Agreement / England and Wales / 30 days / 60000.00 GBP`, and bp_sqldb's real contract
reads `Marketing Agreement / England and Wales / — / 25000.00 GBP`.

### Where the pattern-less audit now stands

| group | fields | read by |
|---|---|---|
| parties | `supplier_id`, `buyer_org_id` | `contract_parties.py` |
| signatories | `contract_signatory_*`, `buyer_signatory_*` | `contract_signatories.py` |
| term | `contract_start_date`, `contract_end_date` | `contract_dates.py` |
| value | `total_contract_value`, `currency` | `contract_value.py` |
| header | `contract_title`, `governing_law`, `payment_terms` | `contract_header.py` |

Thirteen fields across five readers. What remains pattern-less and unread is
`spend_category`, `auto_renew_flag`, `renewal_term` and `contract_type` — and those are NULL
**honestly**: none of these seven documents states them. `jurisdiction` stays with the entity
sweep, which answers it correctly.


## 18. The proposer is wired, 2026-10-04

Unproven item 6 above was the one that mattered: every claim in §7.3 was true and nothing in
the product ever triggered it. A contract's parent was proposed when a person ran a Python
function, and not otherwise.

**What Nick settled.** The question that kept it unwired was never technical — may proposals
appear in a buyer's queue with nobody asking? Answer, 2026-10-04: **yes, on promotion, scoped to
the document that just promoted, with a slow corpus-wide backstop.**

**What was built.**

| piece | where |
|---|---|
| the hook | `extraction/promotion.py::propose_contract_parent`, called from `promote()` |
| the scope | `contract_links.propose_parent_links(contract_id=…)` |
| the backstop | `backend_scheduler`'s `contract-parent-links` job, daily, 30 min after startup |
| the flag | `autonomous_operation.contract_parent_proposals_enabled` (+ `…_sweep_hours`) |
| the migration | `deploy/sql/2026-10-04_contract_parent_proposals.sql`, applied to BOTH databases |
| the guards | `tests/services/test_contract_link_wiring.py`, 15 tests |

**Why `promote()` and not `dispatch`.** `promote()` is the single funnel all three promotion
paths go through — dispatch's inline call, the HITL NOTIFY listener via
`apply_hitl_fixes_and_promote`, and the `promote_pending` catch-up sweep. Field provenance lives
there for exactly this reason. Hooked into `dispatch` instead, a contract promoted by the HITL
listener after a person resolved its discrepancy would never be proposed a parent.

**Live proof, bp_testdb, through the real funnel.** The open proposal for `OF-2026-0211` (written
by hand on 2026-10-03) was deleted, leaving the slot genuinely empty, and `promote(474,
"contract")` was called with nothing touching `contract_links`:

```
AFTER delete : []
INFO src.services.contract_links: contract parent proposals for OF-2026-0211:
     {'proposed': 1, 'contested': 0, 'no_candidate': 0,
      'considered': {'children': 1, 'with_structure': 1, 'with_candidates': 1}}
AFTER promote: [(15140, 'FA-2026-0077', 'open', 'this order_form appears to sit under contract
     FA-2026-0077 (score 85.8, suggested). reference: OK; structure: OK; supplier: OK;
     term: OK; title: MISSING. ...')]
parent_contract_id is still: None
```

`considered.children == 1` is the scope holding: the corpus has five contract documents and the
pass looked at one. `parent_contract_id` still NULL is the rule holding: a proposal links nothing.

**The job schedules off the real policy row**, not a stub — a bare `BackendScheduler` instance
(no threads, so nothing claims the `process_monitor` backlog) registered `contract-parent-links`
with `interval: 1 day, 0:00:00`, first run 30 minutes out, reading `enabled = True` and
`sweep_hours = 24.0` from `AutonomousOperationPolicy` on bp_testdb.

### The guard that passed on the broken state

Every new guard was broken on purpose. One of them **stayed green**, and it is worth recording
why: the first version of "something in the product calls the proposer" grepped all of `src/`
for the name `propose_parent_links`. With the call deleted from `promote()` the words still sat
inside the now-orphaned hook body, so the guard passed on precisely the unwired state it existed
to catch — the same shape of mistake as a profile with no runner.

It was replaced by three link-by-link AST assertions — `promote()` calls the hook, the hook calls
the proposer, the sweep's runner calls the proposer — plus the startup registration. With both
callers broken, ten of the fifteen tests go red. A corpus-wide name scan cannot tell a caller
from a mention.

### What is still unproven after this

1. **bp_sqldb has no contract document to promote** (`proc.bp_contracts` holds 0 rows there), so
   the hook is proven on bp_testdb only. The policy row and every migration ARE on bp_sqldb, so
   the first contract promoted there will be scored.
2. **The running server does not have this code yet.** `procwise.service` must restart before the
   hook or the sweep runs in production, and the restart was deliberately left to Nick: that
   process serves bp_testdb to the UI and to other sessions.
3. **Still no corpus document has ever produced a proposal.** Unproven items 1 and 8 are
   untouched by this: the children are the five verification documents, and the wiring cannot
   manufacture real order forms.
4. **The sweep has never run on a tick.** Its runner is unit-tested and the same function ran by
   hand; what has not been observed is the scheduler firing it 30 minutes after a start.


## 15. Per-role signals, 2026-10-08

Verified on bp_testdb against the branch `contract-link-signals`. Design:
`specs/2026-10-08-contract-link-signals-design.md`.

### What was built (git log 2a3b8f4..HEAD)

```
7c5e834 feat(vocabulary): dpa, side letter, renewal and guaranty are known, proposed, and resolve to nothing
8d0a6c8 test(contracts): matrix pins the counters behind its no-proposal cases; teardown always removes contracts
74be011 test(contracts): every contract child type through the real runner, six situations each
7780cfe feat(contracts): a confirmed parent link records which kind of link it was
c5033ed test(contracts): the live threshold test is live-only
e124491 feat(contracts): the parent-proposal floor and contested gap are governed, not constants
a69c77d feat(contracts): pick the scoring profile by the child's role; carry the corroborating fields; say amend or attach
4b2cc00 feat(contracts): a schedule or SLA is scored for the agreement it attaches to
bfb4a30 feat(contracts): an amendment is scored on what it names, not on its title
7894546 feat(contracts): the hierarchy profile takes the corroborating signals when both sides carry them
92dcebe feat(contracts): seven corroborating link signals - buyer, value, currency, payment terms, law, signatory, cost centre
3a324b2 feat(linking): score a pair on the optional signals both documents can supply
21caaef docs(contracts): close the code fence in the plan's Task 1
```

### Suites (CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1, DB = bp_testdb)

| Run | Result |
|---|---|
| graph_resolution, test_contract_links, test_contract_link_wiring, test_contract_link_matrix, concepts, extraction/test_type_resolver | 435 passed, 1 failed |
| engine golden vectors and every other `score_link` consumer (test_link_proposals, formulas/test_registered_formulas, test_po_revision_deals, test_duplicate_invoice_detector, test_deal_assignment_service, test_deal_clustering_awards, tests/governance) | 231 passed, 2 failed |

(The engine has no file of its own; its vectors live in the graph_resolution and
test_link_proposals tests above. `tests/test_linking_engine.py` does not exist.)

Failures, all PRE-EXISTING and not caused by this branch:
- `tests/services/extraction/test_type_resolver.py::test_a_signal_phrase_does_not_leak_aliases_from_inside_itself`
- `tests/governance/test_governed_limits.py::test_every_governed_limit_is_present_in_the_live_policy_set`
- `tests/governance/test_governed_limits.py::test_the_in_memory_seed_matches_the_live_rows`
  (both: live policy slug `supplier_info_request` is absent from the conftest seed.)

The second run is 231 passed, 2 failed (the two governance tests above); the type-resolver
failure belongs to the first run. No other failure.

### Matrix re-run (scratchpad cl_matrix.py, bp_testdb, rows created and deleted by the script)

| Child type | A declared ref + same supplier | B no reference | C dangling ref | D wrong parent type | E two parents | F ref names parent, supplier differs |
|---|---|---|---|---|---|---|
| Statement of Work | 96.9 auto_link | 75.6 review | 75.6 review | no candidate | 75.6 contested | below threshold |
| Call-Off Contract | 96.9 | 75.6 | 75.6 | no candidate | 75.6 contested | below threshold |
| Order Form | 96.9 | 75.6 | 75.6 | no candidate | 75.6 contested | below threshold |
| Variation | **65.9** review | below threshold | below | - | below | below |
| Addendum | **65.9** review | below | below | - | below | below |
| Change Control Note | **65.9** review | below | below | - | below | below |

Before this branch: Variation and Addendum 75.6 on their own titles' terms, CCN below threshold, with
these generic titles. Now all three score 65.9 when they declare a reference that resolves to a
supplier-matched parent, with no buyer on the row; the design measured 78.4 with a buyer. A
variation or addendum that names nothing is still NOT proposed (the amendment profile does not
read a generic title as evidence), and is counted as `below_threshold`, not `no_candidate`.

### Corpus effect (whole bp_testdb parentless set, nothing left behind)

bp_testdb has only 5 parentless contract documents; 2 are child types, both with candidate parents.
Before = five base signals via `score_link(child, parent, "contract_hierarchy")`; after = the
role's profile with the corroborating signals.

| Child | Parent | Before | After |
|---|---|---|---|
| OF-2026-0117 | FA-2026-0042 | 54.6 weak_relation (below floor, no proposal) | 72.6 review (proposed) |
| OF-2026-0211 | FA-2026-0077 | 85.8 auto_link_with_warning (proposed) | 93.0 auto_link (proposed) |
| OF-2026-0211 | FA-2026-0042 | 23.5 | 45.0 weak_relation (loser) |

Proposals before / after: 1 / 2. Contested before / after: 0 / 0. Highest band reached: `auto_link`
(F 93.05) after, `auto_link_with_warning` before. This is the evidence for spec section 8 risk 3:
corroborators DO lift a pair into a higher band. It is still a proposal a person confirms
(the profiles are in `UNCALIBRATED_PROFILES`, so no graph edge is written at auto_link either).
The real `propose_parent_links()` pass returned proposed 2, contested 0, no_candidate 0,
below_threshold 0, considered children 5 / with_structure 2 / with_candidates 2.
Both proposals attach to the two rows that already existed open (the five verification documents'
rows); the pass created 0 new rows, deleted 0, and refreshed the evidence on those 2 (accepted).

### Still unproven / deployment prerequisites

1. bp_sqldb needs `deploy/sql/2026-10-08_contract_parent_thresholds.sql` BEFORE the code is
   deployed (otherwise proposals stop, by design: a missing governed value raises), and
   `deploy/sql/2026-10-08_contract_link_vocabulary.sql`. Both were applied on bp_testdb only.
2. The running procwise server must be restarted to load the new profiles.
3. Out of scope and unbuilt: wording signals, amendment sequence, the calibration loop, graph edges.
4. No real corpus contract document has ever produced a proposal; the only children on bp_testdb are
   the verification documents and the matrix fixtures, so every figure above is on constructed data.
5. The 65.9 / 75.6 / 96.9 figures are declared weights, not calibrated ones.
