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
6. **Nothing in the product calls `src/services/contract_links.propose_parent_links()`.**
   `grep -rn contract_links src/ scripts/` returns the module itself and nothing else: no
   scheduler, watcher or endpoint imports it. §7.3's chain comes from an explicit call made by
   this verification. Note the trap for a future reader — the
   purchase-order sibling `link_proposals.propose_parent_links` IS wired, at
   `src/api/routers/promotion.py:149`; that is a different function on a different table and
   its being live says nothing about the contract one. Until something calls the contract
   runner, a contract's parent is proposed only when a person runs it by hand.

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
