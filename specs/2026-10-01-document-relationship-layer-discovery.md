# Discovery Report — Document Relationship Layer (GPSS build spec)

**Status:** Phase 1 deliverable, awaiting review. No build work started, no code or
schema changed.
**Date:** 2026-10-01
**Scope of this document:** spec §2 in full — §2.1 product map, §2.2 component
locator, §2.3 placement decisions, §2.4 risks. Plus the principle conflicts the
spec requires be flagged rather than resolved silently, and answers to those of
the §10 open questions the product already settles.
**Databases inspected:** `bp_testdb` (what `.env` points at — the seeded corpus).
Figures below are measured there on 2026-10-01 unless stated.

---

## 0. Summary in plain English

Five things decide this build, and three of them are not what the spec assumes.

**1. Nothing in the product classifies a document today.** The spec's Phase 3
step 1 assumes an AI reads the page and proposes a type. In the live path the
type is simply *whatever the person said when they uploaded the file*, read from
one column (`proc.process_monitor.category`), and if it is not one of four words
— invoice, purchase_order, quote, contract — the pipeline stops with an error
(`process_monitor_watcher.py:531`). There is classifier code in the repo, but it
sits on the older path that no longer runs. So this is not "extend the
classifier". It is "there is no classifier, and the four-word gate will reject
every new type the spec introduces on the first upload".

**2. The product already has the right reference-table pattern, and it is good.**
`proc.bp_uom_canonical` (the unit vocabulary) does almost exactly what the spec
asks of Document_Index and Synonym_Index together: one row per concept, aliases
in a list on that row, a `status` of active / proposed / rejected where
*proposed never resolves silently*, a `source`, an observation counter, and
`confirmed_by` / `confirmed_at` for the human who promoted it. That is the spec's
learn-and-confirm loop, already built, already live, with 38 rows. Everything new
should copy it rather than invent a shape.

**3. "Rules score, only impossibilities block" is already how the product works.**
It is a boolean called `blocks_promotion`, on two live finding tables with 5,372
and 8,147 rows. And the scoring half of the spec — signals, weights, clusters,
coverage, decision bands — is already built and faithful to a maths spec, in
`linking_engine.py`, with bands at 92 / 80 / 65 / 45. Phase 3's steps 6 to 9 are
substantially existing work, not new work.

**4. There is no sector and no usable region to resolve scope against.** The word
"sector" appears nowhere in the source. `governing_law` is free text with six
spellings across two conventions ("English Law", "German Civil Code", "GB",
"UK Regulations"). `jurisdiction` holds both "UK" and "United Kingdom".
`bp_deal_documents.region` mixes continents with English counties ("APAC" and
"West Sussex" are both values). And no document table carries a tenant. The
spec's §4.2 scope precedence has nothing underneath it. Built as written it would
resolve every document to GLOBAL and quietly look like it was working.

**5. The document-type vocabulary is currently copied into at least 15 places.**
Fifteen module-level maps across ten files are keyed by document type, plus the
four extraction schemas, plus the DB `category` values. Adding ten new types
without consolidating first means editing fifteen maps and missing some.

My recommendation, in one line: **build the vocabulary tables and the
classification step; do not build scope resolution or the conflict register until
there is a sector/region dimension to resolve against.** Detail in §8.

---

## 1. Coverage and limits of this review

The product is ~208,000 lines of source and ~119,000 lines of tests across 1,410
Python files. I did not read all of it, and I am not claiming to have. Per spec
§2.1's instruction, here is what I reviewed closely versus surveyed.

**Reviewed closely** (read the code, traced the flow, measured the data):
extraction dispatch and its four layers; extraction schemas; the classification
code on both paths; `process_monitor` ingestion and the watcher; `_raw` → `_stg`
→ `_trgt` promotion and its gate; `linking_engine` scoring, profiles and bands;
`graph_resolution` profiles, composition and the edge writer; Neo4j ingestion;
`rule_book` / `policy_engine`; the agent manifest and the four call sites that
inject it; the facts layer (`concept_codes`, `uom`); reference and queue tables
in `proc`; migration, naming and test conventions; the live corpus.

**Surveyed at module level only** (I know what they are and roughly where their
seams are; I have not read them): negotiation and negotiation_advice; email /
mail intake; RGA report generation; sell_side and reseller catalog; the style
engine; i18n internals beyond its public shape; training and finetune; benchmark
pricing; obligations; opportunity mining and critic; triage internals; the Node
gateway and the UI (separate repositories).

**No placement decision in §4 depends on an unreviewed area.** Every component
the spec names lives inside the closely-reviewed set. The one place the two touch
is the agent manifest, whose only prompt-level consumer I found is
`negotiation_agent.py:8046` — that consumer is named in the risk register (§5.4)
rather than silently assumed harmless.

---

## 2. Product map (spec §2.1)

### 2.1 Architecture

A FastAPI service (`src/api/main.py`, 43 routers) over PostgreSQL schema `proc`,
with Neo4j as a secondary graph, Qdrant for vectors, S3 for files, and a local
Ollama model ("AgentNick") for all non-extraction model work. Two repositories
sit in front of it: a Node gateway and the SpendIQ UI.

The layers, in dependency order:

| Layer | Where | Role |
|---|---|---|
| API | `src/api/routers/*` (43) | HTTP surface; thin |
| Agents | `src/agents/*` (33 files, 59.5k lines) | Long-lived role objects. Two are very large; `data_extraction_agent.py` is the legacy extraction path |
| Orchestration | `src/orchestration/*` | `orchestrator.py` plus a declarative DAG `workflow_engine.py`. Both inject the agent manifest |
| Engines | `src/engines/*` (7) | `rule_book` (detection), `policy_engine` (authorisation), `decision_engine`, `query_engine`, `routing_engine` |
| Services | `src/services/*` (451 files, 113k lines) | Where almost all real logic lives, in focused modules |
| Facts | `src/services/facts/*` | A typed fact model with provenance; `concept_codes.py` is the existing concept vocabulary |

The newer work is organised as small single-purpose modules under `src/services/`
with a docstring stating *why* the module exists and what it refuses to do. That
is the house style and the new code should match it.

### 2.2 Data storage — and which store is authoritative for what

| What | Where it lives | Authoritative? |
|---|---|---|
| Field definitions, aliases, regexes, invariants per document type | `extraction_schemas/{contract,invoice,purchase_order,quote}.yaml` | **Yes** — versioned, declarative, read at runtime |
| The concept vocabulary | **Derived** at import from those YAMLs (`facts/concept_codes.py`) | Yes, by derivation — deliberately not a second hand-typed list |
| Unit vocabulary + aliases | `proc.bp_uom_canonical` (38 rows) | Yes, with a 300s TTL cache in `facts/uom.py`, which falls back to its hard-coded seed if the read fails. The migration states a test asserts the two agree |
| Detection rules | `proc.bp_rule` (12 rows) | Yes |
| Authorisation policies | `proc.bp_policy` (47 rows) | Yes |
| Prompts | `proc.bp_prompt`, plus `prompts/*.json` | DB is authoritative; the JSON files are legacy |
| Governed numeric limits | `proc.bp_policy` rows whose `policy_details.rules` hold them (`ReconciliationTolerancePolicy`, `ExtractionEffortPolicy`, `NegotiationBoundsPolicy`, …) | Yes — a missing limit raises |
| Relationship signals, weights, profiles, bands | **Python constants** in `services/linking_engine.py` and `services/graph_resolution/profiles/*.py` | Yes, but code — not data |
| Document-type → table / PK / column maps | **15 module-level dicts in 10 files** | No single owner — see §5.1 |
| Supplier aliases | `proc.bp_supplier_alias` (0 rows) | Table exists, unused |
| Category taxonomy | `proc.bp_category_master` (246 rows, 5 levels, UNSPSC) | Yes |
| UI display strings | `proc.bp_translation` + `bp_translation_language_status` | Yes, display only |
| Graph nodes and edges | Neo4j | No — rebuilt from `_trgt`, which is authoritative |

Schema changes go in `deploy/sql/YYYY-MM-DD_name.sql` with a matching
`_rollback.sql` (177 files). Migrations are additive, idempotent and reversible,
and must be applied to **both** `bp_testdb` and `bp_sqldb`.

### 2.3 Data flow — upload to display, with every model call marked

```
UI / gateway / mail intake
  → S3 + INSERT proc.process_monitor         (category = the user's own label)
  → process_monitor_watcher (polling)
      content-hash dedup → duplicate_of_id
      category → doc_type via a 4-entry map; anything else RAISES
  → services/extraction/dispatch.py — the only live extraction path
      L0 parse      (docling / Paddle / OCR)
      L1 regex      (schema patterns, per-field prior confidence)
      L2 engineered (table extractor, NER validator, address, dates, bbox)
      L3 judge      [MODEL CALL] grounded-last-resort, per unfilled required
                    field only. The returned value must be a verbatim substring
                    of the parsed text or it is discarded
      context_layer [MODEL CALL] identifier recovery, line synthesis
      invariants + completeness → proc.bp_extraction_discrepancy (HITL)
  → *_raw  (flat columns, permanent)
  → *_stg
  → promotion: confidence ≥ 90 AND parent PO found AND link score F ≥ 80
  → *_trgt  (the only tier the product reads)
  → SQL trigger sets deal_id / deal_name / document_id
  → bp_deal_documents, bp_deal_overview, bp_deal_kpis (views)
  → kg_sync → Neo4j  (fire-and-forget, never raises)
  → API → gateway → UI
```

Orchestrated agent work is a separate flow (`orchestrator` / `workflow_engine`),
and it is there that the agent manifest is injected on every step.

Two properties of this flow matter to the build:

- **`_trgt` is the only tier anything reads.** A relationship written anywhere
  else is invisible.
- **The L3 model call is grounded by construction.** The value must appear
  verbatim in the page. This is the product's existing answer to hallucination
  and the spec's "evidence spans" requirement is already satisfied in extraction.

### 2.4 Conventions

- **Tables:** `proc.bp_*`; indexes `ix_bp_<table>_<col>`.
- **Migrations:** `deploy/sql/YYYY-MM-DD_name.sql` + `_rollback.sql`; additive,
  idempotent, `ON CONFLICT DO NOTHING` on seeds so a re-run never clobbers a
  human's confirmation.
- **Reference tables:** the `bp_uom_canonical` shape — `tenant_id` defaulting to
  `'default'`, `aliases text[]`, `status` in (active, proposed, rejected) with
  proposed never resolving, `source`, `observed_count`, `first/last_observed_at`,
  `confirmed_by/at`, `valid_from` / `valid_to` for versioning, `recorded_at`,
  and CHECK constraints that encode the invariants.
- **Versioning:** temporal (`valid_from` / `valid_to`) on reference data; integer
  `version` + `*_status` smallint on governance tables (`bp_rule`, `bp_policy`).
- **Propose-only writes:** `bp_extraction_hint_proposal` is the template —
  `status`, `reviewed_by`, `reviewed_date`, `review_reason`, and a pointer to the
  row the confirmation produced (`resulting_prompt_id`).
- **Model calls:** `services/ollama_client.py`, with a JSON schema passed as
  `format=` for grammar-constrained output. **No unions** — Ollama ignores
  `oneOf`/`discriminator`.
- **Tests:** `tests/` mirrors `src/` (695 files). Markers `gpu` and
  `integration`. DB-backed tests need `PROCWISE_TEST_LIVE_DB=1`; the default is a
  fake DB. Run with `./venv/bin/python` (the test venv is not the runtime venv).
- **CI:** exactly one workflow, `.github/workflows/formula-goldens.yml` — golden
  vectors that fail the import when pinned behaviour changes. That is the
  precedent to extend for §8.1's reference-data checks.
- **Failure posture, and it is not uniform:** `RuleBook` *raises* when the store
  is unreadable or empty, on the stated grounds that a sweep finding nothing
  looks exactly like a clean scan. `PolicyEngine` returns `[]`, so a governance
  outage is indistinguishable from "nothing governs this". New governance reads
  must pick the RuleBook posture deliberately.

### 2.5 Existing overlaps — things already doing part of this job

| Overlap | Where | What it does |
|---|---|---|
| **Document type list** | `process_monitor_watcher.py:523`, `utils/procurement_schema.py:541,561`, plus 12 more maps | Four types, duplicated |
| **Per-field alias lists** | `extraction_schemas/*.yaml` → `canonical_labels` | A Synonym_Index for *field labels*, scoped per document type. Already the right idea, one level down |
| **Hard-coded type synonyms in code** | `data_extraction_agent.py:7606` | Maps "msa", "statement of work", "sow", "service contract" **all onto `Contract`** — the exact collapse this spec exists to undo |
| **Keyword type scoring** | `data_extraction_agent.py:381` (`DOC_TYPE_KEYWORDS`) + `extraction_engine.py:6748` | A real scoring classifier with a margin requirement, on the legacy path |
| **Parent pointer + variation flags** | `bp_contracts` / `bp_contract_master`: `parent_contract_id`, `is_amendment`, `amendment_ref`, `document_version`, `contract_type` | The GPSS raw materials, already extracted — all free text, no vocabulary |
| **Relationship scoring** | `linking_engine.py` | Signals, clusters, weights, coverage, caps, bands. Faithful to a maths spec |
| **Derived-edge writer** | `graph_resolution/edge_writer.py` | The single gate to Neo4j, with redaction and an uncalibrated-profile block |
| **Link proposal queues** | `link_proposals.py`, `deal_link_proposals.py` | Propose-never-link, into the queue a buyer already works |
| **Human-declared precedence** | `declared_linkage.py` | A confirmed human grouping outranks any inferred correlation — spec principle 3, already implemented |
| **Blocking vs non-blocking findings** | `blocks_promotion` on `bp_extraction_discrepancy`, `bp_detection_finding` | HARD vs SOFT, already live |
| **Review queues** | `bp_extraction_discrepancy` (5,372 live rows), `extraction_review_queue` (0), `bp_extraction_hint_proposal` (0), `bp_deal_proposal`, `bp_approval`, `/promotion/review-queue` | Six surfaces; the discrepancy table is the one humans actually work |
| **Placeholder nodes** | `kg_ingestion_service.py:208` | `MERGE (t:Target {key: $fk})` already mints a node from an unresolved foreign key — placeholders exist, unlabelled and uncounted |

The GPSS name itself appears in the repo once, as a decision **against** it:
`facts/concept_codes.py` records (2026-08-07) that there is no GPSS dictionary
here and that the extraction schemas already are the data dictionary, naming the
column `concept_code` rather than `gpss_code` precisely so no reader assumes
external authority. That ruling was about *field* concepts. This spec is about
*document* and *relationship* concepts, which that file does not cover — so it is
not contradicted, but the naming choice should be honoured.

### 2.6 Constraints

- **No tenant dimension on any document table.** `tenant_id` exists on 12 newer
  tables; `process_monitor`, every `_raw`/`_stg`/`_trgt` table, `bp_contracts`
  and `bp_deal*` have none. Tenant-scoped behaviour cannot be implemented.
- **No sector dimension anywhere.** "sector" appears zero times in `src/`.
- **Region is not a jurisdiction.** Measured: `bp_deal_documents.region` holds
  continents (Europe 9,506; North America 5,240; APAC 2,126) *and* English
  counties (West Sussex 10; Buckinghamshire 7; Greater Manchester 4).
  `governing_law` has six free-text values across two conventions.
  `jurisdiction` holds both "UK" (641) and "United Kingdom" (32).
- **`process_monitor.document_type` is a trap.** It holds the *file format* —
  measured values are `xlsx` (162), `pdf` (13), `docx` (2). The document type is
  in `category`. A build that reaches for the obviously-named column gets the
  file extension.
- **`get_conn()` is autocommit.** Rollback is a no-op and `FOR UPDATE` locks end
  with the statement. Multi-statement consistency needs an explicit transaction.
- **Governance reads are fail-open at every layer except RuleBook.** `facts/uom.py`
  is the same shape: if `bp_uom_canonical` cannot be read it continues on its
  hard-coded seed. A new reference table copied from this pattern inherits that
  posture, so decide it deliberately rather than by copy.
- **Four document types is enforced, not conventional.** `process_monitor_watcher.py:531`
  raises `unsupported doc_type` for anything else.
- **Shared checkout.** Another session's work sits in this repository's git index.
  Never `git add -A`; commit named paths only.
- **`docs/` is gitignored** — anything written there is uncommittable. Specs go in
  `specs/`.
- **The corpus is mostly seeded, not ingested.** `_trgt` holds 12,408 invoices,
  21,054 quotes and 5,042 POs, but `process_monitor` has only 177 rows — so
  fewer than 200 documents ever came through the live extraction path.
  `proc.bp_contracts` (the extraction destination for contracts) is **empty**;
  the 3,051 contracts are seeded into `bp_contract_master`.

---

## 3. Component locator (spec §2.2)

| Component | Exists? | Where | Schema / shape | Read at runtime | Edited by |
|---|---|---|---|---|---|
| **Concept_Library** | **No** | — | — | — | — |
| ↳ nearest thing | Yes | `facts/concept_codes.py` | `FrozenSet[str]` of field names | Imported once | Not edited — derived from YAML |
| **Tax_Scope** | **No** | — | Tax is per-document rates in `triage/tolerance.py`, `extraction_v2/invariants.py` | — | — |
| **UoM_Index** | **Yes** | `proc.bp_uom_canonical` (38 rows) | `uom_code` PK, `tenant_id`, `dimension`, `aliases text[]`, `factor_days`, `factor_convention`, `is_billing_basis`, `status`, `source`, `observed_count`, `first/last_observed_at`, `confirmed_by/at`, `valid_from/to`, `recorded_at`, `non_unit_kind` | `facts/uom.py`, cached with TTL | `UPDATE`; seeded by `deploy/sql/2026-08-07_uom_canonical.sql`; new rows arrive as `status='proposed'` from observation |
| **Document_Index** | **No** | — | — | — | — |
| **Synonym_Index** | **No table** | — | No `scope` / `region` / `confidence` columns exist anywhere | — | — |
| ↳ field-label aliases | Yes | `extraction_schemas/*.yaml` → `canonical_labels` | List of strings per field, with `patterns` carrying `prior_confidence` | `pattern_registry` → `pattern_extractor` | Edit the YAML, deploy |
| ↳ type aliases | Yes, hard-coded | `data_extraction_agent.py:7606` | Python dict; **MSA / SOW / "statement of work" → `Contract`** | Legacy path only | Code edit |
| ↳ supplier aliases | Table only | `proc.bp_supplier_alias` (0 rows) | `alias_id`, `alias_name`, `supplier_id`, `created_by`, `created_date` | — | — |
| **Conflict_Register** | **No** | — | — | — | — |
| **conflict_rulings** | **No** | — | — | — | — |
| **Sector_Glossary** | **No** | — | "sector" appears 0 times in `src/` | — | — |
| **Language_Index** | **Yes, equivalent** | `proc.bp_translation`, `bp_translation_language_status`; `services/i18n/*` | UI strings keyed by a hash of source text, per locale | `i18n/service.py` at render time | AI fill + human override |
| ↳ *is it read by matching code?* | **No** | — | The only exact-match path in i18n is `registry.py:121 _exact()`, which matches **language names** for the language picker, not concepts | — | — |
| **AGENT_MANIFEST** | **Yes, equivalent** | `services/agent_manifest.py` → `AgentManifestService.build_manifest(agent_key)` | Returns `{task, policies, knowledge}`. `knowledge` = **every** table profile (all columns + synonyms) + 6 hard-coded relationships. **No `loads` filter, no `max_rows`, no per-task row** | `orchestrator.py:496,1512,1653,3270`; `workflow_engine.py:424`; `GET` via `routers/agents.py:114` | Code edit |
| **Rules engine** | **Yes** | `proc.bp_rule` (12 rows) + `engines/rule_book.py` | `rule_id bigint` (not `R-*`), `rule_name`, `detector_slug`, `finding_type`, `scope`, `required_fields jsonb`, `conditions jsonb`, `severity`, `rule_status smallint`, `version int` | Loaded once, cached by slug, reload on demand. **Raises** if unreadable or empty | `UPDATE` / migration |
| ↳ rule-group convention | Partial | `finding_type` ∈ {opportunity, non_conformance, anomaly}; `scope` holds dataset names (`po_lines`, `contracts`, `supplier_master`, `invoice_lines`) | A rule is bound to a **detector in code** by `detector_slug` — a new rule without a detector does nothing | | |
| ↳ HARD / SOFT | **Yes, differently named** | `blocks_promotion boolean` on `bp_extraction_discrepancy` (5,372 rows) and `bp_detection_finding` (8,147 rows). `bp_rule.severity` is low/medium/high — advisory, not blocking | | | |
| **scope_loading_guide** | **No** | — | No jurisdiction → rule-group mapping exists | — | — |
| **Relationship registry** | **Yes, as code** | `services/linking_engine.py` | `_COMMON_SIGNALS` (7 signals: `id`, `cluster`, `tier`, `weight`, `appl`, `cap`, `kind`); `PROFILES` = `invoice_po`, `quote_po` with `p0`, `alpha`, `floor`; bands `_BAND_AUTO 92` / `_BAND_WARN 80` / `_BAND_REVIEW 65` / `_BAND_WEAK 45`, else block; `register_profile()` extension seam | Imported | Code edit |
| ↳ further profiles | Yes | `graph_resolution/profiles/*.py` — `contract_coverage`, `contract_succession`, `item_equivalence`, `supplier_identity` | All four listed in `edge_writer.UNCALIBRATED_PROFILES`, so their edges **never auto-link** regardless of F, until a labelled sample exists | | |
| ↳ *does it copy types/roles?* | **It has no roles at all.** No role vocabulary exists. Document types appear only as profile-name strings (`"invoice_po"`) and dict keys | | | | |
| **Stage 0 / relationship engine** | **Yes** | `linking_engine.score_link()`; entry via `promote_ready()`, `POST /promotion/run`, `/promotion/review-queue`, `/promotion/link-proposals` | In: two rows + lines + profile name. Out: `F`, `band`, `P_raw`, per-signal detail with status OK/MISSING/CONFLICT | | |
| **Knowledge graph** | **Yes** | Neo4j; `kg_ingestion_service.py`, `extraction/kg_sync.py`, `graph_resolution/edge_writer.py` | Node label from the table category, keyed `row_id`. Edges `HAS_<LABEL>`, `LINKED_TO_<LABEL>` from foreign keys; derived edges `SAME_ENTITY`, `OF_ITEM`. Mirrors `_trgt`; never authoritative; sync never raises | | |
| ↳ placeholder nodes | **Yes, accidentally** | `kg_ingestion_service.py:208` `MERGE (t:{target} {{{key}: $val}})` creates the target from an unresolved FK | Not labelled as a placeholder, not counted, not aged, not reconciled | | |
| ↳ merge | Yes | Neo4j `MERGE` on `row_id` makes re-ingestion idempotent | | | |
| **Extraction pipeline** | **Yes** | `services/extraction/dispatch.py` — **the only live path** | Model calls: L3 `judge_gate` (per unfilled required field, verbatim-substring enforced) and `context_layer`. Prompts get a **field-level slice**, never a whole table. JSON schema via Ollama `format=` | | |
| ↳ legacy path | Yes, dormant | `agents/data_extraction_agent.py` (reachable via `POST /workflows/extract`) | Holds the classifier and the type-synonym map | | |

---

## 4. Placement decisions (spec §2.3)

Each decision cites the product-map section that justifies it.

| # | Component | Decision | Where, and why |
|---|---|---|---|
| 1 | **Concept_Library** | **CREATE** as `proc.bp_concept` | §2.2: the only concept vocabulary is *derived* from the extraction schemas and covers field names, not document types or relationships. A derived set cannot hold a definition, a `not_to_be_confused_with` pointer or a status, so there is nothing to extend. Build it in the `bp_uom_canonical` shape (§2.4). Name the key `concept_code`, matching `facts/concept_codes.py`'s stated reason for avoiding `gpss_code`. |
| 2 | **Document_Index** | **CREATE** as `proc.bp_document_type`, modelled on `bp_uom_canonical` | §3 row "UoM_Index": that table is the live sibling the spec points at, and it already carries aliases, status, source, observation counts and confirmation — so `Document_Index` and `Synonym_Index` collapse into **one** table with an `aliases text[]`, exactly as units do. One table, not two. |
| 3 | **Synonym_Index** | **CREATE — folded into #2** | §2.2 + §2.4: the product's established way to hold aliases for a concept is an array on the concept row (`bp_uom_canonical.aliases`). A second table would be a second place to edit the same fact, against spec principle 6. **Consequence to accept:** one alias meaning two concepts is then two rows containing the same alias, which a validation query detects (§8.1) rather than a foreign key preventing. |
| 4 | **Conflict_Register** + **conflict_rulings** | **ASK — recommend deferring** | §2.6: a ruling is `scope` + `region` keyed, and the product has no sector dimension and no normalised region. Built now, every ruling would be GLOBAL and the evidence tests would run unconditionally — which is not a conflict register, it is a hard-coded `if`. Recommend: hold the ruling machinery until #11 lands, and in the meantime let a genuine collision resolve to `UNRESOLVED` and go to review (principle 4), which costs nothing and fabricates nothing. **If the ruling is to build it anyway: one table, no mirror** — see §7.3. |
| 5 | **Sector_Glossary** | **ASK — blocked** | §2.6: no sector exists to glossarise. A tier table keyed on a dimension no document carries cannot be populated or tested. |
| 6 | **Language_Index** | **EXTEND — but no work needed now** | §3: `bp_translation` already holds display labels per locale and no matching path reads it. New concept labels should flow through that service when the concepts reach a screen. The spec's test "fails if any lookup path reads this table" is worth adding as a guard (§8.1) and currently passes. |
| 7 | **AGENT_MANIFEST** | **EXTEND** `services/agent_manifest.py` | §3: `AgentManifestService` is the real manifest and every orchestrated step already consumes it. It has no `loads` / filter / `max_rows` concept — that is the gap to close, and closing it also fixes risk §5.4. Add the four task rows as filtered bundles; move the hard-coded `_PROC_RELATIONSHIPS` out of the code path that reaches a prompt. |
| 8 | **Rules engine, group R-REL** | **EXTEND** `proc.bp_rule` + `engines/rule_book.py` | §3: the table and loader exist with `scope`, `conditions jsonb`, `severity` and `version`. Two adaptations are forced by what is there: (a) `rule_id` is a bigint, so `R-REL` must be a value in a *group* column (or in `scope`), not the id — follow the existing convention rather than imposing `R-*`; (b) `detector_slug` binds every rule to code, so each R-REL rule needs a detector function or it silently does nothing. For HARD/SOFT, **reuse `blocks_promotion`** (§2.5) rather than adding a third vocabulary for the same idea. |
| 9 | **scope_loading_guide** | **ASK — blocked**, same reason as #5 | §2.6. |
| 10 | **Relationship registry** | **EXTEND in place, no schema change** — and the spec's verification **passes** | §3: it holds no copies of types or roles because no role vocabulary exists at all. What it does hold is profile names coupled to document-type pairs. New relationship profiles register through the existing `register_profile()` seam, and new profiles must be added to `edge_writer.UNCALIBRATED_PROFILES` until a labelled sample exists — that is the product's own rule and it applies to this work. |
| 11 | **Scope resolution (§4.2)** | **ASK — blocked** | §2.6, measured: no tenant on document tables, no sector, `region` polluted with counties, `governing_law` free text in two conventions, `jurisdiction` spelled two ways. Building §4.2 on this resolves everything to GLOBAL while appearing to work — a silent-failure shape this codebase explicitly designs against (`RuleBook`, `UOM_UNMAPPED`, `bp_uom_canonical` status). **Prerequisite work, if wanted:** normalise `governing_law`/`jurisdiction` into a region code via a reference table in the `bp_uom_canonical` shape, with unmapped values recorded as `proposed` rather than guessed. |
| 12 | **Review queue (§3.10)** | **EXTEND** `proc.bp_extraction_discrepancy` | §2.5: six queue-like surfaces exist and this is the only one humans actually work (5,372 rows). It already has `issue_type`, `severity`, `status`, `blocks_promotion`, `resolved_by/at/value/action` and evidence columns (`evidence_page`, `evidence_bbox`, `evidence_text`) — which is what `UNRESOLVED_CONFLICT` and `UNKNOWN_TYPE` items need. The four item types become `issue_type` values. For promotion-on-confirm, copy `bp_extraction_hint_proposal`'s shape (§2.4): a `resulting_*_id` pointing at the row the confirmation created. |
| 13 | **Classification step (§4.1 step 1)** | **CREATE** a new module under `src/services/extraction/`, called from `dispatch.py` | §2.3: `dispatch.py` is the only live path, and the classifier that exists is on the dormant path. New code goes in a small single-purpose service module (§2.1 house style), not into `data_extraction_agent.py` (59.5k lines, legacy). |
| 14 | **The four-type gate** | **EXTEND** `process_monitor_watcher.py:523-532` — **and this is the first blocker** | §2.6: the gate *raises* on any fifth type. Until it reads the type vocabulary from #2 instead of a 4-entry literal, every new document type fails at upload. Nothing else in Phase 3 can be demonstrated before this changes. |
| 15 | **Document-type map consolidation** | **EXTEND — recommend doing it first** | §2.5 / §5.1: 15 maps in 10 files are keyed by document type. Adding ten types without consolidating means fifteen edits and a near-certain miss. |
| 16 | **Placeholder nodes (§4.4)** | **EXTEND** `kg_ingestion_service.py` + `graph_resolution/edge_writer.py` | §3: FK-derived placeholder nodes are already being created, just not labelled, counted, aged or reconciled. Marking what already happens is smaller and safer than building a parallel mechanism. |
| 17 | **Link-writing policy (§4.3)** | **EXTEND** `linking_engine` + `edge_writer` | §3: bands, the auto gate, the review queue and the discard path all exist with thresholds 92/80/65/45. This is configuration and wiring, not new machinery. |

---

## 5. Risk register (spec §2.4)

### 5.1 Fifteen copies of the document-type vocabulary — *not* in the spec's list, and the largest practical risk

| File | Constant |
|---|---|
| `agents/data_extraction_agent.py` | `DOC_TYPE_CONTEXT`, `DOC_TYPE_KEYWORDS`, `DOC_TYPE_EXTRA_KEYWORDS`, `DOC_TYPE_EXTRA_INSTRUCTIONS` |
| `utils/procurement_schema.py` | `BP_DOC_TYPE_TO_TABLE`, `DOC_TYPE_TO_TABLE`, `CATEGORY_TO_DOC_TYPE` |
| `services/extraction/persistence.py` | `_DOC_PK_FIELD` |
| `services/extraction/dispatch.py` | `_RECOVERED_LINE_AMOUNT_COL` |
| `services/extraction/kg_sync.py` | `_TRGT_TABLE`, `_PK_COL` |
| `services/extraction_v2/provenance.py` | `PARENT_TABLE_FOR_DOC_TYPE` |
| `services/extraction_v2/locator/strategies/header.py` | `_DOC_TYPE_WORDS` (regex) |
| `services/benchmark_live.py` | `_DOC_TYPE_POOL_PREFIX` |
| `services/price_outlier/detector.py` | `_POOL_PREFIX_BY_DOC_TYPE` |
| `services/process_monitor_watcher.py:523` | inline `doc_type_map` + the raising gate |

Plus the four `extraction_schemas/*.yaml` and the `category` values in the DB.
Some of these are legitimately physical (a type → table map must exist
somewhere); the point is that no single one of them is the vocabulary, so
"the list of document types" has no owner.

### 5.2 Conflict_Register / conflict_rulings both hand-edited — **not applicable**
Neither table exists, so there is no dual-master problem to fix. §7.3 recommends
never creating one.

### 5.3 Matching code reading Language_Index — **clean**
`bp_translation` is read only by `i18n/service.py` at render time. The one
exact-match path in i18n (`registry.py:121`) matches **language names** for the
picker, not concepts. The spec's guard test is worth adding and will pass today.

### 5.4 An agent prompt that loads whole reference tables — **confirmed, one place**
`AgentManifestService.build_manifest()` returns a `knowledge` bundle containing
**every** table profile — all columns, all field synonyms — plus six hard-coded
relationships, with no task filter. `orchestrator.py` injects it on every step
(`:1512`, `:1653`, `:3270`, `:496`) and `workflow_engine.py:424` does too.
`base_agent.py` strips heavy knowledge from the *snapshot* (`_remove_knowledge_blocks`),
which contains the blast radius — but `negotiation_agent.py:8046` serialises
`context.knowledge_base` straight into its prompt. So the risk is real, has
exactly one prompt-level consumer today, and spec §3.7 is the fix.

### 5.5 The relationship registry holding copies of types or roles — **clean, trivially**
There are no roles anywhere in the product to copy, and document types appear in
the registry only as profile-name strings. Verification passes. The inverse risk
applies instead: when roles are introduced, the registry must reference them
rather than gaining the fifteenth-and-sixteenth copy (§5.1).

### 5.6 Additional risks found
- **`process_monitor.document_type` holds the file format**, not the document
  type (measured: xlsx/pdf/docx). A build reaching for the obvious column gets
  the extension.
- **The type is user-asserted and never verified.** Nothing today reads the page
  and disagrees with the uploader. Introducing a classifier that *can* disagree
  is a behaviour change on existing uploads, not only on new types, and needs a
  stated precedence rule: who wins, the uploader or the evidence?
- **Contracts do not flow through the live path at all.** `proc.bp_contracts` is
  empty; the 3,051 contracts are seeded in `bp_contract_master`. Every GPSS type
  in the spec (framework, MSA, SOW, call-off, variation) is a contract-family
  document, so the type most affected by this build is the one with zero live
  ingestion evidence. A golden set cannot be drawn from history here; it must be
  built from real documents.
- **The data already contains the collisions, and already resolves them wrongly.**
  `contract_type` free text, measured: Consulting 645, NDA 612, SLA 585,
  **Master Agreement 581**, Service Agreement 579, Service Contract 7,
  Amendment 2, Service 2 — plus misfiled Invoice 4, Policy 3, Purchase Order 1.
  No framework, call-off or SOW at all. And `data_extraction_agent.py:7606`
  maps msa, sow and "statement of work" all onto `Contract`.
- **`parent_contract_id` is populated on 1,561 of 3,051 contracts and resolves on
  zero** — the references were minted in a different namespace (`C1543` against
  an actual `C00002`). Any "exact identifier match links automatically" rule that
  trusts this column links nothing, and would link *wrongly* if the namespaces
  ever partially overlapped.
- **`bp_supplier_alias` has existed and stayed empty.** The nearest precedent for
  an alias table in this product was built and never filled. Worth knowing before
  building another one.

---

## 6. Where the spec conflicts with the product — flagged, not resolved

Spec §1 says to stop and flag when an implementation choice conflicts with a
design principle. Four do.

**6.1 Principle 1 ("the AI reads documents and answers evidence questions") vs
the product's grounding contract.** The live L3 judge accepts a value only if it
is a *verbatim substring* of the parsed page. A ruling test answer ("does this
document state an order of precedence?") is a judgement about meaning; "yes" is
not a substring of anything. So ruling answers cannot be validated the way every
other model output in extraction is. The honest shapes are: require the model to
return the **clause text** it relied on and substring-check *that* while treating
yes/no as unverified, or route every ruling-derived type through review. I
recommend the first, and `not_found` must stay a first-class answer.

**6.2 Principle 3 ("exact identifiers link automatically") vs measured identifier
quality.** `parent_contract_id`: 1,561 populated, 0 resolve. PO references needed
`_norm_po` normalisation *and* a canonicalisation pass before exact joins worked
at all. An identifier match in this corpus is evidence, not proof. Recommend the
auto-link requires an exact match **after** declared normalisation, and that any
new identifier pattern ships with a measured resolve rate before it is trusted.

**6.3 Principle 5 ("load only what the task needs") vs the current manifest.**
Already covered in §5.4. It is a pre-existing violation, and §3.7 fixes it — but
it means the manifest change is a *refactor of live behaviour*, not an additive
one, so it needs its own tests.

**6.4 Principle 4 ("nothing is forced") vs the four-type gate.** The gate does
not record "unknown" — it raises and the document fails extraction. Making
"unknown" expressible is a change to the failure path, which currently surfaces
as a hard error a human sees. Preserve that visibility: an `UNKNOWN_TYPE` review
item must be at least as loud as today's error, or this change *loses* signal.

---

## 7. The §10 open questions the product already settles

**7.1 Score band thresholds (Q4) — already decided.** Auto ≥ 92, warning ≥ 80,
review ≥ 65, weak ≥ 45, below that block (`linking_engine.py:77-81`); the
promotion gate uses confidence ≥ 90 and F ≥ 80. Recommend reusing these
unchanged — they are calibrated against a maths spec and a live corpus.
**One addition the product insists on:** a new profile is listed in
`edge_writer.UNCALIBRATED_PROFILES` and cannot auto-link at any score until a
labelled sample exists. Four existing profiles are still in that state.

**7.2 Full initial list of document types (Q2) — partly answerable from data.**
Live `contract_type` values are in §5.6. Nothing in the corpus says framework,
call-off or SOW, so the initial seed is a product decision, not a data one. But
the seed must include the 13 observed values (including the misfiled ones, as
`rejected` or as aliases of the right concept) or re-ingestion will contradict it.

**7.3 Which of Conflict_Register / conflict_rulings is master (Q3) — recommend
the question be dissolved.** The spec asks for one master and a generated mirror
with a CI check that fails when they differ. The product has no precedent for a
generated mirror of a reference table, and one source of truth per fact
(principle 6) is better served by **one table** holding the collision and its
ruling together. A mirror plus a drift check is machinery that exists only to
detect a problem that not having the mirror prevents.

**7.4 Roles (Q1), sectors (Q2), queue ownership and SLA (Q5), override-rate
threshold (Q6) — genuinely need your ruling.** Nothing in the product implies
them. Note that Q6 cannot be measured until the review queue carries these item
types, so a rollout threshold cannot gate Phase 2 or 3.

---

## 8. Recommended Phase 2 scope (one design, for approval)

Taking §4's blocked items seriously, the buildable half is coherent on its own
and delivers the spec's actual goal — *stop confusing similar terms* — without
building anything resting on a dimension that does not exist.

**Build:**
1. `proc.bp_concept` and `proc.bp_document_type` in the `bp_uom_canonical` shape,
   aliases as an array, `status` where `proposed` never resolves. Seeded with the
   13 observed `contract_type` values plus the agreed GPSS types, and with the
   RELATIONSHIP domain (roles, link types, execution modes, event kinds).
2. Consolidate the 15 document-type maps onto that table, and replace the
   `process_monitor_watcher` gate so a new type no longer raises (§4 #14, #15).
3. The classification step in `dispatch.py`, with the uploader-vs-evidence
   precedence rule stated explicitly, `UNKNOWN_TYPE` to the discrepancy queue,
   and the clause-text grounding compromise from §6.1.
4. R-REL in `bp_rule`, reusing `blocks_promotion` for HARD/SOFT, with a detector
   per rule.
5. Manifest task slices (§3.7), which also closes risk §5.4.
6. Validation checks as a second CI workflow in the `formula-goldens.yml` mould.

**Do not build yet:** scope resolution and precedence (§4.2), Sector_Glossary,
scope_loading_guide, and the conflict register / rulings — all four need a
sector or normalised-region dimension that does not exist (§4 #4, #5, #9, #11).
Collisions resolve to `UNRESOLVED` and reach the review queue in the meantime,
which is principle 4 working as intended.

**Prerequisite, if scope resolution is wanted:** normalise `governing_law` and
`jurisdiction` into region codes via a reference table in the same shape, with
unmapped values recorded as `proposed`.

---

## 9. Acceptance check against spec §2

| Requirement | Status |
|---|---|
| Full product review before placement | Done, with coverage limits stated (§1) |
| Product map included | §2 |
| EXTEND / CREATE / ASK for every component, citing the map | §4 — 17 rows, every one cites a map section |
| Overlaps found in 2.1 have a stated plan | §2.5 catalogues 13; §4 assigns each one consolidate-or-leave |
| §2.4 risks explicitly flagged | §5 — one not applicable, one clean, one confirmed with its single consumer, plus 7 found beyond the spec's list |
| Nothing built | Confirmed: no code, schema or data changed |
