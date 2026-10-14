# Agent Policy Governance — Stage 2 (Extraction Agent) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** People upload many policy documents at once. One extraction agent (AgentNick, local) reads them all in one run and proposes agent policies as Draft versions, plus a "Not enforceable" list with reasons. Results stream into the list while the run is working. A revised document maps each clause onto the existing policy ID: unchanged clauses get no new version, changed ones get a new draft, new ones get a new policy, and removed ones are marked "Proposed retire". Reviewers can "Ask the agent to fix it" for one policy.

**Architecture:**
- **Upload:** documents go browser → S3 directly, through a presigned PUT that the backend issues via the gateway. Nothing travels through the gateway's Lambda body, and nothing touches the procurement upload pipeline.
- **Registration:** the backend hashes each document, versions it, and stores its parsed text.
- **Extraction run:** an extraction run is a restart-safe background job, following `src/services/rga/job_store.py` and `job_runner.py`. It splits each document into section chunks and asks AgentNick for schema-constrained JSON per chunk.
- **Conversion:** plain code converts each proposed policy into the stage-1 form state, matches it to existing policies, and saves drafts through the stage-1 repository.
- **Screens:** the screens poll the run for new items.

**Tech Stack:** Python/FastAPI, psycopg2, boto3 (S3), docling via `services/extraction/parser.parse`, Ollama via `services/ollama_client`, NestJS gateway, `engine.js` + vitest.

**Spec:** `specs/2026-10-08-agent-policy-governance-brief.md` §1, §2, §3.2 (fix), §3.6, §4 (upload, grouping), §5, §9 (revised-document ruling). `specs/2026-10-08-agent-policy-governance-design.md` (rulings). Stage 1 code is the base; its interfaces are reused unchanged unless noted.

## Global Constraints

Everything in stage 1's Global Constraints still binds (`specs/2026-10-08-agent-policy-governance-plan-1-foundation.md`). In addition:

- **Model:**
  - The model is AgentNick only: `ollama_client.DEFAULT_MODEL` (`BeyondProcwise/AgentNick:unified`). Never another model.
  - Always call it with `think=False`.
- **Response schema:**
  - Pass a JSON schema as `format=`.
  - No `oneOf` or `anyOf` unions, and no recursive `$ref`. Use one typed array per kind and `Literal` enums.
- **Load options:** every model call sends `ollama_client.load_options()`. Otherwise the shared 18 GB model reloads, which costs minutes for every session. Other callers' behaviour must not change.
- **Agent output is a proposal:**
  - It never sets `checked`, and never saves anything as `live`.
  - Every policy it produces is saved with `intent="draft"`.
  - `hidden.setBy="extraction_agent"`.
- **Never invent:**
  - A name the registry lacks goes into `hidden.unknownNames`.
  - An input that is not available at the checkpoint goes into `hidden.missingInputs`.
  - The code verifies both against the registry; it never trusts the model.
- **Excerpts are checked word for word** with `services/obligations/grounding.is_quote_grounded`. Do NOT use `extraction_v3.grounding.is_value_grounded`: its digit fallback lets fabricated sentences through.
- **Upload limits** come from the existing `document_intake_authority` policy: 20 files per request and 26,214,400 bytes per file. They are read-only, and a missing value refuses the upload.
- **Accepted types:** `.pdf`, `.docx`, `.txt`, `.md`.
- **S3:**
  - Every object goes under `agent-policy-documents/` in the backend's `settings.s3_bucket_name`.
  - Keys are `agent-policy-documents/uploads/<uuid4>/<sanitised filename>`.
  - Write probes are deleted after use.
- **Revised documents (ruling):**
  - An unchanged clause gets no new version.
  - A changed clause gets a new draft version, and the live version stays live.
  - A new clause gets a new draft.
  - A removed clause is marked "Proposed retire" and is never retired automatically.
  - The screen shows the document before and after.
- **Prompts** live in `proc.bp_prompt` (`prompts_desc` jsonb, key `prompt_template`) and are inserted into BOTH databases. If the row is missing, the run fails with `prompt unavailable`. There is no text held in code.

## Review Focus

1. **A 60-page document** must be chunked under the context budget, with no section silently dropped. Every chunk is either extracted or recorded as an error item. Test: `test_every_section_lands_in_exactly_one_chunk` (Task 3).
2. **The model returns malformed JSON or an invented tool name.** The item becomes an error item, or the policy carries `unknownNames`, and the run continues. It never crashes and never saves a policy using the invented name as if it were known. Test: `test_invented_tool_name_goes_to_unknown_names` (Task 4) and `test_bad_chunk_is_an_error_item_and_run_continues` (Task 6).
3. **The server restarts mid-run.** The run is healed to `failed` on its next read, and its partial items stay visible. Test: `test_stale_run_heals_to_failed` (Task 5).
4. **The same file is uploaded twice, or the same filename with different content.** The first is the same version, with no new extraction needed. The second is a new version of the same document. Test: `test_same_bytes_same_version_new_bytes_new_version` (Task 2).
5. **Re-extracting an unchanged document creates no new versions and no new policies.** Test: `test_reextracting_unchanged_document_writes_nothing` (Task 6).

---

## File Structure

**BP_Backend**

| Path | Responsibility |
|---|---|
| `deploy/sql/2026-10-09_bp_agent_policy_extraction.sql` (+ `_rollback.sql`) | document, version, run, item tables, and source columns on `bp_agent_policy` |
| `deploy/sql/2026-10-09_agent_policy_extraction_prompts.sql` (+ `_rollback.sql`) | the two governed prompts |
| `src/services/ollama_client.py` | add the opt-in `use_load_options` flag (Task 3) |
| `src/services/agent_policy/documents.py` | presign, register, hash, version, parse text |
| `src/services/agent_policy/sections.py` | split text into numbered sections; chunk under a budget; before/after diff |
| `src/services/agent_policy/extraction_schema.py` | pydantic schema the model must fill (flat, no unions) |
| `src/services/agent_policy/converter.py` | proposed item → stage-1 form state |
| `src/services/agent_policy/matching.py` | stable-ID matching and revision decisions (pure) |
| `src/services/agent_policy/extractor.py` | build the prompt, call AgentNick, validate the reply |
| `src/services/agent_policy/run_store.py`, `run_runner.py` | restart-safe job store and runner |
| `src/api/routers/agent_policies.py` | new endpoints (same gateway-key trust) |
| `tests/agent_policy/fixtures_docs/` | three short policy documents (Finance, Customer operations, Security), one with a tiered clause |

**Gateway:** the `agent-policy` module gains the new routes.
**UI:** `agentPolicy/upload.js` and `agentPolicy/runView.js` (pure), plus new `ap*` functions in `engine.js`.

---

### Task 1: Migration — documents, versions, runs, items, source columns

**Files:** `deploy/sql/2026-10-09_bp_agent_policy_extraction.sql`, `_rollback.sql`, and `tests/migrations/test_2026_10_09_bp_agent_policy_extraction.py` (live, both databases).

The DDL, verbatim except for the formatting choices left to the implementer:

```sql
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_policy_document (
    document_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title         TEXT NOT NULL,                 -- from the first upload's filename, editable later
    match_name    TEXT NOT NULL,                 -- normalised filename used to recognise a revision
    latest_version INTEGER NOT NULL DEFAULT 1,
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_document_match ON proc.bp_policy_document (match_name);

CREATE TABLE IF NOT EXISTS proc.bp_policy_document_version (
    document_id   BIGINT NOT NULL REFERENCES proc.bp_policy_document (document_id),
    version       INTEGER NOT NULL CHECK (version >= 1),
    filename      TEXT NOT NULL,
    s3_key        TEXT NOT NULL,
    byte_size     BIGINT NOT NULL CHECK (byte_size > 0),
    content_hash  TEXT NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    parsed_text   TEXT,                          -- NULL until the first run parses it
    parsed_at     TIMESTAMPTZ,
    uploaded_by   TEXT NOT NULL,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, version),
    CONSTRAINT ux_bp_policy_document_version_hash UNIQUE (document_id, content_hash)
);

CREATE TABLE IF NOT EXISTS proc.bp_policy_extraction_run (
    run_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('extract','fix')),
    status        TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
    request       JSONB NOT NULL,                -- extract: {"documents":[{"documentId","version"}]}; fix: {"policyKey","baseVersion","form"}
    owner         TEXT,                          -- process that claimed it
    heartbeat_at  TIMESTAMPTZ,
    counts        JSONB NOT NULL DEFAULT '{}',
    error         TEXT,
    started_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_extraction_run_status ON proc.bp_policy_extraction_run (status);

CREATE TABLE IF NOT EXISTS proc.bp_policy_extraction_item (
    run_id        BIGINT NOT NULL REFERENCES proc.bp_policy_extraction_run (run_id),
    seq           INTEGER NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('policy','not_enforceable','proposed_retire','fix','error','note')),
    document_id   BIGINT,
    document_version INTEGER,
    reference     TEXT,
    payload       JSONB NOT NULL,
    policy_key    TEXT,
    decision      TEXT CHECK (decision IS NULL OR decision IN ('new','changed','unchanged','proposed_retire')),
    saved_version INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);

ALTER TABLE proc.bp_agent_policy
    ADD COLUMN IF NOT EXISTS source_document_id BIGINT REFERENCES proc.bp_policy_document (document_id),
    ADD COLUMN IF NOT EXISTS source_reference TEXT,
    ADD COLUMN IF NOT EXISTS source_split TEXT;   -- distinguishes tiered policies from one clause (e.g. the outcome)
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_source ON proc.bp_agent_policy (source_document_id, source_reference);
COMMIT;
```

Rollback: drop the indexes, columns and tables in reverse order.

Tests:
- the tables and columns exist in both databases;
- the unique `(document_id, content_hash)` constraint holds (insert twice inside a rolled-back transaction);
- `proc.bp_policy` is untouched (copy stage 1's assertion).

Apply to both databases only after the tests fail red. Commit.

---

### Task 2: Documents — presign, register, version, parse

**Files:** `src/services/agent_policy/documents.py`, `tests/agent_policy/test_documents.py` (unit, with S3 and the DB faked), `tests/agent_policy/test_documents_live.py` (bp_testdb plus one real S3 round trip, cleaned up).

**Interfaces (produced):**
- `ACCEPTED = {".pdf", ".docx", ".txt", ".md"}`
- `intake_limits() -> tuple[int, int]` reuses `api.routers.documents._intake_limits`. Import it, or move it into a shared module and re-export it from the old place so existing callers don't change.
- `presign_uploads(files: list[dict], *, actor) -> list[dict]`
  - Input: `files` = `[{"name", "size", "contentType"}]`.
  - Validates the count, each size against the limits and each suffix against `ACCEPTED`. The whole request is refused with `ValueError(message)` on the first violation.
  - Returns `[{"uploadId": <uuid4>, "key": "agent-policy-documents/uploads/<uuid>/<safe name>", "url": <presigned PUT, 900 s>, "headers": {"Content-Type": ...}}]`.
  - The safe name: basename only, characters `[A-Za-z0-9._ -]`, others replaced with `_`, at most 120 characters.
- `register_uploads(conn, uploads: list[dict], *, actor) -> list[dict]`
  - Input: `uploads` = `[{"key", "name", "revisionOf": documentId | None}]`.
  - For each upload:
    - `head_object` checks it exists and its size is within the limit.
    - Download it and compute sha256 over the bytes.
    - Resolve the document. Use `revisionOf` if given. Otherwise, a document whose `match_name == normalise(name)` is a revision. Otherwise create a new document (title = name without the extension).
    - If `(document_id, content_hash)` already exists, return that version with `"duplicate": True` and write nothing.
    - Otherwise insert version = `latest_version + 1` (1 for a new document) and bump `latest_version`, in one transaction.
  - Returns `[{"documentId", "version", "title", "isRevision", "duplicate"}]`.
  - `normalise(name)`: lowercase, drop the extension, drop the trailing ` v2` / `(1)` / `_final` suffixes (`r"[\s_-]*(v\d+|\(\d+\)|final|draft|rev\d*)$"`, repeated), and collapse whitespace.
- `document_text(conn, document_id, version) -> str`
  - Returns `parsed_text` if it is set.
  - Otherwise download from S3. PDF/DOCX → write to a temporary file → `services.extraction.parser.parse(path).full_text`. TXT/MD → UTF-8 decode, with replacement.
  - Store `parsed_text` and `parsed_at`, then return the text.
  - Raise `DocumentUnreadable(reason)` on empty text.
- `list_documents(conn) -> list[dict]` returns each document with all its versions (no text).

**Required tests:**
- `test_same_bytes_same_version_new_bytes_new_version`
- suffix and size refusals
- the 21st file is refused
- name normalisation table: `"Refund Policy v2.docx"` → `"refund policy"`, `"refund_policy (1).pdf"` → `"refund policy"`
- `revisionOf` overrides name matching
- one live round trip on bp_testdb:
  - presign;
  - PUT a 1 KB `.txt` with `requests`;
  - register;
  - confirm `document_text` returns the content;
  - delete the S3 object;
  - check the S3 write permission first, and report BLOCKED if the PUT is refused.

---

### Task 3: Sections, chunking, and the model call

**Files:**
- `src/services/agent_policy/sections.py`
- `src/services/ollama_client.py`: add `use_load_options: bool = False` to `_ollama_generate` and `ollama_generate`. When it is True, merge `load_options()` into `options`, after the temperature and num_predict keys.
- Tests: `tests/agent_policy/test_sections.py`, and an addition to the existing ollama_client tests asserting the default request body is unchanged.

**Interfaces:**
- `split_sections(text) -> list[Section]`
  - `Section = {"reference": str | None, "heading": str, "text": str, "start": int}`.
  - A section starts at a markdown heading line (`^#{1,6} `) or a numbered clause line (`^\s*(\d+(\.\d+)*)[.)]?\s+\S`).
  - The reference is the clause number if there is one, otherwise the heading text.
  - Text before the first marker becomes `{"reference": None, "heading": "Preamble"}`.
  - Every character of the input belongs to exactly one section, and the sections concatenate back to the input.
- `chunk_sections(sections, *, max_chars=9000) -> list[list[Section]]`
  - Packs consecutive sections greedily.
  - A single section longer than `max_chars` is split on paragraph boundaries into parts with the same reference, suffixed ` (part n)`.
  - No section is dropped or duplicated.
  - 9000 characters is about 2.3k tokens, which leaves room for the prompt, the registry and the output within 12,288.
- `diff_sections(old_text, new_text) -> list[dict]`
  - Aligns by reference, falling back to the heading.
  - Returns `[{"reference", "status": "unchanged"|"changed"|"added"|"removed", "before", "after"}]`.
  - Used for the before/after view.

**Required tests:**
- `test_every_section_lands_in_exactly_one_chunk`, a property-style check over three fixtures plus a synthetic 60-section text;
- the concatenation identity;
- numbered clauses `1.`, `1.1`, `4.2)`;
- the oversize split;
- `diff_sections` on a pair where 1.1 changed, 1.2 was removed and 1.3 was added;
- the `ollama_client` default request body is byte-identical to before when `use_load_options` is not passed;
- with `use_load_options=True`, the body contains `num_ctx == load_options()["num_ctx"]`.

---

### Task 4: Extraction schema, prompt, and converter

**Files:**
- `src/services/agent_policy/extraction_schema.py`
- `src/services/agent_policy/converter.py`
- `src/services/agent_policy/extractor.py`
- `deploy/sql/2026-10-09_agent_policy_extraction_prompts.sql` (+ rollback)
- Tests: `tests/agent_policy/test_converter.py`, `tests/agent_policy/test_extractor.py` (the model is stubbed)

**The schema** (pydantic v2; flat; no unions; every optional field is `Optional[...] = None`, and lists default to empty):

```python
OUTCOMES = Literal["approve", "block", "notify"]
OPS = Literal["gt", "gte", "lt", "lte", "eq", "ne", "in", "not_in", "exists"]
RESULTS = Literal["approve", "block", "notify", "none"]
CHECKPOINTS = Literal["tool.call.before", "message.send.before", "data.egress.before", "record.write.before"]

class Rule(BaseModel):
    field: str; op: OPS
    value_number: Optional[float] = None; value_text: Optional[str] = None; value_list: List[str] = []
class ExampleValue(BaseModel):
    field: str; value_number: Optional[float] = None; value_text: Optional[str] = None
class Example(BaseModel):
    values: List[ExampleValue]; expected: RESULTS
class InputSpec(BaseModel):
    name: str; field: str; type: Literal["string", "number", "boolean", "date", "list"]
    is_amount: bool = False; unit: Optional[str] = None
    source: Literal["action", "lookup", "total"] = "action"
    show_approver: bool = True; sensitive: bool = False
class Missing(BaseModel):
    name: str; reason: str
class ProposedPolicy(BaseModel):
    name: str; category: str; business_area: str; sub_area: str
    situation: str
    match: Literal["all", "any"]; rules: List[Rule]
    outcome: OUTCOMES; outcome_phrase: str
    deciders: List[str] = []; notify: List[str] = []
    reference: str; excerpt: str
    examples: List[Example]
    checkpoint: CHECKPOINTS; action_tools: List[str]; action_plain: str
    time_window_from: Optional[str] = None; time_window_to: Optional[str] = None; time_zone: Optional[str] = None
    currency: Optional[str] = None; amounts_include_tax: Optional[bool] = None
    inputs: List[InputSpec]
    missing_inputs: List[Missing] = []; unknown_names: List[str] = []
    reason_code: str; message_for_agent: str; message_for_person: Optional[str] = None
    owner: Optional[str] = None
class NotEnforceable(BaseModel):
    reference: str; excerpt: str; reason: str
class ChunkResult(BaseModel):
    policies: List[ProposedPolicy]; not_enforceable: List[NotEnforceable]
```

**The prompts:** two governed rows, `agent_policy_extract` and `agent_policy_fix`. The extraction prompt must say every item the brief's §5 lists. In particular:
- business area and sub-area come from where the policy comes from, never from what it applies to, and must be chosen from the supplied taxonomy;
- one situation and one outcome per policy, with tiered rules split;
- use only registry names, and list anything else in `unknown_names` or `missing_inputs`, never invented;
- the excerpt is copied word for word;
- give 3–6 examples, with inside, outside and on-the-boundary cases for each number, and in and out of each list;
- never set confirmation or Active;
- a guideline that can't be checked when an agent acts goes in `not_enforceable` with a reason.

Placeholders, replaced by name: `{taxonomy}`, `{registry}`, `{document_title}`, `{document_version}`, `{sections}`.
- `{registry}` is the compact listing produced by `extractor.registry_digest(registry)`: each checkpoint with its plain text and status, then each live checkpoint's actions and inputs, as `field (type, from action)`.
- Planned checkpoints are listed as "not checked yet".
- Insert into both databases.

**The fix prompt** gets:
- the policy's excerpt and its current situation, condition and examples;
- the flipped examples as constraints ("these inputs must give <result>");
- the same registry.

It returns one `ProposedPolicy`.

**`extractor.py`:**
- `registry_digest(registry) -> str`
- `load_prompt(name) -> str`: reads through `PromptEngine(connection_factory=get_conn)`, matching `promptName == name`. It raises `PromptUnavailable` if the row is missing.
- `extract_chunk(sections, *, document, taxonomy, registry, call=ollama_generate) -> ChunkResult`
  - Fills the prompt.
  - Calls with `format=ChunkResult.model_json_schema()`, `think=False`, `temperature=0`, `num_predict=8192`, `background=True`, `use_load_options=True`.
  - Parses with `model_validate_json`.
  - `None` or invalid JSON → raise `ExtractionError("the model did not return a usable answer")`.
- `fix_policy(form, flipped, *, registry, taxonomy, call=ollama_generate) -> ProposedPolicy`: same pattern.

**`converter.to_form(p: ProposedPolicy, *, document_title, document_version, registry, taxonomy) -> dict`** produces stage 1's form state exactly (see `tests/agent_policy/fixtures.py::FORM_EXAMPLE` for the keys).

Mapping:
- `condition = {p.match: [leaf...]}` where each `leaf = {"field", "op", "value"}`:
  - `value` is `value_list` for `in`/`not_in`, else `value_number` if not None, else `value_text`;
  - `exists` has no value.
  - A tool name in `action_tools` adds a leading `{"field": "tool.name", "op": "in", "value": action_tools}` leaf when no rule already names `tool.name`.
- `hidden.inputs`: map `from` as `action` → `"action"`, `lookup` → `"lookup:unregistered"`, `total` → `"total:unregistered"`.
  - Lookups and totals are not registered in this release, so these show as "Can't be enforced yet".
  - Always ensure `tool.name` and `agent.reason` are present as inputs; add them if they are absent.
- `examples`: `input = {v.field: (v.value_number if not None else v.value_text)}`, `agentExpected = expected`, `flipped = False`.
- **Business area:**
  - If `business_area` is not a taxonomy area, set `businessArea = None` and add the note `"The agent proposed business area 'X', which is not in the taxonomy."` to a new `hidden.agentNotes` list.
  - Do the same for a sub-area that is not in that area's list.
- `source = {"document": document_title, "documentVersion": document_version, "reference": p.reference, "excerpt": p.excerpt}`.
- `outcomeBecause = p.outcome_phrase`; `checked = None`; `hidden.setBy = "extraction_agent"`; `responseTime = None`; `limit = {"on": False, "text": ""}`.
- `hidden.unknownNames` = `p.unknown_names` ∪ (rule fields with no registry input row at the checkpoint) ∪ (action tools the registry doesn't know). The code verifies; it never trusts the model.
- `hidden.missingInputs = p.missing_inputs`.
- `hidden.timeWindow = {"from", "to", "timeZone"}` when either end is set, else None.
- `hidden.units = {"currency": p.currency, "convertOther": "rate_on_action_date", "amountsIncludeTax": p.amounts_include_tax}`.

**Required tests:**
- `test_invented_tool_name_goes_to_unknown_names`;
- a tiered clause (two ProposedPolicy objects, same reference, outcomes approve and block) converts to two forms with the same `source.reference`;
- an off-taxonomy area becomes None plus a note;
- a lookup input is marked unregistered and `readiness.how_enforced` reports "Can't be enforced yet";
- the converted form passes `compile_policy` and `contract.validate` produces only registry problems on a known-bad registry;
- `extract_chunk` with a stub call that returns invalid JSON raises `ExtractionError`;
- the call is made with `think=False`, `use_load_options=True` and a schema with no `oneOf`/`anyOf` anywhere (walk it).

---

### Task 5: Run store and runner

**Files:** `src/services/agent_policy/run_store.py`, `src/services/agent_policy/run_runner.py`, and `tests/agent_policy/test_run_store_live.py` (bp_testdb).

Copy the shape of `src/services/rga/job_store.py` and `job_runner.py` (read them first):

- `create(conn, *, kind, request, actor) -> dict`
- `claim(conn, run_id, owner) -> bool`: queued → running.
- `beat(conn, run_id, owner)`
- `append_item(conn, run_id, *, kind, payload, document_id=None, document_version=None, reference=None, policy_key=None, decision=None, saved_version=None) -> int`: assigns the next `seq` atomically (`SELECT COALESCE(MAX(seq),0)+1 ... FOR UPDATE` on the run row).
- `finish(conn, run_id, status, *, counts, error=None)`
- `get(conn, run_id, *, after_seq=0) -> dict`: returns `{run..., "items": [...]}`.
  - Heals first: a `running` run whose `heartbeat_at` is older than 120 s becomes `failed`, with error "The server restarted while this run was working. Start it again; the policies already listed were saved."
- `list_recent(conn, limit=20)`
- Runner: a single-worker `ThreadPoolExecutor`.
  - `submit(run_id, work)` starts a heartbeat thread that beats every 30 s.
  - `work(conn, run, emit)` does the run.
  - Any exception means `finish(..., "failed", error=<one-line message>)`. The runner never raises.

**Required tests:** `test_stale_run_heals_to_failed`, seq ordering under two concurrent `append_item` calls (two connections), and `get(after_seq=n)` returns only newer items.

---

### Task 6: Matching, revision decisions, and the extraction run

**Files:** `src/services/agent_policy/matching.py`, `src/services/agent_policy/extraction_run.py`, `tests/agent_policy/test_matching.py`, and `tests/agent_policy/test_extraction_run.py` (repo live on bp_testdb; the model stubbed with canned `ChunkResult`s).

**`matching.py` (pure):**
- `split_key(form) -> str`: `outcome` plus a sorted digest of the condition's numeric boundaries. This is what distinguishes the tiers of one clause (for example "approve|gt:500" and "block|gt:10000").
- `match(existing: list[dict], proposed: list[dict]) -> list[dict]`
  - `existing` holds the document's current policies: `{policyKey, reference, split, form}`, where the form is the latest version's.
  - `proposed` holds the converted forms in document order.
  - Matching order:
    - exact `(reference, split_key)`;
    - else same reference, and the only unmatched existing policy with the same outcome;
    - else excerpt similarity of at least 0.85 (`difflib.SequenceMatcher` over normalised text), same outcome, unmatched.
  - Returns, per proposed form, `{"policyKey": key | None, "decision": "new" | "changed" | "unchanged"}`, plus `{"policyKey", "decision": "proposed_retire"}` for each existing policy left unmatched.
- `substantively_equal(old_form, new_form) -> bool`: compares `situation`, `outcome`, `hidden.condition`, `deciders`, `notify`, `source.excerpt`, the `hidden.inputs` fields, `hidden.checkpoint`, and the example inputs. It ignores `examples[].agentExpected`, `checked`, `changeNote`, owner and dates.

**`extraction_run.run_extract(conn, run, emit)`:**
1. Load the registry, the settings, and the taxonomy (from `bp_business_area`).
2. For each requested document version:
   - `text = documents.document_text(...)`. If the document is unreadable, emit `error` and continue.
   - `sections = split_sections(text)`; `chunks = chunk_sections(sections)`.
   - For each chunk:
     - Call `extract_chunk`. On failure, emit an `error` item naming the chunk's references, then continue.
     - Convert each proposed policy.
     - Each excerpt must be grounded in the document with `is_quote_grounded(excerpt, text, min_words=4)`. If it isn't, keep the policy but add the note `"The excerpt was not found word for word in the document."`. Confidence will also show it.
     - Emit `not_enforceable` items as they arrive.
3. Per document, after all its chunks: match the proposed forms against existing policies with `source_document_id = document_id`. For each result:
   - `new`: `create_draft`, then set `source_document_id`, `source_reference` and `source_split` on the new row. Emit `policy` with `decision="new"`, the `policyKey` and `saved_version=1`.
   - `changed`: `save_version(intent="draft", base_version=latest, change_note="Re-extracted from <title> v<n>, section <ref>.")`, passing the document text so confidence is computed. Emit `policy` with `decision="changed"`. The live version stays live (stage-1 behaviour).
   - `unchanged`: write nothing, and emit `policy` with `decision="unchanged"`.
   - `proposed_retire`: write nothing, and emit `proposed_retire` with the policy key and the old excerpt.
   - Every save uses `actor = run.started_by`. Its form has `checked=None` and `setBy="extraction_agent"`.
4. Emit items as soon as each is decided, so the list fills during the run.
5. Counts: `{"documents", "chunks", "policies", "new", "changed", "unchanged", "proposedRetire", "notEnforceable", "errors"}`.

`create_draft` needs a way to set the source columns. Add an optional `source: dict | None` parameter to `create_draft` (and update stage-1 callers, which pass nothing). That keeps the write inside the same transaction.

**`run_fix(conn, run, emit)`:**
- Load the policy's base version.
- Call `fix_policy` with the run request's flipped examples.
- Convert the result.
- Emit one `fix` item: `{"proposed": <form fields situation, hidden, examples, outcome, deciders, notify, messages>, "basedOn": baseVersion}`.
- It never saves. The reviewer accepts it in the form, and the normal save follows.

**Required tests:**
- `test_reextracting_unchanged_document_writes_nothing`;
- `test_bad_chunk_is_an_error_item_and_run_continues`;
- a tiered clause produces two `new` policies with the same `source_reference` and different `source_split`;
- a revised document: section 1.1 changed → `changed` with a new draft, while the live version stays live; 1.2 removed → `proposed_retire`; 1.3 added → `new`; nothing is retired;
- a fix run emits one `fix` item and writes no version;
- every saved version has `checked` None and `saved_as` draft.

---

### Task 7: Endpoints and gateway routes

**Backend** (`src/api/routers/agent_policies.py`, same `gateway_principal` and `_require`; every write is audited):

| Method + path | Role | Body | Returns |
|---|---|---|---|
| POST `/agent-policies/documents/upload-urls` | Buyer | `{files:[{name,size,contentType}]}` | `{uploads:[...]}`; 422 `{problems:[{field:"files",code,message}]}` when refused |
| POST `/agent-policies/documents` | Buyer | `{uploads:[{key,name,revisionOf?}]}` | `{documents:[...]}` |
| GET `/agent-policies/documents` | Viewer | — | `{documents:[...]}` |
| GET `/agent-policies/documents/{id}/compare?from=v&to=w` | Viewer | — | `{sections: diff_sections(...)}` |
| POST `/agent-policies/extraction-runs` | Buyer | `{documents:[{documentId,version}]}` | 202 `{runId}` |
| GET `/agent-policies/extraction-runs/{runId}?afterSeq=n` | Viewer | — | run + items |
| GET `/agent-policies/extraction-runs` | Viewer | — | recent runs |
| POST `/agent-policies/{key}/agent-fix` | Buyer | `{baseVersion, flipped:[{input, expects}]}` | 202 `{runId}` |

- Register `GET /agent-policies/documents*` and `/extraction-runs*` before the `/{key}` route.
- `upload-urls` and `documents` refuse a key that is not under `agent-policy-documents/uploads/`.
- **Gateway, local bypass group (ruling):** in `src/auth/guards/cognito.guard.ts`, the local bypass (which already needs AUTH_BYPASS, IS_OFFLINE, non-production NODE_ENV and no Lambda) reads its groups from `AUTH_BYPASS_GROUPS` (a comma list). The default stays `['Admin']`, so nothing changes for anyone who doesn't set it. Then set `AUTH_BYPASS_GROUPS=PROCWISE_ADMIN` for the local demo, so gateway writes work locally. Add a spec for the default and for the override. Nothing changes outside bypass mode.
- **Gateway:** add the same routes to `agent-policy.controller.ts`, with the same roles, `need()`, and an id allow-list:
  - `documentId` and `runId`: `^[0-9]{1,18}$`;
  - `from`/`to`: `^[0-9]{1,6}$`.
  - The 202 status is passed through.
  - Add `agent-policy.yml` entries.

**Tests:** router tests for each route (roles, the 422 shape, the key-prefix refusal, the audit row on writes), gateway specs for each route (role, allow-list, forwarding), and the router-authentication guard test extended.

---

### Task 8: Screens — upload, live run list, Not enforceable, Proposed retire, before/after, Ask the agent to fix it

**Pure modules (vitest):**
- `agentPolicy/upload.js`:
  - `planUploads(files, existingDocuments) -> [{file, suffixOk, sizeOk, revisionOf, title}]`, which uses the same `normalise` rule as the backend (table-tested on the same examples);
  - `summariseRun(run) -> {status label, counts line}`;
  - `mergeItems(prev, newItems)`, which dedupes by seq.
- `agentPolicy/runView.js`:
  - `groupRunItems(items)` → by document, then reference, keeping tiered policies together;
  - the labels "New", "Changed (new draft)", "Unchanged", "Proposed retire", "Not enforceable", "Could not read".

**`engine.js` (new `ap*` functions only):**
- **"Upload documents"** button on the Agent policies tab, Buyer and above:
  - multi-file picker (`.pdf,.docx,.txt,.md`);
  - shows the plan, with each file marked "new document" or "revision of <title>" and the revision switchable;
  - then `upload-urls`, PUT each file to S3 with `fetch`. Use `fetch`, not axios, so no auth headers are added, and send the given Content-Type;
  - then `documents`, then `extraction-runs`.
- **Run panel:** polls `GET /extraction-runs/{id}?afterSeq=` every 2 s until `done`/`failed`. Items appear as they arrive, grouped by document and section. Each policy item has an "Open" link. Not-enforceable items show their reason. Proposed-retire items show "Proposed retire" and a "Retire…" link that opens the policy's normal two-step retire.
- **"Compare versions"** for a document with more than one version: side-by-side before/after by section, using `compare`, with changed, added and removed sections marked.
- **Form:**
  - "Ask the agent to fix it" is enabled when an example is flipped.
  - It posts `agent-fix` with the flipped examples, then polls. On a `fix` item it shows a "Proposed change" box with the new situation and recomputed examples (via preview), plus Accept and Reject.
  - Accept applies the proposal to the form and clears the confirmation (`editClearsConfirmation`). Nothing is saved until the owner saves.
- **List:** show `hidden.agentNotes` and the "Check before this goes Active" confidence notice. Both exist; check the agent's notes are visible.
- Stop polling when the panel is closed.

**Tests:**
- vitest for the pure modules;
- contract and behaviour tests in the existing no-DOM harness:
  - upload sends no Authorization header to S3;
  - polling stops on done and on close;
  - Accept clears `checked` and does not save;
  - Proposed retire never calls retire without the two-step confirm.

---

### Task 9: Acceptance on the real model, and the live demonstration

**Files:**
- `tests/agent_policy/fixtures_docs/finance_payments_policy.md`: Finance; section 1.1 has "Refunds or credits above $500 need approval from the Finance Manager. Refunds above $10,000 are not allowed.", which is tiered.
- `tests/agent_policy/fixtures_docs/customer_refund_standard.md`: Customer operations; one notify rule and one guideline.
- `tests/agent_policy/fixtures_docs/security_data_handling.md`: Security; one block rule on data leaving the system, plus one "should" guideline.
- `tests/agent_policy/test_acceptance_live_model.py`: skipped unless `AGENT_POLICY_LIVE_MODEL=1` and Ollama answers.

The documents are short (under 1 page each) and written for this test. The amounts use real tool names from the registry where possible. Where the registry has no such tool, that is fine: the policy will carry `unknownNames`, and that is the behaviour under test.

**Acceptance (brief §8):**
1. The three documents from three business areas go through **one run**. Every policy has a business area (or the agent note explaining why not), a sub-area, a source and extraction confidence.
2. The tiered clause produces two policies grouped under one source, with the same `source_reference` and different `source_split`. There is no conflict flag; conflicts are stage 4.
- At least one `not_enforceable` item exists, with a reason.
- Re-running the same versions creates nothing new (all `unchanged`).
- Uploading a revised Finance document (1.1 amount changed, 1.2 removed) gives `changed` + `proposed_retire`, and the live version (activate one first) stays live.

**Live demonstration:** run the stack as stage 1's Task 11 did (backend :8010, gateway :3011, UI :3010, bp_testdb, and stop only your own processes). Then:
- upload the three files through the gateway;
- watch the run fill;
- open a policy;
- flip an example and use Ask the agent to fix it;
- check headless Chrome for console errors.

Record the results in `specs/2026-10-08-agent-policy-governance-stage2-verification.md`, with timings per chunk, and commit.
