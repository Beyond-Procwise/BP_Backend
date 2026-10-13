# Prompt engine, phase 1 — design note

13 Oct 2026. Status: **building. Milestones 2 (registry, import, routes, policies, agent
instructions) and 3 (typed templates and fragments) are done. Milestone 4 (render, manifest
store, retention) is done and its tables are on both databases. Milestone 5 (client, retry,
the demand-intake agent's two paths) is built; its typed prompt awaits promotion.**
Built on the Step 0 reuse audit of the same date and the owner's seven answers.

## 1. What is being built

A render-only component. It stores, versions, composes, validates and renders
agent instructions and returns a manifest that proves what it rendered. It
never decides policy, calls a model, runs a tool, retrieves documents, holds
run state, truncates or retries.

## 2. The owner's rulings (13 Oct 2026)

| # | Question | Ruling | Effect on this design |
|---|---|---|---|
| 1 | Gemini / OpenAI acceptance test | Wait for the parked model-provider decision | That one criterion is deferred. Phase 1 is proven on AgentNick. `tested_models` is stored and returned but lists only what has been run. |
| 2 | Environments | One pointer per database. Dev / staging / prod is a separate activity | One pointer named `live` per prompt. Pointers are rows with a name, so adding `dev` / `staging` / `prod` later is data, not a schema change. |
| 3 | Who approves promotion | Approver or above, and not the author | Same rule created agents already follow. |
| 4 | Retention of rendered text | Yes. **7 days**, held as a policy with the same structure as the other policies | A `limit` row in `proc.bp_policy`, read through `governed_limits`. No number in code. |
| 5 | Latency / cache | Acceptable. It should retry until complete | See section 7. |
| 6 | OpenTelemetry | Yes, on the same principle as the embeddings architecture. Reading confirmed "for now": our own infrastructure, nothing to an outside vendor, checked at start-up, health monitored | See section 8. Revisit if the principle changes. |
| 7 | Data residency | None today. It should be a policy | A policy row is created. The prompt engine does not evaluate it (section 9). |

## 3. Where it lives

A module inside the backend, `src/services/prompt_engine/`, with its own
router and an in-process client. Not a separately deployed service.

- Prompts are rows, so a prompt change needs no deploy, and an orchestrator
  deploy changes no prompt. That is the independent-shipping principle.
- The boundary is the interface: the orchestrator and agents may import only
  the client. A test fails if anything else imports the engine's internals or
  reads the prompt tables directly.
- Moving it to its own service later means putting HTTP behind the same
  client. Nothing else changes.

The old `src/orchestration/prompt_engine.py` is replaced. Its unused
query-analysis feature and its model and GPU imports are dropped.

## 4. Data model

All tables in schema `proc`, prefix `bp_`, applied to **both** databases.

**`bp_prompt` (exists, NOT altered).** It stays as the table the older readers
use. A promotion writes the live wording back into the matching row, so the
ten places that still read it directly go on working until each is migrated.
(As first drafted this note extended the table; see the decision log for why
that changed.)

**`bp_prompt_registry` (new).** One row per prompt or fragment, never deleted.

| Column | Meaning |
|---|---|
| `prompt_key` | Stable name, unique. Everything refers to a prompt by this, never by number: numbers differ between the two databases. For an imported prompt it is the name it already had. |
| `kind` | `prompt` or `fragment`; cannot be changed. Routing, rewrite, repair and handoff prompts are ordinary prompts. |
| `owner` | The person or business area answerable for it. |

**`bp_prompt_version` (new, immutable).** A trigger refuses update and delete,
copied from `bp_agent_policy_version`. Tied to `bp_prompt_registry`.

| Column | Meaning |
|---|---|
| `prompt_key`, `version` | Primary key. |
| `syntax` | `legacy` (imported as it was) or `v1` (typed template). |
| `template` | The text. |
| `variables` | Declared blanks: name, type (string, enum, number, boolean, object with schema), required. |
| `fragment_refs` | Every `id@version` it uses. Indexed, so "who uses this fragment" is one query (needed in phase 2). |
| `settings` | Output schema, examples, default parameters, tested models. |
| `change_note`, `author`, `created_at`, `content_sha256` | The record of the save. |

**`bp_prompt_pointer_move` (new, append-only).** Every promotion and rollback
is a row: prompt, pointer name, from version, to version, who approved, when,
note. The current pointer is the latest row. This one table is both the
pointer and its full history, and the approval record.

**`bp_prompt_render` (new).** The manifest store. Primary key is the sha256
of the rendered output. Holds the manifest and the rendered text **once**.

**`bp_prompt_render_use` (new, append-only).** One row per render call:
run id, step id, attempt, decision id, hash, time. This is what answers
"show me every render in this run".

## 5. Templates

- A blank is `{{name}}`. A fragment is `{{> fragment-key@3}}`. That is the
  whole language: no conditions, no loops, no expressions. A template cannot
  branch on anything, and cannot run code. The unrestricted template library
  used today by the email agent and orchestrator is not used here.
- **On save:** every blank is declared, every declaration is used, every
  fragment reference exists at that version. Otherwise the save is refused.
- **On render:** a missing, extra or wrongly typed value is refused with
  `VAR_MISSING`, `VAR_UNDECLARED` or `VAR_TYPE`. No text is returned.
- **Imported prompts** keep their text byte for byte as version 1 with
  `syntax = legacy`, and are served exactly as today. Each becomes a typed
  `v1` template, as a new version, when its call site is migrated.

## 6. Render

Request and response are as in the build prompt's contract.

1. Resolve the pointer to a version (`POINTER_UNRESOLVED`,
   `VERSION_NOT_FOUND`).
2. Check every obligation against the vocabulary: `append:<fragment>@<v>`,
   `include_pack:<pack>`, `redact:<path>`. Anything else is
   `OBLIGATION_UNKNOWN` and nothing is rendered.
3. Apply `redact` first: the value is removed before any text is built. A
   blank that pointed at it renders as `[redacted]`. The manifest records the
   path, never the value.
4. Validate variables, resolve fragments (`FRAGMENT_MISSING`), fill, append.
5. Add descriptions for the tools in `allowed_tools`, and only those, from
   the existing `Tool` definitions in `tool_runtime`.
6. Count tokens. Report the count; never shorten anything.
7. Hash the canonical form of the output, store manifest and text under that
   hash if new, record the use, return.

No clock, random value or ordering accident enters the text, so the same
request gives the same bytes and the same hash.

`experiment_key` is accepted and ignored in phase 1.

## 7. Client, cache and "retry until complete"

The engine never retries. The **client** the orchestrator calls does:

- A failure that can pass (the database not answering) is retried with a
  growing wait until it succeeds or the step's own deadline, which the
  orchestrator owns, runs out. It never falls back to built-in wording.
- A failure that cannot pass (a missing variable, an unknown obligation) is
  returned at once. Retrying it would never complete.
- The cache is keyed on the exact request and the pointer's current version,
  so it can only ever return the render that would have been produced anyway.
  After a rollback the next render uses the rolled-back version.

Integration point: once per step, where instructions are assembled today
(`BaseAgent.governance_brief` / `resolve_prompt`, and the orchestrator's
prompt injection).

## 8. Telemetry

- One OpenTelemetry span per render carrying run id, step id, attempt,
  decision id, prompt key, version, hash, fragment versions, token count and
  latency. It joins the orchestrator's trace and never starts its own.
- No prompt text, variable value or context content in any span or log line.
  A test renders with marker values and scans everything emitted for them.
- The orchestrator has no step id or attempt number today. Adding them is
  part of milestone 5.
- Following the embeddings architecture as I read it: the trace collector and
  store run on our own infrastructure, nothing is sent to an outside vendor,
  the service checks it can reach them at start-up, and their health is a
  monitored signal.

## 9. Governance rows created (both databases)

| Row | Purpose |
|---|---|
| Permissions `prompt.read`, `prompt.author`, `prompt.promote` (milestone 2); `prompt.render`, `prompt.manifest_read` (milestone 4) | Separate rights, each with an explicit policy row stating the role. |
| `PromptEnginePolicy`: `manifest_retention_days` = 7 | How long rendered text is kept. A missing value refuses to sweep rather than guessing. |
| `PromptEnginePolicy`: `wording_read_passthrough` = true | Whether a prompt's wording reaches its own screens unaltered. |
| `PromptEnginePolicy`: `render_retry_seconds` = 30 | How long a caller keeps retrying a render that can pass before the step is refused. |
| Policy `data_residency` | States where prompt and manifest data may be stored. Today: no restriction. Enforcing it is the platform's job, not the prompt engine's. |

## 10. Decision log

Every decision the build rests on, in the order it was made. Add to this; do
not rewrite it.

| Date | Decision | Made by |
|---|---|---|
| 13 Oct | The pasted target design replaces the earlier, simpler proposal. Policy decides, the orchestrator sequences and enforces, the prompt engine renders. | Owner |
| 13 Oct | Prompts are written by users and are also created when an agent is created; both land in the same registry, and an agent uses prompts from that list. | Owner |
| 13 Oct | Prompts are structured and managed the way policies are. | Owner |
| 13 Oct | Gemini / OpenAI acceptance test waits on the parked model-provider decision. | Owner |
| 13 Oct | One pointer per database. Dev / staging / prod is a separate activity. | Owner |
| 13 Oct | Promotion needs Approver or above, and not the author. | Owner |
| 13 Oct | Rendered text is kept 7 days, as a policy row shaped like the other policies. | Owner |
| 13 Oct | p95 under 50 ms in-process; cache keyed on the exact request; the client retries until complete. | Owner |
| 13 Oct | OpenTelemetry approved; self-hosted, nothing to an outside vendor ("yes for now"). | Owner |
| 13 Oct | No data-residency rule today; one is created as a policy. | Owner |
| 13 Oct | A module inside the backend, not a separate service (section 3). | Builder, in the note the owner reviewed |
| 13 Oct | Own minimal template language; imported prompts kept byte for byte as `legacy` version 1 (section 5). | Builder, same |
| 13 Oct | "Retry until complete" is read as: retry failures that can pass until the step deadline; return at once those that cannot; never fall back to built-in wording (section 7). | Builder's reading |
| 13 Oct | Retention confirmed at 7 days; telemetry reading confirmed "yes for now"; owner said go on milestone 2. | Owner |
| 13 Oct | The older `bp_prompt` table is left unaltered and a new `bp_prompt_registry` table holds one row per prompt. Reason found while building: existing code retires a prompt row and inserts a replacement, so that table cannot give a prompt one stable row to hang versions on. | Builder, during milestone 2 |
| 13 Oct | An imported prompt's key is the name it already has, so call sites can be moved over without renaming anything. The import refuses a name two active rows share. | Builder, during milestone 2 |
| 13 Oct | The import writes nothing to the older table. A later promotion steps that table's own version number on by one, never back. | Builder, during milestone 2 |
| 13 Oct | A rollback may be approved by anyone with the right to promote, the author included, because the version returned to was approved by a second person when it first went into use. | Builder, during milestone 2 — owner to confirm |
| 13 Oct | Registry tests run only against a throwaway database: its rows cannot be deleted, so a shared database must never be a test target. | Builder, during milestone 2 |
| 13 Oct | Apply the registry tables and run the import on both databases: approved, and done (33 prompts on bp_testdb, 19 on bp_sqldb; the existing prompt table verified unchanged). | Owner |
| 13 Oct | Rollback may be approved by anyone with the right to promote, the author included: confirmed. | Owner |
| 13 Oct | A prompt's wording may be shown unaltered on its own screens (not scrubbed by the output-safety layer) provided every read is logged with when it happened, it is internal, and it is a policy decision. | Owner |
| 13 Oct | How that ruling is met: the two wording reads ask the gate for `prompt.read`, which records every attempt, allowed or refused, before answering; a stated role is required; and `PromptEnginePolicy.wording_read_passthrough` switches the pass-through. Missing, unreadable or false means scrubbed. | Builder |
| 13 Oct | Roles set in the policy rows: read wording = Buyer or above; save a version = Buyer or above; promote or roll back = Approver or above. These are rows, so they can be changed without a release. | Builder — owner to confirm |
| 13 Oct | An agent's own instructions go into the registry when the agent is created or edited, and are retired when it is deleted. They are recorded as in use at once (`agent_saved`), not promoted here, because the agent's own draft-and-publish by a second person is what governs their release. | Builder |
| 13 Oct | A registry failure does not fail an agent's save while agents still read instructions from the older table; it is logged as an error and the import script reconciles. This reverses to "fail the save" when rendering moves to the registry (milestone 5). | Builder |
| 13 Oct | A prompt can be retired: nothing is in use, its versions and history stay, and a later promotion brings it back. | Builder |
| 13 Oct | Over HTTP, refusal codes are sent in lower case (`version_not_found`). Error answers are always scrubbed, and the output-safety layer withholds the upper-case form as if it were an environment variable. In-process callers get the contract's exact codes. | Builder |
| 13 Oct | Milestone 3. A fragment cannot use another fragment, and a prompt cannot be used as a fragment. Keeps "which wording does this prompt contain" a one-step answer. | Builder |
| 13 Oct | A fragment's own blanks must be declared, with the same type, by every prompt that uses it; checked when the prompt is saved. | Builder |
| 13 Oct | Which fragments a version uses is read from its template, not taken from the caller. | Builder |
| 13 Oct | A value put into a blank is never read again as template, so text from a supplier or a document cannot become instructions. | Builder |
| 13 Oct | A refusal names the variable and the expected type, never the value: values are customer data and refusals reach logs. | Builder |
| 13 Oct | A blank declared `required: false` and left out renders as nothing. | Builder |
| 13 Oct | Milestone 4. Redaction is applied before any text is built, whatever position the obligation was given in; the masked value shows as `[redacted]`, and a redacted object field is masked inside the object too. | Builder |
| 13 Oct | A refused obligation is reported by its position in the list, not its text: an obligation is another system's output and a malformed one may carry anything. | Builder |
| 13 Oct | Tool descriptions are wording, so they are versioned with the prompt (`settings.tool_descriptions`). The engine returns them for allowed tools only; a tool the prompt has no wording for is left to the orchestrator's own description. This replaces "taken from tool_runtime" in the first draft, which would have made the engine depend on the tool layer. | Builder |
| 13 Oct | The trace (run, step, attempt, decision) is recorded with each use but is not part of the hash: the same wording has the same hash in every run. | Builder |
| 13 Oct | Retention is counted from a stored render's LAST use, not its first, and deletes only the text. The record of which run used which hash, prompt and version is kept. | Builder — confirmed by the owner's ruling below |
| 13 Oct | If the render cannot be stored, the render fails. | Builder |
| 13 Oct | Token counts use the `cl100k_base` tokenizer, and the manifest says how the count was made; it is an approximation for models that tokenize differently. | Builder |
| 13 Oct | No HTTP render endpoint in phase 1: the orchestrator calls the engine in-process, and reading stored text is by script on the server. Fewer doors onto text that carries customer values. | Builder — confirmed by the owner's ruling below |
| 13 Oct | Apply the two store tables to both databases: approved. | Owner |
| 13 Oct | Rendered text goes after 7 days; the metadata is kept. | Owner |
| 13 Oct | How that is met: after 7 days unused, only the text is removed. The manifest (prompt version, fragment versions, obligations, token count, hash) and the record of which run used it stay; neither holds wording or values. Rendering the same text again stores it again. The 7 days run from the text's last use. | Builder |
| 13 Oct | No web route for rendering or for reading stored text, for now. | Owner |
| 13 Oct | The hash covers what the text was made from (prompt, version, fragment versions, obligations) as well as the text. Found by a test: two prompts with identical wording shared one stored record, whose manifest named the wrong prompt. | Builder |
| 13 Oct | Milestone 5. Measured on the real database: a prompt lookup took about 200 ms because every query opened a new connection. The engine now keeps one connection per thread and remembers saved versions (they cannot change). The pointer is never remembered. Measured after: lookup 11 ms, full render including storing it 27 ms at p95. | Builder |
| 13 Oct | The "cache keyed on the exact request" the owner accepted is not built. Every render must be recorded against its run, so a cache could only skip reading the pointer, which is the one thing that must be read fresh for a rollback to be seen. Remembering immutable versions gives the speed without that risk. | Builder |
| 13 Oct | "Retry until complete" is bounded by `PromptEnginePolicy.render_retry_seconds` = 30 unless the caller allows its own time. When it runs out the step is refused; it never runs on other wording. A refusal (missing value, unknown obligation, nothing in use) is raised at once. | Builder — owner to confirm the 30 |
| 13 Oct | First agent moved: demand intake (one prompt, the smallest). Which path runs is the registry's decision: a typed version in use is rendered by the engine; imported wording is filled as before. Promoting switches the agent over; rolling back switches it back, with no deploy. | Builder |
| 13 Oct | Conversion to a typed template is mechanical and refused unless it is lossless and fills to the same text character for character. | Builder |
| 13 Oct | One intended difference: the older fill replaced blanks one after another, so a value containing the name of a later blank had other text substituted into it. The typed fill puts each value in once. | Builder |
| 13 Oct | Trace ids for now: the run id is the caller's, or a new one if it has none; the step id names the agent step; the attempt is 1. Orchestrator-wide step ids and attempt counting are still to come, with tracing. | Builder |
| 13 Oct | An agent's instructions failing to reach the registry stays a logged error, not a failed save, until created agents themselves are rendered through the registry. | Builder |

## 11. Things that move out of prompt-building code

- `governance_brief` writes policy text into the system prompt. It becomes an
  `append` obligation the orchestrator passes in.
- The opportunity miner lets prompt text change scope and thresholds. The
  agent spine spec already removes this.
- The policy engine returns no obligations today. Until it does, the
  orchestrator passes an empty list and the engine's obligation handling is
  proven by tests.

## 12. Trade-offs taken

- **Module, not service.** Less to run and no network hop; costs a discipline
  rule (import only the client), which a test enforces.
- **Own tiny template language, not a library.** Less power on purpose: no
  logic means no hidden branching and no code execution.
- **Rendered text in Postgres, not S3.** Simple joins by run id and one access
  rule; revisit if volume grows.
- **Legacy syntax flag.** Adds a second path for a while, in exchange for an
  import that changes no behaviour.
- **API spec.** Generated from the router and checked in at milestone 2,
  rather than hand-written here.

## 13. Milestones

1. This note (done, awaiting approval).
2. Registry: tables on both databases, immutable versions, pointer moves with
   approval, import of existing prompts as version 1 with no behaviour change.
3. Templates and fragments: typed variables, save and render checks.
4. Render and manifest store: obligations, fail-closed errors, hashing,
   retention sweep.
5. Client and one agent end to end: step id and attempt, retry rule, one
   agent migrated as the worked example.
6. Telemetry, the log-scanning test, and the runbook (promote, roll back,
   investigate a run).

After phase 1: migrate the remaining call sites agent by agent (about 35
hard-coded prompts in 25 files, 10 direct table readers).

## 14. Acceptance criteria

All nine from the build prompt, each with a test, except the last (one step
on Gemini and one on OpenAI), which waits on ruling 1 and is replaced for now
by one step end to end on AgentNick.
