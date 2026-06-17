# Requirements Agent — Design Spec

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Author:** brainstorming session

## 1. Problem & Motivation

The ProcWise backend pipeline is entirely **downstream**. It begins when a document
(quote / PO / invoice) lands in `proc.process_monitor`, then links documents → deals →
opportunities → summaries. `procurement_context_service` names an 11-stage procurement
lifecycle that *starts* with **"Need Identified" → "RFQ Generated"**, but nothing in the
codebase implements those first two stages. There is no requirement, sourcing-request, or
intake entity (`requisition_id` exists only as a stray reference column on PO/invoice
schemas).

The **RequirementsAgent** fills this empty front of the funnel: it turns a vague buyer need
into a complete, structured, sourcing-ready **requirement**, then hands it off to the
existing acting agents. It does not rank suppliers or draft RFQs itself.

## 2. Scope

**In scope (v1):**
- Conversation-led, multi-turn elicitation of a single procurement requirement.
- **History seeding**: pre-fill / inform questions from past deals, spend, and prior
  conversation memory so the agent only asks what it cannot infer.
- **Unstructured brief ingestion**: a pasted/uploaded free-text brief, email, or
  requisition is parsed (reusing existing extraction) to pre-fill fields before clarifying
  the gaps.
- Persist a completed requirement to a new `proc.bp_requirement` table.
- Emit hand-off signals so existing agents (supplier ranking, RFQ/email drafting) can pick
  it up — **gathering stays decoupled from acting**.
- Surface: a dedicated `POST /requirements/message` endpoint **plus** SSE progress events
  for a live chat UI.

**Out of scope:**
- Supplier ranking, RFQ drafting, negotiation, opportunity mining (existing agents own these).
- Auto-launching a sourcing workflow on completion (hand-off is signal-only; the
  orchestrator/UI decides whether to chain).
- Multi-requirement batch intake.

## 3. Architecture — three units, clean seams

The agent stays small because state and persistence are factored out — directly addressing
the documented god-class concern.

| Unit | Responsibility | Reuses |
|---|---|---|
| `RequirementsAgent(BaseAgent)` | Per-turn brain. Given one user message + the session, decides: ask the next question, parse a pasted brief, or finalize. **Stateless per call** (fits `run(context) -> AgentOutput`). | `BaseAgent`, governance resolver (`resolve_prompt`, `governing_policy`), AgentNick **local** model |
| `RequirementSession` | Multi-turn state: the partial requirement, turn history, completeness, status. Redis-backed hot state. | Direct clone of the `NegotiationSession` pattern (`src/services/negotiation_session.py`) |
| `RequirementService` | Persistence + seeding. Writes/reads `proc.bp_requirement`; assembles historical seed context; parses unstructured briefs. | `procurement_context_service` (history), `ConversationMemoryService` (semantic recall), existing extraction for brief parse |

**Boundaries:**
- The agent never touches Redis or SQL directly — it calls `RequirementSession` and
  `RequirementService`.
- `RequirementSession` is the hot, per-turn state; `proc.bp_requirement` is the durable
  source of truth. The `draft`/`gathering` row backs the session.
- `RequirementService` is the only unit that reads history/extraction, keeping the agent
  brain focused on conversation logic.

## 4. Data model — `proc.bp_requirement`

New table, `bp_` prefix convention, indexes `ix_bp_requirement_*`. Keyed **independently of
`deal_id`** — a requirement precedes any deal.

| Column | Type | Notes |
|---|---|---|
| `requirement_id` | text PK | e.g. `REQ-20260617-<short>` |
| `session_id` | text | Links to the Redis session |
| `status` | text | `draft` → `gathering` → `complete` → `handed_off`; `abandoned` |
| `created_by` | text | Buyer / user id |
| `created_at`, `updated_at` | timestamptz | |
| `title` | text | Filled by conversation |
| `category` | text | Procurement category |
| `description` | text | |
| `quantity` | numeric | |
| `unit` | text | |
| `target_budget` | numeric | |
| `currency` | text | |
| `needed_by_date` | date | |
| `delivery_location` | text | |
| `priority` | text | |
| `specifications` | jsonb | Free-form spec key/values |
| `constraints` | jsonb | Compliance, preferred suppliers, approval thresholds, etc. |
| `completeness_score` | numeric | Drives "are we done yet?" |
| `missing_fields` | jsonb | List of still-required fields |
| `seed_context` | jsonb | What history/brief contributed (auditability) |

The exact required-field set is **not** hardcoded — it is governed by policy (see §6).

## 5. Conversation flow

```
turn → load session (or create on turn 0)
     → merge new input:
         • plain user message → interpret answer
         • pasted/uploaded brief → parse via extraction → pre-fill fields
     → recompute completeness vs the governed requirement schema
     → if gaps remain:
         pick the highest-value missing field → ask ONE crisp question (local LLM)
     → if complete:
         persist bp_requirement(status=complete)
         → emit hand-off signals
         → return a human-readable summary
```

**Turn 0 seeding.** `RequirementService` pre-fills category norms, typical suppliers, and
last-purchase specs from `procurement_context_service` + `ConversationMemoryService`, so the
first question is already informed (e.g. *"Last time you bought X at £Y from Z — same
spec?"*).

**Brief ingestion.** If the user pastes free text, it is routed through the existing
extraction muscle to pre-fill fields, then the agent only clarifies the gaps. This keeps the
conversation short when the buyer already has a written brief.

**One question per turn.** The agent asks a single, crisp question at a time (governed
prompt template), never a wall of fields.

## 6. Governance & completeness (`bp_prompt` / `bp_policy`)

- **`bp_prompt`** (agent-scoped via `prompt_linked_agents`): the elicitation system prompt
  and the "ask one crisp question" template. Editable via `POST /agents/reload-governance`
  with no redeploy — tone and questioning style are tunable at runtime.
- **`bp_policy`** (agent-scoped via `policy_linked_agents`): `requirement_required_fields`
  and per-category overrides (e.g. capex requires `approval_threshold`). The completeness
  check reads this policy, so **"what makes a requirement complete" is data, not code**.
- **Model routing:** AgentNick **local** model for elicitation (per the documented model
  routing policy); optional **cloud** model only for a final human-readable summary.

## 7. Hand-off (no downstream coupling)

On reaching `complete`, the agent emits via the shared `WorkflowContext` blackboard:
- `SUGGEST_AGENT → supplier_ranking` (requirement as query) and/or `email_drafting`
  (RFQ draft).
- `AgentOutput.data = {requirement_id, requirement, completeness_score}` with
  `next_agents = []` — the orchestrator/UI decides whether to chain into a sourcing
  workflow.

Existing agents pick the requirement up through the blackboard already built into both
orchestrator paths. Gathering and acting stay decoupled.

## 8. Surface (API)

New router `src/api/routers/requirements.py`, registered in `src/api/main.py` like the
other routers, behind `verify_api_key`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/requirements/message` | `{session_id?, message, brief?}` → next question, or the completed requirement |
| GET | `/requirements/{requirement_id}` | Fetch a requirement record |
| GET | `/requirements` | List requirements (paged) |

**SSE progress events** are emitted during a turn (seeding, parsing, question, completion)
for a live chat UI, reusing the existing streaming event shape in `stream.py`.

## 9. Registration

- Add agent class `src/agents/requirements_agent.py`.
- Register in `agent_definitions.json` (slug `requirements`, capability
  `requirements_gathering`, required input `message`, optional `brief` / `session_id`,
  outputs `requirement_id` / `requirement` / `completeness_score`).
- Add `REQUIREMENTS_GATHERING` to the `AgentCapability` enum and capability-role mapping.
- Add module/class to `agent_factory.py` maps and a contract to `AGENT_CONTRACTS` (if the
  factory path is used alongside auto-registry).

## 10. Testing

Unit tests per unit, matching the codebase's defensive (degrade-gracefully) style:
- `RequirementSession`: state transitions (`draft`→`gathering`→`complete`/`abandoned`),
  round/turn history, Redis mocked.
- Completeness check reads `requirement_required_fields` from a mocked policy; per-category
  override applies.
- Brief-parse pre-fill: a free-text brief populates fields, leaving only true gaps.
- Seeding: history pre-fills `seed_context` and informs the first question.
- **Full multi-turn elicitation** reaching `complete`, persisting a `bp_requirement` row,
  and emitting the correct hand-off signals.
- Redis / RAG / extraction dependencies mocked so tests are isolated from infrastructure.

## 11. Files touched (summary)

**New:**
- `deploy/sql/2026-06-17_bp_requirement.sql` — table + indexes
- `src/agents/requirements_agent.py`
- `src/services/requirement_session.py`
- `src/services/requirement_service.py`
- `src/api/routers/requirements.py`
- `tests/...` for each unit

**Modified:**
- `agent_definitions.json` — register agent
- `src/agents/agent_interface.py` — `AgentCapability.REQUIREMENTS_GATHERING`
- `src/agents/agent_factory.py` — maps + contract (if used)
- `src/api/main.py` — include router
- `bp_prompt` / `bp_policy` seed rows (governance) — via SQL or governance load

## 12. Open questions / future

- Linking a requirement to the deal that eventually results from it (reverse FK once a deal
  forms) — deferred; not needed for v1.
- A requirements dashboard surface — deferred; `GET /requirements` covers v1 listing.
