# Query Intake & Clarification Layer — Design

**Date:** 2026-06-20
**Status:** Approved design (pre-implementation)
**Author:** Nick + Claude (brainstorming session)

## Purpose

Give the procurement assistant a front-door **intake layer** so it can take any
procurement question, ground it in live context, and — when the question is
ambiguous or missing a required detail — **ask the user a reasoned clarifying
question instead of guessing or giving up.**

The layer must make a smaller local model (AgentNick) behave reliably for this
job by keeping the *judgement* in deterministic code and using the model only
for *extraction*. The hard rule throughout: **answers must never be invented.**

## Constraints & context

- **Independent of the in-flight product upgrade.** The upgrade changes the UI,
  the database, and adds new agents. None of those touch this layer directly;
  this layer plugs into them through stable interfaces so it survives the
  upgrade and can be added once the upgrade lands.
- **Builds on what exists — no reinvention.** It sits in front of the current
  `/ask` → `RAGPipeline.answer_question` flow and the existing messaging screen.
  `answer_question` is **not** rewritten.
- **Clarify turn is structured.** Returns a `needs_clarification` flag + a
  reasoned question + clickable answer chips the upgraded UI renders specially.

## Today's state (what we are extending)

- `/ask` (`src/api/routers/workflows.py`) calls `RAGPipeline.answer_question`
  (`src/services/model_selector.py:2321`), a retrieve-then-answer RAG flow.
- It has **no clarification gate**: when retrieval is empty it returns a canned
  "couldn't find that"; otherwise it synthesises an answer that **can assert
  facts not present in the retrieved context.**
- `PromptEngine.deconstruct_query` (`src/orchestration/prompt_engine.py:397`)
  was meant to parse intent but is **dead code** (zero callers) and
  ranking-shaped only.
- `AutoRegistry.describe_for_llm()` already enumerates available agents.
- `proc.bp_prompt` is the governance table for editable prompts.

## Core principle

**The model proposes, the data disposes.** The model extracts; deterministic
code decides. Three rules enforce "never invent" end to end:

1. **Fail safe, not fail open.** If intake cannot run, degrade to honesty
   ("I couldn't read that clearly — can you rephrase?"), never silently fall
   through to ungated answering.
2. **Gate the output, not just the question.** Every factual claim in a
   generated answer must trace to a retrieved snippet or the live snapshot.
   Ungrounded claim → abstain ("I don't have that in your records"), never
   assert.
3. **Resolve in code.** Entities and option chips are validated against the live
   snapshot. The model's output is never trusted as fact: an extracted entity
   that doesn't resolve to a real record is treated as *missing* → ask. Option
   chips are built in code from the snapshot, never emitted by the model.

## Architecture

One new self-contained module plus a thin wiring change. `answer_question` is
untouched.

```
src/orchestration/query_intake.py            ← new, the whole feature
  ├─ QueryIntake               orchestrates one intake turn
  ├─ IntakeContextProvider     Protocol → ContextSnapshot   (decouples from DB)
  ├─ DefaultContextProvider    pulls snapshot from existing services/DB
  ├─ IntentSpec / INTENT_SPECS intent → required slots → clarify template
  ├─ EntityResolver            resolves model output against the snapshot
  ├─ AnswerGroundingGate       output check: claim → source, else abstain
  └─ IntakeResult              structured output

src/api/routers/workflows.py   ← ~10 lines in /ask: run intake before RAG
```

### Data flow inside `/ask`

1. `QueryIntake.handle(query, session)` runs first.
2. Build a `ContextSnapshot` (suppliers / categories / open_deals + open
   `extras`) from the provider.
3. One AgentNick call with the **`intake_classify`** prompt → strict JSON
   (intent, entities, missing). `format=json` enforced.
4. `EntityResolver` resolves every extracted entity against the snapshot.
   Unresolvable → drop to *missing*.
5. **Deterministic** required-slot check for the intent (code, not model).
6. If a slot is missing → return the **clarify payload**, skip RAG entirely.
7. If complete → attach resolved entities and call existing `answer_question`.
8. `AnswerGroundingGate` checks the generated answer against
   `retrieved_documents`; ungrounded claims are suppressed / the answer abstains.

### Failure behaviour (fail safe)

- Classify call errors/times out → intake returns a safe "couldn't process,
  please rephrase" turn. **Never** falls through to ungated answering. Loud log.
- Empty snapshot for a required slot → honest limitation / clarify, never a
  guess.

## Interfaces

### Context contract (decouples from the DB upgrade)

```python
@dataclass
class ContextSnapshot:
    suppliers: list[str]
    categories: list[str]
    open_deals: list[dict]          # [{"deal_id":.., "name":..}]
    extras: dict = field(default_factory=dict)   # future fields, no signature change

class IntakeContextProvider(Protocol):
    def snapshot(self, session_id: str) -> ContextSnapshot: ...
```

`DefaultContextProvider` reads from existing services today. When the DB upgrade
changes the schema, **only this provider changes** — prompts and intake logic do
not.

### Intent catalogue (decouples from new agents)

- Available agents come from existing `AutoRegistry.describe_for_llm()` — new
  agents from the upgrade appear automatically.
- `INTENT_SPECS`: a small table mapping `intent → required_slots →
  clarify-question template`. Adding an intent is one entry.

```python
@dataclass
class IntentSpec:
    name: str                       # rank | compare | status | spend | policy | lookup | negotiate
    required_slots: list[str]
    clarify_templates: dict[str, str]   # slot -> question template
```

## Prompt set (lives in `proc.bp_prompt`, editable without code)

- **`intake_classify`** — structured classifier. Injects the `ContextSnapshot`
  and the agent catalogue; enforces `format=json`. Returns `intent`, `entities`,
  `confidence`, `missing`. It does **not** decide whether to clarify and does
  **not** emit option chips. Low `confidence` is treated by code as a clarify.
- **`clarify_templates`** — per-slot question templates so good phrasing is our
  asset, not something the small model must invent. Code fills the blank and
  supplies option chips from the snapshot.
- **`grounded_answer`** — the **existing** `_compose_llm_prompt`, reused as-is,
  with resolved entities passed in. Not rewritten. Paired with
  `AnswerGroundingGate` on its output.

## Clarify response contract (for the UI)

Structured flag (option 3) + clickable chips (option 2):

```json
{
  "needs_clarification": true,
  "answer": "You have 3 open deals — which one?",
  "clarification": {
    "slot": "deal",
    "question": "You have 3 open deals — which one?",
    "options": [
      {"label": "Techworld", "value": "deal:123"},
      {"label": "Dixon Reynolds", "value": "deal:456"}
    ]
  },
  "follow_ups": [],
  "retrieved_documents": []
}
```

- `answer` carries the question as plain text too, so the turn is readable even
  without special rendering.
- `options` are built in code from the snapshot — a chip can only ever be a real
  entity.
- When the question is clear, `/ask` returns the **normal** `answer_question`
  payload, unchanged.

### Multi-turn

When the user answers (typed or chip), intake merges the new slot with the
pending intake state (keyed by session) and re-checks required slots. State is
small and lives in the module.

## Output grounding gate

`AnswerGroundingGate` runs after `answer_question` produces text:

- Each factual claim must map to a `retrieved_documents` entry or a snapshot
  value.
- Unmapped claim → suppressed; if the core answer cannot be grounded, the
  response abstains: *"I don't have that in your records."*
- Lightweight by design (no citation-verification model): claims are checked
  against the retrieved set. Proportional to the need (YAGNI).
- Every intake decision and every answer's grounding sources are logged so the
  no-invention guarantee is auditable.

## Error handling summary

- Classify failure → fail-safe rephrase turn, never ungated answering.
- Unresolvable entity → treated as missing → clarify.
- Empty snapshot for a required slot → honest limitation, never a guess.
- Ungrounded answer claim → suppressed / abstain.

## Testing (no live model required)

- **Slot-checking** — pure-function unit tests per intent.
- **Entity resolution** — exact + fuzzy match against snapshot fixtures;
  unresolvable → missing.
- **Schema validation** — `intake_classify` output validated against JSON
  fixtures (valid, malformed, hallucinated-entity).
- **Fail-safe path** — mocked failing classify call → rephrase turn, RAG not
  called.
- **Grounding gate** — answer with an ungrounded claim → suppressed/abstained;
  fully-grounded answer → passes through.
- **Golden intake examples** — ambiguous → asks; clear → passes through;
  multi-turn merge resolves.

## Out of scope (YAGNI)

- Rewriting `answer_question` / the RAG retrieval.
- Fine-tuning AgentNick (intake is a prompting problem; prior analysis advises
  against retraining the local model on the current corpus).
- A heavyweight citation-verification model.
- Routing the chat path through `ReasoningEngine` (flaky planner; not put on the
  live messaging path). The intake module can be *exposed* as an agent later if
  wanted.

## Decoupling summary (survives the upgrade)

| Upgrade area | Absorbed by | Prompts/logic change? |
|---|---|---|
| Database schema/fields | `DefaultContextProvider` / `ContextSnapshot.extras` | No |
| New agents | `AutoRegistry.describe_for_llm()` | No |
| New intents | one `INTENT_SPECS` entry | No |
| UI messaging screen | clarify response contract | No |
