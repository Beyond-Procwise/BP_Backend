# Requirements scope definition — two-agent design

**Date:** 2026-07-29
**Branch:** Development
**Status:** approved, ready for implementation planning

## 1. What this is

Two agents that turn a demand into a supplier-ready **scope of requirement**
through conversation, and a place to put the resulting summary.

- **Universal requirements agent** — the existing `RequirementsAgent`, reworked.
  Covers the scope areas any commodity needs.
- **Commodity specialist agent** — new. Goes deep on the areas specific to the
  requirement's commodity.

Either agent serves a buyer or a requester; there is no role branching. The
conversation captures detail, the agent summarises it into an actual requirement
document, and the person commits it when they judge it defined.

Explicitly out of scope: a new approvals chain (`ApprovalsAgent` already owns
that), auto-commit at a strength threshold, and any change to the demand intake
itself.

## 2. The problem being fixed

There are two incompatible notions of "requirement" in the product.

**The backend agent fills five blanks.** `src/services/requirement_service.py:13`
defines completeness as five non-empty fields:

```python
DEFAULT_REQUIRED_FIELDS = (
    "title", "category", "quantity", "needed_by_date", "delivery_location",
)
```

`evaluate_completeness()` scores `filled / required`. When all five are present
the requirement is "complete" and hands off to sourcing. Quantity and delivery
address are order details. Nothing in this model describes scope.

That model is also already broken for services: lump-sum services deals
legitimately have no quantity, so a services requirement can never reach 100%.

**The UI has a scope-depth conversation designed but stubbed.**
`src/modules/SpendIQ/engine.js:4183` states it outright:

> Prototype · the live agent (universal → specialist → validation → document)
> plugs in behind this same UI

| Stubbed thing | Location | What it actually does |
|---|---|---|
| Universal question bank | `engine.js:4021` `UNIVERSAL_QS` | 5 hardcoded strings |
| Specialist question bank | `engine.js:4028` `SPECIALIST_QS` | 5 hardcoded strings |
| Agent reply | `engine.js:4076` | `qs[(n-1)%qs.length]` — the 11th answer re-asks question 1 |
| Strength score | `engine.js:4077` | `min(98, strength+3)` — a flat bump per turn |
| "Areas covered" | `engine.js` stat row | the string literals `'4 / 7'` and `'6 / 9'` |
| Generate requirement | `engine.js:4065` `reqGenerate()` | sets `stage='done'`, `strength=max(strength,88)` |
| Requirement document | `engine.js:4204` `docPanel` | fully hardcoded prose |
| Specialist handoff | `engine.js:4064` `reqHandoff()` | flips a string |

The transcript does survive a reload — `reqPersist()` (`engine.js:4043`) PATCHes
`/spendiq/requirements/{id}`, and the seed in `sql/bp_requirement_seed.sql`
mirrors that shape into `proc.bp_requirement.seed_context`. So persistence is
real; the intelligence is not.

**Nothing separates given facts from built scope.** The requirement screen has
one "Ingested from demand" panel plus a chat log. Facts that already exist in
`proc.bp_demand.payload` (`sql/bp_demand_tprm.sql:4`) are dropped:

| Fact | Where it already is | On the requirement screen |
|---|---|---|
| Cost centre | `payload.finance.cc` — e.g. `"IT-3300"` | absent |
| Budget status | `payload.finance.budget` — e.g. `"Budgeted"` | absent |
| Cost structure | `payload.finance.struct` — e.g. `"Opex · recurring"` | absent |
| Problem statement | `payload.problem.{cur,des,inaction}` | partially (one line) |
| Success criteria | `payload.criteria` | yes |
| Incumbent supplier | `payload.party` — e.g. `"Northwind Cloud"` | absent |
| Prior suppliers for category | computed by `requirement_service.seed_context()` from `proc.bp_purchase_order_trgt` | absent |

## 3. Why this shape

The load-bearing pieces exist already:

| Capability | Where it lives | State |
|---|---|---|
| Requirement entity + persistence | `proc.bp_requirement`, `requirement_service.persist()` | live |
| Requirement conversation endpoint | `POST /requirements/message` (`src/api/routers/requirements.py:125`) | live |
| Agent registry from one catalogue | `agent_definitions.json`, `src/agents/auto_registry.py` | live; 14/14 consistent |
| DB-governed prompts | `proc.bp_prompt`, `BaseAgent.resolve_prompt()` (`base_agent.py:326`) | live; already used by `RequirementsAgent` |
| DB-governed policy | `proc.bp_policy`, `PolicyEngine` (`src/engines/policy_engine.py`) | live |
| Governance hot reload | `POST /agents/reload-governance` | live |
| Category spend history | `requirement_service.seed_context()` | live |
| Requirement UI incl. `.asec` document panel | `engine.js:4186` `requirementDefineView()` | live, stubbed |

So this is wiring real intelligence behind an existing screen, not new
infrastructure.

## 4. The three zones

The requirement is presented as three explicitly separated zones. This is the
structural fix: the screen currently conflates given facts with built scope,
which is why cost centre and prior suppliers have nowhere to live.

### Zone 1 — Context (given, never asked)

Inherited from the demand or looked up from history. Four groups:

- **Commercial envelope** — cost centre, budget status, cost structure,
  estimated value, duration, entity
- **Problem statement** — current situation, desired outcome, cost of inaction
- **Success criteria**
- **Incumbency** — current supplier from the demand, plus prior suppliers for
  the category from `proc.bp_purchase_order_trgt`

Both agents read Zone 1 and are forbidden from asking for any of it. This makes
good on what the UI already promises in words: *"the agent starts here and only
clarifies the gaps, no re-asking."*

### Zone 2 — Scope of requirement (built)

The artifact the agents produce. Today it exists only as a chat transcript plus
a hardcoded document. It becomes structured **scope areas** (§5), a **summary**
generated from them (§7), and a **commit** action (§8).

### Zone 3 — Readiness (measured)

`strength` becomes weighted coverage of the areas that apply to this commodity.
Mandatory unanswered areas are the critical gaps. It informs and never gates:
commit is available at any point (§8).

## 5. Data model — scope areas

A scope area is the unit of state. The transcript remains for audit, but the
requirement's state is its areas — that is what makes the summary reproducible
and the strength score real.

Each area carries:

| Field | Meaning |
|---|---|
| `area_key` | stable identifier, e.g. `service_levels`, `data_residency` |
| `label` | display name, e.g. "Service levels" |
| `owner` | `universal` or `specialist` |
| `commodity` | `*` for universal, else the commodity it belongs to |
| `mandatory` | whether an unanswered area counts as a critical gap |
| `weight` | contribution to strength |
| `status` | `open`, `answered`, `needs_depth` |
| `answer` | the agreed content, in the person's own terms |
| `source_turn` | which conversation turn produced the answer (provenance) |

**Area definitions are governed data, not code.** They live in the governance
tables (`proc.bp_prompt` / `proc.bp_policy`) that `BaseAgent.resolve_prompt()`
and `PolicyEngine` already read, so a commodity's area set is editable without a
deploy, versioned, and hot-reloadable via `POST /agents/reload-governance`.

**Answers are stored per requirement**, keyed `(requirement_id, area_key)`, in a
new `proc.bp_requirement_area` table (§10).

### Why areas are governed rather than model-invented

Asking AgentNick to invent which areas matter for a commodity was rejected. For
Works & construction, CDM and H&S is a legal duty — whether it gets asked cannot
depend on a model's run-to-run variation. It also makes coverage unmeasurable:
"6 of 9 areas" means nothing if the model picks a different nine each time.

Hardcoding them in Python was also rejected — it reproduces the JavaScript
prototype's problem one layer down, needing a code change per commodity.

**Deterministic spine, LLM surface.** The areas are data; AgentNick phrases the
next question in context, reading Zone 1 and prior answers so it reads as a
conversation rather than a form. This mirrors extraction's established
regex-primary / AI-judge-final shape.

### Universal areas (`commodity = '*'`)

`scope_boundaries`, `volumes_and_scale`, `service_levels`, `transition`,
`governance`, `commercial_model`, `exit_and_data_return`

### Specialist areas (per commodity)

Resolved by `commodity`, falling back to `category`. Seeded from the prototype's
own bank plus the seed data's construction example:

- **Technology & digital** — `licensing`, `data_residency`, `integration`,
  `security_posture`, `exit_data_format`
- **Works & construction** — `cdm_and_hs` (mandatory), `principal_designer`,
  `site_risks`
- **Supply of goods** — `delivery_and_logistics`, `quality_and_returns`,
  `packaging_and_labelling`

Commodities without a seeded set fall back to universal areas only, and the UI
says so rather than inventing depth.

### The five legacy fields

Demoted. `title` and `category` remain requirement identity. The other three
become ordinary scope areas — asked when the commodity makes them relevant,
never blockers — mapping onto existing areas rather than adding duplicates:

| Legacy field | Area it becomes | Owner |
|---|---|---|
| `quantity` | `volumes_and_scale` | universal |
| `needed_by_date` | `transition` (mobilisation and go-live timing) | universal |
| `delivery_location` | `delivery_and_logistics` | Supply of goods specialist |

Readiness is purely scope coverage.

`DEFAULT_REQUIRED_FIELDS` stops being a completeness gate. Because
`supplier_ranking` and RFQ drafting read these fields downstream, each consumer
is checked to handle their absence as part of implementation, and they continue
to be persisted to their existing `bp_requirement` columns when answered.

## 6. The two agents

Both are registered in `agent_definitions.json` and instantiated by
`auto_registry`, both implement `run(context: AgentContext) -> AgentOutput` like
every other agent, and neither branches on whether the person is a buyer or a
requester.

### Universal requirements agent (`requirements`)

The existing `RequirementsAgent`, reworked from field-slot filling to area
clarification:

1. Load or resume the requirement session.
2. Load Zone 1 context (demand payload + category history).
3. Resolve the universal area set from governance.
4. Ask one focused question for the highest-weighted open area, phrased by
   AgentNick from Zone 1 and prior answers.
5. Attribute the person's answer to an area, store it with its source turn.
6. Recompute strength; regenerate the summary (§7).

### Commodity specialist agent (`requirements_specialist`)

New agent, same loop, differing in step 3: it resolves the specialist area set
for the requirement's commodity, and its prompt is instructed to build on the
universal answers rather than restate them.

Entered explicitly via the existing handoff button (`reqHandoff()`), and
re-enterable. It is optional — commit does not require it (§8).

### Stage model

`stage` and `status` are separate and both are columns (§10).

`stage` is the UI's existing four steps, retained and made real: `universal →
specialist → validation → done`, where `validation` is the strength and critical
gap check.

`status` is the requirement's lifecycle: `draft, gathering, defined, complete,
handed_off, abandoned`.

Commit (§8) sets `status = 'defined'` **and** `stage = 'done'` together, from
whatever stage the conversation had reached.

## 7. Summarising the requirement

A summarise step turns answered areas into the requirement document, in the
`.asec` sectioned shape the screen already uses: **Overview**, **Scope** (in /
out / volumes), **Specification & standards**, **KPIs & SLAs**, **Commercial**,
**Exit**.

It regenerates as the conversation grows, so the panel below the chat is a live
draft rather than a reveal at the end — the person can watch the requirement
take shape and see which sections are still thin.

### Grounding rule — the summary may only contain what was said

Every sentence must trace to a scope area answer or a Zone 1 fact. Each rendered
section stores the `source_turn`s it was built from, and the UI can show them.

This is a named risk, not a formality. The current hardcoded panel asserts
"1,200 users · 40TB · 12% YoY growth" and "99.95% availability, 15 min P1
response". If the live summariser invents figures of that kind, a fabricated
requirement gets issued to suppliers.

The existing grounding guard is **not sufficient here**: it validates that
extracted *values* appear in a source, and a fabricated *clause* passes it. So
the summariser is constrained by construction rather than by post-hoc checking —
each section is rendered from specific area answers, and an area with no answer
renders as an explicit gap rather than as plausible prose. A section that cannot
be traced is not emitted.

## 8. Commit

An explicit human action — "Commit requirement — defined". Never automatic.

- Sets `status = 'defined'` and `stage = 'done'`.
- Freezes the summary as the requirement of record.
- Unlocks "Send to suppliers" (`reqSendSuppliers()`), which stays blocked before
  commit.
- **Available at any point**, at the person's judgement. Not gated on the
  specialist agent having run, and not gated on a strength threshold. Strength
  informs the decision; it never makes it.
- Re-opening after commit creates a **new version** rather than overwriting, so
  what was issued to suppliers stays recoverable.

This replaces `reqGenerate()`, which currently fakes the outcome by setting
`strength = max(strength, 88)`.

## 9. Readiness scoring

```
strength = Σ(weight of answered applicable areas) / Σ(weight of applicable areas)
critical_gaps = applicable areas where mandatory = true and status != 'answered'
```

Applicable areas = universal areas + the specialist set for this commodity. This
replaces both the `filled / required` field count and the UI's `strength + 3`
per turn, and makes the "Areas covered" stat honest instead of the literals
`'4 / 7'` / `'6 / 9'`.

## 10. Schema changes

Additive and idempotent, `bp_` prefixed, `ix_bp_*` indexes, per project
convention.

**New — `proc.bp_requirement_area`**: `requirement_id`, `area_key`, `label`,
`owner`, `commodity`, `mandatory`, `weight`, `status`, `answer`, `source_turn`,
`created_at`, `updated_at`. Primary key `(requirement_id, area_key)`. Index on
`requirement_id`.

**New — `proc.bp_requirement_version`**: frozen summaries per commit —
`requirement_id`, `version`, `summary` (JSONB), `committed_by`, `committed_at`.
Primary key `(requirement_id, version)`.

**Altered — `proc.bp_requirement`**: the status check currently allows only
`draft, gathering, complete, handed_off, abandoned`. Add `defined`. Add
`commodity`, `strength` and `stage` columns so the UI's model is backed by
columns rather than living inside `seed_context` JSON.

**Governance seed**: universal and per-commodity area definitions inserted into
the governance tables, plus the two agents' prompts into `proc.bp_prompt`.

`deal_id` is untouched — it remains owned by the database stored procedure.

## 11. API surface

Existing endpoints keep working:

- `POST /requirements/message` — one turn. Gains an optional request field
  `agent` (`requirements` | `requirements_specialist`) so the caller can direct
  the turn at either agent; it defaults to the requirement's current `stage`. The
  response reports which agent answered.
- `GET /requirements/{id}` — gains `context` (Zone 1), `areas` (Zone 2),
  `strength`/`critical_gaps` (Zone 3) and the current `summary`.

New:

- `POST /requirements/{id}/commit` — commit the requirement; returns the frozen
  version.
- `POST /requirements/{id}/handoff` — enter the specialist stage.

The gateway's existing `PATCH /spendiq/requirements/{id}` continues to serve the
UI; it is repointed at these rather than storing UI state in `seed_context`.

## 12. Testing

- **Area resolution** — universal set for an unknown commodity; specialist set
  for Technology & digital and Works & construction; mandatory `cdm_and_hs`
  present for construction.
- **No re-asking Zone 1** — given a demand carrying cost centre, budget and
  incumbent, no generated question asks for any of them.
- **Answer attribution** — an answer is stored against the area asked about,
  with its source turn.
- **Strength** — weighted coverage arithmetic; a services requirement with no
  quantity can still reach 100%; critical gaps list only mandatory unanswered
  areas.
- **Summary grounding** — a summary never contains a figure absent from the
  answers or Zone 1; an unanswered area renders as a gap, not as prose.
- **Commit** — available at `universal` stage without the specialist having run;
  status becomes `defined`; re-opening creates version 2; "send to suppliers" is
  blocked before commit.
- **Transport failure** — an Ollama failure degrades the turn without failing it
  (the regression already covered by
  `test_llm_transport_failure_keeps_the_turn_alive`).
- **Registration** — both agents resolve from `agent_definitions.json` and
  expose `run(context)`, per `tests/test_required_agents_are_registerable.py`.

## 13. Risks

| Risk | Handling |
|---|---|
| Summariser fabricates specifics | Sections rendered only from traced area answers; untraceable sections not emitted (§7) |
| Existing grounding guard gives false assurance on prose | Not relied on; constraint by construction instead (§7) |
| Downstream sourcing assumes the five fields | Each consumer checked for absence handling; fields still persisted when answered (§5) |
| Commodity mapping wrong or missing | Falls back to universal areas and says so, rather than inventing depth (§5) |
| Area sets grow stale | They are governed data, editable and hot-reloadable without deploy (§5) |

## 14. Not doing

- Auto-commit at a strength threshold — a percentage should not decide a
  requirement is done.
- A separate approval chain — `ApprovalsAgent` already owns approvals.
- Changes to demand intake — the demand is read, never written.
- Role-specific agents — both agents serve buyers and requesters identically.
