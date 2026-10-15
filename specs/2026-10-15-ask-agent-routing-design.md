# Ask agent routing: the screen's agent answers, and a person can pick another

**Date:** 2026-10-15
**Status:** Design, awaiting review.
**Decision taken (Nick, 2026-10-15):** on a section's screen, that section's agent answers
automatically; the answer names the agent, and the person can switch to another agent or to
General.

---

## 1. Problem

Every question typed into an Ask box goes down one fixed pipeline, whatever screen it was
asked on: the exact-figure analytics layer, otherwise document search plus one model call
(`src/api/routers/workflows.py:881`, `src/services/model_selector.py:2532`). No agent is
involved.

Two things look like routing and are not:

- **Chat's agent picker is cosmetic.** A conversation started "with" an agent stores
  `agent_id` in browser storage only. `askRequestBody` never sends it
  (`beyond_procwise_ui/src/modules/Chat/askRequest.js`), and `AskRequest` has no field to
  receive it.
- **SpendIQ's Ask panel says which screen it is on**, as prose in `context`. The backend
  folds it into the prompt as background and never uses it to choose who answers.

Each agent can already answer a task under its own instructions, its own permitted tools
and the agent-policy gate: `BaseAgent.reason` (`src/agents/base_agent.py:349`), which
returns a `ToolRunResult` (`src/services/tool_runtime.py:100`). Nothing on the Ask path
calls it.

## 2. What success looks like

1. A question asked on the Negotiations screen is answered by the negotiation agent, and the
   answer says so.
2. A person who picks an agent (in Chat, or with the switch in SpendIQ's Ask panel) gets
   that agent, whatever screen they are on.
3. A person can switch back to General and get exactly today's answer.
4. Questions with one exact answer (supplier spend rankings and the other analytic
   questions) keep their exact figures, on every screen.
5. A named agent that does not exist, or that this person may not use, never fails
   silently: the question is answered by General and the answer says why.

## 3. Design

### 3.1 Request

`AskRequest` gains two optional fields:

| Field | Meaning |
|---|---|
| `agent` | The agent slug the person chose, or the literal `general` to force today's path. Absent means "nobody chose". |
| `screen` | The screen identifier the question was asked on (SpendIQ's `current`, e.g. `negotiations`). Absent from Chat. |

Both are optional, so every existing caller keeps working unchanged.

### 3.2 Choosing who answers (`src/services/ask_routing.py`, new, pure)

One function, `choose_responder(agent, screen, available, screen_map) -> Responder`, where
`Responder` is `{slug | None, source, note}`. Order:

1. `agent == "general"` → General, source `chosen`.
2. `agent` given and in `available` → that agent, source `chosen`.
3. `agent` given and not in `available` → General, source `fallback`, note
   "<name> is not available, so this was answered generally."
4. `screen` given and mapped → the mapped agent, source `screen`, provided it is in
   `available`; otherwise General, source `fallback`.
5. Otherwise → General, source `default`.

`available` is the set of agents loaded in `agent_nick.agents` (built-in slugs and created
agents), intersected with what this person's role may run. The role check reuses the
existing card-action and agent-policy rules; it adds no new permission model.

The function is pure (no database, no model) so the whole decision table is unit-tested.

### 3.3 The screen-to-agent map

The map lives in governed settings (a `bp_policy` row, `ask_screen_agents`), so an admin
can change it without a release. It is read on each request with a short cache, like the
other governed limits. If it cannot be read, every screen answers as General. A routing
outage must not make every question fail.

Seed values, all slugs confirmed registered in `src/api/main.py`:

| Screen (`current`) | Agent |
|---|---|
| `negotiations` | `negotiation` |
| `quotes`, `quotebuilder` | `quote_comparison` |
| `suppliers`, `tprm` | `supplier_ranking` |
| `actions`, `invoices`, `pos`, `compliance` | `discrepancy_detection` |

SpendIQ has no screen of its own for approvals or opportunities (checked against every
`current ===` and `go('…')` in `engine.js`), so `approvals` and `opportunity_miner` are not
mapped. They can still be chosen with the switch.

Every other screen (Home, Dashboard, Analyse, Contracts, Renewals, Pipeline, Settings and
the admin screens) answers as General. No contract or renewals agent exists, and routing
there would be pretending.

### 3.4 Answering

In `RAGPipeline.answer_question`, after the presentation-directive check and before
anything else:

1. **Exact figures first, always.** `analytics_ask.analytic_answer` runs before routing on
   every screen. If it answers, that answer is returned. It is the same figure the tile
   beside it shows, and an agent must not restate it differently. The answer is labelled
   "Answered from the records".
2. If the responder is an agent: call `agent.reason(query, user_id=..., extra_system=<screen
   context>)` with the session's recent history as context. Map the `ToolRunResult` to the
   same response shape the Ask path returns today (`answer`, `follow_ups: []`,
   `retrieved_documents` from the tool calls' sources, `next_steps: []`).
3. If `reason` raises or returns `error`, the question is answered by General, with the note
   "The <agent> agent could not answer, so this was answered generally." It is never a 500.
4. Otherwise, General: today's path, unchanged.

Every response gains:

```json
"answered_by": {"agent": "negotiation", "label": "Negotiation agent",
                "source": "screen", "note": null}
```

and `/ask/stream` sends it in a `stage` event as soon as the responder is chosen, so the
panel can show who is answering before the answer arrives.

The choice is recorded with the existing `record_action` call (agent, source, screen), so
it is auditable which agent answered what.

**Streaming.** `reason` does not stream prose. On an agent route, `/ask/stream` sends
`stage: thinking` events (one per tool round, using the tool's plain-English name), then
the whole answer in a single `delta`, then `done`. That is slower to first words than
General. This trade-off was accepted with the decision to answer automatically.

### 3.5 Screens (beyond_procwise_ui)

- **Chat:** `askRequestBody` sends `agent` from the conversation's `agent_id`, with the
  `wf:` prefix stripped. A General conversation sends `agent: "general"`. The picker
  becomes real with no layout change.
- **SpendIQ Ask panel:** sends `screen: current` on every message. Above the thread, a
  small line reads "Answering: Negotiation agent ▾". The dropdown lists General plus the
  available agents (from the existing `GET /agents` listing) and holds the person's choice
  for the rest of that panel session. Choosing one sends `agent` from then on. The line is
  built from `answered_by` on each answer, so it always shows who actually answered,
  including a fallback with its note. It uses the existing popover style.

## 4. Out of scope

- Routing a question by its *content* (intent detection). The screen and the person's
  choice decide; the model does not.
- Agents handing a question on to each other.
- New agents for Contracts or Renewals.

## 5. Testing

- `choose_responder`: every row of 3.2, including an unknown agent, a mapped agent not
  loaded, `general` on a mapped screen, and an unreadable map.
- `answer_question`: an analytic question on the Negotiations screen still returns the
  analytic answer. A non-analytic one calls `negotiation.reason` (stubbed). A `reason` that
  raises falls back to General with the note. `agent: "general"` on a mapped screen
  matches today's output exactly.
- `/ask` and `/ask/stream`: `answered_by` is present, and the stream emits it before the
  first `delta`.
- UI: `askRequestBody` sends `agent`/`screen`, and the panel line follows `answered_by`.
- Guard check: break the screen map on purpose (map `negotiations` to a slug that doesn't
  exist) and see the fallback note in the response, so the fallback is proven, not assumed.
- Live: on the local server against bp_testdb, ask one question on Negotiations, one on
  Quotes and one on Home. Then switch the Negotiations panel to General and confirm the
  answer matches today's.
