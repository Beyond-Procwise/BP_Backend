# Ask Agent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A question typed into an Ask box is answered by the agent that owns the screen it
was asked on, or by the agent the person picked, and every answer says who answered and why.

**Architecture:** A pure decision function (`src/services/ask_routing.py`) picks the
responder from the person's choice, the screen and a governed screen→agent map. The answer
pipeline asks it once per question. Exact-figure analytic answers still win first. An agent
answers through its existing `BaseAgent.reason`, and any failure falls back to today's
General path with a note. Both UIs send `agent` / `screen` and show `answered_by`.

**Tech Stack:** Python 3 / FastAPI / pytest (BP_Backend); React + a classic-script engine /
vitest (beyond_procwise_ui); PostgreSQL `proc.bp_policy` for the map.

**Spec:** `specs/2026-10-15-ask-agent-routing-design.md` (commit d45e9579).

## Global Constraints

- Every new `AskRequest` field is optional. A caller that sends neither `agent` nor `screen` gets exactly today's behaviour.
- The literal `general` as `agent` forces today's path.
- Analytic (exact-figure) answers run before routing, on every screen.
- A routing or agent failure never returns an error to the person: General answers, with a note naming the agent.
- The screen map lives in `proc.bp_policy` with `policy_identifier = 'ask_screen_agents'`. An unreadable map means every screen answers as General.
- New tables or indexes: none. New policy rows use the existing `bp_policy` shape.
- Do not commit with `-a`, `git add .` or `git commit -o` without a pathspec. The checkout is shared with other sessions; stage only the files a task names.
- Backend tests run as:
  `set -a; . ./.env; set +a; export CUDA_VISIBLE_DEVICES="" OLLAMA_BASE_URL=http://127.0.0.1:9 OLLAMA_HOST=http://127.0.0.1:9; ./venv/bin/python -m pytest <path> -q -p no:randomly`
- UI tests run from `../beyond_procwise_ui` as `npx vitest run <path>`.

### Deviations from the spec (stated, so review can reject them)

1. **Per-person agent permission.** The spec says "available" is intersected with what the
   person's role may run. No per-agent role rule exists today, and inventing one is out of
   scope. The per-person check is the existing agent-policy gate, which sees every tool call
   with the `user_id` (`agentnick_control.reason(user_id=...)`). Agents that send or act
   (email dispatch, drafting, watcher, extraction, supplier interaction) are not offered for
   Ask at all.
2. **Streaming.** `reason` has no per-round hook. The agent route emits one
   `stage: agent_working` event, then the whole answer as one `delta`. It does not emit one
   stage per tool round.
3. **`retrieved_documents`** is `[]` on an agent answer. `ToolCall` carries no document
   sources to report.

## Review Focus

1. **A conversation started with a created agent whose definition has since been deleted.** Chat still sends its slug. Expected: General answers with "… is not available", never a 500. Pinned in Task 1 (`test_unknown_agent_falls_back_with_a_note`) and Task 4 (`test_a_slug_not_loaded_answers_generally`).
2. **An agent whose `reason` returns `error` with an empty answer** (model down, tool loop exhausted). Expected: General answers with the note. Pinned in Task 3 (`test_an_errored_run_is_not_an_answer`).
3. **An analytic question asked on a mapped screen** ("top 10 suppliers by spend" on Negotiations). Expected: the exact-figure answer, labelled as from the records. Pinned in Task 4 (`test_an_analytic_question_beats_the_screen_agent`).
4. **The `wf:` prefix on Chat agent ids.** Chat stores promoted agents as `wf:<slug>`. Expected: the backend receives `<slug>`. Pinned in Task 6 (`strips the wf: prefix`).
6. **A question asked with an uploaded file on a mapped screen.** Expected: General reads the file; the agent is not asked (it cannot see uploads). Pinned in Task 4 (`test_a_question_with_an_uploaded_file_is_answered_generally`).
5. **A map row naming a sending agent** (an admin maps a screen to `email_dispatch`). Expected: General, never an agent that can send. Pinned in Task 1 (`test_a_mapped_sending_agent_is_never_offered`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/services/ask_routing.py` (new) | Decide who answers (pure), list askable agents, read the screen map, run one agent answer. |
| `deploy/sql/2026-10-15_ask_screen_agents.sql` (+ `_rollback.sql`) (new) | The governed screen→agent map. |
| `src/services/model_selector.py` (modify `answer_question`) | Ask the router, take the agent branch, stamp `answered_by`. |
| `src/api/routers/workflows.py` (modify) | `agent` / `screen` on `AskRequest`; `answered_by` on `done`; `GET /workflows/ask/agents`. |
| `tests/services/test_ask_routing.py` (new) | Decision table, map reader, agent answer. |
| `tests/services/test_ask_routes_to_agents.py` (new) | The pipeline branch. |
| `tests/api/test_ask_agent_fields.py` (new) | The endpoint contract. |
| `beyond_procwise_ui/src/modules/Chat/askRequest.js`, `useChat.js`, `ChatWorkspace.jsx` | Send the conversation's agent; show a fallback note. |
| `beyond_procwise_ui/src/modules/SpendIQ/askAgent.js` (new) | Pure helpers for the panel's agent line. |
| `beyond_procwise_ui/src/modules/SpendIQ/index.jsx`, `engine.js` | Send `screen`/`agent`; draw "Answering: X ▾". |

---

### Task 1: The decision function and the askable-agent list

**Files:**
- Create: `src/services/ask_routing.py`
- Test: `tests/services/test_ask_routing.py`

**Interfaces:**
- Produces:
  - `GENERAL = "general"`
  - `NOT_ASKABLE: frozenset[str]`
  - `@dataclass(frozen=True) class Responder: slug: Optional[str]; source: str; note: Optional[str] = None` with `def answered_by(self, label: Optional[str] = None) -> Dict[str, Any]`
  - `def agent_label(slug: Optional[str]) -> str`
  - `def askable_agents(registry: Mapping[str, Any]) -> List[str]`
  - `def choose_responder(agent: Optional[str], screen: Optional[str], available: Iterable[str], screen_map: Mapping[str, str]) -> Responder`

- [ ] **Step 1: Write the failing tests**

```python
"""Who answers a question typed into an Ask box.

The person's choice wins, then the screen's agent, then General. A choice or a
map entry that names an agent that is not there is answered by General, with a
note, never by an error.
"""

from src.services.ask_routing import (
    GENERAL, Responder, agent_label, askable_agents, choose_responder,
)

AVAILABLE = ["negotiation", "quote_comparison", "supplier_ranking"]
SCREENS = {"negotiations": "negotiation", "quotes": "quote_comparison",
           "invoices": "discrepancy_detection", "outbox": "email_dispatch"}


def test_a_chosen_agent_answers():
    r = choose_responder("quote_comparison", "negotiations", AVAILABLE, SCREENS)
    assert r == Responder("quote_comparison", "chosen")


def test_general_chosen_on_a_mapped_screen_is_general():
    r = choose_responder(GENERAL, "negotiations", AVAILABLE, SCREENS)
    assert r == Responder(None, "chosen")


def test_unknown_agent_falls_back_with_a_note():
    r = choose_responder("deleted_agent", "negotiations", AVAILABLE, SCREENS)
    assert r.slug is None and r.source == "fallback"
    assert r.note == "Deleted agent is not available, so this was answered generally."


def test_the_screen_agent_answers_when_nobody_chose():
    assert choose_responder(None, "negotiations", AVAILABLE, SCREENS) == Responder("negotiation", "screen")


def test_a_mapped_agent_that_is_not_loaded_falls_back():
    r = choose_responder(None, "invoices", AVAILABLE, SCREENS)
    assert r.slug is None and r.source == "fallback"
    assert "Discrepancy detection agent" in r.note


def test_a_mapped_sending_agent_is_never_offered():
    # email_dispatch can be loaded and still must not answer a question.
    registry = {s: object() for s in AVAILABLE + ["email_dispatch"]}
    r = choose_responder(None, "outbox", askable_agents(registry), SCREENS)
    assert r.slug is None and r.source == "fallback"


def test_an_unmapped_screen_and_no_choice_is_general():
    assert choose_responder(None, "home", AVAILABLE, SCREENS) == Responder(None, "default")
    assert choose_responder(None, None, AVAILABLE, {}) == Responder(None, "default")


def test_blank_and_cased_inputs_are_normalised():
    assert choose_responder("  Negotiation ", None, AVAILABLE, {}) == Responder("negotiation", "chosen")
    assert choose_responder("", "NEGOTIATIONS", AVAILABLE, SCREENS) == Responder("negotiation", "screen")


def test_askable_agents_drops_the_ones_that_send_or_act():
    registry = {s: object() for s in ["negotiation", "email_dispatch", "email_drafting",
                                     "email_watcher", "data_extraction", "supplier_interaction"]}
    assert askable_agents(registry) == ["negotiation"]


def test_labels_read_as_names():
    assert agent_label("quote_comparison") == "Quote comparison agent"
    assert agent_label(None) == "General"


def test_answered_by_shape():
    assert Responder("negotiation", "screen").answered_by() == {
        "agent": "negotiation", "label": "Negotiation agent", "source": "screen", "note": None}
    assert Responder(None, "default").answered_by()["agent"] is None
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `./venv/bin/python -m pytest tests/services/test_ask_routing.py -q -p no:randomly` (with the env from Global Constraints)
Expected: FAIL, `ModuleNotFoundError: No module named 'src.services.ask_routing'`

- [ ] **Step 3: Implement**

```python
"""Who answers a question typed into an Ask box.

Until this module, every question went down one pipeline whatever screen it was
asked on, and Chat's agent picker was a label: the agent a conversation was
started "with" was never sent. Now the person's choice wins, then the agent the
screen belongs to, then General, which is the pipeline as it always was.

`choose_responder` is pure, so the whole table is tested without a database or a
model. Spec: specs/2026-10-15-ask-agent-routing-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

GENERAL = "general"

#: Agents that send, act on documents, or exist as pipeline steps. They are never
#: offered a question, even when a map row names one: an Ask must not be the way
#: an email goes out.
NOT_ASKABLE = frozenset({
    "email_dispatch", "email_drafting", "email_watcher",
    "data_extraction", "supplier_interaction",
})


def _norm(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text or None


def agent_label(slug: Optional[str]) -> str:
    if not slug:
        return "General"
    return slug.replace("_", " ").strip().capitalize() + " agent"


@dataclass(frozen=True)
class Responder:
    slug: Optional[str]          # None means General
    source: str                  # chosen | screen | fallback | default | records
    note: Optional[str] = None

    def answered_by(self, label: Optional[str] = None) -> Dict[str, Any]:
        return {"agent": self.slug, "label": label or agent_label(self.slug),
                "source": self.source, "note": self.note}


def _unavailable(slug: str) -> Responder:
    return Responder(None, "fallback",
                     f"{agent_label(slug)} is not available, so this was answered generally.")


def askable_agents(registry: Mapping[str, Any]) -> List[str]:
    return sorted(s for s in registry.keys() if s not in NOT_ASKABLE)


def choose_responder(agent: Optional[str], screen: Optional[str],
                     available: Iterable[str], screen_map: Mapping[str, str]) -> Responder:
    pool = {s.lower() for s in available}
    chosen = _norm(agent)
    if chosen == GENERAL:
        return Responder(None, "chosen")
    if chosen:
        return Responder(chosen, "chosen") if chosen in pool else _unavailable(chosen)
    mapped = _norm((screen_map or {}).get(_norm(screen) or ""))
    if mapped:
        return Responder(mapped, "screen") if mapped in pool else _unavailable(mapped)
    return Responder(None, "default")
```

`screen_map` keys are stored lower-case (Task 2 guarantees it), so `_norm(screen)` matches them.

- [ ] **Step 4: Run the tests to see them pass**

Run: same as Step 2. Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/ask_routing.py tests/services/test_ask_routing.py
git commit -m "feat(ask): decide who answers — the person's choice, the screen's agent, or General" -- src/services/ask_routing.py tests/services/test_ask_routing.py
```

---

### Task 2: The governed screen→agent map

**Files:**
- Create: `deploy/sql/2026-10-15_ask_screen_agents.sql`, `deploy/sql/2026-10-15_ask_screen_agents_rollback.sql`
- Modify: `src/services/ask_routing.py` (add `screen_agent_map`)
- Test: `tests/services/test_ask_routing.py` (append)

**Interfaces:**
- Consumes: `src.services.governed_limits.rules(policy: str) -> Dict[str, Any]` (raises `LimitUnavailable`)
- Produces: `POLICY = "ask_screen_agents"`; `def screen_agent_map(read: Callable[[str], Dict[str, Any]] = None) -> Dict[str, str]`

- [ ] **Step 1: Write the failing tests** (append)

```python
from src.services import ask_routing
from src.services.governed_limits import LimitUnavailable


def test_the_map_is_read_from_policy_and_lower_cased():
    read = lambda policy: {"screens": {"Negotiations": "Negotiation", "quotes": "quote_comparison"}}
    assert ask_routing.screen_agent_map(read) == {
        "negotiations": "negotiation", "quotes": "quote_comparison"}


def test_an_unreadable_map_routes_nothing():
    def read(policy):
        raise LimitUnavailable("no active ask_screen_agents policy")
    assert ask_routing.screen_agent_map(read) == {}


def test_a_malformed_map_routes_nothing():
    assert ask_routing.screen_agent_map(lambda p: {"screens": ["negotiations"]}) == {}
    assert ask_routing.screen_agent_map(lambda p: {}) == {}


def test_the_policy_name_read_is_ask_screen_agents():
    seen = []
    ask_routing.screen_agent_map(lambda p: seen.append(p) or {"screens": {}})
    assert seen == ["ask_screen_agents"]
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: ... has no attribute 'screen_agent_map'`)

- [ ] **Step 3: Implement** (append to `ask_routing.py`)

```python
import logging
from typing import Callable

logger = logging.getLogger(__name__)

POLICY = "ask_screen_agents"


def screen_agent_map(read: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, str]:
    """The screen→agent map an admin governs, or {} when it cannot be read.

    Empty is safe here, unlike a governed limit: no map means every screen
    answers as General, which is the Ask box as it was. A routing outage must
    not make every question fail.
    """
    if read is None:
        from src.services import governed_limits
        read = governed_limits.rules
    try:
        screens = (read(POLICY) or {}).get("screens")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask screen map unreadable; every screen answers as General: %s", exc)
        return {}
    if not isinstance(screens, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip().lower()
            for k, v in screens.items() if str(k).strip() and str(v).strip()}
```

`governed_limits.rules` caches per process. A map edit applies after `governed_limits.reset_cache()` or a restart. That is the same behaviour as every other governed rule read through `rules`.

- [ ] **Step 4: Write the migration and rollback**

`deploy/sql/2026-10-15_ask_screen_agents.sql`:

```sql
-- Which agent answers a question typed on each SpendIQ screen.
--
-- Keys are SpendIQ screen ids (engine.js `current`), values are agent slugs
-- loaded in agent_nick.agents. A screen not listed answers as General, and so
-- does a screen whose agent is not loaded. Sending agents are never used even
-- if listed (src/services/ask_routing.py NOT_ASKABLE).
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code that reads it.

BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'AskScreenAgentsPolicy', 'routing',
       'Which agent answers a question asked on each screen. A screen not listed answers as General.',
       jsonb_build_object(
         'policy_identifier', 'ask_screen_agents',
         'rules', jsonb_build_object('screens', jsonb_build_object(
             'negotiations', 'negotiation',
             'quotes',       'quote_comparison',
             'quotebuilder', 'quote_comparison',
             'suppliers',    'supplier_ranking',
             'tprm',         'supplier_ranking',
             'actions',      'discrepancy_detection',
             'invoices',     'discrepancy_detection',
             'pos',          'discrepancy_detection',
             'compliance',   'discrepancy_detection'))),
       '', 1, 1, now(), 'ask_routing', now(), 'ask_routing'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'ask_screen_agents'
       AND p.policy_status = 1);

COMMIT;
```

`deploy/sql/2026-10-15_ask_screen_agents_rollback.sql`:

```sql
-- Rollback for 2026-10-15_ask_screen_agents.sql. Safe at any time: with the row
-- gone every screen answers as General, which is the Ask box before this change.

BEGIN;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'ask_screen_agents'
   AND created_by = 'ask_routing';

COMMIT;
```

- [ ] **Step 5: Check that `policy_type = 'routing'` is accepted.** Run:
  `psql "$DATABASE_URL" -c "select distinct policy_type from proc.bp_policy"` (connection from `.env`, `bp_testdb`).
  If a check constraint or the policy engine rejects unknown types (`grep -n "policy_type" src/engines/policy_engine.py`), use `'limit'` instead, as the sales-quote row does.

- [ ] **Step 6: Apply to bp_testdb only, and confirm the engine reads it**

```bash
psql "$DATABASE_URL" -f deploy/sql/2026-10-15_ask_screen_agents.sql
./venv/bin/python -c "from src.services import ask_routing; print(ask_routing.screen_agent_map())"
```
Expected: the nine-entry dict. **Do not apply to bp_sqldb.** That needs Nick's go-ahead, so record it as pending in the final report.

- [ ] **Step 7: Run tests, expect PASS (15 in the file). Commit.**

```bash
git add deploy/sql/2026-10-15_ask_screen_agents.sql deploy/sql/2026-10-15_ask_screen_agents_rollback.sql src/services/ask_routing.py tests/services/test_ask_routing.py
git commit -m "feat(ask): the screen-to-agent map is governed policy" -- deploy/sql/2026-10-15_ask_screen_agents.sql deploy/sql/2026-10-15_ask_screen_agents_rollback.sql src/services/ask_routing.py tests/services/test_ask_routing.py
```

---

### Task 3: One agent answers one question

**Files:**
- Modify: `src/services/ask_routing.py` (add `agent_answer`)
- Test: `tests/services/test_ask_routing.py` (append)

**Interfaces:**
- Consumes: `agent.reason(task: str, **kwargs) -> ToolRunResult` (`src/agents/base_agent.py:349`); `ToolRunResult.answer: str`, `.error: Optional[str]`
- Produces: `def agent_answer(agent: Any, query: str, *, user_id: Optional[str], context: Optional[str]) -> Optional[str]`. Returns the answer text, or `None` meaning "fall back".

- [ ] **Step 1: Write the failing tests** (append)

```python
from src.services.tool_runtime import ToolRunResult


class _Agent:
    def __init__(self, result=None, raises=None):
        self.result, self.raises, self.seen = result, raises, {}

    def reason(self, task, **kwargs):
        self.seen = {"task": task, **kwargs}
        if self.raises:
            raise self.raises
        return self.result


def test_the_agent_answers_with_the_person_and_the_screen():
    agent = _Agent(ToolRunResult(answer="Counter at £455."))
    text = ask_routing.agent_answer(agent, "what should I counter at?",
                                    user_id="nick", context="Screen: Negotiations")
    assert text == "Counter at £455."
    assert agent.seen["task"] == "what should I counter at?"
    assert agent.seen["user_id"] == "nick"
    assert "Screen: Negotiations" in agent.seen["extra_system"]


def test_an_errored_run_is_not_an_answer():
    assert ask_routing.agent_answer(_Agent(ToolRunResult(answer="", error="model down")),
                                    "q", user_id="u", context=None) is None


def test_a_blank_answer_is_not_an_answer():
    assert ask_routing.agent_answer(_Agent(ToolRunResult(answer="   ")),
                                    "q", user_id="u", context=None) is None


def test_a_raising_agent_is_not_an_answer():
    assert ask_routing.agent_answer(_Agent(raises=RuntimeError("boom")),
                                    "q", user_id="u", context=None) is None


def test_no_context_sends_no_extra_system():
    agent = _Agent(ToolRunResult(answer="ok"))
    ask_routing.agent_answer(agent, "q", user_id="u", context=None)
    assert "extra_system" not in agent.seen
```

- [ ] **Step 2: Run, expect FAIL** (no `agent_answer`)

- [ ] **Step 3: Implement** (append)

```python
def agent_answer(agent: Any, query: str, *, user_id: Optional[str],
                 context: Optional[str]) -> Optional[str]:
    """The agent's answer to one question, or None, meaning "answer it generally".

    The agent reasons under its own instructions, tools and policies
    (BaseAgent.reason). `user_id` reaches the agent-policy gate on every tool
    call, which is the per-person check on what this agent may do for them.
    """
    kwargs: Dict[str, Any] = {"user_id": user_id}
    if context:
        kwargs["extra_system"] = context
    try:
        result = agent.reason(query, **kwargs)
    except Exception:  # noqa: BLE001 - a broken agent is a General answer, not a 500
        logger.exception("ask agent %s failed", type(agent).__name__)
        return None
    text = str(getattr(result, "answer", "") or "").strip()
    if getattr(result, "error", None) and not text:
        logger.warning("ask agent %s errored: %s", type(agent).__name__, result.error)
        return None
    return text or None
```

- [ ] **Step 4: Run, expect PASS (20). Commit.**

```bash
git commit -m "feat(ask): one agent answers one question, or hands it back" -- src/services/ask_routing.py tests/services/test_ask_routing.py
```

---

### Task 4: The pipeline routes

**Files:**
- Modify: `src/services/model_selector.py`: `answer_question` (line 2532) signature and body
- Test: `tests/services/test_ask_routes_to_agents.py` (new)

**Interfaces:**
- Consumes: Tasks 1–3 (`choose_responder`, `askable_agents`, `screen_agent_map`, `agent_answer`, `Responder.answered_by`, `agent_label`)
- Produces: `RAGPipeline.answer_question(..., agent: Optional[str] = None, screen: Optional[str] = None)`. Every returned dict carries `answered_by: {agent, label, source, note}`. Emits `stage` events `routed` (with `answered_by`) and `agent_working`.

- [ ] **Step 1: Write the failing tests**

```python
"""The ask pipeline hands a question to the agent the screen or the person names.

Built without its constructor, as test_ask_routes_analytic_questions does: the
branch under test runs before retrieval, so a pipeline with a history manager,
an agent registry and nothing else is enough.
"""

import types

import pytest

from src.services import ask_routing
from src.services.analytics import ask as analytics_ask
from src.services.tool_runtime import ToolRunResult


class _History:
    def __init__(self, items=None):
        self.items, self.saved = list(items or []), None

    def get_history(self, user_id):
        return list(self.items)

    def save_history(self, user_id, history):
        self.saved = history


class _Agent:
    def __init__(self, answer="Counter at £455.", raises=None):
        self.answer, self.raises, self.calls = answer, raises, []

    def reason(self, task, **kwargs):
        self.calls.append((task, kwargs))
        if self.raises:
            raise self.raises
        return ToolRunResult(answer=self.answer)


def _pipeline(agents, history=None):
    from src.services.model_selector import RAGPipeline
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_llm_model = "test-model"
    p.history_manager = _History(history)
    p.agent_nick = types.SimpleNamespace(agents=agents)
    return p


@pytest.fixture(autouse=True)
def _no_analytic_no_audit(monkeypatch):
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: None)
    monkeypatch.setattr(ask_routing, "screen_agent_map",
                        lambda read=None: {"negotiations": "negotiation"})
    monkeypatch.setattr("src.services.model_selector.record_ask_route", lambda **k: None)


def test_the_screen_agent_answers():
    agent = _Agent()
    result = _pipeline({"negotiation": agent}).answer_question(
        query="what should I counter at?", user_id="nick", screen="negotiations")
    assert result["answer"] == "Counter at £455."
    assert result["answered_by"] == {"agent": "negotiation", "label": "Negotiation agent",
                                     "source": "screen", "note": None}
    assert agent.calls[0][1]["user_id"] == "nick"


def test_the_chosen_agent_beats_the_screen():
    neg, quotes = _Agent("neg"), _Agent("quotes")
    result = _pipeline({"negotiation": neg, "quote_comparison": quotes}).answer_question(
        query="q", user_id="u", screen="negotiations", agent="quote_comparison")
    assert result["answer"] == "quotes" and not neg.calls


def test_an_analytic_question_beats_the_screen_agent(monkeypatch):
    payload = {"answer": "<section>top 10</section>", "follow_ups": [],
               "retrieved_documents": [], "next_steps": []}
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: dict(payload))
    agent = _Agent()
    result = _pipeline({"negotiation": agent}).answer_question(
        query="top 10 suppliers by spend", user_id="u", screen="negotiations")
    assert result["answer"] == payload["answer"]
    assert result["answered_by"]["source"] == "records"
    assert not agent.calls


def test_a_failing_agent_falls_back_to_general_with_a_note(monkeypatch):
    # General is stubbed at its first step after routing: reaching it is the proof.
    from src.services import model_selector
    monkeypatch.setattr(model_selector.RAGPipeline, "_is_presentation_directive",
                        lambda self, q, has_history: False, raising=False)
    pipeline = _pipeline({"negotiation": _Agent(raises=RuntimeError("down"))})
    with pytest.raises(AttributeError):   # no retriever on this stub = General was reached
        pipeline.answer_question(query="q", user_id="u", screen="negotiations")


def test_a_slug_not_loaded_answers_generally():
    pipeline = _pipeline({})
    with pytest.raises(AttributeError):   # General reached: the stub has no retriever
        pipeline.answer_question(query="q", user_id="u", agent="deleted_agent")


def test_the_answer_is_kept_in_the_history():
    pipeline = _pipeline({"negotiation": _Agent()})
    pipeline.answer_question(query="q", user_id="u", screen="negotiations")
    assert pipeline.history_manager.saved[-1]["answer"] == "Counter at £455."


def test_recent_turns_reach_the_agent_as_context():
    agent = _Agent()
    _pipeline({"negotiation": agent},
              history=[{"query": "who is the supplier?", "answer": "Acme Ltd"}]).answer_question(
        query="and their last offer?", user_id="u", screen="negotiations")
    assert "Acme Ltd" in agent.calls[0][1]["extra_system"]


def test_a_question_with_an_uploaded_file_is_answered_generally():
    agent = _Agent()
    with pytest.raises(AttributeError):   # General reached: the stub has no retriever
        _pipeline({"negotiation": agent}).answer_question(
            query="q", user_id="u", screen="negotiations", files=[(b"x", "a.pdf")])
    assert not agent.calls


def test_the_route_is_announced_before_the_work():
    events = []
    _pipeline({"negotiation": _Agent()}).answer_question(
        query="q", user_id="u", screen="negotiations",
        on_event=lambda kind, payload: events.append((kind, payload)))
    stages = [p.get("stage") for k, p in events if k == "stage"]
    assert stages[:2] == ["routed", "agent_working"]
    assert events[0][1]["answered_by"]["agent"] == "negotiation"
    assert ("delta", {"text": "Counter at £455."}) in events
```

Fallback-to-General tests use the stub's missing retriever. If General is reached, it fails with `AttributeError` on a missing attribute, which is the same proof `test_a_question_the_layer_declines_goes_down_the_old_path` uses. For the fallback *note*, the endpoint test in Task 5 checks the stamped `answered_by` with a stub pipeline. Here, add one direct check of the stamping helper:

```python
def test_a_fallback_note_is_stamped_on_the_general_answer(monkeypatch):
    from src.services import model_selector
    pipeline = _pipeline({"negotiation": _Agent(raises=RuntimeError("down"))})
    monkeypatch.setattr(model_selector.RAGPipeline, "_answer_general",
                        lambda self, **kw: {"answer": "general"})
    result = pipeline.answer_question(query="q", user_id="u", screen="negotiations")
    assert result["answer"] == "general"
    assert result["answered_by"]["source"] == "fallback"
    assert result["answered_by"]["note"] == (
        "The Negotiation agent could not answer, so this was answered generally.")
```

- [ ] **Step 2: Run, expect FAIL** (`TypeError: answer_question() got an unexpected keyword argument 'screen'`)

- [ ] **Step 3: Implement.** In `src/services/model_selector.py`:

(a) Rename the existing method `answer_question` to `_answer_general` and give it one extra keyword-only parameter `routing: Optional[Dict[str, Any]] = None`. Leave its body unchanged except for (c).

(b) Add the new public method directly above it:

```python
    def answer_question(
        self,
        query: str,
        user_id: str,
        *,
        agent: Optional[str] = None,
        screen: Optional[str] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> Dict:
        """Answer a question: the agent the person chose, else the screen's agent, else General.

        General is the pipeline this method used to be (now ``_answer_general``),
        and it still answers first when the question has one exact answer.
        Every result says who answered in ``answered_by``. Spec:
        specs/2026-10-15-ask-agent-routing-design.md.
        """
        from src.services import ask_routing

        registry = getattr(getattr(self, "agent_nick", None), "agents", None) or {}
        responder = ask_routing.choose_responder(
            agent, screen, ask_routing.askable_agents(registry),
            ask_routing.screen_agent_map() if screen and not agent else {})
        routing = {"responder": responder, "answered_by": responder.answered_by()}
        if on_event is not None:
            try:
                on_event("stage", {"stage": "routed", "answered_by": routing["answered_by"]})
            except Exception:  # pragma: no cover - a broken consumer must not stop the answer
                logger.debug("ask on_event consumer raised", exc_info=True)
        result = self._answer_general(query=query, user_id=user_id, on_event=on_event,
                                      routing=routing, **kwargs)
        if isinstance(result, dict):
            result.setdefault("answered_by", routing["answered_by"])
        record_ask_route(user_id=user_id, screen=screen, answered_by=result.get("answered_by")
                         if isinstance(result, dict) else routing["answered_by"])
        return result
```

(c) Inside `_answer_general`, change the analytic block's return path and add the agent branch **immediately after** the `if analytic is not None:` block and **before** `_emit("stage", stage="retrieving")`:

```python
        if analytic is not None:
            history.append({"query": self._redact_identifiers(query),
                            "answer": analytic["answer"]})
            self.history_manager.save_history(user_id, history)
            _emit("stage", stage="answer_complete")
            analytic["answered_by"] = _records_answered_by()
            return analytic

        responder = (routing or {}).get("responder")
        # An uploaded file is read by the general path only; an agent would answer without it.
        if responder is not None and responder.slug and not files:
            agent_obj = (getattr(self.agent_nick, "agents", None) or {}).get(responder.slug)
            _emit("stage", stage="agent_working", answered_by=routing["answered_by"])
            context = self._agent_context(history, screen_context)
            text = ask_routing.agent_answer(agent_obj, query, user_id=user_id, context=context) \
                if agent_obj is not None else None
            if text:
                _emit("delta", text=text)
                _emit("stage", stage="answer_complete")
                history.append({"query": self._redact_identifiers(query), "answer": text})
                self.history_manager.save_history(user_id, history)
                return {"answer": text, "follow_ups": [], "retrieved_documents": [],
                        "next_steps": [], "answered_by": routing["answered_by"]}
            routing["answered_by"] = ask_routing.Responder(
                None, "fallback",
                f"The {ask_routing.agent_label(responder.slug)} could not answer, "
                "so this was answered generally.").answered_by()
```

with these module-level helpers in `model_selector.py`:

```python
from src.services import ask_routing  # top-level import, beside the analytics_ask import


def _records_answered_by() -> Dict[str, Any]:
    return ask_routing.Responder(None, "records").answered_by("Answered from the records")


def record_ask_route(*, user_id: Optional[str], screen: Optional[str],
                     answered_by: Optional[Dict[str, Any]]) -> None:
    """Which agent answered, for the record. Best-effort, like every read's audit row."""
    ab = answered_by or {}
    record_action(phase="ask", action_type="ask_routed", agent=ab.get("agent") or "general",
                  status="ok", summary=ab.get("label"),
                  details={"source": ab.get("source"), "screen": screen,
                           "note": ab.get("note"), "user_id": user_id})
```

`record_action` is imported from `src.services.agent_actions` (add the import if `model_selector.py` lacks it: `grep -n "record_action" src/services/model_selector.py`).

and this method on `RAGPipeline`:

```python
    def _agent_context(self, history: List[Dict[str, Any]], screen_context: Optional[str]) -> Optional[str]:
        """What the agent is told besides the question: the screen, and the last three turns."""
        parts: List[str] = []
        if isinstance(screen_context, str) and screen_context.strip():
            parts.append("What the person is looking at: " + screen_context.strip()[:500])
        turns = [
            f"Q: {str(h.get('query', '')).strip()}\nA: {self._history_answer_as_text(str(h.get('answer', '')).strip())[:600]}"
            for h in (history or [])[-3:] if h.get("query")
        ]
        if turns:
            parts.append("Earlier in this conversation:\n" + "\n\n".join(turns))
        return "\n\n".join(parts) or None
```

The order is right: presentation directive, then analytic, then agent, then retrieval. `screen_context` is already a parameter of `_answer_general`.

- [ ] **Step 4: Run the new tests, plus the existing ask tests that call `answer_question`, so the rename broke nothing**

Run: `./venv/bin/python -m pytest tests/services/test_ask_routes_to_agents.py tests/services/test_ask_routes_analytic_questions.py tests/services/test_ask_stream_answer_complete.py tests/services/test_ask_keep_alive_top_level.py tests/services/test_ask_persona_currency_rule.py tests/test_rag_pipeline_policy_mode.py -q -p no:randomly`
Expected: all pass. If an existing test compares a whole result dict with `==`, it now also sees `answered_by`. Update that assertion to compare `{k: v for k, v in result.items() if k != "answered_by"}`. Do not drop the stamp.

- [ ] **Step 5: Prove the fallback guard fails when broken.** Temporarily change `if text:` to `if True:`, then run `test_a_fallback_note_is_stamped_on_the_general_answer`. It must FAIL. Revert, run it again, and it must PASS.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(ask): the screen's or the chosen agent answers; every answer says who" -- src/services/model_selector.py tests/services/test_ask_routes_to_agents.py
```
(Add any existing test file you touched in Step 4 to the pathspec.)

---

### Task 5: The endpoint carries `agent` / `screen`, returns `answered_by`, and lists askable agents

**Files:**
- Modify: `src/api/routers/workflows.py`: `AskRequest` (line 55), `ask_question` (881), `ask_question_stream` (933), new route
- Test: `tests/api/test_ask_agent_fields.py` (new)

**Interfaces:**
- Consumes: `RAGPipeline.answer_question(agent=, screen=)`; `ask_routing.askable_agents`, `agent_label`
- Produces: `AskRequest.agent: Optional[str]` (max 128), `AskRequest.screen: Optional[str]` (max 64); `done` event field `answered_by`; `GET /workflows/ask/agents -> {"agents": [{"slug", "label"}]}`

- [ ] **Step 1: Write the failing tests**

```python
"""The ask endpoint carries who should answer, and says who did."""

import asyncio
import json
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.routers import workflows  # noqa: E402
from src.api.routers.workflows import AskRequest, ask_question, ask_question_stream  # noqa: E402

ANSWERED_BY = {"agent": "negotiation", "label": "Negotiation agent", "source": "screen", "note": None}


class _Pipeline:
    def __init__(self):
        self.seen = {}

    def answer_question(self, **kwargs):
        self.seen = kwargs
        cb = kwargs.get("on_event")
        if cb:
            cb("stage", {"stage": "routed", "answered_by": ANSWERED_BY})
        return {"answer": "x", "follow_ups": [], "retrieved_documents": [],
                "next_steps": [], "answered_by": ANSWERED_BY}


class _Request:
    headers: dict = {}


def test_agent_and_screen_reach_the_pipeline():
    p = _Pipeline()
    asyncio.run(ask_question(AskRequest(query="q", user_id="u", agent="negotiation",
                                        screen="negotiations"), _Request(), p, principal=object()))
    assert p.seen["agent"] == "negotiation" and p.seen["screen"] == "negotiations"


def test_absent_fields_reach_the_pipeline_as_none():
    p = _Pipeline()
    asyncio.run(ask_question(AskRequest(query="q", user_id="u"), _Request(), p, principal=object()))
    assert p.seen["agent"] is None and p.seen["screen"] is None


def test_answered_by_comes_back():
    result = asyncio.run(ask_question(AskRequest(query="q", user_id="u"), _Request(),
                                      _Pipeline(), principal=object()))
    assert result["answered_by"] == ANSWERED_BY


def _frames(response):
    async def drain():
        out = []
        async for chunk in response.body_iterator:
            out.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(out)
    text = asyncio.run(drain())
    return [json.loads(l[6:]) for l in text.split("\n") if l.startswith("data: ")]


def test_the_stream_says_who_answers_before_the_answer_and_in_done():
    resp = asyncio.run(ask_question_stream(AskRequest(query="q", user_id="u", screen="negotiations"),
                                           _Request(), _Pipeline(), principal=object()))
    frames = _frames(resp)
    routed = next(f for f in frames if f.get("stage") == "routed")
    done = next(f for f in frames if f["type"] == "done")
    assert routed["answered_by"] == ANSWERED_BY
    assert done["answered_by"] == ANSWERED_BY
    assert frames.index(routed) < frames.index(done)


def test_the_askable_agents_are_listed_with_names():
    nick = types.SimpleNamespace(agents={"negotiation": object(), "email_dispatch": object()})
    result = asyncio.run(workflows.list_ask_agents(agent_nick=nick, principal=object()))
    assert result == {"agents": [{"slug": "negotiation", "label": "Negotiation agent"}]}
```

If `ask_question_stream` returns a `StreamingResponse` whose iterator needs a running loop of its own, adapt `_frames` to `asyncio.run` a coroutine that both calls the endpoint and drains it. Check the existing `tests/services/test_ask_stream_answer_complete.py` for the pattern this repo already uses, and copy it.

- [ ] **Step 2: Run, expect FAIL** (`AskRequest` rejects or ignores `agent`; no `list_ask_agents`)

- [ ] **Step 3: Implement**

In `AskRequest`, after `action_id`:

```python
    agent: Optional[str] = Field(
        default=None,
        max_length=128,
        description="The agent the person chose to answer (an agent slug), or 'general' "
        "for the general answer. Absent: the screen's agent answers, if it has one.",
    )
    screen: Optional[str] = Field(
        default=None,
        max_length=64,
        description="The screen the question was asked on (SpendIQ's screen id). Picks "
        "the screen's agent when the person chose none.",
    )
```

In both `ask_question` and `_work` inside `ask_question_stream`, add to the `answer_question(...)` call:

```python
        agent=req.agent,
        screen=req.screen,
```

In the stream's `done` event dict, add:

```python
                    "answered_by": result.get("answered_by"),
```

`stage` events are already forwarded verbatim by `_on_event`, so `routed` and `agent_working` reach the client with their `answered_by` and need no further change.

New route, placed after `ask_question_stream`:

```python
@router.get("/ask/agents")
async def list_ask_agents(
    agent_nick=Depends(get_agent_nick),
    principal: object = Depends(require_user),
):
    """The agents a person can pick to answer in an Ask box: every loaded agent
    that does not send or act (ask_routing.NOT_ASKABLE)."""
    from src.services import ask_routing

    slugs = ask_routing.askable_agents(getattr(agent_nick, "agents", None) or {})
    return {"agents": [{"slug": s, "label": ask_routing.agent_label(s)} for s in slugs]}
```

Find `get_agent_nick`: `grep -n "def get_agent_nick" -r src/api`. Import it the way `src/api/routers/agents.py` does.

**Output safety:** the response boundary withholds JSON fields that name internal routes or agent/class names (memory: "Output-safety withholds route paths"). Read the gate (`grep -n "def " src/api/output_safety*.py` or the boundary gate in `src/api/main.py`) and confirm that `slug` values like `negotiation` and the `answered_by.agent` field survive it. If they are scrubbed, register these two fields with the gate's allow mechanism, the same way other user-facing identifiers are allowed. Do not disable the gate. Test this live in Task 8.

- [ ] **Step 4: Run, expect PASS. Then run the existing endpoint tests:** `tests/api/test_ask_carries_currency_and_steps.py`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(ask): the endpoint takes agent and screen, says who answered, lists askable agents" -- src/api/routers/workflows.py tests/api/test_ask_agent_fields.py
```

---

### Task 6: Chat sends the conversation's agent

**Files (repo `../beyond_procwise_ui`):**
- Modify: `src/modules/Chat/askRequest.js`, `src/modules/Chat/useChat.js` (ask mutation, ~line 172–330), `src/modules/Chat/ChatWorkspace.jsx` (assistant message render, ~line 650)
- Test: `src/modules/Chat/askAgent.contract.test.js` (new)

**Interfaces:**
- Consumes: backend `agent` field, `answered_by` on `stage:routed` and `done`
- Produces: `askRequestBody({..., agentId})`; `export function agentSlug(agentId)`; assistant messages carry `answered_by`

- [ ] **Step 1: Write the failing tests**

```js
// A chat started with an agent is answered by that agent. Until now the agent a
// conversation was started "with" was stored and never sent.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { askRequestBody, agentSlug } from './askRequest.js';

const here = dirname(fileURLToPath(import.meta.url));
const hook = readFileSync(join(here, 'useChat.js'), 'utf8');
const workspace = readFileSync(join(here, 'ChatWorkspace.jsx'), 'utf8');

describe('the agent a conversation is held with', () => {
  it('is sent as the agent to answer', () => {
    const body = askRequestBody({ query: 'q', userId: 'u', currency: () => 'GBP', agentId: 'negotiation' });
    expect(body.agent).toBe('negotiation');
  });

  it('strips the wf: prefix promoted agents carry', () => {
    expect(agentSlug('wf:supplier_scout')).toBe('supplier_scout');
    expect(askRequestBody({ query: 'q', currency: () => 'GBP', agentId: 'wf:supplier_scout' }).agent)
      .toBe('supplier_scout');
  });

  it('a General conversation asks for the general answer, not the screen default', () => {
    expect(askRequestBody({ query: 'q', currency: () => 'GBP', agentId: null }).agent).toBe('general');
  });

  it('the hook reads the conversation agent and passes it', () => {
    expect(hook).toMatch(/agentId:\s*conv(Agent|\.agent_id)/);
  });

  it('a fallback note is shown under the answer', () => {
    expect(hook).toMatch(/answered_by/);
    expect(workspace).toMatch(/answered_by\??\.note/);
  });
});
```

- [ ] **Step 2: Run, expect FAIL** — `npx vitest run src/modules/Chat/askAgent.contract.test.js`

- [ ] **Step 3: Implement**

`askRequest.js`, add:

```js
// Promoted workspace agents are listed as `wf:<slug>`; the backend knows the slug.
export function agentSlug(agentId) {
  const id = String(agentId || '').trim();
  return id.startsWith('wf:') ? id.slice(3) : id;
}
```

and in `askRequestBody`, take `agentId` in the destructured args and add before `return body`:

```js
  // A conversation held with an agent is answered by it; a General one asks for
  // the general answer by name, so no screen default can stand in for it.
  body.agent = agentSlug(agentId) || 'general';
```

`useChat.js`, in the `ask` mutation before the `fetch`:

```js
      const convAgent = (readStore(userId).find((c) => c.conversation_id === id) || {}).agent_id || null;
```

change the body line to `askRequestBody({ query, userId, sessionId: id, actionId, agentId: convAgent })`, and in the stream loop:

```js
          if (evt.type === 'stage') {
            patchLast({
              stage: evt.stage,
              ...(evt.sources ? { sources: evt.sources } : {}),
              ...(evt.answered_by ? { answered_by: evt.answered_by } : {}),
            });
```

In the `done` branch, keep `evt.answered_by` (`answeredBy = evt.answered_by || null;`, declared beside `nextSteps`) and add `answered_by: answeredBy` to the final `patchLast`. Add `stage: 'agent_working'` to the stage-label map the chat uses (`grep -n "retrieving" src/modules/Chat/ChatWorkspace.jsx`) with the text `t('chat.stageAgentWorking')`, and add that key to the i18n source catalogue next to the existing `chat.*` stage keys (English: "The agent is working on it…").

`ChatWorkspace.jsx`: under the assistant message body (beside where `follow_ups`/`next_steps` render), add:

```jsx
{m.answered_by?.note && (
  <div style={css('margin-top:6px;font-size:11.5px;color:#9aa3ad;')}>{m.answered_by.note}</div>
)}
```

- [ ] **Step 4: Run, expect PASS, plus `npx vitest run src/modules/Chat`.** Expected: all Chat tests pass.

- [ ] **Step 5: Commit (UI repo; check `git status` first, and stage only these files)**

```bash
git -C ../beyond_procwise_ui add src/modules/Chat/askRequest.js src/modules/Chat/useChat.js src/modules/Chat/ChatWorkspace.jsx src/modules/Chat/askAgent.contract.test.js <i18n catalogue file>
git -C ../beyond_procwise_ui commit -m "feat(chat): a conversation held with an agent is answered by it" -- <same paths>
```

---

### Task 7: SpendIQ's Ask panel sends the screen and shows "Answering: X ▾"

**Files (repo `../beyond_procwise_ui`):**
- Create: `src/modules/SpendIQ/askAgent.js`
- Modify: `src/modules/SpendIQ/index.jsx` (`askBody`, ~line 776), `src/modules/SpendIQ/engine.js` (panel state ~18925, `askPanelSend` ~18993, `askPanel()` ~19273, `newAskPanelThread`)
- Test: `src/modules/SpendIQ/askAgent.contract.test.js` (new)

**Interfaces:**
- Consumes: `GET /workflows/ask/agents`; `answered_by` on `stage:routed` and `done`
- Produces: `askRouteFields({screen, choice})`, `answeringLabel(answeredBy, fallbackLabel)`; engine globals `askPanelAgent` (person's choice; `null` = screen default) and `askPanelAnsweredBy`

- [ ] **Step 1: Write the failing tests**

```js
// The Ask panel tells the backend which screen it is on, lets the person pick
// who answers, and names who actually did.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { askRouteFields, answeringLabel } from './askAgent.js';

const here = dirname(fileURLToPath(import.meta.url));
const engine = readFileSync(join(here, 'engine.js'), 'utf8');
const bridge = readFileSync(join(here, 'index.jsx'), 'utf8');

describe('what the panel sends', () => {
  it('sends the screen, and no agent when the person chose none', () => {
    expect(askRouteFields({ screen: 'negotiations', choice: null }))
      .toEqual({ screen: 'negotiations' });
  });
  it('sends the chosen agent', () => {
    expect(askRouteFields({ screen: 'negotiations', choice: 'quote_comparison' }))
      .toEqual({ screen: 'negotiations', agent: 'quote_comparison' });
  });
  it('sends general when the person chose General', () => {
    expect(askRouteFields({ screen: 'quotes', choice: 'general' }).agent).toBe('general');
  });
  it('the bridge puts them on the request', () => {
    expect(bridge).toMatch(/opts\.route/);
  });
});

describe('the Answering line', () => {
  it('names who answered', () => {
    expect(answeringLabel({ label: 'Negotiation agent', source: 'screen' })).toBe('Negotiation agent');
  });
  it('before any answer, names the default it will ask', () => {
    expect(answeringLabel(null, 'General')).toBe('General');
  });
  it('the panel draws the line and the switch, and reads the routed stage', () => {
    expect(engine).toMatch(/askPanelAnsweredBy/);
    expect(engine).toMatch(/stage === 'routed'|stage==='routed'/);
    expect(engine).toMatch(/\/workflows\/ask\/agents/);
  });
});
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `askAgent.js`**

```js
// The Ask panel's routing fields and its "Answering: X" label. Pure, so they are
// tested here; engine.js is a classic script and reaches them through the bridge.

export function askRouteFields({ screen, choice } = {}) {
  const out = {};
  if (screen) out.screen = String(screen);
  if (choice) out.agent = String(choice);
  return out;
}

export function answeringLabel(answeredBy, fallbackLabel = 'General') {
  return (answeredBy && answeredBy.label) || fallbackLabel;
}
```

- [ ] **Step 4: Wire the bridge (`index.jsx`)**

Import `askRouteFields` at the top beside the other SpendIQ imports. In `askBody`, add:

```js
        ...(opts && opts.route ? askRouteFields(opts.route) : {}),
```

and expose the agent list for the engine, beside the other `window.__SPENDIQ_*` bridges:

```js
      window.__SPENDIQ_ASK_AGENTS__ = () =>
        axios.get(`${AI_API}/workflows/ask/agents`).then((r) => (r.data && r.data.agents) || []);
```

In the streaming reader, the `done` branch: before `final = pickAnswer(ev);` add
`if (ev.answered_by && typeof handlers.onAnsweredBy === 'function') handlers.onAnsweredBy(ev.answered_by);`.
`stage` events already reach `handlers.onStage(ev.stage, ev)`.

- [ ] **Step 5: Wire the panel (`engine.js`)**

State, beside `askPanelStage`:

```js
// Who answers. `askPanelAgent` is the person's pick from the switch (null = the
// screen's own agent); `askPanelAnsweredBy` is what the backend said actually
// answered, so the line never claims an agent that did not.
let askPanelAgent = null;
let askPanelAnsweredBy = null;
let askPanelAgentList = null;   // [{slug,label}] once fetched; null = not yet
```

`newAskPanelThread()`: add `askPanelAnsweredBy = null;` (the person's pick survives "New").

In `askPanelSend`, change the call's 4th argument from `opts` to
`Object.assign({}, opts, { route: { screen: current, choice: askPanelAgent } })`, and add two handlers:

```js
      onAnsweredBy: (ab) => { askPanelAnsweredBy = ab; paint(); },
```

and in `onStage`, first line:

```js
        if(stage === 'routed' && ev && ev.answered_by){ askPanelAnsweredBy = ev.answered_by; }
```

and add `agent_working: 'The agent is working on it…',` to the stage map. If the answered-by carries a `note`, push it under the answer by setting `slot.note = ab.note` in `onAnsweredBy`, and render it in `askPanel()`'s message markup after `body`:

```js
${m.note ? `<div class="p-sub" style="margin-top:4px">${escH(m.note)}</div>` : ''}
```

The line and switch replace the existing `contextChip` in `askPanel()`:

```js
  const ansLabel = askPanelAnsweredBy && askPanelAnsweredBy.label
    ? askPanelAnsweredBy.label
    : (askPanelAgent ? (askPanelAgentList||[]).find(a=>a.slug===askPanelAgent)?.label || askPanelAgent
                     : T('askp.screenDefault','This screen\'s agent'));
  const opts = [`<option value="">${escH(T('askp.screenDefault','This screen\'s agent'))}</option>`,
                `<option value="general"${askPanelAgent==='general'?' selected':''}>${escH(T('askp.general','General'))}</option>`]
    .concat((askPanelAgentList||[]).map(a =>
      `<option value="${escH(a.slug)}"${askPanelAgent===a.slug?' selected':''}>${escH(a.label)}</option>`)).join('');
  const contextChip = `<div style="padding:5px 14px;font-size:.62rem;font-weight:700;letter-spacing:.02em;color:var(--ink-3);background:var(--surface-2);border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;justify-content:space-between">
      <span>${escH(T('askp.context','Context'))}: ${escH(screenViewName())}</span>
      <label style="display:flex;gap:4px;align-items:center">${escH(T('askp.answering','Answering'))}:
        <select class="siq-askp-agent" onchange="askPanelPickAgent(this.value)" onfocus="askPanelLoadAgents()" aria-label="${escH(T('askp.answering','Answering'))}" title="${escH(ansLabel)}">${opts}</select>
      </label></div>`;
```

with:

```js
function askPanelPickAgent(v){ askPanelAgent = v || null; askPanelAnsweredBy = null; renderAskPanel(); }
function askPanelLoadAgents(){
  if(askPanelAgentList || typeof window.__SPENDIQ_ASK_AGENTS__ !== 'function') return;
  askPanelAgentList = [];   // one fetch per load, even on repeated focus
  window.__SPENDIQ_ASK_AGENTS__().then(list => { askPanelAgentList = list; renderAskPanel(); })
    .catch(() => { askPanelAgentList = null; });
}
```

Call `askPanelLoadAgents()` once from `openAskPanel()` and `openAskPanelFromReport()`, so the list is ready before the first click. The visible "Answering" text is the `<select>`'s selected option plus its `title`. After an answer, the select shows the person's pick, and `ansLabel` (who actually answered, including a fallback) is shown in the `title` and in the message's note. Add `askp.answering`, `askp.screenDefault`, `askp.general` and the `agent_working` text to the i18n catalogue next to the other `askp.*` keys. Use the existing small-select style if the panel has one (`grep -n "select" src/modules/SpendIQ/*.css | head`); otherwise inherit, per "Reuse the existing UI look".

- [ ] **Step 6: Run, expect PASS, plus `npx vitest run src/modules/SpendIQ`** (all existing SpendIQ contract tests must still pass), then `npm run build`. The build is known broken by a missing `mockPipelineDeals` file. If it fails, confirm it is that same error and not a new one.

- [ ] **Step 7: Commit (UI repo, named paths only)**

```bash
git -C ../beyond_procwise_ui commit -m "feat(spendiq): the Ask panel asks the screen's agent, names who answered, and can switch" -- src/modules/SpendIQ/askAgent.js src/modules/SpendIQ/askAgent.contract.test.js src/modules/SpendIQ/index.jsx src/modules/SpendIQ/engine.js <i18n catalogue file>
```
(`git add` the new files first.)

---

### Task 8: Prove it on the local server against bp_testdb

**Files:** none changed. Record results in the spec's section 5 as a dated "Verified" note.

- [ ] **Step 1: Full backend regression under the isolated env.** Run `./venv/bin/python -m pytest tests/ -q -p no:randomly`, with the env from Global Constraints, on the commit before Task 1 (`git stash` is NOT allowed; use `git worktree add /tmp/claude-1001/.../ask-base d45e9579` for the baseline) and on HEAD. Compare the two failure lists. Expected: no new failures.

- [ ] **Step 2: Restart the local procwise service** (`sudo systemctl restart procwise`, or however `procwise.service` is run here). Other sessions restart it too, so check `systemctl status procwise` first and say in the report that you restarted it.

- [ ] **Step 3: Ask three questions through the API** with a valid token (`$TOKEN` from a signed-in session):

```bash
for body in \
 '{"query":"what should I counter at on the open negotiation?","user_id":"t","screen":"negotiations"}' \
 '{"query":"which quote is best value?","user_id":"t","screen":"quotes"}' \
 '{"query":"what needs my attention?","user_id":"t","screen":"home"}' \
 '{"query":"what should I counter at on the open negotiation?","user_id":"t","screen":"negotiations","agent":"general"}' \
 '{"query":"top 10 suppliers by spend","user_id":"t","screen":"negotiations"}'; do
  curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
    -d "$body" http://localhost:8000/workflows/ask | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("answered_by"), "|", str(d.get("answer"))[:160])'
done
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/workflows/ask/agents
```

Expected, in order: `negotiation/screen`; `quote_comparison/screen`; `None/default`; `None/chosen`; `None/records`. The agent list must name `negotiation` and must not name `email_dispatch`. If `answered_by.agent` or the slugs come back withheld or blank, the output-safety gate is scrubbing them, so return to Task 5's output-safety step.

- [ ] **Step 4: Prove the fallback live.** On bp_testdb, map `negotiations` to `nonexistent_agent`, then reset the cache by restarting or calling `governed_limits.reset_cache()` in-process. Ask again and confirm `source: fallback` with the note. Restore the row's value to `negotiation` afterwards, and confirm with one more ask.

- [ ] **Step 5: In the browser** (signed in, SpendIQ), open Negotiations, open Ask and type a question. The line shows the screen's agent, the answer arrives, and its `title` reads "Negotiation agent". Switch to General and ask again. Then open Chat, start a chat with an agent and confirm the request body carries `agent` (DevTools → Network). Chrome MCP cannot reach this host's localhost (memory), so this step is done by Nick or skipped with a note.

- [ ] **Step 6: Report**, with no pushing and no bp_sqldb migration. State: commits (both repos), test counts before and after, the five live answers, the fallback proof, and what is pending (bp_sqldb migration, push, browser check).
