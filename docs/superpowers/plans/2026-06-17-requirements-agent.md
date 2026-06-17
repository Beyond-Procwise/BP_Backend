# RequirementsAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conversation-led `RequirementsAgent` that turns a vague buyer need into a complete, structured, sourcing-ready procurement requirement, persists it to `proc.bp_requirement`, and hands it off to existing acting agents via the shared workflow blackboard.

**Architecture:** Three units with clean seams — `RequirementsAgent(BaseAgent)` is the per-turn brain (stateless per call), `RequirementSession` holds Redis-backed multi-turn state (cloned from the `NegotiationSession` pattern), and `RequirementService` owns SQL persistence, history seeding, and completeness evaluation. A single LLM call per turn both extracts answers from the user's text and asks the next question. On completion the agent emits `SUGGEST_AGENT` signals; it never ranks suppliers or drafts RFQs itself.

**Tech Stack:** Python 3, FastAPI, psycopg2 (`proc` schema, PostgreSQL), Redis (`redis-py`), Ollama (AgentNick local model via `BaseAgent.call_ollama`), pytest.

## Global Constraints

- All new DB tables/indexes use the `bp_` prefix; indexes named `ix_bp_<table>_<col>`.
- Elicitation and brief-parsing run on the **AgentNick local model** (`BeyondProcwise/AgentNick:latest`), never cloud. Use `self.call_ollama(...)` which already resolves the local model.
- **No fabrication:** if a field is absent from the user's input, leave it unset (do not invent values). Completeness is driven by genuinely-filled fields only.
- SQL migrations are additive and idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`), wrapped in `BEGIN; ... COMMIT;`, filed under `deploy/sql/`.
- Governance: agent-scoped prompts/policies resolve via `self.resolve_prompt(name, **fmt)` and `self.governing_policy(name)`; both degrade to `None` when unavailable.
- Git commit messages: NO `Co-Authored-By` / Claude attribution lines.
- Services degrade gracefully when Redis / RAG / DB are unavailable (return empty/`None`, never raise to the caller) — match the defensive style of `ConversationMemoryService` and `redis_client.get_redis_client`.

---

### Task 1: `proc.bp_requirement` table migration

**Files:**
- Create: `deploy/sql/2026-06-17_bp_requirement.sql`
- Test: `tests/sql/test_bp_requirement_sql.py`

**Interfaces:**
- Produces: table `proc.bp_requirement` with PK `requirement_id` and columns consumed by `RequirementService` (Task 3): `session_id`, `status`, `created_by`, `title`, `category`, `description`, `quantity`, `unit`, `target_budget`, `currency`, `needed_by_date`, `delivery_location`, `priority`, `specifications` (jsonb), `constraints` (jsonb), `completeness_score`, `missing_fields` (jsonb), `seed_context` (jsonb), `created_at`, `updated_at`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sql/test_bp_requirement_sql.py
from pathlib import Path

SQL = Path("deploy/sql/2026-06-17_bp_requirement.sql").read_text()


def test_table_and_key_columns_present():
    lowered = SQL.lower()
    assert "create table if not exists proc.bp_requirement" in lowered
    for col in (
        "requirement_id", "session_id", "status", "created_by",
        "title", "category", "description", "quantity", "unit",
        "target_budget", "currency", "needed_by_date", "delivery_location",
        "priority", "specifications", "constraints",
        "completeness_score", "missing_fields", "seed_context",
        "created_at", "updated_at",
    ):
        assert col in lowered, f"missing column {col}"


def test_status_check_and_indexes_and_idempotent():
    lowered = SQL.lower()
    assert "begin;" in lowered and "commit;" in lowered
    assert "ix_bp_requirement_status" in lowered
    assert "ix_bp_requirement_category" in lowered
    assert "ix_bp_requirement_created_by" in lowered
    for state in ("draft", "gathering", "complete", "handed_off", "abandoned"):
        assert state in lowered, f"missing status {state} in CHECK"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sql/test_bp_requirement_sql.py -v`
Expected: FAIL — `FileNotFoundError` (SQL file does not exist yet).

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-06-17_bp_requirement.sql
-- 2026-06-17 Upstream procurement requirement entity ("Need Identified" stage).
-- Gathered by RequirementsAgent via conversation; precedes any deal_id.
-- Additive + idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_requirement (
    requirement_id      VARCHAR PRIMARY KEY,
    session_id          VARCHAR,
    status              VARCHAR NOT NULL DEFAULT 'gathering',
    created_by          VARCHAR,
    title               TEXT,
    category            VARCHAR,
    description         TEXT,
    quantity            NUMERIC,
    unit                VARCHAR,
    target_budget       NUMERIC,
    currency            VARCHAR,
    needed_by_date      DATE,
    delivery_location   TEXT,
    priority            VARCHAR,
    specifications      JSONB DEFAULT '{}'::jsonb,
    constraints         JSONB DEFAULT '{}'::jsonb,
    completeness_score  NUMERIC DEFAULT 0,
    missing_fields      JSONB DEFAULT '[]'::jsonb,
    seed_context        JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT bp_requirement_status_check CHECK (
        status IN ('draft', 'gathering', 'complete', 'handed_off', 'abandoned'))
);

CREATE INDEX IF NOT EXISTS ix_bp_requirement_status      ON proc.bp_requirement (status);
CREATE INDEX IF NOT EXISTS ix_bp_requirement_category    ON proc.bp_requirement (category);
CREATE INDEX IF NOT EXISTS ix_bp_requirement_created_by  ON proc.bp_requirement (created_by);

COMMIT;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sql/test_bp_requirement_sql.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add deploy/sql/2026-06-17_bp_requirement.sql tests/sql/test_bp_requirement_sql.py
git commit -m "feat(requirements): add proc.bp_requirement table migration"
```

---

### Task 2: `RequirementSession` — Redis-backed multi-turn state

**Files:**
- Create: `src/services/requirement_session.py`
- Test: `tests/services/test_requirement_session.py`

**Interfaces:**
- Consumes: `redis.Redis`-like client (or `None`).
- Produces:
  - `RequirementSession(session_id: str, requirement_id: str, created_by: str = "", status: str = "gathering", requirement: Dict[str, Any] = {}, turn_history: List[Dict] = [], missing_fields: List[str] = [], completeness_score: float = 0.0)`
  - `.add_turn(role: str, content: str) -> None`
  - `.apply_fields(updates: Dict[str, Any]) -> None` (ignores `None`/empty values — no fabrication)
  - `.mark_complete() -> None`, `.mark_abandoned() -> None`
  - `.to_dict() -> Dict`, classmethod `.from_dict(data: Dict) -> RequirementSession`
  - `.save(redis_client) -> None` (no-op if client is `None`)
  - classmethod `.load(session_id: str, redis_client) -> Optional[RequirementSession]` (returns `None` if client is `None` or key absent)
  - `.redis_key` attribute == `f"requirement_session:{session_id}"`

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_requirement_session.py
from src.services.requirement_session import RequirementSession


class _FakeRedis:
    def __init__(self):
        self.store = {}
    def set(self, k, v):
        self.store[k] = v
    def get(self, k):
        return self.store.get(k)


def _new():
    return RequirementSession(session_id="S1", requirement_id="REQ-1", created_by="alice")


def test_redis_key_and_defaults():
    s = _new()
    assert s.redis_key == "requirement_session:S1"
    assert s.status == "gathering"
    assert s.requirement == {}


def test_apply_fields_ignores_empty_values():
    s = _new()
    s.apply_fields({"title": "Laptops", "quantity": 10, "unit": None, "category": ""})
    assert s.requirement == {"title": "Laptops", "quantity": 10}


def test_add_turn_and_status_transitions():
    s = _new()
    s.add_turn("user", "I need laptops")
    assert s.turn_history[-1]["role"] == "user"
    assert s.turn_history[-1]["content"] == "I need laptops"
    s.mark_complete()
    assert s.status == "complete"
    s.mark_abandoned()
    assert s.status == "abandoned"


def test_save_load_round_trip():
    r = _FakeRedis()
    s = _new()
    s.apply_fields({"title": "Laptops"})
    s.add_turn("user", "hi")
    s.save(r)
    loaded = RequirementSession.load("S1", r)
    assert loaded is not None
    assert loaded.requirement == {"title": "Laptops"}
    assert loaded.turn_history[-1]["content"] == "hi"
    assert loaded.requirement_id == "REQ-1"


def test_none_client_degrades_gracefully():
    s = _new()
    s.save(None)  # must not raise
    assert RequirementSession.load("S1", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_requirement_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.requirement_session'`.

- [ ] **Step 3: Write the implementation**

```python
# src/services/requirement_session.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequirementSession:
    """Multi-turn elicitation state for a single procurement requirement.

    Hot state lives in Redis (best-effort); the durable source of truth is the
    ``proc.bp_requirement`` row written by ``RequirementService``.
    """

    session_id: str
    requirement_id: str
    created_by: str = ""
    status: str = "gathering"
    requirement: Dict[str, Any] = field(default_factory=dict)
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    completeness_score: float = 0.0

    def __post_init__(self) -> None:
        self.redis_key = f"requirement_session:{self.session_id}"

    def add_turn(self, role: str, content: str) -> None:
        self.turn_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def apply_fields(self, updates: Dict[str, Any]) -> None:
        """Merge non-empty field values. Empty/None values are ignored so the
        agent never fabricates or blanks an already-known field."""
        if not isinstance(updates, dict):
            return
        for key, value in updates.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            self.requirement[key] = value

    def mark_complete(self) -> None:
        self.status = "complete"

    def mark_abandoned(self) -> None:
        self.status = "abandoned"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "requirement_id": self.requirement_id,
            "created_by": self.created_by,
            "status": self.status,
            "requirement": self.requirement,
            "turn_history": self.turn_history,
            "missing_fields": self.missing_fields,
            "completeness_score": self.completeness_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementSession":
        return cls(
            session_id=data["session_id"],
            requirement_id=data["requirement_id"],
            created_by=data.get("created_by", ""),
            status=data.get("status", "gathering"),
            requirement=data.get("requirement", {}),
            turn_history=data.get("turn_history", []),
            missing_fields=data.get("missing_fields", []),
            completeness_score=data.get("completeness_score", 0.0),
        )

    def save(self, redis_client: Any) -> None:
        if redis_client is None:
            return
        try:
            redis_client.set(self.redis_key, json.dumps(self.to_dict()))
        except Exception:  # pragma: no cover - infra failure
            logger.exception("Failed to save requirement session %s", self.session_id)

    @classmethod
    def load(cls, session_id: str, redis_client: Any) -> Optional["RequirementSession"]:
        if redis_client is None:
            return None
        key = f"requirement_session:{session_id}"
        try:
            data = redis_client.get(key)
        except Exception:  # pragma: no cover - infra failure
            logger.exception("Failed to load requirement session %s", session_id)
            return None
        if not data:
            return None
        try:
            return cls.from_dict(json.loads(data))
        except Exception:  # pragma: no cover - corrupt payload
            logger.exception("Corrupt requirement session payload for %s", session_id)
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_requirement_session.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/requirement_session.py tests/services/test_requirement_session.py
git commit -m "feat(requirements): add Redis-backed RequirementSession state"
```

---

### Task 3: `RequirementService` — persistence, seeding, completeness

**Files:**
- Create: `src/services/requirement_service.py`
- Test: `tests/services/test_requirement_service.py`

**Interfaces:**
- Consumes: `src.services.db.get_conn` (context manager yielding a DB connection with `.cursor()`); the `proc.bp_requirement` table (Task 1).
- Produces (module-level functions, mirroring `deal_summary`'s functional style):
  - `DEFAULT_REQUIRED_FIELDS: tuple = ("title", "category", "quantity", "needed_by_date", "delivery_location")`
  - `mint_requirement_id(created_by: str = "") -> str` → `"REQ-YYYYMMDD-<8hex>"`
  - `evaluate_completeness(requirement: Dict, required_fields: Iterable[str]) -> Tuple[float, List[str]]` → `(score 0..1, missing_fields)`
  - `seed_context(category: str) -> Dict[str, Any]` → history summary (`{}` on any failure)
  - `persist(record: Dict[str, Any]) -> None` → upsert one `proc.bp_requirement` row
  - `get_requirement(requirement_id: str) -> Optional[Dict[str, Any]]`
  - `list_requirements(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_requirement_service.py
import src.services.requirement_service as rs


class _FakeCursor:
    def __init__(self, table_data=None):
        self._table_data = table_data or {}
        self.description = []
        self._rows = []
        self.executed = []
    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []
    def fetchall(self):
        return self._rows
    def fetchone(self):
        return self._rows[0] if self._rows else None
    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data=None):
        self.cur = _FakeCursor(table_data)
        self.committed = False
    def cursor(self):
        return self.cur
    def commit(self):
        self.committed = True
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_mint_requirement_id_format():
    rid = rs.mint_requirement_id("alice")
    assert rid.startswith("REQ-")
    parts = rid.split("-")
    assert len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 8


def test_evaluate_completeness_partial():
    req = {"title": "Laptops", "category": "IT"}
    score, missing = rs.evaluate_completeness(req, rs.DEFAULT_REQUIRED_FIELDS)
    assert missing == ["quantity", "needed_by_date", "delivery_location"]
    assert abs(score - 2 / 5) < 1e-6


def test_evaluate_completeness_full():
    req = {f: "x" for f in rs.DEFAULT_REQUIRED_FIELDS}
    score, missing = rs.evaluate_completeness(req, rs.DEFAULT_REQUIRED_FIELDS)
    assert missing == []
    assert score == 1.0


def test_persist_executes_upsert(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(rs, "get_conn", lambda: conn)
    rs.persist({"requirement_id": "REQ-1", "status": "complete", "title": "Laptops"})
    joined = " ".join(sql for sql, _ in conn.cur.executed).lower()
    assert "insert into proc.bp_requirement" in joined
    assert "on conflict (requirement_id) do update" in joined
    assert conn.committed is True


def test_get_requirement_returns_row(monkeypatch):
    data = {"proc.bp_requirement": (
        ["requirement_id", "status", "title"],
        [("REQ-1", "complete", "Laptops")],
    )}
    monkeypatch.setattr(rs, "get_conn", lambda: _FakeConn(data))
    row = rs.get_requirement("REQ-1")
    assert row == {"requirement_id": "REQ-1", "status": "complete", "title": "Laptops"}


def test_get_requirement_missing_returns_none(monkeypatch):
    monkeypatch.setattr(rs, "get_conn", lambda: _FakeConn({}))
    assert rs.get_requirement("REQ-x") is None


def test_seed_context_degrades_to_empty(monkeypatch):
    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(rs, "get_conn", _boom)
    assert rs.seed_context("IT") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_requirement_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.requirement_service'`.

- [ ] **Step 3: Write the implementation**

```python
# src/services/requirement_service.py
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.services.db import get_conn

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_FIELDS: Tuple[str, ...] = (
    "title", "category", "quantity", "needed_by_date", "delivery_location",
)

# Columns persisted to proc.bp_requirement (excludes DB-defaulted timestamps).
_PERSIST_COLUMNS: Tuple[str, ...] = (
    "requirement_id", "session_id", "status", "created_by", "title",
    "category", "description", "quantity", "unit", "target_budget",
    "currency", "needed_by_date", "delivery_location", "priority",
    "specifications", "constraints", "completeness_score",
    "missing_fields", "seed_context",
)
_JSONB_COLUMNS = {"specifications", "constraints", "missing_fields", "seed_context"}


def mint_requirement_id(created_by: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"REQ-{stamp}-{uuid.uuid4().hex[:8]}"


def evaluate_completeness(
    requirement: Dict[str, Any], required_fields: Iterable[str]
) -> Tuple[float, List[str]]:
    """Return (score, missing_fields). A field counts as filled when present
    and not None/blank. No fabrication: only genuinely-filled fields score."""
    required = list(required_fields) or list(DEFAULT_REQUIRED_FIELDS)
    missing: List[str] = []
    for name in required:
        value = requirement.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    filled = len(required) - len(missing)
    score = filled / len(required) if required else 1.0
    return score, missing


def seed_context(category: str) -> Dict[str, Any]:
    """Best-effort history summary for a category from final (_trgt) tables.
    Returns {} on any failure so a turn never breaks on infra issues."""
    if not category:
        return {}
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "select supplier_name, count(*) as n, avg(total_amount) as avg_amount "
                "from proc.bp_purchase_order_trgt "
                "where lower(coalesce(category, '')) = lower(%s) "
                "group by supplier_name order by n desc limit 5",
                (category,),
            )
            cols = [d[0] for d in (cur.description or [])]
            suppliers = [dict(zip(cols, r)) for r in cur.fetchall()]
        return {"category": category, "recent_suppliers": suppliers}
    except Exception:
        logger.debug("seed_context failed for category=%s", category, exc_info=True)
        return {}


def persist(record: Dict[str, Any]) -> None:
    """Upsert one proc.bp_requirement row keyed by requirement_id."""
    values = []
    for col in _PERSIST_COLUMNS:
        val = record.get(col)
        if col in _JSONB_COLUMNS and val is not None and not isinstance(val, str):
            val = json.dumps(val)
        values.append(val)
    placeholders = ", ".join(["%s"] * len(_PERSIST_COLUMNS))
    update_cols = [c for c in _PERSIST_COLUMNS if c != "requirement_id"]
    set_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
    sql = (
        f"insert into proc.bp_requirement ({', '.join(_PERSIST_COLUMNS)}) "
        f"values ({placeholders}) "
        f"on conflict (requirement_id) do update set {set_clause}, updated_at = now()"
    )
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(values))
        conn.commit()


def get_requirement(requirement_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "select * from proc.bp_requirement where requirement_id = %s",
            (requirement_id,),
        )
        cols = [d[0] for d in (cur.description or [])]
        row = cur.fetchone()
    if not row:
        return None
    return dict(zip(cols, row))


def list_requirements(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "select * from proc.bp_requirement order by created_at desc "
            "limit %s offset %s",
            (limit, offset),
        )
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_requirement_service.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/requirement_service.py tests/services/test_requirement_service.py
git commit -m "feat(requirements): add RequirementService persistence/seeding/completeness"
```

---

### Task 4: `RequirementsAgent` — per-turn elicitation brain

**Files:**
- Create: `src/agents/requirements_agent.py`
- Test: `tests/agents/test_requirements_agent.py`

**Interfaces:**
- Consumes: `BaseAgent` (`__init__(agent_nick)`, `call_ollama`, `resolve_prompt`, `governing_policy`, `emit_signal`); `RequirementSession` (Task 2); `requirement_service` functions (Task 3); `redis_client.get_redis_client` (module-level, monkeypatchable).
- Produces:
  - `class RequirementsAgent(BaseAgent)` with `run(self, context: AgentContext) -> AgentOutput`.
  - Input `context.input_data` keys: `message` (str, optional), `brief` (str, optional), `session_id` (str, optional), `created_by` (str, optional), `category` (str, optional).
  - On incomplete: `AgentOutput.data = {session_id, requirement_id, complete: False, next_question, completeness_score, missing_fields, requirement}`.
  - On complete: `AgentOutput.data = {session_id, requirement_id, complete: True, requirement, completeness_score, summary}`; emits `SUGGEST_AGENT` signals for `supplier_ranking` and `email_drafting`.
  - `_required_fields() -> List[str]` (policy `requirement_required_fields` → `DEFAULT_REQUIRED_FIELDS`).
  - `_advance(session, user_text: str) -> str` (one LLM call: fills fields + returns next question).

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_requirements_agent.py
import src.agents.requirements_agent as ra_mod
from src.agents.requirements_agent import RequirementsAgent
from src.agents.base_agent import AgentContext, AgentStatus


class _FakeRedis:
    def __init__(self):
        self.store = {}
    def set(self, k, v):
        self.store[k] = v
    def get(self, k):
        return self.store.get(k)


def _make_agent(monkeypatch, llm_payloads, redis=None):
    """llm_payloads: list of dicts returned (as JSON) by successive call_ollama calls."""
    agent = RequirementsAgent.__new__(RequirementsAgent)  # bypass heavy __init__
    agent._workflow_context = None
    calls = {"i": 0}

    def fake_call_ollama(prompt=None, model=None, format=None, messages=None, **kw):
        payload = llm_payloads[calls["i"]]
        calls["i"] += 1
        import json as _json
        return {"response": _json.dumps(payload)}

    agent.call_ollama = fake_call_ollama
    agent.resolve_prompt = lambda name, **fmt: None
    agent.governing_policy = lambda name: None
    agent._with_plan = lambda ctx, out: out  # bypass BaseAgent plan internals
    agent.emit_signal = lambda st, msg, data=None: None
    monkeypatch.setattr(ra_mod, "get_redis_client", lambda: redis)
    monkeypatch.setattr(ra_mod.requirement_service, "seed_context", lambda c: {})
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: None)
    return agent


def _ctx(data):
    return AgentContext(workflow_id="W1", agent_id="requirements", user_id="alice", input_data=data)


def test_incomplete_turn_asks_next_question(monkeypatch):
    redis = _FakeRedis()
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Laptops", "category": "IT"},
                       "next_question": "How many laptops do you need?"}],
        redis=redis,
    )
    out = agent.run(_ctx({"message": "I need laptops for the IT team", "created_by": "alice"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False
    assert out.data["next_question"] == "How many laptops do you need?"
    assert "quantity" in out.data["missing_fields"]
    assert out.data["session_id"] in [k.split(":")[1] for k in redis.store]


def test_complete_turn_persists_and_emits_signals(monkeypatch):
    redis = _FakeRedis()
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {
            "title": "Laptops", "category": "IT", "quantity": 10,
            "needed_by_date": "2026-07-01", "delivery_location": "London HQ"},
            "next_question": ""}],
        redis=redis,
    )
    emitted = []
    agent.emit_signal = lambda st, msg, data=None: emitted.append((st, data))
    persisted = []
    monkeypatch.setattr(ra_mod.requirement_service, "persist", lambda rec: persisted.append(rec))

    out = agent.run(_ctx({"message": "10 laptops to London HQ by July 1", "created_by": "alice"}))
    assert out.data["complete"] is True
    assert out.data["completeness_score"] == 1.0
    assert persisted and persisted[0]["status"] == "complete"
    signal_targets = [d.get("agent") for _, d in emitted]
    assert "supplier_ranking" in signal_targets
    assert "email_drafting" in signal_targets


def test_brief_is_parsed_on_first_turn(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        llm_payloads=[{"updates": {"title": "Office chairs", "category": "Furniture"},
                       "next_question": "What quantity?"}],
        redis=_FakeRedis(),
    )
    out = agent.run(_ctx({"brief": "We urgently need office chairs for the new floor."}))
    assert out.data["requirement"]["title"] == "Office chairs"
    assert out.data["complete"] is False


def test_malformed_llm_json_does_not_crash(monkeypatch):
    agent = _make_agent(monkeypatch, llm_payloads=[], redis=_FakeRedis())
    agent.call_ollama = lambda **kw: {"response": "not json at all"}
    out = agent.run(_ctx({"message": "hello"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["complete"] is False  # nothing filled, still gathering
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_requirements_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agents.requirements_agent'`.

- [ ] **Step 3: Write the implementation**

```python
# src/agents/requirements_agent.py
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.requirement_session import RequirementSession
from services.redis_client import get_redis_client
from services import requirement_service

logger = logging.getLogger(__name__)

_DEFAULT_ELICITATION_PROMPT = (
    "You are a procurement requirements assistant. Given the current requirement "
    "(JSON) and the buyer's latest message, extract any NEW field values the "
    "message provides and ask ONE concise question for the single most important "
    "still-missing field. Never invent values not stated by the buyer.\n"
    "Allowed fields: title, category, description, quantity, unit, target_budget, "
    "currency, needed_by_date, delivery_location, priority.\n"
    "Current requirement: {requirement}\n"
    "Still missing: {missing}\n"
    "Buyer message: {message}\n"
    'Respond ONLY with JSON: {{"updates": {{<field>: <value>, ...}}, '
    '"next_question": "<one question, or empty string if nothing missing>"}}'
)


class RequirementsAgent(BaseAgent):
    """Conversation-led elicitation of a single procurement requirement.

    Stateless per call: loads the session, merges the turn's input via one LLM
    call, recomputes completeness, then either asks the next question or
    finalizes and hands off. Acting (ranking/RFQ) is left to other agents.
    """

    AGENTIC_PLAN_STEPS = (
        "Load the requirement session and seed it with category history.",
        "Extract field values from the buyer's message or pasted brief.",
        "Ask the next question, or finalize and hand off when complete.",
    )

    def _required_fields(self) -> List[str]:
        policy = None
        try:
            policy = self.governing_policy("requirement_required_fields")
        except Exception:  # pragma: no cover - defensive
            policy = None
        if policy:
            details = policy.get("policy_details") or policy.get("details")
            fields = None
            if isinstance(details, dict):
                fields = details.get("required_fields")
            elif isinstance(details, str):
                try:
                    fields = json.loads(details).get("required_fields")
                except Exception:
                    fields = None
            if isinstance(fields, list) and fields:
                return [str(f) for f in fields]
        return list(requirement_service.DEFAULT_REQUIRED_FIELDS)

    def _advance(self, session: RequirementSession, user_text: str) -> str:
        missing = session.missing_fields or list(requirement_service.DEFAULT_REQUIRED_FIELDS)
        template = self.resolve_prompt("requirements_elicitation") or _DEFAULT_ELICITATION_PROMPT
        prompt = template.format(
            requirement=json.dumps(session.requirement),
            missing=", ".join(missing),
            message=user_text,
        )
        try:
            result = self.call_ollama(prompt=prompt, format="json")
            raw = result.get("response") if isinstance(result, dict) else result
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            logger.debug("elicitation LLM parse failed", exc_info=True)
            parsed = {}
        if isinstance(parsed, dict):
            session.apply_fields(parsed.get("updates") or {})
            return str(parsed.get("next_question") or "")
        return ""

    def run(self, context: AgentContext) -> AgentOutput:
        try:
            data = context.input_data or {}
            created_by = str(data.get("created_by") or context.user_id or "")
            redis = get_redis_client()

            session_id = str(data.get("session_id") or uuid.uuid4().hex)
            session = RequirementSession.load(session_id, redis)
            if session is None:
                session = RequirementSession(
                    session_id=session_id,
                    requirement_id=requirement_service.mint_requirement_id(created_by),
                    created_by=created_by,
                )
                category = str(data.get("category") or "")
                if category:
                    session.requirement.setdefault("category", category)
                    session.apply_fields({"category": category})
                    session_seed = requirement_service.seed_context(category)
                else:
                    session_seed = {}
                session.requirement.setdefault("_seed_context", session_seed)

            user_text = " ".join(
                str(part) for part in (data.get("brief"), data.get("message")) if part
            ).strip()

            next_question = ""
            if user_text:
                session.add_turn("user", user_text)
                next_question = self._advance(session, user_text)

            required = self._required_fields()
            score, missing = requirement_service.evaluate_completeness(
                session.requirement, required
            )
            session.completeness_score = score
            session.missing_fields = missing

            requirement_out = {
                k: v for k, v in session.requirement.items() if not k.startswith("_")
            }

            if not missing:
                session.mark_complete()
                record = self._build_record(session, requirement_out, score, missing)
                requirement_service.persist(record)
                session.save(redis)
                self._emit_handoff(session, requirement_out)
                summary = self._summary(session, requirement_out)
                return self._with_plan(context, AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={
                        "session_id": session.session_id,
                        "requirement_id": session.requirement_id,
                        "complete": True,
                        "requirement": requirement_out,
                        "completeness_score": score,
                        "summary": summary,
                    },
                    next_agents=[],
                    confidence=score,
                ))

            session.save(redis)
            return self._with_plan(context, AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "session_id": session.session_id,
                    "requirement_id": session.requirement_id,
                    "complete": False,
                    "next_question": next_question,
                    "completeness_score": score,
                    "missing_fields": missing,
                    "requirement": requirement_out,
                },
                next_agents=[],
                confidence=score,
            ))
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("RequirementsAgent.run failed")
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    def _build_record(self, session, requirement_out, score, missing) -> Dict[str, Any]:
        record = {
            "requirement_id": session.requirement_id,
            "session_id": session.session_id,
            "status": "complete",
            "created_by": session.created_by,
            "completeness_score": score,
            "missing_fields": missing,
            "seed_context": session.requirement.get("_seed_context", {}),
        }
        for key in (
            "title", "category", "description", "quantity", "unit",
            "target_budget", "currency", "needed_by_date",
            "delivery_location", "priority",
        ):
            if key in requirement_out:
                record[key] = requirement_out[key]
        specs = requirement_out.get("specifications")
        if isinstance(specs, dict):
            record["specifications"] = specs
        constraints = requirement_out.get("constraints")
        if isinstance(constraints, dict):
            record["constraints"] = constraints
        return record

    def _emit_handoff(self, session, requirement_out) -> None:
        query = requirement_out.get("title") or requirement_out.get("category") or ""
        payload = {
            "requirement_id": session.requirement_id,
            "requirement": requirement_out,
            "query": query,
        }
        self.emit_signal("SUGGEST_AGENT",
                         "Requirement ready for supplier sourcing",
                         {**payload, "agent": "supplier_ranking"})
        self.emit_signal("SUGGEST_AGENT",
                         "Requirement ready for RFQ drafting",
                         {**payload, "agent": "email_drafting"})

    def _summary(self, session, requirement_out) -> str:
        title = requirement_out.get("title", "requirement")
        qty = requirement_out.get("quantity")
        loc = requirement_out.get("delivery_location")
        by = requirement_out.get("needed_by_date")
        bits = [f"Requirement {session.requirement_id} captured: {title}"]
        if qty:
            bits.append(f"qty {qty}")
        if loc:
            bits.append(f"to {loc}")
        if by:
            bits.append(f"by {by}")
        return ", ".join(bits) + "."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_requirements_agent.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/agents/requirements_agent.py tests/agents/test_requirements_agent.py
git commit -m "feat(requirements): add RequirementsAgent elicitation brain"
```

---

### Task 5: Register the agent (definitions + capability)

**Files:**
- Modify: `agent_definitions.json` (append one entry to the JSON array)
- Modify: `src/agents/agent_interface.py:27-43` (add `REQUIREMENTS_GATHERING` to `AgentCapability`) and the `CAPABILITY_ROLES` map (~line 130-145)
- Test: `tests/agents/test_requirements_registration.py`

**Interfaces:**
- Consumes: `RequirementsAgent` class (Task 4); `AgentCapability` enum.
- Produces: a discoverable `requirements` slug in `agent_definitions.json`; `AgentCapability.REQUIREMENTS_GATHERING == "requirements_gathering"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_requirements_registration.py
import json
from pathlib import Path

from src.agents.agent_interface import AgentCapability, CAPABILITY_ROLES, AgentRole


def test_capability_enum_present():
    assert AgentCapability.REQUIREMENTS_GATHERING.value == "requirements_gathering"
    assert CAPABILITY_ROLES[AgentCapability.REQUIREMENTS_GATHERING] == AgentRole.SOURCE


def test_agent_definition_registered():
    defs = json.loads(Path("agent_definitions.json").read_text())
    entry = next((d for d in defs if d.get("slug") == "requirements"), None)
    assert entry is not None
    assert entry["class_path"] == "agents.requirements_agent.RequirementsAgent"
    assert "requirements_gathering" in entry["capabilities"]
    assert "message" in entry["inputs"]["optional"]
    assert "requirement_id" in entry["outputs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_requirements_registration.py -v`
Expected: FAIL — `AttributeError: REQUIREMENTS_GATHERING` (enum member missing).

- [ ] **Step 3a: Add the capability enum member**

In `src/agents/agent_interface.py`, inside `class AgentCapability`, add after `RAG_QUERY = "rag_query"`:

```python
    RAG_QUERY = "rag_query"
    REQUIREMENTS_GATHERING = "requirements_gathering"
```

- [ ] **Step 3b: Map the capability to a role**

In `src/agents/agent_interface.py`, inside the `CAPABILITY_ROLES` dict, add:

```python
    AgentCapability.OPPORTUNITY_MINING: AgentRole.SOURCE,
    AgentCapability.REQUIREMENTS_GATHERING: AgentRole.SOURCE,
```

(Insert the `REQUIREMENTS_GATHERING` line; keep the existing `OPPORTUNITY_MINING` line as-is — shown for placement context.)

- [ ] **Step 3c: Append the agent definition**

Append this object as the last element of the JSON array in `agent_definitions.json` (add a comma after the previous final entry):

```json
  {
    "agentId": 14,
    "agentType": "RequirementsAgent",
    "slug": "requirements",
    "class_path": "agents.requirements_agent.RequirementsAgent",
    "description": "Gathers procurement requirements through conversation, seeded by history and able to parse an unstructured brief; persists a structured requirement and hands off to sourcing agents.",
    "role": "source",
    "capabilities": ["requirements_gathering"],
    "required_inputs": [],
    "output_fields": ["requirement_id", "requirement", "completeness_score"],
    "dependencies": ["db_client", "redis_client", "ollama_client"],
    "inputs": {
      "required": [],
      "optional": ["message", "brief", "session_id", "created_by", "category"]
    },
    "outputs": ["requirement_id", "requirement", "completeness_score"],
    "version": "1.0.0"
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_requirements_registration.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add agent_definitions.json src/agents/agent_interface.py tests/agents/test_requirements_registration.py
git commit -m "feat(requirements): register RequirementsAgent + capability"
```

---

### Task 6: API router (`/requirements`) + SSE progress

**Files:**
- Create: `src/api/routers/requirements.py`
- Modify: `src/api/main.py:48` (import) and `:279` area (include_router)
- Test: `tests/api/test_requirements_router.py`

**Interfaces:**
- Consumes: `requirement_service.get_requirement`, `requirement_service.list_requirements` (Task 3); the orchestrator's agent execution path to run `RequirementsAgent` for a turn.
- Produces endpoints:
  - `POST /requirements/message` — body `{session_id?, message?, brief?, created_by?, category?}` → the agent's `AgentOutput.data` (next question or completed requirement) + SSE-style `events` list.
  - `GET /requirements/{requirement_id}` → the persisted record (404 if absent).
  - `GET /requirements` → `{requirements: [...], count}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_requirements_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.requirements as rq


def _client(monkeypatch, run_result=None, get_result=None, list_result=None):
    monkeypatch.setattr(rq, "_run_requirements_turn",
                        lambda app_state, payload: run_result or {"complete": False, "next_question": "?"})
    monkeypatch.setattr(rq.requirement_service, "get_requirement", lambda rid: get_result)
    monkeypatch.setattr(rq.requirement_service, "list_requirements",
                        lambda limit=50, offset=0: list_result or [])
    app = FastAPI()
    app.include_router(rq.router)
    return TestClient(app)


def test_message_returns_next_question(monkeypatch):
    client = _client(monkeypatch, run_result={
        "complete": False, "next_question": "How many?", "session_id": "S1"})
    resp = client.post("/requirements/message", json={"message": "I need laptops"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["next_question"] == "How many?"
    assert isinstance(body["events"], list) and body["events"]


def test_get_requirement_404(monkeypatch):
    client = _client(monkeypatch, get_result=None)
    resp = client.get("/requirements/REQ-missing")
    assert resp.status_code == 404


def test_get_requirement_found(monkeypatch):
    client = _client(monkeypatch, get_result={"requirement_id": "REQ-1", "status": "complete"})
    resp = client.get("/requirements/REQ-1")
    assert resp.status_code == 200
    assert resp.json()["requirement_id"] == "REQ-1"


def test_list_requirements(monkeypatch):
    client = _client(monkeypatch, list_result=[{"requirement_id": "REQ-1"}])
    resp = client.get("/requirements")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_requirements_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.routers.requirements'`.

- [ ] **Step 3: Write the router**

```python
# src/api/routers/requirements.py
"""Procurement requirements gathering API.

POST /requirements/message      — one elicitation turn (next question or completed requirement)
GET  /requirements/{id}         — fetch a persisted requirement
GET  /requirements              — list requirements
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services import requirement_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/requirements", tags=["Requirements"])


class RequirementMessage(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    brief: Optional[str] = None
    created_by: Optional[str] = None
    category: Optional[str] = None


def _run_requirements_turn(app_state: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one RequirementsAgent turn via the app's orchestrator/agent_nick.

    Isolated for testability — patched in unit tests so the route can be
    exercised without the full agent stack. Mirrors run.py's pattern of
    reaching the orchestrator off ``request.app.state``.
    """
    import uuid
    from agents.agent_factory import AgentFactory
    from agents.base_agent import AgentContext

    orchestrator = getattr(app_state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator unavailable")
    agent = AgentFactory(orchestrator.agent_nick).create("requirements")
    ctx = AgentContext(
        workflow_id=payload.get("session_id") or uuid.uuid4().hex,
        agent_id="requirements",
        user_id=payload.get("created_by") or "api",
        input_data=dict(payload),
    )
    output = agent.run(ctx)
    return dict(output.data or {})


def _events_for(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build SSE-style progress events describing the turn for a live chat UI."""
    events: List[Dict[str, str]] = [{"event": "thinking", "message": "Reviewing requirement"}]
    if result.get("complete"):
        events.append({"event": "complete",
                       "message": result.get("summary", "Requirement captured.")})
    else:
        events.append({"event": "question",
                       "message": result.get("next_question", "")})
    return events


@router.post("/message", summary="Run one requirements elicitation turn")
def post_message(body: RequirementMessage, request: Request) -> Dict[str, Any]:
    try:
        result = _run_requirements_turn(request.app.state, body.model_dump(exclude_none=True))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("requirements turn failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "result": result,
        "events": _events_for(result),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{requirement_id}", summary="Fetch a procurement requirement")
def get_requirement(requirement_id: str) -> Dict[str, Any]:
    try:
        row = requirement_service.get_requirement(requirement_id)
    except Exception as exc:
        logger.exception("requirement fetch failed for %s", requirement_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail=f"No requirement {requirement_id}")
    return row


@router.get("", summary="List procurement requirements")
def list_requirements(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    try:
        items = requirement_service.list_requirements(limit=limit, offset=offset)
    except Exception as exc:
        logger.exception("requirement list failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"requirements": items, "count": len(items)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_requirements_router.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire the router into the app**

In `src/api/main.py` line 48, add `requirements` to the existing router import list:

```python
from api.routers import agents as agents_router_mod, documents, email, metrics, run, stream, system, training, vendors, workflows, deal_summary, promotion, summary, negotiate, opportunities, requirements
```

Then add alongside the other `include_router` calls (after line 279 `app.include_router(summary.router)`):

```python
app.include_router(requirements.router)
```

- [ ] **Step 6: Run the import smoke check**

Run: `python -c "import src.api.main"`
Expected: no ImportError (app imports cleanly with the new router registered).

- [ ] **Step 7: Commit**

```bash
git add src/api/routers/requirements.py src/api/main.py tests/api/test_requirements_router.py
git commit -m "feat(requirements): add /requirements API router with SSE events"
```

---

### Task 7: Governance seed (prompt + policy)

**Files:**
- Create: `deploy/sql/2026-06-17_requirements_governance.sql`
- Test: `tests/sql/test_requirements_governance_sql.py`

**Interfaces:**
- Consumes: `proc.bp_prompt`, `proc.bp_policy` (existing governance tables).
- Produces: an agent-scoped elicitation prompt (`requirements_elicitation`) resolvable by `self.resolve_prompt(...)` (Task 4), and a `requirement_required_fields` policy resolvable by `self.governing_policy(...)`. Both linked to `RequirementsAgent` so the slug `requirements_agent` matches the governance resolver.

- [ ] **Step 1: Write the failing test**

```python
# tests/sql/test_requirements_governance_sql.py
from pathlib import Path

SQL = Path("deploy/sql/2026-06-17_requirements_governance.sql").read_text().lower()


def test_prompt_seed_present():
    assert "insert into proc.bp_prompt" in SQL
    assert "requirements_elicitation" in SQL
    assert "prompt_template" in SQL          # template stored in prompts_desc jsonb
    assert "where not exists" in SQL         # idempotent (no unique constraint)
    assert "requirements_agent" in SQL       # linked agent slug


def test_policy_seed_present():
    assert "insert into proc.bp_policy" in SQL
    assert "requirement_required_fields" in SQL
    assert "policy_details" in SQL
    for f in ("title", "category", "quantity", "needed_by_date", "delivery_location"):
        assert f in SQL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sql/test_requirements_governance_sql.py -v`
Expected: FAIL — `FileNotFoundError`.

- [ ] **Step 3: Write the governance seed SQL**

> Column names verified against `deploy/sql/2026-06-08_create_bp_prompt_bp_policy.sql`:
> `bp_prompt` has **no** `template` column — the template is stored in `prompts_desc` (JSONB) under the key `prompt_template`, which `PromptEngine` parses and exposes as `template` (consumed by `resolve_prompt` in Task 4). Status is `prompts_status SMALLINT` (default 1 = active). Neither `prompt_name` nor `policy_name` has a unique constraint, so seeds use the `INSERT ... SELECT ... WHERE NOT EXISTS` idempotency pattern used by the existing seeds — not `ON CONFLICT`. Linked-agent slug is `requirements_agent` (matches `BaseAgent._governance_slug()` for `RequirementsAgent`, and the existing `supplier_ranking_agent` convention).

```sql
-- deploy/sql/2026-06-17_requirements_governance.sql
-- 2026-06-17 Governance seed for RequirementsAgent: elicitation prompt + required-fields policy.
-- Idempotent (INSERT ... WHERE NOT EXISTS) so re-running is safe.
BEGIN;

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT
    'requirements_elicitation',
    'elicitation',
    'requirements_agent',
    '{"prompt_template": "You are a procurement requirements assistant. Given the current requirement (JSON) and the buyer''s latest message, extract any NEW field values the message provides and ask ONE concise question for the single most important still-missing field. Never invent values not stated by the buyer. Allowed fields: title, category, description, quantity, unit, target_budget, currency, needed_by_date, delivery_location, priority. Current requirement: {requirement}. Still missing: {missing}. Buyer message: {message}. Respond ONLY with JSON: {\"updates\": {}, \"next_question\": \"\"}"}'::jsonb
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'requirements_elicitation'
);

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, policy_linked_agents)
SELECT
    'requirement_required_fields',
    'requirements',
    'Fields that must be filled before a procurement requirement is considered complete.',
    '{"required_fields": ["title", "category", "quantity", "needed_by_date", "delivery_location"]}'::jsonb,
    'requirements_agent'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'requirement_required_fields'
);

COMMIT;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sql/test_requirements_governance_sql.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add deploy/sql/2026-06-17_requirements_governance.sql tests/sql/test_requirements_governance_sql.py
git commit -m "feat(requirements): seed elicitation prompt + required-fields policy"
```

---

## Final verification (run after all tasks)

- [ ] Run the full new-test suite:

Run: `pytest tests/sql/test_bp_requirement_sql.py tests/services/test_requirement_session.py tests/services/test_requirement_service.py tests/agents/test_requirements_agent.py tests/agents/test_requirements_registration.py tests/api/test_requirements_router.py tests/sql/test_requirements_governance_sql.py -v`
Expected: all PASS.

- [ ] Apply the two SQL migrations against `bp_sqldb` using the project's standard migration runner (the same path used to apply `deploy/sql/2026-06-15_bp_opportunity.sql`), then reload governance: `POST /agents/reload-governance`.

- [ ] Manual smoke (server running): `POST /requirements/message {"message": "I need 20 office chairs delivered to London HQ by August"}` → expect a `next_question` for the remaining gap (e.g. category); continue the dialogue with the returned `session_id` until `complete: true`, then `GET /requirements/{requirement_id}` returns the persisted row.

## Deferred (not in this plan)

- Reverse-linking a requirement to the deal it eventually produces (FK once a deal forms).
- A dedicated requirements dashboard view (the `GET /requirements` list covers v1).
- Auto-launching a sourcing workflow on completion (hand-off is signal-only by design).

**Note on hand-off signals:** `emit_signal` only lands on the shared blackboard when the agent runs inside the `WorkflowEngine` (which attaches a `WorkflowContext` via `set_workflow_context`). On the direct `/requirements/message` turn there is no active workflow context, so signals are no-ops; the durable hand-off in that path is the persisted `proc.bp_requirement` row plus the returned `requirement_id`/`requirement`. Wiring `RequirementsAgent` as a node in a declarative workflow (so signals drive `supplier_ranking`/`email_drafting`) is a follow-on, consistent with the existing orchestration pattern.

**Note on `agentId`:** Task 5 uses `agentId: 14` (next after the 13 existing agents). If the array has grown, use `max(agentId) + 1`; the value is cosmetic — discovery is by `slug`/`class_path`.
