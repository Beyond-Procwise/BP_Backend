# Orchestration Core Rearchitecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic 2,971-line Orchestrator with a distributed DAG Scheduler + Redis Streams Task Dispatcher + Agent Workers + Result Collector + PostgreSQL State Manager.

**Architecture:** Declarative workflow DAGs (existing `WorkflowGraph`/`WorkflowNode` data structures) are executed by a new DAG Scheduler that dispatches ready nodes via Redis Streams to independent worker processes. Results flow back through a Result Collector that updates durable PostgreSQL state and triggers downstream node evaluation. The existing `WorkflowEngine` sequential executor is replaced; agent internals are untouched.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL (psycopg2), Redis Streams (redis-py), existing agent framework (BaseAgent, AgentContext, AgentOutput)

**Spec:** `docs/superpowers/specs/2026-03-31-orchestration-rearchitecture-design.md`

**Scope:** This plan covers Sections 1-5, 7, 9-10 of the spec (orchestration core). Extraction pipeline (Section 6) and LLM architecture (Section 8) are separate plans.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/orchestration/dag_scheduler.py` | Accepts workflow requests, computes ready nodes from DAG, dispatches to Task Dispatcher, re-evaluates on node completion |
| `src/orchestration/task_dispatcher.py` | Serializes AgentContext to Redis Streams messages, publishes to per-agent-type streams |
| `src/orchestration/result_collector.py` | Consumes result stream, updates PostgreSQL state, notifies DAG Scheduler |
| `src/orchestration/state_manager.py` | Owns workflow/node state in PostgreSQL, atomic transitions, checkpoint/resume, state queries |
| `src/orchestration/worker.py` | Agent worker process: consumes tasks, executes agents, publishes results, heartbeat, timeout guard |
| `src/orchestration/worker_context.py` | Lightweight replacement for `agent_nick` in worker processes |
| `src/orchestration/message_protocol.py` | Dataclasses for task/result/event messages, serialization/deserialization |
| `src/orchestration/cli.py` | `procwise-worker` CLI entry point |
| `src/orchestration/migrations/001_workflow_execution.sql` | DDL for proc.workflow_execution, proc.node_execution, proc.workflow_events tables + indexes |
| `tests/test_state_manager.py` | State Manager unit tests |
| `tests/test_dag_scheduler.py` | DAG Scheduler unit tests |
| `tests/test_task_dispatcher.py` | Task Dispatcher unit tests |
| `tests/test_result_collector.py` | Result Collector unit tests |
| `tests/test_worker.py` | Worker unit tests |
| `tests/test_message_protocol.py` | Message serialization tests |
| `tests/test_integration_workflow.py` | End-to-end workflow integration tests |

### Modified Files

| File | Change |
|------|--------|
| `src/orchestration/workflow_engine.py` | No changes to WorkflowGraph/WorkflowNode/WorkflowEdge. WorkflowEngine class remains but is no longer the primary executor. |
| `src/orchestration/orchestrator.py` | Add `use_dag_scheduler` flag (extends existing `use_workflow_engine` pattern). When enabled, delegate to DAG Scheduler instead of WorkflowEngine. |
| `src/orchestration/__init__.py` | Export new modules via lazy loading |
| `config/settings.py` | Add `use_dag_scheduler`, `redis_streams_url`, `worker_heartbeat_interval`, `worker_task_timeout` settings |
| `src/api/routers/workflows.py` | Add observability endpoints (GET status/events/trace) |

### Unchanged Files

| File | Why |
|------|-----|
| `src/agents/base_agent.py` | Agent internals untouched. `execute()` signature unchanged. |
| `src/agents/agent_factory.py` | Reused inside workers as-is |
| `src/agents/registry.py` | Reused inside workers as-is |
| `src/orchestration/workflow_definitions.py` | DAG definitions reused as-is |
| All 13 agent files in `src/agents/` | Business logic untouched |

---

## Task 1: Message Protocol Dataclasses

**Files:**
- Create: `src/orchestration/message_protocol.py`
- Create: `tests/test_message_protocol.py`

- [ ] **Step 1: Write the failing test for TaskMessage serialization**

```python
# tests/test_message_protocol.py
import json
import pytest
from datetime import datetime, timezone


def test_task_message_round_trip():
    from orchestration.message_protocol import TaskMessage

    msg = TaskMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        context={
            "workflow_id": "wf-123",
            "agent_id": "supplier_ranking",
            "user_id": "user-1",
            "input_data": {"query": "find suppliers"},
            "policy_context": [],
            "knowledge_base": {},
            "routing_history": ["data_extraction"],
            "task_profile": {},
        },
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc),
        timeout_seconds=300,
        attempt=1,
    )
    serialized = msg.to_dict()
    restored = TaskMessage.from_dict(serialized)
    assert restored.task_id == "t-001"
    assert restored.agent_type == "supplier_ranking"
    assert restored.context["input_data"]["query"] == "find suppliers"
    assert restored.timeout_seconds == 300
    # Must be JSON-serializable for Redis
    json.dumps(serialized)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_message_protocol.py::test_task_message_round_trip -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the failing test for ResultMessage serialization**

```python
# tests/test_message_protocol.py (append)


def test_result_message_round_trip():
    from orchestration.message_protocol import ResultMessage

    msg = ResultMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        status="SUCCESS",
        data={"ranking": [{"supplier": "Acme", "score": 0.95}]},
        pass_fields={"ranking": [{"supplier": "Acme", "score": 0.95}]},
        next_agents=["quote_evaluation"],
        error=None,
        confidence=0.92,
        completed_at=datetime(2026, 3, 31, 12, 0, 5, tzinfo=timezone.utc),
        duration_ms=4230,
    )
    serialized = msg.to_dict()
    restored = ResultMessage.from_dict(serialized)
    assert restored.status == "SUCCESS"
    assert restored.duration_ms == 4230
    assert restored.data["ranking"][0]["supplier"] == "Acme"
    json.dumps(serialized)


def test_event_message_round_trip():
    from orchestration.message_protocol import EventMessage

    msg = EventMessage(
        event="node:completed",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        timestamp=datetime(2026, 3, 31, 12, 0, 5, tzinfo=timezone.utc),
        metadata={"duration_ms": 4230, "agent_type": "supplier_ranking"},
    )
    serialized = msg.to_dict()
    restored = EventMessage.from_dict(serialized)
    assert restored.event == "node:completed"
    assert restored.metadata["duration_ms"] == 4230
    json.dumps(serialized)
```

- [ ] **Step 4: Implement message_protocol.py**

```python
# src/orchestration/message_protocol.py
"""Dataclasses for Redis Streams message protocol.

Three message types:
- TaskMessage: Dispatcher -> Workers (via agent:tasks:{agent_type} streams)
- ResultMessage: Workers -> Result Collector (via agent:results stream)
- EventMessage: All components -> Observability (via workflow:events stream)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_iso(val: Optional[str]) -> Optional[datetime]:
    if val is None:
        return None
    return datetime.fromisoformat(val)


@dataclass(frozen=True)
class TaskMessage:
    task_id: str
    workflow_id: str
    node_name: str
    agent_type: str
    context: Dict[str, Any]
    priority: str  # "critical" | "normal"
    dispatched_at: datetime
    timeout_seconds: int
    attempt: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dispatched_at"] = _iso(self.dispatched_at)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TaskMessage:
        d = dict(d)
        d["dispatched_at"] = _parse_iso(d["dispatched_at"])
        return cls(**d)

    def to_redis(self) -> Dict[str, str]:
        return {"payload": json.dumps(self.to_dict())}

    @classmethod
    def from_redis(cls, fields: Dict[bytes, bytes]) -> TaskMessage:
        raw = fields[b"payload"] if isinstance(list(fields.keys())[0], bytes) else fields["payload"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ResultMessage:
    task_id: str
    workflow_id: str
    node_name: str
    agent_type: str
    status: str  # "SUCCESS" | "FAILED"
    data: Dict[str, Any]
    pass_fields: Dict[str, Any]
    next_agents: List[str]
    error: Optional[str]
    confidence: Optional[float]
    completed_at: datetime
    duration_ms: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["completed_at"] = _iso(self.completed_at)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResultMessage:
        d = dict(d)
        d["completed_at"] = _parse_iso(d["completed_at"])
        return cls(**d)

    def to_redis(self) -> Dict[str, str]:
        return {"payload": json.dumps(self.to_dict())}

    @classmethod
    def from_redis(cls, fields: Dict[bytes, bytes]) -> ResultMessage:
        raw = fields[b"payload"] if isinstance(list(fields.keys())[0], bytes) else fields["payload"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class EventMessage:
    event: str
    workflow_id: str
    node_name: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = _iso(self.timestamp)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EventMessage:
        d = dict(d)
        d["timestamp"] = _parse_iso(d["timestamp"])
        return cls(**d)

    def to_redis(self) -> Dict[str, str]:
        return {"payload": json.dumps(self.to_dict())}

    @classmethod
    def from_redis(cls, fields: Dict[bytes, bytes]) -> EventMessage:
        raw = fields[b"payload"] if isinstance(list(fields.keys())[0], bytes) else fields["payload"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))
```

- [ ] **Step 5: Run all message protocol tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_message_protocol.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/orchestration/message_protocol.py tests/test_message_protocol.py
git commit -m "feat: add Redis Streams message protocol dataclasses"
```

---

## Task 2: PostgreSQL Schema Migration

**Files:**
- Create: `src/orchestration/migrations/001_workflow_execution.sql`

- [ ] **Step 1: Create the migration SQL**

```sql
-- src/orchestration/migrations/001_workflow_execution.sql
-- Orchestration rearchitecture: durable workflow state tables
-- Spec: docs/superpowers/specs/2026-03-31-orchestration-rearchitecture-design.md Section 5

BEGIN;

-- Workflow execution state (replaces Redis-based checkpoints)
CREATE TABLE IF NOT EXISTS proc.workflow_execution (
    execution_id    SERIAL PRIMARY KEY,
    workflow_id     TEXT NOT NULL,
    workflow_name   TEXT NOT NULL,
    user_id         TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','running','paused','completed','failed','cancelled')),
    shared_data     JSONB NOT NULL DEFAULT '{}',
    current_round   INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

-- Unique active workflow constraint: only one non-terminal execution per workflow_id
CREATE UNIQUE INDEX IF NOT EXISTS uix_workflow_execution_active
    ON proc.workflow_execution (workflow_id)
    WHERE status NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS ix_workflow_execution_status
    ON proc.workflow_execution (status);

-- Node execution state (one row per node per round)
CREATE TABLE IF NOT EXISTS proc.node_execution (
    node_execution_id   SERIAL PRIMARY KEY,
    execution_id        INTEGER NOT NULL REFERENCES proc.workflow_execution(execution_id),
    node_name           TEXT NOT NULL,
    agent_type          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','ready','running','completed','failed','skipped','timed_out')),
    attempt             INTEGER NOT NULL DEFAULT 0,
    input_data          JSONB,
    output_data         JSONB,
    pass_fields         JSONB,
    error               TEXT,
    dispatched_at       TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_ms         INTEGER,
    round               INTEGER NOT NULL DEFAULT 0,
    UNIQUE (execution_id, node_name, round)
);

CREATE INDEX IF NOT EXISTS ix_node_execution_status
    ON proc.node_execution (execution_id, status);

CREATE INDEX IF NOT EXISTS ix_node_execution_stale
    ON proc.node_execution (status, dispatched_at)
    WHERE status IN ('running', 'ready');

-- Workflow events (observability + audit trail)
CREATE TABLE IF NOT EXISTS proc.workflow_events (
    event_id        SERIAL PRIMARY KEY,
    workflow_id     TEXT NOT NULL,
    node_name       TEXT,
    event_type      TEXT NOT NULL,
    agent_type      TEXT,
    payload         JSONB NOT NULL DEFAULT '{}',
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    round           INTEGER
);

CREATE INDEX IF NOT EXISTS ix_workflow_events_workflow
    ON proc.workflow_events (workflow_id, timestamp);

CREATE INDEX IF NOT EXISTS ix_workflow_events_type
    ON proc.workflow_events (event_type, timestamp);

COMMIT;
```

- [ ] **Step 2: Verify the SQL is syntactically valid**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -c "
with open('src/orchestration/migrations/001_workflow_execution.sql') as f:
    sql = f.read()
print(f'Migration file: {len(sql)} bytes, {sql.count(\"CREATE TABLE\")} tables, {sql.count(\"CREATE INDEX\")} indexes + {sql.count(\"CREATE UNIQUE INDEX\")} unique indexes')
"`
Expected: `Migration file: ... bytes, 3 tables, 4 indexes + 1 unique indexes`

- [ ] **Step 3: Commit**

```bash
git add src/orchestration/migrations/001_workflow_execution.sql
git commit -m "feat: add DDL migration for workflow execution state tables"
```

---

## Task 3: State Manager

**Files:**
- Create: `src/orchestration/state_manager.py`
- Create: `tests/test_state_manager.py`

- [ ] **Step 1: Write failing tests for workflow lifecycle**

```python
# tests/test_state_manager.py
"""State Manager tests using a mock database connection.

The State Manager owns all reads/writes to proc.workflow_execution,
proc.node_execution, and proc.workflow_events.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone


class FakeCursor:
    """Simulates psycopg2 cursor with fetchone/fetchall."""

    def __init__(self, rows=None):
        self._rows = rows or []
        self.description = None
        self.rowcount = len(self._rows)

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_state_manager(cursor_rows=None):
    from orchestration.state_manager import StateManager

    cursor = FakeCursor(cursor_rows)
    conn = FakeConnection(cursor)
    get_conn = MagicMock(return_value=conn)
    sm = StateManager(get_connection=get_conn)
    return sm, cursor, get_conn


def test_create_workflow_execution():
    sm, cursor, _ = _make_state_manager([(42,)])
    cursor.execute = MagicMock()
    cursor.fetchone = MagicMock(return_value=(42,))

    exec_id = sm.create_workflow_execution(
        workflow_id="wf-001",
        workflow_name="supplier_ranking",
        user_id="user-1",
    )
    assert exec_id == 42
    # Verify INSERT was called
    assert cursor.execute.call_count >= 1
    sql = cursor.execute.call_args_list[0][0][0]
    assert "proc.workflow_execution" in sql
    assert "INSERT" in sql.upper()


def test_create_node_executions():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    nodes = [
        {"node_name": "extract", "agent_type": "data_extraction"},
        {"node_name": "rank", "agent_type": "supplier_ranking"},
    ]
    sm.create_node_executions(execution_id=42, nodes=nodes, round_num=0)
    assert cursor.execute.call_count >= 2


def test_get_ready_nodes():
    sm, cursor, _ = _make_state_manager()
    cursor.fetchall = MagicMock(return_value=[
        (1, "extract", "data_extraction", 0),
    ])
    cursor.execute = MagicMock()

    ready = sm.get_ready_nodes(execution_id=42, current_round=0)
    assert len(ready) == 1
    assert ready[0]["node_name"] == "extract"


def test_update_node_status():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.update_node_status(
        execution_id=42,
        node_name="extract",
        round_num=0,
        status="running",
        dispatched_at=datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert cursor.execute.call_count >= 1
    sql = cursor.execute.call_args[0][0]
    assert "UPDATE" in sql.upper()
    assert "proc.node_execution" in sql


def test_record_node_result():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.record_node_result(
        execution_id=42,
        node_name="extract",
        round_num=0,
        status="completed",
        output_data={"details": "extracted"},
        pass_fields={"details": "extracted"},
        duration_ms=1500,
    )
    assert cursor.execute.call_count >= 1


def test_merge_shared_data():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.merge_shared_data(execution_id=42, new_fields={"ranking": [1, 2, 3]})
    sql = cursor.execute.call_args[0][0]
    assert "shared_data" in sql
    assert "proc.workflow_execution" in sql


def test_record_event():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.record_event(
        workflow_id="wf-001",
        event_type="node:completed",
        node_name="extract",
        agent_type="data_extraction",
        payload={"duration_ms": 1500},
        round_num=0,
    )
    sql = cursor.execute.call_args[0][0]
    assert "proc.workflow_events" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_state_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement StateManager**

```python
# src/orchestration/state_manager.py
"""Durable workflow state management backed by PostgreSQL.

Owns all reads/writes to:
- proc.workflow_execution (workflow lifecycle)
- proc.node_execution (per-node state)
- proc.workflow_events (audit trail)

Spec reference: Section 5 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class StateManager:
    def __init__(self, get_connection: Callable):
        self._get_connection = get_connection

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # ── Workflow Execution ──────────────────────────────────────

    def create_workflow_execution(
        self,
        workflow_id: str,
        workflow_name: str,
        user_id: Optional[str] = None,
    ) -> int:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO proc.workflow_execution
                   (workflow_id, workflow_name, user_id, status, created_at, updated_at)
                   VALUES (%s, %s, %s, 'pending', now(), now())
                   RETURNING execution_id""",
                (workflow_id, workflow_name, user_id),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0]

    def update_workflow_status(
        self, execution_id: int, status: str, completed_at: Optional[datetime] = None
    ) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE proc.workflow_execution
                   SET status = %s, updated_at = now(), completed_at = %s
                   WHERE execution_id = %s""",
                (status, completed_at, execution_id),
            )
            conn.commit()

    def get_workflow_execution(self, execution_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT execution_id, workflow_id, workflow_name, user_id,
                          status, shared_data, current_round, created_at,
                          updated_at, completed_at
                   FROM proc.workflow_execution WHERE execution_id = %s""",
                (execution_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "execution_id": row[0],
                "workflow_id": row[1],
                "workflow_name": row[2],
                "user_id": row[3],
                "status": row[4],
                "shared_data": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                "current_round": row[6],
                "created_at": row[7],
                "updated_at": row[8],
                "completed_at": row[9],
            }

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT execution_id, workflow_id, workflow_name, status
                   FROM proc.workflow_execution
                   WHERE status IN ('pending', 'running', 'paused')"""
            )
            return [
                {"execution_id": r[0], "workflow_id": r[1], "workflow_name": r[2], "status": r[3]}
                for r in cur.fetchall()
            ]

    def increment_round(self, execution_id: int) -> int:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE proc.workflow_execution
                   SET current_round = current_round + 1, updated_at = now()
                   WHERE execution_id = %s
                   RETURNING current_round""",
                (execution_id,),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0]

    def merge_shared_data(self, execution_id: int, new_fields: Dict[str, Any]) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE proc.workflow_execution
                   SET shared_data = shared_data || %s::jsonb, updated_at = now()
                   WHERE execution_id = %s""",
                (json.dumps(new_fields), execution_id),
            )
            conn.commit()

    # ── Node Execution ──────────────────────────────────────────

    def create_node_executions(
        self, execution_id: int, nodes: List[Dict[str, str]], round_num: int = 0
    ) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            for node in nodes:
                cur.execute(
                    """INSERT INTO proc.node_execution
                       (execution_id, node_name, agent_type, status, round)
                       VALUES (%s, %s, %s, 'pending', %s)
                       ON CONFLICT (execution_id, node_name, round) DO NOTHING""",
                    (execution_id, node["node_name"], node["agent_type"], round_num),
                )
            conn.commit()

    def update_node_status(
        self,
        execution_id: int,
        node_name: str,
        round_num: int,
        status: str,
        dispatched_at: Optional[datetime] = None,
        attempt: Optional[int] = None,
    ) -> None:
        conn = self._get_connection()
        parts = ["status = %s"]
        params: list = [status]
        if dispatched_at:
            parts.append("dispatched_at = %s")
            params.append(dispatched_at)
        if attempt is not None:
            parts.append("attempt = %s")
            params.append(attempt)
        params.extend([execution_id, node_name, round_num])
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE proc.node_execution
                    SET {', '.join(parts)}
                    WHERE execution_id = %s AND node_name = %s AND round = %s""",
                params,
            )
            conn.commit()

    def record_node_result(
        self,
        execution_id: int,
        node_name: str,
        round_num: int,
        status: str,
        output_data: Optional[Dict] = None,
        pass_fields: Optional[Dict] = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE proc.node_execution
                   SET status = %s, output_data = %s, pass_fields = %s,
                       error = %s, duration_ms = %s, completed_at = now()
                   WHERE execution_id = %s AND node_name = %s AND round = %s""",
                (
                    status,
                    json.dumps(output_data) if output_data else None,
                    json.dumps(pass_fields) if pass_fields else None,
                    error,
                    duration_ms,
                    execution_id,
                    node_name,
                    round_num,
                ),
            )
            conn.commit()

    def get_ready_nodes(
        self, execution_id: int, current_round: int
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT node_execution_id, node_name, agent_type, attempt
                   FROM proc.node_execution
                   WHERE execution_id = %s AND round = %s AND status = 'ready'""",
                (execution_id, current_round),
            )
            return [
                {
                    "node_execution_id": r[0],
                    "node_name": r[1],
                    "agent_type": r[2],
                    "attempt": r[3],
                }
                for r in cur.fetchall()
            ]

    def get_node_statuses(
        self, execution_id: int, current_round: int
    ) -> Dict[str, str]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT node_name, status
                   FROM proc.node_execution
                   WHERE execution_id = %s AND round = %s""",
                (execution_id, current_round),
            )
            return {r[0]: r[1] for r in cur.fetchall()}

    def get_node_result(
        self, execution_id: int, node_name: str, round_num: int
    ) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT output_data, pass_fields, status
                   FROM proc.node_execution
                   WHERE execution_id = %s AND node_name = %s AND round = %s""",
                (execution_id, node_name, round_num),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "output_data": row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}"),
                "pass_fields": row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}"),
                "status": row[2],
            }

    def check_task_completed(self, task_id: str) -> bool:
        """Idempotency check: has this task_id already been completed?"""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT 1 FROM proc.node_execution
                   WHERE input_data->>'task_id' = %s AND status = 'completed'
                   LIMIT 1""",
                (task_id,),
            )
            return cur.fetchone() is not None

    # ── Workflow Events ─────────────────────────────────────────

    def record_event(
        self,
        workflow_id: str,
        event_type: str,
        node_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        payload: Optional[Dict] = None,
        round_num: Optional[int] = None,
    ) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO proc.workflow_events
                   (workflow_id, event_type, node_name, agent_type, payload, round)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    workflow_id,
                    event_type,
                    node_name,
                    agent_type,
                    json.dumps(payload or {}),
                    round_num,
                ),
            )
            conn.commit()

    def get_events(
        self, workflow_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT event_id, event_type, node_name, agent_type,
                          payload, timestamp, round
                   FROM proc.workflow_events
                   WHERE workflow_id = %s
                   ORDER BY timestamp
                   LIMIT %s""",
                (workflow_id, limit),
            )
            return [
                {
                    "event_id": r[0],
                    "event_type": r[1],
                    "node_name": r[2],
                    "agent_type": r[3],
                    "payload": r[4],
                    "timestamp": r[5],
                    "round": r[6],
                }
                for r in cur.fetchall()
            ]

    # ── Crash Recovery ──────────────────────────────────────────

    def find_stale_running_nodes(self, stale_threshold_seconds: int = 600) -> List[Dict]:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT ne.node_execution_id, ne.execution_id, ne.node_name,
                          ne.agent_type, ne.attempt, we.workflow_id
                   FROM proc.node_execution ne
                   JOIN proc.workflow_execution we ON ne.execution_id = we.execution_id
                   WHERE ne.status = 'running'
                     AND ne.dispatched_at < now() - make_interval(secs => %s)""",
                (stale_threshold_seconds,),
            )
            return [
                {
                    "node_execution_id": r[0],
                    "execution_id": r[1],
                    "node_name": r[2],
                    "agent_type": r[3],
                    "attempt": r[4],
                    "workflow_id": r[5],
                }
                for r in cur.fetchall()
            ]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_state_manager.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/state_manager.py tests/test_state_manager.py
git commit -m "feat: add PostgreSQL-backed StateManager for workflow state"
```

---

## Task 4: Task Dispatcher

**Files:**
- Create: `src/orchestration/task_dispatcher.py`
- Create: `tests/test_task_dispatcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_task_dispatcher.py
import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timezone


def test_dispatch_task_publishes_to_correct_stream():
    from orchestration.task_dispatcher import TaskDispatcher
    from orchestration.message_protocol import TaskMessage

    redis_client = MagicMock()
    redis_client.xlen = MagicMock(return_value=0)
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=10000)

    msg = TaskMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        context={"workflow_id": "wf-123", "agent_id": "supplier_ranking",
                 "user_id": "u1", "input_data": {}},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300,
        attempt=1,
    )
    dispatcher.dispatch(msg)
    redis_client.xadd.assert_called_once()
    stream_name = redis_client.xadd.call_args[0][0]
    assert stream_name == "agent:tasks:supplier_ranking"


def test_dispatch_publishes_event():
    from orchestration.task_dispatcher import TaskDispatcher
    from orchestration.message_protocol import TaskMessage

    redis_client = MagicMock()
    redis_client.xlen = MagicMock(return_value=0)
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=10000)

    msg = TaskMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        context={},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300,
        attempt=1,
    )
    dispatcher.dispatch(msg)
    # Should also publish to workflow:events stream
    event_call = [c for c in redis_client.xadd.call_args_list if c[0][0] == "workflow:events"]
    assert len(event_call) == 1


def test_dispatch_respects_maxlen():
    from orchestration.task_dispatcher import TaskDispatcher
    from orchestration.message_protocol import TaskMessage

    redis_client = MagicMock()
    redis_client.xlen = MagicMock(return_value=0)
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=5000)

    msg = TaskMessage(
        task_id="t-001", workflow_id="wf-123", node_name="n", agent_type="a",
        context={}, priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300, attempt=1,
    )
    dispatcher.dispatch(msg)
    xadd_kwargs = redis_client.xadd.call_args_list[0]
    # maxlen should be passed
    assert "maxlen" in xadd_kwargs.kwargs or len(xadd_kwargs.args) > 2


def test_publish_cancellation():
    from orchestration.task_dispatcher import TaskDispatcher

    redis_client = MagicMock()
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=10000)

    dispatcher.publish_cancellation(workflow_id="wf-123", agent_types=["supplier_ranking", "negotiation"])
    assert redis_client.xadd.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_task_dispatcher.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TaskDispatcher**

```python
# src/orchestration/task_dispatcher.py
"""Publishes agent tasks to Redis Streams for worker consumption.

Each agent type gets its own stream: agent:tasks:{agent_type}
Also publishes lifecycle events to workflow:events stream.

Spec reference: Section 2 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from orchestration.message_protocol import EventMessage, TaskMessage

logger = logging.getLogger(__name__)


class TaskDispatcher:
    TASK_STREAM_PREFIX = "agent:tasks:"
    EVENT_STREAM = "workflow:events"

    def __init__(self, redis_client: Any, max_stream_len: int = 10000):
        self._redis = redis_client
        self._max_stream_len = max_stream_len

    def dispatch(self, task: TaskMessage) -> str:
        stream = f"{self.TASK_STREAM_PREFIX}{task.agent_type}"
        msg_id = self._redis.xadd(
            stream,
            task.to_redis(),
            maxlen=self._max_stream_len,
        )
        event = EventMessage(
            event="node:dispatched",
            workflow_id=task.workflow_id,
            node_name=task.node_name,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "agent_type": task.agent_type,
                "task_id": task.task_id,
                "attempt": task.attempt,
            },
        )
        self._redis.xadd(
            self.EVENT_STREAM,
            event.to_redis(),
            maxlen=self._max_stream_len * 10,
        )
        logger.info(
            "Dispatched task %s to %s (workflow=%s, attempt=%d)",
            task.task_id, stream, task.workflow_id, task.attempt,
        )
        return msg_id

    def publish_cancellation(
        self, workflow_id: str, agent_types: List[str]
    ) -> None:
        for agent_type in agent_types:
            stream = f"{self.TASK_STREAM_PREFIX}{agent_type}"
            cancel_msg = {
                "payload": json.dumps({
                    "cancel": True,
                    "workflow_id": workflow_id,
                })
            }
            self._redis.xadd(stream, cancel_msg, maxlen=self._max_stream_len)
            logger.info("Published cancellation for %s to %s", workflow_id, stream)

    def publish_event(self, event: EventMessage) -> None:
        self._redis.xadd(
            self.EVENT_STREAM,
            event.to_redis(),
            maxlen=self._max_stream_len * 10,
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_task_dispatcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/task_dispatcher.py tests/test_task_dispatcher.py
git commit -m "feat: add TaskDispatcher for Redis Streams publishing"
```

---

## Task 5: DAG Scheduler

**Files:**
- Create: `src/orchestration/dag_scheduler.py`
- Create: `tests/test_dag_scheduler.py`

- [ ] **Step 1: Write failing tests for ready-node computation**

```python
# tests/test_dag_scheduler.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


def _build_simple_graph():
    """Build: extract -> rank -> draft (linear DAG)."""
    from orchestration.workflow_engine import WorkflowGraph, WorkflowNode

    g = WorkflowGraph(name="test_linear")
    g.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    g.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
    g.add_node(WorkflowNode(name="draft", agent_type="email_drafting"))
    g.add_edge("extract", "rank")
    g.add_edge("rank", "draft")
    return g


def _build_parallel_graph():
    """Build: extract -> [eval, compare] -> draft (diamond DAG)."""
    from orchestration.workflow_engine import WorkflowGraph, WorkflowNode

    g = WorkflowGraph(name="test_parallel")
    g.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    g.add_node(WorkflowNode(name="eval", agent_type="quote_evaluation"))
    g.add_node(WorkflowNode(name="compare", agent_type="quote_comparison"))
    g.add_node(WorkflowNode(name="draft", agent_type="email_drafting"))
    g.add_edge("extract", "eval")
    g.add_edge("extract", "compare")
    g.add_edge("eval", "draft")
    g.add_edge("compare", "draft")
    return g


def test_initial_ready_nodes_linear():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_simple_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "pending", "rank": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["extract"]


def test_initial_ready_nodes_parallel():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "pending", "eval": "pending", "compare": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["extract"]


def test_parallel_dispatch_after_predecessor():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "pending", "compare": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert set(ready) == {"eval", "compare"}


def test_join_node_waits_for_all_predecessors():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "completed", "compare": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    # draft should NOT be ready because compare is still pending
    assert "draft" not in ready
    assert ready == ["compare"]


def test_join_node_ready_when_all_predecessors_done():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "completed", "compare": "completed", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["draft"]


def test_skipped_predecessor_unblocks_successor():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "completed", "compare": "skipped", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["draft"]


def test_workflow_complete_detection():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_simple_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "rank": "completed", "draft": "completed"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == []
    assert scheduler.is_workflow_complete(graph, node_statuses)


def test_conditional_edge_skips_node():
    from orchestration.dag_scheduler import DAGScheduler
    from orchestration.workflow_engine import WorkflowGraph, WorkflowNode

    g = WorkflowGraph(name="test_conditional")
    g.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    g.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
    g.add_edge("extract", "rank", condition=lambda state: False)  # Always skip

    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "rank": "pending"}
    shared_data = {}
    ready, skipped = scheduler.compute_ready_nodes_with_conditions(
        graph=g,
        node_statuses=node_statuses,
        shared_data=shared_data,
    )
    assert ready == []
    assert skipped == ["rank"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_dag_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DAGScheduler**

```python
# src/orchestration/dag_scheduler.py
"""DAG-based workflow scheduler with parallel node dispatch.

Replaces the sequential WorkflowEngine executor. Uses WorkflowGraph
data structures from workflow_engine.py but executes nodes in parallel
via TaskDispatcher + Redis Streams.

Spec reference: Sections 1, 3 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orchestration.message_protocol import EventMessage, TaskMessage
from orchestration.state_manager import StateManager
from orchestration.task_dispatcher import TaskDispatcher
from orchestration.workflow_engine import NodeStatus, WorkflowEdge, WorkflowGraph, WorkflowState

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "skipped", "failed", "timed_out"})
_DONE_STATUSES = frozenset({"completed", "skipped"})


class DAGScheduler:
    def __init__(
        self,
        state_manager: StateManager,
        task_dispatcher: TaskDispatcher,
        agent_registry: Any,
        manifest_service: Any = None,
        default_timeout: int = 300,
    ):
        self._state = state_manager
        self._dispatcher = task_dispatcher
        self._agents = agent_registry
        self._manifest = manifest_service
        self._default_timeout = default_timeout

    # ── Public API ──────────────────────────────────────────────

    def start_workflow(
        self,
        graph: WorkflowGraph,
        input_data: Dict[str, Any],
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        workflow_id = workflow_id or f"wf-{uuid.uuid4().hex[:12]}"
        execution_id = self._state.create_workflow_execution(
            workflow_id=workflow_id,
            workflow_name=graph.name,
            user_id=user_id,
        )
        self._state.merge_shared_data(execution_id, input_data)
        nodes = [
            {"node_name": name, "agent_type": node.agent_type}
            for name, node in graph.nodes.items()
        ]
        self._state.create_node_executions(execution_id, nodes, round_num=0)
        self._state.update_workflow_status(execution_id, "running")
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:started",
            payload={"workflow_name": graph.name, "node_count": len(nodes)},
        )
        self._evaluate_and_dispatch(graph, execution_id, workflow_id, round_num=0)
        return execution_id

    def on_node_completed(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        node_name: str,
        round_num: int,
    ) -> None:
        node_statuses = self._state.get_node_statuses(execution_id, round_num)
        wf = self._state.get_workflow_execution(execution_id)
        shared_data = wf["shared_data"] if wf else {}

        ready, skipped = self.compute_ready_nodes_with_conditions(
            graph, node_statuses, shared_data,
        )
        for skip_name in skipped:
            self._state.update_node_status(execution_id, skip_name, round_num, "skipped")
            self._state.record_event(
                workflow_id=workflow_id,
                event_type="node:skipped",
                node_name=skip_name,
                round_num=round_num,
            )

        if ready:
            self._dispatch_nodes(graph, execution_id, workflow_id, ready, round_num, shared_data)
        elif self.is_workflow_complete(graph, node_statuses):
            self._complete_workflow(execution_id, workflow_id)
        elif self._has_failed_required(graph, node_statuses):
            self._fail_workflow(execution_id, workflow_id, graph, node_statuses)

    def on_node_failed(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        node_name: str,
        round_num: int,
    ) -> None:
        node = graph.nodes.get(node_name)
        if node and node.required:
            self._fail_workflow(execution_id, workflow_id, graph,
                                self._state.get_node_statuses(execution_id, round_num))
        else:
            self._state.update_node_status(execution_id, node_name, round_num, "skipped")
            self.on_node_completed(graph, execution_id, workflow_id, node_name, round_num)

    # ── Negotiation Re-entry ────────────────────────────────────

    def start_next_round(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        subgraph_nodes: List[str],
    ) -> int:
        new_round = self._state.increment_round(execution_id)
        nodes = [
            {"node_name": name, "agent_type": graph.nodes[name].agent_type}
            for name in subgraph_nodes
            if name in graph.nodes
        ]
        self._state.create_node_executions(execution_id, nodes, round_num=new_round)
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:resumed",
            payload={"round": new_round, "nodes": subgraph_nodes},
            round_num=new_round,
        )
        self._evaluate_and_dispatch(graph, execution_id, workflow_id, round_num=new_round)
        return new_round

    # ── Ready-Node Computation ──────────────────────────────────

    def compute_ready_nodes(
        self, graph: WorkflowGraph, node_statuses: Dict[str, str]
    ) -> List[str]:
        ready, _ = self.compute_ready_nodes_with_conditions(graph, node_statuses, {})
        return ready

    def compute_ready_nodes_with_conditions(
        self,
        graph: WorkflowGraph,
        node_statuses: Dict[str, str],
        shared_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        predecessors = self._build_predecessor_map(graph)
        incoming_edges = self._build_incoming_edge_map(graph)
        ready = []
        skipped = []

        for node_name, status in node_statuses.items():
            if status != "pending":
                continue
            preds = predecessors.get(node_name, set())
            if not preds:
                ready.append(node_name)
                continue
            all_preds_done = all(
                node_statuses.get(p) in _DONE_STATUSES for p in preds
            )
            if not all_preds_done:
                continue
            edges_to_this = incoming_edges.get(node_name, [])
            if edges_to_this and shared_data is not None:
                # Convert string statuses to NodeStatus enums for WorkflowState
                _STATUS_MAP = {
                    "pending": NodeStatus.PENDING, "ready": NodeStatus.PENDING,
                    "running": NodeStatus.RUNNING, "completed": NodeStatus.COMPLETED,
                    "failed": NodeStatus.FAILED, "skipped": NodeStatus.SKIPPED,
                    "timed_out": NodeStatus.FAILED,
                }
                enum_statuses = {
                    k: _STATUS_MAP.get(v, NodeStatus.PENDING)
                    for k, v in node_statuses.items()
                }
                mock_state = WorkflowState(
                    workflow_id="", workflow_name="", user_id="",
                    status="running", current_node=None,
                    node_statuses=enum_statuses,
                    node_results={}, shared_data=shared_data or {},
                    routing_history=[], errors=[], checkpoints=[],
                )
                any_traversable = any(
                    e.should_traverse(mock_state) for e in edges_to_this
                    if e.condition is not None
                )
                has_conditions = any(e.condition is not None for e in edges_to_this)
                if has_conditions and not any_traversable:
                    skipped.append(node_name)
                    continue
            ready.append(node_name)

        return ready, skipped

    def is_workflow_complete(
        self, graph: WorkflowGraph, node_statuses: Dict[str, str]
    ) -> bool:
        return all(
            status in _TERMINAL_STATUSES
            for status in node_statuses.values()
        )

    # ── Internal Helpers ────────────────────────────────────────

    def _evaluate_and_dispatch(
        self, graph: WorkflowGraph, execution_id: int, workflow_id: str, round_num: int
    ) -> None:
        node_statuses = self._state.get_node_statuses(execution_id, round_num)
        wf = self._state.get_workflow_execution(execution_id)
        shared_data = wf["shared_data"] if wf else {}
        ready, skipped = self.compute_ready_nodes_with_conditions(
            graph, node_statuses, shared_data,
        )
        for skip_name in skipped:
            self._state.update_node_status(execution_id, skip_name, round_num, "skipped")
        if ready:
            self._dispatch_nodes(graph, execution_id, workflow_id, ready, round_num, shared_data)

    def _dispatch_nodes(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        node_names: List[str],
        round_num: int,
        shared_data: Dict[str, Any],
    ) -> None:
        for node_name in node_names:
            node = graph.nodes[node_name]
            task_id = f"t-{uuid.uuid4().hex[:12]}"
            context = {
                "workflow_id": workflow_id,
                "agent_id": node.agent_type,
                "user_id": "",
                "input_data": node.build_input_data(
                    WorkflowState(
                        workflow_id=workflow_id, workflow_name=graph.name,
                        user_id="", status="running", current_node=node_name,
                        node_statuses={}, node_results={},
                        shared_data=shared_data, routing_history=[],
                        errors=[], checkpoints=[],
                    )
                ),
                "policy_context": [],
                "knowledge_base": {},
                "routing_history": [],
                "task_profile": {},
                "task_id": task_id,
            }
            msg = TaskMessage(
                task_id=task_id,
                workflow_id=workflow_id,
                node_name=node_name,
                agent_type=node.agent_type,
                context=context,
                priority="normal",
                dispatched_at=datetime.now(timezone.utc),
                timeout_seconds=node.timeout_seconds or self._default_timeout,
                attempt=1,
            )
            self._state.update_node_status(
                execution_id, node_name, round_num, "running",
                dispatched_at=msg.dispatched_at, attempt=1,
            )
            self._dispatcher.dispatch(msg)

    def _complete_workflow(self, execution_id: int, workflow_id: str) -> None:
        now = datetime.now(timezone.utc)
        self._state.update_workflow_status(execution_id, "completed", completed_at=now)
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:completed",
            payload={},
        )
        logger.info("Workflow %s completed (execution_id=%d)", workflow_id, execution_id)

    def _fail_workflow(
        self, execution_id: int, workflow_id: str,
        graph: WorkflowGraph, node_statuses: Dict[str, str],
    ) -> None:
        now = datetime.now(timezone.utc)
        self._state.update_workflow_status(execution_id, "failed", completed_at=now)
        running_types = [
            graph.nodes[n].agent_type
            for n, s in node_statuses.items()
            if s == "running"
        ]
        if running_types:
            self._dispatcher.publish_cancellation(workflow_id, running_types)
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:failed",
            payload={"reason": "required_node_failed"},
        )
        logger.error("Workflow %s failed (execution_id=%d)", workflow_id, execution_id)

    def _has_failed_required(
        self, graph: WorkflowGraph, node_statuses: Dict[str, str]
    ) -> bool:
        for name, status in node_statuses.items():
            if status in ("failed", "timed_out"):
                node = graph.nodes.get(name)
                if node and node.required:
                    return True
        return False

    @staticmethod
    def _build_predecessor_map(graph: WorkflowGraph) -> Dict[str, set]:
        preds: Dict[str, set] = {name: set() for name in graph.nodes}
        for edge in graph.edges:
            preds[edge.target].add(edge.source)
        return preds

    @staticmethod
    def _build_incoming_edge_map(graph: WorkflowGraph) -> Dict[str, List[WorkflowEdge]]:
        incoming: Dict[str, List[WorkflowEdge]] = {name: [] for name in graph.nodes}
        for edge in graph.edges:
            incoming[edge.target].append(edge)
        return incoming
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_dag_scheduler.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/dag_scheduler.py tests/test_dag_scheduler.py
git commit -m "feat: add DAG Scheduler with parallel node dispatch"
```

---

## Task 6: Result Collector

**Files:**
- Create: `src/orchestration/result_collector.py`
- Create: `tests/test_result_collector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_result_collector.py
import pytest
from unittest.mock import MagicMock, patch, call
import json
from datetime import datetime, timezone


def _make_result_payload(task_id="t-001", workflow_id="wf-123", node_name="extract",
                          agent_type="data_extraction", status="SUCCESS"):
    from orchestration.message_protocol import ResultMessage
    return ResultMessage(
        task_id=task_id, workflow_id=workflow_id, node_name=node_name,
        agent_type=agent_type, status=status,
        data={"details": "extracted"}, pass_fields={"details": "extracted"},
        next_agents=[], error=None, confidence=0.95,
        completed_at=datetime(2026, 3, 31, 12, 0, 5, tzinfo=timezone.utc),
        duration_ms=1500,
    )


def test_process_success_result_updates_state():
    from orchestration.result_collector import ResultCollector

    state_manager = MagicMock()
    dag_scheduler = MagicMock()
    state_manager.get_workflow_execution.return_value = {
        "execution_id": 42, "workflow_id": "wf-123",
        "current_round": 0, "status": "running",
    }

    collector = ResultCollector(
        state_manager=state_manager,
        dag_scheduler=dag_scheduler,
        workflow_graphs={},
    )
    result = _make_result_payload()
    collector.process_result(result, execution_id=42)

    state_manager.record_node_result.assert_called_once()
    state_manager.merge_shared_data.assert_called_once_with(42, {"details": "extracted"})


def test_process_failure_result_calls_on_node_failed():
    from orchestration.result_collector import ResultCollector

    state_manager = MagicMock()
    dag_scheduler = MagicMock()
    state_manager.get_workflow_execution.return_value = {
        "execution_id": 42, "workflow_id": "wf-123",
        "current_round": 0, "status": "running",
    }

    collector = ResultCollector(
        state_manager=state_manager,
        dag_scheduler=dag_scheduler,
        workflow_graphs={"test": MagicMock()},
    )
    result = _make_result_payload(status="FAILED")
    collector.process_result(result, execution_id=42, graph_name="test")

    state_manager.record_node_result.assert_called_once()
    dag_scheduler.on_node_failed.assert_called_once()


def test_process_result_records_event():
    from orchestration.result_collector import ResultCollector

    state_manager = MagicMock()
    dag_scheduler = MagicMock()
    state_manager.get_workflow_execution.return_value = {
        "execution_id": 42, "workflow_id": "wf-123",
        "current_round": 0, "status": "running",
    }

    collector = ResultCollector(
        state_manager=state_manager,
        dag_scheduler=dag_scheduler,
        workflow_graphs={},
    )
    result = _make_result_payload()
    collector.process_result(result, execution_id=42)

    state_manager.record_event.assert_called()
    event_call = state_manager.record_event.call_args
    assert event_call.kwargs["event_type"] == "node:completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_result_collector.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ResultCollector**

```python
# src/orchestration/result_collector.py
"""Consumes agent results from Redis Streams and updates workflow state.

Listens on agent:results stream, matches results to workflows,
updates PostgreSQL state, and notifies DAG Scheduler for downstream evaluation.

Spec reference: Section 1.4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from orchestration.message_protocol import ResultMessage
from orchestration.state_manager import StateManager

logger = logging.getLogger(__name__)

RESULT_STREAM = "agent:results"


class ResultCollector:
    def __init__(
        self,
        state_manager: StateManager,
        dag_scheduler: Any,
        workflow_graphs: Dict[str, Any],
    ):
        self._state = state_manager
        self._scheduler = dag_scheduler
        self._graphs = workflow_graphs

    def process_result(
        self,
        result: ResultMessage,
        execution_id: int,
        graph_name: Optional[str] = None,
    ) -> None:
        wf = self._state.get_workflow_execution(execution_id)
        if not wf:
            logger.warning("No workflow execution found for id=%d", execution_id)
            return

        current_round = wf["current_round"]
        workflow_id = wf["workflow_id"]

        # Record node result in PostgreSQL
        self._state.record_node_result(
            execution_id=execution_id,
            node_name=result.node_name,
            round_num=current_round,
            status="completed" if result.status == "SUCCESS" else "failed",
            output_data=result.data,
            pass_fields=result.pass_fields,
            error=result.error,
            duration_ms=result.duration_ms,
        )

        # Merge pass_fields into shared_data
        if result.pass_fields and result.status == "SUCCESS":
            self._state.merge_shared_data(execution_id, result.pass_fields)

        # Record observability event
        event_type = "node:completed" if result.status == "SUCCESS" else "node:failed"
        self._state.record_event(
            workflow_id=workflow_id,
            event_type=event_type,
            node_name=result.node_name,
            agent_type=result.agent_type,
            payload={
                "task_id": result.task_id,
                "duration_ms": result.duration_ms,
                "confidence": result.confidence,
                "error": result.error,
            },
            round_num=current_round,
        )

        # Notify DAG Scheduler
        graph = self._graphs.get(graph_name) if graph_name else None
        if graph is None:
            wf_name = wf.get("workflow_name", "")
            graph = self._graphs.get(wf_name)

        if graph is None:
            logger.warning("No graph found for workflow %s", workflow_id)
            return

        if result.status == "SUCCESS":
            self._scheduler.on_node_completed(
                graph=graph,
                execution_id=execution_id,
                workflow_id=workflow_id,
                node_name=result.node_name,
                round_num=current_round,
            )
        else:
            self._scheduler.on_node_failed(
                graph=graph,
                execution_id=execution_id,
                workflow_id=workflow_id,
                node_name=result.node_name,
                round_num=current_round,
            )

        logger.info(
            "Processed result for %s/%s: %s (duration=%dms)",
            workflow_id, result.node_name, result.status, result.duration_ms,
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_result_collector.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/result_collector.py tests/test_result_collector.py
git commit -m "feat: add ResultCollector for processing agent results"
```

---

## Task 7: Worker Context & Agent Worker

**Files:**
- Create: `src/orchestration/worker_context.py`
- Create: `src/orchestration/worker.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write failing tests for WorkerContext**

```python
# tests/test_worker.py
import pytest
from unittest.mock import MagicMock, patch
import json
import time
from datetime import datetime, timezone


def test_worker_context_provides_settings():
    from orchestration.worker_context import WorkerContext

    settings = MagicMock()
    settings.db_host = "localhost"
    ctx = WorkerContext(settings=settings, agent_types=["data_extraction"])
    assert ctx.settings.db_host == "localhost"


def test_worker_context_creates_agent_instances():
    from orchestration.worker_context import WorkerContext

    settings = MagicMock()
    mock_agent = MagicMock()
    agent_factory = MagicMock()
    agent_factory.create.return_value = mock_agent

    ctx = WorkerContext(
        settings=settings,
        agent_types=["data_extraction"],
        agent_factory=agent_factory,
    )
    agent = ctx.get_agent("data_extraction")
    assert agent == mock_agent
    agent_factory.create.assert_called_with("data_extraction")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write failing tests for Worker execution**

```python
# tests/test_worker.py (append)


def test_worker_executes_agent_and_publishes_result():
    from orchestration.worker import AgentWorker
    from orchestration.message_protocol import TaskMessage
    from agents.base_agent import AgentOutput, AgentStatus

    mock_agent = MagicMock()
    mock_agent.execute.return_value = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={"details": "extracted"},
        pass_fields={"details": "extracted"},
    )

    worker_ctx = MagicMock()
    worker_ctx.get_agent.return_value = mock_agent
    redis_client = MagicMock()

    worker = AgentWorker(
        agent_type="data_extraction",
        worker_context=worker_ctx,
        redis_client=redis_client,
    )

    task = TaskMessage(
        task_id="t-001", workflow_id="wf-123", node_name="extract",
        agent_type="data_extraction",
        context={"workflow_id": "wf-123", "agent_id": "data_extraction",
                 "user_id": "u1", "input_data": {"file": "test.pdf"},
                 "policy_context": [], "knowledge_base": {},
                 "routing_history": [], "task_profile": {}},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300, attempt=1,
    )

    worker.execute_task(task)

    mock_agent.execute.assert_called_once()
    # Should publish result to agent:results stream
    result_call = [c for c in redis_client.xadd.call_args_list if c[0][0] == "agent:results"]
    assert len(result_call) == 1


def test_worker_handles_agent_failure():
    from orchestration.worker import AgentWorker
    from orchestration.message_protocol import TaskMessage

    mock_agent = MagicMock()
    mock_agent.execute.side_effect = RuntimeError("LLM timeout")

    worker_ctx = MagicMock()
    worker_ctx.get_agent.return_value = mock_agent
    redis_client = MagicMock()

    worker = AgentWorker(
        agent_type="data_extraction",
        worker_context=worker_ctx,
        redis_client=redis_client,
    )

    task = TaskMessage(
        task_id="t-002", workflow_id="wf-123", node_name="extract",
        agent_type="data_extraction",
        context={"workflow_id": "wf-123", "agent_id": "data_extraction",
                 "user_id": "u1", "input_data": {},
                 "policy_context": [], "knowledge_base": {},
                 "routing_history": [], "task_profile": {}},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300, attempt=1,
    )

    worker.execute_task(task)

    result_call = [c for c in redis_client.xadd.call_args_list if c[0][0] == "agent:results"]
    assert len(result_call) == 1
    payload = json.loads(result_call[0].args[1]["payload"])
    assert payload["status"] == "FAILED"
    assert "LLM timeout" in payload["error"]
```

- [ ] **Step 4: Implement WorkerContext**

```python
# src/orchestration/worker_context.py
"""Lightweight replacement for agent_nick in worker processes.

Only initializes dependencies needed by the specific agent types
this worker serves. Implements the same interface agents expect.

Spec reference: Section 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class WorkerContext:
    def __init__(
        self,
        settings: Any,
        agent_types: List[str],
        agent_factory: Any = None,
        db_pool: Any = None,
        redis_client: Any = None,
        process_routing_service: Any = None,
    ):
        self.settings = settings
        self._agent_types = agent_types
        self._agent_factory = agent_factory
        self._db_pool = db_pool
        self._redis_client = redis_client
        self.process_routing_service = process_routing_service
        self._agents: Dict[str, Any] = {}

    def get_agent(self, agent_type: str) -> Any:
        if agent_type not in self._agents:
            if self._agent_factory:
                self._agents[agent_type] = self._agent_factory.create(agent_type)
            else:
                raise ValueError(f"No factory to create agent: {agent_type}")
        return self._agents[agent_type]

    def get_connection(self) -> Any:
        if self._db_pool:
            return self._db_pool.getconn()
        raise RuntimeError("No database pool configured")
```

- [ ] **Step 5: Implement AgentWorker**

```python
# src/orchestration/worker.py
"""Agent worker process: consumes tasks from Redis Streams, executes agents.

Each worker serves one agent type. It reads from agent:tasks:{agent_type},
executes the agent, and publishes results to agent:results.

Spec reference: Section 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from orchestration.message_protocol import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)

RESULT_STREAM = "agent:results"


class AgentWorker:
    def __init__(
        self,
        agent_type: str,
        worker_context: Any,
        redis_client: Any,
        consumer_group: str = "workers",
        consumer_name: Optional[str] = None,
    ):
        self.agent_type = agent_type
        self._ctx = worker_context
        self._redis = redis_client
        self._group = consumer_group
        self._consumer = consumer_name or f"worker-{agent_type}-{id(self)}"
        self._stream = f"agent:tasks:{agent_type}"

    def execute_task(self, task: TaskMessage) -> None:
        start = time.monotonic()
        try:
            agent = self._ctx.get_agent(task.agent_type)
            context = AgentContext(
                workflow_id=task.context.get("workflow_id", ""),
                agent_id=task.context.get("agent_id", task.agent_type),
                user_id=task.context.get("user_id", ""),
                input_data=task.context.get("input_data", {}),
                policy_context=task.context.get("policy_context", []),
                knowledge_base=task.context.get("knowledge_base", {}),
                routing_history=task.context.get("routing_history", []),
                task_profile=task.context.get("task_profile", {}),
            )
            result = agent.execute(context)
            duration_ms = int((time.monotonic() - start) * 1000)
            self._publish_result(task, result, duration_ms)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Agent %s failed for task %s", task.agent_type, task.task_id)
            error_result = AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(exc),
            )
            self._publish_result(task, error_result, duration_ms)

    def _publish_result(
        self, task: TaskMessage, result: AgentOutput, duration_ms: int
    ) -> None:
        msg = ResultMessage(
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            node_name=task.node_name,
            agent_type=task.agent_type,
            status=result.status.value.upper() if hasattr(result.status, 'value') else str(result.status).upper(),
            data=result.data or {},
            pass_fields=result.pass_fields or {},
            next_agents=result.next_agents or [],
            error=result.error,
            confidence=result.confidence,
            completed_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
        )
        self._redis.xadd(RESULT_STREAM, msg.to_redis())
        logger.info(
            "Published result for %s/%s: %s (%dms)",
            task.workflow_id, task.node_name, msg.status, duration_ms,
        )
```

- [ ] **Step 6: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_worker.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/orchestration/worker_context.py src/orchestration/worker.py tests/test_worker.py
git commit -m "feat: add WorkerContext and AgentWorker for distributed execution"
```

---

## Task 8: Settings & Feature Flag

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Read current settings file**

Read: `config/settings.py` to find where to add new settings.

- [ ] **Step 2: Add new settings fields**

Add the following fields to the Settings class (after existing redis settings):

```python
# Orchestration DAG Scheduler
use_dag_scheduler: bool = False  # Feature flag: use new DAG scheduler vs legacy
redis_streams_url: Optional[str] = None  # Falls back to redis_url if not set
worker_heartbeat_interval: int = 30  # Seconds between heartbeats
worker_task_timeout: int = 300  # Default task timeout in seconds
worker_max_stream_len: int = 10000  # Redis Stream MAXLEN cap
```

- [ ] **Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat: add DAG scheduler feature flag and worker settings"
```

---

## Task 9: Orchestrator Integration (Feature Flag)

**Files:**
- Modify: `src/orchestration/orchestrator.py` (lines 227-261, the workflow execution branch)

- [ ] **Step 1: Read the execute_workflow method**

Read: `src/orchestration/orchestrator.py:173-314`

- [ ] **Step 2: Add DAG Scheduler branch**

In `execute_workflow()`, before the existing `use_workflow_engine` check (around line 227), add a new branch:

```python
# --- New DAG Scheduler path (Phase 3 migration) ---
if getattr(self.settings, 'use_dag_scheduler', False) and workflow_name in self._workflow_registry:
    scheduler = self._get_dag_scheduler()  # Lazy singleton
    graph = get_workflow(workflow_name)
    execution_id = scheduler.start_workflow(
        graph=graph,
        input_data=enriched_input,
        workflow_id=workflow_id,
        user_id=user_id,
    )
    return {
        "workflow_id": workflow_id,
        "execution_id": execution_id,
        "status": "running",
        "execution_mode": "dag_scheduler",
    }
```

Also add this helper method to the Orchestrator class:

```python
def _get_dag_scheduler(self):
    """Lazy-initialize DAG Scheduler (singleton per Orchestrator instance)."""
    if not hasattr(self, '_dag_scheduler'):
        from orchestration.dag_scheduler import DAGScheduler
        from orchestration.state_manager import StateManager
        from orchestration.task_dispatcher import TaskDispatcher
        import redis

        redis_client = redis.from_url(
            self.settings.redis_streams_url or self.settings.redis_url
        )
        state_mgr = StateManager(get_connection=self.agent_nick.get_connection)
        dispatcher = TaskDispatcher(
            redis_client=redis_client,
            max_stream_len=getattr(self.settings, 'worker_max_stream_len', 10000),
        )
        self._dag_scheduler = DAGScheduler(
            state_manager=state_mgr,
            task_dispatcher=dispatcher,
            agent_registry=self.agents,
            manifest_service=getattr(self, 'manifest_service', None),
        )
    return self._dag_scheduler
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_orchestrator_supplier_workflow.py tests/test_procurement_workflow.py tests/test_generic_workflow_context.py -v 2>&1 | head -50`
Expected: Existing tests unchanged (feature flag defaults to False)

- [ ] **Step 4: Commit**

```bash
git add src/orchestration/orchestrator.py
git commit -m "feat: add DAG scheduler feature flag branch in Orchestrator"
```

---

## Task 10: Observability API Endpoints

**Files:**
- Modify: `src/api/routers/workflows.py`

- [ ] **Step 1: Read current workflows router for import patterns**

Read: `src/api/routers/workflows.py:1-50`

- [ ] **Step 2: Add observability endpoints**

Append the following endpoints to workflows.py:

```python
# ── Observability Endpoints ─────────────────────────────────────

@router.get("/workflows/{workflow_id}/status")
def get_workflow_status(workflow_id: str, request: Request):
    """Get current workflow state and all node statuses."""
    from orchestration.state_manager import StateManager
    orchestrator = get_orchestrator(request)
    sm = StateManager(get_connection=orchestrator.agent_nick.get_connection)
    # Find execution by workflow_id
    conn = orchestrator.agent_nick.get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT execution_id, workflow_name, status, shared_data,
                      current_round, created_at, updated_at, completed_at
               FROM proc.workflow_execution
               WHERE workflow_id = %s
               ORDER BY created_at DESC LIMIT 1""",
            (workflow_id,),
        )
        row = cur.fetchone()
        if not row:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Workflow not found")
        execution_id = row[0]
        node_statuses = sm.get_node_statuses(execution_id, row[4])
        return {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "workflow_name": row[1],
            "status": row[2],
            "current_round": row[4],
            "created_at": str(row[5]),
            "updated_at": str(row[6]),
            "completed_at": str(row[7]) if row[7] else None,
            "nodes": node_statuses,
        }


@router.get("/workflows/{workflow_id}/events")
def get_workflow_events(workflow_id: str, request: Request, limit: int = 100):
    """Get event timeline for a workflow."""
    from orchestration.state_manager import StateManager
    orchestrator = get_orchestrator(request)
    sm = StateManager(get_connection=orchestrator.agent_nick.get_connection)
    events = sm.get_events(workflow_id, limit=limit)
    return {"workflow_id": workflow_id, "events": events}


@router.get("/system/workflows/active")
def get_active_workflows(request: Request):
    """List all running/paused workflows."""
    from orchestration.state_manager import StateManager
    orchestrator = get_orchestrator(request)
    sm = StateManager(get_connection=orchestrator.agent_nick.get_connection)
    return {"workflows": sm.get_active_workflows()}
```

- [ ] **Step 3: Commit**

```bash
git add src/api/routers/workflows.py
git commit -m "feat: add workflow observability API endpoints"
```

---

## Task 11: Worker CLI Entry Point

**Files:**
- Create: `src/orchestration/cli.py`

- [ ] **Step 1: Implement CLI**

```python
# src/orchestration/cli.py
"""CLI entry point for running agent workers.

Usage:
    python -m src.orchestration.cli --all
    python -m src.orchestration.cli --agents supplier_ranking,negotiation
    python -m src.orchestration.cli --agents data_extraction --concurrency 4

Spec reference: Section 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import List

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProcWise Agent Worker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run workers for all agent types")
    group.add_argument("--agents", type=str, help="Comma-separated agent types to serve")
    parser.add_argument("--concurrency", type=int, default=1, help="Workers per agent type")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


ALL_AGENT_TYPES = [
    "data_extraction", "supplier_ranking", "quote_comparison",
    "quote_evaluation", "opportunity_miner", "email_drafting",
    "email_dispatch", "email_watcher", "negotiation",
    "supplier_interaction", "approvals", "discrepancy_detection", "rag",
]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.all:
        agent_types = ALL_AGENT_TYPES
    else:
        agent_types = [a.strip() for a in args.agents.split(",")]

    logger.info(
        "Starting workers for %d agent type(s): %s (concurrency=%d)",
        len(agent_types), agent_types, args.concurrency,
    )

    # Import here to avoid loading heavy deps at parse time
    import redis
    from config.settings import Settings
    from orchestration.worker import AgentWorker
    from orchestration.worker_context import WorkerContext

    settings = Settings()
    redis_url = settings.redis_streams_url or settings.redis_url
    redis_client = redis.from_url(redis_url)

    worker_ctx = WorkerContext(settings=settings, agent_types=agent_types)

    shutdown = threading.Event()

    def _signal_handler(signum, frame):
        logger.info("Shutdown signal received, finishing current tasks...")
        shutdown.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    threads: List[threading.Thread] = []
    for agent_type in agent_types:
        for i in range(args.concurrency):
            worker = AgentWorker(
                agent_type=agent_type,
                worker_context=worker_ctx,
                redis_client=redis_client,
                consumer_name=f"worker-{agent_type}-{i}",
            )
            t = threading.Thread(
                target=_run_worker_loop,
                args=(worker, redis_client, shutdown),
                name=f"worker-{agent_type}-{i}",
                daemon=True,
            )
            threads.append(t)
            t.start()

    logger.info("All %d worker threads started", len(threads))
    shutdown.wait()
    logger.info("Shutdown complete")


def _run_worker_loop(
    worker: "AgentWorker", redis_client, shutdown: threading.Event
) -> None:
    from orchestration.message_protocol import TaskMessage

    stream = f"agent:tasks:{worker.agent_type}"
    group = "workers"

    # Ensure consumer group exists
    try:
        redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception:
        pass  # Group already exists

    while not shutdown.is_set():
        try:
            messages = redis_client.xreadgroup(
                group, worker._consumer, {stream: ">"}, count=1, block=5000,
            )
            if not messages:
                continue
            for _stream, entries in messages:
                for msg_id, fields in entries:
                    task = TaskMessage.from_redis(fields)
                    worker.execute_task(task)
                    redis_client.xack(stream, group, msg_id)
        except Exception:
            if not shutdown.is_set():
                logger.exception("Worker %s error", worker._consumer)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses correctly**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m src.orchestration.cli --help`
Expected: Shows usage with --all, --agents, --concurrency options

- [ ] **Step 3: Commit**

```bash
git add src/orchestration/cli.py
git commit -m "feat: add procwise-worker CLI entry point"
```

---

## Task 12: Integration Test

**Files:**
- Create: `tests/test_integration_workflow.py`

- [ ] **Step 1: Write end-to-end test with mock Redis and mock agents**

```python
# tests/test_integration_workflow.py
"""Integration test: full workflow through DAG Scheduler with mocked I/O.

Validates the complete flow: start_workflow -> dispatch -> execute -> collect -> complete
without requiring Redis or PostgreSQL.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from orchestration.workflow_engine import WorkflowGraph, WorkflowNode
from orchestration.message_protocol import TaskMessage, ResultMessage
from orchestration.state_manager import StateManager
from orchestration.task_dispatcher import TaskDispatcher
from orchestration.dag_scheduler import DAGScheduler
from orchestration.result_collector import ResultCollector
from agents.base_agent import AgentOutput, AgentStatus


class InMemoryStateManager:
    """StateManager substitute backed by dicts instead of PostgreSQL."""

    def __init__(self):
        self._executions = {}
        self._nodes = {}
        self._events = []
        self._next_id = 1

    def create_workflow_execution(self, workflow_id, workflow_name, user_id=None):
        eid = self._next_id
        self._next_id += 1
        self._executions[eid] = {
            "execution_id": eid, "workflow_id": workflow_id,
            "workflow_name": workflow_name, "user_id": user_id,
            "status": "pending", "shared_data": {}, "current_round": 0,
        }
        return eid

    def update_workflow_status(self, execution_id, status, completed_at=None):
        self._executions[execution_id]["status"] = status

    def get_workflow_execution(self, execution_id):
        return self._executions.get(execution_id)

    def get_active_workflows(self):
        return [e for e in self._executions.values() if e["status"] in ("pending", "running", "paused")]

    def increment_round(self, execution_id):
        self._executions[execution_id]["current_round"] += 1
        return self._executions[execution_id]["current_round"]

    def merge_shared_data(self, execution_id, new_fields):
        self._executions[execution_id]["shared_data"].update(new_fields)

    def create_node_executions(self, execution_id, nodes, round_num=0):
        for n in nodes:
            key = (execution_id, n["node_name"], round_num)
            self._nodes[key] = {"status": "pending", "agent_type": n["agent_type"], "attempt": 0}

    def update_node_status(self, execution_id, node_name, round_num, status, dispatched_at=None, attempt=None):
        key = (execution_id, node_name, round_num)
        if key in self._nodes:
            self._nodes[key]["status"] = status

    def record_node_result(self, execution_id, node_name, round_num, status, output_data=None, pass_fields=None, error=None, duration_ms=None):
        key = (execution_id, node_name, round_num)
        if key in self._nodes:
            self._nodes[key]["status"] = status
            self._nodes[key]["output_data"] = output_data
            self._nodes[key]["pass_fields"] = pass_fields

    def get_ready_nodes(self, execution_id, current_round):
        return [
            {"node_name": k[1], "agent_type": v["agent_type"], "attempt": v["attempt"]}
            for k, v in self._nodes.items()
            if k[0] == execution_id and k[2] == current_round and v["status"] == "ready"
        ]

    def get_node_statuses(self, execution_id, current_round):
        return {
            k[1]: v["status"]
            for k, v in self._nodes.items()
            if k[0] == execution_id and k[2] == current_round
        }

    def record_event(self, workflow_id, event_type, node_name=None, agent_type=None, payload=None, round_num=None):
        self._events.append({"workflow_id": workflow_id, "event_type": event_type, "node_name": node_name})

    def get_events(self, workflow_id, limit=100):
        return [e for e in self._events if e["workflow_id"] == workflow_id][:limit]


def test_linear_workflow_end_to_end():
    """extract -> rank -> draft: all succeed."""
    graph = WorkflowGraph(name="test_linear")
    graph.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    graph.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
    graph.add_node(WorkflowNode(name="draft", agent_type="email_drafting"))
    graph.add_edge("extract", "rank")
    graph.add_edge("rank", "draft")

    state_mgr = InMemoryStateManager()
    dispatched_tasks = []
    redis_mock = MagicMock()
    redis_mock.xadd = MagicMock()

    dispatcher = TaskDispatcher(redis_client=redis_mock, max_stream_len=1000)
    # Capture dispatched tasks
    original_dispatch = dispatcher.dispatch
    def capture_dispatch(task):
        dispatched_tasks.append(task)
        return original_dispatch(task)
    dispatcher.dispatch = capture_dispatch

    scheduler = DAGScheduler(
        state_manager=state_mgr,
        task_dispatcher=dispatcher,
        agent_registry=MagicMock(),
    )
    collector = ResultCollector(
        state_manager=state_mgr,
        dag_scheduler=scheduler,
        workflow_graphs={"test_linear": graph},
    )

    # Start workflow
    exec_id = scheduler.start_workflow(graph, {"query": "test"}, workflow_id="wf-e2e")
    assert state_mgr._executions[exec_id]["status"] == "running"
    assert len(dispatched_tasks) == 1
    assert dispatched_tasks[0].node_name == "extract"

    # Simulate extract completion
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[0].task_id, workflow_id="wf-e2e",
            node_name="extract", agent_type="data_extraction", status="SUCCESS",
            data={"details": "ok"}, pass_fields={"details": "ok"},
            next_agents=[], error=None, confidence=0.9,
            completed_at=datetime.now(timezone.utc), duration_ms=100,
        ),
        execution_id=exec_id,
    )
    assert len(dispatched_tasks) == 2
    assert dispatched_tasks[1].node_name == "rank"

    # Simulate rank completion
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[1].task_id, workflow_id="wf-e2e",
            node_name="rank", agent_type="supplier_ranking", status="SUCCESS",
            data={"ranking": [1]}, pass_fields={"ranking": [1]},
            next_agents=[], error=None, confidence=0.8,
            completed_at=datetime.now(timezone.utc), duration_ms=200,
        ),
        execution_id=exec_id,
    )
    assert len(dispatched_tasks) == 3
    assert dispatched_tasks[2].node_name == "draft"

    # Simulate draft completion
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[2].task_id, workflow_id="wf-e2e",
            node_name="draft", agent_type="email_drafting", status="SUCCESS",
            data={"drafts": ["email1"]}, pass_fields={},
            next_agents=[], error=None, confidence=0.95,
            completed_at=datetime.now(timezone.utc), duration_ms=300,
        ),
        execution_id=exec_id,
    )

    # Workflow should be complete
    assert state_mgr._executions[exec_id]["status"] == "completed"
    events = state_mgr.get_events("wf-e2e")
    event_types = [e["event_type"] for e in events]
    assert "workflow:started" in event_types
    assert "workflow:completed" in event_types


def test_parallel_dispatch():
    """extract -> [eval, compare]: both dispatched simultaneously."""
    graph = WorkflowGraph(name="test_parallel")
    graph.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    graph.add_node(WorkflowNode(name="eval", agent_type="quote_evaluation"))
    graph.add_node(WorkflowNode(name="compare", agent_type="quote_comparison"))
    graph.add_edge("extract", "eval")
    graph.add_edge("extract", "compare")

    state_mgr = InMemoryStateManager()
    dispatched_tasks = []
    redis_mock = MagicMock()

    dispatcher = TaskDispatcher(redis_client=redis_mock, max_stream_len=1000)
    original_dispatch = dispatcher.dispatch
    def capture_dispatch(task):
        dispatched_tasks.append(task)
        return original_dispatch(task)
    dispatcher.dispatch = capture_dispatch

    scheduler = DAGScheduler(
        state_manager=state_mgr,
        task_dispatcher=dispatcher,
        agent_registry=MagicMock(),
    )
    collector = ResultCollector(
        state_manager=state_mgr,
        dag_scheduler=scheduler,
        workflow_graphs={"test_parallel": graph},
    )

    exec_id = scheduler.start_workflow(graph, {}, workflow_id="wf-par")
    assert len(dispatched_tasks) == 1  # extract first

    # Complete extract
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[0].task_id, workflow_id="wf-par",
            node_name="extract", agent_type="data_extraction", status="SUCCESS",
            data={}, pass_fields={}, next_agents=[], error=None, confidence=0.9,
            completed_at=datetime.now(timezone.utc), duration_ms=100,
        ),
        execution_id=exec_id,
    )

    # Both eval and compare should now be dispatched
    assert len(dispatched_tasks) == 3
    dispatched_names = {t.node_name for t in dispatched_tasks[1:]}
    assert dispatched_names == {"eval", "compare"}
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_integration_workflow.py -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_workflow.py
git commit -m "feat: add integration tests for DAG Scheduler workflow execution"
```

---

## Task 13: Update orchestration __init__.py exports

**Files:**
- Modify: `src/orchestration/__init__.py`

- [ ] **Step 1: Read current __init__.py**

Read: `src/orchestration/__init__.py`

- [ ] **Step 2: Add lazy exports for new modules**

Add the new module names to the lazy loading `__getattr__` function so they can be imported from `src.orchestration`:

```python
# Add to the lazy-loaded module mapping:
"dag_scheduler": "src.orchestration.dag_scheduler",
"task_dispatcher": "src.orchestration.task_dispatcher",
"result_collector": "src.orchestration.result_collector",
"state_manager": "src.orchestration.state_manager",
"worker": "src.orchestration.worker",
"worker_context": "src.orchestration.worker_context",
"message_protocol": "src.orchestration.message_protocol",
```

- [ ] **Step 3: Commit**

```bash
git add src/orchestration/__init__.py
git commit -m "feat: export new orchestration modules via lazy loading"
```

---

## Task 14: Run full test suite

- [ ] **Step 1: Run all new tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_message_protocol.py tests/test_state_manager.py tests/test_task_dispatcher.py tests/test_dag_scheduler.py tests/test_result_collector.py tests/test_worker.py tests/test_integration_workflow.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/ -v --ignore=tests/test_message_protocol.py --ignore=tests/test_state_manager.py --ignore=tests/test_task_dispatcher.py --ignore=tests/test_dag_scheduler.py --ignore=tests/test_result_collector.py --ignore=tests/test_worker.py --ignore=tests/test_integration_workflow.py 2>&1 | tail -20`
Expected: No new failures (feature flag defaults to False, so all existing paths unchanged)

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address test regressions from orchestration rearchitecture"
```
