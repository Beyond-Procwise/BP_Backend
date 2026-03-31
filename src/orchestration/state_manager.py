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
