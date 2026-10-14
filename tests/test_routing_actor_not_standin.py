"""A routing record names the person who acted, or nobody -- never "AgentNick".

proc.routing had 48 rows created by 'AgentNick'. ProcessRoutingService wrote
settings.script_user into created_by, modified_by and the run's triggered_by
whenever a caller named nobody -- and BaseAgent wrote it into user_name on every
agent run, a column the gateway fills with a real person's display name. On a
record whose other rows carry people, a stand-in reads as a person, and "the
service did it" cannot be told apart from "somebody called AgentNick did it".

Told nobody, the record now says nobody. All four columns are nullable, the
gateway only writes them, and nothing in this repo reads them back.
"""

from __future__ import annotations

import ast
import json
import pathlib
from datetime import datetime
from types import SimpleNamespace

from services.process_routing_service import ProcessRoutingService

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Cur:
    def __init__(self, log):
        self._log = log

    def execute(self, sql, params=None):
        self._log.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self):
        self.executed = []

    def cursor(self):
        return _Cur(self.executed)

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _service():
    conn = _Conn()
    agent = SimpleNamespace(get_db_connection=lambda: conn,
                            settings=SimpleNamespace(script_user="AgentNick"))
    return ProcessRoutingService(agent), conn


def _routing_updates(conn):
    return [(sql, params) for sql, params in conn.executed
            if sql.startswith("UPDATE proc.routing") and "modified_by" in sql]


def test_a_process_created_by_nobody_is_created_by_nobody(monkeypatch):
    prs, conn = _service()
    monkeypatch.setattr(ProcessRoutingService, "_generate_workflow_id",
                        staticmethod(lambda: "wf-1"))

    prs.log_process("foo", {"a": 1})

    (sql, params), = [e for e in conn.executed if e[0].startswith("INSERT INTO proc.routing")]
    assert params[3] is None, f"created_by was {params[3]!r}"


def test_a_process_created_by_a_person_keeps_that_person(monkeypatch):
    prs, conn = _service()
    monkeypatch.setattr(ProcessRoutingService, "_generate_workflow_id",
                        staticmethod(lambda: "wf-1"))

    prs.log_process("foo", {"a": 1}, created_by="sub-alice")

    (sql, params), = [e for e in conn.executed if e[0].startswith("INSERT INTO proc.routing")]
    assert params[3] == "sub-alice"


def test_details_modified_by_nobody_are_modified_by_nobody():
    prs, conn = _service()

    prs.update_process_details(7, {"a": 1})

    updates = _routing_updates(conn)
    assert updates, conn.executed
    assert all(params[-2] is None for _, params in updates), updates


def test_a_status_set_by_nobody_is_set_by_nobody():
    prs, conn = _service()

    prs.update_process_status(7, "completed", process_details={"a": 1})

    updates = _routing_updates(conn)
    assert updates, conn.executed
    assert all(params[-2] is None for _, params in updates), updates


def test_a_run_triggered_by_nobody_is_triggered_by_nobody():
    prs, conn = _service()
    now = datetime.utcnow()

    prs.log_run_detail(7, "completed", process_details={"a": 1},
                       process_start_ts=now, process_end_ts=now)

    updates = _routing_updates(conn)
    assert updates, conn.executed
    for _, params in updates:
        assert params[-2] is None, f"modified_by was {params[-2]!r}"
        raw = json.loads(params[2])
        assert raw.get("triggered_by") is None, f"raw_data.triggered_by was {raw!r}"


def test_the_service_identity_is_never_read_as_a_value():
    """The guard. settings.script_user is the service's own name; any read of it
    in src is a stand-in waiting to be written somewhere a person belongs."""
    hits = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "script_user"
                    and isinstance(node.ctx, ast.Load)):
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not hits, "settings.script_user read as a value:\n  " + "\n  ".join(hits)


def test_the_guard_can_see_a_read():
    tree = ast.parse("x = created_by or self.settings.script_user")
    assert any(isinstance(n, ast.Attribute) and n.attr == "script_user"
               for n in ast.walk(tree))
