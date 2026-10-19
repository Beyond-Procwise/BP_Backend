"""The gateway audit trail in Postgres (proc.bp_gateway_call): what is written, that it is optional, and that it cannot break a call."""
import datetime as dt
import decimal
import json

import pytest

from src.services import ask_flags
from src.services.gateway import AuditSink, Backends, CallerContext, build_default_gateway
from src.services.gateway.audit import DbAuditWriter, default_audit_sink


class RecDB:
    def __init__(self, fail=False):
        self.executed, self.fail, self.commits = [], fail, 0
        self.autocommit = True

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        db = self

        class C:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def execute(s, sql, params=()):
                if db.fail:
                    raise RuntimeError("relation does not exist")
                db.executed.append((sql, params))
        return C()


ROW = {"call_id": "c1", "ts": "2026-10-19T09:00:00+0000", "caller_subject": "sub", "role": "Admin", "tenant_id": "default", "agent": "ask_records",
       "request_id": "r", "function": "get_record", "function_version": 1, "params": {"id": "PO1", "when": dt.date(2026, 1, 2), "amt": decimal.Decimal("1.5")},
       "result_record_ids": [f"id{i}" for i in range(500)], "row_count": 1, "truncated": False, "ok": True, "error_code": None,
       "latency_ms": 12, "scope_violation": False}


def test_a_row_is_written_to_the_call_table_with_every_audit_field():
    db = RecDB()
    DbAuditWriter(db)(ROW)
    sql, params = db.executed[0]
    assert "insert into proc.bp_gateway_call" in sql.lower() and "on conflict (call_id) do nothing" in sql.lower()
    for col in ("call_id", "caller_subject", "role", "tenant_id", "agent", "request_id", "function", "function_version", "params",
                "result_record_ids", "row_count", "truncated", "ok", "error_code", "latency_ms", "scope_violation"):
        assert col in sql, col
    assert "c1" in params and "sub" in params


def test_params_with_dates_and_decimals_are_stored_as_json_and_the_id_list_is_capped():
    db = RecDB()
    DbAuditWriter(db)(ROW)
    params = db.executed[0][1]
    stored = [p for p in params if isinstance(p, str) and p.startswith("{")][0]
    assert json.loads(stored)["when"] == "2026-01-02" and json.loads(stored)["amt"] == "1.5"
    ids = json.loads([p for p in params if isinstance(p, str) and p.startswith("[")][0])
    assert len(ids) == 200


def test_a_failed_insert_never_breaks_the_call_and_the_row_stays_in_memory():
    sink = AuditSink(db_writer=DbAuditWriter(RecDB(fail=True)))
    sink.record(ROW)
    assert sink.rows[-1]["call_id"] == "c1"


def test_a_gateway_call_is_written_to_the_database_when_a_writer_is_attached():
    db = RecDB()
    g = build_default_gateway(Backends(connect=None), audit=AuditSink(db_writer=DbAuditWriter(db)))
    g.call(CallerContext("s", "Admin", None, agent="x"), "get_inventory_position", {"sku": "A"})
    assert len(db.executed) == 1 and "UNSUPPORTED" in db.executed[0][1]


def test_the_database_trail_is_off_unless_the_flag_is_on(monkeypatch):
    monkeypatch.delenv(ask_flags.GATEWAY_AUDIT, raising=False)
    assert default_audit_sink()._db_writer is None
    monkeypatch.setenv(ask_flags.GATEWAY_AUDIT, "1")
    assert default_audit_sink()._db_writer is not None


def test_the_flag_is_opt_in_even_in_development(monkeypatch):
    monkeypatch.delenv(ask_flags.GATEWAY_AUDIT, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert not ask_flags.enabled(ask_flags.GATEWAY_AUDIT)


def test_the_default_gateway_uses_the_default_sink(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY_AUDIT, "1")
    assert build_default_gateway().audit._db_writer is not None


def test_the_migration_makes_both_tables_append_only_and_has_a_rollback():
    sql = open("deploy/sql/2026-10-19_bp_gateway_audit.sql").read().lower()
    for t in ("proc.bp_gateway_call", "proc.bp_gateway_write"):
        assert f"create table if not exists {t}" in sql
        assert f"before update or delete on {t}" in sql and f"before truncate on {t}" in sql
    assert "unique" in sql and "idempotency_key" in sql
    rb = open("deploy/sql/2026-10-19_bp_gateway_audit_rollback.sql").read().lower()
    assert "drop table if exists proc.bp_gateway_call" in rb and "drop table if exists proc.bp_gateway_write" in rb
