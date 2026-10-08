"""Run store against bp_testdb. Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb."""
import os
import threading

import pytest

from services.agent_policy import run_store as rs

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "run-store live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


def _new(conn):
    return rs.create(conn, kind="extract", request={"documents": []}, actor="test")


def test_create_claim_finish_roundtrip(conn):
    run = _new(conn)
    assert run["status"] == "queued" and run["owner"] == rs.OWNER
    assert rs.claim(conn, run["run_id"], rs.OWNER) is True
    assert rs.claim(conn, run["run_id"], rs.OWNER) is False
    rs.finish(conn, run["run_id"], "done", counts={"policies": 2})
    got = rs.get(conn, run["run_id"])
    assert got["status"] == "done" and got["counts"] == {"policies": 2} and got["finished_at"]


def test_stale_run_heals_to_failed(conn):
    run = _new(conn)
    rs.claim(conn, run["run_id"], rs.OWNER)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_policy_extraction_run "
                    "SET heartbeat_at = now() - interval '10 minutes' WHERE run_id = %s",
                    (run["run_id"],))
    got = rs.get(conn, run["run_id"])
    assert got["status"] == "failed" and got["error"] == rs.HEALED_ERROR
    assert got["error"] == ("The server restarted while this run was working. "
                            "Start it again; the policies already listed were saved.")


def test_fresh_heartbeat_is_not_healed(conn):
    run = _new(conn)
    rs.claim(conn, run["run_id"], rs.OWNER)
    rs.beat(conn, run["run_id"], rs.OWNER)
    assert rs.get(conn, run["run_id"])["status"] == "running"


def test_append_waits_for_the_run_row_lock_then_seqs_are_consecutive():
    """Deterministic: hold the run-row lock in one transaction; a second append must block."""
    import time

    from services.db import get_conn
    with get_conn() as holder:
        run_id = _new(holder)["run_id"]
        holder.autocommit = False
        with holder.cursor() as cur:
            cur.execute("SELECT run_id FROM proc.bp_policy_extraction_run "
                        "WHERE run_id = %s FOR UPDATE", (run_id,))
        got, errors = [], []

        def second():
            try:
                with get_conn() as c:
                    got.append(rs.append_item(c, run_id, kind="note", payload={"n": 2}))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=second)
        t.start()
        t.join(1.0)
        assert t.is_alive(), "append_item did not wait for the run-row lock"
        with holder.cursor() as cur:   # the lock holder appends item 1 in its own transaction
            cur.execute("INSERT INTO proc.bp_policy_extraction_item (run_id, seq, kind, payload) "
                        "VALUES (%s, 1, 'note', '{}'::jsonb)", (run_id,))
        holder.commit()
        t.join(10)
        holder.autocommit = True
        assert not t.is_alive() and not errors
        assert got == [2]


def test_concurrent_appends_get_distinct_consecutive_seqs():
    from services.db import get_conn
    with get_conn() as c:
        run_id = _new(c)["run_id"]
    seqs, errors = [], []

    def worker():
        try:
            with get_conn() as c:
                for _ in range(10):
                    seqs.append(rs.append_item(c, run_id, kind="note", payload={"n": 1}))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sorted(seqs) == list(range(1, 21))


def _age(conn, run_id, col, secs, owner=None):
    with conn.cursor() as cur:
        cur.execute(f"UPDATE proc.bp_policy_extraction_run SET {col} = now() - make_interval(secs => %s)"
                    + (", owner = %s" if owner else "") + " WHERE run_id = %s",
                    (secs, *( (owner,) if owner else ()), run_id))


def test_heartbeat_boundary_60s_stays_running_130s_heals(conn):
    a, b = _new(conn)["run_id"], _new(conn)["run_id"]
    for r in (a, b):
        rs.claim(conn, r, rs.OWNER)
    _age(conn, a, "heartbeat_at", 60)
    _age(conn, b, "heartbeat_at", 130)
    assert rs.get(conn, a)["status"] == "running"
    got = rs.get(conn, b)
    assert got["status"] == "failed" and got["error"] == rs.HEALED_ERROR


def test_finish_after_heal_returns_false_and_keeps_failed(conn):
    run_id = _new(conn)["run_id"]
    rs.claim(conn, run_id, rs.OWNER)
    _age(conn, run_id, "heartbeat_at", 600)
    rs.get(conn, run_id)  # heals
    assert rs.finish(conn, run_id, "done", counts={"x": 1}) is False
    assert rs.get(conn, run_id)["status"] == "failed"
    other = _new(conn)["run_id"]
    assert rs.finish(conn, other, "done", counts={}) is True


def test_orphaned_queued_run_heals_but_own_and_young_ones_do_not(conn):
    orphan = _new(conn)["run_id"]
    _age(conn, orphan, "created_at", 600, owner="proc-dead")
    nobody = _new(conn)["run_id"]
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_policy_extraction_run SET owner = NULL, "
                    "created_at = now() - interval '600 seconds' WHERE run_id = %s", (nobody,))
    young_foreign = _new(conn)["run_id"]
    _age(conn, young_foreign, "created_at", 10, owner="proc-dead")
    own_old = _new(conn)["run_id"]
    _age(conn, own_old, "created_at", 600)
    assert rs.get(conn, orphan)["error"] == rs.NOT_STARTED_ERROR
    assert rs.get(conn, orphan)["status"] == "failed"
    assert rs.NOT_STARTED_ERROR == "The server restarted before this run started. Start it again."
    assert rs.get(conn, nobody)["status"] == "failed"
    assert rs.get(conn, young_foreign)["status"] == "queued"
    assert rs.get(conn, own_old)["status"] == "queued"


def test_heal_all_via_list_recent(conn):
    stale = _new(conn)["run_id"]
    rs.claim(conn, stale, rs.OWNER)
    _age(conn, stale, "heartbeat_at", 600)
    orphan = _new(conn)["run_id"]
    _age(conn, orphan, "created_at", 600, owner="proc-dead")
    fresh = _new(conn)["run_id"]
    rs.claim(conn, fresh, rs.OWNER)
    rows = {r["run_id"]: r for r in rs.list_recent(conn, 200)}
    assert rows[stale]["status"] == "failed" and rows[orphan]["status"] == "failed"
    assert rows[fresh]["status"] == "running"


def test_append_error_is_not_masked_by_a_broken_rollback():
    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): raise ValueError("original")

    class Broken:
        autocommit = True
        def cursor(self): return Cur()
        def rollback(self): raise RuntimeError("connection gone")

    with pytest.raises(ValueError, match="original"):
        rs.append_item(Broken(), 1, kind="note", payload={})


def test_append_restores_autocommit_and_get_after_seq(conn):
    run_id = _new(conn)["run_id"]
    before = conn.autocommit
    for i in range(3):
        rs.append_item(conn, run_id, kind="policy", payload={"i": i}, reference=f"c{i}",
                       policy_key="GEN-1", decision="new", document_id=None)
    assert conn.autocommit == before
    got = rs.get(conn, run_id, after_seq=1)
    assert [i["seq"] for i in got["items"]] == [2, 3]
    assert got["items"][0]["payload"] == {"i": 1}
    assert [i["seq"] for i in rs.get(conn, run_id)["items"]] == [1, 2, 3]


def test_list_recent_includes_new_run(conn):
    run_id = _new(conn)["run_id"]
    assert run_id in [r["run_id"] for r in rs.list_recent(conn, 20)]
