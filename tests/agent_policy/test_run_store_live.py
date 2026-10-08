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
