"""Handing a finding to a person, against bp_testdb (PROCWISE_TEST_LIVE_DB=1)."""
import os
import uuid
from datetime import datetime, timezone

import pytest

from services.people import work as w
from services.people.admin import PeopleError

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
NOW = datetime(2026, 10, 22, 19, 30, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn):
    tag = uuid.uuid4().hex[:8]
    me, other = f"tst-me-{tag}", f"tst-other-{tag}"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_person (tenant_id, subject, display_name, is_active, roles, last_modified_by) "
                    "VALUES ('default', %s, %s, true, '{}', 'test'), ('default', %s, %s, true, '{}', 'test')",
                    (me, f"Me {tag}", other, f"Other {tag}"))
        cur.execute("INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, field_name, issue_type, severity, status) "
                    "VALUES ('invoice', %s, 'invoice_number', 'duplicate_invoice', 'warning', 'open') RETURNING discrepancy_id",
                    (f"tst/{tag}.pdf",))
        fid = cur.fetchone()[0]
    yield {"me": me, "other": other, "fid": str(fid), "tag": tag}
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_notification WHERE event_id LIKE %s", (f"{w.EVENT_TYPE}:finding:{fid}:%",))
        cur.execute("DELETE FROM proc.bp_domain_event WHERE event_id LIKE %s", (f"{w.EVENT_TYPE}:finding:{fid}:%",))
        cur.execute("DELETE FROM proc.bp_work_assignment WHERE entity_type='finding' AND entity_id=%s", (str(fid),))
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (fid,))
        cur.execute("DELETE FROM proc.bp_person WHERE tenant_id='default' AND subject = ANY(%s)", ([me, other],))


@live
def test_hand_on_claim_release_and_who_is_told(conn, world):
    me, other, fid = world["me"], world["other"], world["fid"]
    with pytest.raises(PeopleError) as e:
        w.assign(conn, me, "finding", "999999999", other)
    assert e.value.status == 404
    with pytest.raises(PeopleError) as e:
        w.assign(conn, me, "finding", fid, "nobody-" + world["tag"])
    assert e.value.status == 422 and "not an active" in e.value.detail

    a = w.assign(conn, me, "finding", fid, other, note="Yours, I think.", now=NOW)
    assert a["assignee_subject"] == other and a["assignee_name"] == f"Other {world['tag']}" and a["by_name"] == f"Me {world['tag']}"
    assert a["note"] == "Yours, I think." and a["released_at"] is None
    with conn.cursor() as cur:
        cur.execute("SELECT summary, data FROM proc.bp_domain_event WHERE event_id = %s",
                    (f"{w.EVENT_TYPE}:finding:{fid}:{a['assignment_id']}",))
        summary, data = cur.fetchone()
    assert summary == f"Me {world['tag']} handed you a finding: duplicate invoice on invoice {world['tag']}.pdf"
    assert data["explicit"] == [other] and data["link"] == f"/actions/queue?finding={fid}"
    assert data["body"] == f"Yours, I think.\nAbout finding {fid}: duplicate invoice on invoice {world['tag']}.pdf"

    rows = w.active(conn, "finding", [fid, "0"])
    assert [r["assignee_subject"] for r in rows] == [other]
    assert [r["entity_id"] for r in w.mine(conn, other)] == [fid]

    # claiming it yourself releases the other hand and tells nobody
    b = w.assign(conn, other, "finding", fid, other, now=NOW)
    assert b["assignment_id"] != a["assignment_id"]
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_work_assignment WHERE entity_type='finding' AND entity_id=%s AND released_at IS NULL", (fid,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM proc.bp_domain_event WHERE event_id LIKE %s", (f"{w.EVENT_TYPE}:finding:{fid}:%",))
        assert cur.fetchone()[0] == 1

    assert w.release(conn, me, "finding", fid) is True
    assert w.release(conn, me, "finding", fid) is False
    assert w.active(conn, "finding", [fid]) == [] and w.mine(conn, other) == []
