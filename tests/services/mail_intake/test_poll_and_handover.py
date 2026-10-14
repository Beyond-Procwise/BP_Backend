"""The two connections Plan 1 never made: relay -> pipeline, and READY -> upload queue."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.services import mail_handover
from src.services.mail_intake import poll
from src.services.mail_intake.sources.base import RawMessage


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 1
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append((" ".join(sql.split()), params))
        self._rows = self.conn.answer(sql, params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    def __init__(self, answers=None, autocommit=False):
        self.autocommit = autocommit
        self.answers = answers or {}
        self.log = []
        self.commits = 0
        self.rollbacks = 0

    def answer(self, sql, params):
        for needle, rows in self.answers.items():
            if needle in sql:
                return rows(params) if callable(rows) else rows
        return []

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def sql(self, needle):
        return [(s, p) for s, p in self.log if needle in s]


T0 = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)


def _msg(i, at):
    return RawMessage(raw=b"x", provider_message_id=f"emails/m{i}",
                      received_at=at, source_ref=f"s3://b/emails/m{i}")


class FakeSource:
    def __init__(self, messages):
        self.messages = messages
        self.since = "unset"

    def iter_messages(self, binding, since):
        self.since = since
        return iter([m for m in self.messages if m.received_at >= since])


class FakePipeline:
    def __init__(self, fail_on=()):
        self.seen = []
        self.fail_on = set(fail_on)

    def ingest(self, cur, binding, message):
        if message.provider_message_id in self.fail_on:
            raise RuntimeError("database went away")
        self.seen.append(message.provider_message_id)
        return SimpleNamespace(ready=("a",), quarantined=(), duplicates=())


BINDING = {"binding_id": 7, "owner_ref": "owner-1", "last_sync_at": T0}


# -- poll -----------------------------------------------------------------

def test_poll_refuses_an_autocommit_connection():
    with pytest.raises(RuntimeError, match="non-autocommit"):
        poll.poll_binding(FakeConn(autocommit=True), BINDING, FakeSource([]), FakePipeline())


def test_a_new_binding_starts_from_now_and_ingests_no_history():
    source = FakeSource([_msg(1, T0 - timedelta(days=300))])
    conn = FakeConn()
    counts = poll.poll_binding(conn, dict(BINDING, last_sync_at=None), source, FakePipeline())
    assert counts["messages"] == 0
    assert source.since == "unset"  # the relay was not even listed
    assert conn.sql("SET last_sync_at = %s WHERE binding_id = %s AND last_sync_at IS NULL")


def test_each_message_is_ingested_and_committed_and_the_cursor_follows_it():
    pipeline = FakePipeline()
    conn = FakeConn()
    source = FakeSource([_msg(1, T0 + timedelta(minutes=1)), _msg(2, T0 + timedelta(minutes=2))])
    counts = poll.poll_binding(conn, BINDING, source, pipeline)
    assert source.since == T0
    assert pipeline.seen == ["emails/m1", "emails/m2"]
    assert counts["messages"] == 2 and counts["ready"] == 2
    assert conn.commits == 2
    cursor_moves = [p[0] for _, p in conn.sql("SET last_sync_at = %s WHERE binding_id = %s AND last_sync_at < %s")]
    assert cursor_moves == [T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)]


def test_a_failing_message_is_rolled_back_and_the_rest_still_ingest():
    pipeline = FakePipeline(fail_on={"emails/m1"})
    conn = FakeConn()
    source = FakeSource([_msg(1, T0 + timedelta(minutes=1)), _msg(2, T0 + timedelta(minutes=2))])
    counts = poll.poll_binding(conn, BINDING, source, pipeline)
    assert counts["failed"] == 1 and counts["messages"] == 1
    assert pipeline.seen == ["emails/m2"]
    assert conn.rollbacks == 1


def test_poll_all_reads_only_active_relay_intake_bindings_and_owner_is_user_ref():
    row = {"binding_id": 7, "user_ref": "owner-1", "mailbox_address": "intake@x",
           "folder_scope": "emails/", "last_sync_at": T0}
    conn = FakeConn({"FROM proc.bp_mailbox_binding": [row]})
    seen = []
    poll.poll_all(conn, lambda b: seen.append(b) or FakeSource([]), FakePipeline())
    sql, params = conn.sql("FROM proc.bp_mailbox_binding")[0]
    assert "role = 'intake'" in sql and "is_active" in sql and params == ("relay",)
    assert seen[0]["owner_ref"] == "owner-1"


# -- hand-over ------------------------------------------------------------

class FakeStore:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.puts = []

    def get(self, key):
        return self.objects.get(key)

    def put_immutable(self, key, content):
        self.puts.append(key)
        self.objects[key] = content
        return key


QUOTE = SimpleNamespace(pipeline_doc_type="quote", aliases=("quotation", "quote"))
VOCAB = SimpleNamespace(document_types={"doctype.quote": QUOTE})


def _route(label, vocabulary=None):
    if label in ("quote", "quotation"):
        return ("quote", "doctype.quote")
    raise ValueError(label)


ART = {"artifact_id": "a1b2", "object_key": "mail-intake/artifacts/o/ab/cd/abcd",
       "detected_mime": "application/pdf", "declared_filename": "Quote QT/4471 (final).PDF"}


def _resolution(status, concept=None, candidates=()):
    return SimpleNamespace(status=status, evidence_concept=concept, candidates=candidates)


def _handover_conn():
    return FakeConn({
        "FROM proc.bp_ingest_artifact a": [ART],
        "INSERT INTO proc.process_monitor": [{"id": 9001}],
    })


def test_a_quote_is_copied_under_the_uploads_prefix_and_queued_like_an_upload():
    store = FakeStore({ART["object_key"]: b"%PDF-1.4 quote"})
    conn = _handover_conn()
    counts = mail_handover.hand_over(
        conn, store, read=lambda c, s, vocab: _resolution("matched", "doctype.quote"),
        vocab_loader=lambda: VOCAB, route=_route)
    assert counts == {"queued": 1, "held": 0, "failed": 0}
    # Built by us: detected-type suffix, sanitised stem, never the sender's raw name.
    assert store.puts == ["documents/quote/mail-intake/a1b2-Quote_QT_4471_final.pdf"]
    _, pm = conn.sql("INSERT INTO proc.process_monitor")[0]
    assert pm == (store.puts[0], "quote", "pdf", "mail-intake:a1b2")
    _, deriv = conn.sql("INSERT INTO proc.bp_artifact_derivation")[0]
    assert deriv == ("a1b2", mail_handover.PIPELINE_VERSION, "process_monitor:9001")


def test_a_document_whose_type_is_not_settled_is_held_not_guessed():
    store = FakeStore({ART["object_key"]: b"%PDF-1.4 ?"})
    conn = _handover_conn()
    counts = mail_handover.hand_over(
        conn, store,
        read=lambda c, s, vocab: _resolution("unresolved", None, ("doctype.quote", "doctype.invoice")),
        vocab_loader=lambda: VOCAB, route=_route)
    assert counts == {"queued": 0, "held": 1, "failed": 0}
    assert store.puts == [] and not conn.sql("INSERT INTO proc.process_monitor")
    _, held = conn.sql("INSERT INTO proc.bp_artifact_derivation")[0]
    assert held[0] == "a1b2" and "doctype.quote,doctype.invoice" in held[2]


def test_missing_stored_bytes_fail_and_record_nothing_so_the_next_pass_retries():
    conn = _handover_conn()
    counts = mail_handover.hand_over(
        conn, FakeStore({}), read=lambda *a, **k: pytest.fail("must not read"),
        vocab_loader=lambda: VOCAB, route=_route)
    assert counts["failed"] == 1 and conn.rollbacks == 1
    assert not conn.sql("INSERT INTO")


def test_hand_over_refuses_an_autocommit_connection():
    with pytest.raises(RuntimeError, match="non-autocommit"):
        mail_handover.hand_over(FakeConn(autocommit=True), FakeStore({}), vocab_loader=lambda: VOCAB)


def test_category_is_one_the_watcher_routes_back_to_the_same_type():
    assert mail_handover.category_for("doctype.quote", VOCAB, _route) == "quote"
    assert mail_handover.category_for("doctype.unknown", VOCAB, _route) is None


def test_settle_marks_the_artifact_complete_only_with_the_extraction_outcome():
    conn = FakeConn({"JOIN proc.process_monitor pm": [
        {"derivation_id": 3, "artifact_id": "a1b2", "status": "Extracted", "doc_action": None}]})
    assert mail_handover.settle(conn) == {"settled": 1}
    sql, params = conn.sql("JOIN proc.process_monitor pm")[0]
    assert params[2] == ["Completed", "Running", "Extracting"]
    assert conn.sql("SET completed_at = now(), outcome = %s")[0][1] == ("Extracted", 3)
    _, moved = conn.sql("UPDATE proc.bp_ingest_artifact")[0]
    assert moved[0] == "extraction_complete" and moved[4] == "ready_for_extraction"


@pytest.mark.parametrize("name", ["../../etc/passwd.pdf", "..\\..\\x.pdf", "/abs/path.pdf", "", None])
def test_a_sender_filename_cannot_escape_the_prefix(name):
    stem = mail_handover.safe_stem(name)
    assert "/" not in stem and "\\" not in stem and not stem.startswith(".")
    assert stem
