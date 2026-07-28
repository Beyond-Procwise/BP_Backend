"""A user's attachment must reach the MIME path, and never the extraction pipeline.

The SMTP layer has always supported attachments (EmailService.send_email builds a
MIMEBase part per file). What was missing was any way to get one in: the endpoint
had no field and the draft table had no column.

Regressions guarded here specifically:

1. draft_rfq_emails_repo.load_by_unique_id used to select a fixed column list
   that did not include the new ``attachments`` column, so any caller reading
   ``draft.get("attachments")`` in order to append to it always saw ``None`` --
   every upload would silently REPLACE the previous one instead of adding to
   it. The repo-layer guard for that specific regression is
   test_load_by_unique_id_carries_attachments_column, which drives the real
   ``load_by_unique_id`` against a fake cursor and fails against the old
   (buggy) column list. test_two_successive_uploads_accumulate_not_replace
   monkeypatches ``load_by_unique_id`` out entirely (it stands in an
   in-memory-dict-backed loader) and instead proves the OTHER half of the same
   contract: that ``add_email_attachments`` itself reads whatever the loader
   hands back and appends to it rather than overwriting. Both halves are
   necessary and neither alone is sufficient -- a correct repo with a handler
   that discarded prior attachments would pass the repo test and fail this
   one; a buggy repo behind a correct handler would fail the repo test but
   this test, using its own fake loader, would not catch it.

2. The S3 client must be a bare ``boto3.client("s3")`` reading ``settings.s3_bucket_name``
   -- there is no general ``aws_region`` setting in this project.

3. A file whose display name collides with one already stored, or with
   another file in the same request, must not have its bytes silently
   clobbered in S3 while the earlier record keeps pointing at the same key.
   test_duplicate_filename_against_stored_attachment_gets_unique_key and
   test_two_identically_named_files_in_one_request_get_unique_keys drive the
   real handler and assert on the keys actually written to the fake S3 store,
   not merely on the HTTP-level response shape.
"""
import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import Dict

import pytest

from services.email_dispatch_service import EmailDispatchService

mod = importlib.import_module("src.api.routers.workflows")
draft_repo_mod = importlib.import_module("repositories.draft_rfq_emails_repo")


# ---------------------------------------------------------------------------
# EmailDispatchService._load_attachments -- dispatch-side loading
# ---------------------------------------------------------------------------


def test_dispatch_loads_stored_attachments_as_bytes_and_filename(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    draft = {"attachments": [
        {"filename": "terms.pdf", "s3_key": "email-attachments/wf-1/terms.pdf", "bytes": 4},
    ]}
    monkeypatch.setattr(svc, "_read_s3_bytes", lambda key: b"PDF!", raising=False)
    assert svc._load_attachments(draft) == [(b"PDF!", "terms.pdf")]


def test_an_unreadable_attachment_is_skipped_not_faked(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    draft = {"attachments": [
        {"filename": "gone.pdf", "s3_key": "email-attachments/wf-1/gone.pdf", "bytes": 9},
        {"filename": "ok.pdf", "s3_key": "email-attachments/wf-1/ok.pdf", "bytes": 2},
    ]}

    def read(key):
        if key.endswith("gone.pdf"):
            raise RuntimeError("no such key")
        return b"OK"

    monkeypatch.setattr(svc, "_read_s3_bytes", read, raising=False)
    # An empty part named terms.pdf would be a lie about what was sent.
    assert svc._load_attachments(draft) == [(b"OK", "ok.pdf")]


def test_no_attachments_is_empty_list_not_none(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    assert svc._load_attachments({}) == []


def test_a_json_string_column_is_tolerated(monkeypatch):
    svc = EmailDispatchService.__new__(EmailDispatchService)
    draft = {"attachments": json.dumps([{"filename": "a.txt", "s3_key": "k", "bytes": 1}])}
    monkeypatch.setattr(svc, "_read_s3_bytes", lambda key: b"A", raising=False)
    assert svc._load_attachments(draft) == [(b"A", "a.txt")]


# ---------------------------------------------------------------------------
# EmailDispatchService._read_s3_bytes / _write_s3_bytes -- S3 wiring
# ---------------------------------------------------------------------------


def test_read_and_write_s3_bytes_use_bare_client_and_configured_bucket(monkeypatch):
    """This project has no general aws_region setting -- the house pattern
    (src/services/extraction/parser.py:59) is a bare boto3.client("s3") that
    takes its region from the environment, and the bucket is settings.s3_bucket_name."""
    calls = {}

    class _FakeBody:
        def read(self):
            return b"PDF-BYTES"

    class _FakeS3Client:
        def get_object(self, Bucket, Key):
            calls["get"] = (Bucket, Key)
            return {"Body": _FakeBody()}

        def put_object(self, Bucket, Key, Body, ContentType):
            calls["put"] = (Bucket, Key, Body, ContentType)

    def _fake_client(service_name, *args, **kwargs):
        calls["client_args"] = (service_name, args, kwargs)
        return _FakeS3Client()

    import boto3

    monkeypatch.setattr(boto3, "client", _fake_client)

    from config.settings import settings

    monkeypatch.setattr(settings, "s3_bucket_name", "test-bucket")

    svc = EmailDispatchService.__new__(EmailDispatchService)
    assert svc._read_s3_bytes("email-attachments/wf-1/terms.pdf") == b"PDF-BYTES"
    assert calls["get"] == ("test-bucket", "email-attachments/wf-1/terms.pdf")
    assert calls["client_args"] == ("s3", (), {})

    svc._write_s3_bytes("email-attachments/wf-1/terms.pdf", b"HELLO", "application/pdf")
    assert calls["put"] == (
        "test-bucket",
        "email-attachments/wf-1/terms.pdf",
        b"HELLO",
        "application/pdf",
    )


# ---------------------------------------------------------------------------
# draft_rfq_emails_repo.load_by_unique_id -- the data-loss trap
# ---------------------------------------------------------------------------


def test_load_by_unique_id_carries_attachments_column(monkeypatch):
    """Regression: the SELECT list previously omitted the attachments column
    entirely, so draft.get("attachments") always evaluated to None regardless
    of what was actually stored."""

    class _Cursor:
        def __init__(self, row):
            self._row = row

        def execute(self, sql, params=None):
            self.sql = " ".join(sql.split())
            self.params = params

        def fetchone(self):
            return self._row

        def close(self):
            pass

    class _Conn:
        def __init__(self, row):
            self._row = row
            self._cursor = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            self._cursor = _Cursor(self._row)
            return self._cursor

    stored = [{"filename": "a.pdf", "s3_key": "email-attachments/x/a.pdf", "bytes": 3}]
    row = (
        1, "RFQ-1", "SUP-1", "Acme", "Subject", "Body", "2026-07-28", True,
        "buyer@x.com", {"recipients": []}, "WF-1", "RUN-1", "wf-attach-2",
        "mbox@x.com", stored,
    )

    conn = _Conn(row)
    monkeypatch.setattr(draft_repo_mod, "get_conn", lambda: conn)

    result = draft_repo_mod.load_by_unique_id("wf-attach-2")
    assert result is not None
    assert result["attachments"] == stored
    # column count sanity: 15 columns selected including attachments last
    assert conn._cursor.sql.count(",") + 1 >= 15 or "attachments" in conn._cursor.sql
    assert "attachments" in conn._cursor.sql


def test_load_by_unique_id_returns_none_attachments_when_column_is_null(monkeypatch):
    class _Cursor:
        def __init__(self, row):
            self._row = row

        def execute(self, sql, params=None):
            pass

        def fetchone(self):
            return self._row

        def close(self):
            pass

    class _Conn:
        def __init__(self, row):
            self._row = row

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return _Cursor(self._row)

    row = (
        2, "RFQ-2", "SUP-2", "Acme", "Subject", "Body", "2026-07-28", False,
        "buyer@x.com", {}, "WF-2", "RUN-2", "wf-attach-3", None, None,
    )
    monkeypatch.setattr(draft_repo_mod, "get_conn", lambda: _Conn(row))

    result = draft_repo_mod.load_by_unique_id("wf-attach-3")
    assert result["attachments"] is None


# ---------------------------------------------------------------------------
# Endpoint-level: upload accumulation, rejection reporting, delete
# ---------------------------------------------------------------------------


class _FakeUpload:
    """Duck-types fastapi.UploadFile well enough for the handler under test."""

    def __init__(self, filename, content_type, data):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self):
        return self._data


class _FakeDispatchService:
    """Stands in for EmailDispatchService inside the attachments endpoint so
    the test never constructs a real SES-backed EmailService or touches boto3."""

    _ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
    _ATTACHMENT_MAX_TOTAL_BYTES = 25 * 1024 * 1024

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick

    def _write_s3_bytes(self, s3_key, data, content_type):
        _FAKE_S3[s3_key] = data


_FAKE_S3: Dict = {}


class _FakeCursor:
    def __init__(self, calls, store):
        self._calls = calls
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        normalised = " ".join(sql.split())
        self._calls.append((normalised, params))
        if normalised.startswith("UPDATE proc.draft_rfq_emails SET attachments"):
            records_json, unique_id = params
            self._store[unique_id] = json.loads(records_json)

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, calls, store):
        self.calls = calls
        self.store = store
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor(self.calls, self.store)

    def commit(self):
        self.committed = True


def _make_agent_nick(conn):
    return SimpleNamespace(get_db_connection=lambda: conn)


def _store_backed_loader(store):
    def _loader(unique_id):
        return {"unique_id": unique_id, "attachments": store.get(unique_id)}

    return _loader


def test_two_successive_uploads_accumulate_not_replace(monkeypatch):
    """Proves the HANDLER's own load-then-append logic: given whatever the
    draft loader hands back, add_email_attachments must append to it, never
    replace it. The loader here is an in-memory-dict stand-in
    (_store_backed_loader), not the real draft_rfq_emails_repo.load_by_unique_id
    -- so this test would still pass even if that repo function regressed back
    to omitting the attachments column. That specific regression is guarded
    separately, by test_load_by_unique_id_carries_attachments_column, which
    drives the real function against a fake cursor. The two tests compose:
    together they cover "the repo reads the column back correctly" and "the
    handler appends rather than overwrites what the repo hands it."
    """
    _FAKE_S3.clear()
    store: Dict = {}
    calls = []
    conn = _FakeConn(calls, store)
    agent_nick = _make_agent_nick(conn)

    monkeypatch.setattr(mod, "EmailDispatchService", _FakeDispatchService)
    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    first = asyncio.run(
        mod.add_email_attachments(
            "wf-attach-1",
            files=[_FakeUpload("terms.pdf", "application/pdf", b"first file bytes")],
            user_id="alice",
            agent_nick=agent_nick,
        )
    )
    assert [a["filename"] for a in first["attachments"]] == ["terms.pdf"]
    assert first["rejected"] == []

    second = asyncio.run(
        mod.add_email_attachments(
            "wf-attach-1",
            files=[_FakeUpload("invoice.pdf", "application/pdf", b"second file bytes")],
            user_id="alice",
            agent_nick=agent_nick,
        )
    )
    assert [a["filename"] for a in second["attachments"]] == ["terms.pdf", "invoice.pdf"]
    assert second["rejected"] == []
    assert store["wf-attach-1"] == second["attachments"]
    assert _FAKE_S3["email-attachments/wf-attach-1/terms.pdf"] == b"first file bytes"
    assert _FAKE_S3["email-attachments/wf-attach-1/invoice.pdf"] == b"second file bytes"


def test_duplicate_filename_against_stored_attachment_gets_unique_key(monkeypatch):
    """Re-uploading "terms.pdf" after it is already stored must not clobber the
    first object's bytes in S3 while the first record keeps pointing at that
    same key. Both records keep the human-facing filename "terms.pdf"; the
    STORAGE key for the second one must differ, and both keys' bytes in the
    fake S3 store must match what was actually sent for each."""
    _FAKE_S3.clear()
    store: Dict = {
        "wf-attach-8": [
            {
                "filename": "terms.pdf",
                "content_type": "application/pdf",
                "bytes": len(b"original bytes"),
                "s3_key": "email-attachments/wf-attach-8/terms.pdf",
                "added_by": "alice",
                "added_at": "2026-07-28T00:00:00+00:00",
            }
        ]
    }
    _FAKE_S3["email-attachments/wf-attach-8/terms.pdf"] = b"original bytes"
    calls = []
    conn = _FakeConn(calls, store)
    agent_nick = _make_agent_nick(conn)

    monkeypatch.setattr(mod, "EmailDispatchService", _FakeDispatchService)
    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    result = asyncio.run(
        mod.add_email_attachments(
            "wf-attach-8",
            files=[_FakeUpload("terms.pdf", "application/pdf", b"corrected bytes")],
            user_id="alice",
            agent_nick=agent_nick,
        )
    )

    assert result["rejected"] == []
    filenames = [a["filename"] for a in result["attachments"]]
    assert filenames == ["terms.pdf", "terms.pdf"]

    keys = [a["s3_key"] for a in result["attachments"]]
    assert len(set(keys)) == 2, "the two records must not share a storage key"
    assert keys[0] == "email-attachments/wf-attach-8/terms.pdf"
    assert keys[1] != keys[0]
    assert keys[1].startswith("email-attachments/wf-attach-8/terms__")
    assert keys[1].endswith(".pdf")

    # Neither object's bytes were disturbed by the other.
    assert _FAKE_S3[keys[0]] == b"original bytes"
    assert _FAKE_S3[keys[1]] == b"corrected bytes"


def test_two_identically_named_files_in_one_request_get_unique_keys(monkeypatch):
    """Two files named "dup.pdf" uploaded in the SAME request must not collide
    either -- the second write must not land on the first file's key."""
    _FAKE_S3.clear()
    store: Dict = {}
    calls = []
    conn = _FakeConn(calls, store)
    agent_nick = _make_agent_nick(conn)

    monkeypatch.setattr(mod, "EmailDispatchService", _FakeDispatchService)
    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    result = asyncio.run(
        mod.add_email_attachments(
            "wf-attach-9",
            files=[
                _FakeUpload("dup.pdf", "application/pdf", b"first copy"),
                _FakeUpload("dup.pdf", "application/pdf", b"second copy"),
            ],
            user_id="alice",
            agent_nick=agent_nick,
        )
    )

    assert result["rejected"] == []
    filenames = [a["filename"] for a in result["attachments"]]
    assert filenames == ["dup.pdf", "dup.pdf"]

    keys = [a["s3_key"] for a in result["attachments"]]
    assert len(set(keys)) == 2, "identically-named files in one request must not share a storage key"
    assert keys[0] == "email-attachments/wf-attach-9/dup.pdf"
    assert keys[1] != keys[0]

    assert _FAKE_S3[keys[0]] == b"first copy"
    assert _FAKE_S3[keys[1]] == b"second copy"


def test_disallowed_type_and_oversize_are_reported_not_dropped(monkeypatch):
    """A rejected file must be reported to the caller, never silently dropped."""
    _FAKE_S3.clear()
    store: Dict = {}
    calls = []
    conn = _FakeConn(calls, store)
    agent_nick = _make_agent_nick(conn)

    monkeypatch.setattr(mod, "EmailDispatchService", _FakeDispatchService)
    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    oversized = b"x" * (_FakeDispatchService._ATTACHMENT_MAX_BYTES + 1)

    result = asyncio.run(
        mod.add_email_attachments(
            "wf-attach-4",
            files=[
                _FakeUpload("malware.exe", "application/octet-stream", b"nope"),
                _FakeUpload("huge.pdf", "application/pdf", oversized),
                _FakeUpload("ok.pdf", "application/pdf", b"fine"),
            ],
            user_id="alice",
            agent_nick=agent_nick,
        )
    )

    assert [a["filename"] for a in result["attachments"]] == ["ok.pdf"]
    rejected_names = {r["filename"] for r in result["rejected"]}
    assert rejected_names == {"malware.exe", "huge.pdf"}
    assert len(result["rejected"]) == 2


def test_no_attachment_is_sent_as_an_empty_part_when_upload_fails(monkeypatch):
    """If the S3 write fails, the file must be rejected, never recorded as if
    it had been stored -- a record with no bytes behind it is a false claim."""
    _FAKE_S3.clear()
    store: Dict = {}
    calls = []
    conn = _FakeConn(calls, store)
    agent_nick = _make_agent_nick(conn)

    class _FailingDispatchService(_FakeDispatchService):
        def _write_s3_bytes(self, s3_key, data, content_type):
            raise RuntimeError("S3 unavailable")

    monkeypatch.setattr(mod, "EmailDispatchService", _FailingDispatchService)
    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    result = asyncio.run(
        mod.add_email_attachments(
            "wf-attach-5",
            files=[_FakeUpload("terms.pdf", "application/pdf", b"bytes")],
            user_id="alice",
            agent_nick=agent_nick,
        )
    )
    assert result["attachments"] == []
    assert len(result["rejected"]) == 1
    assert "terms.pdf" == result["rejected"][0]["filename"]


def test_delete_removes_one_attachment_by_index(monkeypatch):
    store = {
        "wf-attach-6": [
            {"filename": "a.pdf", "s3_key": "email-attachments/wf-attach-6/a.pdf", "bytes": 1},
            {"filename": "b.pdf", "s3_key": "email-attachments/wf-attach-6/b.pdf", "bytes": 1},
        ]
    }
    calls = []
    conn = _FakeConn(calls, store)
    agent_nick = _make_agent_nick(conn)

    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    result = mod.remove_email_attachment("wf-attach-6", 0, agent_nick=agent_nick)
    assert [a["filename"] for a in result["attachments"]] == ["b.pdf"]
    assert store["wf-attach-6"] == result["attachments"]


def test_delete_out_of_range_index_404s(monkeypatch):
    from fastapi import HTTPException

    store = {"wf-attach-7": [{"filename": "a.pdf", "s3_key": "k", "bytes": 1}]}
    conn = _FakeConn([], store)
    agent_nick = _make_agent_nick(conn)

    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", _store_backed_loader(store))

    with pytest.raises(HTTPException) as exc_info:
        mod.remove_email_attachment("wf-attach-7", 5, agent_nick=agent_nick)
    assert exc_info.value.status_code == 404


def test_upload_to_unknown_draft_404s(monkeypatch):
    from fastapi import HTTPException

    conn = _FakeConn([], {})
    agent_nick = _make_agent_nick(conn)
    monkeypatch.setattr(mod.draft_rfq_emails_repo, "load_by_unique_id", lambda unique_id: None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            mod.add_email_attachments(
                "does-not-exist",
                files=[_FakeUpload("a.pdf", "application/pdf", b"x")],
                user_id="alice",
                agent_nick=agent_nick,
            )
        )
    assert exc_info.value.status_code == 404
