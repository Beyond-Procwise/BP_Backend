"""Policy documents with S3 and the DB faked. Nothing here reaches AWS or Postgres."""
import hashlib
import io
import uuid
from types import SimpleNamespace

import pytest

from services.agent_policy import documents as d


# ---------------------------------------------------------------- fakes

class _NoSuchKey(Exception):
    response = {"Error": {"Code": "404"}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.head_sizes = {}  # lets a test make head_object disagree with the body

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"ContentLength": self.head_sizes.get(Key, len(self.objects[Key]))}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


class FakeDB:
    """Just enough of the two tables, keyed on the statements documents.py sends."""

    def __init__(self):
        self.docs = {}       # id -> dict
        self.versions = {}   # (id, version) -> dict
        self.autocommit = True
        self.commits = self.rollbacks = 0
        self._staged = None
        self.log = []
        self.extracted = set()  # (document_id, version) an extraction run item references

    # connection API
    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1
        self._staged = None

    def rollback(self):
        self.rollbacks += 1
        if self._staged is not None:
            self.docs, self.versions = self._staged
            self._staged = None

    def _snapshot(self):
        if not self.autocommit and self._staged is None:
            import copy
            self._staged = (copy.deepcopy(self.docs), copy.deepcopy(self.versions))


class FakeCursor:
    def __init__(self, db):
        self.db, self._rows = db, []

    def execute(self, sql, params=()):
        db, s = self.db, " ".join(sql.split())
        db.log.append(s)
        db._snapshot()
        if s.startswith("SELECT document_id, title, latest_version FROM proc.bp_policy_document WHERE document_id"):
            doc = db.docs.get(params[0])
            self._rows = [(doc["id"], doc["title"], doc["latest"])] if doc else []
        elif s.startswith("SELECT document_id, title, latest_version FROM proc.bp_policy_document WHERE match_name"):
            hits = sorted((k, v) for k, v in db.docs.items() if v["match"] == params[0])
            self._rows = [(v["id"], v["title"], v["latest"]) for _, v in hits[:1]]
        elif s.startswith("SELECT version FROM proc.bp_policy_document_version WHERE document_id = %s AND content_hash"):
            self._rows = [(k[1],) for k, v in db.versions.items() if k[0] == params[0] and v["hash"] == params[1]]
        elif s.startswith("SELECT document_id, version FROM proc.bp_policy_document_version WHERE s3_key"):
            self._rows = sorted((k[0], k[1]) for k, v in db.versions.items() if v["key"] == params[0])[:1]
        elif s.startswith("SELECT latest_version FROM proc.bp_policy_document WHERE document_id"):
            self._rows = [(db.docs[params[0]]["latest"],)]
        elif s.startswith("INSERT INTO proc.bp_policy_document ("):
            new_id = max(db.docs, default=0) + 1
            db.docs[new_id] = {"id": new_id, "title": params[0], "match": params[1], "latest": 1, "by": params[2]}
            self._rows = [(new_id,)]
        elif s.startswith("INSERT INTO proc.bp_policy_document_version"):
            doc_id, version, filename, key, size, h, actor = params
            assert (doc_id, version) not in db.versions
            assert not any(k[0] == doc_id and v["hash"] == h for k, v in db.versions.items())
            db.versions[(doc_id, version)] = {"filename": filename, "key": key, "size": size, "hash": h,
                                              "by": actor, "text": None, "parsed_at": None}
        elif s.startswith("UPDATE proc.bp_policy_document SET latest_version"):
            db.docs[params[1]]["latest"] = params[0]
        elif s.startswith("SELECT filename, s3_key, parsed_text FROM proc.bp_policy_document_version"):
            v = db.versions.get((params[0], params[1]))
            self._rows = [(v["filename"], v["key"], v["text"])] if v else []
        elif s.startswith("SELECT EXISTS (SELECT 1 FROM proc.bp_policy_extraction_item WHERE document_id"):
            self._rows = [((params[0], params[1]) in db.extracted,)]
        elif s.startswith("UPDATE proc.bp_policy_document_version SET parsed_text"):
            v = db.versions[(params[1], params[2])]
            v["text"], v["parsed_at"] = params[0], "now"
        else:  # pragma: no cover - a new statement must be taught to the fake
            raise AssertionError(f"unexpected SQL: {s}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(d, "_s3", lambda: fake)
    monkeypatch.setattr(d, "_bucket", lambda: "test-bucket")
    monkeypatch.setattr(d, "intake_limits", lambda: (20, 26_214_400))
    return fake


@pytest.fixture
def db():
    return FakeDB()


def _put(s3, name, data):
    upload_id = str(uuid.uuid4())
    s3.objects[d.upload_key(upload_id, name)] = data
    return {"uploadId": upload_id, "name": name, "revisionOf": None}


def _key(up):
    return d.upload_key(up["uploadId"], up["name"])


# ---------------------------------------------------------------- issue

def test_issue_returns_an_id_name_and_type_per_file_and_no_url_or_key(s3):
    out = d.issue_uploads([{"name": "a/b/Refund Policy?.pdf", "size": 10, "contentType": "text/html"}], actor="t")
    assert len(out) == 1
    item = out[0]
    assert set(item) == {"uploadId", "safeName", "contentType"}
    assert str(uuid.UUID(item["uploadId"])) == item["uploadId"] and uuid.UUID(item["uploadId"]).version == 4
    assert item["safeName"] == "Refund Policy_.pdf"
    assert item["contentType"] == "application/pdf"  # from the suffix, not what the browser claimed
    assert d.upload_key(item["uploadId"], "a/b/Refund Policy?.pdf") == \
        f"agent-policy-documents/uploads/{item['uploadId']}/Refund Policy_.pdf"


@pytest.mark.parametrize("name,ctype", [("a.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                                        ("a.TXT", "text/plain"), ("a.md", "text/markdown")])
def test_issue_content_type_follows_the_suffix(s3, name, ctype):
    assert d.issue_uploads([{"name": name, "size": 1}], actor="t")[0]["contentType"] == ctype


@pytest.mark.parametrize("upload_id", ["not-a-uuid", uuid.uuid4().hex, str(uuid.uuid4()).upper(), "",
                                       None, str(uuid.uuid1()), f"{uuid.uuid4()}/../x"])
def test_upload_key_needs_a_canonical_uuid4(upload_id):
    with pytest.raises(ValueError, match="not an agent-policy upload"):
        d.upload_key(upload_id, "Refund.pdf")


def test_safe_name_is_capped_at_120():
    assert len(d.safe_name("x" * 300 + ".pdf")) == 120


@pytest.mark.parametrize("name", ["policy.exe", "policy.doc", "policy", "policy.PDF.zip"])
def test_suffix_refused(s3, name):
    with pytest.raises(ValueError, match="accepted"):
        d.issue_uploads([{"name": name, "size": 10, "contentType": "x"}], actor="t")


def test_uppercase_accepted_suffix_is_fine(s3):
    assert len(d.issue_uploads([{"name": "P.DOCX", "size": 1, "contentType": "x"}], actor="t")) == 1


def test_size_refused_over_limit_and_whole_request_refused(s3):
    files = [{"name": "ok.pdf", "size": 10, "contentType": "x"},
             {"name": "big.pdf", "size": 26_214_401, "contentType": "x"}]
    with pytest.raises(ValueError, match="big.pdf"):
        d.issue_uploads(files, actor="t")


def test_size_at_limit_accepted_and_empty_refused(s3):
    assert d.issue_uploads([{"name": "a.txt", "size": 26_214_400, "contentType": "x"}], actor="t")
    with pytest.raises(ValueError, match="empty"):
        d.issue_uploads([{"name": "a.txt", "size": 0, "contentType": "x"}], actor="t")


def test_the_21st_file_is_refused(s3):
    files = [{"name": f"f{i}.txt", "size": 1, "contentType": "text/plain"} for i in range(21)]
    with pytest.raises(ValueError, match="At most 20"):
        d.issue_uploads(files, actor="t")
    assert len(d.issue_uploads(files[:20], actor="t")) == 20


def test_missing_limits_refuse(monkeypatch):
    from fastapi import HTTPException

    def refuse():
        raise HTTPException(status_code=503, detail="not configured")
    monkeypatch.setattr(d, "intake_limits", refuse)
    with pytest.raises(HTTPException):
        d.issue_uploads([{"name": "a.txt", "size": 1, "contentType": "x"}], actor="t")


def test_intake_limits_is_the_routers_function(monkeypatch):
    import api.routers.documents as router
    monkeypatch.setattr(router, "_intake_limits", lambda: (7, 99))
    assert d.intake_limits() == (7, 99)


# ---------------------------------------------------------------- normalise

@pytest.mark.parametrize("name,expected", [
    ("Refund Policy v2.docx", "refund policy"),
    ("refund_policy (1).pdf", "refund policy"),
    ("Refund Policy.pdf", "refund policy"),
    ("Refund Policy - final v3.pdf", "refund policy"),
    ("Refund   Policy_draft.md", "refund policy"),
    ("Refund Policy rev2.txt", "refund policy"),
    ("draft.docx", "draft"),  # never normalised to nothing
    ("Refund Policy(2).pdf", "refund policy"),  # a parenthesised number needs no separator
    # a suffix word only counts after a separator: these are words, not version marks
    ("Overdraft.pdf", "overdraft"),
    ("Semifinal.docx", "semifinal"),
    ("Card Overdraft Final.pdf", "card overdraft"),
    ("Preview.md", "preview"),
])
def test_name_normalisation(name, expected):
    assert d.normalise(name) == expected


# ---------------------------------------------------------------- register

def test_same_bytes_same_version_new_bytes_new_version(s3, db):
    first = d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")[0]
    assert first == {"documentId": 1, "version": 1, "title": "Refund Policy", "isRevision": False,
                     "duplicate": False, "extracted": False}
    again = d.register_uploads(db, [_put(s3, "Refund Policy v2.pdf", b"one")], actor="u")[0]
    assert again == {**first, "duplicate": True}
    assert len(db.versions) == 1 and db.docs[1]["latest"] == 1

    second = d.register_uploads(db, [_put(s3, "refund_policy (1).pdf", b"two")], actor="u")[0]
    assert second == {"documentId": 1, "version": 2, "title": "Refund Policy", "isRevision": True,
                      "duplicate": False, "extracted": False}
    assert db.docs[1]["latest"] == 2 and len(db.docs) == 1
    v2 = db.versions[(1, 2)]
    assert v2["hash"] == hashlib.sha256(b"two").hexdigest() and v2["size"] == 3 and v2["by"] == "u"
    assert db.autocommit is True  # restored


def test_unrelated_name_is_a_new_document(s3, db):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")
    other = d.register_uploads(db, [_put(s3, "Travel Policy.pdf", b"one")], actor="u")[0]
    assert other["documentId"] == 2 and other["version"] == 1 and other["title"] == "Travel Policy"


def test_revision_of_overrides_name_matching(s3, db):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")
    d.register_uploads(db, [_put(s3, "Travel Policy.pdf", b"t1")], actor="u")
    up = _put(s3, "Refund Policy v2.pdf", b"t2")  # the name matches document 1 ...
    up["revisionOf"] = 2                            # ... but the caller says document 2
    out = d.register_uploads(db, [up], actor="u")[0]
    assert out["documentId"] == 2 and out["version"] == 2 and out["title"] == "Travel Policy"
    assert db.docs[1]["latest"] == 1


def test_revision_of_unknown_document_is_refused(s3, db):
    up = _put(s3, "Refund Policy.pdf", b"one")
    up["revisionOf"] = 99
    with pytest.raises(ValueError, match="does not exist"):
        d.register_uploads(db, [up], actor="u")
    assert db.docs == {} and db.autocommit is True


def test_register_refuses_a_bad_upload_id_and_oversize(s3, db, monkeypatch):
    s3.objects["documents/o.pdf"] = b"x"
    with pytest.raises(ValueError, match="not an agent-policy upload"):
        d.register_uploads(db, [{"uploadId": "documents", "name": "o.pdf"}], actor="u")
    up = _put(s3, "Big.pdf", b"0123456789")
    monkeypatch.setattr(d, "intake_limits", lambda: (20, 5))
    with pytest.raises(ValueError, match="larger"):
        d.register_uploads(db, [up], actor="u")
    assert db.docs == {}


def test_register_of_an_object_that_was_never_uploaded_is_a_refusal(s3, db):
    class Missing(Exception):
        response = {"Error": {"Code": "404"}}

    def head(Bucket, Key):
        raise Missing()
    s3.head_object = head
    with pytest.raises(ValueError, match="Refund.pdf: the file was not uploaded"):
        d.register_uploads(db, [{"uploadId": str(uuid.uuid4()), "name": "Refund.pdf"}], actor="u")
    assert db.docs == {}


def test_a_failed_insert_rolls_back_and_restores_autocommit(s3, db, monkeypatch):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")

    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(d, "_bump_latest", boom)
    with pytest.raises(RuntimeError):
        d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"two")], actor="u")
    assert (1, 2) not in db.versions and db.docs[1]["latest"] == 1
    assert db.rollbacks >= 1 and db.autocommit is True


# ---------------------------------------------------------------- text

def test_document_text_txt_is_decoded_stored_and_then_served_from_the_row(s3, db):
    d.register_uploads(db, [_put(s3, "Refund.txt", "Refunds over £500 need approval.\xff".encode("utf-8") + b"\xff")],
                       actor="u")
    text = d.document_text(db, 1, 1)
    assert text.startswith("Refunds over £500 need approval.") and "�" in text
    assert db.versions[(1, 1)]["text"] == text and db.versions[(1, 1)]["parsed_at"]
    s3.objects.clear()  # the second read must not touch S3
    assert d.document_text(db, 1, 1) == text


def test_document_text_pdf_goes_through_the_parser(s3, db, monkeypatch):
    from services.extraction import parser
    seen = {}

    class Parsed:
        full_text = "# Refunds\nOver 500 needs approval."

    def fake_parse(path):
        with open(path, "rb") as fh:
            seen["bytes"] = fh.read()
        seen["path"] = path
        return Parsed()
    monkeypatch.setattr(parser, "parse", fake_parse)
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"%PDF-fake")], actor="u")
    assert d.document_text(db, 1, 1) == Parsed.full_text
    assert seen["bytes"] == b"%PDF-fake" and seen["path"].endswith(".pdf")
    import os
    assert not os.path.exists(seen["path"])  # temp file removed


def test_empty_text_is_unreadable(s3, db):
    d.register_uploads(db, [_put(s3, "Blank.md", b"   \n\t")], actor="u")
    with pytest.raises(d.DocumentUnreadable):
        d.document_text(db, 1, 1)
    assert db.versions[(1, 1)]["text"] is None


# ---------------------------------------------------------------- list

def test_list_documents_has_every_version_and_no_text(s3, db, monkeypatch):
    d.register_uploads(db, [_put(s3, "Refund Policy.txt", b"one")], actor="u")
    d.register_uploads(db, [_put(s3, "Refund Policy v2.txt", b"two")], actor="u")

    class ListCursor:
        def __init__(self):
            self.rows = []

        def execute(self, sql, params=()):
            if "FROM proc.bp_policy_document_version" in sql:
                self.rows = [(k[0], k[1], v["filename"], v["size"], v["hash"], v["by"], None, None)
                             for k, v in sorted(db.versions.items())]
            else:
                self.rows = [(v["id"], v["title"], v["match"], v["latest"], v["by"], None)
                             for v in db.docs.values()]

        def fetchall(self):
            return self.rows

    class ListConn:
        def cursor(self):
            return ListCursor()
    out = d.list_documents(ListConn())
    assert len(out) == 1 and [v["version"] for v in out[0]["versions"]] == [1, 2]
    assert out[0]["latestVersion"] == 2
    assert all("parsedText" not in v and "text" not in v for v in out[0]["versions"])


# ---------------------------------------------------------------- fix round 1

def test_key_must_carry_the_issued_name(s3, db):
    up = _put(s3, "Refund Policy.pdf", b"one")
    up["name"] = "Travel Policy.pdf"  # the key is rebuilt from the name, so it names no uploaded object
    with pytest.raises(ValueError, match="Travel Policy.pdf: the file was not uploaded"):
        d.register_uploads(db, [up], actor="u")
    assert db.docs == {}


@pytest.mark.parametrize("segment", ["not-a-uuid", uuid.uuid4().hex, str(uuid.uuid4()).upper(), "",
                                     f"{uuid.uuid4()}/../x"])
def test_upload_id_must_be_a_canonical_uuid4(s3, db, segment):
    s3.objects[f"{d.UPLOAD_PREFIX}{segment}/Refund.pdf"] = b"one"
    with pytest.raises(ValueError, match="not an agent-policy upload"):
        d.register_uploads(db, [{"uploadId": segment, "name": "Refund.pdf"}], actor="u")
    assert db.docs == {}


def test_refusal_messages_never_carry_the_key(s3, db, monkeypatch):
    up = _put(s3, "Refund.pdf", b"0123456789")
    monkeypatch.setattr(d, "intake_limits", lambda: (20, 5))
    with pytest.raises(ValueError) as err:
        d.register_uploads(db, [up], actor="u")
    assert d.UPLOAD_PREFIX not in str(err.value) and up["uploadId"] not in str(err.value)


def test_dot_dot_names(s3, db):
    # "Refund..pdf" is an ordinary file name and registers under its issued key
    out = d.register_uploads(db, [_put(s3, "Refund..pdf", b"one")], actor="u")[0]
    assert out["version"] == 1 and db.versions[(1, 1)]["filename"] == "Refund..pdf"
    # a path in the name is reduced to its basename, so it cannot point anywhere else
    up = _put(s3, "../../etc/Travel.pdf", b"two")
    assert _key(up).endswith("/Travel.pdf")
    assert d.register_uploads(db, [up], actor="u")[0]["title"] == "Travel"


def test_autocommit_is_restored_to_what_the_caller_had(s3, db):
    db.autocommit = False
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"one")], actor="u")
    assert db.autocommit is False
    db.autocommit = True
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"two")], actor="u")
    assert db.autocommit is True


def test_the_document_is_locked_before_the_duplicate_check(s3, db):
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"one")], actor="u")
    db.log.clear()
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"one")], actor="u")
    lock = next(i for i, q in enumerate(db.log) if q.endswith("FOR UPDATE"))
    check = next(i for i, q in enumerate(db.log) if "content_hash = %s" in q and q.startswith("SELECT"))
    assert lock < check


class _UniqueViolation(Exception):
    pgcode = "23505"
    diag = SimpleNamespace(constraint_name="ux_bp_policy_document_version_hash")


def test_a_concurrent_identical_upload_is_a_duplicate_not_an_error(s3, db, monkeypatch):
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"one")], actor="u")
    calls = {"n": 0}
    real = d._version_by_hash

    def racing_check(cur, doc_id, h):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(cur, doc_id, h)  # the other writer lands after our check

    def insert_loses_race(*a, **k):
        raise _UniqueViolation("duplicate key")
    monkeypatch.setattr(d, "_version_by_hash", racing_check)
    monkeypatch.setattr(d, "_insert_version", insert_loses_race)
    out = d.register_uploads(db, [_put(s3, "Refund.pdf", b"one")], actor="u")[0]
    assert out == {"documentId": 1, "version": 1, "title": "Refund", "isRevision": False, "duplicate": True,
                   "extracted": False}
    assert db.docs[1]["latest"] == 1 and db.autocommit is True


def test_other_integrity_errors_still_raise(s3, db, monkeypatch):
    d.register_uploads(db, [_put(s3, "Refund.pdf", b"one")], actor="u")

    class Other(Exception):
        pgcode = "23505"
        diag = SimpleNamespace(constraint_name="bp_policy_document_version_pkey")
    monkeypatch.setattr(d, "_insert_version", lambda *a, **k: (_ for _ in ()).throw(Other("pk")))
    with pytest.raises(Other):
        d.register_uploads(db, [_put(s3, "Refund.pdf", b"two")], actor="u")


def test_body_longer_than_the_limit_is_refused_even_if_head_said_small(s3, db, monkeypatch):
    monkeypatch.setattr(d, "intake_limits", lambda: (20, 5))
    up = _put(s3, "Refund.txt", b"0123456789")
    s3.objects[_key(up)] = b"0123456789"
    s3.head_sizes[_key(up)] = 3  # head under-reports
    reads = []
    real_get = s3.get_object

    def get_object(Bucket, Key):
        body = real_get(Bucket, Key)["Body"]
        orig = body.read
        body.read = lambda n=-1: (reads.append(n), orig(n))[1]
        return {"Body": body}
    monkeypatch.setattr(s3, "get_object", get_object)
    with pytest.raises(ValueError, match="larger"):
        d.register_uploads(db, [up], actor="u")
    assert reads == [6] and db.docs == {}


# ---------------------------------------------------------------- asNew

def test_as_new_makes_a_new_document_despite_a_matching_name(s3, db):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")
    up = dict(_put(s3, "Refund Policy v2.pdf", b"two"), asNew=True)
    out = d.register_uploads(db, [up], actor="u")[0]
    assert out["documentId"] == 2 and out["version"] == 1 and out["isRevision"] is False
    assert db.docs[1]["latest"] == 1  # the matching document was left alone
    # without asNew the same name is a revision of document 1
    again = d.register_uploads(db, [_put(s3, "Refund Policy v3.pdf", b"three")], actor="u")[0]
    assert again["documentId"] == 1 and again["version"] == 2


@pytest.mark.parametrize("flag", [False, None, "true", 1])
def test_only_a_real_true_as_new_skips_matching(s3, db, flag):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")
    out = d.register_uploads(db, [dict(_put(s3, "Refund Policy.pdf", b"two"), asNew=flag)], actor="u")[0]
    assert out["documentId"] == 1 and out["version"] == 2


def test_as_new_with_revision_of_is_refused(s3, db):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")
    up = dict(_put(s3, "Refund Policy.pdf", b"two"), asNew=True, revisionOf=1)
    with pytest.raises(ValueError, match="choose either a revision or a new document"):
        d.register_uploads(db, [up], actor="u")
    assert len(db.docs) == 1 and db.docs[1]["latest"] == 1



# ---------------------------------------------------------------- fix round 2

def test_a_long_name_keeps_its_extension_through_issue_register_and_read(s3, db):
    name = "x" * 125 + ".txt"
    issued = d.issue_uploads([{"name": name, "size": 3}], actor="u")[0]
    assert len(issued["safeName"]) == 120 and issued["safeName"].endswith(".txt")
    assert issued["safeName"] == "x" * 116 + ".txt"
    s3.objects[d.upload_key(issued["uploadId"], name)] = b"Refunds need approval."
    out = d.register_uploads(db, [{"uploadId": issued["uploadId"], "name": name}], actor="u")[0]
    assert out["version"] == 1 and db.versions[(out["documentId"], 1)]["filename"].endswith(".txt")
    assert d.document_text(db, out["documentId"], 1) == "Refunds need approval."


def test_a_long_pdf_name_is_issued_with_its_suffix(s3):
    issued = d.issue_uploads([{"name": "x" * 125 + ".pdf", "size": 3}], actor="u")[0]
    assert issued["safeName"] == "x" * 116 + ".pdf" and issued["contentType"] == "application/pdf"


def test_an_as_new_retry_is_the_version_it_already_made(s3, db):
    d.register_uploads(db, [_put(s3, "Refund Policy.pdf", b"one")], actor="u")
    up = dict(_put(s3, "Refund Policy.pdf", b"two"), asNew=True)
    first = d.register_uploads(db, [up], actor="u")[0]
    assert first["documentId"] == 2 and first["duplicate"] is False
    again = d.register_uploads(db, [up], actor="u")[0]
    assert again == {**first, "duplicate": True}
    assert len(db.docs) == 2 and db.autocommit is True


# ---------------------------------------------------------------- final fix wave (I4)

@pytest.mark.parametrize("bad", ["type", "key", "missing", "size", "revision"])
def test_one_refused_upload_registers_none_of_the_batch(s3, db, monkeypatch, bad):
    """All or nothing: a refusal anywhere in the batch writes nothing, even for the good files
    before it."""
    good = [_put(s3, "Refund Policy.pdf", b"one"), _put(s3, "Travel Policy.pdf", b"two")]
    if bad == "type":
        last = {"uploadId": str(uuid.uuid4()), "name": "Notes.exe"}
    elif bad == "key":
        last = {"uploadId": "documents", "name": "o.pdf"}
    elif bad == "missing":
        last = {"uploadId": str(uuid.uuid4()), "name": "Gone.pdf"}
    elif bad == "size":
        last = _put(s3, "Big.pdf", b"x" * 50)
        monkeypatch.setattr(d, "intake_limits", lambda: (20, 10))
    else:
        last = dict(_put(s3, "Other.pdf", b"three"), revisionOf=99)
    with pytest.raises(ValueError):
        d.register_uploads(db, good + [last], actor="u")
    assert db.docs == {} and db.versions == {} and db.commits == 0


def test_each_result_says_whether_an_extraction_has_read_that_version(s3, db):
    up = _put(s3, "Refund Policy.pdf", b"one")
    first = d.register_uploads(db, [up], actor="u")[0]
    assert first["extracted"] is False
    db.extracted.add((first["documentId"], first["version"]))
    again = d.register_uploads(db, [_put(s3, "Refund Policy v2.pdf", b"one")], actor="u")[0]
    assert again["duplicate"] is True and again["extracted"] is True
    other = d.register_uploads(db, [_put(s3, "Refund Policy v3.pdf", b"two")], actor="u")[0]
    assert other["version"] == 2 and other["extracted"] is False
