"""An upload is bounded, and it belongs to whoever the token says.

``POST /document/embed-document`` took ``List[UploadFile]`` with an extension
allow-list and nothing else: no count cap, no size cap, and no check that the
declared content type had anything to do with the extension. The owner was read
from a Form field, an ``x-user-id`` header or a query parameter -- three places a
caller controls -- so an upload could be attributed to anybody.

The caps are policy (DocumentIntakeAuthorityPolicy, #807), not constants and not
environment variables, because they are a governance limit on what may enter the
product. A missing limit refuses: an unconfigured cap is not an unlimited one.

What these tests do NOT claim, so nobody reads more into them than is there:
the size cap rejects a file that is too big, but FastAPI has already parsed the
multipart body by the time the handler runs, so the bytes have been received.
A true bound before that point belongs in middleware or at the ingress.
"""

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import documents as documents_router

OWNER = "sub-buyer-001"
SOMEONE_ELSE = "sub-someone-else"

_ABSENT = object()


class _Principal:
    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"


class _DummyRAG:
    uploaded_collection = "uploaded_documents"

    def ensure_collection(self, name):
        return None


class _DummyPipeline:
    def __init__(self):
        self.rag = _DummyRAG()
        self.activations = []

    def activate_uploaded_context(self, document_ids, *, metadata=None, session_id=None):
        self.activations.append({"metadata": dict(metadata or {}), "session_id": session_id})


@pytest.fixture
def embedded():
    """Every call to the embedding service, so the metadata can be inspected."""
    return []


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    """The authorization gate is not what these tests are about."""
    monkeypatch.setattr(documents_router, "gate", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _embedding_service(monkeypatch, embedded):
    from types import SimpleNamespace

    class _Service:
        def __init__(self, agent_nick, *, collection_name, **_kw):
            pass

        def embed_document(self, *, filename, file_bytes, metadata):
            embedded.append({"filename": filename, "bytes": len(file_bytes),
                             "metadata": dict(metadata)})
            return SimpleNamespace(document_id=f"doc-{len(embedded)}",
                                   collection="uploaded_documents", chunk_count=1,
                                   metadata={"filename": filename, "doc_name": filename})

    monkeypatch.setattr(documents_router, "DocumentEmbeddingService", _Service)


def _limits(monkeypatch, *, max_files=5, max_bytes=1024):
    """Install the intake policy these tests need.

    ``_ABSENT`` for either value leaves that key out, which is the state a
    deployment is in before the migration runs.
    """
    rules = {"effect": "allow"}
    if max_files is not _ABSENT:
        rules["max_files_per_request"] = max_files
    if max_bytes is not _ABSENT:
        rules["max_bytes_per_file"] = max_bytes
    monkeypatch.setattr(
        documents_router, "_intake_policy",
        lambda: {"policyName": "DocumentIntakeAuthorityPolicy",
                 "details": {"policy_identifier": "document_intake_authority",
                             "rules": rules}},
    )


def _no_policy(monkeypatch):
    monkeypatch.setattr(documents_router, "_intake_policy", lambda: None)


def _client(subject=OWNER):
    app = FastAPI()
    app.include_router(documents_router.router)
    app.state.rag_pipeline = _DummyPipeline()
    app.state.agent_nick = object()
    if subject is None:
        app.dependency_overrides[documents_router.require_user] = lambda: None
    else:
        app.dependency_overrides[documents_router.require_user] = lambda: _Principal(subject)
    return TestClient(app)


def _file(name="invoice.txt", data=b"hello", content_type="text/plain"):
    return (name, io.BytesIO(data), content_type)


# ---------------------------------------------------------------------------
# size and count
# ---------------------------------------------------------------------------
def test_an_oversized_file_is_refused(monkeypatch):
    """Nothing capped this. A single upload could be any size at all.

    Refused per FILE, not per request: this endpoint reports an outcome for each
    document and only fails the whole call when none of them survived. Size is a
    property of one file, so it belongs in that report — the count cap below is
    a property of the request and is a 413.
    """
    _limits(monkeypatch, max_bytes=100)

    response = _client().post(
        "/document/embed-document",
        files={"files": _file(data=b"x" * 5000)},
    )

    assert response.status_code == 400, (
        f"an oversized upload was accepted: {response.status_code} {response.text}")
    assert "100 byte limit" in response.json()["detail"], response.text


def test_an_oversized_file_does_not_take_the_rest_of_the_batch_with_it(monkeypatch, embedded):
    """The other half of refusing per file: the good ones still land, and the
    refused one is reported rather than silently dropped."""
    _limits(monkeypatch, max_bytes=100)

    response = _client().post(
        "/document/embed-document",
        files=[("files", _file("small.txt", b"x" * 10)),
               ("files", _file("huge.txt", b"x" * 5000))],
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [d["filename"] for d in body["failed"]] == ["huge.txt"], body
    assert [e["filename"] for e in embedded] == ["small.txt"]


def test_a_file_within_the_cap_is_accepted(monkeypatch, embedded):
    """The cap must be a cap, not a closed door."""
    _limits(monkeypatch, max_bytes=100)

    response = _client().post(
        "/document/embed-document",
        files={"files": _file(data=b"x" * 50)},
    )

    assert response.status_code == 200, response.text
    assert embedded and embedded[0]["bytes"] == 50


def test_too_many_files_in_one_request_are_refused(monkeypatch):
    _limits(monkeypatch, max_files=2)

    response = _client().post(
        "/document/embed-document",
        files=[("files", _file(f"doc{i}.txt")) for i in range(5)],
    )

    assert response.status_code == 413, (
        f"an unbounded batch was accepted: {response.status_code}")


def test_the_oversized_file_is_never_embedded(monkeypatch, embedded):
    """Refusing after doing the work is not refusing."""
    _limits(monkeypatch, max_bytes=100)

    _client().post("/document/embed-document", files={"files": _file(data=b"x" * 5000)})

    assert embedded == [], f"an oversized file was embedded anyway: {embedded}"


# ---------------------------------------------------------------------------
# the caps are policy, and a missing one refuses
# ---------------------------------------------------------------------------
def test_a_missing_intake_policy_refuses(monkeypatch):
    """An unconfigured cap is not an unlimited one."""
    _no_policy(monkeypatch)

    response = _client().post("/document/embed-document", files={"files": _file()})

    assert response.status_code == 503, (
        f"uploads were accepted with no configured limit: {response.status_code}")


def test_a_missing_size_limit_refuses(monkeypatch):
    _limits(monkeypatch, max_bytes=_ABSENT)

    assert _client().post(
        "/document/embed-document", files={"files": _file()}
    ).status_code == 503


def test_a_missing_count_limit_refuses(monkeypatch):
    _limits(monkeypatch, max_files=_ABSENT)

    assert _client().post(
        "/document/embed-document", files={"files": _file()}
    ).status_code == 503


def test_an_unusable_limit_refuses(monkeypatch):
    """A mistyped policy value narrows, never widens -- the rule this codebase
    already applies to revoke_scope and self_approval."""
    _limits(monkeypatch, max_bytes="quite big please")

    assert _client().post(
        "/document/embed-document", files={"files": _file()}
    ).status_code == 503


# ---------------------------------------------------------------------------
# who the upload belongs to
# ---------------------------------------------------------------------------
def test_a_caller_cannot_attribute_an_upload_to_another_user(monkeypatch, embedded):
    """The finding: user_id came from a Form field, a header or the query
    string, so an upload could be filed under anyone."""
    _limits(monkeypatch)

    response = _client(OWNER).post(
        "/document/embed-document",
        files={"files": _file()},
        data={"user_id": SOMEONE_ELSE},
        headers={"x-user-id": SOMEONE_ELSE},
    )

    assert response.status_code == 200, response.text
    metadata = embedded[0]["metadata"]
    assert metadata.get("uploaded_by") == OWNER, (
        f"the upload was attributed to a caller-supplied id: {metadata}")


def test_a_query_parameter_cannot_attribute_it_either(monkeypatch, embedded):
    _limits(monkeypatch)

    _client(OWNER).post(
        f"/document/embed-document?user_id={SOMEONE_ELSE}",
        files={"files": _file()},
    )

    assert embedded[0]["metadata"].get("uploaded_by") == OWNER


def test_the_caller_supplied_id_is_kept_as_an_unverified_label(monkeypatch, embedded):
    """The old fields still carry something a person typed, and that is worth
    keeping -- as a label, named as one, never as identity."""
    _limits(monkeypatch)

    _client(OWNER).post(
        "/document/embed-document",
        files={"files": _file()},
        data={"user_id": "finance-team"},
    )

    metadata = embedded[0]["metadata"]
    assert metadata.get("uploaded_by") == OWNER
    assert metadata.get("uploaded_by_label") == "finance-team", metadata
    assert "finance-team" != metadata.get("uploaded_by")


def test_an_unauthenticated_upload_is_owned_by_nobody(monkeypatch, embedded):
    """With ASK_AUTH_MODE=off there is no principal. The upload must then claim
    no owner rather than falling back to the field a caller typed -- a fallback
    is the forgery, taken conditionally."""
    _limits(monkeypatch)

    _client(subject=None).post(
        "/document/embed-document",
        files={"files": _file()},
        data={"user_id": SOMEONE_ELSE},
    )

    metadata = embedded[0]["metadata"]
    assert metadata.get("uploaded_by") is None, metadata
    assert metadata.get("uploaded_by_label") == SOMEONE_ELSE


# ---------------------------------------------------------------------------
# what is being uploaded
# ---------------------------------------------------------------------------
def test_a_content_type_that_contradicts_the_extension_is_rejected(monkeypatch, embedded):
    _limits(monkeypatch)

    response = _client().post(
        "/document/embed-document",
        files={"files": ("invoice.pdf", io.BytesIO(b"%PDF-1.4"), "text/html")},
    )

    assert response.status_code == 400, response.text
    assert embedded == []


def test_a_matching_content_type_is_accepted(monkeypatch, embedded):
    _limits(monkeypatch)

    response = _client().post(
        "/document/embed-document",
        files={"files": ("invoice.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 200, response.text
    assert embedded


def test_the_extension_allow_list_still_applies(monkeypatch, embedded):
    """The check that was already there must not be lost to the new one."""
    _limits(monkeypatch)

    response = _client().post(
        "/document/embed-document",
        files={"files": ("payload.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )

    assert response.status_code == 400
    assert embedded == []
