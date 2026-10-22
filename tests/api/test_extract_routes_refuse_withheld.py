"""The two extraction entry points refuse a withheld source as a 400 the caller sees (1.3e).

The agent refuses such a source again inside, but a workflow that fails inside returns 200 with an
empty result, so the refusal would look like "Submit did nothing". Here the orchestrator is a bare
object: reaching it from a withheld request would be a 500, so a 400 proves the check came first.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.routers import agents as ar
from api.routers import workflows as wr
from src.services.mail_intake.store import QUARANTINE_PREFIX, RAW_MIME_PREFIX


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ar, "gate", lambda *a, **k: None)
    monkeypatch.setattr(wr, "gate", lambda *a, **k: None, raising=False)
    app = FastAPI()
    app.include_router(ar.router)
    app.include_router(wr.router)
    for mod in (ar, wr):
        app.dependency_overrides[mod.get_orchestrator] = lambda: object()
        app.dependency_overrides[mod.require_user] = lambda: SimpleNamespace(subject="sub-buyer")
    return TestClient(app, raise_server_exceptions=False)


# Each caller value carries a marker (an owner folder, a message name) that must never come back in the
# refusal; the store's own name may, since that is what the message names.
CASES = [
    pytest.param("/agents/process-document", {"s3_prefix": QUARANTINE_PREFIX + "OWNER-7/"}, id="agents-prefix-quarantine"),
    pytest.param("/agents/process-document", {"s3_object_key": RAW_MIME_PREFIX + "MSG-9.eml"}, id="agents-key-raw"),
    pytest.param("/agents/process-document", {"s3_prefix": "mail-intake/"}, id="agents-prefix-ancestor"),
    pytest.param("/workflows/extract", {"s3_prefix": QUARANTINE_PREFIX + "OWNER-7/"}, id="workflows-prefix-quarantine"),
    pytest.param("/workflows/extract", {"s3_object_key": RAW_MIME_PREFIX + "MSG-9.eml"}, id="workflows-key-raw"),
    pytest.param("/workflows/extract", {"s3_prefix": "mail"}, id="workflows-prefix-ancestor"),
]


@pytest.mark.parametrize("path,body", CASES)
def test_a_withheld_source_is_a_400_before_anything_starts(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 400, (r.status_code, r.text[:200])
    detail = r.json()["detail"]
    assert "withheld" in detail
    assert "OWNER-7" not in detail and "MSG-9" not in detail      # the store is named; the caller's value is not
