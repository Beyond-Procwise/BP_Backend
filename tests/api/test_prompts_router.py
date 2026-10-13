"""The prompt registry's routes: who is asked, what is logged, what is refused."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.prompts as prompts_mod
from api.auth import require_user
from services.prompt_engine import registry as R


class _Registry:
    """Stands in for the tables. The real rules are tested in tests/prompt_engine."""

    def __init__(self):
        self.calls = []
        self.version = {"prompt_key": "joshi", "version": 1, "syntax": "v1",
                        "template": "Answer by 01/04/2026.", "author": "sub-ada",
                        "created_at": "2026-10-13", "change_note": None,
                        "content_sha256": "abc"}

    def list_prompts(self):
        return [{"prompt_key": "joshi", "kind": "prompt", "owner": None,
                 "latest_version": 1, "live_version": 1}]

    def versions(self, key):
        return [self.version] if key == "joshi" else []

    def resolve(self, key):
        return self.version

    def pointer_history(self, key):
        return []

    def get_version(self, key, version):
        if version != 1:
            raise R.VersionNotFound(f"prompt {key!r} has no version {version}")
        return self.version

    def save_version(self, key, **kw):
        self.calls.append(("save", key, kw["author"]))
        return 2

    def promote(self, key, version, *, approved_by, note=None):
        if approved_by == "sub-ada":
            raise R.SelfApproval("a second person promotes it")
        self.calls.append(("promote", key, version, approved_by))
        return version

    def rollback(self, key, *, approved_by, note=None, to_version=None):
        self.calls.append(("rollback", key, approved_by))
        return 1


@pytest.fixture()
def client(monkeypatch):
    fake = _Registry()
    for name in ("list_prompts", "versions", "resolve", "pointer_history", "get_version",
                 "save_version", "promote", "rollback"):
        monkeypatch.setattr(prompts_mod.registry, name, getattr(fake, name))
    asked = []
    refuse = set()

    def gate(action, principal, *, context=None, agent="api", engine=None):
        asked.append((action, getattr(principal, "subject", None), dict(context or {})))
        if action in refuse:
            raise HTTPException(status_code=403, detail="not permitted")

    monkeypatch.setattr(prompts_mod, "gate", gate)
    app = FastAPI()
    app.include_router(prompts_mod.router)
    who = {"subject": "sub-bob"}
    app.dependency_overrides[require_user] = lambda: SimpleNamespace(subject=who["subject"])
    c = TestClient(app)
    c.fake, c.asked, c.refuse, c.who = fake, asked, refuse, who
    return c


def test_reading_a_prompts_wording_is_logged_with_who_and_which_version(client):
    assert client.get("/prompts/joshi/versions/1").json()["template"] == "Answer by 01/04/2026."
    assert client.get("/prompts/joshi").json()["live"]["version"] == 1

    assert client.asked == [
        ("prompt.read", "sub-bob", {"prompt_key": "joshi", "version": 1}),
        ("prompt.read", "sub-bob", {"prompt_key": "joshi"}),
    ]


def test_a_refused_read_returns_no_wording(client):
    client.refuse.add("prompt.read")
    for path in ("/prompts", "/prompts/joshi", "/prompts/joshi/versions/1"):
        answer = client.get(path)
        assert answer.status_code == 403 and "Answer by" not in answer.text


def test_the_listing_carries_no_wording(client):
    assert "template" not in client.get("/prompts").text


def test_saving_records_the_signed_in_person_as_author_and_is_not_in_use(client):
    answer = client.post("/prompts/joshi/versions", json={"template": "New wording."})

    assert answer.status_code == 201
    assert answer.json() == {"prompt_key": "joshi", "version": 2, "in_use": False}
    assert client.fake.calls == [("save", "joshi", "sub-bob")]
    assert client.asked[-1][0] == "prompt.author"


def test_promotion_asks_for_the_promote_right_and_names_the_approver(client):
    answer = client.post("/prompts/joshi/promote", json={"version": 1, "note": "ok"})

    assert answer.json() == {"prompt_key": "joshi", "live_version": 1}
    assert client.fake.calls == [("promote", "joshi", 1, "sub-bob")]
    assert client.asked[-1][:2] == ("prompt.promote", "sub-bob")


def test_the_author_promoting_their_own_version_is_refused(client):
    client.who["subject"] = "sub-ada"
    answer = client.post("/prompts/joshi/promote", json={"version": 1})

    assert answer.status_code == 403
    assert answer.json()["detail"]["code"] == "SELF_APPROVAL"
    assert client.fake.calls == []


def test_rollback_is_one_call_under_the_promote_right(client):
    answer = client.post("/prompts/joshi/rollback", json={})

    assert answer.json() == {"prompt_key": "joshi", "live_version": 1}
    assert client.asked[-1][0] == "prompt.promote"


@pytest.mark.parametrize("method,path,body", [
    ("post", "/prompts/joshi/versions", {"template": "x"}),
    ("post", "/prompts/joshi/promote", {"version": 1}),
    ("post", "/prompts/joshi/rollback", {}),
])
def test_nobody_signed_in_cannot_change_anything(client, method, path, body):
    client.who["subject"] = None
    assert getattr(client, method)(path, json=body).status_code == 401
    assert client.fake.calls == []


def test_a_version_that_is_not_there_is_a_404_with_the_contract_code(client):
    answer = client.get("/prompts/joshi/versions/9")
    assert answer.status_code == 404
    assert answer.json()["detail"]["code"] == "VERSION_NOT_FOUND"
