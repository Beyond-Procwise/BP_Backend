"""A prompt's wording reaches its own screen unaltered only when policy says so."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

main = pytest.importorskip("api.main")


@pytest.fixture()
def policy(monkeypatch):
    state = {"value": True}

    def limit(policy, rule, **kw):
        assert (policy, rule) == ("prompt_engine", "wording_read_passthrough")
        if isinstance(state["value"], Exception):
            raise state["value"]
        return state["value"]

    from src.services import governed_limits
    monkeypatch.setattr(governed_limits, "limit", limit)
    return state


def test_the_two_wording_reads_pass_through_when_policy_allows(policy):
    assert main._prompt_wording_exempt("GET", "/prompts/joshi", 200)
    assert main._prompt_wording_exempt("GET", "/prompts/email_compose_rfq/versions/3", 200)


@pytest.mark.parametrize("value", [False, None, "true", 1, RuntimeError("policy unreadable")])
def test_anything_but_a_stated_true_scrubs(policy, value):
    policy["value"] = value
    assert not main._prompt_wording_exempt("GET", "/prompts/joshi", 200)


@pytest.mark.parametrize("method,path,status", [
    ("GET", "/prompts", 200),                          # the listing
    ("GET", "/prompts/joshi/", 200),                   # a trailing slash
    ("GET", "/prompts/joshi/versions/3/extra", 200),   # a child path
    ("GET", "/prompts/joshi/history", 200),            # another read
    ("POST", "/prompts/joshi/versions", 201),          # a write answer
    ("POST", "/prompts/joshi/promote", 200),
    ("GET", "/prompts/joshi", 404),                    # an error answer
    ("GET", "/prompts/joshi", 403),
    ("GET", "/agents/joshi", 200),                     # somewhere else entirely
])
def test_everything_else_is_still_scrubbed(policy, method, path, status):
    assert not main._prompt_wording_exempt(method, path, status)


# --------------------------------------------------------------------------
# Through the real application, scrubber included
# --------------------------------------------------------------------------
# Holds an internal route, which the output filter withholds unless policy lets the wording through.
WORDING = "Reply to the supplier at /admin/users/list, then retry the tool call."


@pytest.fixture()
def served(policy, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import api.routers.prompts as prompts_mod
    from api.auth import require_user

    version = {"prompt_key": "joshi", "version": 1, "syntax": "v1", "template": WORDING,
               "author": "sub-ada", "created_at": "2026-10-13", "change_note": None,
               "content_sha256": "abc", "variables": [], "fragment_refs": [], "settings": {}}
    monkeypatch.setattr(prompts_mod.registry, "get_version", lambda key, n: version)
    monkeypatch.setattr(prompts_mod, "gate", lambda *a, **k: None)
    main.app.dependency_overrides[require_user] = lambda: SimpleNamespace(subject="sub-bob")
    try:
        yield TestClient(main.app)      # no `with`: start-up is not what is under test
    finally:
        main.app.dependency_overrides.pop(require_user, None)


def test_the_wording_arrives_unaltered_when_policy_allows(served, policy):
    assert served.get("/prompts/joshi/versions/1").json()["template"] == WORDING


def test_the_wording_is_scrubbed_when_policy_says_no(served, policy):
    policy["value"] = False
    assert served.get("/prompts/joshi/versions/1").json()["template"] != WORDING


def test_a_refusal_keeps_its_code_as_a_field_through_the_real_application(served, monkeypatch):
    import api.routers.prompts as prompts_mod
    from services.prompt_engine import registry as R

    def missing(key, n):
        raise R.VersionNotFound(f"prompt {key!r} has no version {n}")

    monkeypatch.setattr(prompts_mod.registry, "get_version", missing)
    answer = served.get("/prompts/joshi/versions/9")

    assert answer.status_code == 404
    assert answer.json()["code"] == "version_not_found"
