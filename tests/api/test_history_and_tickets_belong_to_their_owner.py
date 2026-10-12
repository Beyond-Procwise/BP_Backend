"""A person's chat history and support tickets are theirs, not the room's.

Three endpoints took the caller's word for who they were, or did not ask.
`GET /system/history/{user_id}` returned whichever history the path named.
`/workflows/ask` filed each question and answer under the `user_id` typed in the
request body -- and the UI's default for that field is the constant `ui`, so
signed-in people shared one history and each one's questions became the next
one's context. `GET /support/tickets` listed every ticket, with its author's
name, email and message, to anyone signed in.

The identity is the token's. With sign-in switched off there is no token to
read, and these behave as they did: nothing here can tell callers apart then.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.auth import Principal, require_user  # noqa: E402
from api.routers import support, system  # noqa: E402
from src.api.routers.workflows import AskRequest, ask_question, ask_question_stream  # noqa: E402

ALICE = Principal(subject="sub-alice", email="alice@example.com", username="alice")


# -- chat history -------------------------------------------------------------


class _History:
    def __init__(self):
        self.asked_for = []

    def get_history(self, user_id):
        self.asked_for.append(user_id)
        return [{"query": "q", "answer": "a"}]


class _Pipeline:
    def __init__(self):
        self.history_manager = _History()
        self.seen = {}

    def answer_question(self, **kwargs):
        self.seen = kwargs
        return {"answer": "", "follow_ups": [], "retrieved_documents": [], "next_steps": []}


def _history_client(principal):
    pipeline = _Pipeline()
    app = FastAPI()
    app.include_router(system.router)
    app.dependency_overrides[require_user] = lambda: principal
    app.dependency_overrides[system.get_rag_pipeline] = lambda: pipeline
    return TestClient(app), pipeline


def test_a_person_reads_their_own_history():
    client, pipeline = _history_client(ALICE)
    assert client.get("/system/history/sub-alice").status_code == 200
    assert pipeline.history_manager.asked_for == ["sub-alice"]


def test_a_person_is_refused_someone_elses_history():
    client, pipeline = _history_client(ALICE)
    assert client.get("/system/history/sub-bob").status_code == 403
    assert pipeline.history_manager.asked_for == []


def test_with_sign_in_off_history_reads_as_before():
    client, pipeline = _history_client(None)
    assert client.get("/system/history/anyone").status_code == 200
    assert pipeline.history_manager.asked_for == ["anyone"]


# -- asking -------------------------------------------------------------------


class _Request:
    headers: dict = {}


def _ask(principal, **fields):
    pipeline = _Pipeline()
    request = AskRequest(query="top suppliers", user_id="ui", **fields)
    asyncio.run(ask_question(request, _Request(), pipeline, principal=principal))
    return pipeline.seen


def test_a_question_is_filed_under_the_token_not_the_typed_name():
    assert _ask(ALICE)["user_id"] == "sub-alice"


def test_the_fallback_session_is_the_callers_own():
    # With no session on the request the conversation is keyed by the user; left
    # as the typed name, everyone who sent `ui` would share one conversation.
    assert _ask(ALICE)["session_id"] == "sub-alice"
    assert _ask(ALICE, session_id="s-1")["session_id"] == "s-1"


def test_with_sign_in_off_the_typed_name_is_all_there_is():
    assert _ask(None)["user_id"] == "ui"


def test_the_streamed_question_is_filed_under_the_token_too():
    pipeline = _Pipeline()

    async def _drain():
        response = await ask_question_stream(
            AskRequest(query="top suppliers", user_id="ui"), _Request(), pipeline, principal=ALICE
        )
        async for _ in response.body_iterator:
            pass

    asyncio.run(_drain())
    assert pipeline.seen["user_id"] == "sub-alice"
    assert pipeline.seen["session_id"] == "sub-alice"


# -- support tickets ----------------------------------------------------------


class _Cursor:
    description = [("reference",), ("user_email",)]

    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self._log.append((" ".join(sql.split()), params))

    def fetchall(self):
        return [("SUP-1", "alice@example.com")]


class _Conn:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _Cursor(self._log)


class _Nick:
    def __init__(self):
        self.queries = []

    def get_db_connection(self):
        return _Conn(self.queries)


def _tickets(principal, role, monkeypatch):
    nick = _Nick()
    monkeypatch.setattr(support.rbac, "effective_role", lambda p: role)
    app = FastAPI()
    app.include_router(support.router)
    app.dependency_overrides[require_user] = lambda: principal
    app.dependency_overrides[support.get_agent_nick] = lambda: nick
    return TestClient(app).get("/support/tickets"), nick.queries


def test_a_person_lists_only_the_tickets_they_raised(monkeypatch):
    response, queries = _tickets(ALICE, "Buyer", monkeypatch)
    assert response.status_code == 200
    (sql, params), = queries
    assert "lower(user_email) = %s" in sql
    assert params[0] == "alice@example.com"


def test_an_admin_lists_every_ticket(monkeypatch):
    response, queries = _tickets(ALICE, "Admin", monkeypatch)
    assert response.status_code == 200
    (sql, _), = queries
    assert "user_email) =" not in sql


def test_a_sign_in_with_no_email_owns_no_tickets(monkeypatch):
    nobody = Principal(subject="sub-x")
    response, queries = _tickets(nobody, "Buyer", monkeypatch)
    assert response.json() == {"count": 0, "tickets": []}
    assert queries == []


def test_with_sign_in_off_tickets_list_as_before(monkeypatch):
    response, queries = _tickets(None, "Viewer", monkeypatch)
    assert response.status_code == 200
    (sql, _), = queries
    assert "user_email) =" not in sql


@pytest.mark.parametrize("typed", [None, "someone.else@example.com"])
def test_a_ticket_is_raised_in_the_signed_in_persons_name(typed):
    # The owner filter above matches on the address stored with the ticket, so
    # that address has to be the token's: a typed one could claim another
    # person's tickets, and an absent one would hide the person's own.
    assert support._ticket_email(ALICE, typed) == "alice@example.com"
    assert support._ticket_email(None, typed) == typed
