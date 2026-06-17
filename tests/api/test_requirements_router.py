from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.requirements as rq


def _client(monkeypatch, run_result=None, get_result=None, list_result=None):
    monkeypatch.setattr(rq, "_run_requirements_turn",
                        lambda app_state, payload: run_result or {"complete": False, "next_question": "?"})
    monkeypatch.setattr(rq.requirement_service, "get_requirement", lambda rid: get_result)
    monkeypatch.setattr(rq.requirement_service, "list_requirements",
                        lambda limit=50, offset=0: list_result or [])
    app = FastAPI()
    app.include_router(rq.router)
    return TestClient(app)


def test_message_returns_next_question(monkeypatch):
    client = _client(monkeypatch, run_result={
        "complete": False, "next_question": "How many?", "session_id": "S1"})
    resp = client.post("/requirements/message", json={"message": "I need laptops"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["next_question"] == "How many?"
    assert isinstance(body["events"], list) and body["events"]


def test_get_requirement_404(monkeypatch):
    client = _client(monkeypatch, get_result=None)
    resp = client.get("/requirements/REQ-missing")
    assert resp.status_code == 404


def test_get_requirement_found(monkeypatch):
    client = _client(monkeypatch, get_result={"requirement_id": "REQ-1", "status": "complete"})
    resp = client.get("/requirements/REQ-1")
    assert resp.status_code == 200
    assert resp.json()["requirement_id"] == "REQ-1"


def test_list_requirements(monkeypatch):
    client = _client(monkeypatch, list_result=[{"requirement_id": "REQ-1"}])
    resp = client.get("/requirements")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
