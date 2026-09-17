"""A refused lifecycle move comes back as 409 Conflict with the reason.

Before the lifecycle guard, POST /opportunities/{id}/stage accepted any stage from any
stage. Now the database refuses a backward move; the endpoint must say so as a conflict
the caller can act on, not as a 500 that reads like an outage.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import require_user


def test_a_backward_stage_move_is_a_409_with_the_reason(monkeypatch):
    import api.routers.opportunities as module
    from src.services.lifecycle import IllegalTransition

    def _refuse(*a, **k):
        raise IllegalTransition("opportunity O-1 is realised; it cannot move to identified")

    monkeypatch.setattr(module, "set_stage", _refuse)
    app = FastAPI()
    app.include_router(module.router)
    app.dependency_overrides[require_user] = lambda: object()

    resp = TestClient(app).post("/opportunities/O-1/stage", json={"stage": "identified"})

    assert resp.status_code == 409
    assert "realised" in resp.json()["detail"]
