"""P8 phase 2, the last group: supplier review, vendor onboarding, support, email.

None of these eight endpoints writes an actor column, so there is nothing to
attribute -- but each must still learn who is calling, so that "an agent may
never exceed the human it acts for" has a human to read. Shown the same way as
the other two groups: require_user answers 418, which only a handler that
declares it can produce.

The last test here is the one that makes P8 stick. It parses every router and
fails if any write endpoint anywhere does not take the principal -- so the next
router added without it is caught here, not found by the next audit.
"""

from __future__ import annotations

import ast
import io
import pathlib

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.auth import require_user


class _Tripwire:
    def __getattr__(self, name):
        raise RuntimeError(f"handler ran without resolving the caller (touched .{name})")

    def __call__(self, *a, **k):
        raise RuntimeError("handler ran without resolving the caller")


def _refusing_app(router):
    def _refuse():
        raise HTTPException(status_code=418, detail="principal resolved")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = _refuse
    app.state.orchestrator = _Tripwire()
    app.state.agent_nick = _Tripwire()
    return TestClient(app, raise_server_exceptions=False)


def _vendors_router(monkeypatch):
    from api.routers import vendors
    from src.services.extraction_v2.template_store import InMemoryTemplateStore

    monkeypatch.setattr(vendors, "_run_pipeline", _Tripwire())
    return vendors.build_router(InMemoryTemplateStore())


_ENDPOINTS = [
    # (router module, method, path, request kwargs)
    ("supplier_review", "POST", "/suppliers/reviews/sweep", {}),
    ("support", "POST", "/support/contact", {"json": {"message": "help"}}),
    ("support", "POST", "/support/contact/stream", {"json": {"message": "help"}}),
    ("support", "POST", "/support/REF-1/confirm", {"json": {"resolved": True}}),
    ("email", "POST", "/email/emailwatcher", {"json": {"workflow_id": "WF-1"}}),
    ("vendors", "POST", "/vendors/onboard/upload",
     {"files": {"file": ("q.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
      "data": {"doc_type": "Quote"}}),
    ("vendors", "POST", "/vendors/onboard/S-1/correct", {"json": {"field": "total", "value": 1}}),
    ("vendors", "POST", "/vendors/onboard/S-1/save", {"json": {}}),
]


@pytest.mark.parametrize("module_name,method,path,kwargs", _ENDPOINTS,
                         ids=[f"{m} {p}" for _, m, p, _ in _ENDPOINTS])
def test_the_handler_resolves_the_caller(monkeypatch, module_name, method, path, kwargs):
    if module_name == "vendors":
        router = _vendors_router(monkeypatch)
    else:
        module = __import__(f"api.routers.{module_name}", fromlist=["*"])
        if module_name == "supplier_review":
            monkeypatch.setattr(module, "get_conn", _Tripwire())
        if module_name == "email":
            # Otherwise a handler that never asked falls through to the real
            # watcher, which polls a mailbox.
            monkeypatch.setattr(module, "run_email_watcher_for_workflow", _Tripwire())
        if module_name == "support":
            import services.support_agent as sa
            monkeypatch.setattr(sa, "SupportAgent", _Tripwire())
        router = module.router

    response = _refusing_app(router).request(method, path, **kwargs)

    assert response.status_code == 418, (
        f"{method} {path} never asked who the caller is "
        f"({response.status_code}: {response.text[:160]})")


# ---------------------------------------------------------------------------
# the guard: no write endpoint anywhere may skip the principal
# ---------------------------------------------------------------------------
_ROUTERS = pathlib.Path(__file__).resolve().parents[2] / "src" / "api" / "routers"
_WRITE = {"post", "put", "patch", "delete"}


def _write_endpoints_without_the_principal() -> list[str]:
    """The same parse as scripts/p8_endpoint_scan.py, so the two cannot disagree."""
    missing = []
    for path in sorted(_ROUTERS.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            writes = [d for d in node.decorator_list
                      if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                      and d.func.attr.lower() in _WRITE]
            if not writes:
                continue
            defaults = list(node.args.defaults) + list(node.args.kw_defaults)
            takes_principal = any(
                isinstance(d, ast.Call)
                and getattr(d.func, "id", getattr(d.func, "attr", "")) == "Depends"
                and d.args and getattr(d.args[0], "id", "") == "require_user"
                for d in defaults
            )
            if not takes_principal:
                missing.append(f"{path.name}:{node.lineno} {node.name}")
    return missing


def test_the_scan_sees_the_routers():
    """A guard that parses nothing passes forever. Prove it is looking."""
    assert len(list(_ROUTERS.glob("*.py"))) > 20, _ROUTERS
    tree = ast.parse((_ROUTERS / "promotion.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "post_approve" for n in ast.walk(tree))


def test_every_write_endpoint_resolves_the_caller():
    missing = _write_endpoints_without_the_principal()
    assert not missing, (
        f"{len(missing)} write endpoint(s) never ask who the caller is -- add "
        f"`principal=Depends(require_user)` and attribute any actor to its subject:\n  "
        + "\n  ".join(missing))
