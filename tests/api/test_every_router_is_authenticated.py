"""Every HTTP endpoint must refuse an unauthenticated caller.

Thirty of thirty-four routers answered anyone who could reach the port. The fix
was to mount them all with ``require_user`` — but a fix applied at call sites is
a fix the next router forgets, which is how those thirty came to exist. This
test is the part that makes it stick: it asks the app what it serves and calls
every endpoint with no credentials.

Asserted behaviourally rather than by inspecting dependency objects. The
installed FastAPI keeps included routers as opaque wrappers rather than
flattening them into ``app.routes``, so an introspection-based check found four
routes out of two hundred and passed while checking almost nothing. What a
caller actually gets back cannot be wrong in that way.

No handler runs while this passes: a 401 is returned before the route body is
reached. A handler only executes if the endpoint is already unauthenticated,
which is the failure this test exists to report.

A route that genuinely must be public goes in ``_PUBLIC`` with a reason. That is
a deliberate, reviewable act; forgetting is not.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


_PUBLIC: dict[str, str] = {
    "/": "service banner; no data",
    "/health": "liveness probe — a probe that needs a token cannot report that "
               "auth is misconfigured, which is exactly when you need it",
    "/openapi.json": "schema document served by FastAPI itself",
    "/docs": "FastAPI's own docs UI",
    "/docs/oauth2-redirect": "FastAPI's own docs UI",
    "/redoc": "FastAPI's own docs UI",
}

# A browser cannot set headers on a WebSocket upgrade, so ws.py authenticates on
# a token= query parameter instead. Excluded by protocol, not by exemption.
_WEBSOCKET_PREFIX = "/ws/"

# Returned when the caller is not identified. 403 is included because a
# dependency may reject before the 401 is composed; either is a refusal.
_REFUSALS = {401, 403}


@pytest.fixture(scope="module", autouse=True)
def _enforce_auth():
    """Force enforce mode for this module, whatever the sandbox's .env says.

    This guard proves the ROUTERS are mounted behind ``require_user`` — a claim
    that is only observable while enforcement is on. The sandbox toggles
    ASK_AUTH_MODE to "off" for local demos (the UI runs with the dev auth
    bypass), and reading that toggle here made the mounting guard report the
    demo configuration instead of the mounting. The verifier below refuses
    every token; no request in this module sends one anyway.
    """
    # api.main calls auth.configure() at import time, which would overwrite
    # whatever this fixture sets. Import it FIRST so that configure() has
    # already run (imports are cached — it runs once per process), then force
    # the mode for the module.
    import api.main  # noqa: F401
    from api import auth as _auth

    class _RefuseAll:
        def verify(self, token):
            raise _auth.AuthError("no tokens are valid in this test")

    previous = (_auth._mode, _auth._verifier)
    _auth._mode, _auth._verifier = "enforce", _RefuseAll()
    yield
    _auth._mode, _auth._verifier = previous


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _endpoints() -> list[tuple[str, str]]:
    """(method, path) for everything the app serves, from its own schema."""
    from api.main import app

    out: list[tuple[str, str]] = []
    for path, ops in (app.openapi().get("paths") or {}).items():
        if path in _PUBLIC or path.startswith(_WEBSOCKET_PREFIX):
            continue
        for method in ops:
            if method.lower() in ("get", "post", "put", "patch", "delete"):
                out.append((method.upper(), path))
    return out


def _fill(path: str) -> str:
    """Substitute a placeholder for each {param}. The value never matters: the
    request is refused before any handler reads it."""
    import re
    return re.sub(r"\{[^}]+\}", "auth-probe", path)


def test_there_are_endpoints_to_check():
    """Guards the guard. A collection bug that found nothing would make the
    assertion below vacuously true — which is exactly what happened to the
    first version of this test."""
    endpoints = _endpoints()
    assert len(endpoints) > 50, (
        f"only {len(endpoints)} endpoints collected from the OpenAPI schema — "
        f"the app did not mount properly, so the authentication assertion "
        f"below would pass without checking anything"
    )


def test_every_endpoint_refuses_an_unauthenticated_caller(client):
    served = []
    for method, path in _endpoints():
        response = client.request(method, _fill(path))
        if response.status_code not in _REFUSALS:
            served.append(f"{method} {path} -> {response.status_code}")

    assert not served, (
        f"{len(served)} endpoint(s) answered without authentication:\n  "
        + "\n  ".join(sorted(served))
        + "\n\nAdd the router to _AUTHENTICATED_ROUTERS in api/main.py, or add "
          "the path to _PUBLIC in this test with a reason."
    )


def test_the_public_list_has_not_grown_silently():
    """_PUBLIC is the exemption surface. It should stay small enough that any
    growth is noticed in review rather than discovered later."""
    assert len(_PUBLIC) <= 8, (
        f"_PUBLIC now exempts {len(_PUBLIC)} paths. It is meant to hold health "
        f"and schema routes, not to absorb endpoints that were awkward to "
        f"authenticate."
    )


@pytest.mark.parametrize("method,path", [
    ("GET", "/benchmark/by-deal/{deal_id}"),
    ("POST", "/agents/reason"),
    ("POST", "/suppliers/research/batch"),
    ("GET", "/opportunities"),
    ("GET", "/analysis"),
])
def test_the_endpoints_named_in_the_egress_audit_are_covered(client, method, path):
    """Named explicitly so a refactor that drops one is obvious in the diff.

    These return corpus-wide commercial data, or — /agents/reason — hand a model
    a task with every registered agent exposed to it as a callable tool, and
    /suppliers/research/batch reaches the public internet.
    """
    if (method, path) not in _endpoints():
        pytest.skip(f"{method} {path} is no longer served under that name")
    assert client.request(method, _fill(path)).status_code in _REFUSALS
