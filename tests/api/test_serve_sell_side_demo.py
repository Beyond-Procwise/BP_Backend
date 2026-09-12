"""The sell-side demo server must configure authentication before mounting
its routers, exactly as api.main does at import -- otherwise require_user
answers every request with 503 "authentication is not configured" and the
HTTP walk this script exists for cannot get past the first call.

This module's own conftest fixture (tests/api/conftest.py) forces api.auth
into "off" mode for every test in this directory, which would make a bare
``auth.auth_mode() != "unconfigured"`` assertion pass whether or not the
demo script calls configure() -- "off" is not "unconfigured" either. So this
test resets the mode to "unconfigured" itself, reloads the demo module, and
only then asserts: the assertion is false unless the demo's own import-time
configure() call put it back.
"""
from __future__ import annotations

import importlib

from api import auth


def test_the_demo_app_configures_authentication(monkeypatch):
    monkeypatch.setattr(auth, "_mode", "unconfigured")
    monkeypatch.setattr(auth, "_verifier", None)

    demo = importlib.import_module("scripts.serve_sell_side_demo")
    demo = importlib.reload(demo)

    assert auth.auth_mode() != "unconfigured"

    # The installed FastAPI keeps an included router as an opaque wrapper
    # rather than flattening its routes into app.routes (see the note in
    # test_every_router_is_authenticated.py), so ask the app's own schema
    # instead of walking app.routes directly.
    paths = set((demo.app.openapi().get("paths") or {}).keys())
    assert any(p.startswith("/catalog/") for p in paths)
    assert any(p.startswith("/sales/") for p in paths)
