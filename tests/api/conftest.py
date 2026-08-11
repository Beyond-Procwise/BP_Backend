"""Endpoint tests run with authentication switched off.

Every router is now mounted with ``require_user``, so a TestClient that sends no
Authorization header gets 401 from every endpoint. The suites in this directory
test what the handlers DO — their payloads, their status codes, their conflict
handling — and none of them is about authentication. Making each one mint a
Cognito token would test the token library, not the handler.

So auth is put in ``off`` mode for this directory. Two things keep that from
quietly becoming a hole:

  * ``test_every_router_is_authenticated.py`` opts back IN to ``enforce`` for
    itself, and is the test that proves the mounting works.
  * The mode is restored afterwards, so a test that leaves it off cannot affect
    another module.

If a handler ever needs the caller's identity, it should take a ``principal``
parameter and the test should pass one — not read it from the auth module.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _auth_off(request):
    """Put api.auth in 'off' mode for the duration of each test.

    Skipped for the module that tests authentication itself, which configures
    the mode it needs.
    """
    if request.module.__name__.endswith("test_every_router_is_authenticated"):
        yield
        return

    from api import auth as _auth

    previous_mode = _auth._mode
    previous_verifier = _auth._verifier
    _auth._mode, _auth._verifier = "off", None
    try:
        yield
    finally:
        _auth._mode, _auth._verifier = previous_mode, previous_verifier
