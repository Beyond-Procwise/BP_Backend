"""Bind the verified tenant to each request.

Pure ASGI (not BaseHTTPMiddleware), so the ContextVar set here is the one the endpoint sees, on the
event loop and in the threadpool alike. It only READS the token to find the tenant: refusing a bad or
missing token stays with ``require_user``. Any failure here means "no tenant", never a guessed one.
"""
from __future__ import annotations

import logging

from api import auth
from services import tenant_context

logger = logging.getLogger(__name__)


class TenantContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        token = None
        try:
            tenant = self._tenant_of(scope)
            token = tenant_context.set_for_request(tenant)
        except Exception:  # noqa: BLE001 - no tenant, not a failed request
            logger.warning("tenant: could not read the tenant from the token", exc_info=True)
        try:
            await self.app(scope, receive, send)
        finally:
            if token is not None:
                tenant_context.reset(token)

    @staticmethod
    def _tenant_of(scope):
        header = ""
        for k, v in scope.get("headers") or []:
            if k == b"authorization":
                header = v.decode("latin-1")
                break
        scheme, _, raw = header.strip().partition(" ")
        if scheme.lower() != "bearer" or not raw.strip():
            return None
        try:
            principal = auth.principal_from_token(raw.strip())
        except Exception:  # noqa: BLE001 - require_user will refuse it; here it just has no tenant
            return None
        return tenant_context.from_principal(principal)
