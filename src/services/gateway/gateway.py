"""The gateway: a fixed catalogue of typed business questions, called with a caller context, each call audited.

There is no free-form query function and none can be added by configuration: a function exists only if it is in the
catalogue (``functions.CATALOGUE``) AND declared in ``config/gateway/functions.yaml``.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from .audit import AuditSink
from .context import CONTEXT_KEYS, CallerContext
from .envelope import GatewayError, GatewayFault, GatewayResult, Raw

logger = logging.getLogger(__name__)


@dataclass
class FunctionSpec:
    name: str
    params: Type[BaseModel]
    fn: Callable[[CallerContext, BaseModel, Any], Raw]
    max_rows: int = 25
    kind: str = "read"            # "read" | "write"
    version: int = 1
    description: str = ""


@dataclass
class Backends:
    """What the functions reach the stores through. Only this module's callers hold them."""
    connect: Optional[Callable[[], Any]] = None                 # context-managed DB connection
    search: Optional[Callable[..., Any]] = None                 # (query, top_k, filters: dict) -> hits
    extra: Dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Gateway:
    def __init__(self, functions: Dict[str, FunctionSpec], *, audit: Optional[AuditSink] = None,
                 backends: Optional[Backends] = None) -> None:
        self.functions = functions
        self.audit = audit or AuditSink()
        self.backends = backends

    # ------------------------------------------------------------------------------------------------------------
    def call(self, ctx: CallerContext, name: str, params: Optional[Dict[str, Any]] = None) -> GatewayResult:
        started = time.perf_counter()
        params = dict(params or {})
        call_id = uuid.uuid4().hex
        spec = self.functions.get(name)
        scope_violation = False

        def finish(res: GatewayResult) -> GatewayResult:
            res.function, res.call_id, res.as_of = name, call_id, res.as_of or _now()
            res.latency_ms = int((time.perf_counter() - started) * 1000)
            self._audit(ctx, name, params, res, scope_violation, spec.version if spec else 0)
            return res

        def fail(code: str, message: str, *, retryable: bool = False, candidates: Optional[List[Any]] = None) -> GatewayResult:
            return finish(GatewayResult(function=name, ok=False, data=[], error=GatewayError(code, message, retryable, list(candidates or []))))

        if spec is None:
            return fail("UNSUPPORTED", f"no gateway function named {name!r}; logged to the function backlog")

        forbidden = sorted(k for k in params if k in CONTEXT_KEYS)
        if forbidden:
            scope_violation = True
            logger.warning("gateway: %s tried to supply caller context %s to %s", ctx.agent or "?", forbidden, name)
            return fail("SCHEMA_INVALID", f"parameters {forbidden} are set by the system and cannot be supplied")
        try:
            parsed = spec.params(**params)
        except ValidationError as exc:
            return fail("SCHEMA_INVALID", "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())[:400])

        try:
            raw = spec.fn(ctx, parsed, self.backends)
        except GatewayFault as fault:
            return finish(GatewayResult(function=name, ok=False, data=[], error=fault.error))
        except Exception as exc:  # noqa: BLE001 - a failed store is an error the caller can see, never a silent empty
            logger.exception("gateway function %s failed", name)
            return fail("UNAVAILABLE", f"{type(exc).__name__}: the data source could not be read", retryable=True)

        data, truncated, total = raw.data, bool(raw.truncated), raw.total_available
        if isinstance(data, list) and len(data) > spec.max_rows:
            total = total if total is not None else len(data)
            data, truncated = data[:spec.max_rows], True
        elif total is not None and isinstance(data, list) and total > len(data):
            truncated = True
        if data and not raw.provenance:
            return fail("NO_PROVENANCE", "the function returned data with no source; it is withheld")
        return finish(GatewayResult(function=name, version=spec.version, ok=True, data=data, provenance=list(raw.provenance),
                                    as_of=raw.as_of or "", truncated=truncated, total_available=total))

    # ------------------------------------------------------------------------------------------------------------
    def reject(self, ctx: CallerContext, name: str, params: Dict[str, Any], error: GatewayError) -> GatewayResult:
        """A request refused before it reached a function (allowlist, budget): still audited."""
        res = GatewayResult(function=name, ok=False, data=[], error=error, call_id=uuid.uuid4().hex, as_of=_now())
        self._audit(ctx, name, params, res, False, 0)
        return res

    def _audit(self, ctx: CallerContext, name: str, params: Dict[str, Any], res: GatewayResult, scope_violation: bool, version: int) -> None:
        ids: List[str] = []
        for p in res.provenance:
            ids.extend(str(i) for i in p.get("record_ids", []) or [])
        self.audit.record({
            "call_id": res.call_id, "ts": res.as_of, "caller_subject": ctx.principal_subject, "role": ctx.role,
            "tenant_id": ctx.tenant_id, "agent": ctx.agent, "request_id": ctx.request_id, "function": name,
            "function_version": version, "params": params, "result_record_ids": ids,
            "row_count": len(res.data) if isinstance(res.data, list) else (1 if res.data else 0),
            "truncated": res.truncated, "ok": res.ok, "error_code": res.error.code if res.error else None,
            "latency_ms": res.latency_ms, "scope_violation": scope_violation,
        })
