"""What the orchestrator and agents call. The only door into the prompt engine.

    from services.prompt_engine import client
    out = client.render("claims-triage", vars={...}, allowed_tools=[...],
                        obligations=[...], trace=client.trace(run_id, "triage"))
    out["instructions"], out["manifest"]["hash"]

The engine itself never retries. This does, on the caller's side of the line:

  * a failure that can pass (the database not answering) is tried again, with
    a growing wait, until it succeeds or the time allowed runs out. It never
    falls back to other wording: when the time runs out it raises
    PromptUnavailable and the step does not run.
  * a refusal (a missing value, an unknown obligation, nothing in use) is
    raised at once. Trying it again would never complete.

Nothing is cached here beyond what the registry already remembers (saved
versions, which cannot change). The pointer is read on every call, so a
rollback is seen by the very next render.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Optional

from services.prompt_engine import db, registry, render as _render, templates

logger = logging.getLogger(__name__)

#: Raised as they are, never retried.
REFUSALS = (templates.TemplateError, registry.RegistryError)

_FIRST_WAIT = 0.1
_LONGEST_WAIT = 2.0


class PromptUnavailable(Exception):
    """The prompt could not be rendered in the time allowed. The step must not
    run on other wording."""

    code = "RENDER_UNAVAILABLE"


def trace(run_id: Optional[str], step_id: str, *, attempt: int = 1,
          decision_id: Optional[str] = None) -> Dict[str, Any]:
    """The ids a render is recorded under. A caller with no run of its own gets
    a new run id, so the render can still be found again."""
    return {"run_id": run_id or f"run-{uuid.uuid4().hex}", "step_id": step_id,
            "attempt": int(attempt), "decision_id": decision_id}


def _allowed_seconds() -> float:
    from src.services.governed_limits import limit

    return float(limit("prompt_engine", "render_retry_seconds", cast=float))


def _until_complete(what: str, call: Callable[[], Any], seconds: Optional[float],
                    *, sleep: Callable[[float], None] = time.sleep,
                    clock: Callable[[], float] = time.monotonic) -> Any:
    wait = _FIRST_WAIT
    started = clock()
    allowed: Optional[float] = seconds
    tries = 0
    while True:
        tries += 1
        try:
            return call()
        except REFUSALS:
            raise
        except Exception as exc:  # noqa: BLE001 - anything else may pass on a retry
            db.close()
            if allowed is None:
                # Read only once something has failed: the usual call never
                # needs it, and must not pay a policy lookup to find out.
                allowed = _allowed_seconds()
            spent = clock() - started
            if spent + wait > allowed:
                logger.error("%s did not complete in %.0fs after %d tries (%s)",
                             what, allowed, tries, type(exc).__name__)
                raise PromptUnavailable(
                    f"{what} did not complete in {allowed:.0f}s") from exc
            logger.warning("%s failed (%s); trying again in %.1fs", what,
                           type(exc).__name__, wait)
            sleep(wait)
            wait = min(wait * 2, _LONGEST_WAIT)


def in_use(prompt_id: str, *, pointer: str = registry.LIVE,
           seconds: Optional[float] = None, **timing: Any) -> Optional[Dict[str, Any]]:
    """The version in use, or None if the registry has no such prompt in use.

    None also in a process with no real database (the unit suite).
    """
    if not db.available():
        return None

    def call() -> Optional[Dict[str, Any]]:
        try:
            return registry.resolve(prompt_id, pointer)
        except (registry.PointerUnresolved, registry.PromptNotFound):
            return None

    return _until_complete(f"reading prompt {prompt_id}", call, seconds, **timing)


def render(prompt_id: str, *, trace: Dict[str, Any], vars: Optional[Dict[str, Any]] = None,
           allowed_tools: Iterable[str] = (), obligations: Iterable[str] = (),
           pointer: Optional[str] = None, version: Optional[int] = None,
           experiment_key: Optional[str] = None, seconds: Optional[float] = None,
           **timing: Any) -> Dict[str, Any]:
    """Render one prompt for one step, retrying until complete or out of time."""
    request = {
        "prompt_id": prompt_id, "pointer": pointer, "version": version,
        "experiment_key": experiment_key, "vars": dict(vars or {}),
        "allowed_tools": list(allowed_tools), "obligations": list(obligations),
        "trace": dict(trace),
    }
    return _until_complete(f"rendering prompt {prompt_id}",
                           lambda: _render.render(request), seconds, **timing)
