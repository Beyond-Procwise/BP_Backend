"""The §7 events, in one place, so a stage cannot quietly stop emitting one.

Every event carries the run id — which is the Fact Pack id, because a run and
the snapshot it was computed from are the same thing here — and, where one
exists, the pack hash. That pairing is what makes a released artefact traceable
back to the figures it was built from.

TWO WRITERS, AND THE DIFFERENCE MATTERS

``record_action`` is best-effort and swallows failures; ``record_action_or_fail``
raises. The rule this module applies is the one ``agent_actions`` already sets:
reads and computes use the forgiving writer, because a transient database blip
should not halt a report; **irreversible acts use the writer that raises**, because
an action whose audit cannot be written must not happen. Releasing a report to
someone outside the organisation is irreversible. Building a Fact Pack is not.

A NOTE ON TOKEN COUNTS

§7 asks ``report.composed`` to carry them. It does not, and this is deliberate
rather than forgotten: ``services/ollama_client.ollama_generate`` reads the
response body and returns ``body["response"]`` alone — ``eval_count`` and
``prompt_eval_count`` are discarded inside the function. Surfacing them would
mean changing the return contract of the one client that extraction, the insight
writer, the style compiler and every agent share. That is a deliberate change
for someone to make on purpose, not a side effect of building a reporting
feature, so what is recorded instead is the model, the prompt version and the
prompt hash — enough to identify the call, short of costing it.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

PHASE = "reporting"

# The §7 vocabulary. Spelled once so a typo is a NameError here rather than an
# event that silently never matches anything downstream.
SCOPE_RESOLVED = "report.scope_resolved"
FACTPACK_BUILT = "report.factpack_built"
STYLEBRIEF_RESOLVED = "report.stylebrief_resolved"
COMPOSED = "report.composed"
RENDERED = "report.rendered"
POSTCHECK_PASSED = "report.postcheck_passed"
POSTCHECK_FAILED = "report.postcheck_failed"
APPROVAL_REQUESTED = "report.approval_requested"
APPROVAL_GRANTED = "report.approval_granted"
APPROVAL_DENIED = "report.approval_denied"
RELEASED = "report.released"
STYLE_CHANGE_PROPOSED = "report.style_change_proposed"
STYLE_CHANGE_PROMOTED = "report.style_change_promoted"
# Not in the brief's list. A run that crashed, or was stranded by a restart,
# otherwise leaves a trail that just stops -- indistinguishable from one still
# going. This closes it.
RUN_FAILED = "report.run_failed"
# A person decided a blocked or failed report needs no further action. Not in
# the brief's list either; without it an item leaving the Action Centre would
# leave no trace of who took it off the list, or why.
DISMISSED = "report.dismissed"
# A person saved an edit of a released report as a new version (the light editor,
# 2026-09-24). Irreversible: what the report says has changed, and it is written
# inside the save's transaction so an edit that cannot be audited does not happen.
EDITED = "report.edited"

#: Events that must not happen unless they can be audited.
IRREVERSIBLE = frozenset({
    RELEASED, APPROVAL_GRANTED, APPROVAL_DENIED, STYLE_CHANGE_PROMOTED, EDITED,
})


# WHICH RUN AN EVENT BELONGS TO
#
# The run id is the Fact Pack id, derived from report, scope and as-of, so two
# runs of one quarter on one day share it (by design: DoD3 rests on it). A job
# sets this context around its run; ``emit`` stamps job_id and requested_by onto
# every event written inside it, from whichever module writes it, so each run's
# trail can be separated from another's. The entitlement decision rides here
# too, and the pipeline repeats it on report.scope_resolved only.
_RUN_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "rga_run_context", default={})
_STAMPED = ("job_id", "requested_by")


@contextmanager
def run_context(**values: Any) -> Iterator[None]:
    token = _RUN_CONTEXT.set({k: v for k, v in values.items() if v is not None})
    try:
        yield
    finally:
        _RUN_CONTEXT.reset(token)


def current_context() -> Dict[str, Any]:
    return dict(_RUN_CONTEXT.get())


def prompt_hash(prompt: str) -> str:
    """The sha256 of the exact text a model was given.

    Recorded rather than the prompt itself: a prompt carries the facts, and the
    audit spine is not where a copy of the figures belongs. The hash is enough
    to prove two runs asked the same question.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def emit(
    action_type: str,
    *,
    run_id: str,
    agent: str,
    status: str = "ok",
    summary: str = "",
    pack_hash: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    writer: Any = None,
) -> None:
    """Write one §7 event.

    ``writer`` is injectable so tests can assert on the events without a
    database. Left alone, an irreversible event uses the writer that raises and
    everything else uses the one that does not.
    """
    payload: Dict[str, Any] = {"run_id": run_id}
    context = _RUN_CONTEXT.get()
    payload.update({k: context[k] for k in _STAMPED if k in context})
    if pack_hash:
        payload["pack_hash"] = pack_hash
    payload.update(details or {})

    if writer is None:
        from src.services.agent_actions import record_action, record_action_or_fail

        writer = record_action_or_fail if action_type in IRREVERSIBLE else record_action

    try:
        writer(
            phase=PHASE, action_type=action_type, agent=agent,
            trace_id=run_id, status=status, summary=summary or action_type,
            details=payload,
        )
    except Exception:
        # An irreversible event's writer raises on purpose — let it out, so the
        # caller does not proceed with an unaudited release.
        if action_type in IRREVERSIBLE:
            raise
        logger.exception("rga: could not write %s", action_type)
