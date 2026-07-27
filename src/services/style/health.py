"""Is each bound mailbox still readable?

A permission grant revoked in the customer's tenant does not tell us so. Nothing arrives;
the next read simply fails. Without a check the platform would keep trying — and worse,
would keep drafting against a profile derived from a mailbox it is no longer allowed to
read, producing output that looks personalised and is built on withdrawn consent.

Three outcomes:

* **OK** — a read succeeded.
* **DEGRADED** — the provider did not answer. Probably transient; retry next cycle.
* **REVOKED** — the provider answered, and said no. That is a decision, not an outage.

The distinction matters because they demand opposite responses. A timeout should be
retried; a 403 should stop the platform reading and make every subsequent draft say so.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from services.style.graph_source import (
    GraphExemplarSource,
    MailboxAccessDenied,
    MailboxUnreachable,
)
from services.style.mailbox import (
    HEALTH_DEGRADED,
    HEALTH_OK,
    HEALTH_REVOKED,
    MailboxBinding,
    MailboxBindingRepository,
    PROVIDER_GRAPH,
)

logger = logging.getLogger(__name__)


def _default_probe(binding: MailboxBinding) -> str:
    """Read one message from the bound mailbox and report the outcome."""

    if binding.provider != PROVIDER_GRAPH:
        # IMAP and Gmail bindings have no reader yet. Reporting DEGRADED rather than OK is
        # the honest answer: nothing has confirmed this binding works.
        logger.info(
            "No health probe for provider %r (binding %s); reporting DEGRADED",
            binding.provider, binding.binding_id,
        )
        return HEALTH_DEGRADED

    source = GraphExemplarSource(binding)
    try:
        source.list_folders()
        return HEALTH_OK
    except MailboxAccessDenied:
        # The provider answered and refused. Consent has been withdrawn.
        return HEALTH_REVOKED
    except MailboxUnreachable:
        # Nobody answered. That is not the same thing, and must not be treated as one:
        # marking a binding REVOKED because of a network blip would strip a customer's
        # personalisation over a dropped packet.
        return HEALTH_DEGRADED


def check_binding(
    binding: MailboxBinding,
    *,
    repo: Optional[MailboxBindingRepository] = None,
    probe: Optional[Callable[[MailboxBinding], str]] = None,
) -> str:
    """Probe one binding and record the result. Returns the new health state."""

    repo = repo or MailboxBindingRepository()
    probe = probe or _default_probe

    try:
        state = probe(binding)
    except Exception:
        logger.exception("Health probe raised for binding %s", binding.binding_id)
        state = HEALTH_DEGRADED

    repo.record_health(binding.binding_id, state)
    if state == HEALTH_REVOKED:
        logger.warning(
            "Mailbox binding %s (%s) is REVOKED — drafting for %s now degrades to the "
            "platform baseline with a visible flag",
            binding.binding_id, binding.mailbox_address, binding.user_ref,
        )
    return state


def check_all_bindings(
    *,
    repo: Optional[MailboxBindingRepository] = None,
    probe: Optional[Callable[[MailboxBinding], str]] = None,
) -> Dict[str, int]:
    """Probe every active binding. Returns a tally by state."""

    repo = repo or MailboxBindingRepository()
    bindings: List[MailboxBinding] = repo.list_active()

    tally = {HEALTH_OK: 0, HEALTH_DEGRADED: 0, HEALTH_REVOKED: 0}
    for binding in bindings:
        state = check_binding(binding, repo=repo, probe=probe)
        tally[state] = tally.get(state, 0) + 1

    if bindings:
        logger.info("Mailbox health check: %s", tally)

    try:
        from services.agent_actions import record_action

        record_action(
            phase="style", action_type="mailbox_health", agent="style_health",
            status="ok", summary=f"checked {len(bindings)} mailbox binding(s)",
            details=tally,
        )
    except Exception:  # pragma: no cover - audit must never break the caller
        logger.debug("mailbox health audit write failed", exc_info=True)

    return tally
