"""Mode C2: exemplars read from the mailbox at drafting time.

Every other mode resolves exemplars from a table. C2 does not — it reads the customer's
mailbox as the draft is being written, uses what it finds, and keeps none of it. That is
the whole proposition: the platform can write in your voice from your actual recent mail
without holding a copy of any of it.

Three rules make that claim true rather than aspirational:

* **Nothing fetched is written to a table.** This module returns exemplar records with no
  ``exemplar_id``, because there is no row. A draft cites message ids instead.
* **The cache is memory-only and expires in five minutes**, flushed on unbind or
  revocation. See ``mailbox_cache``.
* **A slow mailbox degrades the draft, it does not delay it.** The circuit breaker below
  gives up rather than leaving someone watching a spinner while Graph is having a bad
  afternoon — and says so, at fallback level 3.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from services.style import mailbox_cache
from services.style.exemplars import ORIGIN_CUSTOMER_RETAINED, ExemplarRecord
from services.style.graph_source import MailboxAccessDenied, MailboxUnreachable
from services.style.mailbox import MailboxBinding
from services.style.redaction import redact

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 3


@dataclass
class MailboxFetchOutcome:
    """What the mailbox read produced, and whether it worked."""

    exemplars: List[ExemplarRecord]
    message_ids: List[str]
    from_cache: bool = False
    unreachable: bool = False
    reason: Optional[str] = None

    @property
    def usable(self) -> bool:
        return bool(self.exemplars)


def message_set_hash(message_ids: List[str]) -> Optional[str]:
    """Fingerprint of the message set, over sorted ids.

    Under C2 this is the only durable record of which emails shaped a draft — the bodies
    are gone by the time anyone asks. Sorted so the fingerprint identifies the set rather
    than the order the provider happened to return it in.
    """

    if not message_ids:
        return None
    joined = ",".join(sorted(message_ids))
    return hashlib.sha256(joined.encode()).hexdigest()


def fetch_live_exemplars(
    binding: MailboxBinding,
    source: Any,
    *,
    user_ref: str,
    intent: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    ttl_seconds: int = mailbox_cache.DEFAULT_TTL_SECONDS,
    use_cache: bool = True,
) -> MailboxFetchOutcome:
    """Read exemplars from the bound mailbox for one draft.

    The circuit breaker is deliberately blunt: any provider failure returns an unreachable
    outcome rather than raising. Drafting then falls to the baseline with a visible reason.
    Someone writing an email should never be blocked because a mail API is slow, and should
    never be told a generic draft is theirs.
    """

    if not binding.usable:
        return MailboxFetchOutcome(
            exemplars=[], message_ids=[], unreachable=True,
            reason="the mailbox binding is inactive or revoked",
        )

    if use_cache:
        cached = mailbox_cache.get(binding.binding_id, intent)
        if cached is not None:
            logger.debug("Mode C2 cache hit for binding %s intent %s",
                         binding.binding_id, intent)
            return MailboxFetchOutcome(
                exemplars=cached,
                message_ids=[e.message_id for e in cached if e.message_id],
                from_cache=True,
            )

    try:
        raw = source.fetch(user_ref, intent)
    except MailboxAccessDenied as exc:
        # The provider refused. Not a blip — the health check will mark this REVOKED, and
        # in the meantime this draft must not pretend to be personalised.
        logger.warning("Mailbox read denied for binding %s: %s", binding.binding_id, exc)
        return MailboxFetchOutcome(
            exemplars=[], message_ids=[], unreachable=True,
            reason="access to your mailbox was refused",
        )
    except MailboxUnreachable as exc:
        logger.warning("Mailbox unreachable for binding %s: %s", binding.binding_id, exc)
        return MailboxFetchOutcome(
            exemplars=[], message_ids=[], unreachable=True,
            reason="your mailbox could not be reached in time",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected mailbox read failure for binding %s", binding.binding_id)
        return MailboxFetchOutcome(
            exemplars=[], message_ids=[], unreachable=True,
            reason=f"the mailbox read failed: {exc}",
        )

    exemplars: List[ExemplarRecord] = []
    message_ids: List[str] = []
    for item in raw[:limit]:
        # Redacted even though these never persist. The bodies are going into a prompt,
        # and a supplier's name and price do not need to be there for the model to copy
        # the writing. Least exposure, not least effort.
        cleaned = redact(item.body, item.subject)
        if not cleaned.has_body:
            continue
        exemplars.append(
            ExemplarRecord(
                exemplar_id=None,          # no row: nothing was stored, and nothing will be
                user_ref=user_ref,
                intent=item.intent or intent or "_all",
                origin=ORIGIN_CUSTOMER_RETAINED,
                profile_version_ref=0,
                subject=item.subject,
                body=cleaned.text,
                is_active=True,
                message_id=item.message_id,
            )
        )
        if item.message_id:
            message_ids.append(item.message_id)

    if use_cache and exemplars:
        mailbox_cache.put(binding.binding_id, intent, exemplars, ttl_seconds=ttl_seconds)

    return MailboxFetchOutcome(exemplars=exemplars, message_ids=message_ids)
