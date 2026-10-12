"""``In-Reply-To``/``References`` that put a reply in the supplier's own conversation.

Moved out of api/routers/workflows.py unchanged, so the agent-written reply to a
supplier's final offer threads exactly as a person's reply from the Action Centre
does. One implementation: the two had no reason to differ, and the agent's reply
was arriving as a new conversation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from repositories import workflow_email_tracking_repo

logger = logging.getLogger(__name__)


def reply_thread_headers(
    *, source_draft: Dict[str, Any], source_payload: Dict[str, Any]
) -> Dict[str, List[str]]:
    """``In-Reply-To``/``References`` that put a reply in the supplier's own thread.

    Where the values come from, in the order they matter:

      * ``proc.workflow_email_tracking.response_message_id`` -- the Message-ID of the
        SUPPLIER'S reply, which is the message we are answering, so it is the correct
        ``In-Reply-To``. Written by the watcher when it matched the reply.
      * ``...message_id`` -- the Message-ID of OUR outbound RFQ. Used as ``In-Reply-To``
        only when the supplier's own is not recorded (an older row), and always folded
        into ``References`` so the chain is complete.
      * any ``References`` already recorded against the thread, kept in front.

    Nothing is fabricated. A thread with no recorded message ids returns ``{}`` and the
    caller reports that, rather than sending a reply that looks like a new conversation
    while claiming otherwise.
    """
    tracked = None
    workflow_id = source_draft.get("workflow_id")
    unique_id = source_draft.get("unique_id")
    if workflow_id and unique_id:
        try:
            tracked = workflow_email_tracking_repo.lookup_dispatch_row(
                workflow_id=str(workflow_id), unique_id=str(unique_id)
            )
        except Exception:  # noqa: BLE001 - absence of threading is reported, not raised
            logger.exception(
                "could not read the dispatch row for unique_id=%s while preparing a reply",
                unique_id,
            )

    # Whatever the thread already recorded, from the tracking row or the draft itself.
    prior: Dict[str, Any] = {}
    for candidate in (
        getattr(tracked, "thread_headers", None),
        source_draft.get("thread_headers"),
        source_payload.get("thread_headers"),
        (source_payload.get("metadata") or {}).get("thread_headers")
        if isinstance(source_payload.get("metadata"), dict) else None,
    ):
        if isinstance(candidate, dict) and candidate:
            prior = candidate
            break

    def _as_list(value: Any) -> List[str]:
        """Message ids in one canonical ``<id>`` form.

        The two sources disagree on form and both end up in ``References``:
        ``workflow_email_tracking._parse_thread_headers`` strips ``<>`` off everything it
        reads back, while the ``message_id``/``response_message_id`` columns come back
        verbatim with their brackets. RFC 5322 requires every msg-id in ``In-Reply-To``/
        ``References`` to be bracketed, so a bare one makes the header malformed; and
        without a single form, the same message listed both ways survives the dedupe
        twice. A ``References`` string may also carry several space-separated ids.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            tokens = [str(v) for v in value]
        else:
            tokens = [str(value)]

        ids: List[str] = []
        for token in tokens:
            for part in token.split():
                bare = part.strip().strip("<>").strip()
                if bare:
                    ids.append(f"<{bare}>")
        return ids

    our_outbound = _as_list(getattr(tracked, "message_id", None) or prior.get("Message-ID"))
    their_reply = _as_list(getattr(tracked, "response_message_id", None))

    # The message we are actually answering. Their reply if we recorded it; our own
    # outbound otherwise -- which still lands in the right conversation, one hop up.
    in_reply_to = (their_reply or our_outbound)[:1]
    if not in_reply_to:
        return {}

    references: List[str] = []
    for item in _as_list(prior.get("References")) + our_outbound + their_reply:
        if item not in references:
            references.append(item)

    headers: Dict[str, List[str]] = {"In-Reply-To": in_reply_to}
    if references:
        headers["References"] = references
    return headers
