"""The seam between the style engine and the platform's one email drafting path.

Invariant 10 says there is a single code path that generates email drafts. Building the
style engine as a separate service created exactly the problem the invariant warns about:
two ways to produce an email, with only one of them recording where the words came from.
This module closes that by making the existing ``EmailDraftingAgent`` the single
generator, style-governed.

Two properties make the integration safe to land on a live RFQ dispatch path:

* **It is inert without a profile.** The style engine's fallback ladder always returns
  something — at worst the platform baseline — but this returns None at that level, so an
  organisation with no compiled profiles drafts exactly as it did before. The engine
  governs only where a human approved a profile for it to govern with.
* **It cannot break drafting.** Every entry point swallows its own failures and returns
  None. A style lookup that fails must not stop an RFQ going out; losing the voice is a
  degraded email, losing the email is a stalled sourcing round.

Provenance is written whenever the engine did govern, and deliberately not written when it
did not — a draft that recorded a profile it never used would be worse than one recording
nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Intent codes the existing agent's interaction types map onto. The style vocabulary was
# seeded as the union of both sets (see the Phase -1 inventory), so this is a rename, not
# a second taxonomy.
_INTERACTION_TO_INTENT = {
    "rfq": "rfq_invite",
    "clarification": "clarification_request",
    "negotiation": "negotiation_counter",
    "award": "award_notification",
    "update": "internal_update",
    "follow_up": "follow_up",
    "reminder": "reminder",
    "thank_you": "thank_you",
}


@dataclass(frozen=True)
class AppliedStyle:
    """What the style engine contributed to a draft, for provenance."""

    rules: str
    user_ref: str
    intent: str
    fallback_level: int
    fallback_reason: str
    profile_id: Optional[int]
    profile_version: Optional[int]
    exemplar_ids: list
    exemplar_set_hash: Optional[str]

    def as_draft_columns(self) -> Dict[str, Any]:
        """The ``style_*`` columns for ``proc.draft_rfq_emails``."""

        return {
            "style_user_ref": self.user_ref,
            "style_intent": self.intent,
            "style_mode": "A",
            "style_profile_id": self.profile_id,
            "style_profile_version": self.profile_version,
            "style_fallback_level": self.fallback_level,
            "style_exemplar_ids": self.exemplar_ids or None,
            "style_exemplar_set_hash": self.exemplar_set_hash,
        }


def resolve_intent(interaction_type: Optional[str]) -> str:
    """Map the agent's ``interaction_type`` onto a style intent code.

    Unknown values fall through to the user-level scope rather than being guessed at. A
    wrong intent selects the wrong profile, which is worse than selecting the general one.
    """

    from services.style.repository import USER_LEVEL_INTENT

    if not interaction_type:
        return USER_LEVEL_INTENT
    key = str(interaction_type).strip().lower()
    return _INTERACTION_TO_INTENT.get(key, USER_LEVEL_INTENT)


def resolve_user_ref(context: Any = None, sender: Optional[str] = None) -> Optional[str]:
    """Who is this email being written on behalf of.

    Prefers an explicit ``user_ref`` — the Cognito sub of the signed-in caller. Falls back
    to the sending mailbox address, because an agent-initiated RFQ has no signed-in user
    but is still written in *somebody's* voice, and that mailbox is the most honest
    identifier available for it.

    Both are stable strings and the column is TEXT, but they are different kinds of
    identifier and that is worth knowing when reading the data back: a value containing
    '@' came from this fallback.
    """

    for source in (context or {},):
        if isinstance(source, dict):
            candidate = source.get("user_ref") or source.get("cognito_sub")
            if candidate:
                return str(candidate)
    if sender and "@" in str(sender):
        return str(sender).strip().lower()
    return None


def style_for_draft(
    *,
    user_ref: Optional[str],
    interaction_type: Optional[str] = None,
    task_text: Optional[str] = None,
    conn: Any = None,
) -> Optional[AppliedStyle]:
    """The style to hold this draft to, or None to leave drafting unchanged.

    None is returned — deliberately — when the ladder lands on the platform baseline.
    The baseline is what the engine falls back to when nobody has a profile, and applying
    it would rewrite the voice of every RFQ in an organisation that never asked for style
    learning. The engine earns its place only where a profile was compiled and approved.
    """

    if not user_ref:
        return None

    try:
        from services.style.exemplars import ExemplarService
        from services.style.rendering import render_profile_rules
        from services.style.resolver import LEVEL_BASELINE, StyleResolver

        intent = resolve_intent(interaction_type)
        resolved = StyleResolver(conn).resolve(user_ref, intent)

        if resolved.fallback_level >= LEVEL_BASELINE:
            # No approved profile anywhere on the ladder. Leave the existing drafting
            # behaviour alone rather than imposing a generic voice nobody chose.
            return None

        exemplars = []
        if resolved.scope_user_ref and resolved.scope_intent:
            try:
                exemplars = ExemplarService(conn).retrieve(
                    resolved.scope_user_ref, resolved.scope_intent, task_text
                )
            except Exception:
                logger.debug("Could not retrieve exemplars for %s", user_ref, exc_info=True)

        from services.style.drafting import _exemplar_set_hash

        exemplar_ids = [e.exemplar_id for e in exemplars if e.exemplar_id is not None]
        return AppliedStyle(
            rules=render_profile_rules(resolved.profile),
            user_ref=user_ref,
            intent=intent,
            fallback_level=resolved.fallback_level,
            fallback_reason=resolved.reason,
            profile_id=resolved.profile_id,
            profile_version=resolved.profile_version,
            exemplar_ids=exemplar_ids,
            exemplar_set_hash=_exemplar_set_hash(exemplar_ids),
        )
    except Exception:
        # A style lookup that fails must not stop an RFQ going out. Losing the voice is a
        # degraded email; losing the email is a stalled sourcing round.
        logger.warning("Style resolution failed for %s; drafting unchanged",
                       user_ref, exc_info=True)
        return None


def augment_system_prompt(base_prompt: str, style: Optional[AppliedStyle]) -> str:
    """Append the style specification to an existing system prompt.

    Appended rather than replacing, because the agent's own prompts carry the procurement
    substance — what an RFQ must contain, how a negotiation round escalates — and the
    style profile carries only the voice. Replacing one with the other would trade
    correct content for correct tone.

    The precedence sentence is repeated here for the same reason it exists in the
    standalone drafter: given concrete instructions and abstract rules, a model follows
    the concrete ones unless told which wins.
    """

    if style is None or not style.rules:
        return base_prompt

    return (
        f"{base_prompt}\n\n"
        "## WRITING STYLE\n"
        "The following describes how this specific person writes. Follow it for voice, "
        "structure and phrasing. Where it conflicts with the instructions above on what "
        "the email must CONTAIN, the instructions above win — this governs how it reads, "
        "not what it says.\n\n"
        f"{style.rules}"
    )
