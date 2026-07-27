"""Which style governs this draft, and how sure are we.

Four rungs, tried in order. Every draft records which one it landed on, and anything
above zero is surfaced to the reader — a draft written in a borrowed voice must say so.
Silent degradation is the failure mode this ladder exists to prevent: a draft that looks
like your writing but is not is worse than one that openly admits it is generic.

    0  an approved profile for exactly this person and this kind of email
    1  their user-level profile, when this kind of email has none of its own
    2  the deployment's house style
    3  the platform baseline — nothing personal at all, or the database was unreachable

Rung 2 is the specification's "tenant default", reinterpreted. This platform has no tenant
dimension (see the Phase -1 inventory), so the house style is a single deployment-wide
profile stored under a reserved user_ref. The rung keeps its meaning: a shared default
that is nobody's personal voice but is better than the platform's.

In practice rung 1 will be the common case early on. The user-level profile is the primary
artifact, and per-intent profiles only exist where an intent accumulated enough emails of
its own — which is why the ladder distinguishes "your general voice" from "the house
style" rather than collapsing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.style.profile import StyleProfile, parse_profile
from services.style.repository import (
    USER_LEVEL_INTENT,
    ProfileRecord,
    StyleProfileRepository,
)

logger = logging.getLogger(__name__)

# The reserved holder of the deployment's house style. Not a real person, so it can never
# collide with a Cognito sub.
HOUSE_STYLE_USER_REF = "_house"

LEVEL_EXACT = 0
LEVEL_USER = 1
LEVEL_HOUSE = 2
LEVEL_BASELINE = 3

# What the platform sounds like when it knows nothing about you. Deliberately unremarkable:
# neutral formality, mild directness, no invented mannerisms. A baseline that had character
# would put words in someone's mouth, which is the opposite of the point.
BASELINE_PROFILE_JSON = {
    "structural": {
        "subject_pattern": "topic — reference",
        "greeting": "Hi {first_name},",
        "opening_move": "context_before_ask",
        "body_form": "short_prose",
        "target_words": [70, 130],
        "sign_off": "{sender_first_name}",
        "signature_block": False,
    },
    "register": {
        "formality": 3,
        "directness": 3,
        "hedging": "low",
        "contractions": True,
        "person": "first_plural",
    },
    "lexical": {
        "preferred_terms": [],
        "banned_phrases": ["I hope this email finds you well"],
        "number_format": "bare_numerals",
        "date_format": "day_month",
    },
    "behavioural": {
        "cta_form": "open_question",
        "deadline_phrasing": "soft_by_date",
        "escalation_ladder": ["neutral", "firm", "formal"],
    },
}

BASELINE_PROFILE = parse_profile(BASELINE_PROFILE_JSON)

_REASONS = {
    LEVEL_EXACT: "using your approved style for this kind of email",
    LEVEL_USER: "no approved style for this kind of email yet — using your general style",
    LEVEL_HOUSE: "no style of your own yet — using the organisation's house style",
    LEVEL_BASELINE: "no style profile available — using the platform default",
}


@dataclass(frozen=True)
class ResolvedStyle:
    """The style a draft will be held to, and where it came from."""

    profile: StyleProfile
    fallback_level: int
    reason: str
    record: Optional[ProfileRecord] = None
    scope_user_ref: Optional[str] = None
    scope_intent: Optional[str] = None
    # True when the ladder fell through because the database could not be read, rather
    # than because no profile existed. Same rung, materially different cause.
    source_unreachable: bool = False

    @property
    def degraded(self) -> bool:
        """Whether the reader should be told. Levels 1-3 surface in the API response."""

        return self.fallback_level > LEVEL_EXACT

    @property
    def profile_id(self) -> Optional[int]:
        return self.record.profile_id if self.record else None

    @property
    def profile_version(self) -> Optional[int]:
        return self.record.version if self.record else None


class StyleResolver:
    """Walks the fallback ladder."""

    def __init__(self, conn: Optional[Any] = None,
                 repo: Optional[StyleProfileRepository] = None,
                 bindings: Optional[Any] = None) -> None:
        self.repo = repo or StyleProfileRepository(conn)
        self._conn = conn
        self._bindings = bindings

    def _revoked_binding(self, user_ref: str):
        """The user's mailbox binding, if it has been revoked.

        A revoked binding is not a missing profile — the profile may still be sitting
        there, compiled from that mailbox weeks ago. Continuing to use it would mean
        drafting in a voice learned from mail the customer has since told us to stop
        reading. So revocation overrides the ladder outright and lands on the baseline,
        visibly, rather than quietly carrying on with what we already had.
        """

        try:
            repo = self._bindings
            if repo is None:
                from services.style.mailbox import MailboxBindingRepository

                repo = MailboxBindingRepository(self._conn)
            binding = repo.get_any_for_user(user_ref)
        except Exception:
            logger.warning("Could not check mailbox binding health", exc_info=True)
            return None

        from services.style.mailbox import HEALTH_REVOKED

        if binding is not None and binding.health_state == HEALTH_REVOKED:
            return binding
        return None

    def resolve(self, user_ref: str, intent: str = USER_LEVEL_INTENT) -> ResolvedStyle:
        try:
            revoked = self._revoked_binding(user_ref)
            if revoked is not None:
                logger.warning(
                    "Mailbox binding %s for %s is revoked; forcing the platform baseline",
                    revoked.binding_id, user_ref,
                )
                return ResolvedStyle(
                    profile=BASELINE_PROFILE,
                    fallback_level=LEVEL_BASELINE,
                    reason=(
                        "access to your mailbox has been revoked — using the platform "
                        "default until it is reconnected"
                    ),
                    source_unreachable=True,
                )
            return self._walk(user_ref, intent)
        except Exception:
            # A database problem must not stop someone writing an email. It must, however,
            # be visible: the draft says it is running on the platform default.
            logger.exception(
                "Could not resolve a style profile for %s/%s; falling back to baseline",
                user_ref, intent,
            )
            return ResolvedStyle(
                profile=BASELINE_PROFILE,
                fallback_level=LEVEL_BASELINE,
                reason="style profiles are temporarily unavailable — using the platform default",
                source_unreachable=True,
            )

    def _walk(self, user_ref: str, intent: str) -> ResolvedStyle:
        # 0 — this person, this kind of email.
        exact = self.repo.get_active(user_ref, intent)
        if exact:
            return self._from(exact, LEVEL_EXACT)

        # 1 — this person, any kind of email. Skipped when the request was already for the
        # user-level scope, since that is the same lookup and would report a fallback that
        # did not happen.
        if intent != USER_LEVEL_INTENT:
            general = self.repo.get_active(user_ref, USER_LEVEL_INTENT)
            if general:
                return self._from(general, LEVEL_USER)

        # 2 — the deployment's house style.
        house = self.repo.get_active(HOUSE_STYLE_USER_REF, USER_LEVEL_INTENT)
        if house:
            return self._from(house, LEVEL_HOUSE)

        # 3 — nothing personal at all.
        return ResolvedStyle(
            profile=BASELINE_PROFILE,
            fallback_level=LEVEL_BASELINE,
            reason=_REASONS[LEVEL_BASELINE],
        )

    def _from(self, record: ProfileRecord, level: int) -> ResolvedStyle:
        try:
            profile = record.as_profile()
        except Exception:
            # A stored profile that no longer validates means the schema moved under a
            # profile someone already approved. Do not draft against it.
            logger.exception(
                "Active profile %s no longer validates; falling back to baseline",
                record.profile_id,
            )
            return ResolvedStyle(
                profile=BASELINE_PROFILE,
                fallback_level=LEVEL_BASELINE,
                reason="your saved style could not be read — using the platform default",
                source_unreachable=True,
            )

        return ResolvedStyle(
            profile=profile,
            fallback_level=level,
            reason=_REASONS[level],
            record=record,
            scope_user_ref=record.user_ref,
            scope_intent=record.intent,
        )
