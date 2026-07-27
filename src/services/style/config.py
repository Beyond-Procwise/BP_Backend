"""Configuration for the style subsystem, read from ``proc.bp_admin_config``.

Settings live in the database rather than ``.env`` so deployment mode can be changed
without a redeploy, matching how the rest of this platform's governance works
(``bp_prompt``, ``bp_policy``, ``bp_admin_config``).

Reads are best-effort: a missing row, an unreachable database or a malformed value
falls back to the safest setting rather than raising. Mode ``A`` — pasted exemplars,
no mailbox access at all — is the safe default, so a configuration failure can never
silently promote the system into reading someone's mailbox.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

CONFIG_KEY = "style_engine"

# Mode A  — exemplars pasted by the user; no mail provider involved.
# Mode B  — platform-generated synthetic exemplars only.
# Mode C1 — mailbox read at COMPILE time only; drafting never touches the provider.
# Mode C2 — mailbox read at DRAFT time, cached, never persisted.
VALID_MODES = ("A", "B", "C1", "C2")

DEFAULT_MODE = "A"
DEFAULT_MIN_EXEMPLARS = 3
DEFAULT_STAGING_TTL_HOURS = 24

# Below this, a profile compiled from the exemplars would be describing noise.
_MIN_ALLOWED_EXEMPLARS = 3


@dataclass(frozen=True)
class StyleConfig:
    """Resolved settings for the style subsystem."""

    deployment_mode: str = DEFAULT_MODE
    min_exemplars: int = DEFAULT_MIN_EXEMPLARS
    staging_ttl_hours: int = DEFAULT_STAGING_TTL_HOURS
    # False when the row could not be read and defaults are standing in. Callers that
    # report configuration to a user should say so rather than present defaults as
    # though they were chosen.
    loaded_from_db: bool = False

    @property
    def reads_mailbox_at_compile_time(self) -> bool:
        return self.deployment_mode in ("C1", "C2")

    @property
    def reads_mailbox_at_draft_time(self) -> bool:
        """Only Mode C2 touches the mail provider while a draft is being generated.

        Under every other mode drafting must complete with the provider unreachable.
        """
        return self.deployment_mode == "C2"


def _coerce_mode(value: Any) -> str:
    candidate = str(value or "").strip().upper()
    if candidate in VALID_MODES:
        return candidate
    if value not in (None, ""):
        logger.warning(
            "style_engine.deployment_mode=%r is not one of %s; falling back to %s",
            value,
            ", ".join(VALID_MODES),
            DEFAULT_MODE,
        )
    return DEFAULT_MODE


def _coerce_positive_int(value: Any, *, default: int, minimum: int, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        if value not in (None, ""):
            logger.warning("style_engine.%s=%r is not an integer; using %s", field, value, default)
        return default
    if number < minimum:
        logger.warning(
            "style_engine.%s=%s is below the minimum of %s; using the minimum",
            field,
            number,
            minimum,
        )
        return minimum
    return number


def _parse(payload: Mapping[str, Any]) -> StyleConfig:
    return StyleConfig(
        deployment_mode=_coerce_mode(payload.get("deployment_mode")),
        min_exemplars=_coerce_positive_int(
            payload.get("min_exemplars"),
            default=DEFAULT_MIN_EXEMPLARS,
            minimum=_MIN_ALLOWED_EXEMPLARS,
            field="min_exemplars",
        ),
        staging_ttl_hours=_coerce_positive_int(
            payload.get("staging_ttl_hours"),
            default=DEFAULT_STAGING_TTL_HOURS,
            minimum=1,
            field="staging_ttl_hours",
        ),
        loaded_from_db=True,
    )


def load_style_config(conn: Optional[Any] = None) -> StyleConfig:
    """Return the style subsystem's settings.

    ``conn`` is accepted so a caller already inside a transaction can reuse it;
    otherwise a connection is opened and closed here.
    """

    if conn is not None:
        return _load_with_conn(conn)

    try:
        with get_conn() as owned_conn:
            return _load_with_conn(owned_conn)
    except Exception:
        logger.exception("Could not read %s config; using safe defaults", CONFIG_KEY)
        return StyleConfig()


def _load_with_conn(conn: Any) -> StyleConfig:
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT config_value FROM proc.bp_admin_config WHERE config_key = %s",
            (CONFIG_KEY,),
        )
        row = cur.fetchone()
        cur.close()
    except Exception:
        logger.exception("Could not read %s config; using safe defaults", CONFIG_KEY)
        return StyleConfig()

    if not row or row[0] is None:
        logger.info(
            "No %s row in proc.bp_admin_config; using defaults (mode=%s, min_exemplars=%s)",
            CONFIG_KEY,
            DEFAULT_MODE,
            DEFAULT_MIN_EXEMPLARS,
        )
        return StyleConfig()

    payload = row[0]
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            logger.warning("%s config is not valid JSON; using safe defaults", CONFIG_KEY)
            return StyleConfig()

    if not isinstance(payload, Mapping):
        logger.warning("%s config is not an object; using safe defaults", CONFIG_KEY)
        return StyleConfig()

    return _parse(payload)
