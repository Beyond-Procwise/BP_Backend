"""Reading and writing style profiles.

The rules this module exists to hold:

* **Versions are append-only.** Recompiling inserts a new version; it never edits an
  existing one. The database enforces this too (see the freeze trigger), because this
  module is not the only thing that will ever hold a connection.
* **Nothing auto-activates.** ``insert_version`` always lands in ``DRAFT``. Becoming the
  active profile requires ``approve`` and a named human.
* **Approval is atomic.** Standing the previous version down and standing the new one up
  happen in one transaction. The partial unique index on ``(user_ref, intent) WHERE
  is_active`` is what makes that a guarantee rather than a hope: if the first step were
  skipped, the second would violate the index and the whole transaction would fail.

``USER_LEVEL_INTENT`` is the default scope. Per-intent profiles are the exception, used
only where an intent genuinely accumulated enough exemplars of its own.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from services.db import get_conn
from services.style.profile import StyleProfile, parse_profile

logger = logging.getLogger(__name__)

# The scope covering every communication type. A real row in bp_style_intent rather than
# a NULL, so it can carry a foreign key and take part in the one-active-per-scope unique
# index — Postgres treats NULLs as distinct, so a nullable intent would happily allow two
# active user-level profiles.
USER_LEVEL_INTENT = "_all"

STATE_UNCOMPILED = "UNCOMPILED"
STATE_DRAFT = "DRAFT"
STATE_APPROVED = "APPROVED"
STATE_SUPERSEDED = "SUPERSEDED"

_COLUMNS = (
    "profile_id, user_ref, intent, version, state, profile_json, exemplar_count, "
    "source_batch_id, compiled_at, approved_by, approved_at, is_active"
)


class ProfileNotFound(Exception):
    """The profile asked for does not exist."""


class ProfileNotApprovable(Exception):
    """The profile exists but is not in a state that can be approved."""


@dataclass(frozen=True)
class ProfileRecord:
    """One row of ``proc.bp_style_profile``."""

    profile_id: int
    user_ref: str
    intent: str
    version: int
    state: str
    profile_json: dict
    exemplar_count: int
    source_batch_id: Optional[str]
    compiled_at: Optional[datetime]
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    is_active: bool

    @property
    def is_user_level(self) -> bool:
        return self.intent == USER_LEVEL_INTENT

    def as_profile(self) -> StyleProfile:
        """The validated profile. Raises if the stored JSON no longer conforms — which
        would mean the schema changed under a profile someone already approved."""

        return parse_profile(self.profile_json)


def _row_to_record(row: Any) -> ProfileRecord:
    payload = row[5]
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    return ProfileRecord(
        profile_id=row[0],
        user_ref=row[1],
        intent=row[2],
        version=row[3],
        state=row[4],
        profile_json=payload,
        exemplar_count=row[6],
        source_batch_id=row[7],
        compiled_at=row[8],
        approved_by=row[9],
        approved_at=row[10],
        is_active=row[11],
    )


class StyleProfileRepository:
    """Data access for style profiles.

    A connection may be supplied so a caller already inside a transaction can reuse it;
    otherwise each call opens and closes its own.
    """

    def __init__(self, conn: Optional[Any] = None) -> None:
        self._conn = conn

    # -- connection plumbing ------------------------------------------------------

    def _cursor(self, conn: Any):
        return conn.cursor()

    def _with_conn(self, fn):
        if self._conn is not None:
            return fn(self._conn)
        with get_conn() as conn:
            return fn(conn)

    # -- reads --------------------------------------------------------------------

    def get_active(self, user_ref: str, intent: str) -> Optional[ProfileRecord]:
        """The profile currently governing drafting for this scope, if there is one."""

        def _run(conn):
            cur = self._cursor(conn)
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_style_profile "
                "WHERE user_ref = %s AND intent = %s AND is_active",
                (user_ref, intent),
            )
            row = cur.fetchone()
            cur.close()
            return _row_to_record(row) if row else None

        return self._with_conn(_run)

    def get(self, profile_id: int) -> Optional[ProfileRecord]:
        def _run(conn):
            cur = self._cursor(conn)
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_style_profile WHERE profile_id = %s",
                (profile_id,),
            )
            row = cur.fetchone()
            cur.close()
            return _row_to_record(row) if row else None

        return self._with_conn(_run)

    def list_versions(self, user_ref: str, intent: str) -> list[ProfileRecord]:
        """Every version for a scope, newest first. The audit view."""

        def _run(conn):
            cur = self._cursor(conn)
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_style_profile "
                "WHERE user_ref = %s AND intent = %s ORDER BY version DESC",
                (user_ref, intent),
            )
            rows = cur.fetchall()
            cur.close()
            return [_row_to_record(r) for r in rows]

        return self._with_conn(_run)

    def next_version(self, conn: Any, user_ref: str, intent: str) -> int:
        cur = self._cursor(conn)
        cur.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM proc.bp_style_profile "
            "WHERE user_ref = %s AND intent = %s",
            (user_ref, intent),
        )
        version = cur.fetchone()[0]
        cur.close()
        return int(version)

    # -- writes -------------------------------------------------------------------

    def insert_version(
        self,
        *,
        user_ref: str,
        intent: str,
        profile: StyleProfile,
        exemplar_count: int,
        source_batch_id: Optional[str] = None,
    ) -> ProfileRecord:
        """Insert the next version for a scope, in ``DRAFT``.

        Never activates. A recompiled profile that quietly took over would change how
        someone's mail reads without anyone agreeing to it, which is why invariant 6
        exists.
        """

        payload = profile.to_json_dict()

        def _run(conn):
            version = self.next_version(conn, user_ref, intent)
            cur = self._cursor(conn)
            cur.execute(
                "INSERT INTO proc.bp_style_profile "
                "(user_ref, intent, version, state, profile_json, exemplar_count, source_batch_id) "
                f"VALUES (%s, %s, %s, '{STATE_DRAFT}', %s, %s, %s) "
                f"RETURNING {_COLUMNS}",
                (user_ref, intent, version, json.dumps(payload), exemplar_count, source_batch_id),
            )
            row = cur.fetchone()
            cur.close()
            self._commit(conn)
            logger.info(
                "Compiled style profile v%s for user_ref=%s intent=%s from %s exemplars (DRAFT)",
                version, user_ref, intent, exemplar_count,
            )
            return _row_to_record(row)

        return self._with_conn(_run)

    def approve(self, profile_id: int, approved_by: str) -> ProfileRecord:
        """Make a DRAFT the active profile, standing the previous one down.

        Both steps happen in one transaction, in this order. The partial unique index on
        ``(user_ref, intent) WHERE is_active`` means a transaction that failed to stand
        the old version down could not commit the new one — so "atomic" here is enforced
        by the schema, not merely intended by this method.
        """

        if not approved_by or not str(approved_by).strip():
            raise ValueError("approve requires the identity of the approver")
        approver = str(approved_by).strip()

        def _run(conn):
            cur = self._cursor(conn)
            try:
                cur.execute(
                    "SELECT user_ref, intent, state FROM proc.bp_style_profile "
                    "WHERE profile_id = %s FOR UPDATE",
                    (profile_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise ProfileNotFound(f"no style profile with profile_id={profile_id}")

                user_ref, intent, state = row[0], row[1], row[2]
                if state != STATE_DRAFT:
                    raise ProfileNotApprovable(
                        f"profile_id={profile_id} is {state}; only a {STATE_DRAFT} "
                        "profile can be approved"
                    )

                # Stand the incumbent down FIRST — the unique index forbids two actives.
                cur.execute(
                    f"UPDATE proc.bp_style_profile SET is_active = FALSE, state = '{STATE_SUPERSEDED}' "
                    "WHERE user_ref = %s AND intent = %s AND is_active AND profile_id <> %s",
                    (user_ref, intent, profile_id),
                )
                superseded = cur.rowcount

                cur.execute(
                    f"UPDATE proc.bp_style_profile SET state = '{STATE_APPROVED}', "
                    "approved_by = %s, approved_at = NOW(), is_active = TRUE "
                    f"WHERE profile_id = %s RETURNING {_COLUMNS}",
                    (approver, profile_id),
                )
                approved = _row_to_record(cur.fetchone())
                cur.close()
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise

            logger.info(
                "Approved style profile v%s for user_ref=%s intent=%s by %s (superseded %s)",
                approved.version, user_ref, intent, approver, superseded,
            )
            return approved

        return self._with_conn(_run)

    def deactivate(self, user_ref: str, intent: str) -> int:
        """Stand down the active profile without approving a replacement.

        Drafting then falls to the next rung of the fallback ladder with a visible flag —
        deleting a profile must degrade openly, never silently.
        """

        def _run(conn):
            cur = self._cursor(conn)
            cur.execute(
                f"UPDATE proc.bp_style_profile SET is_active = FALSE, state = '{STATE_SUPERSEDED}' "
                "WHERE user_ref = %s AND intent = %s AND is_active",
                (user_ref, intent),
            )
            count = cur.rowcount
            cur.close()
            self._commit(conn)
            return count

        return self._with_conn(_run)

    # -- transaction helpers ------------------------------------------------------
    # Only own the transaction when we own the connection: a caller that passed one in
    # is running its own unit of work and gets to decide when it ends.

    def _commit(self, conn: Any) -> None:
        if self._conn is None and hasattr(conn, "commit"):
            conn.commit()

    def _rollback(self, conn: Any) -> None:
        if self._conn is None and hasattr(conn, "rollback"):
            try:
                conn.rollback()
            except Exception:  # pragma: no cover - rollback is best effort
                logger.debug("rollback failed", exc_info=True)
