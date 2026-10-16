"""Mailbox bindings: which mailbox this platform may read, and proof of the limit.

A binding is a claim — "we can read exactly this one mailbox and nothing else". The claim
is worthless unless it was tested, so a binding cannot become active until a **scope
verification** has run: a deliberate read attempt against a control mailbox the customer
admin nominates, which must come back **denied**. A denial is the evidence. A success
means the credential is broader than the binding says, and activation is refused.

That check is the whole point of this module. Anyone can write "we only read your
mailbox" in a data-flow document; this makes the platform prove it before it reads
anything, and stores the proof.

Nothing here sends mail. Bindings whose role is ``draft_target`` are written to in Phase
6 — writing a draft into a Drafts folder is not sending, and the adapter exposes no method
that transmits.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

PROVIDER_GRAPH = "graph"
PROVIDER_GMAIL = "gmail"
PROVIDER_IMAP = "imap"

ROLE_EXEMPLAR_SOURCE = "exemplar_source"
ROLE_DRAFT_TARGET = "draft_target"
ROLE_BOTH = "both"
ROLE_SEND_RECEIVE = "send_receive"
ORG_USER_REF = "org"

HEALTH_OK = "OK"
HEALTH_DEGRADED = "DEGRADED"
HEALTH_REVOKED = "REVOKED"

_COLUMNS = (
    "binding_id, user_ref, provider, mailbox_address, role, credential_ref, "
    "scope_policy_ref, scope_verified_at, scope_evidence_ref, last_health_check, "
    "health_state, is_active, is_default, last_test_send_at"
)


class ScopeVerificationFailed(Exception):
    """The credential could read a mailbox it should not have been able to.

    Deliberately not a warning. If the control read succeeds, the permission grant is
    wider than the binding claims, and every downstream statement about what this
    platform can reach becomes false.
    """


class BindingNotFound(Exception):
    """No such binding."""


@dataclass(frozen=True)
class MailboxBinding:
    """One row of ``proc.bp_mailbox_binding``."""

    binding_id: int
    user_ref: str
    provider: str
    mailbox_address: str
    role: str
    credential_ref: str
    scope_policy_ref: Optional[str]
    scope_verified_at: Optional[datetime]
    scope_evidence_ref: Optional[str]
    last_health_check: Optional[datetime]
    health_state: str
    is_active: bool
    is_default: bool = False
    last_test_send_at: Optional[datetime] = None

    @property
    def can_send(self) -> bool:
        return self.role == ROLE_SEND_RECEIVE

    @property
    def can_read(self) -> bool:
        return self.role in (ROLE_EXEMPLAR_SOURCE, ROLE_BOTH, ROLE_SEND_RECEIVE)

    @property
    def can_receive_drafts(self) -> bool:
        return self.role in (ROLE_DRAFT_TARGET, ROLE_BOTH)

    @property
    def usable(self) -> bool:
        """Whether this binding may be read from right now."""

        return self.is_active and self.health_state != HEALTH_REVOKED


_FIELDS = ("binding_id", "user_ref", "provider", "mailbox_address", "role", "credential_ref",
           "scope_policy_ref", "scope_verified_at", "scope_evidence_ref", "last_health_check",
           "health_state", "is_active", "is_default", "last_test_send_at")


def _row(r) -> MailboxBinding:
    values = [r[k] for k in _FIELDS] if isinstance(r, dict) else list(r)
    return MailboxBinding(**dict(zip(_FIELDS, values)))


def normalise_address(addr: str) -> str:
    return str(addr or "").strip().lower()


def _audit(action_type: str, *, status: str, summary: str, details: Any = None) -> None:
    try:
        from services.agent_actions import record_action

        record_action(
            phase="style", action_type=action_type, agent="mailbox_binding",
            status=status, summary=summary, details=details,
        )
    except Exception:  # pragma: no cover - audit must never break the caller
        logger.debug("mailbox binding audit write failed", exc_info=True)


class MailboxBindingRepository:
    """Data access and lifecycle for mailbox bindings."""

    def __init__(self, conn: Optional[Any] = None) -> None:
        self._conn = conn

    def _with_conn(self, fn):
        if self._conn is not None:
            return fn(self._conn)
        with get_conn() as conn:
            result = fn(conn)
            if hasattr(conn, "commit"):
                conn.commit()
            return result

    # -- creation ------------------------------------------------------------------

    def create(
        self,
        *,
        user_ref: str,
        provider: str,
        mailbox_address: str,
        role: str,
        credential_ref: str,
        scope_policy_ref: Optional[str] = None,
    ) -> MailboxBinding:
        """Register a binding, **inactive**.

        It stays inactive until ``verify_scope`` produces a denial. A binding that could
        be used the moment it was created would make the verification optional, and an
        optional check is one that gets skipped on the day it matters.
        """

        if not str(credential_ref or "").startswith("arn:aws:secretsmanager:"):
            # The database enforces this too. Failing here gives a better message than a
            # constraint violation, and keeps a token from ever being in a query string.
            raise ValueError(
                "credential_ref must be a Secrets Manager ARN — never a token or password"
            )

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO proc.bp_mailbox_binding "
                "(user_ref, provider, mailbox_address, role, credential_ref, "
                " scope_policy_ref, is_active) "
                f"VALUES (%s, %s, %s, %s, %s, %s, FALSE) RETURNING {_COLUMNS}",
                (user_ref, provider, normalise_address(mailbox_address), role, credential_ref,
                 scope_policy_ref),
            )
            record = _row(cur.fetchone())
            cur.close()
            return record

        binding = self._with_conn(_run)
        _audit(
            "binding_create", status="ok",
            summary=f"registered {provider} binding for {mailbox_address} (inactive)",
            details={"binding_id": binding.binding_id, "role": role,
                     "user_ref": user_ref},
        )
        return binding

    # -- scope verification --------------------------------------------------------

    def verify_scope(
        self,
        binding_id: int,
        *,
        control_mailbox: str,
        probe,
    ) -> MailboxBinding:
        """Prove the credential cannot read a mailbox it was not granted.

        ``probe`` is called with the control address and must return ``(allowed, detail)``.
        ``allowed=True`` means the read succeeded — which is a failure, and raises.

        The control address is supplied by the customer's own admin, deliberately: a
        control mailbox we chose ourselves would prove only that we can be trusted to pick
        an easy one.
        """

        binding = self.get(binding_id)
        if binding is None:
            raise BindingNotFound(f"no mailbox binding with binding_id={binding_id}")

        if normalise_address(control_mailbox) == normalise_address(binding.mailbox_address):
            raise ValueError(
                "the control mailbox must be a DIFFERENT mailbox from the bound one — "
                "probing the bound mailbox proves nothing"
            )

        allowed, detail = probe(control_mailbox)
        evidence_ref = f"scope-{uuid.uuid4().hex}"

        _audit(
            "scope_verification",
            status="error" if allowed else "ok",
            summary=(
                f"control read of {control_mailbox} was "
                f"{'ALLOWED (over-scoped)' if allowed else 'denied'}"
            ),
            details={
                "evidence_ref": evidence_ref,
                "binding_id": binding_id,
                "bound_mailbox": binding.mailbox_address,
                "control_mailbox": control_mailbox,
                "allowed": allowed,
                "detail": detail,
            },
        )

        if allowed:
            # Record the failed attempt against the binding so the refusal is traceable,
            # but leave it inactive.
            self._stamp_verification(binding_id, evidence_ref, activate=False)
            raise ScopeVerificationFailed(
                f"the credential for binding {binding_id} successfully read "
                f"{control_mailbox}, which it should not be able to reach. The permission "
                "grant is wider than this binding claims; activation refused."
            )

        return self._stamp_verification(binding_id, evidence_ref, activate=True)

    def _stamp_verification(
        self, binding_id: int, evidence_ref: str, *, activate: bool
    ) -> MailboxBinding:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE proc.bp_mailbox_binding "
                "SET scope_verified_at = NOW(), scope_evidence_ref = %s, is_active = %s "
                f"WHERE binding_id = %s RETURNING {_COLUMNS}",
                (evidence_ref, activate, binding_id),
            )
            record = _row(cur.fetchone())
            cur.close()
            return record

        return self._with_conn(_run)

    # -- reads ---------------------------------------------------------------------

    def get(self, binding_id: int) -> Optional[MailboxBinding]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE binding_id = %s",
                (binding_id,),
            )
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None

        return self._with_conn(_run)

    def get_for_user(self, user_ref: str, *, readable_only: bool = True
                     ) -> Optional[MailboxBinding]:
        """The active binding for a user. ``readable_only`` excludes draft-target-only."""

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding "
                "WHERE user_ref = %s AND is_active "
                "ORDER BY binding_id DESC LIMIT 1",
                (user_ref,),
            )
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None

        binding = self._with_conn(_run)
        if binding and readable_only and not binding.can_read:
            return None
        return binding

    def get_any_for_user(self, user_ref: str) -> Optional[MailboxBinding]:
        """Any binding, active or not.

        Used by the resolver: a REVOKED binding is inactive, but its existence is exactly
        what must force a visible fallback rather than a silent baseline draft.
        """

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE user_ref = %s "
                "ORDER BY binding_id DESC LIMIT 1",
                (user_ref,),
            )
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None

        return self._with_conn(_run)

    def list_active(self) -> List[MailboxBinding]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE is_active "
                "ORDER BY binding_id"
            )
            rows = cur.fetchall()
            cur.close()
            return [_row(r) for r in rows]

        return self._with_conn(_run)

    # -- company (org) mailboxes ----------------------------------------------------

    def list_org(self, *, include_inactive: bool = False) -> List[MailboxBinding]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE user_ref = %s "
                + ("" if include_inactive else "AND is_active ")
                + "ORDER BY binding_id", (ORG_USER_REF,))
            rows = cur.fetchall()
            cur.close()
            return [_row(r) for r in rows]
        return self._with_conn(_run)

    def _one(self, where: str, params) -> Optional[MailboxBinding]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE {where} "
                        "ORDER BY binding_id DESC LIMIT 1", params)
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None
        return self._with_conn(_run)

    def get_default_sender(self) -> Optional[MailboxBinding]:
        return self._one("user_ref = %s AND is_default AND is_active", (ORG_USER_REF,))

    def get_org_default_any(self) -> Optional[MailboxBinding]:
        """An org send_receive mailbox that has ever sent a test email, active or not.

        Its existence turns "no mailbox" into "the company mailbox is unavailable":
        the first sends through SES, the second must be held (spec 5.1).
        """
        return self._one("user_ref = %s AND role = %s AND last_test_send_at IS NOT NULL",
                         (ORG_USER_REF, ROLE_SEND_RECEIVE))

    def _update(self, sql: str, params) -> MailboxBinding:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(sql + f" RETURNING {_COLUMNS}", params)
            row = cur.fetchone()
            cur.close()
            if row is None:
                raise BindingNotFound("no such mailbox binding")
            return _row(row)
        return self._with_conn(_run)

    def record_test_send(self, binding_id: int) -> MailboxBinding:
        b = self._update("UPDATE proc.bp_mailbox_binding SET last_test_send_at = NOW() "
                         "WHERE binding_id = %s", (binding_id,))
        _audit("test_send", status="ok", summary=f"test email sent from {b.mailbox_address}",
               details={"binding_id": binding_id})
        return b

    def set_default(self, binding_id: int) -> MailboxBinding:
        b = self.get(binding_id)
        if b is None:
            raise BindingNotFound(f"no mailbox binding with binding_id={binding_id}")
        if not (b.is_active and b.can_send and b.last_test_send_at):
            raise ValueError("a mailbox can be the default only once it is verified and a "
                             "test email has been sent from it")

        def _run(conn):
            cur = conn.cursor()
            # Clear first: the partial unique index admits one active default at a time.
            cur.execute("UPDATE proc.bp_mailbox_binding SET is_default = FALSE "
                        "WHERE user_ref = %s AND is_default AND binding_id <> %s",
                        (ORG_USER_REF, binding_id))
            cur.execute("UPDATE proc.bp_mailbox_binding SET is_default = TRUE "
                        f"WHERE binding_id = %s RETURNING {_COLUMNS}", (binding_id,))
            row = cur.fetchone()
            cur.close()
            return _row(row)

        out = self._with_conn(_run)
        _audit("default_set", status="ok", summary=f"{b.mailbox_address} is now the default",
               details={"binding_id": binding_id})
        return out

    def deactivate(self, binding_id: int, *, reason: str,
                   forget_verification: bool = False) -> MailboxBinding:
        """Stop using a binding. ``forget_verification`` (an admin removing it) also clears
        scope_verified_at: its proof no longer stands, and a mailbox whose access was
        withdrawn in Microsoft 365 stops being listed once an admin has removed it. The
        evidence ref and last_test_send_at are kept (the latter keeps email held)."""
        b = self._update("UPDATE proc.bp_mailbox_binding SET is_active = FALSE, "
                         "is_default = FALSE"
                         + (", scope_verified_at = NULL" if forget_verification else "")
                         + " WHERE binding_id = %s", (binding_id,))
        _audit("binding_deactivate", status="ok",
               summary=f"{b.mailbox_address} deactivated: {reason}",
               details={"binding_id": binding_id, "reason": reason})
        return b

    # -- health --------------------------------------------------------------------

    def record_health(self, binding_id: int, state: str) -> MailboxBinding:
        """Update health, and deactivate on revocation.

        A revoked binding is not merely unhealthy: the customer has withdrawn permission,
        so it stops being usable immediately rather than at the next retry.
        """

        if state not in (HEALTH_OK, HEALTH_DEGRADED, HEALTH_REVOKED):
            raise ValueError(f"unknown health state {state!r}")

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE proc.bp_mailbox_binding "
                "SET health_state = %s, last_health_check = NOW(), "
                "    is_active = CASE WHEN %s = 'REVOKED' THEN FALSE ELSE is_active END "
                f"WHERE binding_id = %s RETURNING {_COLUMNS}",
                (state, state, binding_id),
            )
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None

        binding = self._with_conn(_run)
        if binding is None:
            raise BindingNotFound(f"no mailbox binding with binding_id={binding_id}")

        if state == HEALTH_REVOKED:
            from services.style.mailbox_cache import flush_binding

            flush_binding(binding_id)
            _audit(
                "binding_revoked", status="warning",
                summary=f"binding {binding_id} revoked; drafting degrades visibly",
                details={"binding_id": binding_id,
                         "mailbox": binding.mailbox_address},
            )
        return binding

    def unbind(self, binding_id: int) -> MailboxBinding:
        """Deactivate a binding and flush anything cached from it.

        The cache flush is not housekeeping. Under Mode C2 the cache holds email bodies
        in memory, and unbinding is the customer saying stop reading — leaving five
        minutes of their mail sitting in a process would make that instruction a
        suggestion.
        """

        from services.style.mailbox_cache import flush_binding

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE proc.bp_mailbox_binding SET is_active = FALSE "
                f"WHERE binding_id = %s RETURNING {_COLUMNS}",
                (binding_id,),
            )
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None

        binding = self._with_conn(_run)
        if binding is None:
            raise BindingNotFound(f"no mailbox binding with binding_id={binding_id}")

        flush_binding(binding_id)
        _audit("binding_unbound", status="ok",
               summary=f"binding {binding_id} unbound and cache flushed",
               details={"binding_id": binding_id})
        return binding
