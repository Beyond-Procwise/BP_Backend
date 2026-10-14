"""The weekly digest must pass the policy gate like any other outbound mail.

``run_weekly_digest`` called ``EmailService.send_email`` directly. Its body is
``build_value_summary()`` — every verified over-billing finding in the corpus,
with supplier names and amounts — and its recipients came from an environment
variable, so no code decided whether that content could go to those addresses.

The gate cannot be ``email_dispatch_guard.authorize_dispatch``: that one is
supplier-shaped (it requires a stored approved draft and recipients on the
supplier master), and the digest is internal mail with neither. So the digest
calls ``guardrail.authorize`` directly, the same way ``value_query_service``
does, plus an internal-domain check on every recipient.

WHO the digest sends as is the load-bearing part. A scheduled job has no
authenticated caller, and inventing a principal that grants itself a role would
be a self-signed permission. Instead ``VALUE_DIGEST_SENT_AS`` names a subject,
and that subject's role is resolved by rbac from ``proc.bp_role_assignment`` —
a governed, auditable grant. Config chooses the identity; the database decides
what the identity may do. Unconfigured, or configured to someone with no grant,
the digest does not send.
"""
from __future__ import annotations

import pytest

from src.services import value_digest


_SUMMARY = {
    "verified_found_gbp": 12345.67,
    "recovered_gbp": 0.0,
    "potential_gbp": 500.0,
    "finding_count": 4,
    "by_supplier": [
        {"supplier_name": "Northgate Ltd", "verified_found_gbp": 12345.67,
         "finding_count": 4},
    ],
    "findings": [],
}


@pytest.fixture(autouse=True)
def _pin_sending_domain(monkeypatch):
    """Set the sending domain explicitly.

    Without this the internal/external split depends on whichever
    SES_DEFAULT_SENDER happened to be in the ambient environment when pytest
    started — so the tests would pass under `set -a; . ./.env` and behave
    differently in CI, which is a test that proves nothing about the code.
    """
    monkeypatch.setenv("SES_DEFAULT_SENDER", "noreply@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")


@pytest.fixture
def sent(monkeypatch):
    """Records every send that reached the transport."""
    calls: list[dict] = []

    def _send(*, to, subject, body, agent_nick=None):
        calls.append({"to": to, "subject": subject, "body": body})
        return True

    monkeypatch.setattr(value_digest, "_send_email", _send)
    monkeypatch.setattr(value_digest, "_load_summary", lambda: _SUMMARY)
    monkeypatch.setattr(
        value_digest, "compose_digest",
        lambda summary, now: {"subject": "Value found this week",
                              "body": "Northgate Ltd  GBP 12,345.67"},
    )
    return calls


def _allow(**_kw):
    from src.services import guardrail
    return guardrail.Decision(allowed=True, reason="test allow")


def _deny(**_kw):
    from src.services import guardrail
    return guardrail.Decision(allowed=False, reason="test deny")


# --------------------------------------------------------------------------
# The gate is consulted at all
# --------------------------------------------------------------------------

def test_the_digest_asks_the_policy_gate_before_sending(monkeypatch, sent):
    asked: list[tuple] = []

    def _authorize(action, action_class, principal, context=None,
                   policy_engine=None):
        asked.append((action, action_class, getattr(principal, "subject", None)))
        return _allow()

    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    monkeypatch.setattr(value_digest.guardrail, "authorize", _authorize)

    value_digest.run_weekly_digest()

    assert asked, "run_weekly_digest sent without consulting guardrail.authorize"
    action, action_class, subject = asked[0]
    assert action == "email.send"
    assert action_class == "communicate"
    assert subject == "digest-service@procwise.co.uk", (
        "the digest must send as a named, governed subject so rbac can resolve "
        "its role from proc.bp_role_assignment"
    )


def test_a_denied_digest_does_not_reach_the_transport(monkeypatch, sent):
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    monkeypatch.setattr(value_digest.guardrail, "authorize",
                        lambda *a, **k: _deny())

    assert value_digest.run_weekly_digest() == 0
    assert sent == [], f"denied digest still sent: {sent}"


def test_an_allowed_digest_still_sends(monkeypatch, sent):
    """The gate must not become a way to break the feature silently."""
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    monkeypatch.setattr(value_digest.guardrail, "authorize",
                        lambda *a, **k: _allow())

    assert value_digest.run_weekly_digest() == 1
    assert len(sent) == 1
    assert sent[0]["to"] == ["ops@procwise.co.uk"]


# --------------------------------------------------------------------------
# Identity must be configured, not invented
# --------------------------------------------------------------------------

def test_an_unconfigured_sender_identity_refuses_rather_than_sending(
    monkeypatch, sent
):
    """No VALUE_DIGEST_SENT_AS means nobody is accountable for this mail.

    A scheduled job with no principal must not fall back to sending anyway;
    guardrail.authorize would see principal=None and, for an irreversible
    class, deny — but relying on that leaves the digest one policy edit away
    from sending unattributed. Refuse here, explicitly.
    """
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.delenv("VALUE_DIGEST_SENT_AS", raising=False)

    called: list = []
    monkeypatch.setattr(value_digest.guardrail, "authorize",
                        lambda *a, **k: called.append(1) or _allow())

    assert value_digest.run_weekly_digest() == 0
    assert sent == []


def test_the_principal_carries_no_claims_so_the_role_comes_from_the_grant_table(
    monkeypatch, sent
):
    """A config-supplied principal must not carry group claims.

    Claims on a principal built from configuration would be a role the config
    granted itself. The subject alone is supplied; rbac.resolve_roles reads
    proc.bp_role_assignment for it.
    """
    seen: list = []
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    monkeypatch.setattr(
        value_digest.guardrail, "authorize",
        lambda a, c, principal, *rest, **kw: seen.append(principal) or _allow(),
    )

    value_digest.run_weekly_digest()

    assert seen, "authorize was never called"
    assert not getattr(seen[0], "claims", None), (
        "the digest principal carries claims; a role granted by config is a "
        "role the config granted itself"
    )


# --------------------------------------------------------------------------
# Recipients
# --------------------------------------------------------------------------

@pytest.mark.parametrize("recipients", [
    "attacker@gmail.com",
    "ops@procwise.co.uk,attacker@gmail.com",
])
def test_an_external_recipient_is_refused(monkeypatch, sent, recipients):
    """The digest names suppliers and amounts. It is internal mail, and an
    address outside the sending domain is not a typo to be delivered anyway."""
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", recipients)
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    monkeypatch.setattr(value_digest.guardrail, "authorize",
                        lambda *a, **k: _allow())

    assert value_digest.run_weekly_digest() == 0
    assert sent == []


def test_no_recipients_configured_is_a_quiet_skip_not_a_denial(monkeypatch, sent):
    """Unchanged behaviour: nothing to send to is not a policy failure."""
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    assert value_digest.run_weekly_digest() == 0
    assert sent == []


def test_no_sending_domain_configured_refuses_every_recipient(monkeypatch, sent):
    """Fail closed on a missing SES_DEFAULT_SENDER.

    Without a sending domain there is no way to tell an internal address from
    an external one. Treating every address as internal would turn a missing
    environment variable into a broadcast of the whole findings list.
    """
    monkeypatch.setenv("SES_DEFAULT_SENDER", "")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    monkeypatch.setattr(value_digest.guardrail, "authorize",
                        lambda *a, **k: _allow())

    assert value_digest.run_weekly_digest() == 0
    assert sent == []


def test_the_disabled_switch_still_short_circuits_everything(monkeypatch, sent):
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "0")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ops@procwise.co.uk")
    monkeypatch.setenv("VALUE_DIGEST_SENT_AS", "digest-service@procwise.co.uk")
    assert value_digest.run_weekly_digest() == 0
    assert sent == []
