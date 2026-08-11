"""A support escalation must not leave the sending domain.

``_notify_admin`` mails the user's verbatim message and the assistant's reply to
``SUPPORT_ADMIN_EMAIL``, whose default is a personal gmail.com address. Nothing
checked where that was going, so an unconfigured deployment exports user support
content — including whatever commercial detail they pasted in — to a third-party
consumer mailbox.

The ticket row is written regardless, so refusing the mail loses nothing: the
escalation still exists to be worked, it is just not couriered off-domain.
"""
from __future__ import annotations

import pytest

from src.services import support_agent


@pytest.fixture
def agent():
    return support_agent.SupportAgent.__new__(support_agent.SupportAgent)


@pytest.mark.parametrize("admin,sender,expected", [
    ("nicholasgeelen@gmail.com", "noreply@procwise.co.uk", False),
    ("support@procwise.co.uk", "noreply@procwise.co.uk", True),
    ("SUPPORT@PROCWISE.CO.UK", "noreply@procwise.co.uk", True),
    ("support@procwise.co.uk.evil.com", "noreply@procwise.co.uk", False),
    ("support@procwise.co.uk", "", False),          # no sending domain -> refuse
    ("", "noreply@procwise.co.uk", False),
])
def test_internal_admin_detection(monkeypatch, admin, sender, expected):
    monkeypatch.setattr(support_agent, "_ADMIN_EMAIL", admin)
    monkeypatch.setenv("SES_DEFAULT_SENDER", sender)
    assert support_agent.SupportAgent._admin_is_internal() is expected


def test_an_external_admin_address_is_not_emailed(monkeypatch, agent):
    """The gmail default. No transport call may happen."""
    monkeypatch.setattr(support_agent, "_ADMIN_EMAIL", "nicholasgeelen@gmail.com")
    monkeypatch.setenv("SES_DEFAULT_SENDER", "noreply@procwise.co.uk")

    import services.email_service as es
    calls = []
    monkeypatch.setattr(
        es.EmailService, "send_email",
        lambda self, **kw: calls.append(kw),
    )

    status, error = agent._notify_admin(
        reference="SUP-1", summary="cannot upload",
        user_name="A Buyer", user_email="buyer@customer.example",
        message="Our Q3 pricing file for Northgate fails to upload",
        agent_reply="Try the Documents screen",
    )

    assert status == "skipped"
    assert calls == [], f"user support content was emailed off-domain: {calls}"


def test_an_internal_admin_address_still_receives_the_escalation(
    monkeypatch, agent
):
    """The guard must not silently disable support escalation when configured."""
    monkeypatch.setattr(support_agent, "_ADMIN_EMAIL", "support@procwise.co.uk")
    monkeypatch.setenv("SES_DEFAULT_SENDER", "noreply@procwise.co.uk")

    class _Settings:
        ses_default_sender = "noreply@procwise.co.uk"

    agent.agent_nick = type("N", (), {"settings": _Settings()})()

    import services.email_service as es
    calls = []
    monkeypatch.setattr(
        es.EmailService, "send_email",
        lambda self, **kw: calls.append(kw) or True,
    )

    status, _ = agent._notify_admin(
        reference="SUP-2", summary="cannot upload",
        user_name="A Buyer", user_email="buyer@customer.example",
        message="Our Q3 pricing file fails to upload",
        agent_reply="Try the Documents screen",
    )

    assert status == "sent"
    assert len(calls) == 1
    assert calls[0]["recipients"] == "support@procwise.co.uk"
