"""Product notices: from the shared mailbox only when the policy allows and the user approves."""
from types import SimpleNamespace

import pytest

from services import mail_notices, mail_policy


class FakeSES:
    def __init__(self):
        self.calls = []

    def send_email(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(success=True)


def _policy(mode="with_user_approval", internal_send="with_user_approval"):
    return SimpleNamespace(notices_from_mailbox=mode, internal_send=internal_send,
                           company_domains=["yourcompany.com"])


@pytest.fixture
def world(monkeypatch):
    w = SimpleNamespace(created=[], ready=True, policy=_policy())

    def load():
        if isinstance(w.policy, Exception):
            raise w.policy
        return w.policy
    monkeypatch.setattr(mail_notices.mail_policy, "load", load)
    monkeypatch.setattr(mail_notices.mail_rules, "classify",
                        lambda addr, p, is_supplier: "internal" if addr.endswith("@yourcompany.com") else "external")
    monkeypatch.setattr(mail_notices, "_mailbox_ready", lambda conn: w.ready)

    def create(conn, **kw):
        w.created.append(kw)
        return {"internal_email_id": 42}
    monkeypatch.setattr(mail_notices.internal_email, "create", create)
    return w


def _go(ses, to=("ann@yourcompany.com",)):
    return mail_notices.notify(object(), for_user="ann@yourcompany.com", recipients=list(to),
                               subject="s", body="b", source_ref="digest", email_service=ses)


def test_policy_off_sends_from_procwise(world):
    world.policy = _policy("off")
    ses = FakeSES()
    assert _go(ses)["route"] == "procwise_address"
    assert len(ses.calls) == 1 and not world.created


def test_policy_on_connected_internal_goes_pending_for_the_user(world):
    ses = FakeSES()
    out = _go(ses)
    assert out == {"route": "pending_approval", "internal_email_id": 42}
    assert not ses.calls
    assert world.created[0]["author"] == "ann@yourcompany.com" and world.created[0]["kind"] == "notice"


def test_not_connected_sends_from_procwise(world):
    world.ready = False
    ses = FakeSES()
    assert _go(ses)["route"] == "procwise_address"
    assert len(ses.calls) == 1 and not world.created


def test_unreadable_policy_still_sends(world):
    world.policy = mail_policy.PolicyUnavailable("gone")
    ses = FakeSES()
    assert _go(ses)["route"] == "procwise_address"
    assert len(ses.calls) == 1


def test_external_recipient_sends_from_procwise(world):
    ses = FakeSES()
    assert _go(ses, ("ann@yourcompany.com", "bob@supplier.com"))["route"] == "procwise_address"
    assert len(ses.calls) == 1 and not world.created


def test_failure_storing_does_not_lose_the_notice(world, monkeypatch):
    def boom(conn, **kw):
        raise RuntimeError("db")
    monkeypatch.setattr(mail_notices.internal_email, "create", boom)
    ses = FakeSES()
    assert _go(ses)["route"] == "procwise_address"
    assert len(ses.calls) == 1


def test_service_subject_without_at_sends_from_procwise(world):
    ses = FakeSES()
    out = mail_notices.notify(object(), for_user="digest-service", recipients=["ann@yourcompany.com"],
                              subject="s", body="b", source_ref="d", email_service=ses)
    assert out["route"] == "procwise_address" and len(ses.calls) == 1 and not world.created


@pytest.mark.parametrize("who", ["@yourcompany.com", "ann@", "", None])
def test_half_addresses_are_not_sign_in_emails(world, who):
    ses = FakeSES()
    out = mail_notices.notify(object(), for_user=who, recipients=["ann@yourcompany.com"],
                              subject="s", body="b", source_ref="d", email_service=ses)
    assert out["route"] == "procwise_address" and not world.created


def test_database_down_still_sends_from_procwise(world):
    def down():
        raise RuntimeError("db down")
    ses = FakeSES()
    out = mail_notices.notify(None, conn_factory=down, for_user="ann@yourcompany.com",
                              recipients=["ann@yourcompany.com"], subject="s", body="b",
                              source_ref="d", email_service=ses)
    assert out["route"] == "procwise_address" and len(ses.calls) == 1


def test_colleague_emails_off_sends_notices_from_procwise(world, caplog):
    """F2: notices queue as emails to colleagues, which internal_send=off refuses; the
    notice must go from the Procwise address, decided up front, not lost or queued."""
    import logging
    world.policy = _policy("with_user_approval", internal_send="off")
    ses = FakeSES()
    with caplog.at_level(logging.INFO, logger=mail_notices.logger.name):
        out = _go(ses)
    assert out["route"] == "procwise_address" and len(ses.calls) == 1 and not world.created
    assert any("emails to colleagues are switched off" in r.getMessage() for r in caplog.records)
