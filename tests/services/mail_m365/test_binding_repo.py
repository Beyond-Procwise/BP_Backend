import os
import pytest

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="live DB test; set PROCWISE_TEST_LIVE_DB=1")

from services.style.mailbox import (ORG_USER_REF, PROVIDER_GRAPH, ROLE_SEND_RECEIVE,
                                    MailboxBindingRepository, normalise_address)

ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/app-AbCdEf"


@pytest.fixture
def repo():
    from dotenv import load_dotenv
    load_dotenv(".env")
    conn = psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
                            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"))
    conn.autocommit = False
    yield MailboxBindingRepository(conn)
    conn.rollback()
    conn.close()


def _active(repo, address):
    b = repo.create(user_ref=ORG_USER_REF, provider=PROVIDER_GRAPH,
                    mailbox_address=address, role=ROLE_SEND_RECEIVE, credential_ref=ARN)
    return repo.verify_scope(b.binding_id, control_mailbox="control@example.test",
                             probe=lambda _a: (False, "denied with HTTP 403"))


def test_normalise_address():
    assert normalise_address("  Procurement@Customer.COM ") == "procurement@customer.com"


def test_create_stores_normalised_address(repo):
    b = repo.create(user_ref=ORG_USER_REF, provider=PROVIDER_GRAPH,
                    mailbox_address="  M365-R0@Example.TEST ", role=ROLE_SEND_RECEIVE,
                    credential_ref=ARN)
    assert b.mailbox_address == "m365-r0@example.test"


def test_send_receive_binding_can_send_and_read(repo):
    b = _active(repo, "m365-r1@example.test")
    assert b.can_send and b.can_read and not b.is_default


def test_default_needs_test_send_first(repo):
    b = _active(repo, "m365-r2@example.test")
    with pytest.raises(ValueError, match="test email"):
        repo.set_default(b.binding_id)


def test_set_default_moves_the_flag(repo):
    a = repo.record_test_send(_active(repo, "m365-r3@example.test").binding_id)
    b = repo.record_test_send(_active(repo, "m365-r4@example.test").binding_id)
    repo.set_default(a.binding_id)
    repo.set_default(b.binding_id)
    assert repo.get_default_sender().binding_id == b.binding_id
    assert repo.get(a.binding_id).is_default is False


def test_deactivate_clears_default(repo):
    a = repo.record_test_send(_active(repo, "m365-r5@example.test").binding_id)
    repo.set_default(a.binding_id)
    repo.deactivate(a.binding_id, reason="removed by admin")
    assert repo.get_default_sender() is None
    assert repo.get(a.binding_id).is_active is False


def test_org_default_any_survives_deactivate(repo):
    a = repo.record_test_send(_active(repo, "m365-r7@example.test").binding_id)
    repo.set_default(a.binding_id)
    repo.deactivate(a.binding_id, reason="disconnect")
    assert repo.get_default_sender() is None
    got = repo.get_org_default_any()
    assert got is not None and got.is_active is False


def test_org_default_any_none_without_test_send(repo):
    _active(repo, "m365-r8@example.test")
    existing = repo.get_org_default_any()
    # a never-tested binding must not count (other rows may exist in the shared DB)
    assert existing is None or existing.last_test_send_at is not None


def test_list_org_excludes_user_bindings(repo):
    _active(repo, "m365-r6@example.test")
    assert all(b.user_ref == ORG_USER_REF for b in repo.list_org())


def test_deactivate_forget_verification_clears_only_the_proof_time(repo):
    b = _active(repo, "m365-forget@example.test")
    repo.record_test_send(b.binding_id)
    kept = repo.deactivate(b.binding_id, reason="health")
    assert kept.scope_verified_at is not None and not kept.is_active
    gone = repo.deactivate(b.binding_id, reason="removed", forget_verification=True)
    assert gone.scope_verified_at is None and gone.scope_evidence_ref == b.scope_evidence_ref
    assert gone.last_test_send_at is not None      # still keeps supplier email held
