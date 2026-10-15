"""Email policy: strict read, validated save, optimistic lock, history."""

import os

import pytest

from src.services import mail_policy as mp
from src.services.db import get_conn

LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not LIVE, reason="needs a real PostgreSQL (PROCWISE_TEST_LIVE_DB=1)")


class _Cur:
    def __init__(self, row):
        self.row = row

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self.row


class _Conn:
    def __init__(self, row):
        self.row = row

    def cursor(self):
        return _Cur(self.row)


@pytest.fixture
def fake_conn_with_rules():
    def make(rules):
        return _Conn((1, 1, {"policy_identifier": "email_policy", "rules": rules, "history": []}))
    return make


@pytest.fixture
def live_policy():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT policy_id, policy_details, version, last_modified_by, last_modified_date
                         FROM proc.bp_policy WHERE policy_name='email_policy' AND policy_status=1""")
        pid, details, version, by, when = cur.fetchone()
    import json
    try:
        yield mp.load()
    finally:
        with get_conn() as conn:
            conn.cursor().execute(
                """UPDATE proc.bp_policy SET policy_details=%s::jsonb, version=%s,
                          last_modified_by=%s, last_modified_date=%s WHERE policy_id=%s""",
                (json.dumps(details), version, by, when, pid))


def test_load_raises_when_a_field_is_missing(fake_conn_with_rules):
    with pytest.raises(mp.PolicyUnavailable):
        mp.load(fake_conn_with_rules({"internal_send": "off"}))


def test_load_raises_when_a_value_is_invalid(fake_conn_with_rules):
    rules = {"internal_send": "sometimes", "company_domains": [], "external_recipients": "any",
             "read_scope": "whole_inbox", "on_disconnect": "hold", "access_level": "read_send",
             "notices_from_mailbox": "off", "poll_seconds": 120, "restriction_group": "G"}
    with pytest.raises(mp.PolicyUnavailable, match="internal_send"):
        mp.load(fake_conn_with_rules(rules))


def test_domains_are_lower_cased_trimmed_and_deduplicated():
    assert mp.FIELDS["company_domains"]([" YourCompany.com", "@yourcompany.com"]) == ("yourcompany.com",)


def test_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="unknown setting: nope"):
        mp.save({"nope": 1}, by="admin@x", expected_version=1)


@pytest.mark.parametrize("field,bad", [("poll_seconds", 5), ("poll_seconds", 99999),
                                       ("restriction_group", "bad/name"),
                                       ("company_domains", ["not a domain"]),
                                       ("read_scope", "everything")])
def test_validators_refuse_bad_values(field, bad):
    with pytest.raises(ValueError):
        mp.FIELDS[field](bad)


@live
def test_save_refuses_internal_send_without_domains(live_policy):
    with pytest.raises(ValueError, match="company domain"):
        mp.save({"company_domains": []}, by="admin@x", expected_version=live_policy.version)
    assert mp.load().version == live_policy.version


@live
def test_save_writes_one_history_entry_per_changed_field(live_policy):
    p = mp.save({"company_domains": ["yourcompany.com"], "read_scope": "whole_inbox"},
                by="admin@x", expected_version=live_policy.version)
    new = [h for h in p.history if h["version"] == p.version]
    assert {h["field"] for h in new} == {"company_domains", "read_scope"}
    assert all(h["changed_by"] == "admin@x" and "before" in h and "after" in h for h in new)
    assert p.version == live_policy.version + 1


@live
def test_concurrent_save_conflicts_and_writes_nothing(live_policy):
    mp.save({"poll_seconds": 300, "company_domains": ["yourcompany.com"]}, by="a@x", expected_version=live_policy.version)
    with pytest.raises(mp.PolicyConflict):
        mp.save({"poll_seconds": 600}, by="b@x", expected_version=live_policy.version)
    now = mp.load()
    assert now.poll_seconds == 300
    assert len([h for h in now.history if h["field"] == "poll_seconds"]) == 1


@live
def test_a_failed_audit_write_means_the_save_does_not_apply(live_policy, monkeypatch):
    def boom(**kw):
        raise RuntimeError("audit down")
    monkeypatch.setattr(mp, "record_action_or_fail", boom)
    with pytest.raises(RuntimeError, match="audit down"):
        mp.save({"poll_seconds": 777, "company_domains": ["yourcompany.com"]}, by="audit@x", expected_version=live_policy.version)
    now = mp.load()
    assert now.poll_seconds == live_policy.poll_seconds
    assert now.version == live_policy.version
    assert now.company_domains == live_policy.company_domains
