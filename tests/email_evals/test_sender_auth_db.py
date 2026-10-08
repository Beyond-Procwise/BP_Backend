"""Checking an inbound reply's sender, recording it, and holding it when the rules say so. Real Postgres."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.services.draft_assurance import inbound, queues, sender_auth as SA
from tests.email_evals.test_learning import db, engine  # noqa: F401  (fixtures)

SES_OK = "amazonses.com; spf=pass smtp.mailfrom=alex@acme.test; dkim=pass header.i=@acme.test; dmarc=pass header.from=acme.test;"
SES_BAD = "amazonses.com; spf=fail smtp.mailfrom=alex@acme.test; dkim=none; dmarc=fail header.from=acme.test;"


@pytest.fixture
def clean(db):
    with db.cursor() as cur:
        cur.execute("TRUNCATE email_agent.bp_inbound_auth, email_agent.bp_inbound_flag RESTART IDENTITY")
        cur.execute("TRUNCATE proc.bp_supplier")
        cur.execute("INSERT INTO proc.bp_supplier (supplier_id, supplier_name, contact_email_1, contact_email_2) "
                    "VALUES ('S-1', 'Acme Ltd', 'alex@acme.test', 'billing@mail.acme.test')")
    return db


def rules(**over):
    base = {"mode": "enforce", "trusted_authserv_ids": ["amazonses.com"], "hold_on_fail": True, "hold_on_missing": False, "hold_on_domain_mismatch": False}
    base.update(over)
    return type("E", (), {"get_policy": staticmethod(lambda slug: {"details": {"rules": base}, "version": 1})})()


def reply(auth=SES_OK, frm="Alex <alex@acme.test>", mid="<m1>", supplier="S-1", headers=None, **over):
    h = headers if headers is not None else {"From": (frm,), "Authentication-Results": (auth,)} if auth else {"From": (frm,)}
    return SimpleNamespace(workflow_id="wf-1", unique_id="PROC-WF-1", supplier_id=supplier, response_message_id=mid, response_from=frm, raw_headers=h, **over)


def factory(conn):
    @contextmanager
    def f():
        yield conn
    return f


def check(conn, row, **r):
    return SA.check_and_record(row, factory(conn), rules(**r))


def auth_rows(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT verdict, spf, dkim, dmarc, from_domain, domain_match, held, reasons, trusted_headers, ignored_headers FROM email_agent.bp_inbound_auth ORDER BY auth_id")
        return cur.fetchall()


def flags(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT kind, status, kinds, terms FROM email_agent.bp_inbound_flag ORDER BY flag_id")
        return cur.fetchall()


def test_an_authenticated_reply_from_the_suppliers_domain_is_recorded_and_not_held(clean):
    assert check(clean, reply()) is None
    ((verdict, spf, dkim, dmarc, domain, match, held, reasons, trusted, ignored),) = auth_rows(clean)
    assert (verdict, spf, dkim, dmarc, domain, match, held, reasons, trusted, ignored) == ("authenticated", "pass", "pass", "pass", "acme.test", True, False, [], 1, 0)
    assert flags(clean) == []


def test_a_failed_reply_is_held_as_a_flag_that_blocks_its_thread(clean):
    fid = check(clean, reply(auth=SES_BAD))
    assert isinstance(fid, int)
    assert flags(clean) == [("sender_not_verified", "open", ["auth_failed"], [])]
    assert auth_rows(clean)[0][0] == "failed" and auth_rows(clean)[0][6] is True
    blocking = inbound.blocking_flags(clean, "wf-1", "S-1")
    assert [f["id"] for f in blocking] == [fid]                      # the same blocking path as a payment-detail flag


def test_shadow_mode_records_the_failure_and_holds_nothing(clean):
    assert check(clean, reply(auth=SES_BAD), mode="shadow") is None
    assert flags(clean) == [] and auth_rows(clean)[0][0] == "failed" and auth_rows(clean)[0][6] is False


def test_a_reply_with_nothing_stamped_is_recorded_missing_and_held_only_if_that_switch_is_on(clean):
    assert check(clean, reply(auth=None)) is None
    assert auth_rows(clean)[0][0] == "missing" and flags(clean) == []
    assert check(clean, reply(auth=None, mid="<m2>"), hold_on_missing=True)
    assert flags(clean)[0][2] == ["auth_missing"]


def test_a_forged_authentication_line_inside_the_message_is_ignored_and_counted(clean):
    forged = reply(headers={"From": ("Alex <alex@acme.test>",), "Authentication-Results": ("evil.example; spf=pass; dkim=pass; dmarc=pass",)})
    check(clean, forged, hold_on_missing=True)
    row = auth_rows(clean)[0]
    assert row[0] == "missing" and row[8] == 0 and row[9] == 1       # nothing trusted, one untrusted line ignored
    assert flags(clean)[0][2] == ["auth_missing"]                     # so it is treated as unauthenticated, not as a pass


def test_a_lookalike_domain_that_authenticates_perfectly_is_caught_by_the_domain_check(clean):
    lookalike = reply(frm="Alex <alex@acme-test.example>", auth="amazonses.com; spf=pass; dkim=pass; dmarc=pass")
    assert check(clean, lookalike) is None                            # off by default: recorded, not held
    verdict, spf, dkim, dmarc, domain, match, held, reasons, trusted, ignored = auth_rows(clean)[0]
    assert (verdict, domain, match, held, reasons) == ("authenticated", "acme-test.example", False, False, ["domain_mismatch"])
    clean.cursor().execute("TRUNCATE email_agent.bp_inbound_auth")
    fid = check(clean, lookalike, hold_on_domain_mismatch=True)
    assert fid and flags(clean)[0][2] == ["domain_mismatch"]


def test_a_subdomain_of_a_known_supplier_domain_matches(clean):
    check(clean, reply(frm="Billing <billing@mail.acme.test>"))
    assert auth_rows(clean)[0][5] is True


def test_a_supplier_the_master_does_not_know_is_not_compared_and_never_held_for_it(clean):
    assert check(clean, reply(supplier="S-404"), hold_on_domain_mismatch=True) is None
    assert auth_rows(clean)[0][5] is None and flags(clean) == []
    clean.cursor().execute("TRUNCATE email_agent.bp_inbound_auth")
    assert check(clean, reply(supplier=None, mid="<m3>"), hold_on_domain_mismatch=True) is None and auth_rows(clean)[0][5] is None


def test_checking_the_same_message_twice_records_and_flags_it_once(clean):
    first = check(clean, reply(auth=SES_BAD))
    again = check(clean, reply(auth=SES_BAD))
    assert first and again is None and len(auth_rows(clean)) == 1 and len(flags(clean)) == 1


def test_the_table_holds_domains_and_result_words_but_no_address_and_no_header_text(clean):
    check(clean, reply(auth=SES_BAD))
    with clean.cursor() as cur:
        cur.execute("SELECT t::text FROM email_agent.bp_inbound_auth t")
        stored = cur.fetchone()[0]
        cur.execute("SELECT t::text FROM email_agent.bp_inbound_flag t")
        flagged = cur.fetchone()[0]
    for text in (stored, flagged):
        for leaked in ("alex@acme.test", "smtp.mailfrom", "header.from", "Alex <"):
            assert leaked not in text, leaked


def test_a_held_reply_appears_in_the_review_queue_with_words_not_codes(clean):
    check(clean, reply(auth=SES_BAD))
    (item,) = queues.list_inbound_flags(clean)
    assert item["what"] == "The sender failed authentication" and item["status"] == "open"
    assert "auth_failed" not in json.dumps(item["what"])


def test_the_hold_can_be_cleared_only_through_the_same_decision_as_any_flag(clean):
    fid = check(clean, reply(auth=SES_BAD))
    assert inbound.decide_flag(clean, fid, "ana", "clear", "rang the supplier")["ok"] is True
    assert inbound.blocking_flags(clean, "wf-1", "S-1") == []


# --- it must never cost a reply ------------------------------------------------------------------------------------------------

def test_no_rules_means_the_check_does_not_run_and_nothing_is_recorded(clean):
    none = type("E", (), {"get_policy": staticmethod(lambda slug: None)})()
    assert SA.check_and_record(reply(auth=SES_BAD), factory(clean), none) is None and auth_rows(clean) == []


def test_bad_rules_also_mean_no_check(clean):
    assert check(clean, reply(auth=SES_BAD), mode="sometimes") is None and auth_rows(clean) == []


def test_a_broken_connection_a_missing_table_and_a_nonsense_row_never_raise(clean):
    @contextmanager
    def broken():
        raise RuntimeError("database down")
        yield
    assert SA.check_and_record(reply(auth=SES_BAD), broken, rules()) is None
    assert SA.check_and_record(None, factory(clean), rules()) is None
    assert SA.check_and_record(SimpleNamespace(), factory(clean), rules()) is None
    with clean.cursor() as cur:
        cur.execute("ALTER TABLE email_agent.bp_inbound_auth RENAME TO bp_inbound_auth_away")
    try:
        assert check(clean, reply(auth=SES_BAD)) is None
    finally:
        with clean.cursor() as cur:
            cur.execute("ALTER TABLE email_agent.bp_inbound_auth_away RENAME TO bp_inbound_auth")


def test_the_real_policy_row_is_valid_and_ships_with_only_the_unambiguous_switch_on(db, engine):
    r = SA.load_rules(engine)
    assert r == {"mode": "enforce", "trusted_authserv_ids": ["amazonses.com"], "hold_on_fail": True,
                 "hold_on_missing": False, "hold_on_domain_mismatch": False}


def test_the_writer_can_record_results_but_never_change_or_delete_them(db):
    with db.cursor() as cur:
        cur.execute("SELECT has_table_privilege('public', 'email_agent.bp_inbound_auth', 'SELECT,INSERT,UPDATE,DELETE')")
        assert cur.fetchone()[0] is False
