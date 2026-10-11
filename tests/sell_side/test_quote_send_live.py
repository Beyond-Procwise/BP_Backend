"""Sending a quote to its customer, which is what issues it (QDA B12).

Live database: what is tested is an issue that is undone when the send cannot happen, a row
lock, and the record that the email went. The mail relay is a stand-in here -- no test
sends an email. Every quote is a LIVETEST one, numbered from a LIVETEST series.
"""
import datetime as dt
import hashlib
import io
import os
import threading
import time
import uuid
from decimal import Decimal as D
from types import SimpleNamespace

import psycopg2
import pytest

from src.services import agent_actions, email_dispatch_guard
from src.services.rga import quote_pipeline
from src.services.rga.models import Finding, FindingCode
from src.services.sell_side import accounts, quote_send, quotes
from src.services.sell_side._db import NotFound, StateConflict
from tests.sell_side.conftest import SENTINEL, live, seed_item

pytestmark = live
TODAY = dt.date.today()
YEAR = TODAY.year
SERIES = f"{SENTINEL}_sales_quote"
SENDER = "quotes@livetest.example"
BUYER = "buyer@livetest.example"
ACCOUNT = "LIVETEST-S"
NUMBER_1 = f"{SENTINEL}-Q-{YEAR}-0001"


class _Relay:
    """Stands in for the mail relay and keeps what it was handed."""

    def __init__(self, success=True, raises=None, delay=0.0):
        self.calls, self.success, self.raises, self.delay = [], success, raises, delay

    def __call__(self, **kw):
        self.calls.append(kw)
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return SimpleNamespace(success=self.success, message_id=f"<m{len(self.calls)}@livetest>")


class _Engine:
    """A policy engine that knows only the daily email limit."""

    def __init__(self, max_per_day):
        self.max_per_day = max_per_day

    def get_policy(self, slug):
        if slug == "email_volume" and self.max_per_day is not None:
            return {"details": {"rules": {"max_per_user_per_day": self.max_per_day}}}
        return None


NO_LIMIT = _Engine(None)


@pytest.fixture
def person():
    """A signed-in sender nobody else is. The audit table is append-only, so the email.send
    rows a test writes stay; a name used once keeps them out of anyone's daily count."""
    return SimpleNamespace(subject=f"{SENTINEL}-sender-{uuid.uuid4().hex[:10]}")


def _connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"], port=os.getenv("DB_PORT", 5432),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=10)


def _quote(conn, dist, *, email=BUYER, contact=True, status="approved", name="Pat Buyer",
           account_name="Livetest Sends Ltd"):
    """One quote for a LIVETEST account, taken to `status`. Returns (quote id, contact id)."""
    accounts.create_account(conn, account_id=ACCOUNT, account_name=account_name)
    contact_id = None
    if contact:
        contact_id = accounts.add_contact(conn, ACCOUNT, contact_name=name,
                                          contact_email=email)["contact_id"]
    item = seed_item(conn, dist, "LIVETEST-S1", cost="10.3700")
    q = quotes.create_draft(
        conn, account_id=ACCOUNT, currency="GBP", created_by="sub-author",
        contact_id=contact_id, valid_until=TODAY + dt.timedelta(days=30),
        lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00")}])
    qid = q["sales_quote_id"]
    if status in ("in_review", "approved"):
        quotes.submit(conn, qid, actor="sub-author")
    if status == "approved":
        quotes.approve(conn, qid, approver="sub-approver")
    return qid, contact_id


def _send(conn, qid, person, relay, **kw):
    kw.setdefault("policy_engine", NO_LIMIT)
    kw.setdefault("sender", SENDER)
    return quote_send.send(conn, qid, principal=person, mailer=relay, **kw)


def _last(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT last_value FROM proc.bp_doc_number_counter "
                    "WHERE doc_type = %s AND year = %s", (SERIES, YEAR))
        row = cur.fetchone()
    conn.rollback()
    return row[0] if row else None


def _events(conn, qid, event):
    with conn.cursor() as cur:
        cur.execute("SELECT details FROM proc.bp_agent_actions WHERE doc_type = 'sales_quote' "
                    "AND doc_pk = %s AND action_type = %s ORDER BY created_at", (str(qid), event))
        rows = [r[0] for r in cur.fetchall()]
    conn.rollback()
    return rows


def _state(conn, qid):
    q = quotes.get_quote(conn, qid)
    conn.rollback()
    return q["status"], q["quote_number"]


def _refused(conn, qid, person, relay, check_id, **kw):
    """The send is refused on `check_id`, nothing went, and the quote is as it was."""
    before = _state(conn, qid)
    with pytest.raises(quote_send.SendRefused) as exc:
        _send(conn, qid, person, relay, **kw)
    failing = [c["id"] for c in exc.value.checks if not c["passed"]]
    assert check_id in failing, failing
    assert relay.calls == []
    assert _state(conn, qid) == before
    return exc.value


# --- the send ----------------------------------------------------------------

def test_sending_an_approved_quote_issues_it_and_emails_its_contact(live_db, person):
    conn, dist = live_db
    qid, contact_id = _quote(conn, dist)
    relay = _Relay()
    out = _send(conn, qid, person, relay)

    assert _state(conn, qid) == ("issued", NUMBER_1)
    assert out["status"] == "sent" and out["quote_number"] == NUMBER_1
    assert out["to"] == BUYER and out["email_id"] == "<m1@livetest>"
    assert out["quote"]["status"] == "issued"
    (mail,) = relay.calls
    assert mail["recipients"] == [BUYER] and mail["sender"] == SENDER
    assert mail["subject"] == f"Quote {NUMBER_1}"
    assert mail["body"] == (f"<p>Dear Pat Buyer,</p><p>Please find attached quote {NUMBER_1} "
                            "for Livetest Sends Ltd.</p>")
    ((content, filename),) = mail["attachments"]
    assert filename == f"Quote-{NUMBER_1}.pdf" and content[:5] == b"%PDF-"


def test_the_attachment_is_the_checked_quote_with_its_number_and_no_cost(live_db, person):
    import pdfplumber

    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    relay = _Relay()
    _send(conn, qid, person, relay)
    ((content, _),) = relay.calls[0]["attachments"]
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    assert NUMBER_1 in text and "Total excluding tax £56" in text
    assert "10.37" not in text and "41.48" not in text       # what it costs us, per unit and in all


def test_the_cover_carries_no_price_date_or_reference_of_ours(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    relay = _Relay()
    _send(conn, qid, person, relay)
    q = quotes.get_quote(conn, qid)
    cover = relay.calls[0]["subject"] + relay.calls[0]["body"]
    for ours in ("56.00", "14.00", "10.37", q["quote_ref"], f"{q['valid_until']:%Y}-",
                 "GBP", "£"):
        assert ours not in cover.replace(NUMBER_1, "")


def test_what_was_sent_is_on_record_with_the_hash_of_the_file(live_db, person):
    conn, dist = live_db
    qid, contact_id = _quote(conn, dist)
    relay = _Relay()
    out = _send(conn, qid, person, relay)
    ((content, filename),) = relay.calls[0]["attachments"]
    (sent,) = _events(conn, qid, quote_send.SENT_EVENT)
    assert sent["artefact"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert sent["artefact"]["bytes"] == len(content) and sent["artefact"]["filename"] == filename
    assert sent["artefact"]["pack_hash"] and sent["artefact"]["ast_hash"]
    assert sent["artefact"]["renderer"] == "quote_pdf"
    assert (sent["actor"], sent["to"], sent["contact_id"]) == (person.subject, BUYER, contact_id)
    assert sent["email_id"] == "<m1@livetest>" and sent["quote_number"] == NUMBER_1
    assert out["artefact"] == sent["artefact"]
    (issued,) = _events(conn, qid, "sales_quote.issued")
    assert issued["quote_number"] == NUMBER_1 and issued["actor"] == person.subject


def test_a_sent_quote_counts_against_the_senders_daily_limit(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    assert email_dispatch_guard._daily_send_count(conn, person.subject) == 0
    _send(conn, qid, person, _Relay())
    assert email_dispatch_guard._daily_send_count(conn, person.subject) == 1
    conn.rollback()


# --- who it goes to ----------------------------------------------------------

def test_a_quote_with_no_contact_is_not_sent_and_not_issued(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, contact=False)
    _refused(conn, qid, person, _Relay(), "has_recipient")
    assert _last(conn) is None                              # the number went back


def test_a_contact_with_no_address_is_not_sent_to(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, email=None)
    _refused(conn, qid, person, _Relay(), "has_recipient")
    assert _last(conn) is None


@pytest.mark.parametrize("address", [
    "buyer@livetest.example\nBcc: someone@else.example",
    "buyer@livetest.example, someone@else.example",
    "Pat <buyer@livetest.example>",
    "not-an-address",
    "buyer@livetest",
])
def test_an_address_that_is_not_one_plain_address_is_not_sent_to(live_db, person, address):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, email=address)
    _refused(conn, qid, person, _Relay(), "has_recipient")


def test_another_contact_on_the_same_account_may_be_chosen(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    other = accounts.add_contact(conn, ACCOUNT, contact_name="Sam Second",
                                 contact_email="sam@livetest.example")["contact_id"]
    relay = _Relay()
    out = _send(conn, qid, person, relay, contact_id=other)
    assert out["to"] == "sam@livetest.example"
    assert relay.calls[0]["recipients"] == ["sam@livetest.example"]
    assert "Dear Sam Second," in relay.calls[0]["body"]


def test_a_contact_on_another_account_is_refused(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    accounts.create_account(conn, account_id="LIVETEST-T", account_name="Livetest Other Ltd")
    stranger = accounts.add_contact(conn, "LIVETEST-T", contact_name="Not Theirs",
                                    contact_email="stranger@livetest.example")["contact_id"]
    refused = _refused(conn, qid, person, _Relay(), "has_recipient", contact_id=stranger)
    assert "not on this quote's account" in str(refused)


# --- when it is refused ------------------------------------------------------

@pytest.mark.parametrize("status", ["draft", "in_review"])
def test_a_quote_that_is_not_approved_is_not_sent(live_db, person, status):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, status=status)
    _refused(conn, qid, person, _Relay(), "quote_approved")
    assert _last(conn) is None


def test_a_quote_that_does_not_exist_is_not_found(live_db, person):
    conn, _ = live_db
    with pytest.raises(NotFound):
        _send(conn, 2_000_000_000, person, _Relay())


def test_nobody_signed_in_means_nothing_is_sent(live_db):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    for nobody in (None, SimpleNamespace(subject=""), SimpleNamespace(subject="  ")):
        _refused(conn, qid, nobody, _Relay(), "sent_by_a_person")
    assert _last(conn) is None


def test_no_sending_address_means_nothing_is_sent(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    _refused(conn, qid, person, _Relay(), "mailbox_can_send", sender=None)
    assert _last(conn) is None


def test_a_sender_at_their_daily_limit_is_refused(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    _refused(conn, qid, person, _Relay(), "within_daily_limit", policy_engine=_Engine(0))
    assert _last(conn) is None
    relay = _Relay()
    _send(conn, qid, person, relay, policy_engine=_Engine(1))      # one is allowed
    assert len(relay.calls) == 1


def test_an_expired_quote_is_not_sent(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_quote SET quote_date = %s, valid_until = %s "
                    "WHERE sales_quote_id = %s",
                    (TODAY - dt.timedelta(days=9), TODAY - dt.timedelta(days=1), qid))
    conn.commit()
    relay = _Relay()
    with pytest.raises(StateConflict, match="expired"):
        _send(conn, qid, person, relay)
    assert relay.calls == [] and _state(conn, qid) == ("approved", None) and _last(conn) is None


def test_a_quote_whose_document_fails_its_checks_is_not_sent_or_issued(live_db, person,
                                                                       monkeypatch):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)

    def _blocked(*a, **k):
        raise quote_pipeline.QuoteBlocked([Finding(
            finding_id="F1", code=FindingCode.QUOTE_TOTAL_MISMATCH, severity="HIGH",
            detail="the total is off")])

    monkeypatch.setattr(quote_pipeline, "produce", _blocked)
    refused = _refused(conn, qid, person, _Relay(), "named_docs_attached")
    assert "the total is off" in str(refused)
    assert _last(conn) is None


def test_a_refused_send_is_on_record_with_what_refused_it(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, contact=False)
    _refused(conn, qid, person, _Relay(), "has_recipient")
    (blocked,) = _events(conn, qid, quote_send.BLOCKED_EVENT)
    assert blocked["failing"] == ["has_recipient"] and blocked["actor"] == person.subject


def test_an_issued_quote_with_no_number_is_not_sent(live_db, person):
    """Quotes issued before numbering existed. A customer is not sent a quote with no number."""
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_quote SET status = 'issued', issued_at = now() "
                    "WHERE sales_quote_id = %s", (qid,))
    conn.commit()
    _refused(conn, qid, person, _Relay(), "quote_numbered")


# --- once, and again after a failure ----------------------------------------

def test_a_quote_is_sent_once(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    relay = _Relay()
    _send(conn, qid, person, relay)
    with pytest.raises(quote_send.SendRefused) as exc:
        _send(conn, qid, person, relay)
    assert [c["id"] for c in exc.value.checks if not c["passed"]] == ["not_already_sent"]
    assert len(relay.calls) == 1 and _last(conn) == 1


def test_an_email_the_relay_does_not_take_leaves_the_quote_issued_and_unsent(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    with pytest.raises(quote_send.SendFailed, match="issued and has not been sent"):
        _send(conn, qid, person, _Relay(success=False))
    assert _state(conn, qid) == ("issued", NUMBER_1)
    assert _events(conn, qid, quote_send.SENT_EVENT) == []
    assert len(_events(conn, qid, quote_send.FAILED_EVENT)) == 1

    relay = _Relay()                                         # the same call, again
    out = _send(conn, qid, person, relay)
    assert out["quote_number"] == NUMBER_1 and len(relay.calls) == 1
    assert f"Quote-{NUMBER_1}.pdf" == relay.calls[0]["attachments"][0][1]
    assert _last(conn) == 1                                  # one quote, one number
    assert len(_events(conn, qid, quote_send.SENT_EVENT)) == 1


def test_a_relay_that_breaks_leaves_the_quote_unsent_and_sendable(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    with pytest.raises(ConnectionError):
        _send(conn, qid, person, _Relay(raises=ConnectionError("relay down")))
    assert _state(conn, qid) == ("issued", NUMBER_1)
    assert _events(conn, qid, quote_send.SENT_EVENT) == []
    assert _send(conn, qid, person, _Relay())["status"] == "sent"


def test_an_email_that_cannot_be_recorded_beforehand_does_not_go(live_db, person, monkeypatch):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    real = agent_actions.record_action_or_fail

    def _no_email_rows(**kw):
        if kw["action_type"] == "email.send":
            raise agent_actions.AuditWriteError("the audit spine is down")
        return real(**kw)

    monkeypatch.setattr(agent_actions, "record_action_or_fail", _no_email_rows)
    relay = _Relay()
    with pytest.raises(agent_actions.AuditWriteError):
        _send(conn, qid, person, relay)
    assert relay.calls == []
    assert _events(conn, qid, quote_send.SENT_EVENT) == []


def test_two_sends_at_once_send_one_email(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    relay, start, sent, refused, errors = _Relay(delay=0.5), threading.Barrier(2), [], [], []

    def _go():
        own = _connect()
        try:
            start.wait(timeout=20)
            sent.append(_send(own, qid, person, relay))
        except quote_send.SendRefused as exc:
            refused.append([c["id"] for c in exc.checks if not c["passed"]])
        except Exception as exc:                            # pragma: no cover - reported below
            errors.append(exc)
        finally:
            own.close()

    threads = [threading.Thread(target=_go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert errors == []
    assert len(relay.calls) == 1 and len(sent) == 1
    assert refused == [["not_already_sent"]]
    assert _last(conn) == 1 and len(_events(conn, qid, quote_send.SENT_EVENT)) == 1


# --- the cover ---------------------------------------------------------------

def test_an_account_with_a_number_in_its_name_is_still_sent(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, account_name="Livetest Unit 4 Fixings", name="Pat 2nd")
    relay = _Relay()
    _send(conn, qid, person, relay)
    assert "for Livetest Unit 4 Fixings." in relay.calls[0]["body"]


@pytest.mark.parametrize("extra, check_id", [
    (" It is valid until 30 November 2026.", "figures_traced"),
    (" The total is 56.00.", "figures_traced"),
    (" Our unit cost is 10.37.", "figures_traced"),
    (" Pay into GB29 NWBK 6016 1331 9268 19.", "no_bank_details"),
])
def test_a_cover_that_carries_a_figure_is_refused(live_db, person, monkeypatch, extra, check_id):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    real = quote_send.body_template
    monkeypatch.setattr(quote_send, "body_template", lambda q, c: real(q, c) + extra)
    _refused(conn, qid, person, _Relay(), check_id)
    assert _last(conn) is None


def test_a_cover_that_carries_our_draft_reference_is_refused(live_db, person, monkeypatch):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    real = quote_send.body_template
    monkeypatch.setattr(quote_send, "body_template",
                        lambda q, c: real(q, c) + f" (our ref {q['quote_ref']})")
    _refused(conn, qid, person, _Relay(), "no_internal_ids")


def test_a_name_cannot_carry_markup_into_the_email(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, name="Pat <img src=x onerror=alert()>")
    relay = _Relay()
    _send(conn, qid, person, relay)
    assert "<img" not in relay.calls[0]["body"] and "&lt;img" in relay.calls[0]["body"]


# --- the draft ---------------------------------------------------------------

def test_the_draft_is_built_from_the_quote_and_changes_nothing(live_db, person):
    conn, dist = live_db
    qid, contact_id = _quote(conn, dist)
    d = quote_send.draft(conn, qid, principal=person, sender=SENDER, policy_engine=NO_LIMIT)
    assert d["can_send"] is True and all(c["passed"] for c in d["checks"])
    assert (d["kind"], d["counterparty_kind"]) == ("quote", "customer")
    assert d["supplier"] == {"id": ACCOUNT, "name": "Livetest Sends Ltd"}
    assert d["recipients"] == {"to": [BUYER]}
    assert d["contact"] == {"contact_id": contact_id, "contact_name": "Pat Buyer"}
    assert d["body_template"] == ("Dear Pat Buyer,\n\nPlease find attached quote {{ref:q1}} "
                                  "for Livetest Sends Ltd.")
    assert d["subject_template"] == "Quote {{ref:q1}}"
    assert d["subject"] is None and d["body"] is None        # no number yet, so not rendered
    assert d["references"] == [{"id": "q1", "kind": "quote", "rendered_value": None}]
    assert d["attachments"] == [{"document_id": f"quote_doc_{qid}", "added_by": "agent",
                                 "reference_id": "q1", "filename": "Quote.pdf"}]
    assert d["required_attachments"] == [f"quote_doc_{qid}"]
    assert d["commercial"] == {"requires_approval": False}
    assert d["policy"]["agent_send_prohibited"] is True and d["sent"] is None
    assert _state(conn, qid) == ("approved", None) and _last(conn) is None


def test_the_draft_says_why_a_quote_cannot_be_sent(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist, contact=False, status="in_review")
    d = quote_send.draft(conn, qid, principal=person, sender=SENDER, policy_engine=NO_LIMIT)
    assert d["can_send"] is False and d["recipients"] == {"to": []}
    failing = {c["id"]: c["message"] for c in d["checks"] if not c["passed"]}
    assert set(failing) == {"quote_approved", "has_recipient", "named_docs_attached"}
    assert failing["has_recipient"] == "The quote has no contact to send to."
    assert all(c["severity"] == "block" for c in d["checks"])


def test_the_draft_of_a_sent_quote_says_it_went(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    _send(conn, qid, person, _Relay())
    d = quote_send.draft(conn, qid, principal=person, sender=SENDER, policy_engine=NO_LIMIT)
    assert d["can_send"] is False
    assert d["sent"]["email_id"] == "<m1@livetest>" and d["sent"]["to"] == BUYER
    assert d["sent"]["sent_by"] == person.subject and d["sent"]["sent_at"] is not None
    assert d["subject"] == f"Quote {NUMBER_1}"
    assert d["references"][0]["rendered_value"] == NUMBER_1
    assert d["attachments"][0]["filename"] == f"Quote-{NUMBER_1}.pdf"


# --- the hook the send uses on issue -----------------------------------------

def test_an_issue_refused_at_the_last_moment_is_undone_with_its_number(live_db):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    seen = []

    def _no():
        seen.append(_state_inside(conn, qid))
        raise StateConflict("not after all")

    with pytest.raises(StateConflict, match="not after all"):
        quotes.issue(conn, qid, actor="sub-issuer", before_commit=_no)
    assert seen == [("issued", NUMBER_1)]                    # it saw the quote as it would be
    assert _state(conn, qid) == ("approved", None) and _last(conn) is None
    assert _events(conn, qid, "sales_quote.issued") == []


def _state_inside(conn, qid):
    q = quotes.get_quote(conn, qid)                          # same transaction, not ended
    return q["status"], q["quote_number"]


# --- added after breaking each rule on purpose -------------------------------

def test_an_issued_quote_that_has_since_expired_is_not_sent(live_db, person):
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    with pytest.raises(quote_send.SendFailed):
        _send(conn, qid, person, _Relay(success=False))      # issued, not sent
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_quote SET quote_date = %s, valid_until = %s "
                    "WHERE sales_quote_id = %s",
                    (TODAY - dt.timedelta(days=9), TODAY - dt.timedelta(days=1), qid))
    conn.commit()
    _refused(conn, qid, person, _Relay(), "quote_in_date")


def test_a_send_whose_record_cannot_be_written_is_an_error_not_a_success(live_db, person,
                                                                         monkeypatch):
    """The email has gone by then. The call must not answer 'sent' with nothing on record;
    it fails, and the quote reads as not sent."""
    conn, dist = live_db
    qid, _ = _quote(conn, dist)
    real = agent_actions.record_action_or_fail

    def _no_sent_rows(**kw):
        if kw["action_type"] == quote_send.SENT_EVENT:
            raise agent_actions.AuditWriteError("the audit spine is down")
        return real(**kw)

    monkeypatch.setattr(agent_actions, "record_action_or_fail", _no_sent_rows)
    relay = _Relay()
    with pytest.raises(agent_actions.AuditWriteError):
        _send(conn, qid, person, relay)
    assert len(relay.calls) == 1
    assert _state(conn, qid) == ("issued", NUMBER_1)
    assert _events(conn, qid, quote_send.SENT_EVENT) == []
