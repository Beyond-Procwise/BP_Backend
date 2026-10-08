"""The negotiation agent's own draft stub is a record, not a sendable draft.

``NegotiationAgent._build_email_draft_stub`` returns a dict (subject, body, recipients, unique_id...) that
ends up in the agent's ``drafts`` output. It is NOT wrapped in draft assurance, on the grounds that it
cannot reach a supplier: the negotiation agent never persists it, and ``EmailDispatchService.send_draft``
sends only a draft that is STORED. The text that does go out for a negotiation round is written and
stored by ``EmailDraftingAgent`` (whose counter path is assured), and a body handed to the dispatcher
over that stored draft is refused by the approval's content hash (tests/approvals/test_content_binding_enforced.py).

These tests pin the first half of that argument. If ``send_draft`` ever starts accepting a draft that is
not in ``proc.draft_rfq_emails``, they go red and the stub must be wrapped.
"""

import logging
from types import SimpleNamespace

import pytest

# The shape _build_email_draft_stub returns (src/agents/negotiation_agent.py), with the model's own words.
STUB = {
    "supplier_id": "S-1", "supplier_name": "Acme", "subject": "Re: pricing",
    "body": "We will pay 12.75 per unit and award you the whole volume.<!-- PROCWISE_MARKER:TRACKING:PROC-WF-STUB -->",
    "text": "We will pay 12.75 per unit and award you the whole volume.",
    "recipients": ["someone@elsewhere.example"], "receiver": "someone@elsewhere.example",
    "sent_status": False, "unique_id": "PROC-WF-STUB", "session_reference": "PROC-WF-STUB",
    "workflow_id": "WF-1", "metadata": {"unique_id": "PROC-WF-STUB"},
}


class _NoRows:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def commit(self):
        pass


def _service():
    from services.email_dispatch_service import EmailDispatchService

    sent = []
    svc = object.__new__(EmailDispatchService)
    conn = _NoRows()
    svc.agent_nick = SimpleNamespace(get_db_connection=lambda: conn, settings=SimpleNamespace(ses_default_sender="buyer@x.test"))
    svc.settings = svc.agent_nick.settings
    svc.logger = logging.getLogger("stub-test")
    svc._thread_table_name = "proc.email_thread_map"
    svc._thread_table_ready = False
    svc.email_service = SimpleNamespace(send_email=lambda *a, **k: sent.append((a, k)))
    return svc, sent, conn


def test_a_stub_that_was_never_stored_cannot_be_sent():
    svc, sent, _ = _service()
    with pytest.raises(ValueError, match="No stored draft"):
        svc.send_draft(STUB["unique_id"], recipients=STUB["recipients"], sender="buyer@x.test",
                       subject_override=STUB["subject"], body_override=STUB["body"], principal=None, agent_name="NegotiationAgent")
    assert sent == []                                # nothing reached the mail service


def test_the_stub_cannot_be_sent_by_its_session_reference_either():
    svc, sent, _ = _service()
    with pytest.raises(ValueError, match="No stored draft"):
        svc.send_draft(STUB["session_reference"], body_override=STUB["body"], agent_name="NegotiationAgent")
    assert sent == []


def test_the_lookup_that_refuses_it_reads_the_stored_drafts_table_only():
    svc, _, conn = _service()
    with pytest.raises(ValueError):
        svc.send_draft(STUB["unique_id"], body_override=STUB["body"], agent_name="NegotiationAgent")
    assert conn.statements and all("proc.draft_rfq_emails" in s or "draft_rfq_emails" in s for s in conn.statements)
