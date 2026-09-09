"""An RFQ is addressed from the supplier master, not from what it was handed.

``SupplierRankingAgent`` republished ``contact_email_1`` off the supplier frame
as ``contact_email`` on every ranking entry, and ``_resolve_receiver`` took that
as its first candidate -- ahead of two further fallbacks, none of which read
proc.bp_supplier. Policy #674 says the master is the only acceptable source, and
the send path enforced that afterwards; these tests move the rule to the point
where the address is chosen.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from agents.email_drafting_agent import EmailDraftingAgent


class _Master:
    """A supplier master holding one address for one supplier."""

    def __init__(self, mapping):
        self._mapping = mapping

    def lookup_supplier_emails(self, supplier_id):
        return self._mapping.get(supplier_id, [])


def _agent_reading(master):
    agent = EmailDraftingAgent()

    @contextmanager
    def _conn():
        yield master

    agent.agent_nick = SimpleNamespace(get_db_connection=_conn)
    return agent


def test_the_carried_address_loses_to_the_master():
    """The payload names one address; the master names another. The master wins."""

    agent = _agent_reading(_Master({"SUP-1": ["real@supplier.test"]}))

    receiver = agent._resolve_receiver(
        {
            "supplier_id": "SUP-1",
            "contact_email": "attacker@example.com",
            "contact_email_1": "also-wrong@example.com",
            "contact_email_2": "wrong-again@example.com",
        },
        {"contacts": [{"email": "profile@example.com"}]},
    )

    assert receiver == "real@supplier.test"


def test_a_supplier_with_no_address_on_file_yields_no_recipient():
    """Not a fallback to the payload -- no recipient, so the draft is held."""

    agent = _agent_reading(_Master({}))

    receiver = agent._resolve_receiver(
        {"supplier_id": "SUP-2", "contact_email": "carried@example.com"},
        {"contacts": [{"email": "profile@example.com"}]},
    )

    assert receiver is None


def test_no_supplier_id_yields_no_recipient():
    """Without a supplier there is nothing to look up, and nothing to trust."""

    agent = _agent_reading(_Master({"SUP-3": ["real@supplier.test"]}))

    receiver = agent._resolve_receiver(
        {"contact_email": "carried@example.com"}, {}
    )

    assert receiver is None


def test_the_second_master_address_is_used_when_the_first_is_absent():
    agent = _agent_reading(_Master({"SUP-4": ["second@supplier.test"]}))

    receiver = agent._resolve_receiver({"supplier_id": "SUP-4"}, {})

    assert receiver == "second@supplier.test"


def test_the_master_is_consulted_once_per_supplier(monkeypatch):
    """One draft, one lookup.

    The salutation, the rendered address and the recipient all come from the
    same resolved contact. Resolving separately for the greeting and again for
    the recipient is a second round-trip per supplier -- and per supplier is
    exactly the shape that becomes N+1 on a batch.
    """

    from agents.base_agent import AgentContext
    from src.services.supplier_contact import SupplierContact

    calls = []

    monkeypatch.setattr(
        EmailDraftingAgent,
        "_master_contact",
        lambda self, supplier_id: (
            calls.append(supplier_id),
            SupplierContact(emails=["quotes@acme.test"], name="Acme Sales"),
        )[1],
    )

    EmailDraftingAgent().run(
        AgentContext(
            workflow_id="W",
            agent_id="email_drafting",
            user_id="u",
            input_data={
                "ranking": [{"supplier_id": "SUP-1", "supplier_name": "Acme"}],
                "supplier_profiles": {"SUP-1": {}},
                "policies": [],
            },
        )
    )

    assert calls == ["SUP-1"], f"expected one lookup, got {len(calls)}: {calls}"
