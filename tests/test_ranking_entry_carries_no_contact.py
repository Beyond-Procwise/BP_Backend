"""A ranking result is a commercial comparison, not a contact card.

``_prepare_ranking_entry`` republished ``contact_name_1`` / ``contact_email_1``
off the supplier data frame as ``contact_name`` / ``contact_email`` on every
ranking entry. ``EmailDraftingAgent`` then took that address as the RFQ
recipient, which made a value carried through the shared workflow context decide
where mail went. Drafting now resolves the address from ``proc.bp_supplier`` by
``supplier_id``, so nothing needs the entry to carry it -- and while it carries
it, a personal email address travels through every downstream prompt and every
serialised run record for no reason at all.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.supplier_ranking_agent import SupplierRankingAgent


def _nick():
    return SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        prompt_engine=SimpleNamespace(),
        policy_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model={}),
        query_engine=SimpleNamespace(),
    )


def _entry():
    agent = SupplierRankingAgent(_nick())
    row = pd.Series(
        {
            "supplier_id": "SUP-1",
            "supplier_name": "Acme",
            "final_score": 0.82,
            "contact_name_1": "Dana Okafor",
            "contact_email_1": "dana@acme.test",
        }
    )
    return agent._prepare_ranking_entry(row, None, {"price": 1.0})


def test_a_ranking_entry_carries_no_contact_email():
    assert "contact_email" not in _entry()


def test_a_ranking_entry_carries_no_contact_name():
    assert "contact_name" not in _entry()


def test_the_ranking_entry_still_identifies_the_supplier():
    """Removing the contact must not remove the means of looking one up."""

    entry = _entry()
    assert entry["supplier_id"] == "SUP-1"
    assert entry["supplier_name"] == "Acme"
