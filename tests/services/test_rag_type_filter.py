"""A document-type word in a question ranks documents; it no longer excludes every other one (found while fixing C2).

The main collection holds a lowercase ``document_type`` and no ``source_type``, so the old hard filter
``source_type == 'Contract'`` (and PO / Invoice / Quote) matched nothing and retrieval returned no documents.
"""
import pytest
from qdrant_client import models

from src.services import ask_flags
from src.services.rag_service import RAGService


def _svc():
    return RAGService.__new__(RAGService)


def _keys(conds):
    return [(c.key, c.match.value) for c in conds]


@pytest.fixture(autouse=True)
def _flag(monkeypatch):
    monkeypatch.delenv(ask_flags.TYPE_FILTER_FIX, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


@pytest.mark.parametrize("question,word", [
    ("When does the contract with Ironbridge Managed IT Ltd expire?", "contract"),
    ("Show the invoice from Meridia Cloud Platforms", "invoice"),
    ("What quote did we get for the service desk?", "quote"),
    ("Is PO 2025 approved yet?", "po"),
    ("What does the agreement say about termination?", "agreement"),
])
def test_document_type_words_are_a_preference_not_a_filter(question, word):
    _, conditions, focus = _svc()._rewrite_query(question)
    assert conditions == []                                   # nothing is excluded
    assert word in focus or "contract" in focus               # but matching documents are still ranked first


def test_policy_stays_a_hard_filter_because_the_policy_collection_carries_it():
    _, conditions, focus = _svc()._rewrite_query("What is the procurement policy for sole sourcing?")
    assert _keys(conditions) == [("source_type", "Policy")] and "policy" in focus


def test_a_policy_question_naming_a_contract_keeps_only_the_policy_filter():
    _, conditions, _ = _svc()._rewrite_query("What does the policy say about contract renewals?")
    assert _keys(conditions) == [("source_type", "Policy")]


def test_the_flag_restores_the_old_behaviour(monkeypatch):
    monkeypatch.setenv(ask_flags.TYPE_FILTER_FIX, "0")
    _, conditions, _ = _svc()._rewrite_query("Show the contract")
    assert _keys(conditions) == [("source_type", "Contract")]
