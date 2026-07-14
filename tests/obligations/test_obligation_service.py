"""The service's invariants, without calling a model.

The one that matters: an obligation whose quote is not in the document never reaches the
database. `test_extraction_live.py` proves the same thing against the real model.
"""
from unittest.mock import patch

# hyperextract's __init__ does not re-export its submodules, so a dotted patch target
# cannot traverse to it. Import the module, then patch the attribute on it.
import hyperextract.types.hypergraph as hypergraph_module

from src.services.obligations.obligation_service import (
    STATUS_EXTRACTED,
    STATUS_NO_GROUNDED,
    ObligationRun,
    _key,
    extract_obligations,
)
from src.services.obligations.schema import ContractObligation, RelType

CONTRACT = "27.7. The risk in any over delivered Goods shall remain with the Contractor."

REAL = ContractObligation(
    name="Risk stays with Contractor", type=RelType.risk_transfer,
    participants=["the Contractor", "Goods"], clause_ref="27.7",
    source_quote="The risk in any over delivered Goods shall remain with the Contractor.",
)
FABRICATED = ContractObligation(
    name="Invented indemnity", type=RelType.penalised_by,
    participants=["the Contractor"], clause_ref="27.7",
    source_quote="The Contractor shall indemnify the Authority for all consequential loss.",
)


class _FakeGraph:
    nodes: list = []

    def __init__(self, edges):
        self.edges = edges

    def feed_text(self, text):
        return self


def _run_with(edges):
    with patch.object(hypergraph_module, "AutoHypergraph", return_value=_FakeGraph(edges)):
        return extract_obligations("DOC-1", CONTRACT, embedding_model=object())


def test_grounded_obligation_survives():
    run = _run_with([REAL])
    assert [o.name for o in run.obligations] == ["Risk stays with Contractor"]
    assert run.status == STATUS_EXTRACTED


def test_fabricated_obligation_is_dropped_not_stored():
    run = _run_with([REAL, FABRICATED])
    assert [o.name for o in run.obligations] == ["Risk stays with Contractor"]
    assert [e.name for e, _ in run.dropped] == ["Invented indemnity"]


def test_all_ungrounded_is_not_reported_as_a_clean_zero():
    """If every obligation was invented, that is a distinct state — not 'extracted 0'."""
    run = _run_with([FABRICATED])
    assert run.obligations == []
    assert run.status == STATUS_NO_GROUNDED


def test_empty_contract_text_raises():
    """Never quietly succeed on a document we could not read."""
    try:
        extract_obligations("DOC-1", "   ", embedding_model=object())
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_entity_keys_are_normalised():
    assert _key("The Contractor") == _key("contractor") == "contractor"


def test_persisted_quotes_are_spans_of_the_document():
    """The feature's core invariant, stated as a test."""
    run = ObligationRun(document_id="DOC-1", obligations=_run_with([REAL, FABRICATED]).obligations)
    for ob in run.obligations:
        assert ob.source_quote.lower() in CONTRACT.lower()
