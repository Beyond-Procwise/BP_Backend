"""Shared RFQ reference = tier-1 linked identifier for rivalry clustering.

The Test Data_300726 batch: SaaS bids (AUR/LSL/MCP) all print
"RFQ reference: PROC-2025-RFQ-SAA-042" and correlate at only 0.58 on fuzzy
text — below the 0.70 threshold. A shared RFQ reference is decisive linkage
("linked identifiers: decisive where present", spec §Data flow), so bids
citing the same RFQ must cluster regardless of description fuzziness.
"""
from src.services.deal_clustering import (
    apply_rfq_linkage,
    cluster_batch,
    extract_rfq_reference,
)


# ---- extraction ----------------------------------------------------------

def test_extracts_labelled_rfq_reference():
    text = "| Quotation to | RFQ reference:  PROC-2025-RFQ-SAA-042 |\n| Term | 36 months |"
    assert extract_rfq_reference(text) == "PROC-2025-RFQ-SAA-042"


def test_extracts_bare_rfq_token_from_jumbled_layout():
    # NHP's PDF detaches the label: only "RFQ-FRT-021" survives as a token.
    text = "RFQ-FRT-021\n\nDate\n\nRFQ reference\n\nValidity\n\n21 days"
    assert extract_rfq_reference(text) == "RFQ-FRT-021"


def test_full_and_bare_forms_normalise_to_the_same_event():
    # PROC-2025-RFQ-FRT-021 (BWT/KLG) and bare RFQ-FRT-021 (NHP) are the
    # same competition: canonicalisation anchors on the RFQ stem.
    from src.services.deal_clustering import _canon_rfq
    assert _canon_rfq("PROC-2025-RFQ-FRT-021") == _canon_rfq("RFQ-FRT-021")
    assert _canon_rfq("PROC-2025-RFQ-SAA-042") != _canon_rfq("RFQ-FRT-021")


def test_no_rfq_or_ambiguous_rfq_yields_none():
    assert extract_rfq_reference("Invoice for services rendered") is None
    two = "RFQ reference: PROC-2025-RFQ-SAA-042\nalso cites PROC-2025-RFQ-MAN-030"
    assert extract_rfq_reference(two) is None  # ambiguous -> conservative None


# ---- linkage boost -------------------------------------------------------

def _bids(*specs):
    return [{"quote_id": q, "supplier_id": s, "rfq_reference": r}
            for q, s, r in specs]


def test_shared_rfq_lifts_pair_above_threshold_with_evidence():
    bids = _bids(("A", "SUP-1", "PROC-2025-RFQ-SAA-042"),
                 ("B", "SUP-2", "PROC-2025-RFQ-SAA-042"))
    matrix = {frozenset(("A", "B")): {"correlation": 0.5877}}
    apply_rfq_linkage(bids, matrix)
    res = matrix[frozenset(("A", "B"))]
    assert res["correlation"] >= 0.95
    assert res["rfq_shared"] == "PROC-2025-RFQ-SAA-042"


def test_different_or_missing_rfq_changes_nothing():
    bids = _bids(("A", "SUP-1", "PROC-2025-RFQ-SAA-042"),
                 ("B", "SUP-2", "PROC-2025-RFQ-MAN-030"),
                 ("C", "SUP-3", None))
    matrix = {frozenset(("A", "B")): {"correlation": 0.4},
              frozenset(("A", "C")): {"correlation": 0.6}}
    apply_rfq_linkage(bids, matrix)
    assert matrix[frozenset(("A", "B"))]["correlation"] == 0.4
    assert matrix[frozenset(("A", "C"))]["correlation"] == 0.6
    assert "rfq_shared" not in matrix[frozenset(("A", "B"))]


# ---- batch-fetch enrichment ----------------------------------------------

def test_fetch_enrichment_attaches_rfq_from_raw_text():
    from src.api.routers.deal_proposals import _attach_rfq_refs

    class _Cur:
        def execute(self, sql, params=()):
            self._rows = [
                ("Q1", "| RFQ reference:  PROC-2025-RFQ-SAA-042 |"),
                ("Q2", None),
            ]

        def fetchall(self):
            return self._rows

    quotes = [{"quote_id": "Q1"}, {"quote_id": "Q2"}]
    _attach_rfq_refs(_Cur(), quotes)
    assert quotes[0]["rfq_reference"] == "PROC-2025-RFQ-SAA-042"
    assert quotes[1]["rfq_reference"] is None


# ---- attachment provenance -----------------------------------------------

def test_po_and_invoice_members_carry_link_evidence(monkeypatch):
    # The engine KNOWS why it attached the PO (it cites the winning bid's
    # reference) and each invoice (it cites the PO number) — that provenance
    # must survive into the stored members so the UI can show per-connection
    # confidence instead of a bare grouping.
    import src.services.deal_clustering as dc
    monkeypatch.setattr(dc, "rivalry_score",
                        lambda a, b, la, lb: {"correlation": 0.1, "F": 10.0})
    quotes = [
        {"quote_id": "A-1", "supplier_id": "SUP-A", "quote_date": None,
         "rfq_reference": "RFQ-X-1"},
        {"quote_id": "B-1", "supplier_id": "SUP-B", "quote_date": None,
         "rfq_reference": "RFQ-X-1"},
    ]
    pos = [{"po_id": "PO-9", "quote_reference": "A-1"}]
    invoices = [{"invoice_id": "INV-7", "po_id": "PO-9"}]
    out = dc.cluster_batch(quotes=quotes, quote_lines={}, purchase_orders=pos,
                           po_lines={}, invoices=invoices)
    members = {(m["doc_type"], m["doc_pk"]): m
               for p in out["proposals"] for m in p["members"]}
    po_m = members[("po", "PO-9")]
    assert po_m["match_evidence"]["linked_by"] == "quote_reference"
    assert po_m["match_evidence"]["cites"] == "A-1"
    assert po_m["match_score"] == 100.0
    inv_m = members[("invoice", "INV-7")]
    assert inv_m["match_evidence"]["linked_by"] == "po_reference"
    assert inv_m["match_evidence"]["cites"] == "PO-9"
    assert inv_m["match_score"] == 100.0


# ---- end to end through cluster_batch ------------------------------------

def test_cluster_batch_attaches_each_document_once_despite_duplicate_rows(monkeypatch):
    # Re-extraction leaves multiple raw rows per document, so the batch fetch
    # can hand cluster_batch the same invoice/PO twice. Members must still be
    # unique or store_proposals violates its primary key (seen live 2026-07-30:
    # (35, invoice, MCP-INV-1120) duplicate).
    import src.services.deal_clustering as dc
    monkeypatch.setattr(dc, "rivalry_score",
                        lambda a, b, la, lb: {"correlation": 0.1, "F": 10.0})
    quotes = [
        {"quote_id": "A-1", "supplier_id": "SUP-A", "quote_date": None,
         "rfq_reference": "RFQ-X-1"},
        {"quote_id": "B-1", "supplier_id": "SUP-B", "quote_date": None,
         "rfq_reference": "RFQ-X-1"},
    ]
    pos = [{"po_id": "PO-9", "quote_reference": "A-1"},
           {"po_id": "PO-9", "quote_reference": "A-1"}]          # duplicate row
    invoices = [{"invoice_id": "INV-7", "po_id": "PO-9"},
                {"invoice_id": "INV-7", "po_id": "PO-9"}]        # duplicate row
    out = dc.cluster_batch(quotes=quotes, quote_lines={}, purchase_orders=pos,
                           po_lines={}, invoices=invoices)
    members = [(m["doc_type"], m["doc_pk"]) for p in out["proposals"] for m in p["members"]]
    assert len(members) == len(set(members))
    assert ("invoice", "INV-7") in members and ("po", "PO-9") in members


def test_cluster_batch_groups_low_correlation_bids_sharing_an_rfq(monkeypatch):
    # Two rival bids whose text correlation is hopeless (0.1) but which cite
    # the same RFQ must come out as ONE proposed event, not two ungrouped rows.
    import src.services.deal_clustering as dc
    monkeypatch.setattr(dc, "rivalry_score",
                        lambda a, b, la, lb: {"correlation": 0.1, "F": 10.0})
    quotes = [
        {"quote_id": "AUR-1", "supplier_id": "SUP-A", "quote_date": None,
         "rfq_reference": "PROC-2025-RFQ-SAA-042"},
        {"quote_id": "MCP-1", "supplier_id": "SUP-B", "quote_date": None,
         "rfq_reference": "PROC-2025-RFQ-SAA-042"},
    ]
    out = dc.cluster_batch(quotes=quotes, quote_lines={}, purchase_orders=[],
                           po_lines={}, invoices=[])
    assert len(out["proposals"]) == 1
    pks = {m["doc_pk"] for m in out["proposals"][0]["members"]}
    assert pks == {"AUR-1", "MCP-1"}
    assert not any(u["doc_type"] == "quote" for u in out["ungrouped"])
