"""An executive summary must exist the moment the analysis is ready — the
report card said "Summary not available yet" until a 15-minute sweep ran,
and never ran at all for held documents. Session completion now stores a
deterministic summary (documents, proposed deals, where the issues are);
the LLM narrative on the sweep still owns real confirmed deals.
"""
from src.services.session_postprocess import compose_session_summary


def _facts(**over):
    facts = {
        "total": 35, "linked": 15, "held": 20, "duplicates": 0,
        "proposals": [
            {"proposed_name": "Southampton - Birmingham (Aston) -FTL — 3 bidders",
             "confidence": 95.0, "bids": 3, "invoices": 0, "pos": 0},
            {"proposed_name": "Platform licence — Enterprise (240 seats) — 3 bidders",
             "confidence": 95.0, "bids": 3, "invoices": 2, "pos": 1},
        ],
        "critical": 3, "warnings": 7,
        "top_issues": [
            "3 documents cite a purchase order that is not in the system (PO-2025-0208)",
        ],
    }
    facts.update(over)
    return facts


def test_summary_leads_with_documents_and_deals():
    text = compose_session_summary(**_facts())
    assert "35 documents" in text
    assert "15" in text and "20" in text
    assert "Southampton - Birmingham" in text
    assert "95%" in text


def test_summary_states_where_the_issues_are():
    text = compose_session_summary(**_facts())
    assert "3 critical" in text
    assert "PO-2025-0208" in text


def test_clean_session_reads_clean():
    text = compose_session_summary(**_facts(critical=0, warnings=0, top_issues=[]))
    assert "no critical issues" in text.lower()


def test_no_proposals_prompts_nothing_confusing():
    text = compose_session_summary(**_facts(proposals=[]))
    assert "confirm" not in text.lower() or "proposed" not in text.lower()
