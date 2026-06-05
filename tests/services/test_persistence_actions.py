import src.services.agent_actions as aa
from src.services.extraction import persistence
from src.services.extraction.persistence import Discrepancy


def test_write_discrepancies_records_validation_actions(monkeypatch):
    captured = {}

    def fake_bulk_record(rows, *, conn=None):
        captured["rows"] = list(rows)
        captured["conn_passed"] = conn is not None

    monkeypatch.setattr(persistence, "bulk_record", fake_bulk_record)

    discs = [
        Discrepancy(
            field_name="tax_amount", raw_value="10", expected_value="12",
            computed_value="12", issue_type="tax_mismatch", severity="critical",
            blocks_promotion=True, evidence_page=1, evidence_bbox=None,
            evidence_text="Tax 12", notes="",
        ),
    ]
    persistence.write_discrepancies(
        doc_type="invoice", raw_id=42, source_file="/tmp/x.pdf",
        doc_pk_candidate="INV-1", discrepancies=discs,
    )
    assert captured["conn_passed"] is True
    assert len(captured["rows"]) == 1
    row = captured["rows"][0]
    assert row["phase"] == aa.PHASE_VALIDATION
    assert row["action_type"] == "discrepancy"
    assert row["field_name"] == "tax_amount"
    assert row["doc_pk"] == "INV-1"
