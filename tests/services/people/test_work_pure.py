import pytest

from services.people import work as w
from services.people.admin import PeopleError


def test_only_known_kinds_can_be_handed_on():
    assert w.check_entity_type("finding") == "finding"
    with pytest.raises(PeopleError) as e:
        w.check_entity_type("po")
    assert e.value.status == 422 and "finding" in e.value.detail


def test_note_is_trimmed_and_bounded():
    assert w.clean_note("  keep an eye on this \n") == "keep an eye on this"
    assert w.clean_note("") is None and w.clean_note(None) is None and w.clean_note(5) is None
    with pytest.raises(PeopleError):
        w.clean_note("x" * (w.MAX_NOTE + 1))


def test_the_context_line_names_the_finding_without_its_values():
    f = {"discrepancy_id": 24718, "issue_type": "duplicate_invoice", "doc_type": "invoice",
         "source_file": "documents/invoice/MASTER Invoice for PO1.pdf"}
    assert w.what_finding(f) == "duplicate invoice on invoice MASTER Invoice for PO1.pdf"
    assert w.about_finding(f) == "About finding 24718: duplicate invoice on invoice MASTER Invoice for PO1.pdf"
    assert w.about_finding({"discrepancy_id": 1}) == "About finding 1: finding on document a document"


def test_the_link_opens_the_queue_on_the_item():
    assert w.link_for("finding", "24718") == "/actions/queue?finding=24718"
