import pytest

from services.people import intake, orgchart
from services.people.intake import IntakeError

OWNER = {"subject": "owen", "roles": ["Admin"], "approval_limit": -1}


def plan(existing, records, **kw):
    return orgchart.plan_import(existing, records, **kw)


def test_reports_listed_before_their_manager_are_applied_managers_first():
    p = plan([], [{"subject": "bea", "manager_subject": "owen"}, dict(OWNER)])
    assert [s for s, _ in p["writes"]] == ["owen", "bea"] and not p["errors"]


def test_unknown_manager_is_refused_and_named():
    p = plan([], [{"subject": "bea", "manager_subject": "ghost"}])
    assert p["errors"] and "ghost" in p["errors"][0]["problem"] and not p["writes"]


def test_reporting_loop_inside_the_batch_is_refused():
    p = plan([], [{"subject": "a", "manager_subject": "b"}, {"subject": "b", "manager_subject": "a"}])
    assert len(p["errors"]) == 2 and not p["writes"]


def test_bad_field_is_refused_with_the_same_rules_as_the_single_edit():
    p = plan([], [{"subject": "a", "approval_limit": -5}])
    assert p["errors"] and "approval_limit" in p["errors"][0]["problem"]


def test_duplicate_and_blank_subjects_are_refused():
    p = plan([], [{"subject": "a"}, {"subject": "a"}, {"subject": " "}])
    assert len(p["errors"]) == 2


def test_manager_may_already_exist():
    existing = [{"subject": "owen", "roles": ["Admin"], "approval_limit": None, "is_active": True}]
    p = plan(existing, [{"subject": "bea", "manager_subject": "owen"}])
    assert not p["errors"] and p["created"] == ["bea"]


def test_missing_people_untouched_unless_asked_and_never_the_last_admin():
    existing = [{"subject": "owen", "roles": ["Admin"], "approval_limit": None, "is_active": True},
                {"subject": "old", "roles": [], "approval_limit": None, "is_active": True}]
    assert plan(existing, [{"subject": "bea"}])["deactivated"] == []
    p = plan(existing, [{"subject": "bea"}], deactivate_missing=True)
    assert p["errors"] and p["deactivated"] == []          # would drop the only Admin
    p = plan(existing, [{"subject": "owen"}], deactivate_missing=True)
    assert p["deactivated"] == ["old"] and not p["errors"]


def test_starter_chart_passes_its_own_validation():
    assert not plan([], orgchart.starter_chart("owen", "o@x.com"))["errors"]


def test_external_event_sentence_is_ours_and_unknown_types_refused():
    e = intake.build_event("sap", "requisition.submitted", "PR-1", "bea", amount="1200")
    assert e.summary == "Requisition PR-1 was submitted for approval" and e.entity_type == "requisition"
    assert e.data["amount"] == "1200" and e.data["requester"] == "bea"
    for bad in ("approval.approved", "message.received", "nope"):
        with pytest.raises(IntakeError):
            intake.build_event("sap", bad, "X", None)


def test_retry_is_the_same_event_and_a_new_version_is_not():
    a = intake.build_event("sap", "po.changed", "PO9", None, version="1")
    assert a.event_id == intake.build_event("sap", "po.changed", "PO9", None, version="1").event_id
    assert a.event_id != intake.build_event("sap", "po.changed", "PO9", None, version="2").event_id


@pytest.mark.parametrize("kw", [dict(external_id=""), dict(external_id="x" * 81), dict(amount="abc"), dict(amount=-1)])
def test_bad_inputs_refused(kw):
    base = dict(source="s", event_type="po.issued", external_id="P1", requester=None)
    base["requester"] = base.pop("requester")
    args = {**base, **kw}
    with pytest.raises(IntakeError):
        intake.build_event(args["source"], args["event_type"], args["external_id"], args["requester"],
                           amount=args.get("amount"))


def test_link_must_be_local():
    assert "link" not in intake.build_event("s", "po.issued", "P", None, link="https://evil.example").data
    assert "link" not in intake.build_event("s", "po.issued", "P", None, link="//evil.example").data
    assert intake.build_event("s", "po.issued", "P", None, link="/po/P").data["link"] == "/po/P"
