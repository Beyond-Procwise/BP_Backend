"""One person's daily brief: who reads what, and whose records.

No database. The reads are stood in for; what is under test is the part that
decides -- which view, whose records, and what a section says when it cannot
answer.
"""
import copy
from types import SimpleNamespace

import pytest

from src.services import rbac
from src.services.daily_brief import service, settings
from src.services.report_data import scope as report_scope
from tests.services.test_daily_brief_settings import seeded_rules


@pytest.fixture
def rules(monkeypatch) -> dict:
    saved = copy.deepcopy(seeded_rules())
    monkeypatch.setattr(settings, "load", lambda: saved)
    return saved


def sign_in(monkeypatch, *, role, groups=(), buyers=(), subject="sub-1", email="pat@example.com"):
    principal = SimpleNamespace(subject=subject, email=email, username=None)
    monkeypatch.setattr(report_scope, "resolve", lambda p: report_scope.Scope(
        role, role == "Admin", tuple(buyers), subject))
    monkeypatch.setattr(rbac, "role_basis", lambda p, policy_engine=None: {
        "groups": list(groups), "direct_grants": []})
    return principal


class FakeConn:
    def cursor(self):
        return object()


def stub_sections(monkeypatch, **answers):
    """Every section answers from ``answers``; one not named there fails the test if called."""
    called = []
    for key in settings.SECTIONS:
        def run(self, key=key):
            called.append(key)
            if key not in answers:
                raise AssertionError(f"{key} was read and should not have been")
            if isinstance(answers[key], Exception):
                raise answers[key]
            return answers[key]
        monkeypatch.setattr(service._Brief, key, run)
    return called


# ── who reads what ───────────────────────────────────────────────────────────

def test_the_chief_procurement_officer_gets_the_executive_view_of_everything(monkeypatch, rules):
    cpo = sign_in(monkeypatch, role="Approver", groups=["PROCWISE_CHIEF_PROCUREMENT_OFFICER"])
    who = service.reader(cpo, rules)
    assert (who["view"], who["basis"], who["buyers"]) == ("executive", "group", None)


def test_a_buyer_gets_the_buyer_view_of_their_own_deals(monkeypatch, rules):
    buyer = sign_in(monkeypatch, role="Buyer", buyers=["CC000247"])
    who = service.reader(buyer, rules)
    assert (who["view"], who["basis"], who["buyers"]) == ("buyer", "role", ("CC000247",))


def test_an_admin_put_in_a_personal_view_still_sees_every_record(monkeypatch, rules):
    """An Admin holds no buyer assignment and needs none; narrowing them to
    nothing would show an Admin an empty brief."""
    rules["view_by_role"]["Admin"] = "buyer"
    who = service.reader(sign_in(monkeypatch, role="Admin"), rules)
    assert (who["view"], who["buyers"]) == ("buyer", None)


def test_changing_the_setting_changes_the_view(monkeypatch, rules):
    buyer = sign_in(monkeypatch, role="Buyer", buyers=["CC000247"])
    rules["view_by_person"] = {"pat@example.com": "executive"}
    who = service.reader(buyer, rules)
    assert (who["view"], who["basis"], who["buyers"]) == ("executive", "person", None)


# ── the brief itself ─────────────────────────────────────────────────────────

def test_the_brief_shows_the_view_s_sections_in_the_view_s_order(monkeypatch, rules):
    cpo = sign_in(monkeypatch, role="Approver", groups=["PROCWISE_CHIEF_PROCUREMENT_OFFICER"])
    called = stub_sections(monkeypatch, momentum={"m": 1}, critical_aged={"c": 1}, upcoming=None)
    brief = service.build(cpo, conn=FakeConn())
    assert called == ["momentum", "critical_aged", "upcoming"]
    assert brief["view"] == "executive" and brief["scope"] == "everyone"
    assert brief["sections"] == [
        {"key": "momentum", "state": "ok", "data": {"m": 1}},
        {"key": "critical_aged", "state": "ok", "data": {"c": 1}},
        {"key": "upcoming", "state": "empty"},
    ]


def test_a_buyer_with_no_assignment_is_told_so_and_shown_nobody_s_records(monkeypatch, rules):
    """No assignment means NO records, never everyone's. The sections are not
    even read: stub_sections fails the test if one is."""
    buyer = sign_in(monkeypatch, role="Buyer", buyers=[])
    stub_sections(monkeypatch)
    brief = service.build(buyer, conn=FakeConn())
    assert brief["scope"] == "own" and brief["assigned"] is False
    assert brief["sections"] == [{"key": k, "state": "unassigned"}
                                 for k in ("pipeline", "supplier_emails", "opportunities")]


def test_an_assigned_buyer_s_sections_are_read_for_their_buyer_codes_only(monkeypatch, rules):
    buyer = sign_in(monkeypatch, role="Buyer", buyers=["CC000247"])
    seen = {}

    def pipeline(self):
        seen["buyers"] = self.buyers
        return {"deals": 16}
    stub_sections(monkeypatch, supplier_emails=None, opportunities=None)
    monkeypatch.setattr(service._Brief, "pipeline", pipeline)
    brief = service.build(buyer, conn=FakeConn())
    assert seen["buyers"] == ("CC000247",)
    assert brief["assigned"] is True and brief["sections"][0]["state"] == "ok"


def test_one_failing_section_costs_only_itself(monkeypatch, rules):
    cpo = sign_in(monkeypatch, role="Approver", groups=["PROCWISE_CHIEF_PROCUREMENT_OFFICER"])
    stub_sections(monkeypatch, momentum=RuntimeError("boom"), critical_aged={"c": 1}, upcoming={"u": 1})
    states = {s["key"]: s["state"] for s in service.build(cpo, conn=FakeConn())["sections"]}
    assert states == {"momentum": "failed", "critical_aged": "ok", "upcoming": "ok"}


def test_no_brief_is_built_when_the_settings_cannot_be_read(monkeypatch):
    def refuse():
        raise settings.BriefSettingsUnavailable("gone")
    monkeypatch.setattr(settings, "load", refuse)
    with pytest.raises(settings.BriefSettingsUnavailable):
        service.build(SimpleNamespace(subject="s", email=None, username=None), conn=FakeConn())


def test_rows_outside_a_buyer_s_assignment_are_dropped():
    brief = service._Brief(None, {}, buyers=("B1",), principal=None, now=None, today=None)
    rows = [{"id": 1, "buyer_id": "B1"}, {"id": 2, "buyer_id": "B2"}, {"id": 3, "buyer_id": None}]
    assert [r["id"] for r in brief._mine(rows)] == [1]
    everyone = service._Brief(None, {}, buyers=None, principal=None, now=None, today=None)
    assert len(everyone._mine(rows)) == 3


# ── saving the settings ──────────────────────────────────────────────────────

class SettingsDb:
    """Stands in for the one policy row: answers the read, records the write."""

    def __init__(self, rules, version=3, rowcount=1):
        self.rules, self.version, self.rowcount = rules, version, rowcount
        self.updates, self.description = [], None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        if sql.lstrip().upper().startswith("UPDATE"):
            self.updates.append(params)
            return
        self.description = [(c,) for c in ("policy_id", "version", "last_modified_by",
                                           "last_modified_date", "rules")]
        self._rows = [(42, self.version, "someone", None, self.rules)]

    def fetchall(self):
        return self._rows


@pytest.fixture
def db(monkeypatch):
    store = SettingsDb(copy.deepcopy(seeded_rules()))
    monkeypatch.setattr(service, "get_conn", lambda: store)
    return store


def test_a_valid_change_is_saved_over_the_rest_and_bumps_the_version(db):
    import json
    service.save_settings({"critical_aged_min_days": 30}, by="admin@example.com", expected_version=3)
    saved_rules, by, policy_id, version = db.updates[0]
    saved = json.loads(saved_rules)
    assert saved["critical_aged_min_days"] == 30
    assert saved["momentum_window_days"] == 7            # everything else is kept
    assert (by, policy_id, version) == ("admin@example.com", 42, 3)


def test_an_invalid_change_is_refused_and_nothing_is_written(db):
    with pytest.raises(ValueError, match="critical_aged_min_days"):
        service.save_settings({"critical_aged_min_days": -5}, by="admin@example.com")
    assert db.updates == []


def test_a_save_against_an_older_version_is_refused_and_nothing_is_written(db):
    with pytest.raises(service.SettingsConflict):
        service.save_settings({"critical_aged_min_days": 30}, by="a", expected_version=2)
    assert db.updates == []


def test_a_save_that_lost_the_race_is_reported_not_swallowed(db):
    db.rowcount = 0
    with pytest.raises(service.SettingsConflict):
        service.save_settings({"critical_aged_min_days": 30}, by="a", expected_version=3)


def test_the_brief_says_whether_to_offer_the_settings(monkeypatch, rules):
    stub_sections(monkeypatch, momentum=None, critical_aged=None, upcoming=None, approvals=None,
                  pipeline=None, supplier_emails=None, opportunities=None)
    monkeypatch.setattr(rbac, "may", lambda role, action_class, policy_engine=None:
                        role == "Admin" and action_class == "configure")
    admin = service.build(sign_in(monkeypatch, role="Admin"), conn=FakeConn())
    buyer = service.build(sign_in(monkeypatch, role="Buyer", buyers=["B1"]), conn=FakeConn())
    assert admin["canConfigure"] is True and buyer["canConfigure"] is False
