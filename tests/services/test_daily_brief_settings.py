"""The daily brief's settings: what may be saved, and which view a reader gets.

The rule these tests hold: a setting that is missing or does not hold together
REFUSES. Nobody is quietly given a view the code chose for them.
"""
import copy
import json
import re
from pathlib import Path

import pytest

from src.services import governed_limits
from src.services.daily_brief import settings

MIGRATION = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-18_daily_brief_settings.sql"


def seeded_rules() -> dict:
    """The rules exactly as the migration writes them."""
    block = re.search(r"\$rules\$(.*?)\$rules\$", MIGRATION.read_text(), re.S)
    assert block, "the migration no longer carries its rules between $rules$ markers"
    return json.loads(block.group(1))


@pytest.fixture
def rules() -> dict:
    return copy.deepcopy(seeded_rules())


def test_the_settings_the_migration_ships_are_valid(rules):
    assert settings.validate(rules) == []


def test_every_setting_the_code_reads_is_in_the_migration(rules):
    assert set(rules) == settings._KNOWN


@pytest.mark.parametrize("key", sorted(settings._KNOWN))
def test_a_missing_setting_is_a_problem(rules, key):
    del rules[key]
    assert any(f"'{key}' is missing" in p for p in settings.validate(rules))


def test_a_setting_nobody_reads_is_a_problem(rules):
    """A misspelt key would otherwise be saved, ignored, and look like it had worked."""
    rules["momentum_window_day"] = 7
    assert "'momentum_window_day' is not a daily brief setting" in settings.validate(rules)


def test_a_personal_view_may_not_show_a_company_wide_section(rules):
    rules["views"]["buyer"]["sections"].append("momentum")
    problems = settings.validate(rules)
    assert any("'momentum'" in p and "cannot be narrowed" in p for p in problems)


def test_the_same_section_is_allowed_in_a_view_of_everyone(rules):
    rules["views"]["buyer"]["scope"] = "everyone"
    rules["views"]["buyer"]["sections"].append("momentum")
    assert settings.validate(rules) == []


def test_a_view_may_not_show_a_section_that_does_not_exist(rules):
    rules["views"]["executive"]["sections"].append("horoscope")
    assert any("does not exist" in p and "horoscope" in p for p in settings.validate(rules))


@pytest.mark.parametrize("key", settings.VIEW_MAPS)
def test_nobody_may_be_sent_to_a_view_that_is_not_defined(rules, key):
    rules[key]["someone"] = "board"
    assert any(f"'{key}'" in p and "'board'" in p for p in settings.validate(rules))


def test_the_default_view_must_be_defined(rules):
    rules["default_view"] = "board"
    assert any("'default_view'" in p for p in settings.validate(rules))


def test_view_order_must_list_every_view_once(rules):
    rules["view_order"] = ["executive", "manager", "buyer"]
    assert "'view_order' must list every view exactly once" in settings.validate(rules)


@pytest.mark.parametrize("value", [0, 91, 7.5, "7", True, None])
def test_a_number_outside_its_range_or_not_whole_is_a_problem(rules, value):
    rules["momentum_window_days"] = value
    assert any("'momentum_window_days'" in p for p in settings.validate(rules))


def test_the_value_floor_may_be_unset_or_an_amount_and_nothing_else(rules):
    rules["upcoming_min_value_gbp"] = 250000
    assert settings.validate(rules) == []
    rules["upcoming_min_value_gbp"] = -1
    assert any("'upcoming_min_value_gbp'" in p for p in settings.validate(rules))


def test_a_severity_list_may_not_be_empty(rules):
    rules["critical_aged_severities"] = []
    assert any("'critical_aged_severities'" in p for p in settings.validate(rules))


# ── which view a reader gets ─────────────────────────────────────────────────

def view(rules, role=None, groups=(), identities=()):
    return settings.resolve_view(rules, role=role, groups=groups, identities=identities)


def test_a_role_gets_the_view_the_settings_give_it(rules):
    assert view(rules, role="Buyer") == ("buyer", "role")
    assert view(rules, role="Admin") == ("manager", "role")


def test_a_group_on_the_sign_in_outranks_the_role(rules):
    """The chief procurement officer's group resolves to the Approver role, which a
    finance approver holds too. Only the group tells them apart."""
    assert view(rules, role="Approver", groups=["PROCWISE_CHIEF_PROCUREMENT_OFFICER"]) == ("executive", "group")
    assert view(rules, role="Approver", groups=["PROCWISE_FINANCE_REVIEWER_APPROVER"]) == ("manager", "role")


def test_a_named_person_outranks_their_group_and_role(rules):
    rules["view_by_person"] = {"Dana@Example.com": "buyer"}
    got = view(rules, role="Approver", groups=["PROCWISE_CHIEF_PROCUREMENT_OFFICER"],
               identities=["sub-123", "dana@example.com"])
    assert got == ("buyer", "person")


def test_two_groups_pointing_at_different_views_follow_view_order(rules):
    rules["view_by_group"] = {"G_BUY": "buyer", "G_EXEC": "executive"}
    assert view(rules, groups=["G_BUY", "G_EXEC"])[0] == "executive"
    rules["view_order"] = ["buyer", "manager", "executive", "requester"]
    assert view(rules, groups=["G_BUY", "G_EXEC"])[0] == "buyer"


def test_an_unknown_role_gets_the_default_and_nothing_wider(rules):
    assert view(rules, role="Contractor") == ("requester", "default")
    assert view(rules) == ("requester", "default")


# ── reading them ─────────────────────────────────────────────────────────────

def test_load_refuses_when_the_settings_are_not_there(monkeypatch):
    def gone(policy, fresh=False):
        raise governed_limits.LimitUnavailable("no active daily_brief policy")
    monkeypatch.setattr(governed_limits, "rules", gone)
    with pytest.raises(settings.BriefSettingsUnavailable):
        settings.load()


def test_load_refuses_settings_that_do_not_hold_together(monkeypatch, rules):
    rules["view_by_role"]["Buyer"] = "board"
    monkeypatch.setattr(governed_limits, "rules", lambda policy, fresh=False: rules)
    with pytest.raises(settings.BriefSettingsUnavailable, match="board"):
        settings.load()


def test_load_reads_the_saved_row_fresh(monkeypatch, rules):
    """A change an administrator saves must apply to the next brief, not the next restart."""
    seen = {}

    def read(policy, fresh=False):
        seen.update(policy=policy, fresh=fresh)
        return rules
    monkeypatch.setattr(governed_limits, "rules", read)
    assert settings.load() == rules
    assert seen == {"policy": "daily_brief", "fresh": True}


def test_the_migration_also_lets_an_admin_change_the_settings():
    """Saving is a configure action, and the gate refuses those unless a policy says
    allow. Shipping the settings without it left nobody able to open them, an Admin
    included (found on the first live call, 2026-10-13)."""
    from src.services import actions
    sql = MIGRATION.read_text()
    assert actions.action_class("brief.settings.write") == "configure"
    assert "jsonb_build_array('brief.settings.write')" in sql
    assert "'required_role', 'Admin'" in sql
