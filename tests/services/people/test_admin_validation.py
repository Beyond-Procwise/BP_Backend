from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from services.people import admin as a
from tests.services.people.test_people import NOW, org

T = "default"
NOWREAL = datetime.now(timezone.utc)


def v(subject, fields, store=None):
    return a.validate_person_update(store or org(), T, subject, fields, now=NOWREAL)


def bad(subject, fields, text, store=None):
    with pytest.raises(a.PeopleError) as e:
        v(subject, fields, store)
    assert text in e.value.detail


def test_unknown_field_refused():
    bad("buyer", {"tenant_id": "x"}, "cannot set")


def test_manager_rules():
    bad("buyer", {"manager_subject": "buyer"}, "own manager")
    bad("buyer", {"manager_subject": "ghost"}, "not a known person")
    bad("cpo", {"manager_subject": "buyer"}, "reporting loop")        # cpo -> buyer -> cm -> cpo
    assert v("buyer", {"manager_subject": "fin"}) == {"manager_subject": "fin"}
    assert v("buyer", {"manager_subject": None}) == {"manager_subject": None}


def test_approval_limit_rules():
    assert v("buyer", {"approval_limit": 5000})["approval_limit"] == Decimal(5000)
    assert v("buyer", {"approval_limit": -1})["approval_limit"] == Decimal(-1)
    assert v("buyer", {"approval_limit": None})["approval_limit"] is None
    bad("buyer", {"approval_limit": -5}, "0 or more")
    bad("buyer", {"approval_limit": "lots"}, "number")


def test_delegate_rules():
    soon = (NOWREAL + timedelta(days=2)).isoformat()
    out = v("cm", {"delegate_subject": "fin", "delegate_until": soon})
    assert out["delegate_subject"] == "fin" and out["delegate_until"] > NOWREAL
    bad("cm", {"delegate_subject": "cm", "delegate_until": soon}, "themselves")
    bad("cm", {"delegate_subject": "ghost", "delegate_until": soon}, "active, known")
    bad("cm", {"delegate_subject": "fin", "delegate_until": None}, "end date")
    bad("cm", {"delegate_subject": "fin", "delegate_until": (NOWREAL - timedelta(days=1)).isoformat()}, "future")
    bad("cm", {"delegate_until": "tomorrow-ish"}, "ISO")
    s = org(fin=lambda p: replace(p, is_active=False))
    bad("cm", {"delegate_subject": "fin", "delegate_until": soon}, "active, known", s)
    cleared = v("cm", {"delegate_subject": None})
    assert cleared == {"delegate_subject": None, "delegate_until": None}


def test_types():
    bad("buyer", {"is_active": "yes"}, "true or false")
    bad("buyer", {"roles": "Admin"}, "list")
    assert v("buyer", {"roles": [" Buyer", "Buyer"]})["roles"] == ["Buyer"]


def test_rule_validation():
    assert a.validate_rule("po.issued", ["entity_owner", "entity_owner"], True) == (["entity_owner"], True)
    with pytest.raises(a.PeopleError) as e: a.validate_rule("nope.x", ["manager"], False)
    assert e.value.status == 404
    with pytest.raises(a.PeopleError) as e: a.validate_rule("po.issued", ["everyone"], False)
    assert "unknown targets" in e.value.detail
    with pytest.raises(a.PeopleError): a.validate_rule("po.issued", [], False)


def test_prefs_validation():
    assert a.validate_prefs({"channels": ["email"], "quiet_hours": ["22:00", "07:00"], "muted_events": ["po.changed"]})
    for body, text in [({"channels": ["sms"]}, "channels may only"), ({"quiet_hours": ["25:00", "07:00"]}, "HH:MM"),
                       ({"muted_events": ["approval.requested"]}, "cannot be muted"),
                       ({"muted_events": ["nope"]}, "known event"), ({"theme": "x"}, "unknown preference"), ("x", "object")]:
        with pytest.raises(a.PeopleError) as e:
            a.validate_prefs(body)
        assert text in e.value.detail
    assert a.validate_prefs({"quiet_hours": None}) == {"quiet_hours": None}
