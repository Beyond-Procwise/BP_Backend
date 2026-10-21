from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from services.people.hierarchy import HierarchyService
from services.people.models import EntityRef, Person
from services.people.resolver import RecipientResolver, StaticRules
from services.people.stores import InMemoryPeopleStore

NOW = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
D = Decimal


def org(**over):
    """CPO -> Category Manager -> Buyer; plus a Finance approver and an admin."""
    people = {
        "cpo": Person("cpo", display_name="Cora", roles=("Approver",), approval_limit=D(-1), department_id="proc", is_department_head=True),
        "cm": Person("cm", display_name="Carl", manager_subject="cpo", approval_limit=D(50000), department_id="proc"),
        "buyer": Person("buyer", display_name="Jane", manager_subject="cm", department_id="proc"),
        "fin": Person("fin", display_name="Fay", roles=("Approver",), approval_limit=D(10_000_000), department_id="fin"),
        "adm": Person("adm", display_name="Ada", roles=("Admin",)),
    }
    for k, v in over.items():
        people[k] = v(people[k]) if callable(v) else v
    return InMemoryPeopleStore(people.values())


def h(store):
    return HierarchyService(store, now=lambda: NOW)


def test_chain_and_manager():
    s = org(); hs = h(s)
    assert [p.subject for p in hs.get_chain_of_command(s.get("default", "buyer"))] == ["cm", "cpo"]
    assert [p.subject for p in hs.get_chain_of_command(s.get("default", "buyer"), max_levels=1)] == ["cm"]
    assert hs.get_manager(s.get("default", "cpo")) is None
    assert [p.subject for p in hs.get_direct_reports(s.get("default", "cpo"))] == ["cm"]


def test_cycle_does_not_hang():
    from dataclasses import replace
    s = org(cpo=lambda p: replace(p, manager_subject="buyer"))
    chain = h(s).get_chain_of_command(s.get("default", "buyer"))
    assert [p.subject for p in chain] == ["cm", "cpo"]


def test_missing_manager_falls_back_to_head_then_admin():
    from dataclasses import replace
    s = org(buyer=lambda p: replace(p, manager_subject="ghost"))
    buyer = s.get("default", "buyer")
    assert h(s).get_manager(buyer) is None
    assert h(s).get_approver_for(buyer, "po.approve", D(1000))[0].subject == "cpo"  # dept head
    s2 = org(buyer=lambda p: replace(p, manager_subject="ghost", department_id=None))
    assert h(s2).get_approver_for(s2.get("default", "buyer"), "po.approve", D(1000))[0].subject == "adm"


def test_approver_by_amount():
    s = org(); hs = h(s); buyer = s.get("default", "buyer")
    assert hs.get_approver_for(buyer, "po.approve", D(20000))[0].subject == "cm"
    assert hs.get_approver_for(buyer, "po.approve", D(60000))[0].subject == "cpo"  # unlimited


def test_requester_never_approves_own():
    s = org(); cm = s.get("default", "cm")
    assert h(s).get_approver_for(cm, "po.approve", D(20000))[0].subject == "cpo"


def test_delegation_active_expired_and_inactive():
    from dataclasses import replace
    until = NOW + timedelta(days=3)
    s = org(cm=lambda p: replace(p, delegate_subject="fin", delegate_until=until))
    buyer = s.get("default", "buyer")
    who, frm = h(s).get_approver_for(buyer, "po.approve", D(20000))
    assert (who.subject, frm) == ("fin", "cm")
    s = org(cm=lambda p: replace(p, delegate_subject="fin", delegate_until=NOW - timedelta(days=1)))
    assert h(s).get_approver_for(s.get("default", "buyer"), "po.approve", D(20000))[0].subject == "cm"
    s = org(cm=lambda p: replace(p, delegate_subject="fin", delegate_until=until),
            fin=lambda p: replace(p, is_active=False))
    assert h(s).get_approver_for(s.get("default", "buyer"), "po.approve", D(20000))[0].subject == "cm"


def test_delegation_loop_terminates():
    from dataclasses import replace
    until = NOW + timedelta(days=1)
    s = org(cm=lambda p: replace(p, delegate_subject="fin", delegate_until=until),
            fin=lambda p: replace(p, delegate_subject="cm", delegate_until=until))
    who, _ = h(s).effective_person(s.get("default", "cm"))
    assert who.subject == "fin"


def test_other_tenant_manager_is_absent():
    from dataclasses import replace
    s = org(); s.add(replace(s.get("default", "cm"), tenant_id="t2"))
    from dataclasses import replace as r
    s.add(r(s.get("default", "buyer"), tenant_id="t2"))  # t2 buyer's manager 'cm' exists in t2
    s2 = InMemoryPeopleStore([Person("b", tenant_id="t2", manager_subject="cm")])  # cm only in default
    assert h(s2).get_manager(s2.get("t2", "b")) is None


# ------------------------------------------------------------------ resolver
def resolver(store, rules=None, can_view=lambda p, e: True):
    return RecipientResolver(store, h(store), StaticRules(rules or {}), can_view)


def subjects(rs):
    return [r.person.subject for r in rs]


def test_requisition_submitted_notifies_manager_and_approver_not_actor():
    s = org(); buyer = s.get("default", "buyer")
    rs = resolver(s).resolve(buyer, "requisition.submitted", EntityRef("requisition", "R1", amount=D(60000)))
    assert subjects(rs) == ["cpo", "cm"] or set(subjects(rs)) == {"cm", "cpo"}
    assert "buyer" not in subjects(rs)
    by = {r.person.subject: r for r in rs}
    assert by["cm"].reason == "manager" and "Jane's manager" in by["cm"].detail
    assert by["cpo"].reason == "approver"


def test_dedup_keeps_strongest_reason():
    s = org(); buyer = s.get("default", "buyer")
    rs = resolver(s).resolve(buyer, "requisition.submitted", EntityRef("requisition", "R1", amount=D(100)),
                             explicit=["cm"])
    assert subjects(rs).count("cm") == 1
    assert [r.reason for r in rs if r.person.subject == "cm"] == ["explicit"]


def test_delegate_receives_with_note():
    from dataclasses import replace
    s = org(cm=lambda p: replace(p, delegate_subject="fin", delegate_until=NOW + timedelta(days=2)))
    rs = resolver(s).resolve(s.get("default", "buyer"), "requisition.submitted",
                             EntityRef("requisition", "R1", amount=D(100)))
    fin = [r for r in rs if r.person.subject == "fin"][0]
    assert fin.delegated_from == "cm" and "covering for cm" in fin.detail


def test_permission_filter_drops_unauthorised_even_if_explicit():
    s = org()
    rs = resolver(s, can_view=lambda p, e: p.subject != "fin").resolve(
        s.get("default", "buyer"), "x.y", EntityRef("po", "P1", owner="fin"), explicit=["fin", "cm"])
    assert subjects(rs) == ["cm"]


def test_cross_tenant_explicit_ignored():
    s = org(); s.add(Person("outsider", tenant_id="t2"))
    rs = resolver(s).resolve(s.get("default", "buyer"), "x.y", EntityRef("po", "P1"), explicit=["outsider"])
    assert rs == []


def test_tenant_rule_overrides_default_and_notify_actor():
    s = org()
    rules = {("default", "po.issued"): (["entity_owner", "requester", "team", "watcher"], True)}
    e = EntityRef("po", "P1", owner="cm", requester="buyer", team=("fin",), watchers=("cpo",))
    rs = resolver(s, rules).resolve(s.get("default", "buyer"), "po.issued", e)
    assert {r.person.subject: r.reason for r in rs} == {
        "cm": "entity_owner", "buyer": "requester", "fin": "team_member", "cpo": "watcher"}


def test_broken_rule_source_uses_safe_default():
    class Boom:
        def targets_for(self, *a): raise RuntimeError("db down")
    s = org()
    r = RecipientResolver(s, h(s), Boom(), lambda p, e: True)
    rs = r.resolve(s.get("default", "buyer"), "approval.requested", EntityRef("requisition", "R", amount=D(100)))
    assert subjects(rs) == ["cm"]


def test_inactive_recipient_dropped():
    from dataclasses import replace
    s = org(cm=lambda p: replace(p, is_active=False))
    rs = resolver(s).resolve(s.get("default", "buyer"), "requisition.submitted",
                             EntityRef("requisition", "R", amount=D(100)))
    assert "cm" not in subjects(rs)


def test_actor_excluded_when_they_qualify_unless_configured():
    s = org(); buyer = s.get("default", "buyer")
    e = EntityRef("po", "P1", requester="buyer", owner="cm")
    rs = resolver(s, {("default", "po.changed"): (["requester", "entity_owner"], False)}).resolve(buyer, "po.changed", e)
    assert subjects(rs) == ["cm"]
    rs = resolver(s, {("default", "po.changed"): (["requester", "entity_owner"], True)}).resolve(buyer, "po.changed", e)
    assert set(subjects(rs)) == {"cm", "buyer"}
