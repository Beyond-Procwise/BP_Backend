"""Getting the organisation into bp_person: a small-company starter chart, or a feed from the company's own system.

Two ways in, one gate:
  * onboarding: starter_chart() is the smallest editable template: the signed-in admin as Owner (unlimited); the rest is added by hand;
  * a connected org-chart product (HR system, Microsoft 365 directory, ERP) is an OrgChartProvider whose
    fetch() returns PersonRecords. It is treated like the mailbox connection: the provider is configured by
    an admin, and its output goes through the same import as a hand-entered chart.

import_people is ALL OR NOTHING. The whole batch is checked first (every field through the same validator as
PUT /admin/people, managers resolved inside the batch, reporting loops refused); if anything is wrong nothing
is written and every problem is returned. Records are applied managers-first so a manager always exists
before their reports. People missing from the batch are left alone unless deactivate_missing is set, and
even then they are deactivated, never deleted.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from services.people import admin as svc
from services.people.models import Person
from services.people.stores import InMemoryPeopleStore


class OrgChartProvider(Protocol):
    """A connected system that knows who works here. fetch() returns plain dicts shaped like PUT /admin/people
    plus a 'subject' (the sign-in identity the person will arrive with)."""

    name: str

    def fetch(self) -> List[Dict[str, Any]]: ...


def starter_chart(owner_subject: str, owner_email: Optional[str] = None) -> List[Dict[str, Any]]:
    """The smallest chart that makes approvals route: an owner who approves anything, and a place to add the rest."""
    return [
        {"subject": owner_subject, "display_name": "Owner", "email": owner_email, "title": "Owner / Finance lead",
         "roles": ["Admin"], "approval_limit": -1, "is_department_head": True, "department_id": "Management"},
    ]


def _as_person(subject: str, p: Dict[str, Any]) -> Person:
    return Person(subject=subject, display_name=p.get("display_name") or "", manager_subject=p.get("manager_subject"),
                  department_id=p.get("department_id"), is_department_head=bool(p.get("is_department_head")),
                  roles=tuple(p.get("roles") or ()), approval_limit=p.get("approval_limit"),
                  is_active=p.get("is_active", True) is not False, title=p.get("title"))


def plan_import(existing: List[Dict[str, Any]], records: List[Dict[str, Any]], *,
                deactivate_missing: bool = False) -> Dict[str, Any]:
    """Check a batch against what exists. Returns {'writes': [(subject, clean)], 'errors': [...], 'created', 'updated', 'deactivated'}.
    Pure: touches nothing."""
    errors: List[Dict[str, str]] = []
    batch: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(records):
        subject = str(rec.get("subject") or "").strip()
        if not subject:
            errors.append({"row": str(i + 1), "subject": "", "problem": "every person needs a subject"})
        elif subject in batch:
            errors.append({"row": str(i + 1), "subject": subject, "problem": "listed twice"})
        else:
            batch[subject] = {k: v for k, v in rec.items() if k != "subject"}

    store = InMemoryPeopleStore(
        _as_person(e["subject"], {**e, "approval_limit": e["approval_limit"]}) for e in existing)
    known = {e["subject"] for e in existing}

    # managers first: a record waits until its manager is either already known or already placed
    ordered: List[str] = []
    placed = set(known)
    pending = list(batch)
    while pending:
        progress = [s for s in pending if not batch[s].get("manager_subject")
                    or batch[s]["manager_subject"] in placed or batch[s]["manager_subject"] == s]
        if not progress:
            break
        for s in progress:
            ordered.append(s); placed.add(s); pending.remove(s)
    for s in pending:
        m = batch[s].get("manager_subject")
        errors.append({"row": "", "subject": s,
                       "problem": (f"manager '{m}' is not in the chart" if m not in batch
                                   else "this would make a reporting loop")})

    writes = []
    for s in ordered:
        try:
            clean = svc.validate_person_update(store, svc.TENANT, s, batch[s])
        except svc.PeopleError as exc:
            errors.append({"row": "", "subject": s, "problem": exc.detail})
            continue
        prior = store.get(svc.TENANT, s)
        store.add(_as_person(s, {**(_asdict(prior) if prior else {}), **clean}))
        writes.append((s, clean))

    deactivated: List[str] = []
    if deactivate_missing and not errors:
        deactivated = sorted(e["subject"] for e in existing if e.get("is_active") and e["subject"] not in batch)
        # never leave the company without an active Admin
        survivors = [e for e in existing if e["subject"] not in deactivated and e.get("is_active")] + \
                    [{"roles": b.get("roles") or []} for b in batch.values() if b.get("is_active", True) is not False]
        if not any("Admin" in (x.get("roles") or []) for x in survivors):
            errors.append({"row": "", "subject": "", "problem": "that would leave no active Admin"})
            deactivated = []
    return {"writes": writes, "errors": errors, "created": sorted(s for s, _ in writes if s not in known),
            "updated": sorted(s for s, _ in writes if s in known), "deactivated": deactivated}


def _asdict(p: Person) -> Dict[str, Any]:
    return {"display_name": p.display_name, "manager_subject": p.manager_subject, "department_id": p.department_id,
            "is_department_head": p.is_department_head, "roles": list(p.roles), "approval_limit": p.approval_limit,
            "is_active": p.is_active, "title": p.title}


def import_people(conn, records: List[Dict[str, Any]], by: str, *, dry_run: bool = False,
                  deactivate_missing: bool = False) -> Dict[str, Any]:
    """Apply a batch. Raises PeopleError(422) with every problem and writes nothing when any record is wrong."""
    from services.agent_policy.approvals import _tx
    plan = plan_import(svc.list_people(conn, include_inactive=True), records, deactivate_missing=deactivate_missing)
    if plan["errors"]:
        raise svc.PeopleError(422, "; ".join(f"{e['subject'] or 'row ' + e['row']}: {e['problem']}" for e in plan["errors"][:20]))
    result = {k: plan[k] for k in ("created", "updated", "deactivated")}
    result["dry_run"] = dry_run
    if dry_run:
        return result
    with _tx(conn):
        for subject, clean in plan["writes"]:
            svc.upsert_person(conn, subject, clean, by)
        for subject in plan["deactivated"]:
            svc.upsert_person(conn, subject, {"is_active": False}, by)
    return result
