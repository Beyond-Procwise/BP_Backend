"""PeopleStore implementations: in-memory (tests, seed demos) and Postgres (proc.bp_person)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from services.people.models import Person


class InMemoryPeopleStore:
    def __init__(self, people: Iterable[Person] = ()) -> None:
        self._by_key: Dict[Tuple[str, str], Person] = {(p.tenant_id, p.subject): p for p in people}

    def add(self, p: Person) -> None:
        self._by_key[(p.tenant_id, p.subject)] = p

    def get(self, tenant_id, subject):
        return self._by_key.get((tenant_id, subject))

    def direct_reports(self, tenant_id, subject):
        return [p for (t, _), p in self._by_key.items() if t == tenant_id and p.manager_subject == subject]

    def department_head(self, tenant_id, department_id):
        for (t, _), p in self._by_key.items():
            if t == tenant_id and p.department_id == department_id and p.is_department_head and p.is_active:
                return p
        return None

    def admins(self, tenant_id):
        return [p for (t, _), p in self._by_key.items() if t == tenant_id and p.is_admin]


_COLS = ("subject, tenant_id, display_name, manager_subject, department_id, is_department_head, "
         "roles, approval_limit, is_active, delegate_subject, delegate_until, title")


def _row(r) -> Person:
    return Person(subject=r[0], tenant_id=r[1], display_name=r[2] or "", manager_subject=r[3],
                  department_id=r[4], is_department_head=bool(r[5]), roles=tuple(r[6] or ()),
                  approval_limit=r[7], is_active=bool(r[8]), delegate_subject=r[9], delegate_until=r[10], title=r[11])


class PgPeopleStore:
    """Read-only; writes go through the admin endpoints (Phase 5) so every change is audited."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def _q(self, where: str, args: tuple) -> List[Person]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {_COLS} FROM proc.bp_person WHERE {where}", args)
            return [_row(r) for r in cur.fetchall()]

    def get(self, tenant_id, subject):
        rows = self._q("tenant_id=%s AND subject=%s", (tenant_id, subject))
        return rows[0] if rows else None

    def direct_reports(self, tenant_id, subject):
        return self._q("tenant_id=%s AND manager_subject=%s", (tenant_id, subject))

    def department_head(self, tenant_id, department_id):
        rows = self._q("tenant_id=%s AND department_id=%s AND is_department_head AND is_active "
                       "ORDER BY subject LIMIT 1", (tenant_id, department_id))
        return rows[0] if rows else None

    def admins(self, tenant_id):
        return self._q("tenant_id=%s AND 'Admin' = ANY(roles) AND is_active ORDER BY subject", (tenant_id,))
