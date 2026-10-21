"""Reporting lines, approver lookup and out-of-office delegation.

Every lookup is tenant-scoped: a manager or delegate in another tenant is treated as absent.
Loops in manager_subject (A -> B -> A) end the walk instead of hanging it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Tuple

from services.people.models import UNLIMITED, PeopleStore, Person

_MAX_DELEGATION_HOPS = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HierarchyService:
    def __init__(self, store: PeopleStore, now=_now) -> None:
        self._store = store
        self._now = now

    def get_manager(self, user: Person) -> Optional[Person]:
        if not user.manager_subject or user.manager_subject == user.subject:
            return None
        return self._store.get(user.tenant_id, user.manager_subject)

    def get_chain_of_command(self, user: Person, max_levels: int = 10) -> List[Person]:
        chain: List[Person] = []
        seen = {user.subject}
        cur = user
        while len(chain) < max_levels:
            mgr = self.get_manager(cur)
            if mgr is None or mgr.subject in seen:
                break
            chain.append(mgr)
            seen.add(mgr.subject)
            cur = mgr
        return chain

    def get_direct_reports(self, user: Person) -> List[Person]:
        return [p for p in self._store.direct_reports(user.tenant_id, user.subject) if p.is_active]

    def get_department_head(self, user: Person) -> Optional[Person]:
        if not user.department_id:
            return None
        head = self._store.department_head(user.tenant_id, user.department_id)
        return head if head and head.subject != user.subject and head.is_active else None

    def effective_person(self, person: Person) -> Tuple[Person, Optional[str]]:
        """Who actually receives for ``person``: their delegate while out of office, else themselves.

        Returns (recipient, original_subject_if_delegated). A delegate who is inactive, in another
        tenant, or unknown is ignored; delegate-of-delegate is followed a few hops, loops stop.
        """
        cur, original, seen = person, None, {person.subject}
        for _ in range(_MAX_DELEGATION_HOPS):
            until = cur.delegate_until
            if not (cur.delegate_subject and until and until >= self._now()):
                break
            nxt = self._store.get(cur.tenant_id, cur.delegate_subject)
            if nxt is None or not nxt.is_active or nxt.subject in seen:
                break
            original = original or person.subject
            seen.add(nxt.subject)
            cur = nxt
        return cur, original

    def get_approver_for(
        self, user: Person, action: str, amount: Optional[Decimal] = None
    ) -> Optional[Tuple[Person, Optional[str]]]:
        """First person up the chain whose limit covers ``amount``; never the requester.

        Falls back to the department head, then an org admin. ``action`` is accepted so limits
        can become per-action later; today one ``approval_limit`` applies to all actions.
        Returns (recipient, delegated_from) after delegation, or None if nobody can be found.
        """
        need = Decimal(amount) if amount is not None else Decimal(0)

        def covers(p: Person) -> bool:
            if not p.is_active or p.subject == user.subject or p.approval_limit is None:
                return False
            return p.approval_limit == UNLIMITED or p.approval_limit >= need

        candidates = list(self.get_chain_of_command(user))
        head = self.get_department_head(user)
        if head:
            candidates.append(head)
        for p in candidates:
            if covers(p):
                return self.effective_person(p)
        for admin in self._store.admins(user.tenant_id):
            if admin.is_active and admin.subject != user.subject:
                return self.effective_person(admin)
        return None
