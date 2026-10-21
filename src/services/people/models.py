"""Value objects shared by the hierarchy service and the recipient resolver."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple

UNLIMITED = Decimal(-1)


@dataclass(frozen=True)
class Person:
    subject: str
    tenant_id: str = "default"
    display_name: str = ""
    manager_subject: Optional[str] = None
    department_id: Optional[str] = None
    is_department_head: bool = False
    roles: Tuple[str, ...] = ()
    approval_limit: Optional[Decimal] = None
    is_active: bool = True
    delegate_subject: Optional[str] = None
    delegate_until: Optional[datetime] = None

    @property
    def is_admin(self) -> bool:
        return "Admin" in self.roles


@dataclass(frozen=True)
class EntityRef:
    """The thing an event is about, with the people who stand in a fixed relation to it."""
    entity_type: str
    entity_id: str
    owner: Optional[str] = None
    requester: Optional[str] = None
    category_owner: Optional[str] = None
    team: Tuple[str, ...] = ()
    watchers: Tuple[str, ...] = ()
    amount: Optional[Decimal] = None


@dataclass(frozen=True)
class Recipient:
    person: Person
    reason: str                       # explicit|manager|chain_of_command|approver|entity_owner|...
    detail: str = ""                  # human text shown to the user: "You're Jane's manager"
    delegated_from: Optional[str] = None


class PeopleStore(Protocol):
    def get(self, tenant_id: str, subject: str) -> Optional[Person]: ...
    def direct_reports(self, tenant_id: str, subject: str) -> List[Person]: ...
    def department_head(self, tenant_id: str, department_id: str) -> Optional[Person]: ...
    def admins(self, tenant_id: str) -> List[Person]: ...
