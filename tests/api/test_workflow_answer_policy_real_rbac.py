"""The answer rule against the REAL rbac role ranks (only the policy store is a stub)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services import workflow_answer_policy as wap

RULES = {
    "role_definition": {"roles": {"Viewer": {"rank": 1}, "Buyer": {"rank": 2},
                                  "Approver": {"rank": 3}, "Admin": {"rank": 4}}},
    "role_assignment": {"claim": "cognito:groups", "no_principal_role": "Viewer",
                        "unmapped_group_role": "Viewer", "multiple_groups": "highest_rank",
                        "group_to_role": {"PROCWISE_VIEWER": "Viewer", "PROCWISE_APPROVER": "Approver",
                                          "PROCWISE_ADMIN": "Admin"}},
}


class StubEngine:
    def get_policy(self, slug):
        return {"details": {"rules": RULES[slug]}} if slug in RULES else None


@dataclass
class P:
    subject: str
    groups: list
    claims: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.claims = {"cognito:groups": self.groups}


def test_real_ranks_approver_allowed_viewer_refused(monkeypatch):
    from services import rbac
    monkeypatch.setattr(rbac, "_direct_roles", lambda subject: [])
    e = StubEngine()
    ok = wap.decide(P("m", ["PROCWISE_APPROVER"]), "starter", True, mode="approver_only", engine=e)
    no = wap.decide(P("v", ["PROCWISE_VIEWER"]), "starter", True, mode="approver_only", engine=e)
    unmapped = wap.decide(P("u", ["SOMETHING_ELSE"]), "starter", True, mode="approver_only", engine=e)
    admin = wap.decide(P("a", ["PROCWISE_ADMIN"]), "starter", True, mode="approver_only", engine=e)
    assert (ok.allowed, ok.role) == (True, "Approver")
    assert (no.allowed, no.role) == (False, "Viewer")
    assert not unmapped.allowed and admin.allowed
