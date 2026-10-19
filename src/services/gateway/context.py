"""Who is asking. Built by the orchestrator from the authenticated session and never by an agent or a pipeline stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CallerContext:
    principal_subject: str
    role: str
    # None = unrestricted (an admin); a tuple restricts to those buyer ids; an empty tuple sees nothing.
    buyer_scope: Optional[Tuple[str, ...]] = None
    tenant_id: str = "default"                 # one tenant today; no reseller mapping exists yet
    reseller_account_ids: Tuple[str, ...] = ()
    request_id: str = ""
    session_id: str = ""
    agent: str = ""                            # the registry name of the agent running this step, set by the orchestrator

    def for_agent(self, agent: str) -> "CallerContext":
        """The same caller, for the step an agent is running. Only the orchestrator calls this."""
        return CallerContext(self.principal_subject, self.role, self.buyer_scope, self.tenant_id,
                             self.reseller_account_ids, self.request_id, self.session_id, agent)


# Parameter names that would let a caller choose its own identity or scope. They are refused wherever they appear in
# agent-supplied params. ``reseller_id`` is deliberately NOT here: a function about a reseller (get_partner_tier) takes the
# reseller as the SUBJECT of the question. Such a function must itself intersect that id with ``ctx.reseller_account_ids``
# and treat an id outside it as not found; that rule comes with the reseller/tenant model and its cross-tenant tests (Step 10).
CONTEXT_KEYS = frozenset({"user_id", "user", "role", "tenant_id", "tenant", "buyer_scope", "buyer_id", "buyer_ids",
                          "reseller_account_ids", "principal_subject", "session_id", "agent", "caller"})
