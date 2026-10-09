"""Shared fixtures for the agent-policy tests."""
from services.agent_policy import registry, settings

FORM_EXAMPLE = {
  "name": "Refund or credit over $500", "category": "Approval",
  "businessArea": "Finance", "subArea": "Refunds and credits",
  "situation": "The agent is about to issue a refund or credit above $500.",
  "source": {"document": "Finance Payments Policy", "documentVersion": 1, "reference": "1.1",
             "excerpt": "Refunds or credits above $500 need approval from the Finance Manager."},
  "outcome": "approve",                      # approve | block | notify | None
  "outcomeBecause": "need approval from",    # the document phrase behind the suggestion, or None
  "deciders": ["Finance Manager", "CFO"],    # approve only, ordered
  "responseTime": None,                      # None = company default, else ISO 8601 duration
  "notify": [],                              # notify (required) / block (optional)
  "limit": {"on": False, "text": ""},
  "owner": "Chief Financial Officer", "effectiveFrom": "2026-02-01", "reviewBy": None,
  "messageForAgent": "Refunds over $500 need Finance approval.",
  "messageForPerson": "Your refund needs a manager's approval. You will hear back within 4 hours.",
  "hidden": {
    "checkpoint": "tool.call.before",
    "actions": {"tools": ["refund.issue", "credit.issue"], "plain": "issuing a refund or credit"},
    "timeWindow": None,                      # or {"days": [...], "from": "18:00", "to": "08:00", "timeZone": "Europe/London"}
    "units": {"currency": "USD", "convertOther": "rate_on_action_date", "amountsIncludeTax": True},
    "inputs": [
      {"name": "Refund amount", "field": "args.amount", "type": "number", "isAmount": True, "unit": "USD",
       "from": "action", "showApprover": True, "sensitive": False},
      {"name": "Tool", "field": "tool.name", "type": "string", "from": "action", "showApprover": False, "sensitive": False},
      {"name": "Agent's reason", "field": "agent.reason", "type": "string", "from": "action", "showApprover": True, "sensitive": False},
    ],
    "missingInputs": [],                     # [{"name": "...", "reason": "..."}] from the agent
    "unknownNames": [],                      # names the agent needed but the registry lacks
    "condition": {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                          {"field": "args.amount", "op": "gt", "value": 500}]},
    "onMissingData": None,                   # None = company setting
    "reasonCode": "over_limit",
    "whilePaused": "no_retry",
    "setBy": "extraction_agent",             # or "person"
  },
  "examples": [
    {"input": {"tool.name": "refund.issue", "args.amount": 501}, "agentExpected": "approve", "flipped": False},
    {"input": {"tool.name": "refund.issue", "args.amount": 500}, "agentExpected": "none", "flipped": False},
    {"input": {"tool.name": "refund.issue", "args.amount": 499}, "agentExpected": "none", "flipped": False},
    {"input": {"tool.name": "supplier_ranking", "args.amount": 900}, "agentExpected": "none", "flipped": False},
  ],
  "checked": None,                           # {"by": "<subject>", "at": "<ISO>"} once confirmed
  "changeNote": "",
}


SETTINGS = settings.merge(None)


def _a(name):
    return {"kind": "action", "name": name, "checkpoint": "tool.call.before", "plain": name, "status": "live"}


def _i(name, plain, vt, source="action", status="live"):
    return {"kind": "input", "name": name, "checkpoint": "tool.call.before", "plain": plain,
            "value_type": vt, "source": source, "status": status}


REGISTRY = registry.snapshot_from_rows([
    {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None,
     "plain": "before a tool runs", "status": "live"},
    _a("refund.issue"), _a("credit.issue"), _a("supplier_ranking"),
    _i("tool.name", "tool name", "string"),
    _i("agent.name", "agent name", "string"),
    _i("agent.reason", "the agent's reason", "string"),
    _i("args.amount", "amount", "number"),
    _i("args.currency", "currency", "string"),
    _i("agg.refunds_30d", "refunds in 30 days", "number", "total:refunds_30d", "planned"),
])


def precedent_n(monkeypatch, n, *, missing=False):
    """Set the governed precedent count (agent_policy_conflicts.precedent_count) every fresh read
    sees. missing=True: the row does not exist, so the read raises LimitUnavailable."""
    from src.services import governed_limits as GL

    class _Engine:
        def get_policy(self, slug):
            if missing or slug != "agent_policy_conflicts":
                return None
            return {"policyName": "AgentPolicyConflictPolicy",
                    "details": {"policy_identifier": slug, "rules": {"precedent_count": n}}}

    monkeypatch.setattr(GL, "_fresh_engine", lambda: _Engine())
