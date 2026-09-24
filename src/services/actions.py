"""The canonical action names a policy may be written against.

``guardrail.authorize`` matches a policy to an action by name, through
``policies_for_action`` reading ``policy_details->'applies_to'``. A name that can
be spelled two ways cannot be aggregated, cannot be reported on, and -- worst --
a policy written against ``email_send`` simply never matches ``email.send``, so
the rule silently governs nothing. That failure looks exactly like the rule
working.

So the vocabulary is closed and it lives here, in code, next to the gate that
consumes it. Adding an action is a deliberate edit with a test behind it, not
something a policy row can invent.

SHAPE: ``domain.verb``. The domain is the thing acted on, the verb is what is
done to it. Both lower case, dot separated, no plurals.

Each entry carries the action class it belongs to, because that is what decides
whether it is irreversible and therefore default-refused. The class vocabulary
is RoleDefinitionPolicy's, not this module's -- these names must match the ones
in ``rules.roles.*.allow``.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

#: action name -> the class it belongs to (see RoleDefinitionPolicy)
ACTIONS: Dict[str, str] = {
    # --- reading ---------------------------------------------------------
    "supplier.read": "read",
    "contract.read": "read",
    "quote.read": "read",
    # Internal cost and margin. A read, but restricted by required_role in
    # SalesMarginReadAuthorityPolicy -- and the sales router refuses it unless
    # that stated permit, not the reversible-read default, is what allowed it.
    "margin.read": "read",
    "invoice.read": "read",
    "deal.read": "read",
    "finding.read": "read",
    "report.read": "read",
    "policy.read": "read",
    # --- computing -------------------------------------------------------
    "quote.rank": "compute",
    "quote.compare": "compute",
    "opportunity.mine": "compute",
    "document.extract": "compute",
    "sales.calibrate": "compute",
    # Building a report inside the tenant changes nothing; sending one out is
    # report.export, a share, and is refused unless a rule says otherwise.
    "report.generate": "compute",
    # --- writing ---------------------------------------------------------
    "supplier.write": "write",
    "contract.write": "write",
    "deal.write": "write",
    "finding.resolve": "write",
    "document.upload": "write",
    "document.promote": "write",
    "email.draft": "write",
    "catalog.write": "write",
    "account.write": "write",
    "sales.write": "write",
    # --- communicating ---------------------------------------------------
    "email.send": "communicate",
    "email.reply": "communicate",
    "sales_quote.issue": "communicate",
    # --- transacting -----------------------------------------------------
    "spend.approve": "transact",
    "negotiation.counter": "transact",
    "sales_quote.approve": "transact",
    # Signing off a report commits the company to what it says before it leaves;
    # who may is policy (report_signoff_authority), which reports need it too.
    "report.signoff": "transact",
    # --- approving -------------------------------------------------------
    "approval.email": "approve_email",
    "approval.record": "approve_email",
    "approval.revoke": "approve_email",
    # --- sharing (leaves the tenant) -------------------------------------
    "report.export": "share",
    "research.web": "share",
    "supplier.clearance.set": "share",
    # --- configuring -----------------------------------------------------
    "policy.write": "configure",
    "policy.reload": "configure",
    "prompt.write": "configure",
    "model.train": "configure",
    "mailbox.bind": "configure",
    # --- delegating ------------------------------------------------------
    "agent.create": "delegate",
    "agent.delete": "delegate",
    "agent.run": "delegate",
    "workflow.save": "delegate",
    "workflow.run": "delegate",
}

KNOWN: FrozenSet[str] = frozenset(ACTIONS)


def action_class(action: str) -> str:
    """The class ``action`` belongs to.

    Raises for an unknown name rather than guessing. A typo that resolved to a
    default class would be governed by the wrong rules, which is worse than
    being governed by none.
    """

    try:
        return ACTIONS[action]
    except KeyError:
        raise KeyError(
            f"{action!r} is not a known action. Add it to services/actions.ACTIONS "
            "with its class, or correct the spelling -- a policy written against "
            "an unknown name governs nothing."
        ) from None


def is_known(action: str) -> bool:
    return action in KNOWN
