"""A question about who may approve something is a policy question, not a question about the thing.

`_INTENTS` is first-match, and the `policies` pattern only knew polic*/rule/governance/
complian*/threshold/"approval limit"/mandate. "What is the delegation of authority limit
for approving a purchase order?" carries none of them, so it fell through to
`purchase_orders` on the noun alone and the answer opened with purchase-order totals
instead of the approval rules.

The fix names the approval/authority *phrases*. It deliberately does not add bare
`approv*`, `authori*` or `limit`: "approved suppliers", "orders awaiting approval" and
"who approved PO-2025-0270" are questions about records, and must keep their intents.
"""

import pytest

from services.corpus_facts import detect_intent


class TestApprovalAuthorityQuestionsRouteToPolicies:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the delegation of authority limit for approving a purchase order?",
            "What is the delegated authority for contracts?",
            "Who can approve a purchase order over £50,000?",
            "Who has to sign off on invoices above 10k?",
            "Who is allowed to approve a quote?",
            "What is the spend limit for a buyer?",
            "What is the authorisation limit for purchase orders?",
            "Which approval level applies to a £20,000 invoice?",
            "What is the signing authority for supplier contracts?",
        ],
    )
    def test_routes_to_policies(self, question):
        assert detect_intent(question) == "policies"


class TestRecordQuestionsKeepTheirIntent:
    """Words near the new phrases must not hijack questions about records."""

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("Show me our approved suppliers", "suppliers"),
            ("Which purchase orders are awaiting approval?", "purchase_orders"),
            ("Who approved PO-2025-0270?", "purchase_orders"),
            ("What is a purchase order?", "purchase_orders"),
            ("How many POs are still open?", "purchase_orders"),
            ("List invoices that were authorised last month", "invoices"),
            ("What is the credit limit on invoice INV-1042?", "invoices"),
        ],
    )
    def test_stays_on_the_entity(self, question, expected):
        assert detect_intent(question) == expected

    def test_a_spend_question_stays_spend(self):
        assert detect_intent("How much did we spend with Kestrel last quarter?") == "spend"
