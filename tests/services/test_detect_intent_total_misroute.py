"""A measure word must not drag a question about one entity into the `spend` intent.

`_INTENTS` is a first-match ordered list, and `spend` sits above `quotes`,
`purchase_orders`, `invoices` and `suppliers`. Its pattern matches the generic measure
words — total, value, amount, cost — that appear in questions *about* those four
entities. So "what is the total value of our purchase orders" matched `spend` before
`purchase_orders` was ever tested.

That is not a mislabel. `_fetch("spend")` reads proc.bp_invoice_trgt, so a question
about purchase orders was answered with invoice figures: a confidently wrong number,
which is worse than no answer. `deals`, `findings` and `policies` were never exposed to
this only because they are scanned before `spend`.

`_count_subject_intent` already rescued the "how many X ... in total" phrasing. These
cover the phrasings it does not reach, and pin the two boundaries that make the rescue
safe: a question carrying a real expenditure word (spend/spent/invoiced/money) stays on
`spend` however many entities it names, and the count-subject rescue keeps precedence.
"""

import pytest

from services.corpus_facts import detect_intent


class TestMeasureWordDoesNotHijackTheSubject:
    """The entity the question is about wins over a bare measure word."""

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("What is the total value of our purchase orders?", "purchase_orders"),
            ("What is the total value of our open quotes?", "quotes"),
            ("Which quote has the highest total?", "quotes"),
            ("What is the total amount on invoice INV-1042?", "invoices"),
            ("List invoices and their total values", "invoices"),
            ("What is the cost of each purchase order?", "purchase_orders"),
        ],
    )
    def test_routes_to_the_entity_not_to_spend(self, question, expected):
        assert detect_intent(question) == expected


class TestPurchaseOrderPluralMatches:
    """The plural is the form people type, and it did not match at all.

    In ``\\b(purchase order|po|pos)\\b`` the trailing boundary binds after ``order``, and
    "purchase orders" has no boundary between "order" and "s". So the whole intent was
    reachable only in the singular: "list all purchase orders" matched no intent, returned
    None, and fell through to vector search — the PO facts were never fetched. Separate
    from the measure-word hijack, but it is why the hijack had nothing to hand back to.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "Show me our purchase orders",
            "List all purchase orders",
            "Which purchase orders are open?",
            "What is a purchase order?",
            "How many POs are still open?",
            "Is there a PO for this invoice?",
        ],
    )
    def test_singular_and_plural_both_route(self, question):
        assert detect_intent(question) == "purchase_orders"

    @pytest.mark.parametrize(
        "question",
        [
            "What does our policy say about single sourcing?",
            "Is that possible under the rules?",
        ],
    )
    def test_the_po_abbreviation_does_not_match_inside_a_longer_word(self, question):
        """`po` must stay boundary-anchored — "policy" and "possible" are not POs."""
        assert detect_intent(question) != "purchase_orders"


class TestGenuineSpendQuestionsStaySpend:
    """An expenditure word means the question really is about spend.

    This is the guard that stops the fix becoming a new mis-route in the other
    direction. "How much have we spent with each supplier" names a supplier, but it is
    a spend question and `spend` is the intent that carries the per-supplier amounts.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "What is our total spend?",
            "How much have we spent in total this year?",
            "What is the total invoiced amount by supplier?",
            "How much have we spent with each supplier?",
            "Show me spend by supplier",
            "How much money went out the door last quarter?",
        ],
    )
    def test_expenditure_word_keeps_the_spend_intent(self, question):
        assert detect_intent(question) == "spend"


class TestCountSubjectRescueStillWins:
    """The existing "how many X" rescue keeps precedence over the entity scan.

    "How many suppliers do we have invoices from" names invoices, so a plain entity
    scan would route it to `invoices` and answer with a document list instead of the
    count of suppliers that was asked for. The count subject is authoritative.
    """

    @pytest.mark.parametrize(
        "question, expected",
        [
            (
                "How many suppliers do we actually have invoices from, "
                "and how many are on record in total?",
                "suppliers",
            ),
            ("How many suppliers in total?", "suppliers"),
            ("What is the total number of invoices?", "invoices"),
            ("How many purchase orders do we have in total?", "purchase_orders"),
        ],
    )
    def test_count_subject_is_authoritative(self, question, expected):
        assert detect_intent(question) == expected


class TestHigherRankedIntentsAreUntouched:
    """Intents scanned above `spend` never reach the rescue and must not change."""

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("What is the total cost of the Techworld deal?", "deals"),
            ("Show me the total value of each deal", "deals"),
            ("What is the total value of our open discrepancies?", "findings"),
            ("What approval limit applies to spend over ten thousand?", "policies"),
        ],
    )
    def test_topic_outranks_the_measure_word(self, question, expected):
        assert detect_intent(question) == expected


class TestNonQuestionsAreUnchanged:
    @pytest.mark.parametrize("value", ["", "   ", None, 42])
    def test_no_intent_without_a_query(self, value):
        assert detect_intent(value) is None

    def test_a_question_matching_nothing_falls_through_to_vector_search(self):
        assert detect_intent("What did we discuss at the offsite?") is None
