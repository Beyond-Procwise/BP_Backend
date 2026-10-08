"""Entry guard for supplier names: table-header fragments must not become vendors.

The FALSE-POSITIVE direction (a real supplier wrongly rejected) is the more
costly failure, so the accept-list here is deliberately awkward: real names that
look like noise.
"""
import pytest

from src.services.extraction_v3 import supplier_resolver as SR


# Confirmed junk currently sitting in / reaching proc.bp_supplier.
HEADER_FRAGMENTS = [
    "Description Qty Unit Price",
    "days Tax",
    "Delivery Deadline",
    "CONDITIONS",
    # generalisation: never-seen permutations of the same furniture
    "Item No Qty Rate Amount",
    "Terms and Conditions",
    "Unit Price Total",
    "Payment Due Date",
    "Qty",
    "Subtotal Tax Total",
    "Shipping and Handling Charges",
]

CELL_ARTIFACTS = [
    "Lester Group 807",
    "Ergeremiuc School C1 6s",
    # generalisation
    "Wade Solutions 1420",
    "Acme Ltd 92",
]

# Real supplier names, including awkward ones, that MUST survive the guard.
REAL_NAMES = [
    "extraBIZ",
    "T-Shirt Dreams",
    "City of Newport",
    "180 CAR DEALER BAY, RACCO, 18105 NY",
    "Hannah M. Cold VENDOR COMPANY",
    "Veruca Organic Nuts",
    # names that legitimately reuse one or two label words
    "Total Fitness",
    "Express Delivery Ltd",
    "Rate My Agent Ltd",
    "Value Added Distribution",
    "Price Waterhouse",
    "First Rate Exchange Services Ltd",
    # names built on a short alphanumeric token
    "O2 Telefonica",
    "Level 3 Communications",
    # ordinary names
    "Assurity Ltd",
    "NexaSpark Marketing Ltd",
    "TechWorld Global",
    "Gomez, Good and Cross Trading Ltd",
    "Dixon, Reynolds and Solomon",
    "Eleanor Price Creative Studio",
    "Sarah Thompson Tech Consultant",
    "Rubilogy (Owned by Dragon Ang Enterprise)",
]


@pytest.mark.parametrize("name", HEADER_FRAGMENTS)
def test_header_fragments_rejected(name):
    # The new rule must recognise it as furniture...
    assert SR._is_header_fragment(name), f"{name!r} not seen as a header fragment"
    # ...and the guard as a whole must reject it (an older rule may fire first,
    # e.g. "payment" is an existing noise token).
    assert SR._garbage_reason(name) is not None, f"{name!r} should be rejected"


@pytest.mark.parametrize("name", CELL_ARTIFACTS)
def test_cell_artifacts_rejected(name):
    assert SR._garbage_reason(name) is not None, f"{name!r} should be rejected"


@pytest.mark.parametrize("name", REAL_NAMES)
def test_real_names_accepted(name):
    reason = SR._garbage_reason(name)
    assert reason is None, f"{name!r} wrongly rejected as {reason!r}"


def test_new_rules_do_not_fire_on_real_names():
    """The two new rules specifically must be silent on every real name.

    "3M" is included here but NOT in REAL_NAMES: it is rejected by the
    pre-existing _MIN_NAME_LEN=3 rule, which is a known, pre-existing false
    positive unrelated to this guard.
    """
    for name in REAL_NAMES + ["3M"]:
        assert not SR._is_header_fragment(name), name
        assert not SR._is_table_cell_artifact(name), name


def test_rejection_is_recorded_not_dropped():
    """A rejected candidate is written to the audit log, never silently lost."""
    logged = []

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            if "INSERT INTO proc.bp_supplier_name_reject" in sql:
                logged.append(params)

    class _Conn:
        def cursor(self):
            return _Cur()

    assert SR.resolve_or_create_supplier("Description Qty Unit Price", _Conn()) is None
    assert logged, "rejection was not recorded"
    assert logged[0][0] == "Description Qty Unit Price"  # literal value preserved
    assert logged[0][1] == "header_fragment"


def test_reject_log_failure_never_breaks_resolution():
    class _Conn:
        def cursor(self):
            raise RuntimeError("table missing")

    assert SR.resolve_or_create_supplier("days Tax", _Conn()) is None


# --- Banking words inside real company names (2026-10-08) ------------------------------------
# The noise list was matched as a SUBSTRING, so "Swift Distribution Partners Ltd" -- the winning
# bidder on the freight tender -- was rejected as "noise_token" (swift, as in SWIFT/BIC) and its
# quotes were stored with no supplier. A banking word marks a bank-details fragment only when it
# is used as one: labelled, followed by a code, or as an instruction.

NAMES_WITH_BANKING_WORDS = [
    "Swift Distribution Partners Ltd",
    "Swift Logistics",
    "Wellcome Trust",
    "National Trust Enterprises",
    "Branch Logistics Ltd",
    "Payment Solutions Ltd",
    "Invoice Cloud",
    "Premiter Ltd",            # contains "remit"
    "Tiban Engineering",       # contains "iban"
    "Barclays Bank PLC",
    "Savings Direct",
]

BANK_FRAGMENTS = [
    "SWIFT: BARCGB22",
    "SWIFT/BIC BARCGB22",
    "Swift code BARCGB22",
    "IBAN GB29 NWBK 6016 1331 9268 19",
    "Sort code 20-00-00",
    "Bank: Barclays",
    "BSB 062-000",
    "Routing number 021000021",
    "Remit to",
    "Bill To",
    "Payable to",
    "Bank",
    "Banking",
]


@pytest.mark.parametrize("name", NAMES_WITH_BANKING_WORDS)
def test_a_banking_word_inside_a_company_name_is_not_noise(name):
    reason = SR._garbage_reason(name)
    assert reason is None, f"{name!r} wrongly rejected as {reason!r}"


@pytest.mark.parametrize("name", BANK_FRAGMENTS)
def test_bank_detail_fragments_are_still_rejected(name):
    assert SR._garbage_reason(name) is not None, f"{name!r} should be rejected"


# The document-reference rule accepted letters as the reference ("INV" + "OICE"), so any name
# starting INV/PO/REC/ORD/REF/DOC/SER/QUO/BILL was rejected. A reference carries digits.
NAMES_STARTING_LIKE_A_REFERENCE = [
    "Polymer Products Ltd", "Service Masters", "Docusign", "Inventory Partners",
    "Quotient Ltd", "Recorded Books", "Ordnance Supplies", "Reference Point Ltd", "Billington Foods",
]
DOCUMENT_REFERENCES = ["INV-B-23476 PO", "PO-12345", "QUOT2024-17", "REF 99812", "ORD#88123", "INV00045"]


@pytest.mark.parametrize("name", NAMES_STARTING_LIKE_A_REFERENCE)
def test_a_name_that_starts_like_a_reference_is_not_one(name):
    reason = SR._garbage_reason(name)
    assert reason is None, f"{name!r} wrongly rejected as {reason!r}"


@pytest.mark.parametrize("name", DOCUMENT_REFERENCES)
def test_document_references_are_still_rejected(name):
    assert SR._garbage_reason(name) is not None, f"{name!r} should be rejected"
