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
