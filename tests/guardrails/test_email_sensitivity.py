"""Sensitivity classification, proven by the cases that must be blocked.

Every detector gets a test that fires it and a test that does not, because a
detector that fires on everything is as useless as one that never fires.
"""

import pytest

from src.services import email_sensitivity as sens
from tests.guardrails.test_rbac import FakePolicyEngine


SENSITIVITY_POLICY = {
    "policyId": "email_sensitivity",
    "policyName": "EmailSensitivityPolicy",
    "details": {
        "policy_identifier": "email_sensitivity",
        "required_role": "Admin",
        "rules": {
            "classes": ["public", "internal", "commercial_confidential", "personal"],
            "order": {
                "public": 1,
                "internal": 2,
                "commercial_confidential": 3,
                "personal": 4,
            },
            "default_supplier_clearance": "internal",
            "detectors": {
                "third_party_price": {
                    "enabled": True,
                    "raises_to": "commercial_confidential",
                },
                "contract_prose": {
                    "enabled": True,
                    "raises_to": "commercial_confidential",
                },
                "internal_staff_contact": {"enabled": True, "raises_to": "personal"},
                "source_document_attached": {
                    "enabled": True,
                    "raises_to": "commercial_confidential",
                },
            },
            "rule": "content_class <= recipient_clearance",
            "on_undetermined_class": "deny",
            "on_missing_clearance": "use_default",
        },
    },
}


@pytest.fixture
def engine():
    return FakePolicyEngine({"email_sensitivity": SENSITIVITY_POLICY})


def classify(engine, **kwargs):
    defaults = dict(
        subject="Request for quotation",
        body="Please quote for 100 units.",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=["ourcompany.com"],
        sender=None,
    )
    defaults.update(kwargs)
    return sens.classify(policy_engine=engine, **defaults)


def test_plain_rfq_is_internal(engine):
    result = classify(engine)
    assert result.content_class == "internal"
    assert result.detectors_fired == []


def test_competitor_price_raises_to_commercial_confidential(engine):
    result = classify(
        engine,
        body="Supplier B quoted 12,450.00 for the same line.",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert result.content_class == "commercial_confidential"
    assert "third_party_price" in result.detectors_fired


def test_own_price_does_not_fire_the_peer_detector(engine):
    """Quoting the recipient their own number is not a leak."""
    result = classify(
        engine,
        body="You quoted 12,450.00 last month.",
        peer_prices=[{"supplier_id": "SUP-1", "amount": "12450.00"}],
    )
    assert "third_party_price" not in result.detectors_fired


@pytest.mark.parametrize(
    "body",
    [
        "PO-991245000-A",
        "Part no 4512450008812",
        "We need 5000 units by Friday.",
        "We need 12 pallets, 450 units total",
    ],
)
def test_unrelated_digit_runs_do_not_fire_third_party_price(engine, body):
    """A digit substring of a PO number, part number, or quantity is not a price.

    Flattening the whole message into one digit string and substring-matching
    destroys every number boundary -- a reference code or a quantity can
    contain the same digits as an unrelated peer amount by pure coincidence.
    """
    result = classify(
        engine,
        body=body,
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert "third_party_price" not in result.detectors_fired


def test_join_across_subject_and_body_does_not_fabricate_a_price(engine):
    """Digits must not glue across the subject/body join into a false match."""
    result = classify(
        engine,
        subject="RE: PO 12",
        body="Invoice 450.00 due",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert "third_party_price" not in result.detectors_fired


@pytest.mark.parametrize(
    "body",
    [
        "Their price was 12,450.00",
        "Their bid was 12450 net.",
        "Quote came in at 12,450",
    ],
)
def test_verbatim_competitor_totals_still_fire(engine, body):
    """The fix for false positives must not be satisfied by never matching."""
    result = classify(
        engine,
        body=body,
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert "third_party_price" in result.detectors_fired


def test_contract_prose_raises_to_commercial_confidential(engine):
    result = classify(
        engine,
        body=(
            "6.2 Limitation of Liability. Neither party shall be liable for "
            "indirect or consequential loss arising under this Agreement."
        ),
    )
    assert result.content_class == "commercial_confidential"
    assert "contract_prose" in result.detectors_fired


@pytest.mark.parametrize(
    "text",
    [
        "Delivery is 3.5 Working Days.",
        "Our ref 12.5 kg per pallet.",
        "Lead time 2.5 Weeks Maximum.",
    ],
)
def test_ordinary_quote_text_is_not_contract_prose(engine, text):
    """A decimal in a sentence is not a numbered contract clause.

    This detector firing here would block routine mail to every supplier.
    """
    result = classify(engine, body=text)
    assert "contract_prose" not in result.detectors_fired
    assert result.content_class == "internal"


@pytest.mark.parametrize(
    "text",
    [
        "6.2 Limitation of Liability. Neither party shall be liable for indirect loss.",
        "7.1 Governing Law. This Agreement is governed by English law.",
        "12.4 Payment Terms. Net 30 days from invoice date.",
    ],
)
def test_contract_headings_still_fire_without_the_clause_number_pattern(engine, text):
    """Real clauses must still fire on their headings alone."""
    result = classify(engine, body=text)
    assert "contract_prose" in result.detectors_fired
    assert result.content_class == "commercial_confidential"


@pytest.mark.parametrize(
    "text",
    [
        "3.5 Weeks Delivery Included.",
        "3.2 Revised Quote Attached.",
        "Please see quote below:\n3.5 Weeks Delivery Included.",
    ],
)
def test_numbered_lead_times_are_not_contract_prose(engine, text):
    """A numbered-clause pattern is inherently ambiguous with quantities.

    Precision beats recall here: an unusual clause slipping through means
    routine mail flows; a false positive blocks every supplier.
    """
    result = classify(engine, body=text)
    assert "contract_prose" not in result.detectors_fired
    assert result.content_class == "internal"


def test_internal_staff_contact_raises_to_personal(engine):
    result = classify(
        engine, body="Call Jane on +44 20 7946 0812 or jane.doe@ourcompany.com."
    )
    assert result.content_class == "personal"
    assert "internal_staff_contact" in result.detectors_fired


def test_supplier_own_address_is_not_internal_staff_contact(engine):
    result = classify(engine, body="Reply to sales@supplier-b.com.")
    assert "internal_staff_contact" not in result.detectors_fired


SIGNATURE_BLOCK = "Kind regards,\nJane Doe\nProcurement Manager\njane.doe@ourcompany.com"


def test_own_signature_is_not_internal_staff_contact(engine):
    """A supplier must be able to reply; the sign-off is the point of the message."""
    result = classify(
        engine, body=SIGNATURE_BLOCK, sender="jane.doe@ourcompany.com"
    )
    assert "internal_staff_contact" not in result.detectors_fired
    assert result.content_class == "internal"


def test_a_colleagues_address_in_the_body_is_internal_staff_contact(engine):
    """Excluding the sender must not blind the detector to a third colleague."""
    result = classify(
        engine, body=SIGNATURE_BLOCK, sender="buyer@ourcompany.com"
    )
    assert "internal_staff_contact" in result.detectors_fired
    assert result.content_class == "personal"


def test_attached_source_document_raises_to_commercial_confidential(engine):
    result = classify(
        engine, attachments=[{"filename": "PO-10021.pdf", "is_source_document": True}]
    )
    assert result.content_class == "commercial_confidential"
    assert "source_document_attached" in result.detectors_fired


def test_non_source_attachment_does_not_fire_the_document_detector(engine):
    """An attachment that isn't flagged as a source document is not a leak."""
    result = classify(
        engine,
        attachments=[
            {"filename": "compliance-certificate.pdf", "is_source_document": False}
        ],
    )
    assert "source_document_attached" not in result.detectors_fired
    assert result.content_class == "internal"


def test_highest_firing_detector_wins(engine):
    result = classify(
        engine,
        body="Supplier B quoted 12,450.00. Call jane.doe@ourcompany.com.",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert result.content_class == "personal"
    assert set(result.detectors_fired) >= {"third_party_price", "internal_staff_contact"}


def test_a_disabled_detector_does_not_fire(engine):
    policy = {
        "policyId": "email_sensitivity",
        "policyName": "EmailSensitivityPolicy",
        "details": {
            "policy_identifier": "email_sensitivity",
            "rules": {
                **SENSITIVITY_POLICY["details"]["rules"],
                "detectors": {
                    **SENSITIVITY_POLICY["details"]["rules"]["detectors"],
                    "internal_staff_contact": {
                        "enabled": False,
                        "raises_to": "personal",
                    },
                },
            },
        },
    }
    disabled = FakePolicyEngine({"email_sensitivity": policy})
    result = sens.classify(
        subject="s",
        body="Call jane.doe@ourcompany.com.",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=["ourcompany.com"],
        policy_engine=disabled,
    )
    assert "internal_staff_contact" not in result.detectors_fired


def test_missing_policy_yields_undetermined(engine):
    empty = FakePolicyEngine({})
    result = sens.classify(
        subject="s",
        body="b",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=[],
        policy_engine=empty,
    )
    assert result.content_class == sens.CLASS_UNDETERMINED


def test_a_detector_that_raises_yields_undetermined(engine, monkeypatch):
    """A classifier that cannot decide must never read as 'safe'."""
    monkeypatch.setattr(
        sens,
        "_detect_contract_prose",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = classify(engine, body="anything")
    assert result.content_class == sens.CLASS_UNDETERMINED
    assert (
        sens.clearance_permits(
            result.content_class, "personal", policy_engine=engine
        )
        is False
    )


def test_clearance_comparison(engine):
    assert sens.clearance_permits("internal", "internal", policy_engine=engine) is True
    assert sens.clearance_permits("public", "internal", policy_engine=engine) is True
    assert (
        sens.clearance_permits("commercial_confidential", "internal", policy_engine=engine)
        is False
    )
    assert (
        sens.clearance_permits(
            "commercial_confidential", "commercial_confidential", policy_engine=engine
        )
        is True
    )


def test_undetermined_is_never_permitted(engine):
    assert (
        sens.clearance_permits(sens.CLASS_UNDETERMINED, "personal", policy_engine=engine)
        is False
    )


def test_missing_clearance_uses_the_policy_default(engine):
    assert sens.clearance_permits("internal", None, policy_engine=engine) is True
    assert (
        sens.clearance_permits("commercial_confidential", None, policy_engine=engine)
        is False
    )


def test_the_live_email_sensitivity_policy_gates_both_directions():
    """A fixture richer than production data is how a dead guard looks alive.

    This reads the real ``email_sensitivity`` row via PolicyEngine instead of
    the fixture above, and checks both directions: internal content must be
    permitted to an internal-clearance supplier (every supplier today), and
    commercial_confidential content must not be.
    """
    import os

    import psycopg2
    from dotenv import load_dotenv

    from src.engines.policy_engine import PolicyEngine

    load_dotenv()

    def factory():
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10,
        )

    engine = PolicyEngine(connection_factory=factory)

    assert sens.clearance_permits("internal", "internal", policy_engine=engine) is True
    assert (
        sens.clearance_permits(
            "commercial_confidential", "internal", policy_engine=engine
        )
        is False
    )
