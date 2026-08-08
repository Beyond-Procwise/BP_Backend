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
    "raw_row": {"version": 1},
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
    ],
)
def test_unrelated_digit_runs_do_not_fire_third_party_price(engine, body):
    """A digit substring of a PO number or part number is not a price.

    Flattening the whole message into one digit string and substring-matching
    destroys every number boundary -- a reference code can contain the same
    digits as an unrelated peer amount by pure coincidence. With a peer
    amount of 12450.00, the old substring match fired on these; the new
    boundary-tokenised comparison does not.
    """
    result = classify(
        engine,
        body=body,
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
    )
    assert "third_party_price" not in result.detectors_fired


@pytest.mark.parametrize(
    "body, peer_amount",
    [
        # The needle (12450.00, 7 digits) is longer than any digit run in
        # these two bodies, so under the *old* substring match a shorter
        # peer amount is required to actually distinguish old from new
        # behaviour -- these two peer amounts are the ones the old code
        # would have matched by gluing unrelated digits together.
        ("We need 5000 units by Friday.", "500.00"),
        ("We need 12 pallets, 450 units total", "124.50"),
    ],
)
def test_unrelated_quantities_do_not_fire_third_party_price(engine, body, peer_amount):
    """A quantity is not a price, even when its digits could be glued into one."""
    result = classify(
        engine,
        body=body,
        peer_prices=[{"supplier_id": "SUP-2", "amount": peer_amount}],
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
        # This first case fires under both the old and the new code, so it is
        # a sanity check rather than a regression guard. "12450 net." and
        # "12,450" are the load-bearing cases: a bare integer and a
        # thousands-separated figure with no decimal, both of which the
        # tokeniser must still recognise as the same value as "12450.00".
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


def test_a_peer_amount_below_the_floor_does_not_fire_but_is_recorded(engine):
    """A sub-100 competitor price is a deliberate, recorded blind spot.

    Two-digit figures collide with quantities and dates too often to carry
    signal, so they are not detected -- but that must be visible to an
    auditor, not silent.
    """
    result = classify(
        engine,
        body="Their price was 85.00 for the same line.",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "85.00"}],
    )
    assert "third_party_price" not in result.detectors_fired
    assert result.evidence["third_party_price_skipped_below_floor"] == ["85.00"]


@pytest.mark.parametrize(
    "body",
    ["Their bid was 12450.5 net.", "Their bid was 12450.50 net."],
)
def test_one_and_two_decimal_amounts_both_fire(engine, body):
    """A one-decimal-place figure must resolve to the same value as two."""
    result = classify(
        engine,
        body=body,
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.5"}],
    )
    assert "third_party_price" in result.detectors_fired


def test_small_decimal_in_prose_is_blocked_by_the_floor_not_the_tokeniser(engine):
    """"12.5 kg" must not misfire against a peer amount of "12.50".

    The tokeniser now recognises one-decimal-place numbers, so without the
    floor this would fire. Confirms the floor -- not accidental non-matching
    -- is what keeps this quiet.
    """
    result = classify(
        engine,
        body="Our ref 12.5 kg per pallet.",
        peer_prices=[{"supplier_id": "SUP-2", "amount": "12.50"}],
    )
    assert "third_party_price" not in result.detectors_fired
    assert result.evidence["third_party_price_skipped_below_floor"] == ["12.50"]


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


@pytest.mark.parametrize(
    "sender",
    [
        "jane.doe@ourcompany.com",
        "Jane Doe <jane.doe@ourcompany.com>",
        "  JANE.DOE@OURCOMPANY.COM  ",
        '"Doe, Jane" <jane.doe@ourcompany.com>',
    ],
)
def test_the_senders_own_signature_is_not_a_leak_in_any_address_form(engine, sender):
    """A From: header arrives in several shapes; all name the same person.

    Comparing the raw header string against a bare address would only match
    the plain form and silently re-open the signature-block false positive
    for every "Name <addr>" sender, which is how a From: header usually
    looks in practice.
    """
    result = classify(engine, body=SIGNATURE_BLOCK, sender=sender)
    assert "internal_staff_contact" not in result.detectors_fired


def test_a_colleagues_address_in_the_body_is_internal_staff_contact(engine):
    """Documents intended behaviour: excluding the sender must not blind the
    detector to a third colleague's address appearing in the body.

    This does not by itself guard the sender-exclusion fix -- the old code
    (which fired on any internal address, sender or not) passes this case
    identically. The guarding tests are the sender-address-form pair above
    and ``test_own_signature_is_not_internal_staff_contact``.
    """
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


@pytest.mark.parametrize("bad_raises_to", ["Commercial_Confidential", "confidential", "", "  "])
def test_a_typo_d_raises_to_denies_rather_than_silently_scoring_zero(bad_raises_to):
    """A raises_to that does not resolve to a declared class used to score 0
    in the rank comparison, which never beats "internal"'s rank of 2 -- so
    the detector fired, was recorded in detectors_fired, and the content
    was sent anyway. This is the same failure class as guardrail.py's
    required_role typo: an unresolvable target class must deny, not
    silently lose the comparison."""
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
                        "enabled": True,
                        "raises_to": bad_raises_to,
                    },
                },
            },
        },
    }
    typo_engine = FakePolicyEngine({"email_sensitivity": policy})
    result = sens.classify(
        subject="s",
        body="Call jane.doe@ourcompany.com.",
        attachments=None,
        recipient_supplier_id="SUP-1",
        peer_prices=None,
        internal_domains=["ourcompany.com"],
        policy_engine=typo_engine,
    )
    assert result.content_class == sens.CLASS_UNDETERMINED
    assert "internal_staff_contact" in result.detectors_fired
    assert sens.clearance_permits(
        result.content_class, "personal", policy_engine=typo_engine
    ) is False


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


def test_no_engine_supplied_reuses_rbacs_shared_cache_not_a_fresh_one(monkeypatch):
    """Before this fix, classify(policy_engine=None) built a brand-new
    PolicyEngine on every call -- two full bp_policy reads per send (one
    here, one in rbac). It must instead resolve rbac's own TTL-cached
    engine, the same way guardrail.authorize and email_dispatch_guard do."""
    from src.services import rbac

    builds = []

    class Counting:
        def __init__(self):
            builds.append(1)

        def get_policy(self, slug):
            return None

    monkeypatch.setattr(rbac, "_build_engine", lambda: Counting(), raising=False)
    rbac.reset_policy_cache()

    sens.classify(
        subject="s", body="b", attachments=None, recipient_supplier_id="SUP-1",
        peer_prices=None, internal_domains=[],
    )
    sens.classify(
        subject="s", body="b", attachments=None, recipient_supplier_id="SUP-1",
        peer_prices=None, internal_domains=[],
    )

    assert len(builds) == 1, "each call built its own engine instead of reusing rbac's"
    rbac.reset_policy_cache()
