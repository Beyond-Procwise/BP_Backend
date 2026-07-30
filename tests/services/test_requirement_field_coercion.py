"""Typed field values must be validated before they reach proc.bp_requirement.

Found live: asked for a scope for a 3-year platform, the model returned
``needed_by_date = "3 years from contract start"``. bp_requirement.needed_by_date
is a DATE, so the insert raised InvalidDatetimeFormat and the WHOLE turn failed —
the buyer's scope was lost to one unparseable field.

The rule is reject, never repair: "3 years from contract start" is not a date, and
inventing one would be fabrication. The field stays missing, so the agent asks for
it — which is the correct behaviour for something the buyer never actually stated.
"""
from datetime import date, datetime
from decimal import Decimal

from src.services import requirement_service as svc


def test_iso_date_passes_through():
    clean, rejected = svc.coerce_updates({"needed_by_date": "2026-10-01"})
    assert clean["needed_by_date"] == "2026-10-01"
    assert rejected == []


def test_common_written_date_formats_are_accepted():
    for raw, expected in (
        ("01/10/2026", "2026-10-01"),          # UK order — the corpus is UK/EU
        ("1 October 2026", "2026-10-01"),
        ("15th August 2026", "2026-08-15"),    # ordinal suffix
        ("Oct 1 2026", "2026-10-01"),
        ("2026/10/01", "2026-10-01"),
    ):
        clean, rejected = svc.coerce_updates({"needed_by_date": raw})
        assert clean.get("needed_by_date") == expected, raw
        assert rejected == []


def test_date_objects_are_normalised():
    clean, _ = svc.coerce_updates({"needed_by_date": date(2026, 10, 1)})
    assert clean["needed_by_date"] == "2026-10-01"
    clean, _ = svc.coerce_updates({"needed_by_date": datetime(2026, 10, 1, 9, 30)})
    assert clean["needed_by_date"] == "2026-10-01"


def test_prose_dates_are_rejected_not_guessed():
    for raw in ("3 years from contract start", "ASAP", "TBC", "end of Q3",
                "as soon as possible", "next year", "3"):
        clean, rejected = svc.coerce_updates({"needed_by_date": raw})
        assert "needed_by_date" not in clean, raw
        assert rejected and rejected[0]["field"] == "needed_by_date"
        assert rejected[0]["value"] == raw


def test_numeric_fields_accept_numbers_and_formatted_money():
    clean, rejected = svc.coerce_updates(
        {"quantity": 1200, "target_budget": "£480,000"})
    assert clean["quantity"] == 1200
    assert float(clean["target_budget"]) == 480000.0
    assert rejected == []
    clean, _ = svc.coerce_updates({"quantity": "1,200", "target_budget": Decimal("99.50")})
    assert clean["quantity"] == 1200
    assert float(clean["target_budget"]) == 99.5


def test_numeric_fields_reject_prose_rather_than_extracting_a_number():
    # "3 years from contract start" must NOT become quantity 3.
    for field, raw in (("quantity", "10 units"), ("quantity", "3 years from contract start"),
                       ("quantity", "approx 500"), ("target_budget", "mid six figures"),
                       ("quantity", "")):
        clean, rejected = svc.coerce_updates({field: raw})
        assert field not in clean, (field, raw)


def test_free_text_fields_are_untouched():
    updates = {"title": "Managed cloud data platform", "unit": "licence",
               "delivery_location": "Group · UK", "currency": "GBP", "priority": "high"}
    clean, rejected = svc.coerce_updates(updates)
    assert clean == updates
    assert rejected == []


def test_one_bad_field_does_not_discard_the_good_ones():
    clean, rejected = svc.coerce_updates({
        "title": "Managed cloud data platform",
        "quantity": 1200,
        "needed_by_date": "3 years from contract start",
    })
    assert clean["title"] == "Managed cloud data platform"
    assert clean["quantity"] == 1200
    assert "needed_by_date" not in clean
    assert len(rejected) == 1


def test_only_elicitable_fields_may_be_set_by_the_model():
    """Found live: after the buyer accepted an 18-area proposed scope, the next
    elicitation call returned ``updates.specifications`` with 2 areas of its own —
    and it was applied, wiping the adopted scope and its provenance. The prompt
    named the allowed fields; nothing enforced them. Now the code does."""
    clean, rejected = svc.coerce_updates({
        "title": "Cloud platform",
        "specifications": {"scope_areas": [{"area": "Support", "requirement": "24/7"}]},
        "constraints": {"anything": 1},
        "status": "complete",
        "requirement_id": "REQ-hijack",
        "completeness_score": 1.0,
        "_seed_context": {"proposed_scope": {}},
    })
    assert clean == {"title": "Cloud platform"}
    assert {r["field"] for r in rejected} == {
        "specifications", "constraints", "status", "requirement_id",
        "completeness_score", "_seed_context",
    }
    assert all(r["reason"] == "not an elicitable field" for r in rejected)


def test_every_elicitable_field_is_actually_accepted():
    # Guard against the allow-list drifting from the prompt's "Allowed fields".
    values = {"title": "t", "category": "c", "description": "d", "quantity": 1,
              "unit": "ea", "target_budget": 10, "currency": "GBP",
              "needed_by_date": "2026-10-01", "delivery_location": "London",
              "priority": "high"}
    assert set(values) == set(svc.ELICITABLE_FIELDS)
    clean, rejected = svc.coerce_updates(values)
    assert set(clean) == set(values)
    assert rejected == []


class TestGrounding:
    """Values for hard facts must trace back to something the buyer actually wrote.

    Found live: given a robotics brief with no address and no date, the model
    returned delivery_location "Main Distribution Center, London, UK" and
    needed_by_date 2025-12-31. Both were accepted, the requirement scored
    complete, and it handed off to sourcing with an invented delivery address.

    Free text (title, description) is NOT guarded — paraphrasing a brief into a
    title is legitimate. Dates, places and numbers are guarded, because those are
    the ones that get acted on.
    """

    BRIEF = ("Warehouse robotics line. Manual pick-pack limiting throughput at peak. "
             "+35% throughput; payback under 3 years. Estimated value GBP 610,000 over 5 years.")

    def test_invented_location_and_date_are_rejected(self):
        clean, rejected = svc.coerce_updates({
            "title": "Warehouse Robotics Line Upgrade",
            "target_budget": 610000,
            "delivery_location": "Main Distribution Center, London, UK",
            "needed_by_date": "2025-12-31",
        }, source_text=self.BRIEF)
        assert clean["title"] == "Warehouse Robotics Line Upgrade"   # paraphrase allowed
        assert clean["target_budget"] == 610000                      # digits are in the brief
        assert "delivery_location" not in clean
        assert "needed_by_date" not in clean
        reasons = {r["field"]: r["reason"] for r in rejected}
        assert reasons["delivery_location"] == "not grounded in the buyer's message"
        assert reasons["needed_by_date"] == "not grounded in the buyer's message"

    def test_stated_location_and_date_survive_normalisation(self):
        src = "10 laptops to London HQ by July 1, budget £4,500"
        clean, rejected = svc.coerce_updates(
            {"delivery_location": "London HQ", "needed_by_date": "2026-07-01",
             "quantity": 10, "target_budget": "£4,500"},
            source_text=src,
        )
        assert clean["delivery_location"] == "London HQ"
        assert clean["needed_by_date"] == "2026-07-01"   # month named in the text
        assert clean["quantity"] == 10
        assert float(clean["target_budget"]) == 4500.0
        assert rejected == []

    def test_numeric_dates_count_as_grounding(self):
        for src in ("we need it live by 01/10/2026", "go live 01/10", "by 01.10.2026"):
            clean, _ = svc.coerce_updates({"needed_by_date": "2026-10-01"}, source_text=src)
            assert clean.get("needed_by_date") == "2026-10-01", src

    def test_a_decimal_number_is_not_a_date(self):
        """Found in the browser: the brief said "99.95% SLA" and the model returned
        needed_by_date 2025-03-31. "99.95" matched a d/m date pattern, so an invented
        date passed the guard and scored the requirement complete."""
        src = ("Managed cloud data platform. Consolidate to one platform; "
               "at least 15% unit-rate reduction; 99.95% SLA. Term: 3 years + 2 optional.")
        clean, rejected = svc.coerce_updates({"needed_by_date": "2025-03-31"}, source_text=src)
        assert "needed_by_date" not in clean
        assert rejected[0]["reason"] == "not grounded in the buyer's message"

    def test_a_numeric_range_is_not_a_date(self):
        clean, _ = svc.coerce_updates({"needed_by_date": "2026-05-03"},
                                      source_text="we want 3-5 years of cover")
        assert "needed_by_date" not in clean

    def test_the_word_may_alone_does_not_ground_a_date(self):
        # "may" is a month AND an everyday verb; only a month beside a number counts.
        clean, _ = svc.coerce_updates({"needed_by_date": "2026-05-01"},
                                      source_text="we may need extra licences later")
        assert "needed_by_date" not in clean
        clean, _ = svc.coerce_updates({"needed_by_date": "2026-05-01"},
                                      source_text="live by 1 May")
        assert clean.get("needed_by_date") == "2026-05-01"

    def test_invented_quantity_is_rejected(self):
        clean, rejected = svc.coerce_updates({"quantity": 250},
                                             source_text="we need laptops for the new floor")
        assert "quantity" not in clean
        assert rejected[0]["reason"] == "not grounded in the buyer's message"

    def test_no_source_text_means_no_grounding_check(self):
        # Callers that have no buyer text (e.g. a reload path) keep the old
        # behaviour rather than losing every typed field.
        clean, rejected = svc.coerce_updates(
            {"needed_by_date": "2026-10-01", "delivery_location": "London HQ"})
        assert clean["needed_by_date"] == "2026-10-01"
        assert clean["delivery_location"] == "London HQ"
        assert rejected == []


def test_non_dict_input_is_safe():
    assert svc.coerce_updates(None) == ({}, [])
    assert svc.coerce_updates("nonsense") == ({}, [])
