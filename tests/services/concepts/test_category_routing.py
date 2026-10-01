"""What the uploader typed, turned into a physical pipeline — or refused.

Review Focus 1 and 2 live here: every spelling the old four-entry map accepted
must still route, and an empty category must refuse rather than default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import routing as R  # noqa: E402
from src.services.concepts.vocabulary import SEED_VOCABULARY, build_vocabulary  # noqa: E402
from utils.procurement_schema import CATEGORY_TO_DOC_TYPE  # noqa: E402

V = SEED_VOCABULARY


@pytest.mark.parametrize("category,expected", [
    ("invoice", "invoice"),
    ("Invoice", "invoice"),
    ("purchase_order", "purchase_order"),
    ("PurchaseOrder", "purchase_order"),
    ("po", "purchase_order"),
    ("PO", "purchase_order"),
    ("quote", "quote"),
    ("Quote", "quote"),
    ("contract", "contract"),
    ("Contract", "contract"),
])
def test_every_spelling_the_old_map_accepted_still_routes(category, expected):
    """Review Focus 1. Losing one of these breaks a working upload path."""
    pipeline, concept = R.pipeline_for_category(category, vocabulary=V)
    assert pipeline == expected
    assert concept.startswith("doctype.")


@pytest.mark.parametrize("category", ["", "   ", None])
def test_an_empty_category_refuses_rather_than_defaulting(category):
    """Review Focus 2. Guessing a type for a document nobody labelled is
    exactly the forcing the spec's principle 4 forbids."""
    # match= pins the DELIBERATE guard: without it the alias lookup would also
    # raise (nothing claims ""), and removing the guard would stay green.
    with pytest.raises(R.UnknownDocumentCategory, match="no document category was supplied"):
        R.pipeline_for_category(category, vocabulary=V)


def test_an_unrecognised_category_refuses_and_says_what_it_tried():
    with pytest.raises(R.UnknownDocumentCategory) as exc:
        R.pipeline_for_category("bill of lading", vocabulary=V)
    assert "bill of lading" in str(exc.value)


def test_a_new_type_routes_as_soon_as_it_is_in_the_vocabulary():
    """The point of the exercise: adding a type is a row, not a code edit."""
    pipeline, concept = R.pipeline_for_category("framework agreement", vocabulary=V)
    assert pipeline == "contract"
    assert concept == "doctype.framework_agreement"


def test_a_recognised_type_with_no_pipeline_refuses():
    """(The brief probed the bare word "notice", which is NOT an alias and so
    only proves the unknown path; "general notice" is the real alias.)
    doctype.notice_general is a real concept that nothing can ingest yet.
    Routing it at the contract tables because 'contract' is the closest match
    is the forcing principle 4 forbids."""
    with pytest.raises(R.UnknownDocumentCategory) as exc:
        R.pipeline_for_category("general notice", vocabulary=V)
    assert "no pipeline" in str(exc.value).lower()
    assert "doctype.notice_general" in str(exc.value)


def test_an_ambiguous_category_refuses_rather_than_choosing():
    """Review Focus 3. If two types claim the alias, routing to either one is
    a coin toss recorded as a fact."""
    colliding = build_vocabulary(
        [
            {"concept_code": "role.master", "domain": "RELATIONSHIP_ROLE",
             "definition": "x", "not_to_be_confused_with": [], "status": "active",
             "rejection_reason": None},
            {"concept_code": "doctype.a", "domain": "DOCUMENT_TYPE",
             "definition": "x", "not_to_be_confused_with": [], "status": "active",
             "rejection_reason": None},
            {"concept_code": "doctype.b", "domain": "DOCUMENT_TYPE",
             "definition": "x", "not_to_be_confused_with": [], "status": "active",
             "rejection_reason": None},
        ],
        [
            {"concept_code": "doctype.a", "role": "role.master",
             "default_parent_type": None, "execution_mode": None,
             "aliases": ["order form"], "identifiers": [],
             "structural_signals": [], "pipeline_doc_type": "contract",
             "status": "active"},
            {"concept_code": "doctype.b", "role": "role.master",
             "default_parent_type": None, "execution_mode": None,
             "aliases": ["order form"], "identifiers": [],
             "structural_signals": [], "pipeline_doc_type": "purchase_order",
             "status": "active"},
        ],
        source="test",
    )
    with pytest.raises(R.AmbiguousDocumentCategory) as exc:
        R.pipeline_for_category("order form", vocabulary=colliding)
    assert "doctype.a" in str(exc.value) and "doctype.b" in str(exc.value)


@pytest.mark.parametrize("key,physical", sorted(CATEGORY_TO_DOC_TYPE.items()))
def test_every_legacy_category_routes_to_the_same_pipeline_as_before(key, physical):
    """Stronger than the single-owner guard, which only proves each key resolves
    to SOME concept. An old spelling that now reaches a different physical
    pipeline is data corruption, not a classification nicety."""
    pipeline, _ = R.pipeline_for_category(key, vocabulary=V)
    assert pipeline == physical.lower()


@pytest.mark.parametrize("category,expected", [
    ("Invoice", "invoice"), ("PurchaseOrder", "purchase_order"),
    ("PO", "purchase_order"), ("Quote", "quote"), ("Contract", "contract"),
])
def test_old_hardcoded_map_spellings_outside_the_legacy_map_still_route(category, expected):
    pipeline, _ = R.pipeline_for_category(category, vocabulary=V)
    assert pipeline == expected


def test_the_collision_raise_is_not_the_unknown_raise():
    """A two-owner tuple is truthy; the gate must not take [0] nor report it as
    merely unknown."""
    rows = [
        {"concept_code": f"doctype.{n}", "role": "role.master",
         "default_parent_type": None, "execution_mode": None,
         "aliases": ["order form"], "identifiers": [], "structural_signals": [],
         "pipeline_doc_type": p, "status": "active"}
        for n, p in (("a", "contract"), ("b", "purchase_order"))
    ]
    v = build_vocabulary([], rows, source="test")
    assert len(R.resolve_alias("order form", v)) == 2
    with pytest.raises(R.AmbiguousDocumentCategory):
        R.pipeline_for_category("order form", vocabulary=v)
    assert not issubclass(R.AmbiguousDocumentCategory, R.UnknownDocumentCategory)


def test_a_proposed_type_never_routes():
    rows = [{"concept_code": "doctype.draftthing", "role": "role.master",
             "default_parent_type": None, "execution_mode": None,
             "aliases": ["draft thing"], "identifiers": [], "structural_signals": [],
             "pipeline_doc_type": "contract", "status": "proposed"}]
    v = build_vocabulary([], rows, source="test")
    with pytest.raises(R.UnknownDocumentCategory):
        R.pipeline_for_category("draft thing", vocabulary=v)
