"""Propose-first requirement scope: intent detection, commodity families, skeleton.

These are the deterministic halves of the fix for "the requirements agent just asks
questions instead of giving me a scope". The LLM tailors the wording; this module
guarantees that a scope EXISTS to tailor — so a buyer who asks for a scope always
gets one, even if the model is slow, terse or unreachable.
"""
from src.services import requirement_scope as rs


# --------------------------------------------------------------------------
# Intent: is the buyer ASKING for a scope, or ANSWERING a question?
# --------------------------------------------------------------------------

def test_asking_for_a_list_of_requirements_is_a_scope_request():
    # The two messages from the reported issue (UI Improvements/requirements agent_issue1.pdf)
    assert rs.wants_scope("tell me the requirements I should have for a managed cloud platform")
    assert rs.wants_scope("provide a list of requirements")


def test_other_phrasings_of_the_same_ask():
    for text in (
        "What should the scope of requirements be for this?",
        "Give me a scope for a managed service",
        "draft the requirements please",
        "suggest the specifications I need",
        "what requirements do I need for office chairs",
        "recommend a checklist of must-haves",
        "you tell me — I don't know what to ask for",
        "stop asking questions and just give me the scope",
        "skip the questions",
    ):
        assert rs.wants_scope(text), text


def test_answering_a_question_is_not_a_scope_request():
    # Regression guard: these contain "scope"/"requirements"/"list" but are ANSWERS.
    # Treating them as scope requests would replace elicitation with advice and
    # silently drop the buyer's stated facts.
    for text in (
        "Migration is in scope; bespoke dashboards stay in-house, so out of scope. "
        "Training is in scope for admins only.",
        "500 users at peak, growing 20% a year",
        "The requirements are already agreed with the business",
        "London HQ, by 15 August",
        "our data must stay in the UK",
    ):
        assert not rs.wants_scope(text), text


def test_blank_message_is_not_a_scope_request():
    assert not rs.wants_scope("")
    assert not rs.wants_scope(None)


# --------------------------------------------------------------------------
# Acceptance: "yes, use that" adopts a pending proposal
# --------------------------------------------------------------------------

def test_acceptance_phrases():
    for text in ("yes", "looks good", "that works, use it", "approve", "accept that scope"):
        assert rs.is_acceptance(text), text


def test_non_acceptance():
    for text in ("no", "not quite — drop the training item", "what about security?"):
        assert not rs.is_acceptance(text), text


# --------------------------------------------------------------------------
# Commodity family classification
# --------------------------------------------------------------------------

def test_family_from_category_and_title():
    assert rs.classify_family("Managed cloud data platform", "SaaS / IT") == rs.FAMILY_SAAS_IT
    assert rs.classify_family("Warehouse robotics line", "Works & construction") == rs.FAMILY_WORKS
    assert rs.classify_family("Office supplies framework", "Office & facilities") == rs.FAMILY_GOODS
    assert rs.classify_family("Cleaning services", "Facilities management") == rs.FAMILY_SERVICES


def test_family_defaults_to_services_when_unknown():
    assert rs.classify_family("", "") == rs.FAMILY_SERVICES


# --------------------------------------------------------------------------
# The skeleton: always a usable scope, never empty
# --------------------------------------------------------------------------

def test_every_family_has_a_substantive_skeleton():
    for family in (rs.FAMILY_SAAS_IT, rs.FAMILY_SERVICES, rs.FAMILY_GOODS, rs.FAMILY_WORKS):
        areas = rs.scope_skeleton(family, {})
        assert len(areas) >= 8, family
        for area in areas:
            assert area["area"] and area["requirement"] and area["why"]
            # Untailored items must declare themselves as templates, so nothing
            # generic is ever presented as if the buyer had stated it.
            assert area["source"] == "template"


def test_skeleton_areas_cover_the_commodity_specifics():
    saas = {a["area"].lower() for a in rs.scope_skeleton(rs.FAMILY_SAAS_IT, {})}
    assert any("residency" in a for a in saas)
    assert any("licen" in a for a in saas)
    works = {a["area"].lower() for a in rs.scope_skeleton(rs.FAMILY_WORKS, {})}
    assert any("cdm" in a or "health" in a for a in works)


def test_skeleton_uses_known_requirement_facts_and_never_invents_them():
    areas = rs.scope_skeleton(
        rs.FAMILY_SAAS_IT,
        {"title": "Managed cloud data platform", "needed_by_date": "2026-10-01"},
    )
    blob = " ".join(a["requirement"] for a in areas)
    assert "Managed cloud data platform" in blob
    # quantity/budget were NOT supplied, so no number may appear as if it were.
    assert "£" not in blob


def test_open_points_flag_what_only_the_buyer_can_answer():
    areas = rs.scope_skeleton(rs.FAMILY_SAAS_IT, {})
    points = rs.open_points(areas)
    assert points, "a template scope must declare what it still needs confirmed"
    assert all(isinstance(p, str) and p for p in points)


# --------------------------------------------------------------------------
# Tailoring merge: LLM wording wins, template fills every gap
# --------------------------------------------------------------------------

def test_tailored_wording_replaces_the_template_and_is_labelled():
    skeleton = rs.scope_skeleton(rs.FAMILY_SAAS_IT, {})
    target = skeleton[0]["area"]
    merged = rs.merge_tailored(
        skeleton, [{"area": target, "requirement": "Supplier consolidates three tools onto one tenant."}]
    )
    hit = next(a for a in merged if a["area"] == target)
    assert hit["requirement"] == "Supplier consolidates three tools onto one tenant."
    assert hit["source"] == "tailored"
    # Every other area survives with its template text — no silent truncation.
    assert len(merged) == len(skeleton)
    assert all(a["requirement"] for a in merged)


def test_merge_tolerates_garbage_from_the_model():
    skeleton = rs.scope_skeleton(rs.FAMILY_GOODS, {})
    for junk in (None, [], "nonsense", [{"nope": 1}], [{"area": "Unknown area", "requirement": "x"}]):
        merged = rs.merge_tailored(skeleton, junk)
        assert len(merged) >= len(skeleton)
        assert all(a["requirement"] for a in merged)


def test_merge_keeps_extra_model_areas_but_labels_them():
    skeleton = rs.scope_skeleton(rs.FAMILY_SAAS_IT, {})
    merged = rs.merge_tailored(
        skeleton, [{"area": "AI model governance", "requirement": "Supplier discloses model provenance."}]
    )
    extra = next(a for a in merged if a["area"] == "AI model governance")
    assert extra["source"] == "tailored"


def test_confirm_question_is_one_question_and_asks_for_confirmation():
    q = rs.confirm_question(rs.scope_skeleton(rs.FAMILY_SERVICES, {}))
    assert q.count("?") == 1
    assert any(w in q.lower() for w in ("confirm", "adjust", "change", "right"))
