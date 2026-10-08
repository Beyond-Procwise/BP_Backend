"""Only some families are things a free-text request can be classified into.

`rfq_batch` and `human_written` are assured paths, not kinds of request: the classifier must never offer them, and a
model that names one is treated like a model that invents a family.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.draft_assurance import family as fam, run as R, stages
from tests.services.test_draft_run import Conn, env
from tests.services.test_draft_assurance import TABLES

SQL = Path(__file__).resolve().parents[2] / "deploy/sql"


def rules_of(name):
    return json.loads((SQL / name).read_text().split("$json$")[1])["rules"]


def engine(rules_by_family):
    listing = [{"policy_type": "email_family", "policy_desc": f"{k} desc", "details": {"rules": {"family_id": k, **v}}}
               for k, v in rules_by_family.items()]
    return SimpleNamespace(list_policies=lambda: listing, get_policy=lambda slug: None)


def test_a_family_that_opts_out_is_not_listed_and_the_default_is_listed():
    got = fam.list_families(engine({"a": {}, "b": {"classifiable": True}, "c": {"classifiable": False}}))
    assert set(got) == {"a", "b"}


def test_the_shipped_rows_say_which_families_are_requests():
    assert rules_of("2026-10-08_email_family_rfq_batch.sql")["classifiable"] is False
    assert rules_of("2026-10-08_email_family_human_written.sql")["classifiable"] is False
    for name in ("2026-10-07_email_family_negotiation_counter.sql", "2026-10-07_email_family_free_prompt.sql"):
        assert rules_of(name).get("classifiable", True) is True


@pytest.mark.parametrize("bad", ["no", 0, None, []])
def test_the_flag_must_be_a_true_or_false(bad):
    rules = {**rules_of("2026-10-07_email_family_free_prompt.sql"), "classifiable": bad}
    with pytest.raises(fam.FamilyConfigUnavailable):
        fam.parse_family(rules)


def test_a_model_that_names_a_family_that_is_not_classifiable_gets_the_fallback():
    listing = fam.list_families(engine({"free_prompt": {}, "negotiation_counter": {}, "rfq_batch": {"classifiable": False}}))
    shown = {}

    def ask(system, user):
        shown["system"] = system
        return json.dumps({"family_id": "rfq_batch", "confidence": 0.95, "candidates": [{"family_id": "rfq_batch", "confidence": 0.95}],
                           "lookup_keys": {}, "user_instruction": "send an RFQ"})

    res = stages.classify_request(ask, "C {families}", "send an RFQ to all suppliers", listing)
    assert res["status"] == "invalid" and "not a configured family" in res["reason"]
    assert "rfq_batch" not in shown["system"]                  # it was never offered either


# --- what the classifier is told about each family -------------------------------------------------------------------

def test_the_classifier_is_shown_the_request_description_when_a_family_has_one():
    got = fam.list_families(engine({"a": {"request_description": "Use this when the person wants X."}, "b": {}}))
    assert got["a"] == "Use this when the person wants X." and got["b"] == "b desc"      # no description: the config text, as before


@pytest.mark.parametrize("bad", ["", "  ", 5, None])
def test_a_request_description_must_be_real_text(bad):
    rules = {**rules_of("2026-10-07_email_family_free_prompt.sql"), "request_description": bad}
    with pytest.raises(fam.FamilyConfigUnavailable):
        fam.parse_family(rules)


def test_every_classifiable_shipped_family_explains_when_to_choose_it():
    for name in ("2026-10-07_email_family_negotiation_counter.sql", "2026-10-07_email_family_free_prompt.sql"):
        text = rules_of(name)["request_description"]
        assert len(text.split()) >= 10 and "guardrail" not in text.lower()          # about the REQUEST, not about our checks


# --- the short name used in the question put to the person ------------------------------------------------------------

def test_the_question_uses_the_families_short_label_not_its_long_description():
    listing = {"negotiation_counter": "A request to counter, negotiate or push back on a price. Long text here.",
               "free_prompt": "Any other supplier correspondence."}
    labels = {"negotiation_counter": "a counter-offer", "free_prompt": "an ordinary supplier message"}

    def ask(system, user):
        return json.dumps({"family_id": "negotiation_counter", "confidence": 0.4, "lookup_keys": {}, "user_instruction": user,
                           "candidates": [{"family_id": "negotiation_counter", "confidence": 0.4}, {"family_id": "free_prompt", "confidence": 0.35}]})

    q = stages.classify_request(ask, "C {families}", "get back to acme", listing, labels=labels)["clarification"]["question"]
    assert q == "Is this a counter-offer, or an ordinary supplier message?"
    q = stages.classify_request(ask, "C {families}", "get back to acme", listing)["clarification"]["question"]    # no labels: as before
    assert q == "Is this A request to counter, negotiate or push back on a price, or Any other supplier correspondence?"


def test_list_labels_reads_the_short_label_and_skips_what_is_not_classifiable():
    got = fam.list_labels(engine({"a": {"request_label": "an A"}, "b": {}, "c": {"classifiable": False, "request_label": "a C"}}))
    assert got == {"a": "an A"}                    # b has no label (the caller falls back to the description); c is not offered


@pytest.mark.parametrize("bad", ["", " ", 3, None])
def test_a_request_label_must_be_real_text(bad):
    with pytest.raises(fam.FamilyConfigUnavailable):
        fam.parse_family({**rules_of("2026-10-07_email_family_free_prompt.sql"), "request_label": bad})


def test_the_shipped_classifiable_families_have_short_labels():
    for name in ("2026-10-07_email_family_negotiation_counter.sql", "2026-10-07_email_family_free_prompt.sql"):
        label = rules_of(name)["request_label"]
        assert 1 <= len(label.split()) <= 4


def test_the_run_puts_the_short_label_in_the_question(monkeypatch):
    from tests.services.test_draft_run import _policies
    eng = _policies()
    listing = [{"policy_type": "email_family", "policy_desc": "Long config description. More.", "details": {"rules": {
        "family_id": "free_prompt", "request_label": "an ordinary supplier message"}}},
        {"policy_type": "email_family", "policy_desc": "Other long description.", "details": {"rules": {
        "family_id": "negotiation_counter", "request_label": "a counter-offer"}}}]
    eng.list_policies = lambda: listing
    low = json.dumps({"family_id": "free_prompt", "confidence": 0.4, "lookup_keys": {}, "user_instruction": "x",
                      "candidates": [{"family_id": "free_prompt", "confidence": 0.4}, {"family_id": "negotiation_counter", "confidence": 0.38}]})
    run = R.begin(env(ask=lambda s, u: low, engine=eng), {"supplier_id": "S-1", "workflow_id": "wf-1"}, slug=None, workflow_id="wf-1",
                  request="get back to acme", classify=True)
    assert run.classification["clarification"]["question"] == "Is this an ordinary supplier message, or a counter-offer?"
