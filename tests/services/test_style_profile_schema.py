"""The profile schema is the first line of defence for invariant 1.

A profile describes writing habits. It must not be able to carry the writing itself.
These tests are mostly about what the schema *refuses*.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.style.profile import StyleProfile, parse_profile


def valid_payload(**overrides) -> dict:
    payload = {
        "structural": {
            "subject_pattern": "topic — reference",
            "greeting": "Hi {first_name},",
            "opening_move": "context_before_ask",
            "body_form": "short_prose",
            "target_words": [60, 110],
            "sign_off": "Nick",
            "signature_block": False,
        },
        "register": {
            "formality": 3,
            "directness": 4,
            "hedging": "low",
            "contractions": True,
            "person": "first_singular",
        },
        "lexical": {
            "preferred_terms": ["unit rate", "volume tiers"],
            "banned_phrases": ["I hope this email finds you well"],
            "number_format": "bare_numerals",
            "date_format": "weekday_name",
        },
        "behavioural": {
            "cta_form": "proposes_specific_time",
            "deadline_phrasing": "soft_by_date",
            "escalation_ladder": ["neutral", "firm", "formal"],
        },
    }
    for section, changes in overrides.items():
        payload[section].update(changes)
    return payload


def test_the_specifications_own_example_validates():
    profile = parse_profile(valid_payload())
    assert profile.structural.target_words == (60, 110)
    assert profile.register_spec.formality == 3
    assert profile.behavioural.escalation_ladder == ["neutral", "firm", "formal"]


def test_round_trips_through_json():
    profile = parse_profile(valid_payload())
    assert parse_profile(profile.to_json_dict()) == profile


def test_the_stored_key_is_register_not_the_internal_attribute_name():
    """'register' shadows a method Pydantic inherits from its metaclass, so the attribute
    is register_spec and the wire format is held in place by an alias. Without this the
    field is silently dropped and an entire missing section validates cleanly."""
    stored = parse_profile(valid_payload()).to_json_dict()
    assert set(stored) == {"structural", "register", "lexical", "behavioural"}
    assert stored["register"]["formality"] == 3


def test_the_register_section_is_actually_validated():
    """Regression: this passed before the alias, because the field did not exist."""
    with pytest.raises(ValidationError):
        parse_profile(valid_payload(register={"formality": 99}))


# --- invariant 1: content cannot hide in the profile -------------------------------

def test_a_sentence_cannot_masquerade_as_a_preferred_term():
    """The four-word cap is the specific mechanism that stops correspondence leaking in
    as vocabulary."""
    with pytest.raises(ValidationError, match="capped at 4"):
        parse_profile(valid_payload(lexical={
            "preferred_terms": ["we would be delighted to discuss this further with you"],
        }))


def test_the_specifications_own_banned_phrase_example_is_accepted():
    """The spec states a four-word cap for both term lists, then gives a seven-word banned
    phrase as its own example. The example wins — a stock opener is long-winded by nature,
    and a cap that rejects the canonical example of the field's purpose is the wrong cap.
    See the note in services/style/profile.py."""
    profile = parse_profile(valid_payload(lexical={
        "banned_phrases": ["I hope this email finds you well"],
    }))
    assert profile.lexical.banned_phrases == ["I hope this email finds you well"]


def test_a_banned_phrase_still_cannot_be_a_paragraph():
    """Looser than preferred_terms, but still bounded. The n-gram test in Phase 2 is what
    catches a genuinely lifted sentence."""
    with pytest.raises(ValidationError, match="capped at 12"):
        parse_profile(valid_payload(lexical={
            "banned_phrases": [
                "I trust this message finds you in good health and that the quarter "
                "has started well for you and the wider team"
            ],
        }))


def test_free_text_fields_are_too_short_to_hold_a_borrowed_sentence():
    lifted = (
        "Thanks for coming back to me so quickly on the racking quote, I have passed "
        "it to finance and will confirm by Thursday at the latest."
    )
    for field in ("greeting", "sign_off", "subject_pattern"):
        with pytest.raises(ValidationError):
            parse_profile(valid_payload(structural={field: lifted}))


def test_closed_vocabularies_cannot_hold_prose():
    """A field that can only be 'short_prose' or 'bullets' cannot hold a sentence."""
    for section, field in (
        ("structural", "body_form"),
        ("structural", "opening_move"),
        ("register", "hedging"),
        ("register", "person"),
        ("lexical", "number_format"),
        ("lexical", "date_format"),
        ("behavioural", "cta_form"),
        ("behavioural", "deadline_phrasing"),
    ):
        with pytest.raises(ValidationError):
            parse_profile(valid_payload(**{section: {field: "We are requesting a quotation"}}))


def test_term_lists_are_bounded():
    with pytest.raises(ValidationError):
        parse_profile(valid_payload(lexical={
            "preferred_terms": [f"term {i}" for i in range(40)],
        }))


# --- unknown keys ------------------------------------------------------------------

def test_unknown_keys_are_rejected_at_every_level():
    with pytest.raises(ValidationError, match="extra"):
        parse_profile({**valid_payload(), "sample_emails": ["..."]})
    with pytest.raises(ValidationError, match="extra"):
        parse_profile(valid_payload(structural={"example_body": "Hi Sam, quick one..."}))


def test_missing_sections_are_rejected():
    payload = valid_payload()
    del payload["register"]
    with pytest.raises(ValidationError):
        parse_profile(payload)


# --- value sanity ------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, 6, -1])
def test_numeric_scales_are_one_to_five(value):
    with pytest.raises(ValidationError):
        parse_profile(valid_payload(register={"formality": value}))


def test_target_words_must_not_be_inverted():
    with pytest.raises(ValidationError, match="inverted"):
        parse_profile(valid_payload(structural={"target_words": [200, 60]}))


def test_target_words_must_be_plausible_for_an_email():
    with pytest.raises(ValidationError, match="implausible"):
        parse_profile(valid_payload(structural={"target_words": [60, 50000]}))
    with pytest.raises(ValidationError, match="positive"):
        parse_profile(valid_payload(structural={"target_words": [0, 110]}))


def test_escalation_ladder_repeats_are_collapsed_not_rejected():
    """The ladder is an ordered sequence of distinct tones, so a repeat carries no
    information to lose. It also has to be tolerated: generation is grammar-constrained
    against this schema, and a schema cannot express "each value used once" — a decoder
    at temperature 0 pads the array with the highest-probability option. Rejecting that
    would fail compilation over a difference that means nothing."""
    profile = parse_profile(valid_payload(behavioural={
        "escalation_ladder": ["neutral", "firm", "firm", "neutral", "formal"],
    }))
    assert profile.behavioural.escalation_ladder == ["neutral", "firm", "formal"]


def test_a_term_cannot_be_both_preferred_and_banned():
    with pytest.raises(ValidationError, match="both preferred and banned"):
        parse_profile(valid_payload(lexical={
            "preferred_terms": ["unit rate"],
            "banned_phrases": ["Unit Rate"],
        }))


def test_whitespace_in_terms_is_normalised_not_rejected():
    profile = parse_profile(valid_payload(lexical={
        "preferred_terms": ["  unit    rate  ", "volume tiers"],
    }))
    assert profile.lexical.preferred_terms == ["unit rate", "volume tiers"]


def test_empty_term_lists_are_allowed():
    profile = parse_profile(valid_payload(lexical={
        "preferred_terms": [], "banned_phrases": [],
    }))
    assert profile.lexical.preferred_terms == []


def test_there_is_no_lenient_parse_mode():
    """A profile that does not validate is not a profile the model should be held to."""
    assert not hasattr(StyleProfile, "model_construct_lenient")
    with pytest.raises(ValidationError):
        parse_profile({"structural": {}})
