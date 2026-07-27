"""The style profile schema — a description of writing HABITS, never of content.

This model is the enforcement point for invariant 1. A profile says *how* someone
writes: how long their emails run, whether they use contractions, where the ask sits
relative to the context, what they sign off with. It must never carry *what* they
wrote.

Two mechanisms keep it that way, and they are deliberately belt-and-braces:

* **Closed vocabularies.** Most fields are ``Literal`` sets rather than free text. A
  field that can only hold ``"short_prose"`` or ``"bullets"`` cannot hold a sentence
  lifted from someone's email. This also gives the compile step a fixed menu, which is
  what makes grammar-constrained generation reliable.
* **Length caps on everything else.** The handful of genuinely free-text fields
  (greeting, sign-off, subject pattern, domain terms) are capped short enough that a
  borrowed sentence will not fit. ``preferred_terms`` and ``banned_phrases`` are capped
  at four words per entry, which is the specific limit that stops content leaking in
  under the guise of vocabulary.

Neither is sufficient alone, and neither replaces the n-gram test in Phase 2 that
compares a compiled profile against the emails it was compiled from. They are the cheap
checks that run first.

``extra="forbid"`` throughout: a key nobody designed is a key nobody validated.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Free-text caps. Deliberately tight — see the module docstring.
_MAX_SHORT_TEXT = 80      # greeting, sign-off, subject pattern
_MAX_TERMS = 25
_MAX_LADDER_STEPS = 5

# The two term lists are capped differently, resolving a contradiction in the spec: it
# states a four-word maximum for BOTH lists, then gives "I hope this email finds you well"
# — seven words — as its own example of a banned phrase.
#
# The example wins, because the two fields do different jobs. A preferred term is domain
# vocabulary ("unit rate", "volume tiers") and four words is already generous. A banned
# phrase is a stock opener, and being long-winded is precisely why it is banned; a
# four-word cap would make the field unable to hold the canonical example of its own
# purpose. The leak risk also runs the other way — banned phrases record what the writer
# never writes, so they are boilerplate rather than their content.
#
# This is a loosening, so it does not carry invariant 1 on its own. The n-gram test in
# Phase 2, which compares the compiled profile against the emails it was compiled from,
# is what catches a real sentence smuggled into either list.
_MAX_PREFERRED_TERM_WORDS = 4
_MAX_PREFERRED_TERM_CHARS = 60
_MAX_BANNED_PHRASE_WORDS = 12
_MAX_BANNED_PHRASE_CHARS = 90


class _Strict(BaseModel):
    """Base for every profile section: unknown keys are an error, not a curiosity."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StructuralSpec(_Strict):
    """Shape of the email: what goes where, and how much of it there is."""

    # A pattern, not a subject line. "topic — reference", not "RFQ for 40 chairs".
    subject_pattern: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    # Placeholders such as {first_name} are expected here.
    greeting: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    opening_move: Literal["context_before_ask", "ask_before_context", "greeting_only"]
    body_form: Literal["short_prose", "long_prose", "bullets", "mixed"]
    target_words: Tuple[int, int]
    sign_off: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    signature_block: bool

    @field_validator("target_words")
    @classmethod
    def _sane_word_range(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        low, high = value
        if low < 1 or high < 1:
            raise ValueError("target_words must be positive")
        if low > high:
            raise ValueError(f"target_words is inverted: {low} > {high}")
        if high > 2000:
            raise ValueError("target_words upper bound is implausible for an email")
        return value


class RegisterSpec(_Strict):
    """Tone, on fixed scales. 1-5 throughout; 3 is unremarkable."""

    formality: int = Field(ge=1, le=5)
    directness: int = Field(ge=1, le=5)
    hedging: Literal["none", "low", "medium", "high"]
    contractions: bool
    person: Literal["first_singular", "first_plural", "impersonal"]


class LexicalSpec(_Strict):
    """Vocabulary habits. Short domain terms only — this is the field most at risk of
    becoming a hiding place for content, so it is the most tightly capped."""

    preferred_terms: List[str] = Field(default_factory=list, max_length=_MAX_TERMS)
    banned_phrases: List[str] = Field(default_factory=list, max_length=_MAX_TERMS)
    number_format: Literal["bare_numerals", "words_under_ten", "grouped_thousands"]
    date_format: Literal["weekday_name", "iso", "day_month", "month_day"]

    @staticmethod
    def _clean(values: List[str], *, max_words: int, max_chars: int, field: str) -> List[str]:
        cleaned: List[str] = []
        for raw in values:
            term = " ".join(str(raw).split())
            if not term:
                continue
            words = term.split(" ")
            if len(words) > max_words:
                raise ValueError(
                    f"{term!r} is {len(words)} words; {field} entries are capped at "
                    f"{max_words} so that correspondence cannot be smuggled in as vocabulary"
                )
            if len(term) > max_chars:
                raise ValueError(f"{term!r} exceeds {max_chars} characters")
            cleaned.append(term)
        return cleaned

    @field_validator("preferred_terms")
    @classmethod
    def _preferred_terms_are_domain_terms(cls, values: List[str]) -> List[str]:
        return cls._clean(
            values,
            max_words=_MAX_PREFERRED_TERM_WORDS,
            max_chars=_MAX_PREFERRED_TERM_CHARS,
            field="preferred_terms",
        )

    @field_validator("banned_phrases")
    @classmethod
    def _banned_phrases_are_stock_phrases_not_paragraphs(cls, values: List[str]) -> List[str]:
        return cls._clean(
            values,
            max_words=_MAX_BANNED_PHRASE_WORDS,
            max_chars=_MAX_BANNED_PHRASE_CHARS,
            field="banned_phrases",
        )


class BehaviouralSpec(_Strict):
    """What the writer habitually *does* — how they ask, and how they escalate."""

    cta_form: Literal[
        "proposes_specific_time", "open_question", "deadline_only", "none"
    ]
    deadline_phrasing: Literal["soft_by_date", "hard_deadline", "none"]
    escalation_ladder: List[
        Literal["neutral", "warm", "firm", "formal", "final"]
    ] = Field(default_factory=list, max_length=_MAX_LADDER_STEPS)

    @field_validator("escalation_ladder")
    @classmethod
    def _collapse_repeated_steps(cls, values: List[str]) -> List[str]:
        """Drop repeats, preserving order.

        Normalisation rather than rejection, and deliberately so. The ladder is an ordered
        sequence of distinct tones, so ``["neutral", "neutral", "firm"]`` and
        ``["neutral", "firm"]`` say the same thing — the repeat carries no information to
        lose.

        It also has to be tolerated. Generation is grammar-constrained against this
        model's JSON schema, and a schema can say "an array of these enum values" but not
        "each used once"; a constrained decoder at temperature 0 will happily pad the
        array with the highest-probability option. Rejecting that would fail compilation
        over a difference that means nothing.
        """

        seen: set = set()
        collapsed: List[str] = []
        for step in values:
            if step not in seen:
                seen.add(step)
                collapsed.append(step)
        return collapsed


class StyleProfile(_Strict):
    """A complete, versionable description of how one person writes email.

    Note the ``register_spec`` field name. The JSON key is ``register`` — that is what the
    specification says and what is stored — but ``register`` cannot be used as the Python
    attribute: it shadows a method Pydantic inherits from its metaclass, and the collision
    is silent. Pydantic drops the field, ``.register`` returns the bound classmethod, and a
    payload missing its entire register section validates cleanly. The alias keeps the wire
    format correct while the attribute stays out of the way.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)

    structural: StructuralSpec
    register_spec: RegisterSpec = Field(alias="register")
    lexical: LexicalSpec
    behavioural: BehaviouralSpec

    @model_validator(mode="after")
    def _preferred_and_banned_do_not_contradict(self) -> "StyleProfile":
        overlap = {t.lower() for t in self.lexical.preferred_terms} & {
            p.lower() for p in self.lexical.banned_phrases
        }
        if overlap:
            raise ValueError(
                "the same term is both preferred and banned: " + ", ".join(sorted(overlap))
            )
        return self

    def to_json_dict(self) -> dict:
        """The form stored in ``bp_style_profile.profile_json``.

        ``by_alias`` so the stored key is ``register``, not the internal attribute name.
        """

        return self.model_dump(mode="json", by_alias=True)


def parse_profile(payload: dict) -> StyleProfile:
    """Validate a compiled profile, raising ``pydantic.ValidationError`` if it is not one.

    Deliberately has no lenient mode. A profile that does not validate is not a profile
    the model should be held to, and quietly accepting a partial one would mean drafts
    generated against rules nobody approved.
    """

    return StyleProfile.model_validate(payload)
