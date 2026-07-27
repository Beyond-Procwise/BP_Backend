"""Render a style profile as instructions a model can follow.

``profile_json`` is a data structure. Pasting it into a prompt asks the model to both
interpret a schema and obey it, and models are markedly better at following prose. So the
profile is rendered into short imperative rules — the same rendering is used to generate
exemplars in Phase 3 and to draft in Phase 4, because an exemplar generated against
different wording than the drafter sees would be illustrating the wrong specification.

Nothing here reads from anywhere but the profile. That is what makes the output safe to
put in a prompt: the profile contains no correspondence, so neither can this.
"""

from __future__ import annotations

from typing import List

from services.style.profile import StyleProfile

_FORMALITY = {
    1: "very informal, as you would write to a close colleague",
    2: "informal",
    3: "neutral — professional but not stiff",
    4: "formal",
    5: "very formal",
}

_DIRECTNESS = {
    1: "very indirect; circle the point before making it",
    2: "indirect",
    3: "balanced",
    4: "direct; state the point plainly",
    5: "very direct; lead with the point and do not soften it",
}

_HEDGING = {
    "none": "Do not hedge. No 'perhaps', 'possibly' or 'I was wondering whether'.",
    "low": "Hedge rarely.",
    "medium": "Hedge where a claim is genuinely uncertain.",
    "high": "Hedge frequently; prefer tentative phrasing.",
}

_PERSON = {
    "first_singular": "Write in the first person singular ('I').",
    "first_plural": "Write in the first person plural ('we').",
    "impersonal": "Write impersonally; avoid 'I' and 'we'.",
}

_OPENING = {
    "context_before_ask": "Give the context first, then make the ask.",
    "ask_before_context": "Make the ask first, then give the context.",
    "greeting_only": "Open with the greeting and go straight into the message.",
}

_BODY_FORM = {
    "short_prose": "Write a few short sentences of prose. No bullet points.",
    "long_prose": "Write flowing prose across several paragraphs. No bullet points.",
    "bullets": "Use bullet points for the substance.",
    "mixed": "Open in prose, then use bullets for any list of items.",
}

_NUMBERS = {
    "bare_numerals": "Write numbers as numerals ('3 units', not 'three units').",
    "words_under_ten": "Spell out numbers under ten; use numerals from ten upwards.",
    "grouped_thousands": "Write numbers as numerals with thousands separators.",
}

_DATES = {
    "weekday_name": "Refer to dates by weekday where possible ('by Thursday').",
    "iso": "Write dates in ISO form (2026-03-14).",
    "day_month": "Write dates as day then month (14 March).",
    "month_day": "Write dates as month then day (March 14).",
}

_CTA = {
    "proposes_specific_time": "Close by proposing a specific time to speak.",
    "open_question": "Close with an open question.",
    "deadline_only": "Close by stating what you need and by when, without proposing a call.",
    "none": "Do not add a call to action.",
}

_DEADLINE = {
    "soft_by_date": "Phrase deadlines softly ('if you could let me know by Friday').",
    "hard_deadline": "State deadlines firmly and unambiguously.",
    "none": "Do not state a deadline unless the task supplies one.",
}


def render_profile_rules(profile: StyleProfile) -> str:
    """The profile as a numbered list of imperative rules."""

    s = profile.structural
    r = profile.register_spec
    lex = profile.lexical
    b = profile.behavioural

    rules: List[str] = [
        f"Subject line pattern: {s.subject_pattern}",
        f"Open with: {s.greeting}",
        _OPENING[s.opening_move],
        _BODY_FORM[s.body_form],
        f"Length: aim for {s.target_words[0]}-{s.target_words[1]} words in the body.",
        f"Tone: {_FORMALITY[r.formality]}; {_DIRECTNESS[r.directness]}.",
        _HEDGING[r.hedging],
        _PERSON[r.person],
        (
            "Use contractions naturally (it's, we're, I'd)."
            if r.contractions
            else "Do not use contractions; write them out in full."
        ),
        _NUMBERS[lex.number_format],
        _DATES[lex.date_format],
        _CTA[b.cta_form],
        _DEADLINE[b.deadline_phrasing],
        f"Sign off with: {s.sign_off}",
        (
            "Include a signature block after the sign-off."
            if s.signature_block
            else "Do not add a signature block, job title or contact details."
        ),
    ]

    if lex.preferred_terms:
        rules.append("Prefer these terms where they fit: " + ", ".join(lex.preferred_terms) + ".")
    if lex.banned_phrases:
        rules.append(
            "Never use these phrases: " + "; ".join(f'"{p}"' for p in lex.banned_phrases) + "."
        )
    if b.escalation_ladder:
        rules.append(
            "Escalation ladder, in order across successive chases: "
            + " -> ".join(b.escalation_ladder) + "."
        )

    rendered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))

    # A glossary, kept off the rule lines themselves. An earlier version put this as a
    # parenthetical next to the sign-off template and the model copied it straight into
    # the email — output read "Alex  (Alex)". Guidance sitting inside a template gets
    # treated as part of the template.
    slots = []
    if "{first_name}" in s.greeting:
        slots.append("{first_name} is the person being written to")
    if "{sender_first_name}" in s.sign_off:
        slots.append("{sender_first_name} is the writer themselves")
    if slots:
        rendered += (
            "\n\nPlaceholder meanings (substitute real values; never output the "
            "placeholder itself or this note): " + "; ".join(slots) + "."
        )

    return rendered
