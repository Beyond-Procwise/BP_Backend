"""Rewrite a grounded email in a different tone, without letting the figures move.

`value_query_service` builds a supplier query with NO model in the path: every figure is
filled from a stored discrepancy row, so the email cannot state a number the database does
not hold. That property is the whole reason a buyer can press send on a message accusing a
named company of over-billing.

A tone control puts a model back in that path. This module is what keeps the guarantee:
the model is shown the finished draft and asked to re-word it, and its answer is then
checked AGAINST that draft. Nothing new may appear (no invented VAT line, no transposed
digits) and nothing that identifies the claim may vanish (the amount, the document
references, the supplier's name). A rewrite that fails either check is discarded and the
grounded draft stands.

Failure is always narrated. Returning the template silently would read to the buyer as
"the tone was applied", and they would send a formal email believing it was conciliatory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# `formal` is not in the model's gift: it is the voice the stored template is already
# written in, so asking for it means "leave the grounded draft exactly as it is".
TONES = ("formal", "direct", "warm", "conciliatory")

_TONE_BRIEF = {
    "direct": "plain and businesslike — short sentences, straight to the point, no "
              "padding beyond the greeting and sign-off",
    "warm": "friendly and cooperative, written as though this is an honest mistake "
            "between two companies that work well together",
    "conciliatory": "careful and non-accusatory, leaving the supplier obvious room to "
                    "give a reasonable explanation without losing face",
}

_SYSTEM = (
    "You re-word procurement emails. You are given a finished email and a tone. Rewrite it "
    "in that tone.\n"
    "ABSOLUTE RULES:\n"
    "1. Every amount, currency, invoice number, purchase-order number and company name must "
    "appear in your rewrite EXACTLY as written in the original, character for character.\n"
    "2. Never introduce a number that is not in the original — no totals, no percentages, "
    "no dates, no VAT.\n"
    "3. Do not change what is being asked for.\n"
    "Reply with the rewritten email only. No preamble, no explanation, no quotation marks "
    "around it."
)

_DEFAULT_MODEL = "AgentNick:unified"

# Digits with optional thousands separators and decimals. Deliberately greedy about what
# counts as a number: the digits inside "INV000047-1A" are matched too, so a rewrite that
# quietly renumbers an invoice is caught by the same rule that catches an invented total.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Retone:
    """What the buyer gets. `body` is always safe to send.

    `applied` is False whenever the body is the original — because the tone was `formal`,
    because the model could not be reached, or because its rewrite was rejected. `note`
    says which, in words a buyer can act on.
    """
    body: str
    applied: bool
    note: Optional[str] = None


def numbers(text: str) -> set:
    """Every numeric token in `text`, with thousands separators removed.

    Normalising the separators means "147,783.11" and "147783.11" compare equal: a rewrite
    that reformats a figure it did not actually change must not be thrown away, or the tone
    control would fail constantly for no reason a buyer could see.
    """
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(str(text or ""))}


def _still_names(text: str, name: str) -> bool:
    """Does `text` still say who this is about?

    Looser than the byte-exact rule the FIGURES get, and deliberately so. A supplier name
    appears in the greeting, where re-wording it is the point of the control — and this
    corpus names suppliers with a trailing counter ("Ashcroft Logistics 14"). Held to the
    letter, a model writing the perfectly correct "Hello Ashcroft Logistics," had its
    rewrite binned for "dropping" the 14, so the tone control fell back on nearly every
    supplier while every figure in the message was untouched.

    Only a trailing number is forgiven. "Ashcroft Logistics" still names the same company;
    "Ashcroft Freight" does not, and is still a violation. Nothing here can change WHO the
    email reaches — that is the `to` field, which no model writes.
    """
    if name in text:
        return True
    core = re.sub(r"[\s,\-]*\d+$", "", name).strip()
    return bool(core) and core != name and core in text


def violations(original: str, rewritten: str, *, must_keep: Iterable[str],
               must_name: Optional[str] = None) -> list:
    """Everything wrong with `rewritten`, in plain English. Empty means it is safe.

    Three checks, because they catch different failures:
      * nothing invented — the rewrite's numbers must be a subset of the draft's. This is
        what stops a model helpfully adding "plus 12,500.00 VAT" to a credit-note request;
      * nothing dropped — each string in `must_keep` THAT THE DRAFT ACTUALLY CONTAINED must
        survive. Scoped to the draft on purpose: a finding with no PO reference must not
        have its rewrite rejected for "dropping" a string that was never there;
      * still named — `must_name` (the supplier) must still be recognisable, by the looser
        rule in `_still_names`. It is a salutation, not a figure.
    """
    problems = []

    invented = numbers(rewritten) - numbers(original)
    if invented:
        problems.append("it introduces figures the finding does not contain: "
                        + ", ".join(sorted(invented)))

    for value in must_keep:
        value = str(value or "").strip()
        if value and value in original and value not in rewritten:
            problems.append(f"it drops {value!r}, which the claim rests on")

    name = str(must_name or "").strip()
    if name and name in original and not _still_names(rewritten, name):
        problems.append(f"it no longer names {name!r}")

    return problems


def retone(body: str, *, tone: str, must_keep: Iterable[str], agent_nick: Any,
           must_name: Optional[str] = None, model: Optional[str] = None) -> Retone:
    """Re-word `body` in `tone`, or say why it still reads as it did.

    Never raises and never returns an unchecked rewrite: every path out of here yields a
    body that either IS the grounded draft or has been compared against it.
    """
    original = str(body or "")
    tone = str(tone or "").strip().lower()
    must_keep = list(must_keep or [])

    if tone in ("", "formal"):
        return Retone(original, False)          # the template is already the formal voice
    if tone not in _TONE_BRIEF:
        return Retone(original, False,
                      f"'{tone}' is not a tone this can write in, so the draft is unchanged.")
    caller = resolve_caller(agent_nick)
    if caller is None:
        return Retone(original, False,
                      "The drafting agent is not available, so the wording is unchanged.")

    prompt = (f"Rewrite this email so it reads {_TONE_BRIEF[tone]}.\n\n"
              f"Tone: {tone}\n\nEmail:\n{original}")
    try:
        response = caller.call_ollama(
            model=model or _resolve_model(agent_nick),
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": prompt}],
            think=False,               # reasoning models return an empty response otherwise
            options={"temperature": 0},  # temperature belongs INSIDE options, or the client raises
        )
    except Exception:  # noqa: BLE001
        logger.exception("re-tone to %s failed", tone)
        return Retone(original, False,
                      "The drafting agent could not be reached, so the wording is unchanged.")

    rewritten = _text(response).strip()
    if not rewritten:
        return Retone(original, False,
                      "The drafting agent returned nothing, so the wording is unchanged.")

    problems = violations(original, rewritten, must_keep=must_keep, must_name=must_name)
    if problems:
        # Logged with the offending text: this is the signal that the tone prompt needs
        # work, and it is invisible if only the buyer-facing sentence survives.
        logger.warning("re-tone to %s rejected (%s): %s", tone, "; ".join(problems), rewritten)
        return Retone(original, False,
                      "The re-worded version was rejected because " + problems[0]
                      + ". The original wording, which is built from the finding, is what "
                        "you see below.")

    return Retone(rewritten, True)


def resolve_caller(agent_nick: Any) -> Any:
    """An object exposing ``call_ollama``, from whatever the caller had to hand.

    `AgentNick` does NOT inherit from `BaseAgent` and has no `call_ollama`, so the
    `app.state.agent_nick` that every router carries cannot be called directly — passing it
    straight through is why the first live run of this module degraded to "the drafting
    agent is not available" on a perfectly healthy server. Same resolution as
    `DecisionEngine._reply_caller`: an already-registered agent instance if there is one
    (they are all BaseAgent subclasses sharing one AgentNick, so model, settings and pool
    are identical whichever is picked), else a bare BaseAgent.

    Returns None if neither is possible. Never raises, and never reaches for a different
    model — AgentNick is the only model in this system.
    """
    if agent_nick is None:
        return None
    if callable(getattr(agent_nick, "call_ollama", None)):
        return agent_nick                      # already a caller (or a test double)
    agents = getattr(agent_nick, "agents", None)
    if isinstance(agents, dict):
        for instance in agents.values():
            if callable(getattr(instance, "call_ollama", None)):
                return instance
    try:
        from src.agents.base_agent import BaseAgent
        return BaseAgent(agent_nick)
    except Exception:  # noqa: BLE001
        logger.exception("no LLM caller could be obtained for a tone re-word")
        return None


def _resolve_model(agent_nick: Any) -> str:
    """AgentNick's configured name. There is no other model to fall back to."""
    settings_obj = getattr(agent_nick, "settings", None)
    if settings_obj is None:
        try:
            from config.settings import settings as settings_obj
        except Exception:  # noqa: BLE001
            return _DEFAULT_MODEL
    return getattr(settings_obj, "local_primary_model", _DEFAULT_MODEL) or _DEFAULT_MODEL


def _field(obj: Any, name: str) -> Any:
    """A mapping key or an attribute, whichever the object has."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _text(response: Any) -> str:
    """The model's text out of whichever shape ``call_ollama`` returned.

    It returns the raw `ollama` client result: the chat shape (``message.content``) when
    called with ``messages=``, or the generate shape (``response``) otherwise. In the live
    client BOTH are OBJECTS, not dicts — a dict-only reader falls through to `str(response)`
    and hands back the whole repr, `total_duration=4723328056` and all.

    That is not hypothetical. The first live run of this module did exactly that: AgentNick
    returned a flawless conciliatory rewrite keeping every figure, the repr's timings and
    ISO timestamp were read as "invented figures", and the guard threw a perfectly good
    rewrite away. Same trap, same fix as `email_intent._extract_ollama_message`, which
    documents it having been caught the same way on 2026-07-28.
    """
    content = _field(_field(response, "message"), "content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    content = _field(response, "response")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""
