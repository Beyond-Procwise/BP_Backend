"""Strip content out of an email, leaving only its shape.

This runs between ingestion and compilation. Whatever survives here is what the model
sees, so this is the last point at which a customer's correspondence can be stopped from
reaching a prompt. Invariant 1 rests on it.

The bias throughout is **over-redaction**. Removing too much costs a little signal;
removing too little puts someone's supplier names and prices into an LLM call. Where a
rule is uncertain, it redacts.

Two things are deliberately *not* redacted, because they are the signal rather than the
content:

* **Dates and their formatting.** ``date_format`` is a profile field — whether someone
  writes "Thursday", "14/03" or "2026-03-14" is a habit worth learning, and replacing
  dates with a placeholder would destroy it.
* **Bare quantities.** "40 chairs" tells you about ``number_format``; it identifies
  nobody. Currency amounts are a different matter and always go.

No NER model is used: none is installed, and the repo's direction is regex-primary. The
rules below are ordered so the more specific ones claim their text first — organisations
before names, so "Techworld Ltd" does not become "[NAME] Ltd".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# Placeholders. The first four are the specification's; the last three were added
# because a redactor that strips a supplier's name while leaving their direct dial and
# email address in the prompt is not a redactor.
NAME = "[NAME]"
ORG = "[ORG]"
AMOUNT = "[AMOUNT]"
REF = "[REF]"
EMAIL = "[EMAIL]"
PHONE = "[PHONE]"
URL = "[URL]"

# ---------------------------------------------------------------------------------
# Block removal — quoted chains, signatures, disclaimers
# ---------------------------------------------------------------------------------

# "On Tue, 14 Mar 2026 at 09:12, Sam Rees <sam@x.com> wrote:" and its many variants.
_REPLY_ATTRIBUTION = re.compile(
    r"^\s*On\b.{0,200}?\bwrote:\s*$", re.IGNORECASE | re.MULTILINE
)
# Outlook's separator, and the header block that follows it.
_ORIGINAL_MESSAGE = re.compile(
    r"^\s*-{2,}\s*(Original Message|Forwarded message)\s*-{2,}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# A bare "From: ... Sent: ... To: ..." header block, which Outlook emits without the
# separator above.
_HEADER_BLOCK = re.compile(
    r"^\s*From:\s.+?^\s*(?:Subject|To):\s.*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
# The conventional signature delimiter: a line of exactly "--" or "-- ".
_SIG_DELIMITER = re.compile(r"^\s*--\s*$", re.MULTILINE)

_CLOSINGS = (
    "kind regards", "best regards", "warm regards", "regards", "many thanks",
    "thanks", "thank you", "cheers", "best wishes", "best", "sincerely",
    "yours sincerely", "yours faithfully", "all the best", "speak soon",
)
_CLOSING_LINE = re.compile(
    r"^\s*(" + "|".join(re.escape(c) for c in sorted(_CLOSINGS, key=len, reverse=True))
    + r")\s*[,.!]?\s*$",
    re.IGNORECASE,
)

_DISCLAIMER_TRIGGERS = (
    "this email and any attachments",
    "this e-mail and any attachments",
    "the information contained in this",
    "this message is intended only",
    "if you are not the intended recipient",
    "please consider the environment",
    "registered in england",
    "confidentiality notice",
    "disclaimer:",
)


def _strip_quoted_chain(text: str) -> tuple[str, List[str]]:
    """Drop everything from the first sign of a quoted reply onward.

    A reply chain is someone else's correspondence sitting inside this one. It is not
    this writer's style and it is the largest single body of content in a typical thread,
    so it goes first and it goes wholesale.
    """

    removed: List[str] = []
    cut = len(text)
    for pattern, label in (
        (_REPLY_ATTRIBUTION, "reply_attribution"),
        (_ORIGINAL_MESSAGE, "original_message_separator"),
        (_HEADER_BLOCK, "quoted_header_block"),
    ):
        match = pattern.search(text)
        if match and match.start() < cut:
            cut, = (match.start(),)
            removed.append(label)

    text = text[:cut]

    # Any remaining "> " quoted lines, wherever they sit.
    lines = text.split("\n")
    kept = [ln for ln in lines if not ln.lstrip().startswith(">")]
    if len(kept) != len(lines):
        removed.append("quoted_lines")
    return "\n".join(kept), removed


def _strip_disclaimer(text: str) -> tuple[str, List[str]]:
    lowered = text.lower()
    cut = len(text)
    for trigger in _DISCLAIMER_TRIGGERS:
        idx = lowered.find(trigger)
        if idx != -1:
            # Cut from the start of the line the trigger sits on.
            line_start = text.rfind("\n", 0, idx) + 1
            cut = min(cut, line_start)
    if cut < len(text):
        return text[:cut].rstrip(), ["disclaimer"]
    return text, []


def _strip_signature(text: str) -> tuple[str, List[str], str]:
    """Remove the signature block, returning the closing line so the sign-off habit
    survives even though the name and contact details do not.

    The sign-off itself is a style signal — "Nick" versus "Kind regards, Nicholas
    Geelen, Procurement Lead" is exactly the kind of habit a profile records. What
    follows the name is contact detail and job title, and none of it is style.
    """

    removed: List[str] = []
    closing = ""

    # Explicit "-- " delimiter wins where present.
    delim = _SIG_DELIMITER.search(text)
    if delim:
        text = text[: delim.start()].rstrip()
        removed.append("signature_delimiter")

    lines = text.split("\n")
    # Find the last closing line ("Kind regards,"). Everything after it is the name and
    # whatever the mail client appended.
    for idx in range(len(lines) - 1, -1, -1):
        if _CLOSING_LINE.match(lines[idx]):
            closing = lines[idx].strip()
            trailing = [ln for ln in lines[idx + 1:] if ln.strip()]
            if trailing:
                removed.append("signature_block")
            # Keep the closing line; drop the name and everything under it. The name is
            # re-inserted as [NAME] so the profile can still learn that a sign-off exists.
            lines = lines[: idx + 1] + ([NAME] if trailing else [])
            break

    return "\n".join(lines).rstrip(), removed, closing


# ---------------------------------------------------------------------------------
# Entity replacement
# ---------------------------------------------------------------------------------

_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?(?:\(?0\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}(?![\w.])"
)

_CURRENCY = r"(?:[£$€¥]|\b(?:GBP|USD|EUR|JPY|CHF|AUD|CAD|SEK|NOK|DKK|PLN|ZAR|INR)\b)"
_MONEY_RE = re.compile(
    rf"{_CURRENCY}\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?"
    rf"|\b\d[\d,]*(?:\.\d+)?\s?{_CURRENCY}",
    re.IGNORECASE,
)

# Identifiers: letter-prefixed codes, long digit runs, slash/hyphen composites.
_REF_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{2,}[-/ ]?\d{3,}[A-Z\d/-]*"      # INV-0042, PO 12345, QUT136586
    r"|[A-Z]+\d{4,}"                          # ABC12345
    r"|\d{5,}"                                # bare long numbers
    r"|\d{2,}/\d{2,}[/\d-]*"                  # 136586/2026
    r")\b"
)

_ORG_SUFFIX = (
    "Ltd", "Limited", "PLC", "Plc", "Inc", "Incorporated", "LLC", "LLP", "GmbH",
    "AG", "BV", "NV", "SA", "SAS", "AB", "Oy", "Pty", "Group", "Holdings",
    "Partners", "Associates", "Solutions", "Services", "Technologies", "Industries",
)
_ORG_RE = re.compile(
    r"\b(?:[A-Z][\w&'’.-]*\s+){0,4}[A-Z][\w&'’.-]*\s+(?:"
    + "|".join(_ORG_SUFFIX) + r")\b\.?"
)

_GREETING_RE = re.compile(
    r"\b(Hi|Hello|Hey|Dear|Morning|Afternoon|Good morning|Good afternoon)\b"
    r"[ \t]+([A-Z][\w'’-]+(?:[ \t]+[A-Z][\w'’-]+){0,2})",
)

# Words that are capitalised for reasons other than being a name. Redacting these would
# strip the writer's actual vocabulary, which is the thing being learned.
_NOT_A_NAME = {
    # calendar — deliberately preserved, see the module docstring
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    # procurement vocabulary
    "rfq", "rfp", "rfi", "po", "vat", "eta", "uom", "qty", "net", "sla", "kpi",
    "invoice", "quote", "quotation", "purchase", "order", "contract", "supplier",
    "delivery", "payment", "terms", "pricing", "price", "rate", "tier", "tiers",
    "lead", "time", "warranty", "framework", "tender", "award", "scope", "spec",
    # greetings and closings — these ARE the habit (`structural.greeting`,
    # `structural.sign_off`), so they must survive even though they always sit next to
    # a name that must not
    "hi", "hello", "hey", "dear", "morning", "afternoon", "evening", "greetings",
    "regards", "kind", "best", "warm", "cheers", "sincerely", "yours", "faithfully",
    "wishes", "speak", "soon",
    # sentence-openers, connectives and other words that are capitalised for reasons
    # other than being a name. Not exhaustive and cannot be — the word-at-a-time rule
    # above is what stops a gap here from taking neighbouring words down with it.
    "i", "we", "you", "the", "this", "that", "it", "if", "as", "at", "in", "on",
    "please", "thanks", "thank", "could", "would", "should", "can", "happy", "just",
    "also", "our", "your", "their", "there", "here", "however", "unfortunately",
    "apologies", "otherwise", "meanwhile", "alternatively", "additionally", "regarding",
    "what", "when", "where", "which", "who", "why", "how", "let", "hope", "sorry",
    "good", "many", "all", "some", "both", "one", "two", "three", "four", "five",
    "agreed", "confirmed", "noted", "understood", "done", "yes", "no", "not",
    "for", "from", "with", "and", "but", "or", "so", "to", "of", "by", "up", "out",
    "given", "based", "following", "further", "final", "next", "last", "first",
    "attached", "below", "above", "subject", "re", "fw", "fwd",
}

# One capitalised word, contraction included, so "What's" is a single token rather than
# "What" plus a stray apostrophe.
_CAPITALISED_WORD = re.compile(r"\b[A-Z][a-z]+(?:['’][a-z]+)?\b")
# Two or more adjacent placeholders collapse into one: "Nicholas Geelen" should read as
# a single [NAME], not "[NAME] [NAME]".
_ADJACENT_NAMES = re.compile(rf"(?:{re.escape(NAME)})(?:\s+{re.escape(NAME)})+")


@dataclass
class RedactionResult:
    """Redacted text, plus enough detail to audit what happened without storing it."""

    text: str
    counts: Dict[str, int] = field(default_factory=dict)
    blocks_removed: List[str] = field(default_factory=list)
    # The redacted body alone, without the subject line ``text`` may carry. Kept separate
    # because an email whose body was entirely a quoted reply chain redacts to nothing,
    # and a surviving subject line would otherwise make it look like a writing sample.
    body_text: str = ""

    @property
    def total_replacements(self) -> int:
        return sum(self.counts.values())

    @property
    def has_body(self) -> bool:
        """Whether anything of the writer's own prose survived redaction."""

        return bool(self.body_text.strip())


def _sub_counting(pattern: re.Pattern, placeholder: str, text: str,
                  counts: Dict[str, int], key: str) -> str:
    text, n = pattern.subn(placeholder, text)
    if n:
        counts[key] = counts.get(key, 0) + n
    return text


def _redact_names(text: str, counts: Dict[str, int]) -> str:
    # Greetings first: "Hi Sam," -> "Hi [NAME]," keeps the greeting habit intact.
    def _greet(match: re.Match) -> str:
        counts["name"] = counts.get("name", 0) + 1
        return f"{match.group(1)} {NAME}"

    text = _GREETING_RE.sub(_greet, text)

    # Then every remaining capitalised word that is not ordinary vocabulary, judged one
    # word at a time. Judging whole runs was wrong: "Otherwise Monday" would be replaced
    # entirely because "Otherwise" was unrecognised, taking a weekday — which the profile
    # needs — down with it.
    #
    # Aggressive by design. A false positive costs one word of signal; a false negative
    # leaks a person's name into an LLM prompt.
    def _word(match: re.Match) -> str:
        token = match.group(0)
        base = re.split(r"['’]", token)[0].lower()
        if base in _NOT_A_NAME:
            return token

        # A capitalised word that opens a sentence AND is followed by lowercase prose is
        # a sentence opener, not a name — "What's driving this?" rather than "Nicholas
        # Geelen". Only spaces and tabs are stripped when looking back, so a newline still
        # reads as a boundary.
        start = match.start()
        preceding = text[:start].rstrip(" \t")
        at_sentence_start = not preceding or preceding[-1] in ".!?:\n"
        if at_sentence_start:
            after = text[match.end():]
            continues_in_prose = bool(re.match(r"[ \t]+[a-z]", after))
            if continues_in_prose:
                return token

        counts["name"] = counts.get("name", 0) + 1
        return NAME

    text = _CAPITALISED_WORD.sub(_word, text)

    def _collapse(match: re.Match) -> str:
        # The run counted once per word; charge it once per person instead.
        counts["name"] = counts.get("name", 0) - (len(match.group(0).split()) - 1)
        return NAME

    return _ADJACENT_NAMES.sub(_collapse, text)


def redact(body: str, subject: str | None = None) -> RedactionResult:
    """Reduce an email to its shape.

    Returns the redacted text (subject prepended when supplied) and a tally of what was
    replaced. The tally is safe to log; the input is not.
    """

    if not body or not isinstance(body, str):
        return RedactionResult(text="")

    text = body.replace("\r\n", "\n").replace("\r", "\n")

    # Our own tracking marker, if this body ever passed through the dispatch path.
    try:
        from utils.email_markers import split_hidden_marker

        _, text = split_hidden_marker(text)
    except Exception:  # pragma: no cover - marker utility is optional here
        pass

    blocks: List[str] = []
    text, removed = _strip_quoted_chain(text)
    blocks += removed
    text, removed = _strip_disclaimer(text)
    blocks += removed
    text, removed, _closing = _strip_signature(text)
    blocks += removed

    # Where the body was entirely a quoted chain there is nothing left, and a subject line
    # would only disguise that. Record the body separately before the subject is prepended.
    body_only = re.sub(r"\n{3,}", "\n\n", text).strip()

    if subject:
        text = f"Subject: {subject.strip()}\n\n{text.lstrip()}"

    counts: Dict[str, int] = {}
    # Order matters: the specific patterns claim their text before the general ones.
    text = _sub_counting(_URL_RE, URL, text, counts, "url")
    text = _sub_counting(_EMAIL_RE, EMAIL, text, counts, "email")
    text = _sub_counting(_MONEY_RE, AMOUNT, text, counts, "amount")
    text = _sub_counting(_REF_RE, REF, text, counts, "ref")
    text = _sub_counting(_PHONE_RE, PHONE, text, counts, "phone")
    text = _sub_counting(_ORG_RE, ORG, text, counts, "org")
    text = _redact_names(text, counts)

    # Collapse the gaps that removed blocks leave behind.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return RedactionResult(
        text=text, counts=counts, blocks_removed=blocks, body_text=body_only
    )
