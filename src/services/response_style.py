"""Enforce the governed response style on a generated answer.

The style itself is authored in ``proc.bp_prompt`` (``ask_persona/joshi``) and
sent with every request. A prompt is a request, though — nothing prevented the
model returning a document outline, and nothing downstream looked. This module
is the enforcement half: it removes presentation markers the style forbids, so
what reaches the reader matches the governed style regardless of what the model
produced.

It is deliberately narrow. It deletes formatting characters and nothing else:

* ATX headers (``## Findings``)      -> the heading's words, as a plain line
* horizontal rules (``---``)         -> dropped (they carry no content)
* blockquote markers (``> ...``)     -> unwrapped, text kept
* a bolded line on its own           -> unwrapped (it is a header in disguise)
* emoji used as decoration           -> stripped from line edges only

Everything else — inline bold, genuine bullet and numbered lists, every word,
digit and currency figure — is passed through untouched. The previous formatter
in this codebase rewrote prose (it read "£1,200." as a list marker and rendered
"200. Delivery was late"), which is why the tests assert byte-level content
preservation rather than eyeballing the output.
"""

from __future__ import annotations

import re
from typing import List, Optional

# `## Heading`, with optional closing hashes. Requires a space after the hashes
# so a bare "#1 supplier" is not mistaken for a header.
_ATX_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.*?)\s*#*\s*$")

# A rule is the whole line: ---, ***, ___ (three or more).
_RULE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")

# Blockquote marker at the head of a line, one level or several ("> > ").
_QUOTE = re.compile(r"^\s{0,3}(?:>\s?)+")

# A line that is entirely bold, optionally ending in a colon: a fake header.
# Inline bold inside a sentence does not match and is left alone.
_WHOLE_LINE_BOLD = re.compile(r"^\s*\*\*(?P<text>.+?)\*\*\s*:?\s*$")

# Pictographs, dingbats and variation selectors. Applied only at line edges so
# an emoji quoted inside a sentence from a source document is preserved.
_EMOJI_CLASS = (
    "\U0001F300-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE0F\U0000200D\U000020E3"
)
_EMOJI_LEAD = re.compile(rf"^[{_EMOJI_CLASS}\s]*[{_EMOJI_CLASS}]\s*")
_EMOJI_TRAIL = re.compile(rf"\s*[{_EMOJI_CLASS}][{_EMOJI_CLASS}\s]*$")


def _strip_decorative_emoji(line: str) -> str:
    line = _EMOJI_LEAD.sub("", line)
    line = _EMOJI_TRAIL.sub("", line)
    return line


def enforce_response_style(text: Optional[str]) -> str:
    """Return ``text`` with forbidden presentation markers removed.

    Content is never altered: the words, numbers and figures that go in come
    out, in the same order. Only formatting characters are dropped.
    """

    if not text or not isinstance(text, str):
        return ""

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    out: List[str] = []

    for raw_line in normalised.split("\n"):
        line = raw_line

        # A rule is pure decoration — drop the line outright.
        if _RULE.match(line):
            continue

        line = _QUOTE.sub("", line)

        header = _ATX_HEADER.match(line)
        if header:
            line = header.group("text")

        bold_line = _WHOLE_LINE_BOLD.match(line)
        if bold_line:
            line = bold_line.group("text")

        line = _strip_decorative_emoji(line)

        # A line that held nothing but decoration disappears rather than
        # leaving a blank gap where a rule or a lone emoji used to be.
        if not line.strip() and raw_line.strip():
            continue

        out.append(line.rstrip())

    # Collapse the runs of blank lines that dropped rules leave behind, without
    # touching the author's own paragraph breaks.
    collapsed: List[str] = []
    for line in out:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)

    return "\n".join(collapsed).strip("\n")
