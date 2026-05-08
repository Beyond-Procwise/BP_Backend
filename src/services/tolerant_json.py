"""Tolerant JSON parser for partially-truncated LLM responses.

When ``num_predict`` runs out mid-generation, the LLM returns a stream
that ends in the middle of an object — no closing brace, sometimes a
half-written number or string. The default ``json.loads`` rejects this
entirely, even though the first 90% of the data is salvageable.

This module provides :func:`parse_tolerant` which:

  1. Strips common wrappers (markdown fences, chat-template echoes).
  2. Locates the **first** ``{`` and walks forward, balancing braces and
     brackets and tracking string state, until either:
        - it finds a clean balanced object (returns parsed),
        - or the input ends mid-token (auto-closes the structure and
          retries until a balanced view parses).
  3. Returns ``ParseResult(data, recovered, recovery_ops)`` where
     ``recovered=True`` indicates the parse needed structural repair.

The output is JSON-loadable; callers get the same shape they would have
gotten from a clean response, plus a flag they can record in logs /
provenance so we know which extractions used the recovery path.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


__all__ = ["ParseResult", "parse_tolerant"]


_FENCE_HEAD = re.compile(r"^\s*```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_TAIL = re.compile(r"\n?```\s*$")
# Some chat models leak the assistant-turn separator before the JSON.
# Examples observed in production: `]{"\\n"}{"\\n"}assistant\n{...` —
# i.e. the literal characters `]{"\n"}{"\n"}assistant\n`. The regex
# matches that drift conservatively: anything up to and including the
# word "assistant" followed by a newline.
_ASSISTANT_TURN = re.compile(r"[^{]*?assistant\s*\n", re.IGNORECASE | re.DOTALL)


@dataclass
class ParseResult:
    data: Optional[Any]
    recovered: bool = False
    recovery_ops: list[str] = field(default_factory=list)
    # Fraction of the input that survived (1.0 = clean parse, 0.0 = nothing salvaged).
    completeness: float = 1.0


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = _FENCE_HEAD.sub("", cleaned)
    cleaned = _FENCE_TAIL.sub("", cleaned)
    # Drop everything before the assistant-turn marker, if present
    # AND if there's a JSON object after it. The marker is typically
    # the literal substring "assistant\n" emitted by some chat-style
    # models when their template leaks. We strip from the start up to
    # AND INCLUDING that marker, but only when a candidate JSON object
    # follows it.
    marker = re.search(r"assistant\s*\n", cleaned, re.IGNORECASE)
    if marker:
        tail = cleaned[marker.end():].lstrip()
        if tail.startswith("{") or tail.startswith("["):
            cleaned = tail
    return cleaned


def _find_first_object_start(s: str) -> int:
    """Return the index of the first `{` that plausibly begins a JSON
    object — followed (after optional whitespace) by either a `"` (key
    starts) or `}` (empty object). This skips `[FOOTER]`-style prose
    markers and `{` that appears inside narrative text.
    Returns -1 if none found.
    """
    for m in re.finditer(r"\{", s):
        i = m.start()
        # peek past whitespace
        j = i + 1
        while j < len(s) and s[j] in " \t\r\n":
            j += 1
        if j < len(s) and s[j] in '"}':
            return i
    return -1


def _auto_close(s: str) -> tuple[str, list[str]]:
    """Walk `s` tracking braces, brackets, and string state. When the
    input ends mid-structure, append the closing characters needed to
    balance. Trims any trailing partial token (number, string, key) so
    the result is loadable.

    Returns ``(repaired, ops)`` where ``ops`` lists the recovery actions
    taken, for diagnostic logging.
    """
    ops: list[str] = []
    stack: list[str] = []  # entries: '{' or '['
    in_string = False
    escape = False
    last_safe_idx = -1  # last index where structure was valid
    n = len(s)
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
                # End of string is a safe point if not inside a value mid-token.
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack:
                # Stray closer — input is malformed before this point.
                return s[:i], ops + [f"trim_stray_closer_at_{i}"]
            opener = stack.pop()
            if (opener, ch) not in (("{", "}"), ("[", "]")):
                # Mismatched — bail at this point.
                return s[:i], ops + [f"trim_mismatched_at_{i}"]
            if not stack:
                last_safe_idx = i  # we've returned to top level
            continue
        if not stack:
            # We're outside any structure — anything here is junk after
            # the top-level object closed (or before it opened). Stop.
            if last_safe_idx >= 0:
                return s[: last_safe_idx + 1], ops
            continue

    # If we reach here, input is truncated mid-structure.
    if not stack and not in_string:
        return s, ops  # nothing to repair

    repaired = s
    if in_string:
        # Trim back to before the unclosed string opener — finding the
        # last unescaped `"` that started a string isn't reliable, so
        # we trim to the last comma or `{` before the truncation.
        cut = max(s.rfind(","), s.rfind("{"), s.rfind("["))
        if cut < 0:
            return s[:0], ops + ["abandon_truncated_string"]
        repaired = s[:cut]
        ops.append(f"trim_unclosed_string_to_{cut}")
        # Recompute stack depth on repaired prefix
        return _auto_close(repaired)
    # Trim trailing partial value (number that ends with `.`, key without
    # value, etc.). Heuristic: cut at last `,` or opener.
    trailing = repaired.rstrip()
    if trailing and trailing[-1] in ":,.":
        cut = max(repaired.rfind(","), repaired.rfind("{"), repaired.rfind("["))
        if cut > 0:
            repaired = repaired[:cut]
            ops.append(f"trim_partial_value_to_{cut}")
            return _auto_close(repaired)
    # Close remaining open braces/brackets in reverse order.
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
        ops.append("close_" + opener)
    return repaired, ops


def parse_tolerant(text: str) -> ParseResult:
    """Parse `text` as JSON, repairing common LLM-output failure modes.

    Returns a :class:`ParseResult`. ``data is None`` only when even the
    repaired input is unparsable.
    """
    if not text:
        return ParseResult(data=None, recovered=False, completeness=0.0)

    cleaned = _strip_wrappers(text)
    # Fast path
    try:
        return ParseResult(data=json.loads(cleaned), recovered=False)
    except json.JSONDecodeError:
        pass

    # Locate the first plausible object start
    start = _find_first_object_start(cleaned)
    if start < 0:
        return ParseResult(data=None, recovered=False, completeness=0.0)

    # Try a balanced regex search (greedy then shrinking)
    candidate = cleaned[start:]
    for body, ops in _shrinking_candidates(candidate):
        try:
            data = json.loads(body)
            return ParseResult(
                data=data,
                recovered=bool(ops),
                recovery_ops=ops,
                completeness=len(body) / len(candidate) if candidate else 1.0,
            )
        except json.JSONDecodeError:
            continue

    # Fallback: some fine-tuned models leak a fake multi-turn structure
    # like ``{"user": "...": {"assistant": '{"header": ..., "line_items":
    # [...]}'}}`` and then keep regenerating. The ACTUAL extraction JSON
    # lives inside as a single-quoted string after ``"assistant":``. We
    # find the first ``{"header"`` substring and balance from there.
    inner_start = re.search(r'\{\s*"header"\s*:', cleaned)
    if inner_start:
        inner = cleaned[inner_start.start():]
        for body, ops in _shrinking_candidates(inner):
            try:
                data = json.loads(body)
                return ParseResult(
                    data=data,
                    recovered=True,
                    recovery_ops=["found_inner_header_object"] + ops,
                    completeness=len(body) / len(inner) if inner else 1.0,
                )
            except json.JSONDecodeError:
                continue

    return ParseResult(data=None, recovered=True, recovery_ops=["all_attempts_failed"])


def _shrinking_candidates(s: str):
    """Yield repair candidates in order of preference."""
    # Attempt 1: as-is
    yield s, []
    # Attempt 2: trim to last `}`, simple greedy close
    last_close = s.rfind("}")
    if last_close > 0:
        yield s[: last_close + 1], ["trim_to_last_close"]
    # Attempt 3: full auto-close with state machine
    repaired, ops = _auto_close(s)
    if repaired and repaired != s:
        yield repaired, ops
    # Attempt 4: progressively trim final field and re-close
    for n_trim in (50, 200, 500, 1000):
        if len(s) > n_trim:
            shrunk = s[:-n_trim]
            cut = max(shrunk.rfind(","), shrunk.rfind("{"))
            if cut > 0:
                trimmed = shrunk[:cut]
                rep, rop = _auto_close(trimmed)
                yield rep, rop + [f"shrink_{n_trim}"]
