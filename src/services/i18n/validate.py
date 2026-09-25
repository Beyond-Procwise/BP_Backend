"""What a translation must preserve: placeholders, ICU structure, markup, link targets.

ICU messages are PARSED, not pattern-matched. A branch body ({# deals}) is text to be
translated, not a placeholder, and a language may legitimately need more plural branches
than English (Russian adds few/many). What must hold:

  * the same arguments, with the same types ({n, plural}, {name}, {d, date}), at any depth;
  * select keywords unchanged (they are keys the app passes in, not words);
  * plural selectors drawn from CLDR's six categories (zero/one/two/few/many/other) or =N,
    always including `other`, which the formatter requires. A category the TARGET language
    does not use (Japanese `one`) is accepted: the formatter never selects it, and AgentNick
    keeps it however the prompt is worded. A translated keyword (`uno`) is refused;
  * `#` kept if the source used it (it is where the count is printed).

printf conversions and HTML tags are compared as multisets over the whole string:
dropping one of two %s loses a value.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional

_PLURAL_TYPES = frozenset({"plural", "selectordinal"})
_CHOICE_TYPES = _PLURAL_TYPES | {"select"}
_ALL_CATEGORIES = frozenset({"zero", "one", "two", "few", "many", "other"})
_ID = re.compile(r"\s*([A-Za-z_][\w.]*|\d+)\s*")
_SELECTOR = re.compile(r"\s*(offset:\s*\d+\s+)?(=\d+|[\w-]+)\s*")

# No space flag: "50% discount" must not read as "% d".
_PRINTF = re.compile(r"%(?:\(\w+\)|\d+\$)?[-+0#]*(?:\d+|\*)?(?:\.\d+)?[sdifuxXeEgGcr]")
_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][\w-]*)[^<>]*>")
_URL_ATTR = re.compile(r"""\b(href|src)\s*=\s*("[^"]*"|'[^']*')""")


class ICUError(ValueError):
    pass


def _parse_message(s: str, i: int, depth: int, args: list, current: Optional[dict]) -> int:
    """Parse message text from i; return the index of the closing '}' (depth>0) or len(s)."""
    while i < len(s):
        c = s[i]
        if c == "{":
            i = _parse_arg(s, i + 1, depth, args, current)
        elif c == "}":
            if depth == 0:
                raise ICUError("unbalanced '}'")
            return i
        else:
            if c == "#" and current is not None:
                current["hash"] += 1
            i += 1
    if depth:
        raise ICUError("unclosed '{'")
    return i


def _parse_arg(s: str, i: int, depth: int, args: list, current: Optional[dict]) -> int:
    m = _ID.match(s, i)
    if not m:
        raise ICUError(f"bad argument at {i}")
    arg = {"name": m.group(1), "type": None, "selectors": [], "hash": 0}
    args.append(arg)
    i = m.end()
    if i < len(s) and s[i] == "}":
        return i + 1
    if i >= len(s) or s[i] != ",":
        raise ICUError(f"expected ',' or '}}' at {i}")
    m = _ID.match(s, i + 1)
    if not m:
        raise ICUError(f"bad argument type at {i}")
    arg["type"] = m.group(1)
    i = m.end()
    if i < len(s) and s[i] == "}":
        return i + 1
    if i >= len(s) or s[i] != ",":
        raise ICUError(f"expected ',' or '}}' at {i}")
    i += 1
    if arg["type"] not in _CHOICE_TYPES:  # {d, date, short}: a style, no nested message
        end = s.find("}", i)
        if end < 0 or "{" in s[i:end]:
            raise ICUError("bad argument style")
        return end + 1
    counts_hash = arg if arg["type"] in _PLURAL_TYPES else current
    while True:
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            raise ICUError("unclosed choice argument")
        if s[i] == "}":
            if not arg["selectors"]:
                raise ICUError("choice argument with no branches")
            return i + 1
        m = _SELECTOR.match(s, i)
        if not m or m.end() >= len(s) or s[m.end()] != "{":
            raise ICUError(f"bad selector at {i}")
        arg["selectors"].append(m.group(2))
        i = _parse_message(s, m.end() + 1, depth + 1, args, counts_hash)
        i += 1  # the branch's closing '}'


def parse_icu(text: str) -> list[dict]:
    args: list[dict] = []
    _parse_message(text, 0, 0, args, None)
    return args


def _check_icu(source: str, translated: str, lang: Optional[str]) -> Optional[str]:
    try:
        src = parse_icu(source)
    except ICUError:
        return None  # prose with stray braces: nothing structural to hold the translation to
    try:
        dst = parse_icu(translated)
    except ICUError as exc:
        return f"ICU syntax broken: {exc}"
    src_sig = {(a["name"], a["type"]) for a in src}
    dst_sig = {(a["name"], a["type"]) for a in dst}
    if src_sig != dst_sig:
        return f"arguments changed: {sorted(src_sig, key=str)} -> {sorted(dst_sig, key=str)}"
    for a in src:
        if a["type"] not in _CHOICE_TYPES:
            continue
        same = [b for b in dst if b["name"] == a["name"] and b["type"] == a["type"]]
        if a["type"] == "select":
            if any(set(b["selectors"]) != set(a["selectors"]) for b in same):
                return f"select keywords of {{{a['name']}}} changed"
            continue
        for b in same:
            if "other" not in b["selectors"]:
                return f"plural {{{a['name']}}} has no 'other' branch"
            bad = [x for x in b["selectors"] if not x.startswith("=") and x not in _ALL_CATEGORIES]
            if bad:
                return f"plural {{{a['name']}}} uses {bad}, which are not plural categories"
            if a["hash"] and not b["hash"]:
                return f"plural {{{a['name']}}} lost '#'"
    return None


def placeholders(text: str) -> Counter:
    rest: Counter = Counter()
    rest.update(_PRINTF.findall(text.replace("%%", "")))
    rest.update(f"<{slash}{name.lower()}>" for slash, name in _TAG.findall(text))
    rest.update(f"{attr}={val[1:-1]}" for attr, val in _URL_ATTR.findall(text))  # quote style is not content
    return rest


def check_pair(source: str, translated: str, lang: Optional[str] = None) -> str | None:
    if source.strip() and not translated.strip():
        return "empty translation"
    reason = _check_icu(source, translated, lang)
    if reason:
        return reason
    sr, tr = placeholders(source), placeholders(translated)
    if sr != tr:
        return f"placeholders/markup changed: {dict(sr)} -> {dict(tr)}"
    return None


_CONFIDENCE = ("high", "medium", "low")


def _strings_of(data: dict) -> dict:
    """The translations in a reply: {"strings": {...}} (prompt v3+) or the whole flat object."""
    inner = data.get("strings")
    return inner if isinstance(inner, dict) else data


def read_flags(raw: str | None) -> tuple[Optional[bool], Optional[str]]:
    """(lang_recognized, confidence) from a reply; None for each one it does not carry."""
    try:
        data = json.loads(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    recognized = data.get("lang_recognized")
    confidence = data.get("confidence")
    return (recognized if isinstance(recognized, bool) else None,
            confidence if confidence in _CONFIDENCE else None)


def validate_batch(sent: dict[str, str], raw: str | None,
                   lang: Optional[str] = None) -> tuple[dict[str, str], dict[str, str]]:
    if raw is None:
        return {}, {k: "no response from the model" for k in sent}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}, {k: "response was not JSON" for k in sent}
    if not isinstance(data, dict):
        return {}, {k: "response was not a JSON object" for k in sent}
    data = _strings_of(data)
    good: dict[str, str] = {}
    bad: dict[str, str] = {}
    for key, source in sent.items():
        if key not in data:
            bad[key] = "missing from response"
            continue
        value = data[key]
        if not isinstance(value, str):
            bad[key] = "value was not a string"
            continue
        reason = check_pair(source, value, lang)
        if reason:
            bad[key] = reason
        else:
            good[key] = value
    return good, bad
