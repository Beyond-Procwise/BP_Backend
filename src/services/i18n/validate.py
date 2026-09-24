"""What a translation must preserve: placeholders, markup, link targets. Nothing else.

Brace arguments ({name}, ICU {n, plural, ...}) are compared as a SET: a language with more
plural categories repeats {n} or # more often, legitimately. printf conversions and HTML tags
are compared as multisets: dropping one of two %s loses a value.
"""
from __future__ import annotations

import json
import re
from collections import Counter

_BRACE = re.compile(r"\{\s*([A-Za-z_][\w.]*|\d+)\s*(?:,\s*([a-z]+)\s*)?[,}]")
# No space flag: "50% discount" must not read as "% d".
_PRINTF = re.compile(r"%(?:\(\w+\)|\d+\$)?[-+0#]*(?:\d+|\*)?(?:\.\d+)?[sdifuxXeEgGcr]")
_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][\w-]*)[^<>]*>")
_URL_ATTR = re.compile(r"""\b(href|src)\s*=\s*("[^"]*"|'[^']*')""")


def placeholders(text: str) -> tuple[frozenset[str], Counter]:
    braces = frozenset(
        "{" + name + ("," + kind if kind else "") + "}" for name, kind in _BRACE.findall(text)
    )
    rest: Counter = Counter()
    rest.update(_PRINTF.findall(text.replace("%%", "")))
    rest.update(f"<{slash}{name.lower()}>" for slash, name in _TAG.findall(text))
    rest.update(f"{attr}={val}" for attr, val in _URL_ATTR.findall(text))
    return braces, rest


def check_pair(source: str, translated: str) -> str | None:
    if source.strip() and not translated.strip():
        return "empty translation"
    sb, sr = placeholders(source)
    tb, tr = placeholders(translated)
    if sb != tb:
        return f"brace placeholders changed: {sorted(sb)} -> {sorted(tb)}"
    if sr != tr:
        return f"placeholders/markup changed: {dict(sr)} -> {dict(tr)}"
    return None


def validate_batch(sent: dict[str, str], raw: str | None) -> tuple[dict[str, str], dict[str, str]]:
    if raw is None:
        return {}, {k: "no response from the model" for k in sent}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}, {k: "response was not JSON" for k in sent}
    if not isinstance(data, dict):
        return {}, {k: "response was not a JSON object" for k in sent}
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
        reason = check_pair(source, value)
        if reason:
            bad[key] = reason
        else:
            good[key] = value
    return good, bad
