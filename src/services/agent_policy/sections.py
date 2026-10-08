"""Split a policy document into sections, pack them for the model, and diff two versions.

Pure text handling: no I/O, no model, no database.

A section starts at a markdown heading line (``# Title``) or a numbered clause line
(``1.``, ``1.1``, ``4.2)``). Every character of the input belongs to exactly one section,
and the sections concatenate back to the input.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TypedDict

_HEADING = re.compile(r"^#{1,6} +(?P<title>.*\S)?")
# A clause line is one of: a dotted number ("1.1 ", "4.2) "), a number with a "." or ")"
# ("1. ", "4) "), or a bare number then an UPPERCASE word ("1 Refunds"). Prose that merely
# starts with a number ("2026 budget", "3 days notice") is not a clause: references drive
# stable-ID matching across revisions, so a false marker is load-bearing.
_CLAUSE = re.compile(
    r"^\s*(?P<num>"
    r"\d{1,3}(?:\.\d{1,3})+(?=[.)]?\s+\S)"
    r"|\d{1,3}(?=[.)]\s+\S)"
    r"|\d{1,3}(?=\s+[A-Z])"
    r")")

DEFAULT_MAX_CHARS = 9000


class Section(TypedDict):
    reference: Optional[str]
    heading: str
    text: str
    start: int


def _marker(line: str) -> Optional[Dict[str, str]]:
    m = _HEADING.match(line)
    if m:
        title = (m.group("title") or line.strip("# \r\n")).strip()
        return {"reference": title, "heading": line.strip()}
    m = _CLAUSE.match(line)
    if m:
        return {"reference": m.group("num"), "heading": line.strip()}
    return None


def split_sections(text: str) -> List[Section]:
    """Split ``text`` into sections; ``"".join(s["text"] for s in result) == text``."""
    if not text:
        return []
    sections: List[Section] = []
    offset = 0
    current: Optional[Section] = None
    for line in text.splitlines(keepends=True):
        marker = _marker(line)
        if marker is not None:
            if current is not None:
                sections.append(current)
            current = {"reference": marker["reference"], "heading": marker["heading"],
                       "text": line, "start": offset}
        elif current is None:
            current = {"reference": None, "heading": "Preamble", "text": line, "start": offset}
        else:
            current["text"] += line
        offset += len(line)
    if current is not None:
        sections.append(current)
    return sections


def _pieces(text: str, max_chars: int) -> List[str]:
    """Cut ``text`` into pieces of at most ``max_chars`` that concatenate back to it,
    preferring paragraph boundaries, then lines, then raw characters."""
    paragraphs = re.findall(r".*?(?:\n[ \t]*\n+|\Z)", text, flags=re.S)
    atoms: List[str] = []
    for para in paragraphs:
        if not para:
            continue
        if len(para) <= max_chars:
            atoms.append(para)
            continue
        for line in para.splitlines(keepends=True):
            if len(line) <= max_chars:
                atoms.append(line)
            else:
                atoms.extend(line[i:i + max_chars] for i in range(0, len(line), max_chars))
    pieces: List[str] = []
    buf = ""
    for atom in atoms:
        if buf and len(buf) + len(atom) > max_chars:
            pieces.append(buf)
            buf = ""
        buf += atom
    if buf:
        pieces.append(buf)
    return pieces


def _split_oversize(section: Section, max_chars: int) -> List[Section]:
    pieces = _pieces(section["text"], max_chars)
    if len(pieces) <= 1:
        return [section]
    parts: List[Section] = []
    start = section["start"]
    for n, piece in enumerate(pieces, 1):
        suffix = f" (part {n})"
        parts.append({
            "reference": None if section["reference"] is None else section["reference"] + suffix,
            "heading": section["heading"] + suffix,
            "text": piece,
            "start": start,
        })
        start += len(piece)
    return parts


def chunk_sections(sections: List[Section], *, max_chars: int = DEFAULT_MAX_CHARS) -> List[List[Section]]:
    """Pack consecutive sections greedily into chunks of at most ``max_chars``.

    A section longer than ``max_chars`` is split on paragraph boundaries into parts that
    carry the same reference with `` (part n)`` appended. Nothing is dropped or duplicated.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: List[List[Section]] = []
    current: List[Section] = []
    size = 0
    for section in sections:
        for part in _split_oversize(section, max_chars):
            length = len(part["text"])
            if current and size + length > max_chars:
                chunks.append(current)
                current, size = [], 0
            current.append(part)
            size += length
    if current:
        chunks.append(current)
    return chunks


def _keyed(text: str) -> Dict[str, Section]:
    keyed: Dict[str, Section] = {}
    seen: Dict[str, int] = {}
    for section in split_sections(text):
        base = section["reference"] or section["heading"]
        seen[base] = seen.get(base, 0) + 1
        key = base if seen[base] == 1 else f"{base} #{seen[base]}"
        keyed[key] = section
    return keyed


def diff_sections(old_text: str, new_text: str) -> List[Dict[str, Any]]:
    """Align two versions by reference (heading when there is none) for a before/after view."""
    old, new = _keyed(old_text), _keyed(new_text)
    order: List[str] = list(old)
    last: Optional[str] = None
    for key in new:
        if key not in old:
            idx = order.index(last) + 1 if last in order else 0
            order.insert(idx, key)
        last = key
    out: List[Dict[str, Any]] = []
    for key in order:
        before = old[key]["text"] if key in old else None
        after = new[key]["text"] if key in new else None
        if before is None:
            status = "added"
        elif after is None:
            status = "removed"
        elif before.strip() == after.strip():
            status = "unchanged"
        else:
            status = "changed"
        out.append({"reference": key, "status": status, "before": before, "after": after})
    return out
