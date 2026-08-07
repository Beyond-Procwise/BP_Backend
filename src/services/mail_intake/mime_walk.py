"""Parse raw MIME into a bundle: headers, bodies, and every attachment part.

Two decisions here decide whether a real quote is ever seen.

**Nesting.** A supplier quote is frequently one or two levels down a forwarded
chain, wrapped in message/rfc822 parts. Walking only the top level finds the
covering note and nothing else. Recursion is bounded (default 3) because depth
is attacker-controlled and unbounded recursion on hostile input is a denial of
service.

**Inline versus CID.** Signature logos and tracking pixels arrive as inline
parts referenced from the HTML body by Content-ID. Those are not documents.
But an inline part that nothing references usually IS a document from a sender
whose client is misconfigured, and dropping every inline part loses it. So the
rule is "referenced by CID", not "inline".

Spec §6 stages 2 and 5.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Dict, List, Optional, Tuple

MAX_DEPTH_DEFAULT = 3

_CID_REF = re.compile(r"""["'(]\s*cid:([^"')\s>]+)""", re.IGNORECASE)


class MalformedMime(Exception):
    """The message could not be parsed into parts.

    Raised rather than returning what was recoverable: spec §1 forbids partial
    ingestion, and half a quote is worse than a quarantine record.
    """


@dataclass(frozen=True)
class ParsedPart:
    filename: Optional[str]
    declared_mime: str
    content: bytes
    disposition: str
    content_id: Optional[str]
    nesting_depth: int


@dataclass(frozen=True)
class ParsedBundle:
    headers: Dict[str, List[str]]
    body_text: Optional[str]
    body_html: Optional[str]
    parts: Tuple[ParsedPart, ...]


def _normalise_cid(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().strip("<>").strip()


def _collect_headers(message: EmailMessage) -> Dict[str, List[str]]:
    headers: Dict[str, List[str]] = {}
    for name, value in message.items():
        headers.setdefault(name.lower(), []).append(str(value))
    return headers


def _walk(message: EmailMessage, depth: int, max_depth: int,
          parts: List[ParsedPart], bodies: Dict[str, str]) -> None:
    for part in message.iter_parts() if message.is_multipart() else ():
        content_type = part.get_content_type()

        if content_type == "message/rfc822":
            if depth + 1 > max_depth:
                continue
            nested = part.get_payload(0)
            _walk(nested, depth + 1, max_depth, parts, bodies)
            continue

        if part.is_multipart():
            _walk(part, depth, max_depth, parts, bodies)
            continue

        disposition = (part.get_content_disposition() or "").lower()
        # Body parts belong to the bundle, not to the artifact list — but only
        # at the outermost level. A forwarded message's own covering note is
        # context, not this bundle's body.
        if disposition in ("", "inline") and content_type in (
            "text/plain",
            "text/html",
        ) and part.get_filename() is None:
            key = "text" if content_type == "text/plain" else "html"
            if depth == 0 and key not in bodies:
                bodies[key] = part.get_content()
            continue

        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        parts.append(
            ParsedPart(
                filename=part.get_filename(),
                declared_mime=content_type,
                content=payload,
                disposition=disposition or "attachment",
                content_id=_normalise_cid(part.get("Content-ID")),
                nesting_depth=depth,
            )
        )


def parse(raw: bytes, *, max_depth: int = MAX_DEPTH_DEFAULT) -> ParsedBundle:
    """Parse raw MIME bytes. Raises MalformedMime rather than partly succeeding."""
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise MalformedMime(str(exc)) from exc

    if message.defects:
        raise MalformedMime(
            "; ".join(type(d).__name__ for d in message.defects)
        )

    parts: List[ParsedPart] = []
    bodies: Dict[str, str] = {}

    if message.is_multipart():
        _walk(message, 0, max_depth, parts, bodies)
        if not parts and not bodies:
            raise MalformedMime("multipart message yielded no parts")
    else:
        try:
            bodies["text"] = message.get_content()
        except Exception as exc:
            raise MalformedMime(str(exc)) from exc

    return ParsedBundle(
        headers=_collect_headers(message),
        body_text=bodies.get("text"),
        body_html=bodies.get("html"),
        parts=tuple(parts),
    )


def drop_cid_referenced(bundle: ParsedBundle) -> Tuple[ParsedPart, ...]:
    """Drop inline parts the HTML body references by Content-ID.

    Those are signature logos and tracking pixels. An inline part that nothing
    references is retained — see the module docstring.
    """
    referenced = {
        _normalise_cid(cid) for cid in _CID_REF.findall(bundle.body_html or "")
    }
    return tuple(
        part
        for part in bundle.parts
        if not (part.disposition == "inline" and part.content_id in referenced)
    )
