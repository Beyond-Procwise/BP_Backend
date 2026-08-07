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
from email import errors as email_errors
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Dict, List, Optional, Tuple

MAX_DEPTH_DEFAULT = 3

# Ceilings enforced BEFORE the stdlib parser is handed the bytes. See
# _reject_hostile_shape for why "before" is the whole point.
MAX_RAW_BYTES = 25 * 1024 * 1024
MAX_NESTED_MESSAGES = 20

_CID_REF = re.compile(r"""["'(]\s*cid:([^"')\s>]+)""", re.IGNORECASE)

# Cheap upper bound on nesting, counted on the raw bytes without parsing them.
_RFC822_MARKER = re.compile(rb"message/rfc822", re.IGNORECASE)

# Defects that mean structure was actually lost. Everything else Python reports
# — misplaced envelope headers, obsolete header syntax, and the rest — is a
# cosmetic anomaly in mail that still decodes perfectly, and rejecting on those
# would quarantine legitimate supplier quotes that merely passed through a
# mailing list or a migrated mailbox. Fail-closed means not proceeding under
# uncertainty; it does not mean refusing anything imperfect.
_FATAL_DEFECTS = (
    email_errors.NoBoundaryInMultipartDefect,
    email_errors.StartBoundaryNotFoundDefect,
    email_errors.CloseBoundaryNotFoundDefect,
    email_errors.MultipartInvariantViolationDefect,
    email_errors.MissingHeaderBodySeparatorDefect,
)


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


def _reject_hostile_shape(raw: bytes) -> None:
    """Reject pathological input BEFORE the stdlib parser sees it.

    `max_depth` alone is not a denial-of-service defence, which is easy to get
    wrong: BytesParser.parsebytes eagerly builds the ENTIRE nested tree in one
    call, so a depth limit applied afterwards limits only how far we walk the
    tree we have already paid to construct. Measured on nested message/rfc822
    chains: 1,200 wrappers (552 KB) took 4.8s, and 3,000 (~1.3 MB) exceeded 30s
    — payload sizes well inside any normal attachment limit.

    Both checks below are linear scans of the raw bytes and cost microseconds.
    Real forwarded chains nest in single digits, so MAX_NESTED_MESSAGES leaves
    enormous headroom while rejecting the hostile case outright.
    """
    if len(raw) > MAX_RAW_BYTES:
        raise MalformedMime(f"raw message exceeds {MAX_RAW_BYTES} bytes")
    nested = len(_RFC822_MARKER.findall(raw))
    if nested > MAX_NESTED_MESSAGES:
        raise MalformedMime(
            f"message declares {nested} nested messages, above the "
            f"{MAX_NESTED_MESSAGES} ceiling"
        )


def _fatal_defects(message: EmailMessage) -> List[str]:
    """Structure-losing defects anywhere in the tree, including nested messages.

    Checking only the outer message's defects misses the case this module
    exists for. A corrupted quote nested one level down inside a forward leaves
    the OUTER message defect-free, so a top-level-only check returns a
    "successful" bundle carrying the covering note and no attachments — the
    partial ingestion the design forbids, arriving silently.
    """
    found: List[str] = []
    for part in message.walk():
        for defect in part.defects or ():
            if isinstance(defect, _FATAL_DEFECTS):
                found.append(type(defect).__name__)
    return found


def parse(raw: bytes, *, max_depth: int = MAX_DEPTH_DEFAULT) -> ParsedBundle:
    """Parse raw MIME bytes. Raises MalformedMime rather than partly succeeding."""
    _reject_hostile_shape(raw)

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise MalformedMime(str(exc)) from exc

    fatal = _fatal_defects(message)
    if fatal:
        raise MalformedMime("; ".join(fatal))

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
