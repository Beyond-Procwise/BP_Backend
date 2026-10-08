"""Screening an INBOUND reply before anything is drafted against it.

Today: the payment-detail-change screen. A reply that asks for NEW or CHANGED bank or payment details is the classic invoice-redirection
fraud, and it arrives by email. The screen is deterministic (no model: a model can be talked out of it by the very email it reads),
biased toward flagging (a false alarm costs a person a glance, a miss costs a payment to a criminal), and defeats the usual disguises:
markup splitting a word, zero-width characters, spaced-out letters, capitals, odd whitespace.

It returns categories and keywords only. It never returns text from the email, so a bank detail in the reply cannot leak through it.
"""

from __future__ import annotations

import html
import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MAX_SCAN = 60_000          # characters scanned from each end of a very long mail: a phrase hidden at the end still counts
NEAR = 100                 # characters between a payment term and a change term for them to count as one request

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­᠎]")
_BLOCK_TAG = re.compile(r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6]|blockquote)\b[^>]{0,200}>", re.I)
_ANY_TAG = re.compile(r"<[^>]{0,300}>")
_SPACED_LETTERS = re.compile(r"(?<![a-z])(?:[a-z] ){2,}[a-z](?![a-z])")

# What is being changed: a payment instrument. Bare "account" is NOT one ("account manager", "cover the account").
_NOT_A_PERSON_OR_TEAM = r"(?!\s+(?:manager|executive|director|rep|representative|team|payable|receivable|management|owner|lead|officer))"
_OBJECT = re.compile(
    r"bank(?:ing)?\s+(?:account|details|information|info|instructions|transfer\s+details)"
    r"|bank\b(?!\s+holiday)"
    r"|account\s+(?:number|details|no\b)"
    r"|sort\s*code|\biban\b|\bswift\b|\bbic\b|routing\s+number|beneficiar(?:y|ies)"
    r"|remittance\s+(?:details|instructions|information)|payment\s+(?:details|instructions|information)|\bpayee\b"
    r"|wire\s+transfer|wire\s+the\s+(?:next|payment|funds)"
    r"|(?:alternative|alternate|new|different|revised|updated|other)\s+account" + _NOT_A_PERSON_OR_TEAM +
    r"|settlement\s+account"
)
# That it is being changed.
_CHANGE = re.compile(
    r"chang(?:e|ed|es|ing)\b|\bnew\b|updat(?:e|ed|es|ing)\b|revis(?:e|ed|ion)\b|replac(?:e|ed|ing|ement)\b|switch(?:ed|ing)?\b"
    r"|migrat(?:e|ed|ing)\b|amend(?:ed)?\b|alternat(?:e|ive)\b|different\b|\binstead\b|no\s+longer|going\s+forward|from\s+now\s+on"
    r"|disregard|\bmoved\b|\bclosed\b|previous\s+details|old\s+(?:account|details)"
)
_PRESSURE = re.compile(
    r"\burgent(?:ly)?\b|immediately|\basap\b|\btoday\b|right\s+away|without\s+delay|do\s+not\s+call|don'?t\s+call|do\s+not\s+contact"
    r"|do\s+not\s+verify|phones?\s+(?:are|is)\s+down|cannot\s+be\s+reached|keep\s+this\s+(?:confidential|between)|time[- ]sensitive"
)
_BANK_DETAIL = re.compile(
    r"\b[a-z]{2}\d{2}[a-z0-9]{11,30}\b"                       # IBAN-shaped
    r"|\b\d{2}-\d{2}-\d{2}\b"                                # UK sort code
    r"|account\s*(?:number|no\.?)\s*:?\s*\d{6,12}"
)


def _normalise(text: Any) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if len(text) > 2 * MAX_SCAN:
        text = text[:MAX_SCAN] + "\n" + text[-MAX_SCAN:]
    text = _ZERO_WIDTH.sub("", text)
    text = _BLOCK_TAG.sub(" ", text)
    text = _ANY_TAG.sub("", text)                              # a tag INSIDE a word is removed, so b<b></b>ank is "bank"
    text = html.unescape(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), text)      # "b a n k" -> "bank"


def _gap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, b[0] - a[1], a[0] - b[1])


def screen_payment_change(subject: Any = "", body: Any = "") -> Dict[str, Any]:
    """{"suspected", "kinds", "terms", "bank_details_present", "pressure"}. Never raises; non-text input is not flagged.

    ``suspected`` means a person must look before anything is drafted or sent against this reply.
    """

    text = _normalise(f"{subject if isinstance(subject, str) else ''}\n{body if isinstance(body, str) else ''}")
    objects = [(m.start(), m.end(), m.group(0)) for m in _OBJECT.finditer(text)]
    changes = [(m.start(), m.end(), m.group(0)) for m in _CHANGE.finditer(text)]
    kinds: Set[str] = set()
    terms: List[str] = []
    for o in objects:
        for c in changes:
            if _gap(o[:2], c[:2]) <= NEAR:
                kinds.add("payment_detail_change")
                for t in (o[2], c[2]):
                    t = t.strip()[:30]
                    if t and t not in terms:
                        terms.append(t)
    bank = bool(_BANK_DETAIL.search(text))
    pressure = bool(_PRESSURE.search(text))
    if bank and pressure:
        kinds.add("bank_details_with_pressure")
    return {"suspected": bool(kinds), "kinds": sorted(kinds), "terms": terms[:8],
            "bank_details_present": bank, "pressure": pressure}


# --- prompt injection -----------------------------------------------------------------------------------------------------------
# An inbound reply reaches a model in two places. Text in it that tries to instruct the assistant must be seen by a person first.
# Narrow on purpose: ordinary supplier mail says "please ignore my previous message", so an override needs an INSTRUCTION-class noun
# (instructions, prompts, rules, guidelines...) and an action directed at the assistant needs the assistant to be addressed.

_OBJ = r"(?:instructions?|prompts?|rules|directions|guidelines|programming|system\s+prompt|constraints)"
_PREV = r"(?:previous|prior|above|earlier|preceding|former|original)"
_OVERRIDE_PATTERNS = {
    "ignore-previous-instructions": re.compile(r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+|these\s+|those\s+|of\s+)*" + _PREV + r"\s+" + _OBJ),
    "forget-everything": re.compile(r"\bforget\s+(?:everything|all)\s+(?:you|that|above|before)"),
    "role-reset": re.compile(
        r"\byou\s+are\s+(?:now|no\s+longer)\s+(?:an?|the|free|unrestricted|in|my|operating|acting|bound)\b"
        r"|\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|should|shall)\b|\bpretend\s+(?:to\s+be|you\s+are)\b"
        r"|\bact\s+as\s+(?:an?|the)\s+(?:ai|assistant|language\s+model|chatbot|different|unrestricted|system)\b"
        r"|\bdeveloper\s+mode\b|\bjailbreak\b|\bdo\s+anything\s+now\b"),
    "reveal-prompt": re.compile(r"\b(?:reveal|print|show|output|repeat|disclose)\s+(?:your|the)\s+(?:system\s+)?prompt\b"
                                r"|\b(?:reveal|print|show|output|repeat|disclose)\s+your\s+(?:system\s+)?instructions\b|\bsystem\s+prompt\b"),
}
_SQUASHED_OVERRIDE = re.compile(r"(?:ignore|disregard|forget|override|bypass)(?:all|any|the|your|my|these|those|of)*" + _PREV.replace("\\", "")
                                + r"(?:instructions?|prompts?|rules|directions|guidelines|programming|systemprompt|constraints)")
_ROLE_MARKERS = {
    "role-marker-tag": re.compile(r"<\s*/?\s*(?:system|assistant|instructions?|prompt)\s*>"),
    "role-marker-bracket": re.compile(r"\[\s*/?\s*(?:inst|sys|system)\s*\]|<<\s*/?\s*sys\s*>>"),
    "role-marker-token": re.compile(r"<\|[a-z_]+\|>"),
    "role-marker-heading": re.compile(r"(?m)^\s*#{2,}\s*(?:system|instructions?)\b"),
}
_AI_TERMS = re.compile(r"\b(?:assistant|ai|chatbot|llm|language\s+model|gpt|claude|bot|automated\s+(?:system|agent|assistant|reader)|procwise|procurement\s+agent)\b")
_DIRECTED = {
    "directed-forward": re.compile(r"\b(?:forward|send|copy|bcc|cc|email|share)\b[^.!?]{0,80}?\bto\s+\S+@\S+"),
    "directed-recipient": re.compile(r"change\s+the\s+recipients?\b|(?:reply|respond)\s+(?:only\s+)?to\s+(?:this|the\s+following)\s+address|use\s+this\s+(?:email\s+)?address\s+instead"),
}
_HIDDEN_ELEMENT = re.compile(
    r"<(\w+)\b[^>]*\bstyle\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d])|opacity\s*:\s*0(?![.\d]))[^\"']*[\"'][^>]*>(.*?)</\1\s*>",
    re.I | re.S)
_COMMENT = re.compile(r"<!--(.*?)-->", re.S)


def _light(text: Any) -> str:
    """Zero-width characters and entities resolved, lower-cased, TAGS KEPT (a fake <system> tag is itself a signal)."""
    if not isinstance(text, str) or not text:
        return ""
    if len(text) > 2 * MAX_SCAN:
        text = text[:MAX_SCAN] + "\n" + text[-MAX_SCAN:]
    return html.unescape(_ZERO_WIDTH.sub("", text)).lower()


def _instruction_hits(norm: str, light: str) -> Dict[str, str]:
    """{pattern id: kind} for every instruction-class signal in the text. ``norm`` has tags stripped, ``light`` has them."""
    hits: Dict[str, str] = {}
    for pid, rx in _OVERRIDE_PATTERNS.items():
        if rx.search(norm):
            hits[pid] = "instruction_override"
    if "ignore-previous-instructions" not in hits and _SQUASHED_OVERRIDE.search(re.sub(r"[^a-z]", "", norm)):
        hits["ignore-previous-instructions"] = "instruction_override"      # words spelled out letter by letter merge into one blob
    for pid, rx in _ROLE_MARKERS.items():
        if rx.search(light):
            hits[pid] = "role_marker"
    if _AI_TERMS.search(norm):
        for pid, rx in _DIRECTED.items():
            if rx.search(norm):
                hits[pid] = "assistant_directed_action"
    return hits


def _hidden_segments(raw: str) -> List[str]:
    out: List[str] = []
    for m in _HIDDEN_ELEMENT.finditer(raw):
        out.append(m.group(2))
    for m in _COMMENT.finditer(raw):
        inner = m.group(1).strip()
        if inner.lower().startswith(("[if", "<![endif", "procwise_marker")) or not inner:
            continue                                                       # conditional comments and our own tracking marker
        out.append(inner)
    return out


def screen_injection(subject: Any = "", body: Any = "", html: Any = None) -> Dict[str, Any]:
    """{"suspected", "kinds", "terms", "hidden_text"}: does this reply try to instruct the assistant? Never raises.

    ``terms`` are pattern ids (e.g. ``ignore-previous-instructions``), never text from the email. ``hidden_text`` says invisible text was
    present at all; it is a reason to suspect only when that text itself carries an instruction.
    """

    raw = "\n".join(x for x in (subject, body, html) if isinstance(x, str) and x)
    norm, light = _normalise(raw), _light(raw)
    hits = _instruction_hits(norm, light)
    kinds = set(hits.values())
    terms = sorted(hits)
    segments = _hidden_segments(light)
    if segments:
        hidden = " ".join(segments)
        hidden_hits = _instruction_hits(_normalise(hidden), hidden)
        if hidden_hits:
            kinds.add("hidden_instruction")
            terms = sorted(set(terms) | {"hidden-text"})
    return {"suspected": bool(kinds), "kinds": sorted(kinds), "terms": terms[:8], "hidden_text": bool(segments)}


# --- recording what the screen found --------------------------------------------------------------------------------------------

class FlagLookupFailed(RuntimeError):
    """The flags could not be read. A caller that must be safe treats this as 'blocked', never as 'nothing flagged'."""


@contextmanager
def _default_factory() -> Iterator[Any]:
    from src.services.db import get_conn

    with get_conn() as conn:
        yield conn


def record_flag(conn: Any, *, workflow_id: Optional[str], unique_id: Optional[str], supplier_id: Optional[str],
                message_id: Optional[str], result: Dict[str, Any], kind: str = "payment_detail_change") -> Optional[int]:
    """Store WHICH signals fired and the keywords, by pointer to the message. Returns the flag id, or None if already flagged."""

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO email_agent.bp_inbound_flag (kind, workflow_id, unique_id, supplier_id, response_message_id, kinds, terms)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
               ON CONFLICT (kind, workflow_id, response_message_id) WHERE response_message_id IS NOT NULL DO NOTHING
               RETURNING flag_id""",
            (kind, workflow_id, unique_id, supplier_id, message_id, json.dumps(result["kinds"]), json.dumps(result["terms"])))
        got = cur.fetchone()
    return int(got[0]) if got else None


def _record_screen(row: Any, screen: Callable[[Any], Dict[str, Any]], kind: str, label: str,
                   conn_factory: Optional[Callable[[], Any]]) -> Optional[int]:
    """Run ``screen`` on a stored reply and flag it as ``kind`` if suspected. NEVER raises and never blocks the ingest: this runs inside
    the path that stores a supplier's reply, and a fault here must cost a missed flag at worst (logged), not a lost reply."""

    try:
        if row is None:
            return None
        result = screen(row)
        if not result["suspected"]:
            return None
        with (conn_factory or _default_factory)() as conn:
            return record_flag(conn, workflow_id=getattr(row, "workflow_id", None), unique_id=getattr(row, "unique_id", None),
                               supplier_id=getattr(row, "supplier_id", None), message_id=getattr(row, "response_message_id", None),
                               result=result, kind=kind)
    except Exception as exc:  # noqa: BLE001
        # An absent email_agent schema is the normal state before the pack is applied: say so quietly, not as an error.
        if "bp_inbound_flag" in str(exc) and ("does not exist" in str(exc) or "UndefinedTable" in type(exc).__name__):
            logger.debug("inbound flag not recorded: the email_agent schema is not applied")
        else:
            logger.exception("inbound %s screen failed; the reply was stored regardless", label)
        return None


def _field(row: Any, name: str) -> str:
    return str(getattr(row, name, None) or "")


def screen_and_record(row: Any, conn_factory: Optional[Callable[[], Any]] = None) -> Optional[int]:
    """Screen an inbound reply for a request to change payment details and, if suspected, flag it."""

    def screen(r: Any) -> Dict[str, Any]:
        text = "\n".join(_field(r, k) for k in ("response_text", "response_body", "body_html"))
        return screen_payment_change(getattr(r, "response_subject", None), text)

    return _record_screen(row, screen, "payment_detail_change", "payment-change", conn_factory)


def screen_injection_and_record(row: Any, conn_factory: Optional[Callable[[], Any]] = None) -> Optional[int]:
    """Screen an inbound reply for text that tries to instruct the assistant and, if suspected, flag it. The flag holds fixed pattern
    ids and the kinds, never the text."""

    def screen(r: Any) -> Dict[str, Any]:
        body = _field(r, "response_text") or _field(r, "response_body")
        return screen_injection(getattr(r, "response_subject", None), body, _field(r, "body_html") or None)

    return _record_screen(row, screen, INJECTION_FLAG, "injection", conn_factory)


INJECTION_FLAG = "injection_suspected"
_INJECTION_KINDS = {"instruction_override", "role_marker", "assistant_directed_action", "hidden_instruction"}
_AUTH_KINDS = {"auth_failed", "auth_missing", "domain_mismatch"}


def block_phrase(flags: List[Dict[str, Any]]) -> str:
    """What the standing flags say about the reply, in words, so a refusal names the real reason (not always a payment change)."""

    kinds = {k for f in flags for k in (f.get("kinds") or [])}
    parts = []
    if kinds - _INJECTION_KINDS - _AUTH_KINDS or not kinds:
        parts.append("asks to change payment details")
    if kinds & _INJECTION_KINDS:
        parts.append("contains text that tries to instruct the assistant")
    if kinds & _AUTH_KINDS:
        parts.append("could not be verified as coming from the supplier")
    return " and ".join(parts)


def blocking_flags(conn: Any, workflow_id: Optional[str], supplier_id: Optional[str]) -> List[Dict[str, Any]]:
    """Flags that stop anything being drafted or sent on this supplier's thread: open and confirmed_fraud (not cleared).

    A flag with no supplier blocks the whole workflow (we cannot say who wrote it). An absent flag table means nothing can have been
    flagged, so nothing blocks. Any OTHER failure raises ``FlagLookupFailed``: the caller must treat that as blocked.
    """

    if not workflow_id:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('email_agent.bp_inbound_flag')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                """SELECT flag_id, status, kinds, response_message_id, created_at FROM email_agent.bp_inbound_flag
                   WHERE workflow_id = %s AND status <> 'cleared' AND (supplier_id IS NULL OR supplier_id IS NOT DISTINCT FROM %s)
                   ORDER BY flag_id""", (workflow_id, supplier_id))
            return [{"id": r[0], "status": r[1], "kinds": r[2], "message_id": r[3], "created_at": r[4]} for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        raise FlagLookupFailed(f"{type(exc).__name__}: {exc}") from exc


def decide_flag(conn: Any, flag_id: int, by: Optional[str], action: str, note: Optional[str] = None) -> Dict[str, Any]:
    """A person clears a flag (lifting the block) or confirms it as fraud (keeping it). From 'open' only, once, by a named person."""

    if action not in ("clear", "confirm") or not isinstance(by, str) or not by.strip():
        return {"ok": False, "error": "action must be clear or confirm, by a named person"}
    status = "cleared" if action == "clear" else "confirmed_fraud"
    with conn.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_inbound_flag SET status = %s, decided_by = %s, decided_at = now(), note = %s "
                    "WHERE flag_id = %s AND status = 'open' RETURNING flag_id",
                    (status, by.strip(), (note or "").strip()[:500] or None, int(flag_id)))
        row = cur.fetchone()
        if row:
            return {"ok": True}
        cur.execute("SELECT 1 FROM email_agent.bp_inbound_flag WHERE flag_id = %s", (int(flag_id),))
        exists = cur.fetchone() is not None
    return {"ok": False, "error": "this flag has already been decided" if exists else "no such flag"}
