"""Shaping what goes into a supplier email, short of writing it.

Recipient lists, subject lines, the greeting test, quoted history folded into an
HTML body. None of this decides what to say — prose.py does that — and none of it
sends anything.

Every function here was a method on NegotiationAgent that never touched `self`.
They are unchanged apart from losing that argument.
"""
from __future__ import annotations

import re
from html import escape
from typing import Any, Dict, List, Optional, Sequence, Set

from agents.base_agent import AgentContext
from agents.email_drafting_agent import EmailDraftingAgent


def normalise_base_subject(subject: Optional[str]) -> Optional[str]:
    if subject is None:
        return None
    if not isinstance(subject, str):
        try:
            subject = str(subject)
        except Exception:
            return None
    trimmed = subject.strip()
    if not trimmed:
        return None
    trimmed = re.sub(r"(?i)^(re|fw|fwd):\s*", "", trimmed)
    cleaned = EmailDraftingAgent._strip_rfq_identifier_tokens(trimmed)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("-–: ")
    return cleaned or None


def collect_supplier_snippets(payload: Dict[str, Any]) -> List[str]:
    snippets: List[str] = []
    for key in (
        "supplier_snippets",
        "snippets",
        "highlights",
        "supplier_highlights",
        "response_text",
        "message",
        "raw_email",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            snippets.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            snippets.append(value.strip())
    return snippets[:5]


def is_likely_identifier(value: str) -> bool:
    token = value.strip()
    if not token:
        return False
    return bool(re.match(r"^[A-Z]{2,}[A-Z0-9._-]*$", token))


def has_explicit_greeting(message: Optional[str]) -> bool:
    if not message:
        return False
    snippet = message.lstrip()
    lowered = snippet.lower()
    greeting_prefixes = (
        "dear ",
        "hi ",
        "hello ",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    )
    return any(lowered.startswith(prefix) for prefix in greeting_prefixes)


def simple_html_from_text(text: str) -> str:
    lines = text.splitlines()
    html_parts: List[str] = []
    bullets: List[str] = []

    def flush() -> None:
        if bullets:
            items = "".join(f"<li>{escape(item)}</li>" for item in bullets)
            html_parts.append(f"<ul>{items}</ul>")
            bullets.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if re.match(r"^[-*•]\s+", stripped):
            bullets.append(stripped[1:].strip())
            continue
        flush()
        html_parts.append(f"<p>{escape(stripped)}</p>")
    flush()
    return "".join(html_parts)


def normalise_recipient_list(value: Any) -> List[str]:
    if value is None:
        return []
    candidates: List[str] = []
    if isinstance(value, str):
        tokens = re.split(r"[;,]", value)
        candidates.extend(token.strip() for token in tokens if token.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            if isinstance(item, str):
                tokens = re.split(r"[;,]", item)
                candidates.extend(token.strip() for token in tokens if token.strip())
    return candidates


def merge_recipients_basic(to_list: List[str], cc_list: List[str]) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for addr in list(to_list) + list(cc_list):
        candidate = addr.strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return merged


def inject_history_into_html(html: str, transcript_html: str) -> str:
    if not html or not transcript_html:
        return html

    insertion_marker = (
        "          <tr>\n"
        "            <td style=\"padding:20px 32px 28px 32px;background-color:#f8fafc;font-family:'Segoe UI',Arial,sans-serif;font-size:12px;line-height:1.6;color:#64748b;border-top:1px solid #e2e8f0;\">\n"
    )

    history_row = (
        "          <tr>\n"
        "            <td style=\"padding:0 32px 32px 32px;font-family:'Segoe UI',Arial,sans-serif;\">\n"
        f"              {transcript_html}\n"
        "            </td>\n"
        "          </tr>\n"
    )

    marker_index = html.find(insertion_marker)
    if marker_index == -1:
        closing_body = "</body>"
        body_index = html.lower().rfind(closing_body)
        if body_index == -1:
            return f"{html}\n{transcript_html}"
        return (
            f"{html[:body_index]}\n{transcript_html}\n{html[body_index:]}"
        )

    return html[:marker_index] + history_row + html[marker_index:]


def build_email_context_snapshot(context: AgentContext) -> Dict[str, Any]:
    snapshot = {
        "workflow_id": getattr(context, "workflow_id", None),
        "agent_id": "EmailDraftingAgent",
        "user_id": getattr(context, "user_id", None),
        "manifest": context.manifest(),
    }
    return {key: value for key, value in snapshot.items() if value}
