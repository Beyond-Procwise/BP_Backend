"""The support agent behind "Contact support".

It greets the user by name, works out what they are actually asking, and then does
one of two things — honestly:

  * **Resolves it.** ProcWise holds a model of itself in the knowledge graph (how a
    document flows, which agent decides what, where a buyer sees something, what is
    known to be broken). Most support questions are "how do I…" or "why did my
    upload…", and that is answerable from the platform's own description. When the
    agent can answer, it walks the user through it properly rather than filing a
    ticket and making them wait.

  * **Escalates it.** When it cannot, it says so plainly, records a ticket, emails
    the admin, and tells the user their reference and that a person will be in touch.

What it must never do is the middle thing: invent a confident-sounding fix. A wrong
answer to "why is my invoice wrong" is worse than "I have raised this with the team",
because the user will act on it.

The ticket is written BEFORE the email is attempted. If SES is down the user's
problem must not vanish with it.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Dict, Optional

from services.tool_runtime import Tool, run_tools

logger = logging.getLogger(__name__)

# Where escalations go. A real address, not a placeholder that silently black-holes.
_ADMIN_EMAIL = os.getenv("SUPPORT_ADMIN_EMAIL", "nicholasgeelen@gmail.com")

_SYSTEM = """You are the ProcWise support agent. You are talking to a real user who has
just clicked "Contact support", so something is not working for them.

Greet them by name, once, and get to the point. They came here with a problem, not for
small talk.

Work out what they actually need, then do ONE of these:

1. If you can genuinely resolve it, do so. Call describe_platform to find out how the
   relevant part of ProcWise really works — the upload pipeline, extraction, promotion
   gates, the agents, the screens, the known gaps — and walk them through it in clear,
   numbered steps they can follow right now. Be specific: name the button, the screen,
   the field.

2. If you cannot resolve it — it needs a person, an account change, a fix, or you simply
   do not know — say so plainly, tell them you have raised it with the team, and stop.
   Do NOT guess. A confident wrong answer to "why is my invoice wrong" is worse than
   "I've passed this to the team", because they will act on it.

HARD RULE ON UI DETAIL. Only name a screen, a button, or a menu path if describe_platform
actually told you about it. Do NOT describe what a button looks like, what colour it is,
what icon it has, or which menu it sits under unless that is in the facts you were given.
If the facts explain the mechanism but not where the control is, say what happens and be
honest that you cannot see their screen — then ask them what they can see, or escalate.
Words like "usually", "typically" or "should be" mean you are guessing: delete the
sentence. An invented button sends the user hunting for something that does not exist,
and they will trust you while they do it.

HARD RULE ON CONTACT DETAILS. Never invent an email address, phone number, URL or team
name. Do not write "contact sales@..." or "reach out to support@..." — you do not know
those addresses and a plausible-looking one is worse than none, because they will email
it and hear nothing back. YOU are the support channel: if a person is needed, say you
have raised it with the team, and stop. Nothing else.

End your reply with exactly one of these markers on its own final line:

RESOLVED: <a short summary of the fix you gave>
ESCALATE: <a one-line summary of the problem, for the admin>

Never write both. Never omit it. Choose ESCALATE whenever you are unsure."""


def _platform_tool() -> list[Tool]:
    """The platform's model of itself — how ProcWise actually works."""

    def _describe(topic: str) -> Any:
        try:
            from services.platform_kg import describe

            facts = describe(str(topic))
            if not facts:
                return {
                    "facts": None,
                    "note": (
                        f"Nothing is recorded about '{topic}'. Do not invent an answer — "
                        "escalate instead."
                    ),
                }
            return {"facts": facts}
        except Exception as exc:  # noqa: BLE001
            logger.warning("platform_kg lookup failed: %s", exc)
            return {"facts": None, "note": "the platform description is unavailable"}

    return [
        Tool(
            name="describe_platform",
            description=(
                "Look up how ProcWise itself works: uploading and ingesting documents, "
                "extraction, promotion gates, the agents and what they decide, the "
                "screens and what is behind them, and the known gaps. Use this before "
                "answering ANY 'how do I' or 'why did it' question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "e.g. 'upload', 'extraction failed', 'invoices screen'",
                    }
                },
                "required": ["topic"],
            },
            handler=_describe,
        )
    ]


def _is_marker(line: str) -> bool:
    s = line.strip().upper()
    return s.startswith("RESOLVED:") or s.startswith("ESCALATE:")


# A prompt rule is a request; this is a guarantee. The model wrote "reach out to ProcWise
# Sales at sales@procwise.com" — an address that does not exist. The user would have
# emailed it and heard nothing, and would have blamed the company, not the bot. Anything
# that looks like a contact route and is not the real admin address is removed before the
# text ever reaches them.
_CONTACT_PATTERN = re.compile(
    r"""(
        [\w.+-]+@[\w-]+\.[\w.-]+           # an email address
      | https?://\S+                       # a URL
      | \+?\d[\d\s()-]{8,}\d                # a phone number
    )""",
    re.VERBOSE,
)


def _scrub_contacts(text: str) -> str:
    """Remove any contact detail the agent invented."""

    def _replace(m: re.Match) -> str:
        found = m.group(0)
        # The admin address is real and may legitimately appear.
        if found.lower() == _ADMIN_EMAIL.lower():
            return found
        logger.warning("support agent invented a contact detail; removed: %s", found)
        return "the team"

    return _CONTACT_PATTERN.sub(_replace, text)


def _split_marker(answer: str) -> tuple[str, str, str]:
    """Return (reply_without_marker, outcome, summary)."""
    reply, outcome, summary = answer.strip(), "escalated", ""
    for line in reversed(reply.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("RESOLVED:"):
            outcome, summary = "resolved", stripped.split(":", 1)[1].strip()
            reply = reply.replace(line, "").strip()
            break
        if stripped.upper().startswith("ESCALATE:"):
            outcome, summary = "escalated", stripped.split(":", 1)[1].strip()
            reply = reply.replace(line, "").strip()
            break
    else:
        # No marker at all. Escalate — an unmarked answer is one we cannot vouch for.
        logger.warning("support agent produced no outcome marker; escalating")
    return reply, outcome, summary or "User contacted support."


class SupportAgent:
    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick

    # ------------------------------------------------------------------
    def handle(
        self,
        message: str,
        *,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        name = (user_name or "").strip() or "there"

        task = (
            f"The user's name is {name}.\n"
            f"Their email is {user_email or 'not provided'}.\n\n"
            f"They wrote:\n{message}"
        )

        result = run_tools(
            task,
            _platform_tool(),
            _SYSTEM,
            max_rounds=4,
            # Support questions are about the product, and the product's behaviour is in
            # the knowledge graph — so an answer given without consulting it is a guess.
            require_tool_use=True,
            nudge=(
                "You answered without looking anything up. Call describe_platform to find "
                "out how this part of ProcWise actually behaves, then answer from that — "
                "or escalate if it does not tell you."
            ),
        )

        if result.error and not result.answer:
            # The model is unreachable. Escalate rather than leaving them with nothing.
            reply = (
                f"Hello {name} — I'm sorry, I can't reach the assistant right now, so I "
                "haven't been able to look into this myself. I've raised it with the team "
                "and someone will come back to you."
            )
            outcome, summary = "escalated", f"Support agent unavailable: {result.error}"
        else:
            reply, outcome, summary = _split_marker(result.answer or "")
            reply = _scrub_contacts(reply)
            if not reply:
                reply = (
                    f"Hello {name} — I've noted this and raised it with the team; "
                    "someone will be in touch."
                )

        return self._finish(
            reply=reply,
            outcome=outcome,
            summary=summary,
            name=name,
            message=message,
            user_name=user_name,
            user_email=user_email,
            session_id=session_id,
            tools_used=result.tools_used,
        )

    # ------------------------------------------------------------------
    def handle_stream(
        self,
        message: str,
        *,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
        emit: Any = None,
    ) -> Dict[str, Any]:
        """Same as handle(), but the reply is streamed out through ``emit``.

        ``emit(kind, payload)`` fires with:
          stage  - 'thinking' | 'looking_up' (with the topic it is checking)
          delta  - a piece of the reply

        The outcome marker (RESOLVED:/ESCALATE:) is an instruction to US, not something
        the user should ever see. It arrives on the final line, so completed lines are
        released as they finish and the trailing partial line is held back until we know
        it is prose rather than a marker. Streaming naively would flash "ESCALATE: user
        cannot restore access" onto their screen.
        """
        from services.tool_runtime import run_tools_stream

        name = (user_name or "").strip() or "there"

        def _emit(kind: str, **payload: Any) -> None:
            if emit is None:
                return
            try:
                emit(kind, payload)
            except Exception:  # pragma: no cover
                logger.debug("support stream consumer raised", exc_info=True)

        _emit("stage", stage="thinking")

        held = ""   # the trailing, not-yet-terminated line

        def _on_delta(fragment: str) -> None:
            nonlocal held
            held += fragment
            # Release only whole lines; keep the last (possibly incomplete) one back.
            #
            # Scrubbing per line matters here: the final reply is scrubbed too, but on the
            # streaming path the text is already on the user's screen by then. An invented
            # "sales@procwise.com" would have been read before the guard ever ran. Whole
            # lines are the right unit — a contact detail cannot straddle a newline.
            while "\n" in held:
                line, held = held.split("\n", 1)
                if _is_marker(line):
                    # A marker means the answer is over. Swallow it and everything after.
                    held = ""
                    return
                _emit("delta", text=_scrub_contacts(line) + "\n")

        def _on_tool(call: Any) -> None:
            topic = (call.arguments or {}).get("topic") if call.arguments else None
            _emit("stage", stage="looking_up", topic=topic)

        result = run_tools_stream(
            f"The user's name is {name}.\nTheir email is {user_email or 'not provided'}.\n\n"
            f"They wrote:\n{message}",
            _platform_tool(),
            _SYSTEM,
            max_rounds=4,
            on_tool=_on_tool,
            on_delta=_on_delta,
        )

        # Flush whatever is left, unless it is the marker.
        if held and not _is_marker(held):
            _emit("delta", text=_scrub_contacts(held))

        if result.error and not result.answer:
            reply = (
                f"Hello {name} — I'm sorry, I can't reach the assistant right now, so I "
                "haven't been able to look into this myself. I've raised it with the team "
                "and someone will come back to you."
            )
            _emit("delta", text=reply)
            outcome, summary = "escalated", f"Support agent unavailable: {result.error}"
        else:
            reply, outcome, summary = _split_marker(result.answer or "")
            reply = _scrub_contacts(reply)

        return self._finish(
            reply=reply,
            outcome=outcome,
            summary=summary,
            name=name,
            message=message,
            user_name=user_name,
            user_email=user_email,
            session_id=session_id,
            tools_used=result.tools_used,
        )

    # ------------------------------------------------------------------
    def _finish(
        self,
        *,
        reply: str,
        outcome: str,
        summary: str,
        name: str,
        message: str,
        user_name: Optional[str],
        user_email: Optional[str],
        session_id: Optional[str],
        tools_used: list,
    ) -> Dict[str, Any]:
        reference = f"SUP-{uuid.uuid4().hex[:6].upper()}"

        # Write the ticket FIRST. If the email then fails, the request still exists.
        # A guided answer is stored too, as 'awaiting_confirmation': we do not get to
        # call it solved until the person says it is.
        ticket_id = self._store(
            reference=reference,
            user_name=user_name,
            user_email=user_email,
            session_id=session_id,
            message=message,
            outcome=outcome,
            agent_reply=reply,
            status="awaiting_confirmation" if outcome == "resolved" else "open",
        )

        email_status = "not_required"
        awaiting = False

        if outcome == "escalated":
            email_status, email_error = self._notify_admin(
                reference=reference,
                user_name=name,
                user_email=user_email,
                message=message,
                summary=summary,
                agent_reply=reply,
            )
            self._update_email_status(reference, email_status, email_error)
            reply = (
                f"{reply}\n\n"
                f"I've logged this as **{reference}** and passed it to the team — "
                "someone will contact you shortly."
            )
        else:
            # Guidance is not the same as a fix. Ask, rather than assuming it landed and
            # quietly closing the ticket — the whole point of support is that the person
            # walks away with the problem gone, and only they can tell us that.
            awaiting = True
            reply = f"{reply}\n\nDid that sort it? If not, tell me and I'll raise it with the team."

        return {
            "reply": reply,
            "outcome": outcome,
            "reference": reference,
            "ticket_id": ticket_id,
            "escalated": outcome == "escalated",
            "admin_notified": email_status == "sent",
            # The UI shows a Yes / Not fixed prompt on this, and calls /confirm.
            "awaiting_confirmation": awaiting,
            "tools_used": tools_used,
        }

    # ------------------------------------------------------------------
    def confirm(
        self, reference: str, *, resolved: bool, note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Close the loop on a guided answer: did it actually fix their problem?

        If it did, the ticket closes. If it did not, this is where the ticket becomes a
        real escalation and the admin is emailed — which is the case that matters, since
        an agent that guesses well enough to sound convincing is precisely the one whose
        advice needs checking.
        """
        ticket = self._fetch(reference)
        if not ticket:
            return {"error": f"unknown support reference {reference}"}

        if resolved:
            self._set_status(reference, status="closed", outcome="resolved", closed=True)
            return {
                "reference": reference,
                "status": "closed",
                "reply": "Glad that sorted it. I'll close this off — shout if it comes back.",
            }

        # It did not work. Escalate for real.
        name = (ticket.get("user_name") or "").strip() or "there"
        summary = f"Guided answer did not resolve it. {note or ''}".strip()
        email_status, email_error = self._notify_admin(
            reference=reference,
            user_name=name,
            user_email=ticket.get("user_email"),
            message=(
                f"{ticket.get('message')}\n\n"
                f"[The assistant's guidance did NOT fix it."
                + (f" User added: {note}]" if note else "]")
            ),
            summary=summary,
            agent_reply=ticket.get("agent_reply") or "",
        )
        self._update_email_status(reference, email_status, email_error)
        self._set_status(reference, status="open", outcome="escalated", closed=False)

        return {
            "reference": reference,
            "status": "escalated",
            "escalated": True,
            "admin_notified": email_status == "sent",
            "reply": (
                f"Sorry that didn't fix it. I've raised **{reference}** with the team and "
                "sent them everything we've discussed — someone will contact you shortly."
            ),
        }

    def _fetch(self, reference: str) -> Optional[Dict[str, Any]]:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT reference, user_name, user_email, message, agent_reply,
                               outcome, status
                        FROM proc.bp_support_ticket WHERE reference = %s
                        """,
                        (reference,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return dict(zip([d[0] for d in cur.description], row))
        except Exception:
            logger.exception("failed to read support ticket %s", reference)
            return None

    def _set_status(
        self, reference: str, *, status: str, outcome: str, closed: bool
    ) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.bp_support_ticket
                           SET status = %s,
                               outcome = %s,
                               closed_at = CASE WHEN %s THEN NOW() ELSE closed_at END
                         WHERE reference = %s
                        """,
                        (status, outcome, closed, reference),
                    )
                conn.commit()
        except Exception:
            logger.exception("failed to update support ticket %s", reference)

    # ------------------------------------------------------------------
    def _store(self, **kw: Any) -> Optional[int]:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.bp_support_ticket
                            (reference, user_name, user_email, session_id, message,
                             outcome, agent_reply, status)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        RETURNING ticket_id
                        """,
                        (
                            kw["reference"],
                            kw.get("user_name"),
                            kw.get("user_email"),
                            kw.get("session_id"),
                            kw["message"],
                            kw["outcome"],
                            kw.get("agent_reply"),
                            kw.get("status", "open"),
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception:
            logger.exception("failed to store support ticket")
            return None

    def _update_email_status(
        self, reference: str, status: str, error: Optional[str]
    ) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.bp_support_ticket
                           SET email_status = %s, email_error = %s, notified_admin = %s
                         WHERE reference = %s
                        """,
                        (status, error, _ADMIN_EMAIL, reference),
                    )
                conn.commit()
        except Exception:
            logger.exception("failed to update support ticket email status")

    def _notify_admin(self, **kw: Any) -> tuple[str, Optional[str]]:
        """Email the admin. Returns (status, error)."""
        subject = f"[ProcWise support] {kw['reference']} — {kw['summary'][:70]}"
        body = (
            f"A user has contacted support and the assistant could not resolve it.\n\n"
            f"Reference : {kw['reference']}\n"
            f"From      : {kw['user_name']} <{kw.get('user_email') or 'no email given'}>\n"
            f"Summary   : {kw['summary']}\n\n"
            f"--- What they said ---\n{kw['message']}\n\n"
            f"--- What the assistant told them ---\n{kw['agent_reply']}\n"
        )
        try:
            from services.email_service import EmailService

            sender = getattr(self.agent_nick.settings, "ses_default_sender", None)
            if not sender:
                return "failed", "no SES sender configured"

            EmailService(self.agent_nick).send_email(
                subject=subject,
                body=body,
                recipients=_ADMIN_EMAIL,
                sender=sender,
            )
            logger.info("support escalation %s emailed to %s", kw["reference"], _ADMIN_EMAIL)
            return "sent", None
        except Exception as exc:  # noqa: BLE001
            # Do NOT let this bubble up. The user was told their issue is noted, and it
            # IS — the ticket row exists. A failed notification is an admin problem, not
            # something to throw in the user's face.
            logger.exception("failed to email support escalation %s", kw["reference"])
            return "failed", str(exc)[:300]
