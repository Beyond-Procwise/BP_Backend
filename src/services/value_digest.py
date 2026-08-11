"""The weekly value digest — what we found, and what came back.

One email a week: value found this week, recovered this week, the top three open findings
by amount with their age, and one link into the drawer.

The rule that matters most is the one about NOT sending. A digest that arrives every week
saying nothing is a digest people stop opening, and then the week it does matter they miss
it. So an empty week — nothing found in the last seven days AND nothing recovered — sends
nothing at all.

Composed from the same GET /spendiq/value-summary data every surface uses, so the email
cannot disagree with the screens. Off by default: VALUE_DIGEST_ENABLED plus
VALUE_DIGEST_RECIPIENTS both have to be set before anything is sent.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.services import guardrail

logger = logging.getLogger(__name__)

WINDOW_DAYS = 7
TOP_N = 3
SUBJECT_PREFIX = "Value found this week"
DRAWER_LINK = "/spendiq"


# ---------------------------------------------------------------------------
# Composition (pure)
# ---------------------------------------------------------------------------

def _at(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _within(value: Any, since: datetime) -> bool:
    moment = _at(value)
    return moment is not None and moment >= since


def _money(amount: Any) -> str:
    """'£950.00'. An amount we could not convert is not a zero — it is left out of totals
    and rendered as an honest dash where it has to appear."""
    if amount is None:
        return "—"
    try:
        return f"£{float(amount):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _count(findings: list) -> str:
    n = len(findings)
    return f"{n} finding{'' if n == 1 else 's'}"


def _age(finding: dict) -> str:
    days = finding.get("age_days")
    if days is None:
        return ""
    if days <= 0:
        return "found today"
    return f"found {days} day{'' if days == 1 else 's'} ago"


def _live(summary: dict) -> list[dict]:
    """Findings that count. A superseded row is the same money as a stronger finding, so
    reporting it as this week's news would state it twice."""
    return [f for f in (summary.get("findings") or [])
            if f and not f.get("superseded_by")]


def compose_digest(summary: Optional[dict], now: datetime) -> Optional[dict]:
    """{'subject', 'body'} — or None when the week has nothing to report."""
    if not summary:
        return None
    since = now - timedelta(days=WINDOW_DAYS)
    live = _live(summary)

    found_this_week = [f for f in live if _within(f.get("found_at"), since)]
    recovered_this_week = [f for f in live
                           if f.get("recovered_gbp") and _within(f.get("resolved_at"), since)]
    if not found_this_week and not recovered_this_week:
        return None

    # A finding whose amount could not be converted is real but unpriced. It is counted,
    # never valued at zero — "£0.00 value found across 1 finding" is a contradiction, and
    # the kind that quietly trains people to distrust the number.
    valued = [f for f in found_this_week if f.get("amount_gbp") is not None]
    unvalued = len(found_this_week) - len(valued)
    found_total = sum(f["amount_gbp"] for f in valued)
    recovered_total = sum(f["recovered_gbp"] for f in recovered_this_week
                          if f.get("recovered_gbp") is not None)
    found_headline = _money(found_total) if valued else _count(found_this_week)

    # The list is what somebody can still act on — a resolved finding is not a to-do. It
    # still counts in the totals above; it is simply not on the list.
    actionable = sorted(
        [f for f in live if f.get("status") == "open" and f.get("amount_gbp") is not None],
        key=lambda f: f["amount_gbp"], reverse=True)[:TOP_N]

    subject = (f"{SUBJECT_PREFIX}: {found_headline}"
               f" · {_money(recovered_total)} recovered")

    found_line = (f"{_money(found_total)} value found this week across "
                  f"{_count(found_this_week)}." if valued
                  else f"{_count(found_this_week)} found this week, none of which "
                       f"could be converted to GBP.")
    if valued and unvalued:
        found_line += (f" A further {unvalued} finding{'' if unvalued == 1 else 's'} "
                       f"could not be converted and {'is' if unvalued == 1 else 'are'} "
                       f"not in that total.")
    lines = [
        found_line,
        f"{_money(recovered_total)} recovered this week.",
        "",
    ]
    if actionable:
        lines.append(f"Biggest open finding{'' if len(actionable) == 1 else 's'}:")
        for f in actionable:
            supplier = f.get("supplier_name") or "Unknown supplier"
            age = _age(f)
            lines.append(f"  · {_money(f['amount_gbp'])} — {f.get('title') or 'Untitled finding'}"
                         f" ({supplier}{', ' + age if age else ''})")
        lines.append("")
    else:
        lines.append("Nothing is open — every finding has been actioned.")
        lines.append("")

    unavailable = [k for k, v in (summary.get("sources") or {}).items() if v == "unavailable"]
    if unavailable:
        # Say what is missing rather than quietly reporting a smaller number.
        lines.append(f"This week excludes {len(unavailable)} unavailable source"
                     f"{'' if len(unavailable) == 1 else 's'}: {', '.join(sorted(unavailable))}.")
        lines.append("")

    lines.append(f"See every finding: {DRAWER_LINK}")
    return {"subject": subject, "body": "\n".join(lines)}


# ---------------------------------------------------------------------------
# The scheduled run
# ---------------------------------------------------------------------------

def _enabled() -> bool:
    return os.environ.get("VALUE_DIGEST_ENABLED", "0").strip() in ("1", "true", "True")


def recipients() -> list[str]:
    raw = os.environ.get("VALUE_DIGEST_RECIPIENTS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class _DigestPrincipal:
    """Who the digest sends as.

    Carries a subject and NOTHING else — deliberately no ``claims``. rbac
    resolves a principal's roles from two places: identity-provider group
    claims on a token, and a direct grant in ``proc.bp_role_assignment``.
    A principal built from configuration must not supply claims, because a
    group claim written into an environment variable is a role the
    configuration granted itself. With subject alone, the only thing that can
    give this identity permission is a row in the governed grant table.
    """

    subject: str


def sent_as() -> Optional[_DigestPrincipal]:
    """The configured sending identity, or None when nobody is accountable.

    A scheduled job has no authenticated caller. ``guardrail.authorize`` would
    refuse a ``None`` principal for an irreversible class anyway, but relying on
    that leaves the digest one policy edit away from sending unattributed mail.
    This refuses explicitly instead, so the reason in the log names the missing
    configuration rather than a generic policy denial.
    """
    subject = os.environ.get("VALUE_DIGEST_SENT_AS", "").strip()
    return _DigestPrincipal(subject=subject) if subject else None


def _sender_domain() -> str:
    sender = os.environ.get("SES_DEFAULT_SENDER", "").strip()
    return sender.partition("@")[2].strip().lower()


def _external(addresses: list[str]) -> list[str]:
    """Recipients outside the sending domain.

    The digest body names suppliers and the amounts we believe they over-billed.
    That is internal commercial analysis, and an address outside our own domain
    is not a typo worth delivering — it is the whole finding set leaving.

    With no sending domain configured we cannot tell internal from external, so
    every address counts as external and the digest does not send. Failing the
    other way would make a missing environment variable into a broadcast.
    """
    domain = _sender_domain()
    if not domain:
        return list(addresses)
    return [a for a in addresses if a.partition("@")[2].strip().lower() != domain]


def _load_summary() -> dict:
    from src.services.value_summary_service import build_value_summary
    return build_value_summary()


def _send_email(*, to: list[str], subject: str, body: str, agent_nick=None) -> bool:
    """The SES boundary, isolated so the digest is testable without a mail server.

    One message to all recipients rather than one each: this is a shared weekly summary,
    not a personalised notice, and N separate sends is N chances to half-deliver it.
    """
    from src.services.email_service import EmailService
    sender = getattr(getattr(agent_nick, "settings", None), "ses_default_sender", None) \
        or "noreply@procwise.co.uk"
    result = EmailService(agent_nick).send_email(subject=subject, body=body,
                                                 recipients=to, sender=sender)
    return bool(getattr(result, "success", result))


def run_weekly_digest(agent_nick=None) -> int:
    """Compose and send this week's digest. Returns 1 if an email went, else 0.

    Never raises: this runs on the scheduler, and a mail or database problem must not break
    the chain that follows it.
    """
    if not _enabled():
        logger.debug("value digest disabled by VALUE_DIGEST_ENABLED")
        return 0
    to = recipients()
    if not to:
        logger.info("value digest: no VALUE_DIGEST_RECIPIENTS configured — skipping")
        return 0

    # --- who is this being sent as -------------------------------------
    principal = sent_as()
    if principal is None:
        logger.warning(
            "value digest: VALUE_DIGEST_SENT_AS is not set, so no identity is "
            "accountable for this mail — not sending. Set it to a subject that "
            "holds a send grant in proc.bp_role_assignment."
        )
        return 0

    # --- who is it going to --------------------------------------------
    # Checked before the summary is built: there is no reason to assemble a
    # corpus-wide findings list for a send that is already refused.
    outside = _external(to)
    if outside:
        logger.warning(
            "value digest: %d recipient(s) outside the sending domain (%s) — "
            "not sending. The digest names suppliers and disputed amounts.",
            len(outside), ", ".join(outside),
        )
        return 0

    # --- may this identity send at all ---------------------------------
    # Same call value_query_service makes. NOT email_dispatch_guard: that path
    # requires a stored approved draft and recipients on the supplier master,
    # and this is internal mail with neither.
    try:
        decision = guardrail.authorize(
            "email.send",
            "communicate",
            principal,
            {"recipients": to, "purpose": "value_digest"},
        )
    except Exception:
        # authorize() is written not to raise, but a gate that fails open
        # because of an unexpected error is not a gate.
        logger.exception("value digest: authorization raised — not sending")
        return 0
    if not decision.allowed:
        logger.warning("value digest: refused by policy — %s", decision.reason)
        return 0

    try:
        digest = compose_digest(_load_summary(), datetime.now(timezone.utc))
    except Exception:
        logger.exception("value digest: could not build the summary")
        return 0
    if digest is None:
        logger.info("value digest: nothing found or recovered this week — not sending")
        return 0
    try:
        if not _send_email(to=to, subject=digest["subject"], body=digest["body"],
                           agent_nick=agent_nick):
            logger.warning("value digest: not accepted for delivery")
            return 0
    except Exception:
        logger.exception("value digest: send failed")
        return 0
    logger.info("value digest sent to %s: %s", ", ".join(to), digest["subject"])
    return 1
