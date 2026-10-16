"""The restrictions the Email page draws, each with how it is guaranteed.

Worked out on the server from live state every time it is asked; the screen only draws
the list. Three kinds of guarantee:

* ``provider`` -- Microsoft enforces it, and we prove it (our token holds no mail
  permissions; each mailbox's access proof is fresh and healthy).
* ``procwise`` -- our own code enforces it (the Graph path allowlist).
* ``policy`` -- the customer's email policy says so; the screen shows the choice.

Nothing is shown as in force that was not proven: a check that cannot run is
``not_proven``, never ``in_force``.
"""
from __future__ import annotations

import html as _h
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

IN_FORCE, FAILED, NOT_PROVEN = "in_force", "failed", "not_proven"
# Only an admin's Add or "Check again" re-runs the control read and stamps verified_at; the
# hourly health check reads the mailbox itself but not the control, so it proves reach, not
# scope, and does not renew this.
PROOF_MAX_AGE = timedelta(hours=2)
POLICY_UNREADABLE = "Email policy could not be read"

# Titles and explanations are the mock-up's Admin artboard words.
_TEXT = {
    "no_mail_permissions": (
        "Procwise has no mail permissions in your Microsoft account",
        "Microsoft shows “Procwise Mail” with sign-in only. Mail access comes solely from the "
        "restriction below."),
    "mailbox_scope": (
        "Only the mailboxes in “{group}” can be reached",
        "Reading each chosen mailbox succeeded. Reading a mailbox outside the group (the "
        "control) was refused by Microsoft — so nothing outside the group is reachable."),
    "actions_limited": (
        "Read and send only — never move, delete or mark as read",
        "Procwise’s own code refuses every other action."),
    "access_level": (
        "Access your IT grants: read and send only",
        "Your policy has the setup script grant read (Mail.Read) and send only. Procwise does "
        "not check what your IT actually ran."),
    "internal_approval": (
        "Emails to colleagues need the user’s approval",
        "Forwards, internal replies and Procwise’s own notices wait until the person who wrote "
        "them approves."),
    "external_suppliers_only": (
        "Emails outside the company go only to known suppliers",
        "Anything else is held with the reason and shown to the sender."),
    "read_scope": (
        "Only replies and supplier mail are read",
        "Replies to emails Procwise sent, and mail from known suppliers. Other mail in the "
        "shared inbox is not opened."),
    "on_disconnect": (
        "If the connection breaks, emails are held",
        "Once your company has a mailbox set up, emails wait while it cannot be used; they do "
        "not go out from the Procwise address instead. Each held email is recorded in the "
        "activity log with the reason; no separate alert is sent."),
}

# When a provider or code check is not in force, the success sentence above would be
# untrue in an evidence document; these say what is actually known.
_NOT_IN_FORCE_TEXT = {
    ("no_mail_permissions", FAILED):
        "Microsoft reports that Procwise’s app holds mail permissions. Ask your IT to remove "
        "them in Microsoft Entra.",
    ("no_mail_permissions", NOT_PROVEN):
        "We could not confirm this with Microsoft just now.",
    ("mailbox_scope", FAILED):
        "Microsoft reports that at least one mailbox no longer responds as limited. Its "
        "access was withdrawn or must be checked again; see the mailboxes below.",
    ("mailbox_scope", NOT_PROVEN):
        "We could not confirm this: a mailbox has no recent proof (older than 2 hours), is "
        "temporarily unavailable, or no mailbox is connected yet. The proof is not renewed "
        "automatically: an admin uses “Check again” on the mailbox to prove it again.",
    ("actions_limited", FAILED):
        "Procwise’s code allows an action that is not described here, so this could not be "
        "confirmed.",
}

# (id, policy field, the narrow value, wider?, title and explanation when the admin chose
# otherwise). The title then says what the chosen option does, never the narrow guarantee.
# "wider?" is False where the other choice is stricter (emails to colleagues switched off).
_POLICY_ITEMS = [
    ("access_level", "access_level", "read_send", True,
     "Access your IT grants: read, send and change",
     "Your policy lets your IT grant read, send and change (Mail.ReadWrite), so Microsoft "
     "would let Procwise change mail in the chosen mailboxes. Procwise’s own code still only "
     "reads and sends. Procwise does not check what your IT actually ran."),
    ("internal_approval", "internal_send", "with_user_approval", False,
     "Emails to colleagues are switched off",
     "Your policy switches emails to colleagues off, so none are sent from the company "
     "mailbox. Procwise’s own notices go from the Procwise address."),
    ("external_suppliers_only", "external_recipients", "known_suppliers_only", True,
     "Emails outside the company can go to any address",
     "Your policy allows any outside address, not only known suppliers."),
    ("read_scope", "read_scope", "replies_and_suppliers", True,
     "Every message in the shared inbox is read",
     "Your policy lets Procwise read the whole shared inbox, not only replies and supplier "
     "mail."),
    ("on_disconnect", "on_disconnect", "hold", True,
     "If the connection breaks, emails go from the Procwise address",
     "Your policy sends from the Procwise address if the connection breaks, the mailbox is "
     "switched off or its access is withdrawn. While a mailbox is only temporarily "
     "unavailable, emails are still held."),
]

SETTING_LABELS = {
    "internal_send": "Emails to colleagues",
    "company_domains": "Company domains",
    "external_recipients": "Emails outside the company",
    "read_scope": "Which incoming mail is read",
    "on_disconnect": "If the connection breaks",
    "access_level": "Access your IT grants",
    "notices_from_mailbox": "Procwise notices from the mailbox",
    "poll_seconds": "Check inbox every (seconds)",
    "restriction_group": "Restriction group",
}
_VALUE_WORDS = {
    ("internal_send", "off"): "Switched off",
    ("internal_send", "with_user_approval"): "Only after the user approves",
    ("notices_from_mailbox", "off"): "Switched off",
    ("notices_from_mailbox", "with_user_approval"): "Only after the user approves",
    ("external_recipients", "known_suppliers_only"): "Known suppliers only",
    ("external_recipients", "any"): "Any address",
    ("read_scope", "replies_and_suppliers"): "Replies and known suppliers only",
    ("read_scope", "whole_inbox"): "The whole shared inbox",
    ("on_disconnect", "hold"): "Hold emails until it is fixed",
    ("on_disconnect", "procwise_address"): "Send from the Procwise address",
    ("access_level", "read_send"): "Read and send only",
    ("access_level", "read_send_change"): "Read, send and change",
}

# Every Graph path our code may call, in plain words. A path the allowlist gains that is
# not described here turns `actions_limited` red until someone says what it does.
_PATH_WORDS = {
    ("GET", r"^/users/[^/]+/mailFolders/inbox$"): "list inbox",
    ("GET", r"^/users/[^/]+/mailFolders/inbox/messages$"): "list inbox",
    ("GET", r"^/users/[^/]+/messages$"): "list inbox",
    ("GET", r"^/users/[^/]+/messages/[^/]+/\$value$"): "open message",
    ("GET", r"^/users/[^/]+/messages/[^/]+$"): "open message",
    ("POST", r"^/users/[^/]+/sendMail$"): "send",
}

_STATE_WORDS = {"OK": "Healthy", "DEGRADED": "Temporarily unavailable",
                "REVOKED": "Access withdrawn in Microsoft 365"}


def setting_words(field: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "none"
    if value is None:
        return "not set"
    return _VALUE_WORDS.get((field, value), str(value))


def _iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
    return str(v)


def _when(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if not isinstance(v, datetime):
        try:
            v = datetime.fromisoformat(str(v))
        except ValueError:
            return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def _item(rid: str, guarantee: str, status: str, evidence: List[dict], *,
          last_checked_at: Any, title: Optional[str] = None,
          explanation: Optional[str] = None) -> Dict[str, Any]:
    t, e = _TEXT[rid]
    e = _NOT_IN_FORCE_TEXT.get((rid, status), e)
    return {"id": rid, "title": title or t, "explanation": explanation or e,
            "guarantee": guarantee, "status": status, "last_checked_at": _iso(last_checked_at),
            "evidence": evidence}


def _no_mail_permissions(conn: Any, provider: Any, now: datetime, proof: Any) -> Dict[str, Any]:
    if provider is None:
        ev = [{"check": "Email connection", "answer": "Email is not connected",
               "result": "unknown", "at": now.isoformat()}]
        return _item("no_mail_permissions", "provider", NOT_PROVEN, ev, last_checked_at=None)
    try:
        rows = proof if proof is not _ASK else provider.prove(conn)
        rows = [r for r in (rows or []) if r.get("restriction") == "no_mail_permissions"]
    except Exception:  # noqa: BLE001 - a check that cannot run proves nothing
        logger.exception("provider proof failed")
        rows = []
    ev = [{k: r.get(k) for k in ("check", "answer", "result", "at")} for r in rows]
    if not ev:
        ev = [{"check": "Mail permissions", "answer": "The check could not be run",
               "result": "unknown", "at": now.isoformat()}]
    results = {r["result"] for r in ev}
    status = (FAILED if "fail" in results
              else IN_FORCE if results == {"pass"} else NOT_PROVEN)
    return _item("no_mail_permissions", "provider", status, ev,
                 last_checked_at=max((r["at"] for r in ev if r.get("at")), default=None))


def _mailbox_scope(conn: Any, provider: Any, policy: Any, now: datetime,
                   given: Any) -> Dict[str, Any]:
    group = getattr(policy, "restriction_group", None)
    title = (_TEXT["mailbox_scope"][0].format(group=group) if group
             else "Only the mailboxes in your restriction group can be reached")

    def done(status, ev, last):
        return _item("mailbox_scope", "provider", status, ev, last_checked_at=last, title=title)

    if provider is None:
        return done(NOT_PROVEN, [{"check": "Email connection", "answer": "Email is not connected",
                                  "result": "unknown", "at": now.isoformat()}], None)
    if given is _ASK:
        given = read_mailboxes(provider, conn)
    if given is None:
        return done(NOT_PROVEN, [{"check": "Mailboxes", "answer": "Could not be read",
                                  "result": "unknown", "at": now.isoformat()}], None)
    mailboxes = list(given)
    if not mailboxes:
        return done(NOT_PROVEN, [{"check": "Mailboxes", "answer": "No mailbox is connected yet",
                                  "result": "unknown", "at": now.isoformat()}], None)
    ev, results, times = [], [], []
    for m in mailboxes:
        state = m.get("health_state")
        at = _when(m.get("verified_at"))
        if m.get("withdrawn") or state == "REVOKED":
            result = "fail"
        elif at is None or now - at > PROOF_MAX_AGE or state != "OK":
            result = "unknown"
        else:
            result = "pass"
        words = _STATE_WORDS.get(state, str(state or "unknown"))
        if at is None:
            words += "; never proven"
        elif now - at > PROOF_MAX_AGE:
            words += "; last proof is more than 2 hours old"
        ev.append({"check": f"Access proof for {m.get('address')}", "answer": words,
                   "result": result, "at": _iso(at)})
        results.append(result)
        if at is not None:
            times.append(at)
    status = (FAILED if "fail" in results
              else IN_FORCE if all(r == "pass" for r in results) else NOT_PROVEN)
    return done(status, ev, min(times) if times else None)


def _actions_limited(now: datetime) -> Dict[str, Any]:
    from services.mail_m365 import graph_client
    words, unknown = [], []
    for method, patterns in graph_client.ALLOWED_PATH_PATTERNS.items():
        for p in patterns:
            w = _PATH_WORDS.get((method, getattr(p, "pattern", str(p))))
            if w is None:
                unknown.append(f"{method} {getattr(p, 'pattern', p)}")
            elif w not in words:
                words.append(w)
    if unknown:
        ev = [{"check": "What Procwise’s code allows in Microsoft 365",
               "answer": "Paths not described: " + "; ".join(unknown), "result": "fail",
               "at": now.isoformat()}]
        return _item("actions_limited", "procwise", FAILED, ev, last_checked_at=now)
    ev = [{"check": "What Procwise’s code allows in Microsoft 365", "answer": ", ".join(words),
           "result": "pass", "at": now.isoformat()}]
    return _item("actions_limited", "procwise", IN_FORCE, ev, last_checked_at=now)


def _policy_items(policy: Any, now: datetime) -> List[Dict[str, Any]]:
    out = []
    for rid, field, narrow, is_wider, other_title, other_text in _POLICY_ITEMS:
        if policy is None:
            ev = [{"check": "Email policy", "answer": "could not be read", "result": "unknown",
                   "at": now.isoformat()}]
            out.append(_item(rid, "policy", NOT_PROVEN, ev, last_checked_at=None,
                             explanation=POLICY_UNREADABLE))
            continue
        value = getattr(policy, field)
        on = value == narrow
        ev = [{"check": f"Email policy, version {policy.version}: {SETTING_LABELS[field]}",
               "answer": setting_words(field, value), "result": "pass" if on else "chosen",
               "at": now.isoformat()}]
        out.append(_item(rid, "policy", IN_FORCE, ev, last_checked_at=now,
                         title=None if on else other_title,
                         explanation=None if on else other_text))
    return out


_WIDER_WHEN_CHOSEN = {item[0] for item in _POLICY_ITEMS if item[3]}


def _is_wider(item: dict) -> bool:
    """A policy line where the admin chose the wider option (its evidence says "chosen")."""
    return (item.get("id") in _WIDER_WHEN_CHOSEN
            and any(e.get("result") == "chosen" for e in item.get("evidence") or []))


_ASK = object()   # "not given: ask the provider"


def read_mailboxes(provider: Any, conn: Any) -> Optional[List[dict]]:
    """The provider's mailboxes, or None when they could not be read."""
    if provider is None:
        return []
    try:
        return list(provider.status(conn).get("mailboxes") or [])
    except Exception:  # noqa: BLE001 - the restriction line says not proven
        logger.exception("mailbox status could not be read")
        return None


def build(conn: Any, *, provider: Any, policy: Any, now: datetime,
          mailboxes: Any = _ASK, proof: Any = _ASK) -> List[Dict[str, Any]]:
    """The eight restrictions, in the screen's order. ``policy`` is None when it could not
    be read; ``provider`` is None when email was never connected. ``mailboxes`` (None =
    could not be read) and ``proof`` may be given so one read feeds the screen and the
    report, and so the provider's check can run after the caller's connection is closed."""
    return [_no_mail_permissions(conn, provider, now, proof),
            _mailbox_scope(conn, provider, policy, now, mailboxes),
            _actions_limited(now),
            *_policy_items(policy, now)]


def summary(restrictions: Iterable[dict]) -> Dict[str, int]:
    """``in_force`` counts lines applied as set; ``wider`` how many of those are policy
    lines where the admin chose the wider option (their titles say what they allow)."""
    items = list(restrictions)
    return {"in_force": sum(1 for i in items if i["status"] == IN_FORCE), "total": len(items),
            "wider": sum(1 for i in items if _is_wider(i))}


# ---- the evidence report ----

_STATUS_WORDS = {IN_FORCE: "In force", FAILED: "Failed", NOT_PROVEN: "Not proven"}
_GUARANTEE_WORDS = {"provider": "Enforced by Microsoft, proven by Procwise",
                    "procwise": "Enforced by Procwise’s own code",
                    "policy": "Your email policy"}
_RESULT_WORDS = {"pass": "Passed", "fail": "Failed", "unknown": "Not proven",
                 "chosen": "Admin’s choice"}

_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: sans-serif; font-size: 10pt; color: #1f2933; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; }
.meta { color: #4a5560; margin: 0 0 10pt; }
table { width: 100%; border-collapse: collapse; margin: 4pt 0 8pt; }
th, td { text-align: left; vertical-align: top; padding: 4pt 6pt; border-bottom: 0.5pt solid #ccd; }
th { font-size: 8.5pt; color: #4a5560; }
.r { border: 0.5pt solid #ccd; padding: 6pt 8pt; margin: 0 0 8pt; page-break-inside: avoid; }
.r h3 { font-size: 11pt; margin: 0 0 2pt; }
.st-in_force { color: #0f5132; } .st-failed { color: #8f1d14; } .st-not_proven { color: #7a4b00; }
.small { font-size: 8.5pt; color: #4a5560; }
"""


def _e(v: Any) -> str:
    return _h.escape("" if v is None else str(v), quote=True)


def _date(v: Any) -> str:
    d = _when(v)
    return d.astimezone(timezone.utc).strftime("%d %b %Y %H:%M UTC") if d else str(v) if v else "—"


def evidence_html(restrictions: List[dict], policy: Any, mailboxes: List[dict],
                  history: List[dict], *, generated_at: datetime, generated_by: str) -> str:
    """A self-contained page (no outside resources) for render_pdf; every value escaped."""
    s = summary(restrictions)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Email restrictions — evidence report</title><style>{_CSS}</style></head><body>",
        "<h1>Email restrictions — evidence report</h1>",
        f"<p class='meta'>Generated {_e(_date(generated_at))} by {_e(generated_by)}. "
        f"{s['in_force']} of {s['total']} restrictions in force"
        + (f" as set; {s['wider']} of them are set wider than the strictest option, and "
           f"their titles below say what they allow.</p>" if s["wider"] else ".</p>"),
        "<h2>Restrictions</h2>",
    ]
    for r in restrictions:
        parts.append(
            f"<div class='r'><h3>{_e(r['title'])}</h3>"
            f"<div class='st-{_e(r['status'])}'><b>{_e(_STATUS_WORDS.get(r['status'], r['status']))}</b>"
            f" · {_e(_GUARANTEE_WORDS.get(r['guarantee'], r['guarantee']))}"
            f" · last checked {_e(_date(r.get('last_checked_at')))}</div>"
            f"<p>{_e(r['explanation'])}</p>"
            "<table><tr><th>Check</th><th>Answer</th><th>Result</th><th>When</th></tr>")
        for ev in r.get("evidence") or []:
            parts.append(
                f"<tr><td>{_e(ev.get('check'))}</td><td>{_e(ev.get('answer'))}</td>"
                f"<td>{_e(_RESULT_WORDS.get(ev.get('result'), ev.get('result')))}</td>"
                f"<td>{_e(_date(ev.get('at')))}</td></tr>")
        parts.append("</table></div>")

    parts.append("<h2>Email policy</h2>")
    if policy is None:
        parts.append(f"<p>{_e(POLICY_UNREADABLE)}.</p>")
    else:
        parts.append(f"<p class='meta'>Current version {_e(policy.version)}.</p>"
                     "<table><tr><th>Setting</th><th>Value</th></tr>")
        for field, label in SETTING_LABELS.items():
            parts.append(f"<tr><td>{_e(label)}</td>"
                         f"<td>{_e(setting_words(field, getattr(policy, field, None)))}</td></tr>")
        parts.append("</table>")

    parts.append("<h2>Mailboxes</h2>")
    if not mailboxes:
        parts.append("<p>No mailbox is connected.</p>")
    else:
        parts.append("<table><tr><th>Mailbox</th><th>State</th><th>Default sender</th>"
                     "<th>Access last proven</th><th>Last test email</th></tr>")
        for m in mailboxes:
            state = _STATE_WORDS.get(m.get("health_state"), m.get("health_state") or "unknown")
            if m.get("withdrawn"):
                state = _STATE_WORDS["REVOKED"]
            parts.append(
                f"<tr><td>{_e(m.get('address'))}</td><td>{_e(state)}</td>"
                f"<td>{'Yes' if m.get('is_default') else 'No'}</td>"
                f"<td>{_e(_date(m.get('verified_at')))}</td>"
                f"<td>{_e(_date(m.get('last_test_send_at')))}</td></tr>")
        parts.append("</table>")

    parts.append("<h2>Change history</h2>")
    if not history:
        parts.append("<p>No changes recorded.</p>")
    else:
        parts.append("<table><tr><th>Version</th><th>When</th><th>By</th><th>Setting</th>"
                     "<th>Before</th><th>After</th></tr>")
        for h in history:
            field = h.get("field")
            parts.append(
                f"<tr><td>{_e(h.get('version'))}</td><td>{_e(_date(h.get('changed_at')))}</td>"
                f"<td>{_e(h.get('changed_by'))}</td>"
                f"<td>{_e(SETTING_LABELS.get(field, field))}</td>"
                f"<td>{_e(setting_words(field, h.get('before')))}</td>"
                f"<td>{_e(setting_words(field, h.get('after')))}</td></tr>")
        parts.append("</table>")
    parts.append("<p class='small'>Generated from the same live checks as the Email settings "
                 "page.</p></body></html>")
    return "".join(parts)
