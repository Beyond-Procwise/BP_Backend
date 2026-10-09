"""Bank and payment details in OUTGOING email: a hard rule, in code, for every family.

Ruling 2026-10-09. An outgoing request or draft that mentions new or changed bank or payment details is the outbound
half of invoice-redirection fraud (or a mistake that looks exactly like it), so:

* it is always held for a person: marked ``needs_review`` and not ready whatever the family's mode, and the send guard
  refuses it unless a person approved it (no agent autonomy reaches it);
* the repair pass never touches it: a model must not rewrite it into something that passes;
* it must point the supplier to the secure supplier portal, and no outgoing email may carry an account number, IBAN,
  sort code, SWIFT/BIC or routing number, approved or not.

Nothing here is read from config, so no family row can switch it off. Detection reuses the inbound screen, biased the
same way: a false hold costs a person a glance, a miss costs a payment.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .inbound import _normalise, screen_payment_change

PORTAL_GUIDANCE = ("Bank and payment details are never sent or changed by email. Point the supplier to the secure supplier "
                   "portal, and include no account numbers.")

# An IBAN is accepted only if its mod-97 check holds, so a long reference that merely looks like one does not count.
_IBAN = re.compile(r"\b([a-z]{2}\d{2}(?:\s?[a-z0-9]{4}){2,7}(?:\s?[a-z0-9]{1,4})?)\b")
_CUED = {
    "sort_code": re.compile(r"sort\s*code\s*(?:is|:|-)?\s*\d{2}[-\s]?\d{2}[-\s]?\d{2}\b"),
    "account_number": re.compile(r"(?:account|acct|a/c)\s*(?:number|no\.?|#)?\s*(?:is|:|-)?\s*\d[\d\s-]{4,16}\d\b"),
    "swift_bic": re.compile(r"\b(?:swift|bic)(?:\s*code)?\s*(?:is|:|-)?\s*[a-z]{6}[a-z0-9]{2}(?:[a-z0-9]{3})?\b"),
    "routing_number": re.compile(r"\b(?:routing|aba)\s*(?:number|no\.?)?\s*(?:is|:|-)?\s*\d{9}\b"),
}
_PORTAL = re.compile(r"\bportal\b")


def _iban_ok(candidate: str) -> bool:
    s = re.sub(r"\s+", "", candidate).upper()
    if not 15 <= len(s) <= 34:
        return False
    digits = "".join(str(int(ch, 36)) for ch in s[4:] + s[:4])
    return int(digits) % 97 == 1


def account_details(*texts: Any) -> List[str]:
    """Which kinds of bank detail appear (never the values themselves, so nothing leaks through a log or a record)."""

    text = _normalise(" \n".join(t for t in texts if isinstance(t, str)))
    kinds = [k for k, rx in _CUED.items() if rx.search(text)]
    if any(_iban_ok(m.group(1)) for m in _IBAN.finditer(text)):
        kinds.insert(0, "iban")
    return kinds


def mentions_change(*texts: Any) -> Optional[Dict[str, Any]]:
    """The screen's verdict when any text announces or asks for new/changed payment details, else None."""

    joined = " \n".join(t for t in texts if isinstance(t, str))
    s = screen_payment_change("", joined)
    return {"kinds": s["kinds"], "terms": s["terms"]} if s["suspected"] else None


def points_to_portal(text: Any) -> bool:
    return bool(_PORTAL.search(_normalise(text if isinstance(text, str) else "")))


def hold(draft_text: Any, request_texts: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
    """Why this outgoing email must be held for a person, or None. Looks at the draft and at what the person asked for."""

    change = mentions_change(draft_text, *list(request_texts))
    details = account_details(draft_text)
    if not change and not details:
        return None
    return {"payment_change": change, "account_details": details, "guidance": PORTAL_GUIDANCE}


def violations(draft_text: Any, request_texts: Iterable[Any] = ()) -> List[Dict[str, str]]:
    """The failing violations the rule adds to a draft. Severity is always ``fail``; no mode softens it."""

    h = hold(draft_text, request_texts)
    if not h:
        return []
    out: List[Dict[str, str]] = []
    if h["account_details"]:
        out.append({"kind": "bank_account_detail", "severity": "fail",
                    "detail": "the email contains " + ", ".join(k.replace("_", " ") for k in h["account_details"])
                              + ". " + PORTAL_GUIDANCE})
    if h["payment_change"]:
        out.append({"kind": "payment_details_change", "severity": "fail",
                    "detail": "the email or the request mentions new or changed bank or payment details; a person must "
                              "review it before it is sent. " + PORTAL_GUIDANCE})
        if not points_to_portal(draft_text):
            out.append({"kind": "payment_details_no_portal", "severity": "fail",
                        "detail": "the email does not point the supplier to the secure supplier portal"})
    return out
