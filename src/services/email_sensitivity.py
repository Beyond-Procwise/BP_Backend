"""Classify outbound email content, deterministically.

The detectors here are code because a gate must be reproducible: the same
message must classify the same way every time, and it must still classify when
the model host is down. Which detectors run, and what class each raises the
content to, is policy — so tightening a rule is a row edit, not a deploy.

An error inside a detector yields ``undetermined``, which policy maps to a
denial. A classifier that cannot decide must not be read as "safe".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CLASS_UNDETERMINED = "undetermined"

_SLUG = "email_sensitivity"

# A clause number followed by a contract-style heading, or the stock phrases
# that only appear in contractual prose.
#
# The clause-number pattern is anchored to the start of a line: a numbered
# contract clause always opens its own line, while an ordinary decimal in
# prose (e.g. "Delivery is 3.5 Working Days.") sits mid-sentence. Without the
# anchor, the pattern fires on routine quote text and would block it to every
# internal-clearance supplier.
_CONTRACT_PATTERNS = (
    re.compile(r"^\s*\d+\.\d+\s+[A-Z][A-Za-z ]{3,40}\.", re.MULTILINE),
    re.compile(
        r"\b(limitation of liability|indemnif(y|ication)|termination for convenience"
        r"|governing law|confidentiality obligations|force majeure"
        r"|consequential loss|this agreement)\b",
        re.IGNORECASE,
    ),
)

# The final label must not absorb a trailing sentence-ending dot: without the
# character class on the last segment, "jane.doe@ourcompany.com." captures a
# domain of "ourcompany.com." (with the full stop attached), which then fails
# to match the configured domain "ourcompany.com" and the detector never
# fires -- on exactly the case, an address at the end of a sentence, where it
# matters most.
_EMAIL_RE = re.compile(r"[\w.+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")


@dataclass(frozen=True)
class ClassificationResult:
    content_class: str
    detectors_fired: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


def _rules(policy_engine: Optional[Any]) -> Dict[str, Any]:
    engine = policy_engine
    if engine is None:
        try:
            from src.engines.policy_engine import PolicyEngine
            from src.services.db import get_conn

            engine = PolicyEngine(connection_factory=get_conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("email_sensitivity: no PolicyEngine: %s", exc)
            return {}
    try:
        policy = engine.get_policy(_SLUG)
    except Exception as exc:  # noqa: BLE001
        logger.error("email_sensitivity: get_policy failed: %s", exc)
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def _order(rules: Dict[str, Any]) -> Dict[str, int]:
    order = rules.get("order")
    if not isinstance(order, dict):
        return {}
    out: Dict[str, int] = {}
    for name, rank in order.items():
        try:
            out[str(name)] = int(rank)
        except (TypeError, ValueError):
            continue
    return out


def _digits(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def _detect_third_party_price(
    text: str, recipient_supplier_id: Optional[str], peer_prices: Iterable[Dict[str, Any]]
) -> Optional[str]:
    """Fires when a figure belonging to another supplier appears in the text."""

    recipient = str(recipient_supplier_id or "").strip()
    haystack = _digits(text)
    for entry in peer_prices or []:
        if not isinstance(entry, dict):
            continue
        owner = str(entry.get("supplier_id") or "").strip()
        if owner and owner == recipient:
            continue
        amount = _digits(entry.get("amount"))
        if len(amount) >= 3 and amount in haystack:
            return f"{owner or 'another supplier'}:{entry.get('amount')}"
    return None


def _detect_contract_prose(text: str) -> Optional[str]:
    for pattern in _CONTRACT_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(0)[:80]
    return None


def _detect_internal_staff_contact(
    text: str, internal_domains: Iterable[str]
) -> Optional[str]:
    """An address on one of our own domains appearing in outbound content."""

    domains = {str(d).strip().lower() for d in (internal_domains or []) if str(d).strip()}
    if not domains:
        return None
    for match in _EMAIL_RE.finditer(text or ""):
        if match.group(1).lower() in domains:
            return match.group(0)
    return None


def _detect_source_document_attached(
    attachments: Optional[Iterable[Any]],
) -> Optional[str]:
    for attachment in attachments or []:
        if isinstance(attachment, dict) and attachment.get("is_source_document"):
            return str(attachment.get("filename") or "attachment")
    return None


def classify(
    subject: Optional[str],
    body: Optional[str],
    attachments: Optional[Iterable[Any]],
    recipient_supplier_id: Optional[str],
    peer_prices: Optional[Iterable[Dict[str, Any]]] = None,
    internal_domains: Optional[Iterable[str]] = None,
    policy_engine: Optional[Any] = None,
) -> ClassificationResult:
    """Classify a message. Returns ``undetermined`` rather than guessing."""

    rules = _rules(policy_engine)
    if not rules:
        return ClassificationResult(
            content_class=CLASS_UNDETERMINED,
            evidence={"error": "email_sensitivity policy unavailable"},
        )

    detectors = rules.get("detectors")
    detectors = detectors if isinstance(detectors, dict) else {}
    order = _order(rules)
    text = f"{subject or ''}\n{body or ''}"

    fired: List[str] = []
    evidence: Dict[str, Any] = {}
    best_class = "internal"
    best_rank = order.get("internal", 2)

    checks = {
        "third_party_price": lambda: _detect_third_party_price(
            text, recipient_supplier_id, peer_prices or []
        ),
        "contract_prose": lambda: _detect_contract_prose(text),
        "internal_staff_contact": lambda: _detect_internal_staff_contact(
            text, internal_domains or []
        ),
        "source_document_attached": lambda: _detect_source_document_attached(attachments),
    }

    for name, check in checks.items():
        config = detectors.get(name)
        if not isinstance(config, dict) or not config.get("enabled"):
            continue
        try:
            hit = check()
        except Exception as exc:  # noqa: BLE001 - cannot decide means deny
            logger.error("email_sensitivity: detector %s failed: %s", name, exc)
            return ClassificationResult(
                content_class=CLASS_UNDETERMINED,
                detectors_fired=fired,
                evidence={"error": f"detector {name} failed: {exc}"},
            )
        if not hit:
            continue
        fired.append(name)
        evidence[name] = hit
        raised = str(config.get("raises_to") or "")
        rank = order.get(raised, 0)
        if rank > best_rank:
            best_class, best_rank = raised, rank

    return ClassificationResult(
        content_class=best_class, detectors_fired=fired, evidence=evidence
    )


def clearance_permits(
    content_class: str,
    clearance: Optional[str],
    policy_engine: Optional[Any] = None,
) -> bool:
    """True when ``content_class`` may be sent to a supplier at ``clearance``."""

    rules = _rules(policy_engine)
    if not rules:
        return False
    if str(content_class) == CLASS_UNDETERMINED:
        return False

    order = _order(rules)
    effective = str(clearance or "").strip() or str(
        rules.get("default_supplier_clearance") or ""
    )

    content_rank = order.get(str(content_class))
    clearance_rank = order.get(effective)
    if content_rank is None or clearance_rank is None:
        return False
    return content_rank <= clearance_rank
