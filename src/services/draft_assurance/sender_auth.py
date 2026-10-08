"""Who an inbound reply is really from.

Two separate questions, both answered without trusting anything the sender controls:

1. Did the message authenticate? Read the SPF / DKIM / DMARC results from an ``Authentication-Results`` header, but ONLY one stamped by an
   authserv-id we trust (our own receiver). A sender can put a forged ``Authentication-Results: ...; dmarc=pass`` line inside a message, so
   an untrusted one is ignored; where several trusted lines disagree the WORST result wins; an unrecognised word is never a pass.
2. Is the sender the supplier? A lookalike domain passes DMARC perfectly well. The From domain is compared with the domains on the supplier
   master (the same domain or a subdomain, never a mere suffix of text).

Every checked reply gets a row recording the results (``bp_inbound_auth``). Whether a reply is HELD for a person is governed config
(``EmailSenderAuthRules``): shadow mode records only; each of a hard failure, nothing stamped, and a domain mismatch has its own switch,
because the second and third depend on how real mail looks and are switched on once that has been seen. A held reply becomes a flag in
``bp_inbound_flag`` (kind ``sender_not_verified``), which blocks drafting and sending on its thread exactly as a payment-detail flag does.
"""

from __future__ import annotations

import logging
import re
from email.utils import parseaddr
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)
SLUG = "email_sender_auth_rules"
METHODS = ("spf", "dkim", "dmarc")
# worst first: where trusted headers (or repeated entries) disagree, the earliest in this list wins
_SEVERITY = ["fail", "permerror", "temperror", "softfail", "unknown", "neutral", "none", "pass"]
_WORDS = {"pass", "fail", "softfail", "neutral", "none", "temperror", "permerror"}
_RESULT = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([A-Za-z]+)", re.I)
_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


class SenderAuthRulesUnavailable(RuntimeError):
    """The switches are missing or unusable: nothing is held on a guess."""


def load_rules(policy_engine: Any) -> Dict[str, Any]:
    if policy_engine is None:
        raise SenderAuthRulesUnavailable("no policy engine available")
    try:
        policy = policy_engine.get_policy(SLUG)
    except Exception as exc:  # noqa: BLE001
        raise SenderAuthRulesUnavailable(f"policy store unreadable: {exc}") from exc
    rules = ((policy or {}).get("details") or {}).get("rules") if isinstance(policy, dict) else None
    if not isinstance(rules, dict):
        raise SenderAuthRulesUnavailable("no sender-authentication rules are defined")
    if rules.get("mode") not in ("shadow", "enforce"):
        raise SenderAuthRulesUnavailable("mode must be shadow or enforce")
    ids = rules.get("trusted_authserv_ids")
    if not (isinstance(ids, list) and ids and all(isinstance(i, str) and i.strip() for i in ids)):
        raise SenderAuthRulesUnavailable("trusted_authserv_ids must be a non-empty list of text values")
    out: Dict[str, Any] = {"mode": rules["mode"], "trusted_authserv_ids": [i.strip().lower() for i in ids]}
    for key in ("hold_on_fail", "hold_on_missing", "hold_on_domain_mismatch"):
        if not isinstance(rules.get(key), bool):
            raise SenderAuthRulesUnavailable(f"{key} must be true or false")
        out[key] = rules[key]
    return out


def _values(raw_headers: Any, name: str) -> List[str]:
    if not isinstance(raw_headers, Mapping):
        return []
    out: List[str] = []
    for key, value in raw_headers.items():
        if isinstance(key, str) and key.lower() == name:
            items = [value] if isinstance(value, str) else (list(value) if isinstance(value, (list, tuple)) else [])
            out += [v for v in items if isinstance(v, str)]
    return out


def _word(raw: str) -> str:
    w = raw.lower()
    if w == "hardfail":
        return "fail"
    return w if w in _WORDS else "unknown"


def parse_results(raw_headers: Any, trusted_ids: Iterable[str]) -> Dict[str, Any]:
    """{"spf","dkim","dmarc": a result word or "missing", "trusted_headers": n, "ignored_headers": n}. Never raises."""

    trusted = {str(t).strip().lower() for t in (trusted_ids or []) if str(t).strip()}
    found: Dict[str, str] = {}
    n_trusted = n_ignored = 0
    for value in _values(raw_headers, "authentication-results"):
        text = re.sub(r"\s+", " ", value).strip()
        serv, _, rest = text.partition(";")
        serv_id = serv.strip().split(" ")[0].lower() if serv.strip() else ""
        if serv_id not in trusted:
            n_ignored += 1
            continue
        n_trusted += 1
        for method, word in _RESULT.findall(rest):
            method, word = method.lower(), _word(word)
            prior = found.get(method)
            if prior is None or _SEVERITY.index(word) < _SEVERITY.index(prior):
                found[method] = word
    out: Dict[str, Any] = {m: found.get(m, "missing") for m in METHODS}
    out.update({"trusted_headers": n_trusted, "ignored_headers": n_ignored})
    return out


def from_domain(raw_headers: Any, fallback_address: Optional[str]) -> Optional[str]:
    """The domain in the From header. None if there are two From headers (ambiguous) or it is not an address."""

    froms = _values(raw_headers, "from")
    if len(froms) > 1:
        return None
    candidate = froms[0] if froms else fallback_address
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    address = parseaddr(candidate)[1]
    if "@" not in address:
        return None
    domain = address.rsplit("@", 1)[1].strip().lower().rstrip(".")
    return domain if _DOMAIN.match(domain) else None


def domain_matches(domain: Optional[str], known: Optional[Iterable[str]]) -> Optional[bool]:
    """True if ``domain`` is one of the supplier's domains or a subdomain of one; False if not; None if either side is unknown."""

    ks = [k.strip().lower().rstrip(".") for k in (known or []) if isinstance(k, str) and k.strip()]
    if not isinstance(domain, str) or not domain.strip() or not ks:
        return None
    domain = domain.strip().lower().rstrip(".")
    return any(domain == k or domain.endswith("." + k) for k in ks)


def decide(results: Mapping[str, str], match: Optional[bool], rules: Mapping[str, Any]) -> Dict[str, Any]:
    spf, dkim, dmarc = (results.get(m, "missing") for m in METHODS)
    if dmarc == "pass":
        verdict = "authenticated"                       # DMARC passes when SPF or DKIM passes in alignment: one failing beside it is normal
    elif "fail" in (spf, dkim, dmarc):
        verdict = "failed"
    elif dkim == "pass" and spf == "pass":
        verdict = "authenticated"
    elif (spf, dkim, dmarc) == ("missing",) * 3:
        verdict = "missing"
    else:
        verdict = "inconclusive"
    reasons: List[str] = []
    if verdict == "failed":
        reasons.append("auth_failed")
    elif verdict in ("missing", "inconclusive"):
        reasons.append("auth_missing")
    if match is False:
        reasons.append("domain_mismatch")
    wanted = {"auth_failed": rules.get("hold_on_fail"), "auth_missing": rules.get("hold_on_missing"),
              "domain_mismatch": rules.get("hold_on_domain_mismatch")}
    hold = rules.get("mode") == "enforce" and any(wanted.get(r) for r in reasons)
    return {"verdict": verdict, "hold": bool(hold), "reasons": reasons}


# --- recording --------------------------------------------------------------------------------------------------------------------

def known_domains(conn: Any, supplier_id: Optional[str]) -> Optional[List[str]]:
    """The email domains on the supplier master for this supplier, or None if they cannot be read."""

    if not supplier_id:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT contact_email_1, contact_email_2 FROM proc.bp_supplier WHERE supplier_id = %s", (supplier_id,))
            domains = []
            for row in cur.fetchall():
                for addr in row:
                    if isinstance(addr, str) and "@" in addr:
                        domains.append(addr.rsplit("@", 1)[1].strip().lower())
            return sorted(set(domains)) or None
    except Exception:  # noqa: BLE001
        logger.debug("supplier domains unreadable", exc_info=True)
        return None


def record_result(conn: Any, *, workflow_id, unique_id, supplier_id, message_id, results: Mapping[str, Any], verdict: Mapping[str, Any],
                  domain: Optional[str], match: Optional[bool]) -> Optional[int]:
    import json

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO email_agent.bp_inbound_auth (workflow_id, unique_id, supplier_id, response_message_id, spf, dkim, dmarc, verdict,
                   from_domain, domain_match, reasons, held, trusted_headers, ignored_headers)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)
               ON CONFLICT (workflow_id, response_message_id) WHERE response_message_id IS NOT NULL DO NOTHING RETURNING auth_id""",
            (workflow_id, unique_id, supplier_id, message_id, results["spf"], results["dkim"], results["dmarc"], verdict["verdict"],
             domain, match, json.dumps(verdict["reasons"]), verdict["hold"], results["trusted_headers"], results["ignored_headers"]))
        got = cur.fetchone()
    return int(got[0]) if got else None


def check_and_record(row: Any, conn_factory: Any = None, policy_engine: Any = None) -> Optional[int]:
    """Check an inbound reply's sender, record the result, and hold it if the rules say so. Returns the flag id when held.

    NEVER raises and never delays the ingest beyond the check: a fault costs a missed check (logged), not a lost reply. No rules means the
    feature is off and nothing is recorded or held.
    """

    try:
        if row is None:
            return None
        from . import inbound

        if policy_engine is None:
            from src.services import rbac
            policy_engine = rbac.policy_engine()
        try:
            rules = load_rules(policy_engine)
        except SenderAuthRulesUnavailable as exc:
            logger.debug("sender authentication not checked: %s", exc)
            return None
        headers = getattr(row, "raw_headers", None)
        results = parse_results(headers, rules["trusted_authserv_ids"])
        domain = from_domain(headers, getattr(row, "response_from", None))
        with (conn_factory or inbound._default_factory)() as conn:
            match = domain_matches(domain, known_domains(conn, getattr(row, "supplier_id", None)))
            verdict = decide(results, match, rules)
            ids = dict(workflow_id=getattr(row, "workflow_id", None), unique_id=getattr(row, "unique_id", None),
                       supplier_id=getattr(row, "supplier_id", None), message_id=getattr(row, "response_message_id", None))
            fresh = record_result(conn, **ids, results=results, verdict=verdict, domain=domain, match=match)
            if verdict["hold"] and fresh is not None:         # a message already checked is not flagged a second time
                return inbound.record_flag(conn, kind="sender_not_verified", **ids, result={"kinds": verdict["reasons"], "terms": []})
        return None
    except Exception as exc:  # noqa: BLE001
        if "bp_inbound_auth" in str(exc) and "does not exist" in str(exc):
            logger.debug("sender check not recorded: the email_agent schema is not applied")
        else:
            logger.exception("sender authentication check failed; the reply was stored regardless")
        return None
