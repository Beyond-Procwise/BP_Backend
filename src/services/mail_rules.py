"""Send-time rules for supplier email leaving from the company mailbox.

Pure: the caller supplies the policy, the supplier lookup and the size limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: Optional[str]
    needs_internal_approval: bool


def bare_address(raw: str) -> str:
    return parseaddr(raw or "")[1].strip().lower()


def classify(address: str, policy, *, is_supplier: Callable[[str], bool]) -> str:
    address = bare_address(address)
    domain = address.rpartition("@")[2] if "@" in address else ""
    if domain and domain in {d.strip().lower() for d in policy.company_domains}:
        return "internal"
    return "supplier" if is_supplier(address) else "other"


def check_supplier_email(recipients: List[str], size_bytes: int, policy, *,
                         is_supplier: Callable[[str], bool], size_limit: int) -> Verdict:
    if size_bytes > size_limit:
        return Verdict(False, f"Too large to send from your company mailbox "
                              f"(limit {size_limit // 1_000_000} MB)", False)
    for raw in recipients:
        addr = bare_address(raw)
        if "@" not in addr or addr.startswith("@") or addr.endswith("@"):
            return Verdict(False, f"'{raw}' is not a single email address", False)
    kinds = [(bare_address(r), classify(r, policy, is_supplier=is_supplier)) for r in recipients]
    colleagues = [a for a, k in kinds if k == "internal"]
    if colleagues:
        if getattr(policy, "internal_send", "with_user_approval") != "with_user_approval":
            return Verdict(False, f"Emails to colleagues are switched off in your email "
                                  f"policy, so {colleagues[0]} cannot be on this email", False)
        return Verdict(False, "Emails to colleagues go through 'Send to a colleague' "
                              "and need your approval", False)
    if policy.external_recipients == "known_suppliers_only":
        for addr, kind in kinds:
            if kind == "other":
                return Verdict(False, f"{addr} is not a known supplier", False)
    return Verdict(True, None, False)
