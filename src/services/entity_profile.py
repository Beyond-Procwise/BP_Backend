"""The selling entity's own details: who a quote is from, where they are registered and
how to pay them. One copy (bp_admin_config `entity_profile`), worn by every document.

Nothing here is defaulted. A field nobody has filled in is None, and the page that
prints the letterhead leaves it out; a placeholder company on a real quote is worse
than a missing line.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

CONFIG_KEY = "entity_profile"
FIELDS = ("legalName", "tradingName", "logoText", "address", "email", "phone", "web",
          "companyNumber", "vatNumber", "footerNote")
BANK_FIELDS = ("name", "sortCode", "accountNumber", "iban")
_LIMITS = {"address": 600, "footerNote": 600, "logoText": 4}


def _text(value: Any, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text")
    limit = _LIMITS.get(what.split(".")[-1], 200)
    s = value.strip()
    if len(s) > limit:
        raise ValueError(f"{what} is longer than {limit} characters")
    return s or None


def shape(stored: Any) -> Dict[str, Any]:
    """Every field the letterhead knows, None where nothing is stored."""
    stored = stored if isinstance(stored, dict) else {}
    bank = stored.get("bank") if isinstance(stored.get("bank"), dict) else {}
    return {**{f: stored.get(f) for f in FIELDS}, "logoId": stored.get("logoId"),
            "bank": {f: bank.get(f) for f in BANK_FIELDS}}


def clean(profile: Dict[str, Any]) -> Dict[str, Any]:
    bank = profile.get("bank") or {}
    if not isinstance(bank, dict):
        raise ValueError("bank must be an object")
    logo = profile.get("logoId")
    if logo is not None and (isinstance(logo, bool) or not isinstance(logo, int) or logo < 1):
        raise ValueError("logoId must be a logo's id")
    return {**{f: _text(profile.get(f), f) for f in FIELDS}, "logoId": logo,
            "bank": {f: _text(bank.get(f), f"bank.{f}") for f in BANK_FIELDS}}


def load(conn: Any) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT config_value, last_modified_by, last_modified_date "
                "FROM proc.bp_admin_config WHERE config_key = %s", (CONFIG_KEY,))
    row = cur.fetchone()
    if row is None:
        return {**shape(None), "updatedBy": None, "updatedAt": None}
    value, by, at = (row["config_value"], row["last_modified_by"], row["last_modified_date"]) \
        if isinstance(row, dict) else row
    if isinstance(value, str):
        value = json.loads(value)
    return {**shape(value), "updatedBy": by, "updatedAt": at}


def save(conn: Any, profile: Dict[str, Any], *, actor: Optional[str]) -> Dict[str, Any]:
    """Replace the profile. The whole of it: a field left out is a field cleared, so the
    stored copy is always exactly what the last editor saw on the form."""
    cleaned = clean(profile)
    cur = conn.cursor()
    try:
        if cleaned["logoId"] is not None:
            from src.services import brand_logos
            if not brand_logos.exists(cur, cleaned["logoId"]):
                raise ValueError(f"logo {cleaned['logoId']} is not one of the company's logos")
        cur.execute(
            "INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) "
            "VALUES (%s, %s::jsonb, %s) ON CONFLICT (config_key) DO UPDATE "
            "SET config_value = EXCLUDED.config_value, "
            "last_modified_by = EXCLUDED.last_modified_by, last_modified_date = now()",
            (CONFIG_KEY, json.dumps(cleaned), actor))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return load(conn)
