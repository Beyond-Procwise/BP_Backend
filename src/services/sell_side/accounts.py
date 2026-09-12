"""The customer we sell TO. Deliberately not bp_supplier (spec §4.4)."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

import psycopg2.errors

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor
from src.services.sell_side.money import iso_currency

_ACCOUNT_FIELDS = ("trading_name", "also_supplier_id", "registration_number", "vat_number",
                   "country", "default_currency", "payment_terms", "credit_limit_amount",
                   "account_owner_email")
SOURCE_KINDS = frozenset({"our_invoices", "customer_shared", "third_party"})
COMPLETENESS = frozenset({"complete", "partial", "unknown"})


def _require_account(cur, account_id: str) -> None:
    cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
    if cur.fetchone() is None:
        raise NotFound(f"account {account_id!r} does not exist")


def create_account(conn: Any, *, account_name: str, account_id: Optional[str] = None,
                   **fields: Any) -> Dict[str, Any]:
    unknown = set(fields) - set(_ACCOUNT_FIELDS)
    if unknown:
        raise ValueError(f"not account fields: {sorted(unknown)}")
    if not (account_name or "").strip():
        raise ValueError("account_name is empty")
    if fields.get("default_currency") is not None:
        fields["default_currency"] = iso_currency(fields["default_currency"])
    account_id = account_id or f"ACC-{uuid.uuid4().hex[:12].upper()}"
    cols = ["account_id", "account_name", *fields]
    cur = dict_cursor(conn)
    try:
        cur.execute(
            f"INSERT INTO proc.bp_account ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))}) RETURNING *",
            (account_id, account_name.strip(), *fields.values()))
        row = dict(cur.fetchone())
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise StateConflict(f"account {account_id!r} already exists") from None
    except psycopg2.errors.ForeignKeyViolation:
        conn.rollback()
        raise ValueError(f"also_supplier_id {fields.get('also_supplier_id')!r} is not a supplier") from None
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def get_account(conn: Any, account_id: str) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute("SELECT * FROM proc.bp_account WHERE account_id = %s", (account_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"account {account_id!r} does not exist")
    out = dict(row)
    cur.execute("SELECT * FROM proc.bp_account_contact WHERE account_id = %s "
                "ORDER BY is_primary DESC, contact_id", (account_id,))
    out["contacts"] = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT * FROM proc.bp_account_history_scope WHERE account_id = %s "
                "ORDER BY source_kind", (account_id,))
    out["history_scope"] = [dict(r) for r in cur.fetchall()]
    return out


def add_contact(conn: Any, account_id: str, *, contact_name: str,
                contact_role: Optional[str] = None, contact_email: Optional[str] = None,
                contact_phone: Optional[str] = None, is_primary: bool = False) -> Dict[str, Any]:
    if not (contact_name or "").strip():
        raise ValueError("contact_name is empty")
    cur = dict_cursor(conn)
    try:
        _require_account(cur, account_id)
        if is_primary:
            cur.execute("UPDATE proc.bp_account_contact SET is_primary = FALSE "
                        "WHERE account_id = %s", (account_id,))
        cur.execute(
            "INSERT INTO proc.bp_account_contact (account_id, contact_name, contact_role, "
            "contact_email, contact_phone, is_primary) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
            (account_id, contact_name.strip(), contact_role, contact_email, contact_phone,
             bool(is_primary)))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def set_history_scope(conn: Any, account_id: str, *, source_kind: str, completeness: str,
                      covers_from: Any = None, covers_to: Any = None,
                      note: Optional[str] = None) -> Dict[str, Any]:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")
    if completeness not in COMPLETENESS:
        raise ValueError(f"completeness must be one of {sorted(COMPLETENESS)}")
    cur = dict_cursor(conn)
    try:
        _require_account(cur, account_id)
        cur.execute(
            "INSERT INTO proc.bp_account_history_scope (account_id, source_kind, covers_from, "
            "covers_to, completeness, note) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (account_id, source_kind) DO UPDATE SET covers_from = EXCLUDED.covers_from, "
            "covers_to = EXCLUDED.covers_to, completeness = EXCLUDED.completeness, "
            "note = EXCLUDED.note RETURNING *",
            (account_id, source_kind, covers_from, covers_to, completeness, note))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row
