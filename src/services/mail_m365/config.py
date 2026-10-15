"""Where Microsoft 365 mail is configured.

Two places, deliberately apart. The server's own settings (our app id, where its
certificate lives, the public address Microsoft returns to) come from the environment: they are the
same for every customer. What the customer's admin did — which tenant approved us, when,
who — is company configuration and lives in proc.bp_admin_config['mail_integration'],
next to the other admin settings.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

CONFIG_KEY = "mail_integration"


@dataclass(frozen=True)
class M365Settings:
    client_id: str
    # Secrets Manager ARN of our app's certificate: {"client_id", "private_key_pem",
    # "thumbprint_sha1_b64url"}. Microsoft sign-in uses this certificate; no client secret.
    cert_arn: str
    public_base_url: str
    state_key_arn: str
    poll_seconds: int
    # The screen's own address, when it is not the server's (the UI and the API run on
    # different origins). The admin's browser lands here after Microsoft's pages.
    app_base_url: str = ""


def load_settings() -> Optional[M365Settings]:
    values = {k: os.getenv(k, "").strip() for k in (
        "M365_CLIENT_ID", "M365_CLIENT_CERT_ARN", "PUBLIC_BASE_URL",
        "MAIL_STATE_SIGNING_KEY_ARN")}
    if not all(values.values()):
        return None
    try:
        poll = max(30, int(os.getenv("M365_POLL_SECONDS", "120")))
    except ValueError:
        poll = 120
    return M365Settings(values["M365_CLIENT_ID"], values["M365_CLIENT_CERT_ARN"],
                        values["PUBLIC_BASE_URL"], values["MAIL_STATE_SIGNING_KEY_ARN"], poll,
                        os.getenv("APP_BASE_URL", "").strip())


def redirect_uri(settings: M365Settings) -> str:
    base = settings.public_base_url.rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("PUBLIC_BASE_URL must be an https address — Microsoft will not "
                         "return an admin to anything else")
    return f"{base}/admin/integrations/mail/m365/callback"


def settings_page_url(settings: M365Settings) -> str:
    """Where the admin's browser is sent once Microsoft is done with it: the screen's
    Settings view. APP_BASE_URL when set (the UI runs on its own origin), else the server."""
    base = (settings.app_base_url or settings.public_base_url).rstrip("/")
    return f"{base}/?view=settings"


def read_integration(conn: Any) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = %s",
                (CONFIG_KEY,))
    row = cur.fetchone()
    cur.close()
    if not row:
        return {}
    value = row["config_value"] if isinstance(row, dict) else row[0]
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def write_integration(conn: Any, value: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) "
        "VALUES (%s, %s::jsonb, %s) ON CONFLICT (config_key) DO UPDATE "
        "SET config_value = EXCLUDED.config_value, "
        "last_modified_by = EXCLUDED.last_modified_by, last_modified_date = now()",
        (CONFIG_KEY, json.dumps(value, default=str), actor))
    cur.close()
    if not getattr(conn, "autocommit", False):
        conn.commit()
    return value


def consume_nonce(conn: Any, nonce: str, *, actor: str) -> bool:
    """Spend a pending consent nonce, atomically. True iff it was pending.

    One UPDATE whose WHERE requires the nonce to be present, so two callbacks racing
    with the same state cannot both win (the row lock serialises them and the loser
    re-checks the WHERE against the already-spent array).
    """
    cur = conn.cursor()
    cur.execute(
        "UPDATE proc.bp_admin_config "
        "SET config_value = jsonb_set(config_value, '{pending_nonces}', "
        "(config_value->'pending_nonces') - %s), "
        "last_modified_by = %s, last_modified_date = now() "
        "WHERE config_key = %s AND config_value->'pending_nonces' ? %s",
        (nonce, actor, CONFIG_KEY, nonce))
    spent = cur.rowcount == 1
    cur.close()
    if not getattr(conn, "autocommit", False):
        conn.commit()
    return spent


def append_nonce(conn: Any, nonce: str, *, actor: str, keep: int = 20) -> None:
    """Add a pending nonce in one statement, trimmed to the newest `keep`.

    Appending inside SQL (not read-modify-write) means a callback spending another
    nonce at the same moment cannot be undone by this write.
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) "
        "VALUES (%s, jsonb_build_object('pending_nonces', jsonb_build_array(%s::text)), %s) "
        "ON CONFLICT (config_key) DO UPDATE SET config_value = jsonb_set("
        "proc.bp_admin_config.config_value, '{pending_nonces}', "
        "(SELECT coalesce(jsonb_agg(e ORDER BY o), '[]'::jsonb) FROM ("
        "SELECT e, o FROM jsonb_array_elements(coalesce("
        "proc.bp_admin_config.config_value->'pending_nonces', '[]'::jsonb) "
        "|| to_jsonb(%s::text)) WITH ORDINALITY AS t(e, o) "
        "ORDER BY o DESC LIMIT %s) s)), "
        "last_modified_by = %s, last_modified_date = now()",
        (CONFIG_KEY, nonce, actor, nonce, keep, actor))
    cur.close()
    if not getattr(conn, "autocommit", False):
        conn.commit()


def merge_integration(conn: Any, fields: Dict[str, Any], *, actor: str,
                      unless_other_tenant: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Merge top-level `fields` into the stored row without touching other keys
    (notably pending_nonces, which other requests may be changing). Returns the row,
    or None when nothing was updated: no row, or (with `unless_other_tenant`) the row
    is already connected to a different tenant. The tenant test is in the WHERE, so
    it is atomic with the write."""
    guard, extra = "", ()
    if unless_other_tenant is not None:
        guard = (" AND NOT (config_value->>'status' = 'connected' "
                 "AND coalesce(config_value->>'tenant_id', '') <> '' "
                 "AND lower(config_value->>'tenant_id') <> lower(%s))")
        extra = (unless_other_tenant,)
    cur = conn.cursor()
    cur.execute(
        "UPDATE proc.bp_admin_config SET config_value = config_value || %s::jsonb, "
        "last_modified_by = %s, last_modified_date = now() "
        "WHERE config_key = %s" + guard + " RETURNING config_value",
        (json.dumps(fields, default=str), actor, CONFIG_KEY) + extra)
    row = cur.fetchone()
    cur.close()
    if not getattr(conn, "autocommit", False):
        conn.commit()
    if not row:
        return None
    value = row["config_value"] if isinstance(row, dict) else row[0]
    return json.loads(value) if isinstance(value, str) else dict(value or {})
