"""Where Microsoft 365 mail is configured.

Two places, deliberately apart. The server's own settings (our app id, where its secret
lives, the public address Microsoft returns to) come from the environment: they are the
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
    secret_arn: str
    public_base_url: str
    state_key_arn: str
    poll_seconds: int


def load_settings() -> Optional[M365Settings]:
    values = {k: os.getenv(k, "").strip() for k in (
        "M365_CLIENT_ID", "M365_CLIENT_SECRET_ARN", "PUBLIC_BASE_URL",
        "MAIL_STATE_SIGNING_KEY_ARN")}
    if not all(values.values()):
        return None
    try:
        poll = max(30, int(os.getenv("M365_POLL_SECONDS", "120")))
    except ValueError:
        poll = 120
    return M365Settings(values["M365_CLIENT_ID"], values["M365_CLIENT_SECRET_ARN"],
                        values["PUBLIC_BASE_URL"], values["MAIL_STATE_SIGNING_KEY_ARN"], poll)


def redirect_uri(settings: M365Settings) -> str:
    base = settings.public_base_url.rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("PUBLIC_BASE_URL must be an https address — Microsoft will not "
                         "return an admin to anything else")
    return f"{base}/admin/integrations/mail/m365/callback"


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
