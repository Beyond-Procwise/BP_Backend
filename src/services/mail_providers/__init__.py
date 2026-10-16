"""Email providers. ``get_provider`` says which one this company has connected."""
from __future__ import annotations

from typing import Any, Optional

from services.mail_providers.base import MailProvider
from services.mail_m365.config import read_integration

__all__ = ["MailProvider", "get_provider"]


def get_provider(conn: Any) -> Optional[MailProvider]:
    """The connected provider, or None when email was never connected."""
    if read_integration(conn).get("provider") == "m365":
        from services.mail_providers.m365 import M365Provider
        return M365Provider()
    return None
