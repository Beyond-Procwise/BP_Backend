# tests/services/mail_m365/test_consent.py
from urllib.parse import parse_qs, urlparse

import pytest

from services.mail_m365 import consent
from services.mail_m365.config import M365Settings

KEY = b"k" * 32
S = M365Settings("app-client-id", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class MemConn:
    """bp_admin_config in memory, enough for read_integration/write_integration."""
    autocommit = True

    def __init__(self):
        self.value = None

    def cursor(self):
        conn = self

        class C:
            rowcount = 0

            def execute(self, sql, params):
                if sql.startswith("INSERT"):
                    conn.value = params[1]
                elif sql.startswith("UPDATE"):
                    import json
                    nonce = params[0]
                    cur = json.loads(conn.value) if conn.value else {}
                    pend = cur.get("pending_nonces") or []
                    if nonce in pend:
                        pend.remove(nonce)
                        conn.value = json.dumps(cur)
                        self.rowcount = 1
                    else:
                        self.rowcount = 0
                self._sql = sql

            def fetchone(self):
                return (conn.value,) if conn.value else None

            def close(self):
                pass
        return C()


def test_state_round_trip():
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    assert consent.verify_state(KEY, s, now=200.0) == ("admin@x", "n1")


@pytest.mark.parametrize("tamper,code", [
    (lambda s: s[:-2] + "xx", "bad_signature"),
    (lambda s: "garbage", "malformed"),
])
def test_tampered_state_is_rejected(tamper, code):
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    with pytest.raises(consent.StateRejected) as e:
        consent.verify_state(KEY, tamper(s), now=101.0)
    assert e.value.code == code


def test_expired_state_is_rejected():
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    with pytest.raises(consent.StateRejected) as e:
        consent.verify_state(KEY, s, now=100.0 + 601)
    assert e.value.code == "expired"


def test_begin_builds_admin_consent_url():
    conn = MemConn()
    url = consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0)
    q = parse_qs(urlparse(url).query)
    assert url.startswith(consent.ADMIN_CONSENT_URL)
    assert q["client_id"] == ["app-client-id"]
    assert q["redirect_uri"] == ["https://app.procwise.co.uk/admin/integrations/mail/m365/callback"]
    assert q["scope"] == ["https://graph.microsoft.com/.default"]


def _state_from(url):
    return parse_qs(urlparse(url).query)["state"][0]


def test_complete_records_the_tenant_once():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    out = consent.complete(conn, S, KEY, state=state, tenant="tenant-1",
                           admin_consent="True", error=None, now=lambda: 120.0)
    assert out["tenant_id"] == "tenant-1" and out["status"] == "connected"
    assert out["consented_by"] == "admin@x"
    with pytest.raises(consent.StateRejected) as e:   # Review Focus 2: replay
        consent.complete(conn, S, KEY, state=state, tenant="tenant-1",
                         admin_consent="True", error=None, now=lambda: 121.0)
    assert e.value.code == "reused"


def test_declined_consent_stores_nothing():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    with pytest.raises(consent.ConsentNotGiven):
        consent.complete(conn, S, KEY, state=state, tenant=None, admin_consent=None,
                         error="access_denied", now=lambda: 120.0)
    from services.mail_m365.config import read_integration
    assert "tenant_id" not in read_integration(conn)
