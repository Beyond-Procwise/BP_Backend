# tests/services/mail_m365/test_consent.py
import json
from urllib.parse import parse_qs, urlparse

import pytest

from services.mail_m365 import consent
from services.mail_m365.config import M365Settings

T1 = "11111111-2222-3333-4444-555555555555"
T2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
KEY = b"k" * 32
S = M365Settings("app-client-id", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class MemConn:
    """bp_admin_config in memory: mirrors the four statements config.py issues."""
    autocommit = True

    def __init__(self):
        self.value = None

    def _obj(self):
        return json.loads(self.value) if self.value else {}

    def cursor(self):
        conn = self

        class C:
            rowcount = 0
            _ret = None

            def execute(self, sql, params):
                if sql.startswith("INSERT") and "jsonb_build_object" in sql:   # append_nonce
                    cur = conn._obj()
                    pend = (cur.get("pending_nonces") or []) + [params[1]]
                    cur["pending_nonces"] = pend[-params[4]:]
                    conn.value = json.dumps(cur)
                elif sql.startswith("INSERT"):                                   # write_integration
                    conn.value = params[1]
                elif sql.startswith("UPDATE") and "||" in sql:                   # merge_integration
                    cur = conn._obj()
                    cur.update(json.loads(params[0]))
                    conn.value = json.dumps(cur)
                    self._ret = (conn.value,)
                elif sql.startswith("UPDATE"):                                   # consume_nonce
                    cur = conn._obj()
                    pend = cur.get("pending_nonces") or []
                    if params[0] in pend:
                        pend.remove(params[0])
                        conn.value = json.dumps(cur)
                        self.rowcount = 1
                    else:
                        self.rowcount = 0
                else:
                    self._ret = (conn.value,) if conn.value else None

            def fetchone(self):
                return self._ret

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
    out = consent.complete(conn, S, KEY, state=state, tenant=T1,
                           admin_consent="True", error=None, now=lambda: 120.0)
    assert out["tenant_id"] == T1 and out["status"] == "connected"
    assert out["consented_by"] == "admin@x"
    with pytest.raises(consent.StateRejected) as e:   # Review Focus 2: replay
        consent.complete(conn, S, KEY, state=state, tenant=T1,
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


def test_bad_tenant_is_refused_and_not_stored():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    with pytest.raises(consent.ConsentNotGiven) as e:
        consent.complete(conn, S, KEY, state=state, tenant="t\nRemove-Item *",
                         admin_consent="True", error=None, now=lambda: 120.0)
    assert e.value.code == "bad_tenant"
    assert "tenant_id" not in json.loads(conn.value)


def _connect(conn, tenant, t0):
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: t0))
    return consent.complete(conn, S, KEY, state=state, tenant=tenant,
                            admin_consent="True", error=None, now=lambda: t0 + 5)


def test_different_tenant_cannot_replace_a_connected_one():
    conn = MemConn()
    _connect(conn, T1, 100.0)
    with pytest.raises(consent.ConsentNotGiven) as e:
        _connect(conn, T2, 200.0)
    assert e.value.code == "tenant_mismatch"
    assert json.loads(conn.value)["tenant_id"] == T1


def test_same_tenant_may_reconsent():
    conn = MemConn()
    _connect(conn, T1, 100.0)
    assert _connect(conn, T1, 200.0)["tenant_id"] == T1


def test_complete_does_not_resurrect_a_nonce_spent_meanwhile():
    conn = MemConn()
    s1 = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    s2 = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    consent.complete(conn, S, KEY, state=s1, tenant=T1, admin_consent="True",
                     error=None, now=lambda: 110.0)
    # s1 is spent; completing never rewrote the pending list from a stale read
    assert len(json.loads(conn.value)["pending_nonces"]) == 1
    with pytest.raises(consent.StateRejected):
        consent.complete(conn, S, KEY, state=s1, tenant=T1, admin_consent="True",
                         error=None, now=lambda: 111.0)


def test_non_ascii_signature_is_bad_signature_not_a_crash():
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    with pytest.raises(consent.StateRejected) as e:
        consent.verify_state(KEY, s.split(".")[0] + ".\u00e9\u00e9", now=101.0)
    assert e.value.code == "bad_signature"


def test_signed_non_object_payload_is_malformed():
    body = consent._b64(b"[1,2]")
    sig = consent._b64(consent.hmac.new(KEY, body.encode(), consent.hashlib.sha256).digest())
    with pytest.raises(consent.StateRejected) as e:
        consent.verify_state(KEY, f"{body}.{sig}", now=1.0)
    assert e.value.code == "malformed"
