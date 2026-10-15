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
                    if "lower(" in sql and cur.get("status") == "connected" \
                            and cur.get("tenant_id") \
                            and cur["tenant_id"].lower() != params[3].lower():
                        self._ret = None
                        return
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


# ---- a stand-in Microsoft: one RSA key, its JWKS, the OpenID config, the token endpoint ----

from cryptography.hazmat.primitives.asymmetric import rsa
import jwt as pyjwt

_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_JWK = json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(_RSA.public_key()))
_JWK.update({"kid": "k1", "use": "sig"})
NOW = 1_000.0


GLOBAL_ADMIN = "62e90394-69f5-4237-9190-012177145e10"
USER_ADMIN = "fe930be7-5e62-47db-91af-98c3a49a38b1"     # an admin role, but not one that may connect


def _id_token(nonce, *, tid=T1, aud="app-client-id", iss=None, exp=NOW + 3600,
              nbf=NOW - 10, key=_RSA, kid="k1", drop=(), wids=(GLOBAL_ADMIN,)):
    claims = {"aud": aud, "iss": iss or f"https://login.microsoftonline.com/{tid}/v2.0",
              "tid": tid, "exp": int(exp), "nbf": int(nbf), "iat": int(nbf),
              "nonce": nonce, "sub": "s", "oid": "o"}
    if wids is not None:
        claims["wids"] = list(wids) if isinstance(wids, (list, tuple)) else wids
    for k in drop:
        claims.pop(k)
    return pyjwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


class FakeMicrosoft:
    """Answers the three sign-in calls. `token` builds the id_token from the nonce the
    authorize URL carried, so a test controls exactly what comes back."""

    def __init__(self, make_token=None, token_status=200):
        self.make_token = make_token or (lambda nonce: _id_token(nonce))
        self.token_status = token_status
        self.nonce = None
        self.calls = []

    def post_form(self, url, data):
        self.calls.append(("POST", url, dict(data)))
        if self.token_status != 200:
            return self.token_status, {"error": "invalid_grant"}
        return 200, {"id_token": self.make_token(self.nonce), "token_type": "Bearer"}

    def get(self, url, headers):
        self.calls.append(("GET", url, None))
        if url.endswith("/.well-known/openid-configuration"):
            tid = url.split("/")[3]
            return 200, {"issuer": f"https://login.microsoftonline.com/{tid}/v2.0",
                         "jwks_uri": f"https://login.microsoftonline.com/{tid}/discovery/v2.0/keys"}
        if url.endswith("/discovery/v2.0/keys"):
            return 200, {"keys": [_JWK]}
        return 404, {}


from tests.services.mail_m365.certs import PUBLIC_KEY, X5T, fake_credentials  # noqa: E402

CREDS = lambda settings: fake_credentials("organizations", "app-client-id")  # noqa: E731


def _consent_return(conn, *, tenant=T1, t0=NOW - 20, actor="admin@x"):
    state = _state_from(consent.begin(conn, S, KEY, actor, now=lambda: t0))
    return consent.handle_callback(conn, S, KEY, state=state, tenant=tenant,
                                   admin_consent="True", now=lambda: t0 + 5)


def _signin_return(conn, signin_url, ms, *, code="the-code", error=None, t=NOW):
    q = parse_qs(urlparse(signin_url).query)
    ms.nonce = q["nonce"][0]
    return consent.handle_callback(conn, S, KEY, state=q["state"][0], code=code, error=error,
                                   transport=ms, app_credentials=CREDS, now=lambda: t)


def _connect(conn, tenant=T1, ms=None, **kw):
    url = _consent_return(conn, tenant=tenant, **kw)
    _signin_return(conn, url, ms or FakeMicrosoft(lambda n: _id_token(n, tid=tenant)))
    from services.mail_m365.config import read_integration
    return read_integration(conn)


def test_consent_return_stores_nothing_and_asks_for_an_openid_sign_in():
    conn = MemConn()
    url = _consent_return(conn)
    p = urlparse(url)
    q = parse_qs(p.query)
    assert f"{p.scheme}://{p.netloc}{p.path}" == \
        f"https://login.microsoftonline.com/{T1}/oauth2/v2.0/authorize"
    assert q["scope"] == ["openid"]                       # no mail permission, ever
    assert q["response_type"] == ["code"] and q["client_id"] == ["app-client-id"]
    assert q["redirect_uri"] == ["https://app.procwise.co.uk/admin/integrations/mail/m365/callback"]
    assert q["nonce"][0] and q["state"][0]
    assert "tenant_id" not in json.loads(conn.value)      # the query's tenant is not stored


def test_happy_path_stores_the_verified_tid():
    conn = MemConn()
    ms = FakeMicrosoft()
    url = _consent_return(conn)
    assert _signin_return(conn, url, ms) is None
    out = json.loads(conn.value)
    assert out["tenant_id"] == T1 and out["status"] == "connected"
    assert out["consented_by"] == "admin@x"
    # the code was redeemed server-side with our certificate (never a secret), openid only
    post = [c for c in ms.calls if c[0] == "POST"][0]
    token_url = f"https://login.microsoftonline.com/{T1}/oauth2/v2.0/token"
    assert post[1] == token_url
    assert "client_secret" not in post[2] and post[2]["scope"] == "openid"
    assert post[2]["client_id"] == "app-client-id"
    assert post[2]["client_assertion_type"] == \
        "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    assertion = pyjwt.decode(post[2]["client_assertion"], PUBLIC_KEY, algorithms=["RS256"],
                             audience=token_url, options={"verify_exp": False,
                                                          "verify_nbf": False})
    assert assertion["iss"] == assertion["sub"] == "app-client-id"
    assert pyjwt.get_unverified_header(post[2]["client_assertion"])["x5t"] == X5T
    assert post[2]["grant_type"] == "authorization_code" and post[2]["code"] == "the-code"


def test_forged_tenant_in_the_query_is_refused():
    """C1: an admin replays their own state with ANOTHER customer's tenant id. They can
    only sign in to their own organisation, so the verified tid disagrees: refused."""
    conn = MemConn()
    url = _consent_return(conn, tenant=T2)                # forged: victim's public tenant id
    with pytest.raises(consent.ConsentNotGiven) as e:
        _signin_return(conn, url, FakeMicrosoft(lambda n: _id_token(n, tid=T1)))
    assert e.value.code == "wrong_account"
    assert "tenant_id" not in json.loads(conn.value)


def test_query_tenant_is_never_written_even_when_sign_in_is_skipped():
    conn = MemConn()
    _consent_return(conn, tenant=T2)
    assert "tenant_id" not in json.loads(conn.value)
    assert json.loads(conn.value).get("status") != "connected"


@pytest.mark.parametrize("make,why", [
    (lambda n: _id_token(n, aud="someone-elses-app"), "audience"),
    (lambda n: _id_token(n, iss=f"https://login.microsoftonline.com/{T2}/v2.0"), "issuer"),
    (lambda n: _id_token(n, iss="https://evil.example/v2.0"), "issuer"),
    (lambda n: _id_token(n, exp=NOW - 3600), "expired"),
    (lambda n: _id_token(n, nbf=NOW + 3600), "not yet valid"),
    (lambda n: _id_token(n, key=_OTHER_RSA), "signature"),
    (lambda n: _id_token(n, kid="unknown"), "key"),
    (lambda n: _id_token("not-our-nonce"), "nonce"),
    (lambda n: _id_token(n, drop=("nonce",)), "nonce"),
    (lambda n: _id_token(n, drop=("nbf",)), "nbf"),
    (lambda n: pyjwt.encode({"aud": "app-client-id", "tid": T1, "nonce": n,
                             "iss": f"https://login.microsoftonline.com/{T1}/v2.0",
                             "exp": int(NOW + 99), "nbf": int(NOW)}, "x" * 32,
                            algorithm="HS256", headers={"kid": "k1"}), "alg"),
])
def test_bad_id_token_is_refused_and_nothing_stored(make, why):
    conn = MemConn()
    url = _consent_return(conn)
    with pytest.raises(consent.ConsentNotGiven) as e:
        _signin_return(conn, url, FakeMicrosoft(make))
    assert e.value.code == "signin_failed", why
    assert "tenant_id" not in json.loads(conn.value)


def test_failed_code_redemption_is_refused():
    conn = MemConn()
    url = _consent_return(conn)
    with pytest.raises(consent.ConsentNotGiven) as e:
        _signin_return(conn, url, FakeMicrosoft(token_status=400))
    assert e.value.code == "signin_failed"


def test_sign_in_cancelled_is_not_granted():
    conn = MemConn()
    url = _consent_return(conn)
    with pytest.raises(consent.ConsentNotGiven) as e:
        _signin_return(conn, url, FakeMicrosoft(), code=None, error="access_denied")
    assert e.value.code == "access_denied"


def test_sign_in_state_is_single_use():
    """Review Focus 2 for the second step: a refreshed sign-in return is refused as reused."""
    conn = MemConn()
    url = _consent_return(conn)
    _signin_return(conn, url, FakeMicrosoft())
    with pytest.raises(consent.StateRejected) as e:
        _signin_return(conn, url, FakeMicrosoft())
    assert e.value.code == "reused"


def test_consent_state_is_single_use():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    consent.handle_callback(conn, S, KEY, state=state, tenant=T1, admin_consent="True",
                            now=lambda: 120.0)
    with pytest.raises(consent.StateRejected) as e:   # Review Focus 2: replay
        consent.handle_callback(conn, S, KEY, state=state, tenant=T1, admin_consent="True",
                                now=lambda: 121.0)
    assert e.value.code == "reused"


def test_a_consent_state_cannot_be_used_as_a_sign_in_state():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    with pytest.raises(consent.StateRejected) as e:
        consent.complete_signin(conn, S, KEY, state=state, code="c", error=None,
                                transport=FakeMicrosoft(), app_credentials=CREDS,
                                now=lambda: 101.0)
    assert e.value.code == "malformed"


def test_declined_consent_stores_nothing():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    with pytest.raises(consent.ConsentNotGiven):
        consent.handle_callback(conn, S, KEY, state=state, tenant=None, admin_consent=None,
                                error="access_denied", now=lambda: 120.0)
    from services.mail_m365.config import read_integration
    assert "tenant_id" not in read_integration(conn)


@pytest.mark.parametrize("bad", ["t\nRemove-Item *", T1 + "\n"])
def test_bad_tenant_is_refused_and_not_stored(bad):
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    with pytest.raises(consent.ConsentNotGiven) as e:
        consent.handle_callback(conn, S, KEY, state=state, tenant=bad,
                                admin_consent="True", now=lambda: 120.0)
    assert e.value.code == "bad_tenant"
    assert "tenant_id" not in json.loads(conn.value)


def test_different_tenant_cannot_replace_a_connected_one():
    conn = MemConn()
    _connect(conn, T1)
    with pytest.raises(consent.ConsentNotGiven) as e:
        _connect(conn, T2)
    assert e.value.code == "tenant_mismatch"
    assert json.loads(conn.value)["tenant_id"] == T1


def test_same_tenant_may_reconsent():
    conn = MemConn()
    _connect(conn, T1)
    assert _connect(conn, T1)["tenant_id"] == T1


def test_complete_does_not_resurrect_a_nonce_spent_meanwhile():
    conn = MemConn()
    s1 = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    consent.handle_callback(conn, S, KEY, state=s1, tenant=T1, admin_consent="True",
                            now=lambda: 110.0)
    # s1 is spent and one sign-in nonce issued; the other consent nonce is untouched
    assert len(json.loads(conn.value)["pending_nonces"]) == 2
    with pytest.raises(consent.StateRejected):
        consent.handle_callback(conn, S, KEY, state=s1, tenant=T1, admin_consent="True",
                                now=lambda: 111.0)


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




# ---- only a Microsoft administrator may connect --------------------------------------------

@pytest.mark.parametrize("wids,why", [
    (None, "no directory roles at all"),
    ([USER_ADMIN], "an admin role that may not connect"),
    ([], "an empty role list"),
    (GLOBAL_ADMIN, "wids is a string, not a list"),
    ({"role": GLOBAL_ADMIN}, "wids is an object, not a list"),
])
def test_a_non_administrator_cannot_connect(wids, why):
    conn = MemConn()
    url = _consent_return(conn)
    with pytest.raises(consent.ConsentNotGiven) as e:
        _signin_return(conn, url, FakeMicrosoft(lambda n: _id_token(n, wids=wids)))
    assert e.value.code == "not_admin", why
    assert "administrator" in e.value.detail
    assert "tenant_id" not in json.loads(conn.value)
    assert json.loads(conn.value).get("status") != "connected"


@pytest.mark.parametrize("role", sorted(consent.ADMIN_ROLE_IDS))
def test_each_connecting_administrator_role_may_connect(role):
    conn = MemConn()
    url = _consent_return(conn)
    _signin_return(conn, url, FakeMicrosoft(lambda n: _id_token(n, wids=[USER_ADMIN, role])))
    assert json.loads(conn.value)["status"] == "connected"


def test_role_ids_compare_without_case():
    conn = MemConn()
    url = _consent_return(conn)
    _signin_return(conn, url, FakeMicrosoft(lambda n: _id_token(n, wids=[GLOBAL_ADMIN.upper()])))
    assert json.loads(conn.value)["tenant_id"] == T1


def test_admin_role_ids_are_exactly_the_four_that_may_consent():
    assert consent.ADMIN_ROLE_IDS == frozenset({
        "62e90394-69f5-4237-9190-012177145e10",   # Global Administrator
        "e8611ab8-c189-46e8-94e1-60213ab1f814",   # Privileged Role Administrator
        "158c047a-c907-4556-b7ef-446551a6b5f7",   # Cloud Application Administrator
        "9b895d92-2cd3-44c7-9d02-a6ac2d5ea5c3",   # Application Administrator
    })


def test_verify_id_token_returns_claims_for_a_global_administrator():
    claims = consent.verify_id_token(_id_token("n1"), client_id="app-client-id",
                                     expected_nonce="n1", transport=FakeMicrosoft(), now=NOW)
    assert claims["tid"] == T1 and claims["wids"] == [GLOBAL_ADMIN]


def test_verify_id_token_refuses_a_token_without_roles():
    with pytest.raises(consent.ConsentNotGiven) as e:
        consent.verify_id_token(_id_token("n1", wids=None), client_id="app-client-id",
                                expected_nonce="n1", transport=FakeMicrosoft(), now=NOW)
    assert e.value.code == "not_admin"


def test_a_forged_role_on_a_badly_signed_token_is_still_a_signature_failure():
    """The role check reads verified claims only: a token signed by someone else that
    claims Global Administrator is refused for its signature, not believed."""
    with pytest.raises(consent.ConsentNotGiven) as e:
        consent.verify_id_token(_id_token("n1", key=_OTHER_RSA), client_id="app-client-id",
                                expected_nonce="n1", transport=FakeMicrosoft(), now=NOW)
    assert e.value.code == "signin_failed"
