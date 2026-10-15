"""Our app signs in to Microsoft with a certificate (a signed client assertion), never a
client secret."""
import json

import jwt
import pytest

from services.mail_m365 import config, graph_client
from services.mail_m365.graph_client import (ASSERTION_TYPE, M365Client, client_assertion,
                                             load_app_certificate)
from tests.services.mail_m365.certs import PEM, PUBLIC_KEY, X5T, fake_credentials

TENANT = "11111111-2222-3333-4444-555555555555"
ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/cert-AbCdEf"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"


def _decode(assertion, aud=TOKEN_URL):
    return jwt.decode(assertion, PUBLIC_KEY, algorithms=["RS256"], audience=aud,
                      options={"verify_exp": False, "verify_nbf": False})


def test_client_assertion_is_signed_by_our_key_for_that_tenant():
    a = client_assertion("app-id", TENANT, PEM, X5T, now=1_000.0)
    claims = _decode(a)
    assert claims["aud"] == TOKEN_URL
    assert claims["iss"] == claims["sub"] == "app-id"
    assert claims["nbf"] == 995 and claims["exp"] == 1_300
    assert claims["jti"]
    header = jwt.get_unverified_header(a)
    assert header["x5t"] == X5T and header["alg"] == "RS256"


def test_each_assertion_is_unique():
    a = _decode(client_assertion("app-id", TENANT, PEM, X5T, now=1_000.0))
    b = _decode(client_assertion("app-id", TENANT, PEM, X5T, now=1_000.0))
    assert a["jti"] != b["jti"]


class Recording:
    def __init__(self):
        self.posts = []

    def post_form(self, url, data):
        self.posts.append((url, dict(data)))
        return 200, {"access_token": "tok", "expires_in": 3600}


def test_token_posts_a_client_assertion_and_no_client_secret():
    t = Recording()
    c = M365Client(fake_credentials(TENANT, "app-id"), transport=t, clock=lambda: 1_000.0)
    assert c.token() == "tok"
    url, form = t.posts[0]
    assert url == TOKEN_URL
    assert "client_secret" not in form
    assert form["client_id"] == "app-id"
    assert form["client_assertion_type"] == ASSERTION_TYPE == \
        "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    assert form["grant_type"] == "client_credentials"
    assert form["scope"] == "https://graph.microsoft.com/.default"
    claims = _decode(form["client_assertion"])
    assert claims["iss"] == "app-id" and claims["exp"] == 1_300


def test_credentials_never_show_the_private_key():
    assert "PRIVATE KEY" not in repr(fake_credentials())
    assert "PRIVATE KEY" not in str(fake_credentials())


# ---- the certificate secret ---------------------------------------------------------------

class FakeSecrets:
    def __init__(self, payload):
        self.payload = payload
        self.reads = 0

    def get_secret_value(self, SecretId):
        self.reads += 1
        return {"SecretString": json.dumps(self.payload)}


GOOD = {"client_id": "app-id", "private_key_pem": PEM, "thumbprint_sha1_b64url": X5T}


@pytest.fixture(autouse=True)
def _fresh_cache():
    graph_client._reset_certificate_cache()
    yield
    graph_client._reset_certificate_cache()


def test_certificate_secret_becomes_credentials_for_the_tenant():
    c = load_app_certificate(ARN, tenant_id=TENANT, client=FakeSecrets(GOOD))
    assert (c.tenant_id, c.client_id, c.private_key_pem, c.x5t) == (TENANT, "app-id", PEM, X5T)


def test_certificate_secret_is_read_once_per_process():
    s = FakeSecrets(GOOD)
    load_app_certificate(ARN, tenant_id=TENANT, client=s)
    load_app_certificate(ARN, tenant_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", client=s)
    assert s.reads == 1


def test_certificate_cache_expires(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(graph_client.time, "monotonic", lambda: now[0])
    s = FakeSecrets(GOOD)
    load_app_certificate(ARN, tenant_id=TENANT, client=s)
    now[0] += graph_client.CERT_TTL_SECONDS + 1
    load_app_certificate(ARN, tenant_id=TENANT, client=s)
    assert s.reads == 2


@pytest.mark.parametrize("missing", ["client_id", "private_key_pem", "thumbprint_sha1_b64url"])
def test_incomplete_certificate_secret_is_refused(missing):
    payload = {k: v for k, v in GOOD.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        load_app_certificate(ARN, tenant_id=TENANT, client=FakeSecrets(payload))


def test_a_secret_holding_only_a_client_secret_is_refused():
    with pytest.raises(ValueError):
        load_app_certificate(ARN, tenant_id=TENANT,
                             client=FakeSecrets({"client_id": "c", "client_secret": "s"}))


def test_certificate_reference_must_be_a_secrets_manager_arn():
    with pytest.raises(ValueError, match="ARN"):
        load_app_certificate("-----BEGIN PRIVATE KEY-----", tenant_id=TENANT,
                             client=FakeSecrets(GOOD))


# ---- settings -------------------------------------------------------------------------------

_ENV = {"M365_CLIENT_ID": "cid", "PUBLIC_BASE_URL": "https://api.procwise.co.uk",
        "MAIL_STATE_SIGNING_KEY_ARN": ARN}


def test_settings_need_the_certificate_arn(monkeypatch):
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("M365_CLIENT_CERT_ARN", raising=False)
    monkeypatch.setenv("M365_CLIENT_SECRET_ARN", ARN)      # the old setting no longer counts
    assert config.load_settings() is None
    monkeypatch.setenv("M365_CLIENT_CERT_ARN", ARN)
    s = config.load_settings()
    assert s is not None and s.cert_arn == ARN
