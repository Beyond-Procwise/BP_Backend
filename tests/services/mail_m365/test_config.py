import json
import pytest

from services.mail_m365 import config
from services.style.graph_source import resolve_credentials

ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/app-AbCdEf"


class FakeSecrets:
    def __init__(self, payload):
        self.payload = payload

    def get_secret_value(self, SecretId):
        return {"SecretString": json.dumps(self.payload)}


def test_settings_absent_means_not_configured(monkeypatch):
    for k in ("M365_CLIENT_ID", "M365_CLIENT_SECRET_ARN", "PUBLIC_BASE_URL",
              "MAIL_STATE_SIGNING_KEY_ARN"):
        monkeypatch.delenv(k, raising=False)
    assert config.load_settings() is None


def test_redirect_uri_requires_https(monkeypatch):
    s = config.M365Settings("cid", ARN, "http://16.61.116.180:8000", ARN, 120)
    with pytest.raises(ValueError, match="https"):
        config.redirect_uri(s)


def test_redirect_uri_shape():
    s = config.M365Settings("cid", ARN, "https://app.procwise.co.uk/", ARN, 120)
    assert config.redirect_uri(s) == \
        "https://app.procwise.co.uk/admin/integrations/mail/m365/callback"


def test_tenant_comes_from_consent_not_the_secret():
    creds = resolve_credentials(ARN, client=FakeSecrets({"client_id": "c", "client_secret": "s"}),
                                tenant_id="tenant-123")
    assert (creds.tenant_id, creds.client_id, creds.client_secret) == ("tenant-123", "c", "s")


def test_old_secret_shape_still_works():
    creds = resolve_credentials(ARN, client=FakeSecrets(
        {"tenant_id": "t", "client_id": "c", "client_secret": "s"}))
    assert creds.tenant_id == "t"
