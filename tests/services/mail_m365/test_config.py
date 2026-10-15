import pytest

from services.mail_m365 import config

ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/app-AbCdEf"


def test_settings_absent_means_not_configured(monkeypatch):
    for k in ("M365_CLIENT_ID", "M365_CLIENT_CERT_ARN", "PUBLIC_BASE_URL",
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


def test_app_base_url_is_read_and_used_for_the_settings_page(monkeypatch):
    for k, v in {"M365_CLIENT_ID": "cid", "M365_CLIENT_CERT_ARN": ARN,
                 "PUBLIC_BASE_URL": "https://api.procwise.co.uk",
                 "MAIL_STATE_SIGNING_KEY_ARN": ARN}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("APP_BASE_URL", "https://ui.procwise.co.uk/")
    assert config.settings_page_url(config.load_settings()) == \
        "https://ui.procwise.co.uk/?view=settings"
    monkeypatch.delenv("APP_BASE_URL")
    assert config.settings_page_url(config.load_settings()) == \
        "https://api.procwise.co.uk/?view=settings"
