from datetime import datetime, timezone

import pytest

from services.mail_m365 import receiving
from services.mail_m365.graph_client import M365Client, Throttled
from services.style import health
from services.style.graph_source import MailboxUnreachable
from tests.services.mail_m365.certs import fake_credentials
from services.style.mailbox import MailboxBinding

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)
B = MailboxBinding(3, "org", "graph", "p@c.com", "send_receive", "arn:aws:secretsmanager:x",
                   None, NOW, "ev", None, "OK", True, True, NOW)


class Repo:
    def __init__(self):
        self.recorded, self.deactivated = [], []

    def record_health(self, i, state):
        self.recorded.append(state)
        return B

    def deactivate(self, i, *, reason):
        self.deactivated.append(reason)
        return B


class Client:
    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    def can_read(self, mailbox):
        if self.exc:
            raise self.exc
        return self.result


def _use(monkeypatch, client):
    monkeypatch.setattr(receiving, "_default_client_for",
                        lambda: None if client is None else (lambda: client))


def test_probe_revoked_deactivates_and_clears_default(monkeypatch):
    _use(monkeypatch, Client((False, "denied with HTTP 403")))
    repo = Repo()
    assert health.check_binding(B, repo=repo) == "REVOKED"
    assert repo.recorded == ["REVOKED"]
    assert repo.deactivated == ["access withdrawn in Microsoft 365"]


def test_probe_readable_is_ok(monkeypatch):
    _use(monkeypatch, Client((True, "ok")))
    repo = Repo()
    assert health.check_binding(B, repo=repo) == "OK"
    assert repo.deactivated == []


@pytest.mark.parametrize("exc", [MailboxUnreachable("down"), Throttled(30)])
def test_transient_error_is_degraded_never_revoked(monkeypatch, exc):
    _use(monkeypatch, Client(exc=exc))
    repo = Repo()
    assert health.check_binding(B, repo=repo) == "DEGRADED"
    assert repo.recorded == ["DEGRADED"]
    assert repo.deactivated == []


def test_m365_not_connected_leaves_state_unchanged(monkeypatch):
    _use(monkeypatch, None)
    repo = Repo()
    assert health.check_binding(B, repo=repo) == "OK"
    assert repo.recorded == [] and repo.deactivated == []


class StatusTransport:
    def __init__(self, status):
        self.status = status

    def post_form(self, url, data):
        return 200, {"access_token": "t", "expires_in": 3600}

    def get(self, url, headers):
        return self.status, {}


@pytest.mark.parametrize("status,state,deactivated", [
    (401, "DEGRADED", False),   # Microsoft rejected OUR token: not the customer's decision
    (403, "REVOKED", True),
    (404, "REVOKED", True),
    (429, "DEGRADED", False),
    (500, "DEGRADED", False),
    (200, "OK", False),
])
def test_http_status_maps_to_state_end_to_end(monkeypatch, status, state, deactivated):
    client = M365Client(fake_credentials(), transport=StatusTransport(status))
    _use(monkeypatch, client)
    repo = Repo()
    assert health.check_binding(B, repo=repo) == state
    assert repo.recorded == [state]
    assert bool(repo.deactivated) is deactivated


def test_revoked_non_send_receive_binding_does_not_call_deactivate():
    from dataclasses import replace
    repo = Repo()
    assert health.check_binding(replace(B, role="exemplar_source"), repo=repo,
                                probe=lambda b: "REVOKED") == "REVOKED"
    assert repo.deactivated == []
