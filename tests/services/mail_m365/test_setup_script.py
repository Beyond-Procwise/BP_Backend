import pytest

# tests/services/mail_m365/test_setup_script.py
from services.mail_m365.setup_script import render


def test_script_names_tenant_app_group_and_both_roles():
    text = render("tenant-1", "app-client-id", "sp-object-id", group="Procwise Mailboxes")
    for needle in ("Connect-ExchangeOnline", "New-ServicePrincipal", "app-client-id",
                   "sp-object-id", "New-ManagementScope", "Procwise Mailboxes",
                   "Application Mail.Send", "Application Mail.ReadWrite",
                   "New-ManagementRoleAssignment", "Test-ServicePrincipalAuthorization"):
        assert needle in text, needle


def test_script_never_contains_a_secret():
    text = render("tenant-1", "app-client-id", None)
    assert "secret" not in text.lower() or "no secret" in text.lower()


@pytest.mark.parametrize("bad", ["a\nb", 'a"b', "a`b", "a\rb"])
def test_script_rejects_characters_that_break_out(bad):
    for args in ((bad, "app", "sp"), ("t", bad, "sp"), ("t", "app", bad)):
        with pytest.raises(ValueError):
            render(*args)
    with pytest.raises(ValueError):
        render("t", "app", "sp", group=bad)
