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
