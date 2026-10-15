import pytest

from services.mail_m365.setup_script import render

T = "11111111-2222-3333-4444-555555555555"
A = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SP = "99999999-8888-7777-6666-555555555555"


def test_script_names_tenant_app_group_and_both_roles():
    text = render(T, A, SP, group="Procwise Mailboxes")
    for needle in ("Connect-ExchangeOnline", "New-ServicePrincipal", A, SP,
                   "New-ManagementScope", "Procwise Mailboxes",
                   "Application Mail.Send", "Application Mail.ReadWrite",
                   "New-ManagementRoleAssignment", "Test-ServicePrincipalAuthorization"):
        assert needle in text, needle


def test_script_never_contains_a_secret():
    text = render(T, A, None)
    assert "secret" not in text.lower() or "no secret" in text.lower()


BAD = ["x; Remove-Mailbox -Identity a", "$(Remove-Item C:\\)", "a\u201cb", "a\nb", 'a"b',
       "a`b", "a|b", "a&b", T + "\n", ""]


@pytest.mark.parametrize("bad", BAD)
def test_script_rejects_anything_outside_the_allow_list(bad):
    for args in ((bad, A, SP), (T, bad, SP), (T, A, bad)):
        with pytest.raises(ValueError):
            render(*args)
    with pytest.raises(ValueError):
        render(T, A, SP, group=bad)
