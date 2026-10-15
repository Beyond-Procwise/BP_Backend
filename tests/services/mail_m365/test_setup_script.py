import pytest

from services.mail_m365.setup_script import render

T = "11111111-2222-3333-4444-555555555555"
A = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SP = "99999999-8888-7777-6666-555555555555"


def test_script_names_tenant_app_group_and_both_roles():
    text = render(T, A, SP, group="Procwise Mailboxes", access_level="read_send_change")
    for needle in ("Connect-ExchangeOnline", "New-ServicePrincipal", A, SP,
                   "New-ManagementScope", "Procwise Mailboxes",
                   "Application Mail.Send", "Application Mail.ReadWrite",
                   "New-ManagementRoleAssignment", "Test-ServicePrincipalAuthorization"):
        assert needle in text, needle


def test_script_never_contains_a_secret():
    text = render(T, A, None, group="Procwise Mailboxes", access_level="read_send_change")
    assert "secret" not in text.lower() or "no secret" in text.lower()


BAD = ["x; Remove-Mailbox -Identity a", "$(Remove-Item C:\\)", "a\u201cb", "a\nb", 'a"b',
       "a`b", "a|b", "a&b", T + "\n", ""]


@pytest.mark.parametrize("bad", BAD)
def test_script_rejects_anything_outside_the_allow_list(bad):
    for args in ((bad, A, SP), (T, bad, SP), (T, A, bad)):
        with pytest.raises(ValueError):
            render(*args, group="Procwise Mailboxes", access_level="read_send_change")
    with pytest.raises(ValueError):
        render(T, A, SP, group=bad, access_level="read_send")


def test_script_has_no_angle_bracket_placeholders_and_uses_variables():
    for sp in (SP, None):
        text = render(T, A, sp, group="Procwise Mailboxes", access_level="read_send_change")
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("<" in ln or ">" in ln for ln in code)
        assert '$TestMailbox = "shared-mailbox@yourdomain"' in text
        assert "-ObjectId $ServicePrincipalId" in text
        assert "-Resource $TestMailbox" in text
    assert f'$ServicePrincipalId = "{SP}"' in render(T, A, SP, group="Procwise Mailboxes", access_level="read_send_change")
    assert '$ServicePrincipalId = "PASTE-OBJECT-ID-HERE"' in render(T, A, None, group="Procwise Mailboxes", access_level="read_send_change")


def test_read_send_grants_read_not_readwrite_and_uses_the_group():
    text = render(T, A, SP, group="Finance Shared", access_level="read_send")
    assert '-Role "Application Mail.Read"' in text
    assert "Mail.ReadWrite" not in text
    assert '-Role "Application Mail.Send"' in text
    assert "Finance Shared" in text


def test_read_send_change_grants_readwrite():
    text = render(T, A, SP, group="Procwise Mailboxes", access_level="read_send_change")
    assert '-Role "Application Mail.ReadWrite"' in text
    assert '-Role "Application Mail.Send"' in text


@pytest.mark.parametrize("bad", ["", "admin", "read_send; Remove-Mailbox", None])
def test_unknown_access_level_is_rejected(bad):
    with pytest.raises(ValueError):
        render(T, A, SP, group="Procwise Mailboxes", access_level=bad)
