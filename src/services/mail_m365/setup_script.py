"""The PowerShell an IT admin runs to restrict Procwise to their chosen mailboxes.

Built from Microsoft's "RBAC for Applications in Exchange Online" guidance: register our
app's service principal in Exchange, scope it to members of one mail-enabled security
group, and grant send + read only within that scope. Permissions granted in Entra would
ADD to this, which is why our app requests none there (spec §2.2). Verify the cmdlets
against a live tenant before release — see docs/m365-setup.md.
"""
from __future__ import annotations

import re
from typing import Optional

from services.mail_m365.consent import GUID

_GROUP = re.compile(r"[A-Za-z0-9 ._-]{1,64}")
# access_level (email policy) -> the Exchange role that lets Procwise read; send is always granted.
_READ_ROLE = {"read_send": "Application Mail.Read", "read_send_change": "Application Mail.ReadWrite"}


def render(tenant_id: str, app_client_id: str, service_principal_object_id: Optional[str],
           group: str = "Procwise Mailboxes", access_level: str = "read_send_change") -> str:
    # Allow-list, not block-list: these land in a PowerShell script an admin runs.
    if not (GUID.fullmatch(tenant_id) and GUID.fullmatch(app_client_id)
            and (service_principal_object_id is None
                 or GUID.fullmatch(service_principal_object_id))
            and _GROUP.fullmatch(group)):
        raise ValueError("setup script input is not in the allowed form")
    read_role = _READ_ROLE.get(access_level)
    if read_role is None:
        raise ValueError("setup script input is not in the allowed form")
    sp_id = service_principal_object_id or "PASTE-OBJECT-ID-HERE"
    return f"""# Restrict Procwise to your chosen shared mailboxes. No secret is needed to run this.
# Tenant: {tenant_id}
# 1. Create a mail-enabled security group called "{group}" and add each shared mailbox
#    Procwise should use. Adding or removing a mailbox later = changing this group.
# Fill these two in before running. ("<" is reserved in PowerShell, so they are plain strings.)
#   ServicePrincipalId: Microsoft Entra > Enterprise applications > Procwise Mail > Object ID
#   TestMailbox: one shared mailbox that is a member of the group, for the check in step 5
$ServicePrincipalId = "{sp_id}"
$TestMailbox = "shared-mailbox@yourdomain"

Connect-ExchangeOnline

# 2. Register Procwise's app in Exchange.
New-ServicePrincipal -AppId {app_client_id} -ObjectId $ServicePrincipalId -DisplayName "Procwise Mail"

# 3. A scope containing only members of the group.
$group = Get-DistributionGroup -Identity "{group}"
New-ManagementScope -Name "Procwise Mailboxes Scope" -RecipientRestrictionFilter "MemberOfGroup -eq '$($group.DistinguishedName)'"

# 4. Send and read (access level: {access_level}), only inside that scope.
New-ManagementRoleAssignment -App {app_client_id} -Role "Application Mail.Send" -CustomResourceScope "Procwise Mailboxes Scope"
New-ManagementRoleAssignment -App {app_client_id} -Role "{read_role}" -CustomResourceScope "Procwise Mailboxes Scope"

# 5. Check (should say InScope = True for a shared mailbox, False for anyone else).
Test-ServicePrincipalAuthorization -Identity {app_client_id} -Resource $TestMailbox
"""
