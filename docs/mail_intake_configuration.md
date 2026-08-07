# Email Intake: Configuration and Access

This document describes what access ProcureIQ needs in order to read documents
attached to email, and what it does with them. It is written for the person
reviewing the request, not for the engineer implementing it.

## What this feature does

Suppliers send quotes, invoices and order confirmations as email attachments.
This feature collects those attachments, records where each one came from, and
passes them to the part of ProcureIQ that reads documents. It does not send
email, and it does not act on anything a document says.

## What access is requested

Three connection methods are supported. **The relay method is preferred and
needs no access to your mail system at all.**

### 1. Relay address (recommended)

You are given an address such as `intake@yourcompany.procureiq.io`. Suppliers
send or copy their quotes to it, or you forward them.

- **Access to your mail system required: none.**
- Mail arrives directly at our system. No credential is issued, no mailbox is
  read, and nothing connects to your network.

This is the option most likely to clear a security review quickly, because
there is nothing to grant.

### 2. Microsoft 365 (Graph)

- **Permission requested:** `Mail.Read`, application permission.
- **Scope:** restricted to named mailboxes or folders using an *application
  access policy*. This is a Microsoft feature that limits the permission to the
  mailboxes you list. Without it, `Mail.Read` would cover every mailbox in the
  tenant, which is more than this feature needs and more than we ask for.
- **Not requested:** sending mail, calendars, contacts, files, directory data,
  or write access of any kind.
- **Verification:** after the permission is granted, the system deliberately
  attempts to read a mailbox it should *not* be able to reach. If that attempt
  succeeds, the connection is refused and reported as over-scoped rather than
  being used. The result of that check is recorded with a timestamp.
- **A connection cannot be switched on without that proof.** This is enforced
  by the database, not by process: a mailbox connection may only be marked
  active if it carries a recorded scope-verification timestamp. Nobody can
  enable one by skipping the check, including us.
- **Credentials are never held by this system.** What is stored is a reference
  to an entry in your organisation's secret store, and the database will only
  accept a value in that form — a pasted password or key is rejected outright
  rather than stored.

### 3. IMAP

- **Permission requested:** read access to one named folder.
- **Not requested:** sending, deleting, moving or modifying messages.
- Used only where API access cannot be granted.

## What is stored

For each email: the sender, the recipients, the subject, the date, the message
body, the raw original message, and the result of the sender-authentication
checks your mail system performed.

For each attachment: the filename, its actual file type, its size, a
cryptographic fingerprint (SHA-256), and the file itself.

Files are stored write-once. Reprocessing a document creates a new record; it
never alters the original.

## What is never stored or logged

Attachment contents and message bodies are never written to system logs. In
non-production environments, email addresses in logs are reduced to the domain
only. Mailbox credentials are never stored by this feature — it holds a
reference to your organisation's secret store, not the secret.

## What is refused automatically

The following are held for a person to review, and are never opened, run or
unpacked:

| Held for review | Why |
|---|---|
| Password-protected documents | Cannot be checked without the password |
| Documents containing macros | Macros are executable code |
| ZIP and other archives | Contents unknown until unpacked; a person decides |
| Programs and scripts | Regardless of what the file is named |
| Files above the size limit | Configurable; 25 MB by default |
| Files whose type does not match their name | A `.pdf` that is really something else |
| Documents containing hidden text | Text present in the file but invisible to a reader — a known method of attempting to influence automated document reading |

Held items are retained with the reason recorded. An authorised person can
release one, and that release is itself recorded.

## A note on hidden text

Scanned documents legitimately contain invisible text — that is how a scan
becomes searchable. This feature distinguishes that from concealment by
checking whether the invisible text sits over a scanned image. Ordinary scanned
invoices are unaffected.

## What this feature cannot do

Nothing in a document can change an approval, a supplier's standing, a risk
rating, or how anything is scored. Those decisions are made by rules applied to
checked data, never by text inside a file. The part of the system that reads
documents cannot reach a mailbox, and cannot change how a document arrived or
who sent it — sender, timestamps and authentication results are recorded before
it runs and it has no route to alter them.

## Where the separation is enforced, and where it is not

We would rather state this precisely than let a reviewer infer more than is
true.

**Enforced for email attachments.** The document reader is never given a
storage location it could vary. It receives the file itself, written to a
temporary location we control and deleted immediately afterwards, so there is
no address it could substitute and no way for it to reach material it was not
handed — including the quarantine area holding rejected files.

**Not enforced in general.** The document reader is also used by other parts of
the product that do pass it a storage location, and it can fetch from storage
when they do. So it is not, today, a component that is *incapable* of network
access — it is one that is never *given* the means to misuse it on this path.
An automated check records the one place this happens and fails the build if a
new one appears, so the gap cannot widen quietly.

Closing it completely means running the document reader as a separate process
with no network route and a read-only database role. That is a deployment
change rather than a code change, and it is not part of this work. If your
review requires the stronger property, that is the piece to ask for.
