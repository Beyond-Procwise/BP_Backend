# Microsoft 365 Shared Mailbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A customer admin connects Microsoft 365 company-wide and the product sends and receives supplier email through admin-chosen shared mailboxes, with scope proven before use and no silent fallback to SES.

**Architecture:** A new package `src/services/mail_m365/` owns everything Microsoft-specific (consent, Graph client, setup script, sending, receiving, admin service). It reuses the existing `proc.bp_mailbox_binding` table and `MailboxBindingRepository` (extended, not duplicated), the existing `proc.bp_admin_config` store (key `mail_integration`), the existing `mailbox.bind` (`configure`) action, mail-intake's `MailSource` / `poll_binding` for attachments, and the email watcher's `parse_email_bytes` for reply matching. `EmailService.send_email` gains one transport decision; its callers do not change.

**Tech Stack:** Python 3 / FastAPI / psycopg2 / pytest (BP_Backend); React + vitest (beyond_procwise_ui); Microsoft Graph v1.0; AWS Secrets Manager.

**Spec:** `specs/2026-10-15-m365-shared-mailbox-design.md` — read it before starting.

## Global Constraints

- Branch: `m365-mailbox`, cut from **`mail-intake-v2`** (it carries Development + the attachment reader that receiving needs). Merge order later: `mail-intake-v2` → Development, then this branch. Never push to `main`.
- Work in a worktree (`superpowers:using-git-worktrees`); the main checkout is shared with other sessions — never bare `git stash`, commit only your own paths (`git commit -o <paths>` after checking `git status`).
- Tests: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest …` from the repo root with `.env` present. DB tests need `PROCWISE_TEST_LIVE_DB=1` or they skip. `.env` points at **bp_testdb**.
- Migrations: bp_testdb only. bp_sqldb only with Nick's explicit OK.
- New tables/indexes use `bp_` / `ix_bp_<table>_<cols>`.
- No customer password or refresh token is ever stored. The only credential is our app secret, referenced by Secrets Manager ARN.
- The Entra app requests **no** mail permissions; mail access is granted in Exchange (RBAC for Applications) scoped to a mail-enabled security group.
- **No silent SES fallback** once M365 is connected.
- We never move, delete, flag or mark-read anything in the customer's mailbox.
- Graph paths reachable from `mail_m365`: `/users/{m}/mailFolders…`, `/users/{m}/messages…` (list, get, `$value`), `/users/{m}/sendMail`. Nothing else.
- Commit message trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- UI repo: `/home/muthu/PycharmProjects/beyond_procwise_ui`, branch `spendiq-ui`; reuse `modalStyles.js` look; new strings go through `tOr`.

## Review Focus

1. **Admin adds the same mailbox twice** → second add is refused with "already connected", not a second binding (pinned in Task 6).
2. **Callback hit twice with the same `state`** (browser back/refresh) → second is refused as reused, first result kept (pinned in Task 5).
3. **Default mailbox removed while drafts are queued** → those drafts are held with a reason, never sent via SES (pinned in Task 8).
4. **Mailbox address typed with capitals/spaces** (` Procurement@Customer.com `) → stored and compared lower-cased and trimmed; control-equals-bound check still fires (pinned in Task 6).
5. **Graph returns 429 on poll** → poll for that binding stops this pass without moving the cursor; no message lost (pinned in Task 9).

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/migrations/2026-10-15-m365-mailbox.sql` (+ `_rollback.sql`) | role `send_receive`, `is_default`, `last_test_send_at`, constraints, default index |
| `tests/sql/test_m365_mailbox_sql.py` | migration test against bp_testdb |
| `src/services/style/mailbox.py` (modify) | binding fields + org mailbox reads/writes |
| `src/services/mail_m365/__init__.py` | package marker |
| `src/services/mail_m365/config.py` | env settings + `mail_integration` row in `bp_admin_config` |
| `src/services/mail_m365/graph_client.py` | the only Graph caller for this feature |
| `src/services/mail_m365/consent.py` | consent URL, signed single-use `state`, callback handling |
| `src/services/mail_m365/setup_script.py` | PowerShell text (pure) |
| `src/services/mail_m365/admin.py` | add / verify / test-send / default / remove / disconnect |
| `src/services/mail_m365/sending.py` | transport decision + Graph send |
| `src/services/mail_m365/receiving.py` | `GraphMailSource` + watcher reply fetch |
| `src/api/routers/mail_integration.py` | HTTP endpoints |
| `src/services/email_service.py`, `email_dispatch_service.py`, `email_watcher.py`, `backend_scheduler.py`, `style/health.py`, `mail_intake/poll.py`, `src/api/main.py` (modify) | wiring |
| `tests/services/mail_m365/test_*.py` | unit tests (stub transport) |
| UI `src/modules/MailConnection/mailboxService.js` (rewrite), `IntegrationsEmailPanel.jsx` (new), `mailbox.contract.test.js` (rewrite), `IntegrationsEmailPanel.render.test.jsx` (new), `ProcurementHome/HomeNavBar.jsx` + `index.jsx` (entry) | admin UI |
| `docs/m365-setup.md` (`git add -f`, docs/ is gitignored) | Entra registration steps + live test runbook |

---

### Task 1: Migration — shared mailboxes on the existing binding table

**Files:**
- Create: `scripts/migrations/2026-10-15-m365-mailbox.sql`, `scripts/migrations/2026-10-15-m365-mailbox_rollback.sql`
- Test: `tests/sql/test_m365_mailbox_sql.py`

**Interfaces:**
- Consumes: `proc.bp_mailbox_binding` as left by `2026-08-07-mail-intake.sql` (constraints `bp_mailbox_binding_provider_check`, `bp_mailbox_binding_role_check`, columns `folder_scope`, `last_sync_at`) and `2026-10-14-mail-intake-relay-binding.sql`.
- Produces: columns `is_default BOOLEAN NOT NULL DEFAULT FALSE`, `last_test_send_at TIMESTAMPTZ`; role value `send_receive`; index `ix_bp_mailbox_binding_default`; constraint `ck_bp_mailbox_binding_default_needs_test_send`; unique index `ix_bp_mailbox_binding_org_address`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sql/test_m365_mailbox_sql.py
"""The shared-mailbox migration, exercised by inserting, not by reading DDL."""
import os
import pytest

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="live DB test; set PROCWISE_TEST_LIVE_DB=1")

ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/app-AbCdEf"


@pytest.fixture
def cur():
    from dotenv import load_dotenv
    load_dotenv(".env")
    conn = psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
                            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"))
    conn.autocommit = False
    c = conn.cursor()
    yield c
    conn.rollback()
    conn.close()


def _insert(cur, address, **extra):
    cols = {"user_ref": "org", "provider": "graph", "mailbox_address": address,
            "role": "send_receive", "credential_ref": ARN, **extra}
    names = ", ".join(cols)
    marks = ", ".join(["%s"] * len(cols))
    cur.execute(f"INSERT INTO proc.bp_mailbox_binding ({names}) VALUES ({marks}) "
                "RETURNING binding_id", list(cols.values()))
    return cur.fetchone()[0]


def test_send_receive_role_is_admitted(cur):
    assert _insert(cur, "m365-test-a@example.test") > 0


def test_default_requires_a_test_send(cur):
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(cur, "m365-test-b@example.test", is_default=True,
                is_active=True, scope_verified_at="2026-10-15T00:00:00Z")


def test_only_one_active_default(cur):
    common = dict(is_default=True, is_active=True,
                  scope_verified_at="2026-10-15T00:00:00Z",
                  last_test_send_at="2026-10-15T00:00:00Z")
    _insert(cur, "m365-test-c@example.test", **common)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(cur, "m365-test-d@example.test", **common)


def test_same_org_address_cannot_be_bound_twice_while_active(cur):
    common = dict(is_active=True, scope_verified_at="2026-10-15T00:00:00Z")
    _insert(cur, "m365-test-e@example.test", **common)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(cur, "m365-test-e@example.test", **common)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/sql/test_m365_mailbox_sql.py -v`
Expected: FAIL — `CheckViolation` on `bp_mailbox_binding_role_check` for `send_receive` (and `UndefinedColumn` for `is_default`). If instead it fails on `provider`/`intake`, the mail-intake migrations are not on bp_testdb — stop and apply those first (they are on the branch you cut from).

- [ ] **Step 3: Write the migration**

```sql
-- scripts/migrations/2026-10-15-m365-mailbox.sql
-- Shared company mailboxes on Microsoft 365 (spec 2026-10-15-m365-shared-mailbox-design §4.2).
-- Runs AFTER 2026-08-07-mail-intake.sql and 2026-10-14-mail-intake-relay-binding.sql.
-- Idempotent. Every row valid before is valid after (no send_receive row exists yet).

ALTER TABLE proc.bp_mailbox_binding
    DROP CONSTRAINT IF EXISTS bp_mailbox_binding_role_check;
ALTER TABLE proc.bp_mailbox_binding
    ADD CONSTRAINT bp_mailbox_binding_role_check
    CHECK (role = ANY (ARRAY['exemplar_source', 'draft_target', 'both', 'intake', 'send_receive']));

ALTER TABLE proc.bp_mailbox_binding
    ADD COLUMN IF NOT EXISTS is_default        BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS last_test_send_at TIMESTAMPTZ;

-- A mailbox may be the default only once a person has watched a test email leave it:
-- sending cannot be probed without sending (spec §2.3 step 3).
ALTER TABLE proc.bp_mailbox_binding
    DROP CONSTRAINT IF EXISTS ck_bp_mailbox_binding_default_needs_test_send;
ALTER TABLE proc.bp_mailbox_binding
    ADD CONSTRAINT ck_bp_mailbox_binding_default_needs_test_send
    CHECK (NOT is_default OR (last_test_send_at IS NOT NULL AND role = 'send_receive'));

-- One default among active bindings.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_mailbox_binding_default
    ON proc.bp_mailbox_binding ((1)) WHERE is_default AND is_active;

-- One active org binding per address (Review Focus 1).
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_mailbox_binding_org_address
    ON proc.bp_mailbox_binding (lower(mailbox_address))
    WHERE user_ref = 'org' AND is_active;
```

```sql
-- scripts/migrations/2026-10-15-m365-mailbox_rollback.sql
DROP INDEX IF EXISTS proc.ix_bp_mailbox_binding_org_address;
DROP INDEX IF EXISTS proc.ix_bp_mailbox_binding_default;
ALTER TABLE proc.bp_mailbox_binding DROP CONSTRAINT IF EXISTS ck_bp_mailbox_binding_default_needs_test_send;
DELETE FROM proc.bp_mailbox_binding WHERE role = 'send_receive';
ALTER TABLE proc.bp_mailbox_binding DROP COLUMN IF EXISTS last_test_send_at;
ALTER TABLE proc.bp_mailbox_binding DROP COLUMN IF EXISTS is_default;
ALTER TABLE proc.bp_mailbox_binding DROP CONSTRAINT IF EXISTS bp_mailbox_binding_role_check;
ALTER TABLE proc.bp_mailbox_binding ADD CONSTRAINT bp_mailbox_binding_role_check
    CHECK (role = ANY (ARRAY['exemplar_source', 'draft_target', 'both', 'intake']));
```

Note: the `DELETE` in rollback fails if `draft_rfq_emails.style_mailbox_binding_id` references a send_receive row; that is intended — investigate rather than force.

- [ ] **Step 4: Apply to bp_testdb and run the test**

Run: `set -a; . ./.env; set +a; PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -f scripts/migrations/2026-10-15-m365-mailbox.sql` (confirm `DB_NAME` is `bp_testdb` first), then the pytest command from Step 2.
Expected: 4 passed. Run the migration a second time: no error (idempotent).

- [ ] **Step 5: Commit**

```bash
git add scripts/migrations/2026-10-15-m365-mailbox*.sql tests/sql/test_m365_mailbox_sql.py
git commit -m "feat(m365): shared mailboxes are send_receive bindings with one tested default"
```

---

### Task 2: Binding repository knows org mailboxes and the default

**Files:**
- Modify: `src/services/style/mailbox.py` (constants near line 30, `_COLUMNS` line ~42, `MailboxBinding` line 63, `_row` line 95, new methods after `list_active`)
- Test: `tests/services/mail_m365/test_binding_repo.py`, `tests/services/mail_m365/__init__.py` (empty)

**Interfaces:**
- Consumes: Task 1 columns.
- Produces:
  - constants `ROLE_SEND_RECEIVE = "send_receive"`, `ORG_USER_REF = "org"`
  - `MailboxBinding` gains `is_default: bool = False`, `last_test_send_at: Optional[datetime] = None`, property `can_send -> bool` (`role == ROLE_SEND_RECEIVE`); `can_read` true for `send_receive` too.
  - `normalise_address(addr: str) -> str` (strip + lower)
  - `MailboxBindingRepository.list_org(include_inactive: bool = False) -> List[MailboxBinding]`
  - `.get_default_sender() -> Optional[MailboxBinding]` (active default)
  - `.get_org_default_any() -> Optional[MailboxBinding]` (default flag regardless of active — used to detect "connected but unusable")
  - `.record_test_send(binding_id: int) -> MailboxBinding`
  - `.set_default(binding_id: int) -> MailboxBinding` (clears others in one statement pair; raises `ValueError` if not active/tested)
  - `.deactivate(binding_id: int, *, reason: str) -> MailboxBinding` (also clears `is_default`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/mail_m365/test_binding_repo.py
import os
import pytest

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="live DB test; set PROCWISE_TEST_LIVE_DB=1")

from services.style.mailbox import (ORG_USER_REF, PROVIDER_GRAPH, ROLE_SEND_RECEIVE,
                                    MailboxBindingRepository, normalise_address)

ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/app-AbCdEf"


@pytest.fixture
def repo():
    from dotenv import load_dotenv
    load_dotenv(".env")
    conn = psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
                            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"))
    conn.autocommit = False
    yield MailboxBindingRepository(conn)
    conn.rollback()
    conn.close()


def _active(repo, address):
    b = repo.create(user_ref=ORG_USER_REF, provider=PROVIDER_GRAPH,
                    mailbox_address=address, role=ROLE_SEND_RECEIVE, credential_ref=ARN)
    return repo.verify_scope(b.binding_id, control_mailbox="control@example.test",
                             probe=lambda _a: (False, "denied with HTTP 403"))


def test_normalise_address():
    assert normalise_address("  Procurement@Customer.COM ") == "procurement@customer.com"


def test_send_receive_binding_can_send_and_read(repo):
    b = _active(repo, "m365-r1@example.test")
    assert b.can_send and b.can_read and not b.is_default


def test_default_needs_test_send_first(repo):
    b = _active(repo, "m365-r2@example.test")
    with pytest.raises(ValueError, match="test email"):
        repo.set_default(b.binding_id)


def test_set_default_moves_the_flag(repo):
    a = repo.record_test_send(_active(repo, "m365-r3@example.test").binding_id)
    b = repo.record_test_send(_active(repo, "m365-r4@example.test").binding_id)
    repo.set_default(a.binding_id)
    repo.set_default(b.binding_id)
    assert repo.get_default_sender().binding_id == b.binding_id
    assert repo.get(a.binding_id).is_default is False


def test_deactivate_clears_default(repo):
    a = repo.record_test_send(_active(repo, "m365-r5@example.test").binding_id)
    repo.set_default(a.binding_id)
    repo.deactivate(a.binding_id, reason="removed by admin")
    assert repo.get_default_sender() is None
    assert repo.get(a.binding_id).is_active is False


def test_list_org_excludes_user_bindings(repo):
    _active(repo, "m365-r6@example.test")
    assert all(b.user_ref == ORG_USER_REF for b in repo.list_org())
```

- [ ] **Step 2: Run to verify failure**

Run: `PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_binding_repo.py -v`
Expected: FAIL — `ImportError: cannot import name 'ORG_USER_REF'`.

- [ ] **Step 3: Implement**

In `src/services/style/mailbox.py`:

```python
ROLE_SEND_RECEIVE = "send_receive"
ORG_USER_REF = "org"
```

Append `, is_default, last_test_send_at` to `_COLUMNS` (it must stay the column order `_row` reads). Add the two fields **with defaults, last** in `MailboxBinding` so existing positional constructions in tests keep working:

```python
    is_default: bool = False
    last_test_send_at: Optional[datetime] = None

    @property
    def can_send(self) -> bool:
        return self.role == ROLE_SEND_RECEIVE
```

Change `can_read` to `return self.role in (ROLE_EXEMPLAR_SOURCE, ROLE_BOTH, ROLE_SEND_RECEIVE)`.

`_row` must accept both tuple and dict cursors (the mail-intake lesson: `RealDictCursor` raised `KeyError: 0` silently):

```python
_FIELDS = ("binding_id", "user_ref", "provider", "mailbox_address", "role", "credential_ref",
           "scope_policy_ref", "scope_verified_at", "scope_evidence_ref", "last_health_check",
           "health_state", "is_active", "is_default", "last_test_send_at")


def _row(r) -> MailboxBinding:
    values = [r[k] for k in _FIELDS] if isinstance(r, dict) else list(r)
    return MailboxBinding(**dict(zip(_FIELDS, values)))


def normalise_address(addr: str) -> str:
    return str(addr or "").strip().lower()
```

New methods on `MailboxBindingRepository` (same `_with_conn` pattern as the existing ones):

```python
    def list_org(self, *, include_inactive: bool = False) -> List[MailboxBinding]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE user_ref = %s "
                + ("" if include_inactive else "AND is_active ")
                + "ORDER BY binding_id", (ORG_USER_REF,))
            rows = cur.fetchall()
            cur.close()
            return [_row(r) for r in rows]
        return self._with_conn(_run)

    def _one(self, where: str, params) -> Optional[MailboxBinding]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(f"SELECT {_COLUMNS} FROM proc.bp_mailbox_binding WHERE {where} "
                        "ORDER BY binding_id DESC LIMIT 1", params)
            row = cur.fetchone()
            cur.close()
            return _row(row) if row else None
        return self._with_conn(_run)

    def get_default_sender(self) -> Optional[MailboxBinding]:
        return self._one("user_ref = %s AND is_default AND is_active", (ORG_USER_REF,))

    def get_org_default_any(self) -> Optional[MailboxBinding]:
        """The mailbox last marked default, even if it has since been deactivated.

        Its existence is what turns "no mailbox" into "the company mailbox is unavailable":
        the first sends through SES, the second must be held (spec §5.1).
        """
        return self._one("user_ref = %s AND role = %s AND last_test_send_at IS NOT NULL",
                         (ORG_USER_REF, ROLE_SEND_RECEIVE))

    def _update(self, sql: str, params) -> MailboxBinding:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(sql + f" RETURNING {_COLUMNS}", params)
            row = cur.fetchone()
            cur.close()
            if row is None:
                raise BindingNotFound("no such mailbox binding")
            return _row(row)
        return self._with_conn(_run)

    def record_test_send(self, binding_id: int) -> MailboxBinding:
        b = self._update("UPDATE proc.bp_mailbox_binding SET last_test_send_at = NOW() "
                         "WHERE binding_id = %s", (binding_id,))
        _audit("test_send", status="ok", summary=f"test email sent from {b.mailbox_address}",
               details={"binding_id": binding_id})
        return b

    def set_default(self, binding_id: int) -> MailboxBinding:
        b = self.get(binding_id)
        if b is None:
            raise BindingNotFound(f"no mailbox binding with binding_id={binding_id}")
        if not (b.is_active and b.can_send and b.last_test_send_at):
            raise ValueError("a mailbox can be the default only once it is verified and a "
                             "test email has been sent from it")
        def _run(conn):
            cur = conn.cursor()
            # Clear first: the partial unique index admits one active default at a time.
            cur.execute("UPDATE proc.bp_mailbox_binding SET is_default = FALSE "
                        "WHERE user_ref = %s AND is_default AND binding_id <> %s",
                        (ORG_USER_REF, binding_id))
            cur.execute("UPDATE proc.bp_mailbox_binding SET is_default = TRUE "
                        f"WHERE binding_id = %s RETURNING {_COLUMNS}", (binding_id,))
            row = cur.fetchone()
            cur.close()
            return _row(row)

        out = self._with_conn(_run)
        _audit("default_set", status="ok", summary=f"{b.mailbox_address} is now the default",
               details={"binding_id": binding_id})
        return out

    def deactivate(self, binding_id: int, *, reason: str) -> MailboxBinding:
        b = self._update("UPDATE proc.bp_mailbox_binding SET is_active = FALSE, "
                         "is_default = FALSE WHERE binding_id = %s", (binding_id,))
        _audit("binding_deactivate", status="ok",
               summary=f"{b.mailbox_address} deactivated: {reason}",
               details={"binding_id": binding_id, "reason": reason})
        return b
```

Note: `get_org_default_any` means "an org mailbox has ever sent a test email", i.e. this company has started sending from its own mailbox ⇒ from then on a broken mailbox holds rather than falling back to SES — including after Disconnect (deliberate: spec §6 "Disconnect while drafts queued → held"). Also: `create()` must store `normalise_address(mailbox_address)`; change its INSERT parameter accordingly, and in `verify_scope` compare with `normalise_address` on both sides.

- [ ] **Step 4: Run the new tests and the existing mailbox suite**

Run: `PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_binding_repo.py tests/services/test_style_mailbox.py tests/services/test_style_draft_writer.py -v`
Expected: all pass (existing suite proves the `_row`/`_COLUMNS` change broke nothing).

- [ ] **Step 5: Commit**

```bash
git add src/services/style/mailbox.py tests/services/mail_m365/
git commit -m "feat(m365): the binding repository knows company mailboxes and which one sends"
```

---

### Task 3: Configuration — env settings and the `mail_integration` row

**Files:**
- Create: `src/services/mail_m365/__init__.py` (empty), `src/services/mail_m365/config.py`
- Modify: `src/services/style/graph_source.py:82-111` (`resolve_credentials` accepts a tenant override)
- Test: `tests/services/mail_m365/test_config.py`

**Interfaces:**
- Produces:
  - `M365Settings(client_id: str, secret_arn: str, public_base_url: str, state_key_arn: str, poll_seconds: int)` dataclass; `load_settings() -> Optional[M365Settings]` (None when any of `M365_CLIENT_ID`, `M365_CLIENT_SECRET_ARN`, `PUBLIC_BASE_URL`, `MAIL_STATE_SIGNING_KEY_ARN` is unset); `redirect_uri(s) -> str` = `f"{s.public_base_url.rstrip('/')}/admin/integrations/mail/m365/callback"`; raises `ValueError` if `public_base_url` is not `https://`.
  - `CONFIG_KEY = "mail_integration"`; `read_integration(conn) -> dict` (empty dict if no row); `write_integration(conn, value: dict, *, actor: str) -> dict`.
  - `resolve_credentials(credential_ref, *, client=None, tenant_id: Optional[str] = None) -> GraphCredentials` — when `tenant_id` is given, the secret need only hold `client_id`/`client_secret`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/mail_m365/test_config.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: services.mail_m365`.

- [ ] **Step 3: Implement**

```python
# src/services/mail_m365/config.py
"""Where Microsoft 365 mail is configured.

Two places, deliberately apart. The server's own settings (our app id, where its secret
lives, the public address Microsoft returns to) come from the environment: they are the
same for every customer. What the customer's admin did — which tenant approved us, when,
who — is company configuration and lives in proc.bp_admin_config['mail_integration'],
next to the other admin settings.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

CONFIG_KEY = "mail_integration"


@dataclass(frozen=True)
class M365Settings:
    client_id: str
    secret_arn: str
    public_base_url: str
    state_key_arn: str
    poll_seconds: int


def load_settings() -> Optional[M365Settings]:
    values = {k: os.getenv(k, "").strip() for k in (
        "M365_CLIENT_ID", "M365_CLIENT_SECRET_ARN", "PUBLIC_BASE_URL",
        "MAIL_STATE_SIGNING_KEY_ARN")}
    if not all(values.values()):
        return None
    try:
        poll = max(30, int(os.getenv("M365_POLL_SECONDS", "120")))
    except ValueError:
        poll = 120
    return M365Settings(values["M365_CLIENT_ID"], values["M365_CLIENT_SECRET_ARN"],
                        values["PUBLIC_BASE_URL"], values["MAIL_STATE_SIGNING_KEY_ARN"], poll)


def redirect_uri(settings: M365Settings) -> str:
    base = settings.public_base_url.rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("PUBLIC_BASE_URL must be an https address — Microsoft will not "
                         "return an admin to anything else")
    return f"{base}/admin/integrations/mail/m365/callback"


def read_integration(conn: Any) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = %s",
                (CONFIG_KEY,))
    row = cur.fetchone()
    cur.close()
    if not row:
        return {}
    value = row["config_value"] if isinstance(row, dict) else row[0]
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def write_integration(conn: Any, value: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) "
        "VALUES (%s, %s::jsonb, %s) ON CONFLICT (config_key) DO UPDATE "
        "SET config_value = EXCLUDED.config_value, "
        "last_modified_by = EXCLUDED.last_modified_by, last_modified_date = now()",
        (CONFIG_KEY, json.dumps(value, default=str), actor))
    cur.close()
    if not getattr(conn, "autocommit", False):
        conn.commit()
    return value
```

In `graph_source.resolve_credentials`, add `tenant_id: Optional[str] = None` and build with `tenant_id=tenant_id or data["tenant_id"]` (keep the `KeyError → ValueError` message).

- [ ] **Step 4: Run tests** — same command as Step 2, plus `tests/services/test_style_mailbox.py`. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/mail_m365/__init__.py src/services/mail_m365/config.py src/services/style/graph_source.py tests/services/mail_m365/test_config.py
git commit -m "feat(m365): server settings from env, the customer's approval in admin config"
```

---

### Task 4: Graph client — the only code that talks to Microsoft for this feature

**Files:**
- Create: `src/services/mail_m365/graph_client.py`
- Modify: `src/services/style/graph_source.py` `RequestsTransport` — add `get_bytes(url, headers) -> Tuple[int, bytes]` and `post_body(url, headers, body: bytes) -> Tuple[int, Any]` (both through `egress` with `Purpose.MAILBOX`, wrapping exceptions in `MailboxUnreachable`). Do **not** add the string `sendMail` to graph_source.py.
- Test: `tests/services/mail_m365/test_graph_client.py`

**Interfaces:**
- Consumes: `GraphCredentials`, `RequestsTransport`, `MailboxUnreachable`, `MailboxAccessDenied`, `GRAPH_BASE`, `TOKEN_URL`, `GRAPH_SCOPE` from `graph_source`.
- Produces: `class M365Client(credentials: GraphCredentials, *, transport=None, clock=time.time)` with
  - `token() -> str` (cached until 60 s before `expires_in`)
  - `can_read(mailbox: str) -> Tuple[bool, str]` — lists Inbox; 2xx → `(True, detail)`, 401/403/404 → `(False, detail)`, else raises `MailboxUnreachable`
  - `probe_control(mailbox: str) -> Tuple[bool, str]` — same semantics as `GraphExemplarSource.probe_control_mailbox`
  - `list_inbox(mailbox: str, since: datetime, *, page_size: int = 50) -> List[Dict]` (`id`, `receivedDateTime`), oldest first, follows `@odata.nextLink`; 429 → raises `Throttled(retry_after: int)`
  - `get_mime(mailbox: str, message_id: str) -> bytes`
  - `send_mime(mailbox: str, mime: bytes) -> None` — POST `/users/{m}/sendMail`, `Content-Type: text/plain`, body = base64 of the MIME; 202 → ok; 401/403 → `MailboxAccessDenied`; 429 → `Throttled`; other → `MailboxUnreachable`
  - exception `Throttled(MailboxUnreachable)` with `.retry_after`
  - module constant `ALLOWED_PATH_PATTERNS` and `_url(path)` that raises `ValueError` for any path not matching it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/mail_m365/test_graph_client.py
import base64
from datetime import datetime, timezone

import pytest

from services.mail_m365.graph_client import M365Client, Throttled
from services.style.graph_source import GraphCredentials, MailboxAccessDenied


class Stub:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.token_posts = 0

    def post_form(self, url, data):
        self.token_posts += 1
        return 200, {"access_token": "tok", "expires_in": 3600}

    def _match(self, method, url):
        self.calls.append((method, url))
        for (m, frag), resp in self.routes.items():
            if m == method and frag in url:
                return resp
        return 404, {}

    def get(self, url, headers):
        return self._match("GET", url)

    def get_bytes(self, url, headers):
        return self._match("GETB", url)

    def post_body(self, url, headers, body):
        self.last_body = body
        self.last_headers = headers
        return self._match("POST", url)


def client(routes, clock=lambda: 1000.0):
    return M365Client(GraphCredentials("t", "c", "s"), transport=Stub(routes), clock=clock)


def test_token_is_cached_until_near_expiry():
    now = [1000.0]
    c = client({}, clock=lambda: now[0])
    c.token(); c.token()
    assert c.transport.token_posts == 1
    now[0] += 3600 - 30
    c.token()
    assert c.transport.token_posts == 2


def test_can_read_allowed_and_denied():
    c = client({("GET", "/users/ok@x/mailFolders/inbox"): (200, {"id": "inbox"}),
                ("GET", "/users/no@x/mailFolders/inbox"): (403, {})})
    assert c.can_read("ok@x")[0] is True
    assert c.can_read("no@x")[0] is False


def test_list_inbox_follows_paging_and_sorts_oldest_first():
    page2 = "https://graph.microsoft.com/v1.0/users/m@x/mailFolders/inbox/messages?page=2"
    c = client({
        ("GET", "page=2"): (200, {"value": [{"id": "a", "receivedDateTime": "2026-10-15T09:00:00Z"}]}),
        ("GET", "/users/m@x/mailFolders/inbox/messages"): (200, {
            "value": [{"id": "b", "receivedDateTime": "2026-10-15T10:00:00Z"}],
            "@odata.nextLink": page2}),
    })
    got = c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert [m["id"] for m in got] == ["a", "b"]


def test_list_inbox_429_raises_throttled():
    c = client({("GET", "/messages"): (429, {"retry_after": 7})})
    with pytest.raises(Throttled):
        c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))


def test_send_mime_posts_base64_text_plain():
    c = client({("POST", "/users/m@x/sendMail"): (202, {})})
    c.send_mime("m@x", b"Subject: hi\r\n\r\nbody")
    assert c.transport.last_headers["Content-Type"] == "text/plain"
    assert base64.b64decode(c.transport.last_body) == b"Subject: hi\r\n\r\nbody"


def test_send_denied_raises_access_denied():
    c = client({("POST", "/sendMail"): (403, {})})
    with pytest.raises(MailboxAccessDenied):
        c.send_mime("m@x", b"x")


@pytest.mark.parametrize("path", ["/users/m@x/messages/1/move", "/users/m@x/messages/1",
                                  "/me/messages", "/users/m@x/mailFolders/inbox/messages/1/forward"])
def test_paths_outside_the_allowlist_are_refused(path):
    c = client({})
    with pytest.raises(ValueError):
        c._url(path, method="POST")


def test_no_delete_or_patch_method_exists():
    assert not any(hasattr(M365Client, n) for n in ("delete", "patch", "move", "mark_read"))
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_graph_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/services/mail_m365/graph_client.py
"""Microsoft Graph for company shared mailboxes: read the Inbox, fetch a message's raw
MIME, send a MIME message. Nothing else is reachable — no move, delete, flag or
mark-read — because we promise the customer we never change their mailbox (spec §5.2).
The allowlist below is checked on every call, and a test tries the paths it refuses.
"""
from __future__ import annotations

import base64
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from services.style.graph_source import (GRAPH_BASE, GRAPH_SCOPE, TOKEN_URL, GraphCredentials,
                                         MailboxAccessDenied, MailboxUnreachable,
                                         RequestsTransport)

_MBX = r"/users/[^/]+"
ALLOWED_PATH_PATTERNS = {
    "GET": [re.compile(p) for p in (
        rf"^{_MBX}/mailFolders/inbox$",
        rf"^{_MBX}/mailFolders/inbox/messages$",
        rf"^{_MBX}/messages(\?.*)?$",
        rf"^{_MBX}/messages/[^/]+/\$value$",
    )],
    "POST": [re.compile(rf"^{_MBX}/sendMail$")],
}


class Throttled(MailboxUnreachable):
    def __init__(self, retry_after: int) -> None:
        super().__init__(f"throttled by Microsoft; retry after {retry_after}s")
        self.retry_after = retry_after


class M365Client:
    def __init__(self, credentials: GraphCredentials, *, transport: Optional[Any] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.credentials = credentials
        self.transport = transport or RequestsTransport()
        self._clock = clock
        self._token: Optional[str] = None
        self._token_expires = 0.0

    def token(self) -> str:
        if self._token and self._clock() < self._token_expires - 60:
            return self._token
        c = self.credentials
        status, payload = self.transport.post_form(
            TOKEN_URL.format(tenant_id=c.tenant_id),
            {"client_id": c.client_id, "client_secret": c.client_secret,
             "scope": GRAPH_SCOPE, "grant_type": "client_credentials"})
        if status != 200 or not isinstance(payload, dict) or not payload.get("access_token"):
            raise MailboxUnreachable(
                f"token request failed with status {status}: "
                f"{(payload or {}).get('error_description', 'no detail')}")
        self._token = payload["access_token"]
        self._token_expires = self._clock() + float(payload.get("expires_in", 3600))
        return self._token

    def _url(self, path: str, *, method: str) -> str:
        bare = path.split("?", 1)[0] if method != "GET" else path
        if not any(p.match(bare) or p.match(path.split("?", 1)[0])
                   for p in ALLOWED_PATH_PATTERNS.get(method, [])):
            raise ValueError(f"{method} {path} is not a Graph path this feature may use")
        return f"{GRAPH_BASE}{path}"

    def _headers(self, **extra: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Accept": "application/json", **extra}

    @staticmethod
    def _mbx(mailbox: str) -> str:
        return f"/users/{quote(mailbox, safe='@')}"

    @staticmethod
    def _retry_after(payload: Any) -> int:
        try:
            return int((payload or {}).get("retry_after", 30))
        except (TypeError, ValueError, AttributeError):
            return 30

    def can_read(self, mailbox: str) -> Tuple[bool, str]:
        status, _ = self.transport.get(
            self._url(f"{self._mbx(mailbox)}/mailFolders/inbox", method="GET"), self._headers())
        if 200 <= status < 300:
            return True, f"read Inbox with HTTP {status}"
        if status in (401, 403, 404):
            return False, f"denied with HTTP {status}"
        raise MailboxUnreachable(f"Inbox read returned HTTP {status}; inconclusive")

    def probe_control(self, mailbox: str) -> Tuple[bool, str]:
        status, payload = self.transport.get(
            self._url(f"{self._mbx(mailbox)}/messages?$top=1", method="GET"), self._headers())
        if status in (401, 403, 404):
            return False, f"denied with HTTP {status}"
        if 200 <= status < 300:
            return True, f"READ SUCCEEDED with HTTP {status}"
        raise MailboxUnreachable(f"control probe returned HTTP {status}; inconclusive")

    def list_inbox(self, mailbox: str, since: datetime, *, page_size: int = 50
                   ) -> List[Dict[str, Any]]:
        stamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (self._url(f"{self._mbx(mailbox)}/mailFolders/inbox/messages", method="GET")
               + f"?$top={page_size}&$select=id,receivedDateTime"
               f"&$filter=receivedDateTime ge {stamp}&$orderby=receivedDateTime asc")
        out: List[Dict[str, Any]] = []
        while url:
            status, payload = self.transport.get(url, self._headers())
            if status == 429:
                raise Throttled(self._retry_after(payload))
            if status in (401, 403, 404):
                raise MailboxAccessDenied(f"Inbox listing denied with HTTP {status}")
            if not 200 <= status < 300:
                raise MailboxUnreachable(f"Inbox listing returned HTTP {status}")
            out.extend((payload or {}).get("value", []))
            url = (payload or {}).get("@odata.nextLink")
            if url and not url.startswith(GRAPH_BASE):
                raise ValueError("nextLink left Microsoft Graph")
        return sorted(out, key=lambda m: m.get("receivedDateTime") or "")

    def get_mime(self, mailbox: str, message_id: str) -> bytes:
        status, body = self.transport.get_bytes(
            self._url(f"{self._mbx(mailbox)}/messages/{quote(message_id, safe='')}/$value",
                      method="GET"), self._headers())
        if status == 429:
            raise Throttled(30)
        if status in (401, 403, 404):
            raise MailboxAccessDenied(f"message fetch denied with HTTP {status}")
        if not 200 <= status < 300:
            raise MailboxUnreachable(f"message fetch returned HTTP {status}")
        return body

    def send_mime(self, mailbox: str, mime: bytes) -> None:
        status, payload = self.transport.post_body(
            self._url(f"{self._mbx(mailbox)}/sendMail", method="POST"),
            self._headers(**{"Content-Type": "text/plain"}), base64.b64encode(mime))
        if status == 202 or 200 <= status < 300:
            return
        if status == 429:
            raise Throttled(self._retry_after(payload))
        if status in (401, 403):
            raise MailboxAccessDenied(f"send denied with HTTP {status}")
        raise MailboxUnreachable(f"send returned HTTP {status}")
```

Add to `RequestsTransport` in graph_source.py:

```python
    def get_bytes(self, url: str, headers: Dict[str, str]) -> Tuple[int, bytes]:
        try:
            response = egress.get(url, purpose=egress.Purpose.MAILBOX,
                                  headers=headers, timeout=self.timeout)
        except Exception as exc:
            raise MailboxUnreachable(str(exc)) from exc
        return response.status_code, response.content

    def post_body(self, url: str, headers: Dict[str, str], body: bytes) -> Tuple[int, Any]:
        try:
            response = egress.post(url, purpose=egress.Purpose.MAILBOX,
                                   headers=headers, data=body, timeout=self.timeout)
        except Exception as exc:
            raise MailboxUnreachable(str(exc)) from exc
        return response.status_code, _safe_json(response)
```

Check: `egress` permits `graph.microsoft.com` and `login.microsoftonline.com` for `Purpose.MAILBOX` (see `src/services/egress.py` around line 102). If not, add them there in this task, with a test in the existing egress test file.

- [ ] **Step 4: Run tests** — Step 2 command plus `tests/services/test_style_mailbox.py`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/mail_m365/graph_client.py src/services/style/graph_source.py tests/services/mail_m365/test_graph_client.py
git commit -m "feat(m365): a Graph client that can read the Inbox and send, and nothing else"
```

---

### Task 5: Consent, signed state, callback, setup script

**Files:**
- Create: `src/services/mail_m365/consent.py`, `src/services/mail_m365/setup_script.py`
- Test: `tests/services/mail_m365/test_consent.py`, `tests/services/mail_m365/test_setup_script.py`

**Interfaces:**
- Consumes: Task 3 `M365Settings`, `redirect_uri`, `read_integration`, `write_integration`.
- Produces:
  - `consent.ADMIN_CONSENT_URL = "https://login.microsoftonline.com/organizations/v2.0/adminconsent"`
  - `sign_state(key: bytes, actor: str, nonce: str, now: float) -> str`; `verify_state(key: bytes, state: str, now: float, max_age: int = 600) -> Tuple[str, str]` (actor, nonce) raising `StateRejected(code: str)` with codes `malformed|bad_signature|expired`.
  - `begin(conn, settings, key: bytes, actor: str, *, now=time.time) -> str` (authorization URL; stores nonce in `mail_integration.pending_nonces`, keeps last 20)
  - `complete(conn, settings, key: bytes, *, state: str, tenant: Optional[str], admin_consent: Optional[str], error: Optional[str], now=time.time) -> Dict` → returns updated integration dict; raises `StateRejected` (`reused` when nonce not pending) or `ConsentNotGiven(code)`.
  - `load_state_key(settings, *, client=None) -> bytes` (reads the secret by ARN; cached per process)
  - `setup_script.render(tenant_id: str, app_client_id: str, service_principal_object_id: Optional[str], group: str = "Procwise Mailboxes") -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/mail_m365/test_consent.py
from urllib.parse import parse_qs, urlparse

import pytest

from services.mail_m365 import consent
from services.mail_m365.config import M365Settings

KEY = b"k" * 32
S = M365Settings("app-client-id", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class MemConn:
    """bp_admin_config in memory, enough for read_integration/write_integration."""
    autocommit = True

    def __init__(self):
        self.value = None

    def cursor(self):
        conn = self

        class C:
            def execute(self, sql, params):
                if sql.startswith("INSERT"):
                    conn.value = params[1]
                self._sql = sql

            def fetchone(self):
                return (conn.value,) if conn.value else None

            def close(self):
                pass
        return C()


def test_state_round_trip():
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    assert consent.verify_state(KEY, s, now=200.0) == ("admin@x", "n1")


@pytest.mark.parametrize("tamper,code", [
    (lambda s: s[:-2] + "xx", "bad_signature"),
    (lambda s: "garbage", "malformed"),
])
def test_tampered_state_is_rejected(tamper, code):
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    with pytest.raises(consent.StateRejected) as e:
        consent.verify_state(KEY, tamper(s), now=101.0)
    assert e.value.code == code


def test_expired_state_is_rejected():
    s = consent.sign_state(KEY, "admin@x", "n1", now=100.0)
    with pytest.raises(consent.StateRejected) as e:
        consent.verify_state(KEY, s, now=100.0 + 601)
    assert e.value.code == "expired"


def test_begin_builds_admin_consent_url():
    conn = MemConn()
    url = consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0)
    q = parse_qs(urlparse(url).query)
    assert url.startswith(consent.ADMIN_CONSENT_URL)
    assert q["client_id"] == ["app-client-id"]
    assert q["redirect_uri"] == ["https://app.procwise.co.uk/admin/integrations/mail/m365/callback"]
    assert q["scope"] == ["https://graph.microsoft.com/.default"]


def _state_from(url):
    return parse_qs(urlparse(url).query)["state"][0]


def test_complete_records_the_tenant_once():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    out = consent.complete(conn, S, KEY, state=state, tenant="tenant-1",
                           admin_consent="True", error=None, now=lambda: 120.0)
    assert out["tenant_id"] == "tenant-1" and out["status"] == "connected"
    assert out["consented_by"] == "admin@x"
    with pytest.raises(consent.StateRejected) as e:   # Review Focus 2: replay
        consent.complete(conn, S, KEY, state=state, tenant="tenant-1",
                         admin_consent="True", error=None, now=lambda: 121.0)
    assert e.value.code == "reused"


def test_declined_consent_stores_nothing():
    conn = MemConn()
    state = _state_from(consent.begin(conn, S, KEY, "admin@x", now=lambda: 100.0))
    with pytest.raises(consent.ConsentNotGiven):
        consent.complete(conn, S, KEY, state=state, tenant=None, admin_consent=None,
                         error="access_denied", now=lambda: 120.0)
    from services.mail_m365.config import read_integration
    assert "tenant_id" not in read_integration(conn)
```

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_consent.py tests/services/mail_m365/test_setup_script.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement**

```python
# src/services/mail_m365/consent.py
"""Microsoft's admin-consent round trip.

The callback is the one endpoint Microsoft calls with no bearer token, so it trusts only
what we signed: `state` is HMAC-signed, names the admin who started, lives ten minutes,
and its nonce can be spent once. Consent creates our app's identity in the customer's
tenant and grants no mail access — that is granted in Exchange (spec §2.2).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlencode

from services.mail_m365.config import (M365Settings, read_integration, redirect_uri,
                                       write_integration)

ADMIN_CONSENT_URL = "https://login.microsoftonline.com/organizations/v2.0/adminconsent"
_SCOPE = "https://graph.microsoft.com/.default"
_KEEP_NONCES = 20


class StateRejected(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(f"consent state rejected: {code}")
        self.code = code


class ConsentNotGiven(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(f"Microsoft did not grant approval: {code}")
        self.code = code


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign_state(key: bytes, actor: str, nonce: str, now: float) -> str:
    body = _b64(json.dumps({"a": actor, "n": nonce, "t": int(now)}).encode())
    sig = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_state(key: bytes, state: str, now: float, max_age: int = 600) -> Tuple[str, str]:
    try:
        body, sig = state.split(".", 1)
        payload = json.loads(_unb64(body))
    except Exception:
        raise StateRejected("malformed")
    expected = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig):
        raise StateRejected("bad_signature")
    if now - float(payload.get("t", 0)) > max_age:
        raise StateRejected("expired")
    return str(payload["a"]), str(payload["n"])


def begin(conn: Any, settings: M365Settings, key: bytes, actor: str, *,
          now: Callable[[], float] = time.time) -> str:
    nonce = secrets.token_urlsafe(16)
    current = read_integration(conn)
    pending = (list(current.get("pending_nonces") or []) + [nonce])[-_KEEP_NONCES:]
    write_integration(conn, {**current, "pending_nonces": pending}, actor=actor)
    return ADMIN_CONSENT_URL + "?" + urlencode({
        "client_id": settings.client_id,
        "redirect_uri": redirect_uri(settings),
        "scope": _SCOPE,
        "state": sign_state(key, actor, nonce, now()),
    })


def complete(conn: Any, settings: M365Settings, key: bytes, *, state: str,
             tenant: Optional[str], admin_consent: Optional[str], error: Optional[str],
             now: Callable[[], float] = time.time) -> Dict[str, Any]:
    actor, nonce = verify_state(key, state, now())
    current = read_integration(conn)
    pending = list(current.get("pending_nonces") or [])
    if nonce not in pending:
        raise StateRejected("reused")
    pending.remove(nonce)
    write_integration(conn, {**current, "pending_nonces": pending}, actor=actor)
    if error or str(admin_consent).lower() != "true" or not tenant:
        raise ConsentNotGiven(error or "not_granted")
    updated = {**current, "pending_nonces": pending, "provider": "m365",
               "tenant_id": tenant, "status": "connected", "consented_by": actor,
               "consented_at": datetime.fromtimestamp(now(), tz=timezone.utc).isoformat()}
    return write_integration(conn, updated, actor=actor)


_KEY_CACHE: Dict[str, bytes] = {}


def load_state_key(settings: M365Settings, *, client: Any = None) -> bytes:
    if settings.state_key_arn in _KEY_CACHE:
        return _KEY_CACHE[settings.state_key_arn]
    if client is None:  # pragma: no cover - real AWS only
        from src.services import egress
        client = egress.aws_client("secretsmanager", purpose=egress.Purpose.SECRETS,
                                   region_name=settings.state_key_arn.split(":")[3])
    raw = client.get_secret_value(SecretId=settings.state_key_arn)["SecretString"]
    key = raw.encode()
    if len(key) < 32:
        raise ValueError("the state signing key must be at least 32 bytes")
    _KEY_CACHE[settings.state_key_arn] = key
    return key
```

```python
# src/services/mail_m365/setup_script.py
"""The PowerShell an IT admin runs to restrict Procwise to their chosen mailboxes.

Built from Microsoft's "RBAC for Applications in Exchange Online" guidance: register our
app's service principal in Exchange, scope it to members of one mail-enabled security
group, and grant send + read only within that scope. Permissions granted in Entra would
ADD to this, which is why our app requests none there (spec §2.2). Verify the cmdlets
against a live tenant before release — see docs/m365-setup.md.
"""
from __future__ import annotations

from typing import Optional


def render(tenant_id: str, app_client_id: str, service_principal_object_id: Optional[str],
           group: str = "Procwise Mailboxes") -> str:
    sp_id = service_principal_object_id or "<Object ID of 'Procwise Mail' under Entra > Enterprise applications>"
    return f"""# Restrict Procwise to your chosen shared mailboxes. No secret is needed to run this.
# Tenant: {tenant_id}
# 1. Create a mail-enabled security group called "{group}" and add each shared mailbox
#    Procwise should use. Adding or removing a mailbox later = changing this group.
Connect-ExchangeOnline

# 2. Register Procwise's app in Exchange.
New-ServicePrincipal -AppId {app_client_id} -ObjectId {sp_id} -DisplayName "Procwise Mail"

# 3. A scope containing only members of the group.
$group = Get-DistributionGroup -Identity "{group}"
New-ManagementScope -Name "Procwise Mailboxes Scope" -RecipientRestrictionFilter "MemberOfGroup -eq '$($group.DistinguishedName)'"

# 4. Send and read, only inside that scope.
New-ManagementRoleAssignment -App {app_client_id} -Role "Application Mail.Send" -CustomResourceScope "Procwise Mailboxes Scope"
New-ManagementRoleAssignment -App {app_client_id} -Role "Application Mail.ReadWrite" -CustomResourceScope "Procwise Mailboxes Scope"

# 5. Check (should say InScope = True for a shared mailbox, False for anyone else).
Test-ServicePrincipalAuthorization -Identity {app_client_id} -Resource <a shared mailbox address>
"""
```

- [ ] **Step 4: Run tests** — Step 2 command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/mail_m365/consent.py src/services/mail_m365/setup_script.py tests/services/mail_m365/test_consent.py tests/services/mail_m365/test_setup_script.py
git commit -m "feat(m365): admin consent with a signed single-use state, and the Exchange scoping script"
```

---

### Task 6: Admin service — add, verify, test send, default, remove, disconnect

**Files:**
- Create: `src/services/mail_m365/admin.py`
- Test: `tests/services/mail_m365/test_admin.py`

**Interfaces:**
- Consumes: Task 2 repo methods, Task 3 config, Task 4 `M365Client`, `MailboxAccessDenied`, `MailboxUnreachable`, `ScopeVerificationFailed`.
- Produces (all take `conn` and an injectable `client_factory: Callable[[str], M365Client]` keyed by tenant id, default builds from settings + `resolve_credentials(settings.secret_arn, tenant_id=...)`):
  - `status(conn, *, settings) -> Dict` → `{configured: bool, connected: bool, tenant_id, consented_by, consented_at, mailboxes: [{binding_id, address, is_default, health_state, verified_at, last_test_send_at, can_be_default}]}`
  - `add_mailbox(conn, address: str, control_mailbox: str, *, actor, settings, client_factory=None) -> Dict` → returns `{ok: True, mailbox}` or raises `MailboxRefused(code, message)` with codes `not_connected`, `duplicate`, `same_as_control`, `cannot_read`, `over_scoped`, `inconclusive`
  - `verify_mailbox(conn, binding_id, control_mailbox, …) -> Dict` (same codes)
  - `test_send(conn, binding_id, to_address: str, …) -> Dict` → sends a fixed plain message, records `last_test_send_at`
  - `make_default(conn, binding_id, *, actor) -> Dict`
  - `remove_mailbox(conn, binding_id, *, actor) -> Dict`
  - `disconnect(conn, *, actor) -> Dict` (deactivate all org bindings; status `disconnected`; keep tenant id for the record)
  - exception `MailboxRefused(code: str, message: str)`

Plain-English messages (shown to the admin as given):
- `over_scoped`: "We could also read {control}, so access is not restricted to your chosen mailboxes yet. Run the commands above, wait up to 30 minutes for Microsoft to apply them, then try again."
- `cannot_read`: "We could not read {address}. Check it is in the “Procwise Mailboxes” group; changes can take up to 30 minutes."
- `duplicate`: "{address} is already connected."
- `same_as_control`: "The control mailbox must be a different mailbox — normally your own."
- `inconclusive`: "Microsoft did not give a clear answer ({detail}). Nothing was changed; try again shortly."
- `not_connected`: "Connect Microsoft 365 first."

- [ ] **Step 1: Write the failing tests**

Use an in-memory fake repo to keep these fast and DB-free; the repo itself is proven in Task 2.

```python
# tests/services/mail_m365/test_admin.py
from datetime import datetime, timezone

import pytest

from services.mail_m365 import admin
from services.mail_m365.config import M365Settings
from services.style.mailbox import MailboxBinding, ScopeVerificationFailed

S = M365Settings("cid", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class FakeClient:
    def __init__(self, readable=(), sent=None):
        self.readable = set(readable)
        self.sent = sent if sent is not None else []

    def can_read(self, m):
        return (m in self.readable, "stub")

    def probe_control(self, m):
        return (m in self.readable, "stub")

    def send_mime(self, m, mime):
        self.sent.append((m, mime))


@pytest.fixture
def env(monkeypatch):
    state = {"integration": {"status": "connected", "tenant_id": "t1"}, "rows": {}, "next": 1}

    class Repo:
        def __init__(self, conn=None): pass

        def list_org(self, include_inactive=False):
            return [b for b in state["rows"].values() if include_inactive or b.is_active]

        def create(self, **kw):
            b = MailboxBinding(state["next"], kw["user_ref"], kw["provider"],
                               kw["mailbox_address"], kw["role"], kw["credential_ref"],
                               None, None, None, None, "OK", False)
            state["rows"][b.binding_id] = b
            state["next"] += 1
            return b

        def verify_scope(self, binding_id, *, control_mailbox, probe):
            allowed, _ = probe(control_mailbox)
            if allowed:
                raise ScopeVerificationFailed("over")
            b = state["rows"][binding_id]
            b = MailboxBinding(*[getattr(b, f) for f in (
                "binding_id", "user_ref", "provider", "mailbox_address", "role",
                "credential_ref", "scope_policy_ref")], datetime.now(timezone.utc),
                "ev", None, "OK", True)
            state["rows"][binding_id] = b
            return b

        def get(self, i):
            return state["rows"].get(i)

        def deactivate(self, i, *, reason):
            b = state["rows"][i]
            state["rows"][i] = MailboxBinding(**{**b.__dict__, "is_active": False})
            return state["rows"][i]

        def record_test_send(self, i):
            b = state["rows"][i]
            state["rows"][i] = MailboxBinding(**{**b.__dict__,
                                                 "last_test_send_at": datetime.now(timezone.utc)})
            return state["rows"][i]

    monkeypatch.setattr(admin, "MailboxBindingRepository", Repo)
    monkeypatch.setattr(admin, "read_integration", lambda conn: state["integration"])
    monkeypatch.setattr(admin, "write_integration",
                        lambda conn, v, actor: state.update(integration=v) or v)
    return state


def test_add_activates_when_control_is_denied(env):
    c = FakeClient(readable={"procurement@cust.com"})
    out = admin.add_mailbox(None, " Procurement@Cust.com ", "me@cust.com", actor="a",
                            settings=S, client_factory=lambda t: c)
    assert out["mailbox"]["address"] == "procurement@cust.com"   # Review Focus 4
    assert out["mailbox"]["active"] is True


def test_over_scoped_is_refused_in_plain_english(env):
    c = FakeClient(readable={"procurement@cust.com", "me@cust.com"})
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)
    assert e.value.code == "over_scoped"
    assert "not restricted" in e.value.message


def test_unreadable_mailbox_is_refused_before_creating_a_binding(env):
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: FakeClient())
    assert e.value.code == "cannot_read"
    assert env["rows"] == {}


def test_duplicate_address_is_refused(env):                      # Review Focus 1
    c = FakeClient(readable={"procurement@cust.com"})
    admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                      settings=S, client_factory=lambda t: c)
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "PROCUREMENT@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)
    assert e.value.code == "duplicate"


def test_control_equal_to_mailbox_is_refused(env):              # Review Focus 4
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "Procurement@cust.com", " procurement@CUST.com", actor="a",
                          settings=S, client_factory=lambda t: FakeClient())
    assert e.value.code == "same_as_control"


def test_not_connected_is_refused(env):
    env["integration"] = {}
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "p@c.com", "me@c.com", actor="a", settings=S,
                          client_factory=lambda t: FakeClient())
    assert e.value.code == "not_connected"


def test_test_send_sends_once_and_records_it(env):
    sent = []
    c = FakeClient(readable={"procurement@cust.com"}, sent=sent)
    b = admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)["mailbox"]
    admin.test_send(None, b["binding_id"], "me@cust.com", actor="a", settings=S,
                    client_factory=lambda t: c)
    assert len(sent) == 1 and sent[0][0] == "procurement@cust.com"
    assert b"To: me@cust.com" in sent[0][1]
    assert env["rows"][b["binding_id"]].last_test_send_at is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_admin.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/services/mail_m365/admin.py
"""What an admin does on Integrations → Email, as plain functions the router calls.

The order inside add_mailbox is the product claim: read the mailbox first (otherwise a
binding row would exist for a mailbox we cannot use), then create it inactive, then prove
the control mailbox is denied — only that activates it.
"""
from __future__ import annotations

from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Any, Callable, Dict, Optional

from services.mail_m365.config import M365Settings, read_integration, write_integration
from services.mail_m365.graph_client import M365Client
from services.style.graph_source import MailboxUnreachable, resolve_credentials
from services.style.mailbox import (ORG_USER_REF, PROVIDER_GRAPH, ROLE_SEND_RECEIVE,
                                    MailboxBinding, MailboxBindingRepository,
                                    ScopeVerificationFailed, normalise_address)

ClientFactory = Callable[[str], M365Client]

_MESSAGES = {
    "over_scoped": ("We could also read {control}, so access is not restricted to your chosen "
                    "mailboxes yet. Run the commands above, wait up to 30 minutes for Microsoft "
                    "to apply them, then try again."),
    "cannot_read": ("We could not read {address}. Check it is in the “Procwise Mailboxes” "
                    "group; changes can take up to 30 minutes."),
    "duplicate": "{address} is already connected.",
    "same_as_control": "The control mailbox must be a different mailbox — normally your own.",
    "inconclusive": ("Microsoft did not give a clear answer ({detail}). Nothing was changed; "
                     "try again shortly."),
    "not_connected": "Connect Microsoft 365 first.",
    "not_found": "That mailbox is not connected.",
}


class MailboxRefused(Exception):
    def __init__(self, code: str, **fmt: Any) -> None:
        self.code = code
        self.message = _MESSAGES[code].format(**fmt)
        super().__init__(self.message)


def _default_factory(settings: M365Settings) -> ClientFactory:
    def build(tenant_id: str) -> M365Client:
        return M365Client(resolve_credentials(settings.secret_arn, tenant_id=tenant_id))
    return build


def _tenant(conn: Any) -> str:
    integration = read_integration(conn)
    if integration.get("status") != "connected" or not integration.get("tenant_id"):
        raise MailboxRefused("not_connected")
    return integration["tenant_id"]


def _view(b: MailboxBinding) -> Dict[str, Any]:
    return {"binding_id": b.binding_id, "address": b.mailbox_address, "active": b.is_active,
            "is_default": b.is_default, "health_state": b.health_state,
            "verified_at": b.scope_verified_at, "last_test_send_at": b.last_test_send_at,
            "can_be_default": bool(b.is_active and b.last_test_send_at and not b.is_default)}


def status(conn: Any, *, settings: Optional[M365Settings]) -> Dict[str, Any]:
    integration = read_integration(conn) if conn is not None else {}
    connected = integration.get("status") == "connected"
    mailboxes = [_view(b) for b in MailboxBindingRepository(conn).list_org()] if connected else []
    return {"configured": settings is not None, "connected": connected,
            "tenant_id": integration.get("tenant_id"),
            "consented_by": integration.get("consented_by"),
            "consented_at": integration.get("consented_at"), "mailboxes": mailboxes}


def add_mailbox(conn: Any, address: str, control_mailbox: str, *, actor: str,
                settings: M365Settings, client_factory: Optional[ClientFactory] = None
                ) -> Dict[str, Any]:
    address, control = normalise_address(address), normalise_address(control_mailbox)
    tenant = _tenant(conn)
    if address == control:
        raise MailboxRefused("same_as_control")
    repo = MailboxBindingRepository(conn)
    if any(b.mailbox_address == address for b in repo.list_org()):
        raise MailboxRefused("duplicate", address=address)
    client = (client_factory or _default_factory(settings))(tenant)
    try:
        readable, detail = client.can_read(address)
    except MailboxUnreachable as exc:
        raise MailboxRefused("inconclusive", detail=str(exc))
    if not readable:
        raise MailboxRefused("cannot_read", address=address)
    binding = repo.create(user_ref=ORG_USER_REF, provider=PROVIDER_GRAPH,
                          mailbox_address=address, role=ROLE_SEND_RECEIVE,
                          credential_ref=settings.secret_arn)
    return {"ok": True, "mailbox": _view(_prove(repo, binding.binding_id, control, client))}


def _prove(repo, binding_id: int, control: str, client: M365Client) -> MailboxBinding:
    try:
        return repo.verify_scope(binding_id, control_mailbox=control, probe=client.probe_control)
    except ScopeVerificationFailed:
        raise MailboxRefused("over_scoped", control=control)
    except MailboxUnreachable as exc:
        raise MailboxRefused("inconclusive", detail=str(exc))


def verify_mailbox(conn: Any, binding_id: int, control_mailbox: str, *, actor: str,
                   settings: M365Settings, client_factory: Optional[ClientFactory] = None
                   ) -> Dict[str, Any]:
    tenant = _tenant(conn)
    repo = MailboxBindingRepository(conn)
    b = repo.get(binding_id)
    if b is None or b.user_ref != ORG_USER_REF:
        raise MailboxRefused("not_found")
    control = normalise_address(control_mailbox)
    if control == b.mailbox_address:
        raise MailboxRefused("same_as_control")
    client = (client_factory or _default_factory(settings))(tenant)
    return {"ok": True, "mailbox": _view(_prove(repo, binding_id, control, client))}


def _test_message(sender: str, to: str) -> bytes:
    msg = EmailMessage()
    msg["From"], msg["To"] = sender, to
    msg["Subject"] = "Procwise test email"
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender.partition("@")[2] or None)
    msg.set_content(f"This is a test from Procwise. It was sent from {sender} through "
                    "Microsoft 365, and a copy should be in that mailbox's Sent Items.")
    return bytes(msg)


def test_send(conn: Any, binding_id: int, to_address: str, *, actor: str,
              settings: M365Settings, client_factory: Optional[ClientFactory] = None
              ) -> Dict[str, Any]:
    tenant = _tenant(conn)
    repo = MailboxBindingRepository(conn)
    b = repo.get(binding_id)
    if b is None or not b.is_active or b.user_ref != ORG_USER_REF:
        raise MailboxRefused("not_found")
    client = (client_factory or _default_factory(settings))(tenant)
    client.send_mime(b.mailbox_address, _test_message(b.mailbox_address,
                                                      normalise_address(to_address)))
    return {"ok": True, "mailbox": _view(repo.record_test_send(binding_id))}


def make_default(conn: Any, binding_id: int, *, actor: str) -> Dict[str, Any]:
    try:
        return {"ok": True, "mailbox": _view(MailboxBindingRepository(conn).set_default(binding_id))}
    except ValueError as exc:
        raise MailboxRefused("not_found") if "no mailbox" in str(exc) else exc


def remove_mailbox(conn: Any, binding_id: int, *, actor: str) -> Dict[str, Any]:
    repo = MailboxBindingRepository(conn)
    if repo.get(binding_id) is None:
        raise MailboxRefused("not_found")
    return {"ok": True, "mailbox": _view(repo.deactivate(binding_id, reason=f"removed by {actor}"))}


def disconnect(conn: Any, *, actor: str) -> Dict[str, Any]:
    repo = MailboxBindingRepository(conn)
    for b in repo.list_org():
        repo.deactivate(b.binding_id, reason=f"Microsoft 365 disconnected by {actor}")
    integration = read_integration(conn)
    write_integration(conn, {**integration, "status": "disconnected",
                             "disconnected_by": actor}, actor=actor)
    return {"ok": True, "note": ("Procwise has stopped using your mailboxes. To remove its "
                                 "access entirely, delete “Procwise Mail” under Microsoft Entra "
                                 "→ Enterprise applications.")}
```

`make_default` must let the repo's `ValueError` ("a mailbox can be the default only once …") reach the router as a 422 with that text; keep the re-raise as written.

- [ ] **Step 4: Run tests** — Step 2 command. Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/services/mail_m365/admin.py tests/services/mail_m365/test_admin.py
git commit -m "feat(m365): adding a shared mailbox reads it, then proves the control is denied"
```

---

### Task 7: Endpoints under Admin → Integrations → Email

**Files:**
- Create: `src/api/routers/mail_integration.py`
- Modify: `src/api/main.py` (import near line 75; register `mail_integration.router` in `_AUTHENTICATED_ROUTERS` list near line 640 **and** `mail_integration.callback_router` with a plain `app.include_router(...)` — the callback has no bearer token)
- Test: `tests/api/test_mail_integration_router.py`

**Interfaces:**
- Consumes: Task 3/5/6 functions; `api.auth.require_user`; `api.endpoint_gate.require` with action `"mailbox.bind"`; `services.db.get_conn`.
- Produces: routes in spec §4.3. Callback redirects (302) to `f"{settings.public_base_url}/?view=settings&mail=connected"` or `…&mail_error=<code>`. `GET /admin/integrations/mail` returns `admin.status(...)` plus `canConfigure: True` (reaching it at all means the gate passed); a non-admin gets 403 from the gate — the UI uses that to hide the entry.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api/test_mail_integration_router.py
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.routers import mail_integration as r
from services.mail_m365 import admin, consent
from services.mail_m365.config import M365Settings

S = M365Settings("cid", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class P:
    subject, email, username = "u1", "admin@cust.com", "admin"


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(r.router)
    app.include_router(r.callback_router)
    app.dependency_overrides[r.require_user] = lambda: P()
    calls = []
    monkeypatch.setattr(r, "gate", lambda action, principal, **kw: calls.append(action))
    monkeypatch.setattr(r, "load_settings", lambda: S)
    monkeypatch.setattr(r, "_conn", lambda: _NullConn())
    monkeypatch.setattr(r, "load_state_key", lambda s: b"k" * 32)
    c = TestClient(app)
    c.gate_calls = calls
    return c


class _NullConn:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_every_admin_route_is_gated(client, monkeypatch):
    monkeypatch.setattr(admin, "status", lambda conn, settings: {"connected": False})
    client.get("/admin/integrations/mail")
    assert client.gate_calls == ["mailbox.bind"]


def test_gate_refusal_is_403(client, monkeypatch):
    def deny(action, principal, **kw):
        raise HTTPException(status_code=403, detail="not permitted")
    monkeypatch.setattr(r, "gate", deny)
    assert client.get("/admin/integrations/mail").status_code == 403


def test_not_configured_says_so(client, monkeypatch):
    monkeypatch.setattr(r, "load_settings", lambda: None)
    res = client.post("/admin/integrations/mail/m365/connect")
    assert res.status_code == 503
    assert "not configured on this server" in res.json()["detail"]


def test_refusal_message_is_passed_through(client, monkeypatch):
    def refuse(*a, **k):
        raise admin.MailboxRefused("duplicate", address="p@c.com")
    monkeypatch.setattr(admin, "add_mailbox", refuse)
    res = client.post("/admin/integrations/mail/mailboxes",
                      json={"address": "p@c.com", "control_mailbox": "me@c.com"})
    assert res.status_code == 409
    assert res.json()["detail"] == {"code": "duplicate", "message": "p@c.com is already connected."}


def test_callback_needs_no_token_and_redirects(client, monkeypatch):
    client.app.dependency_overrides.clear()          # no principal at all
    monkeypatch.setattr(consent, "complete", lambda *a, **k: {"status": "connected"})
    res = client.get("/admin/integrations/mail/m365/callback",
                     params={"state": "s", "tenant": "t", "admin_consent": "True"},
                     follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"].endswith("mail=connected")


def test_callback_with_bad_state_redirects_with_error(client, monkeypatch):
    client.app.dependency_overrides.clear()
    def bad(*a, **k):
        raise consent.StateRejected("reused")
    monkeypatch.setattr(consent, "complete", bad)
    res = client.get("/admin/integrations/mail/m365/callback",
                     params={"state": "s"}, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"].endswith("mail_error=reused")
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/api/test_mail_integration_router.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/api/routers/mail_integration.py
"""Admin → Integrations → Email: connect Microsoft 365 and choose shared mailboxes.

Every route except the callback is `mailbox.bind` (configure — Admin only), checked by the
server; the screen never decides on its own who may connect a company mailbox. The
callback is called by Microsoft's redirect of the admin's browser and carries no bearer
token: it trusts only the signed, single-use state (services/mail_m365/consent.py).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.db import get_conn
from services.mail_m365 import admin, consent
from services.mail_m365.config import load_settings, read_integration
from services.mail_m365.consent import load_state_key
from services.mail_m365.setup_script import render as render_script

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/integrations/mail", tags=["Mail integration"])
callback_router = APIRouter(prefix="/admin/integrations/mail", tags=["Mail integration"])

_AGENT = "MailIntegrationRouter"
_STATUS = {"not_connected": 409, "duplicate": 409, "same_as_control": 422,
           "cannot_read": 422, "over_scoped": 422, "inconclusive": 503, "not_found": 404}


def _conn():
    return get_conn()


def _actor(p: Any) -> str:
    return str(getattr(p, "email", None) or getattr(p, "username", None)
               or getattr(p, "subject", None) or "unknown")


def _settings_or_503():
    s = load_settings()
    if s is None:
        raise HTTPException(503, "Microsoft 365 is not configured on this server.")
    return s


def _refused(exc: admin.MailboxRefused) -> HTTPException:
    return HTTPException(_STATUS.get(exc.code, 422), {"code": exc.code, "message": exc.message})


class AddBody(BaseModel):
    address: str = Field(..., min_length=3)
    control_mailbox: str = Field(..., min_length=3)


class VerifyBody(BaseModel):
    control_mailbox: str = Field(..., min_length=3)


class DefaultBody(BaseModel):
    is_default: bool


@router.get("", summary="Microsoft 365 connection and shared mailboxes")
def get_status(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "read"})
    with _conn() as conn:
        return {**admin.status(conn, settings=load_settings()), "canConfigure": True}


@router.post("/m365/connect", summary="Start Microsoft's admin approval")
def connect(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "connect"})
    s = _settings_or_503()
    with _conn() as conn:
        return {"authorization_url": consent.begin(conn, s, load_state_key(s), _actor(principal))}


@callback_router.get("/m365/callback", include_in_schema=False)
def callback(state: str = "", tenant: Optional[str] = None,
             admin_consent: Optional[str] = None, error: Optional[str] = None):
    s = _settings_or_503()
    back = f"{s.public_base_url.rstrip('/')}/?view=settings"
    try:
        with _conn() as conn:
            consent.complete(conn, s, load_state_key(s), state=state, tenant=tenant,
                             admin_consent=admin_consent, error=error)
    except consent.StateRejected as exc:
        logger.warning("mail consent callback refused: %s", exc.code)
        return RedirectResponse(f"{back}&mail_error={exc.code}", status_code=302)
    except consent.ConsentNotGiven as exc:
        return RedirectResponse(f"{back}&mail_error={exc.code}", status_code=302)
    return RedirectResponse(f"{back}&mail=connected", status_code=302)


@router.get("/m365/setup-script", response_class=PlainTextResponse,
            summary="PowerShell that restricts Procwise to the chosen mailboxes")
def setup_script(principal: Any = Depends(require_user),
                 service_principal_object_id: Optional[str] = None) -> str:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "script"})
    s = _settings_or_503()
    with _conn() as conn:
        tenant = read_integration(conn).get("tenant_id") or "<your tenant id>"
    return render_script(tenant, s.client_id, service_principal_object_id)


@router.post("/mailboxes", summary="Add a shared mailbox and prove access is restricted")
def add(body: AddBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "add", "address": body.address})
    s = _settings_or_503()
    try:
        with _conn() as conn:
            return admin.add_mailbox(conn, body.address, body.control_mailbox,
                                     actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.post("/mailboxes/{binding_id}/verify", summary="Re-run the access proof")
def verify(binding_id: int, body: VerifyBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "verify", "binding_id": binding_id})
    s = _settings_or_503()
    try:
        with _conn() as conn:
            return admin.verify_mailbox(conn, binding_id, body.control_mailbox,
                                        actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.post("/mailboxes/{binding_id}/test-send", summary="Send a test email to yourself")
def test_send(binding_id: int, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "test_send", "binding_id": binding_id})
    s = _settings_or_503()
    to = getattr(principal, "email", None)
    if not to:
        raise HTTPException(422, "Your sign-in has no email address to send the test to.")
    try:
        with _conn() as conn:
            return admin.test_send(conn, binding_id, to, actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.patch("/mailboxes/{binding_id}", summary="Make this the default sending mailbox")
def set_default(binding_id: int, body: DefaultBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "default", "binding_id": binding_id})
    if not body.is_default:
        raise HTTPException(422, "Choose another mailbox as the default instead.")
    try:
        with _conn() as conn:
            return admin.make_default(conn, binding_id, actor=_actor(principal))
    except admin.MailboxRefused as exc:
        raise _refused(exc)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/mailboxes/{binding_id}", summary="Stop using this mailbox")
def remove(binding_id: int, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "remove", "binding_id": binding_id})
    try:
        with _conn() as conn:
            return admin.remove_mailbox(conn, binding_id, actor=_actor(principal))
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.delete("", summary="Disconnect Microsoft 365")
def disconnect(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "disconnect"})
    with _conn() as conn:
        return admin.disconnect(conn, actor=_actor(principal))
```

In `src/api/main.py`: add `from api.routers import mail_integration as mail_integration_router` beside the other router imports; add `mail_integration_router.router,` to `_AUTHENTICATED_ROUTERS`; after that loop add `app.include_router(mail_integration_router.callback_router)` with a comment `# Microsoft redirects the admin's browser here with no bearer token; see the router docstring.` Check how other modules import `services.*` vs `src.services.*` in routers (daily_brief uses `src.services`); match whichever `api.auth` resolution the app uses so `require_user` override in the test is the same object.

- [ ] **Step 4: Run tests** — Step 2 command. Expected: 6 passed. Then `curl -s localhost:8000/openapi.json | python -m json.tool | grep integrations/mail` on a locally started server (see `procwise.service` for the start command) shows the routes.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/mail_integration.py src/api/main.py tests/api/test_mail_integration_router.py
git commit -m "feat(m365): admin endpoints for connecting Microsoft 365 and choosing mailboxes"
```

---

### Task 8: Sending — the company mailbox, or held; never quietly SES

**Files:**
- Create: `src/services/mail_m365/sending.py`
- Modify: `src/services/email_service.py` (`EmailSendResult` line 26: add `held_reason: Optional[str] = None`, `transport: str = "ses"`; `send_email` line 55: call the decision after the MIME message is built, before SMTP credentials are fetched)
- Modify: `src/services/email_dispatch_service.py` (after `send_result = self.email_service.send_email(...)` ~line 475: when `send_result.held_reason`, put it on `dispatch_payload["held_reason"]` before `_update_draft_status`, and record an admin alert)
- Test: `tests/services/mail_m365/test_sending.py`

**Interfaces:**
- Consumes: Task 2 `get_default_sender`, `get_org_default_any`; Task 3 config; Task 4 `M365Client`, `Throttled`, `MailboxAccessDenied`.
- Produces:
  - `Decision(kind: Literal["ses","graph","held"], binding: Optional[MailboxBinding], reason: Optional[str])`
  - `decide(conn=None, *, repo=None, integration=None) -> Decision`
  - `send_via_graph(binding, mime: bytes, *, client_factory=None, settings=None) -> None` (raises on failure)
  - Held reasons (exact strings): `"company mailbox unavailable: {address} is {state}"`, `"company mailbox unavailable: Microsoft 365 is connected but no default mailbox is set"`, `"company mailbox unavailable: {error}"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/mail_m365/test_sending.py
from datetime import datetime, timezone

import pytest

from services import email_service as es
from services.mail_m365 import sending
from services.style.mailbox import MailboxBinding

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def B(active=True, health="OK", default=True):
    return MailboxBinding(7, "org", "graph", "procurement@cust.com", "send_receive",
                          "arn:aws:secretsmanager:x:1:secret:a", None, NOW, "ev", None,
                          health, active, default, NOW)


class Repo:
    def __init__(self, default=None, any_=None):
        self._d, self._a = default, any_

    def get_default_sender(self): return self._d
    def get_org_default_any(self): return self._a


def test_not_connected_uses_ses():
    d = sending.decide(repo=Repo(), integration={})
    assert d.kind == "ses"


def test_connected_and_healthy_uses_graph():
    b = B()
    d = sending.decide(repo=Repo(default=b, any_=b), integration={"status": "connected"})
    assert d.kind == "graph" and d.binding is b


def test_connected_but_revoked_is_held():
    b = B(active=False, health="REVOKED", default=False)
    d = sending.decide(repo=Repo(default=None, any_=b), integration={"status": "connected"})
    assert d.kind == "held" and "REVOKED" in d.reason


def test_connected_without_default_is_held():
    d = sending.decide(repo=Repo(), integration={"status": "connected"})
    assert d.kind == "held" and "no default mailbox" in d.reason


def test_degraded_default_is_held():
    b = B(health="DEGRADED")
    d = sending.decide(repo=Repo(default=b, any_=b), integration={"status": "connected"})
    assert d.kind == "held"


def test_disconnected_after_use_is_held_not_ses():                 # Review Focus 3
    b = B(active=False, default=False)
    d = sending.decide(repo=Repo(default=None, any_=b), integration={"status": "disconnected"})
    assert d.kind == "held"


class _Svc(es.EmailService):
    def __init__(self):  # skip agent_nick wiring
        import logging
        self.logger = logging.getLogger("t")
        self.smtp_calls = 0

    def _fetch_smtp_credentials(self, **k):
        self.smtp_calls += 1
        return "u", "p"

    def _deliver_via_smtp(self, *a, **k):
        self.smtp_calls += 1


def test_held_send_never_touches_smtp(monkeypatch):
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("held", None, "company mailbox unavailable: x"))
    svc = _Svc()
    res = svc.send_email("s", "<p>b</p>", ["sup@x.com"], "ignored@procwise.co.uk")
    assert res.success is False and res.held_reason.startswith("company mailbox unavailable")
    assert svc.smtp_calls == 0


def test_graph_send_uses_the_shared_mailbox_as_from(monkeypatch):
    b = B()
    sent = {}
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("graph", b, None))
    monkeypatch.setattr(sending, "send_via_graph",
                        lambda binding, mime, **k: sent.update(addr=binding.mailbox_address, mime=mime))
    svc = _Svc()
    res = svc.send_email("s", "<p>b</p>", ["sup@x.com"], "ignored@procwise.co.uk",
                         headers={"X-Procwise-RFQ-ID": "RFQ-1"})
    assert res.success and res.transport == "graph" and svc.smtp_calls == 0
    assert b"From: procurement@cust.com" in sent["mime"]
    assert b"X-Procwise-RFQ-ID: RFQ-1" in sent["mime"]


def test_graph_failure_is_held_not_retried_on_ses(monkeypatch):
    b = B()
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("graph", b, None))
    def boom(*a, **k):
        raise RuntimeError("HTTP 500")
    monkeypatch.setattr(sending, "send_via_graph", boom)
    svc = _Svc()
    res = svc.send_email("s", "b", ["sup@x.com"], "x@y")
    assert res.success is False and "HTTP 500" in res.held_reason and svc.smtp_calls == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_sending.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/services/mail_m365/sending.py
"""Which way a supplier email leaves.

Not connected → SES, exactly as before. Connected and the default mailbox healthy →
Microsoft 365, from that mailbox. Anything in between → HELD. A supplier who has been
writing to procurement@customer.com and suddenly hears from a Procwise address is right
to be suspicious, so a broken company mailbox stops sending rather than changing sender
(spec §5.1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from services.style.mailbox import HEALTH_OK, MailboxBinding, MailboxBindingRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    kind: str                       # "ses" | "graph" | "held"
    binding: Optional[MailboxBinding]
    reason: Optional[str]


def decide(conn: Any = None, *, repo: Any = None,
           integration: Optional[Dict[str, Any]] = None) -> Decision:
    if integration is None or repo is None:
        from services.db import get_conn
        from services.mail_m365.config import read_integration
        with get_conn() as c:
            integration = read_integration(c) if integration is None else integration
            repo = repo or MailboxBindingRepository(c)
            return _decide(repo, integration)
    return _decide(repo, integration)


def _decide(repo: Any, integration: Dict[str, Any]) -> Decision:
    default = repo.get_default_sender()
    ever = repo.get_org_default_any()
    connected = integration.get("status") == "connected"
    if not connected and ever is None:
        return Decision("ses", None, None)
    if connected and default is not None and default.is_active and default.health_state == HEALTH_OK:
        return Decision("graph", default, None)
    if default is not None:
        return Decision("held", default,
                        f"company mailbox unavailable: {default.mailbox_address} is {default.health_state}")
    if ever is not None:
        state = "REVOKED" if ever.health_state == "REVOKED" else (
            "disconnected" if not connected else "inactive")
        return Decision("held", ever,
                        f"company mailbox unavailable: {ever.mailbox_address} is {state}")
    return Decision("held", None, "company mailbox unavailable: Microsoft 365 is connected "
                                  "but no default mailbox is set")


def send_via_graph(binding: MailboxBinding, mime: bytes, *,
                   client_factory: Optional[Callable[[str], Any]] = None,
                   settings: Any = None) -> None:
    from services.db import get_conn
    from services.mail_m365.admin import _default_factory
    from services.mail_m365.config import load_settings, read_integration
    settings = settings or load_settings()
    if settings is None:
        raise RuntimeError("Microsoft 365 is not configured on this server")
    with get_conn() as c:
        tenant = read_integration(c).get("tenant_id")
    client = (client_factory or _default_factory(settings))(tenant)
    client.send_mime(binding.mailbox_address, mime)
```

In `email_service.py`:

```python
@dataclass
class EmailSendResult:
    """Outcome of a send attempt."""

    success: bool
    message_id: Optional[str] = None
    held_reason: Optional[str] = None
    transport: str = "ses"
```

In `send_email`, right after the attachments loop and **before** `self._fetch_smtp_credentials()`:

```python
        from services.mail_m365 import sending as _m365
        try:
            decision = _m365.decide()
        except Exception as exc:  # cannot tell whether a company mailbox applies: hold, never guess
            self.logger.error("Could not decide the mail transport: %s", exc)
            return EmailSendResult(False, generated_message_id,
                                   held_reason=f"company mailbox unavailable: {exc}")
        if decision.kind == "held":
            return EmailSendResult(False, generated_message_id, held_reason=decision.reason)
        if decision.kind == "graph":
            del msg["From"]
            msg["From"] = decision.binding.mailbox_address
            try:
                _m365.send_via_graph(decision.binding, msg.as_bytes())
            except Exception as exc:
                self.logger.error("Microsoft 365 send failed: %s", exc)
                return EmailSendResult(False, generated_message_id, transport="graph",
                                       held_reason=f"company mailbox unavailable: {exc}")
            return EmailSendResult(True, generated_message_id, transport="graph")
```

Note the `try/except` on `decide()`: a DB outage must not become an SES send for a company that has connected M365. That means a DB outage on a **never-connected** install also holds — acceptable, since the dispatch itself needs the DB. If an existing email test breaks because it has no DB, monkeypatch `sending.decide` to return `Decision("ses", None, None)` in that test's fixture rather than weakening this.

In `email_dispatch_service.py`, immediately after `message_id = send_result.message_id`:

```python
            held_reason = getattr(send_result, "held_reason", None)
            if held_reason:
                dispatch_payload["held_reason"] = held_reason
                try:
                    from services.agent_actions import record_action
                    record_action(phase="dispatch", action_type="send_held", agent="EmailDispatch",
                                  status="error", summary=held_reason,
                                  details={"draft_id": draft_row[0] if draft_row else None})
                except Exception:  # audit never breaks dispatch
                    logger.debug("send_held audit failed", exc_info=True)
```

Then add a dispatch test in the same file: a held result leaves `sent=False` and `payload["held_reason"]` set (use the existing dispatch test fixtures in `tests/` — find with `grep -rl "send_draft" tests | head`).

- [ ] **Step 4: Prove the guard fails, then run all email tests**

Temporarily change `if decision.kind == "held": return …` to `pass`; run `test_held_send_never_touches_smtp` — it must FAIL. Restore. Then:
Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/ tests -k "email or dispatch" -q`
Expected: all pass (fix any fixture that now needs `decide` patched to `ses`, as noted).

- [ ] **Step 5: Commit**

```bash
git add src/services/mail_m365/sending.py src/services/email_service.py src/services/email_dispatch_service.py tests/services/mail_m365/test_sending.py
git commit -m "feat(m365): supplier email leaves from the company mailbox, or is held — never quietly from ours"
```

---

### Task 9: Receiving — attachments through mail-intake, replies through the watcher

**Files:**
- Create: `src/services/mail_m365/receiving.py`
- Modify: `src/services/mail_intake/poll.py` (`active_bindings`: accept `roles: Tuple[str, ...] = ("intake",)`; select `role = ANY(%s)`), `src/services/email_watcher.py` (`EmailWatcher._fetch_emails` ~line 1008: append Graph replies), `src/services/backend_scheduler.py` (register `mail-m365-poll`, modelled on the existing `mail-intake-poll` registration in the same file — copy how it builds the pipeline and connection)
- Test: `tests/services/mail_m365/test_receiving.py`

**Interfaces:**
- Consumes: mail-intake `RawMessage`, `poll_binding(conn, binding, source, pipeline)`, `active_bindings`; Task 4 `M365Client.list_inbox/get_mime`, `Throttled`; `email_watcher.parse_email_bytes(raw) -> EmailResponse`.
- Produces:
  - `GraphMailSource(client_for: Callable[[], M365Client])` with `iter_messages(binding: Mapping, since: datetime) -> Iterator[RawMessage]` — `source_ref = f"m365:{mailbox}"`, `provider_message_id = Graph id`
  - `fetch_replies(since: datetime, *, repo=None, client_for=None) -> List[EmailResponse]` — every active org `send_receive` binding; failures logged per binding, never raised
  - `poll_m365(conn) -> Dict[str,int]` — `poll_all`-like loop over `active_bindings(cur, provider="graph", roles=("send_receive",))`; on `Throttled` for a binding, that binding is skipped this pass, cursor unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/services/mail_m365/test_receiving.py
from datetime import datetime, timezone

from services.mail_m365 import receiving
from services.mail_m365.graph_client import Throttled

SINCE = datetime(2026, 10, 15, 8, tzinfo=timezone.utc)
MIME = (b"From: sup@supplier.com\r\nTo: procurement@cust.com\r\nSubject: Re: RFQ-1\r\n"
        b"Message-ID: <a@supplier.com>\r\nIn-Reply-To: <rfq1@cust.com>\r\n\r\nOur price is 455.")


class Client:
    def __init__(self, throttle=False):
        self.throttle = throttle
        self.fetched = []

    def list_inbox(self, mailbox, since):
        if self.throttle:
            raise Throttled(10)
        return [{"id": "m1", "receivedDateTime": "2026-10-15T09:00:00Z"}]

    def get_mime(self, mailbox, mid):
        self.fetched.append(mid)
        return MIME


def test_source_yields_raw_mime_with_graph_ids():
    src = receiving.GraphMailSource(lambda: Client())
    msgs = list(src.iter_messages({"mailbox_address": "procurement@cust.com"}, SINCE))
    assert msgs[0].raw == MIME and msgs[0].provider_message_id == "m1"
    assert msgs[0].source_ref == "m365:procurement@cust.com"
    assert msgs[0].received_at == datetime(2026, 10, 15, 9, tzinfo=timezone.utc)


def test_replies_are_parsed_for_the_watcher(monkeypatch):
    from services.style.mailbox import MailboxBinding
    b = MailboxBinding(1, "org", "graph", "procurement@cust.com", "send_receive", "arn:aws:secretsmanager:x",
                       None, SINCE, "ev", None, "OK", True)

    class Repo:
        def list_org(self): return [b]
    got = receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: Client())
    assert len(got) == 1 and "455" in (got[0].body or "")


def test_one_failing_mailbox_does_not_stop_replies(monkeypatch):
    from services.style.mailbox import MailboxBinding
    ok = MailboxBinding(1, "org", "graph", "a@cust.com", "send_receive", "arn:aws:secretsmanager:x",
                        None, SINCE, "ev", None, "OK", True)
    bad = MailboxBinding(2, "org", "graph", "b@cust.com", "send_receive", "arn:aws:secretsmanager:x",
                         None, SINCE, "ev", None, "OK", True)

    class Repo:
        def list_org(self): return [bad, ok]
    clients = {"b@cust.com": Client(throttle=True), "a@cust.com": Client()}

    class Router:
        def list_inbox(self, mailbox, since): return clients[mailbox].list_inbox(mailbox, since)
        def get_mime(self, mailbox, mid): return clients[mailbox].get_mime(mailbox, mid)
    assert len(receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: Router())) == 1


def test_throttled_binding_keeps_its_cursor(monkeypatch):          # Review Focus 5
    src = receiving.GraphMailSource(lambda: Client(throttle=True))
    calls = []

    class Pipeline:
        def ingest(self, cur, binding, message):
            calls.append(message)
    out = receiving.poll_one(None, {"binding_id": 1, "mailbox_address": "a@cust.com",
                                    "last_sync_at": SINCE}, src, Pipeline())
    assert out == {"throttled": 1} and calls == []
```

The `EmailResponse` attribute holding the text may not be `body` — open `email_watcher.py` around `parse_email_bytes` (line 249) and `EmailResponse` and use the real field name in the assertion before running.

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_receiving.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/services/mail_m365/receiving.py
"""Mail arriving in a company shared mailbox.

Two readers, each with its own cursor, both read-only: mail-intake takes every message
(attachments → quarantine/dedup → extraction) and the email watcher takes them as
possible supplier replies. Neither moves, flags or marks anything in the mailbox.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from services.mail_intake.sources.base import RawMessage
from services.mail_m365.graph_client import M365Client, Throttled

log = logging.getLogger(__name__)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class GraphMailSource:
    def __init__(self, client_for: Callable[[], M365Client]) -> None:
        self._client_for = client_for

    def iter_messages(self, binding: Mapping[str, Any], since: datetime) -> Iterator[RawMessage]:
        mailbox = binding["mailbox_address"]
        client = self._client_for()
        for item in client.list_inbox(mailbox, since):
            yield RawMessage(raw=client.get_mime(mailbox, item["id"]),
                             provider_message_id=item["id"],
                             received_at=_parse_dt(item.get("receivedDateTime")),
                             source_ref=f"m365:{mailbox}")


def _default_client_for() -> Callable[[], M365Client]:
    from services.db import get_conn
    from services.mail_m365.admin import _default_factory
    from services.mail_m365.config import load_settings, read_integration
    settings = load_settings()
    if settings is None:
        raise RuntimeError("Microsoft 365 is not configured on this server")
    with get_conn() as c:
        tenant = read_integration(c).get("tenant_id")
    client = _default_factory(settings)(tenant)
    return lambda: client


def fetch_replies(since: datetime, *, repo: Any = None,
                  client_for: Optional[Callable[[], Any]] = None) -> List[Any]:
    from services.email_watcher import parse_email_bytes
    from services.style.mailbox import MailboxBindingRepository
    repo = repo or MailboxBindingRepository()
    bindings = [b for b in repo.list_org() if b.can_send and b.is_active]
    if not bindings:
        return []
    client_for = client_for or _default_client_for()
    out: List[Any] = []
    for b in bindings:
        try:
            for raw in GraphMailSource(client_for).iter_messages(
                    {"mailbox_address": b.mailbox_address}, since):
                out.append(parse_email_bytes(raw.raw))
        except Exception:
            log.warning("m365 reply fetch failed for %s; other mailboxes continue",
                        b.mailbox_address, exc_info=True)
    return out


def poll_one(conn: Any, binding: Mapping[str, Any], source: GraphMailSource,
             pipeline: Any) -> Dict[str, int]:
    from services.mail_intake.poll import poll_binding
    try:
        return poll_binding(conn, binding, source, pipeline)
    except Throttled as exc:
        log.info("m365 poll throttled for %s; retry after %ss, cursor unchanged",
                 binding.get("mailbox_address"), exc.retry_after)
        if conn is not None:
            conn.rollback()
        return {"throttled": 1}


def poll_m365(conn: Any, pipeline: Any) -> Dict[str, int]:
    from services.mail_intake.poll import active_bindings
    with conn.cursor() as cur:
        bindings = active_bindings(cur, provider="graph", roles=("send_receive",))
    conn.commit()
    if not bindings:
        return {"bindings": 0}
    source = GraphMailSource(_default_client_for())
    total: Dict[str, int] = {"bindings": len(bindings)}
    for binding in bindings:
        for k, v in poll_one(conn, binding, source, pipeline).items():
            total[k] = total.get(k, 0) + v
    return total
```

`poll_binding` returns early (counts zero) when `last_sync_at` is NULL, so `poll_one` with a stub needs `last_sync_at` set — the test passes it. Because `poll_binding` sorts the iterator first, `Throttled` raised inside `list_inbox` surfaces before any `ingest` — exactly the property the test pins.

`mail_intake/poll.py` `active_bindings`:

```python
def active_bindings(cur: Any, provider: str = "relay",
                    roles: Tuple[str, ...] = ("intake",)) -> List[Dict[str, Any]]:
    cur.execute(
        """
        SELECT binding_id, user_ref, mailbox_address, folder_scope, last_sync_at
          FROM proc.bp_mailbox_binding
         WHERE role = ANY(%s) AND provider = %s AND is_active
         ORDER BY binding_id
        """,
        (list(roles), provider),
    )
```

(add `Tuple` to the typing import). Existing callers keep the default.

`email_watcher.py` `EmailWatcher._fetch_emails`: change the final returns so the Graph replies are appended. Replace the early `if fetcher is None: … return []` with collecting into `responses: List[EmailResponse] = []` from the existing branches, then:

```python
        try:
            from services.mail_m365.receiving import fetch_replies
            responses.extend(await asyncio.to_thread(fetch_replies, since))
        except Exception:
            logger.warning("Microsoft 365 reply fetch failed; IMAP results kept", exc_info=True)
        return responses
```

Keep the "Email fetcher not configured" warning only when **both** produced nothing and no M365 binding exists.

`backend_scheduler.py`: next to the `mail-intake-poll` registration (on this branch — find it with `grep -n "mail-intake-poll" src/services/backend_scheduler.py`), add `M365_POLL_JOB_NAME = "mail-m365-poll"`, registered only when `os.environ.get("M365_POLL", "on") != "off"` and `load_settings()` is not None, with `interval=timedelta(seconds=settings.poll_seconds)`, running `poll_m365(conn, pipeline)` on a **non-autocommit** connection built the same way the mail-intake job builds its own (it refuses autocommit), and the same pipeline object.

- [ ] **Step 4: Run tests**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_receiving.py tests/services/mail_intake tests -k "watcher or mail_intake" -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/mail_m365/receiving.py src/services/mail_intake/poll.py src/services/email_watcher.py src/services/backend_scheduler.py tests/services/mail_m365/test_receiving.py
git commit -m "feat(m365): mail in the shared mailbox is read for replies and attachments, and never changed"
```

---

### Task 10: Health — revoked access clears the default and alerts

**Files:**
- Modify: `src/services/style/health.py` (`_default_probe`: for `role == "send_receive"` use `M365Client.can_read`; on REVOKED call `repo.deactivate(..., reason="access withdrawn in Microsoft 365")` and record an alert)
- Test: `tests/services/mail_m365/test_health.py`

**Interfaces:**
- Consumes: `check_binding(binding, *, repo, probe)` as it exists in `health.py` — read it first; keep its signature.
- Produces: send_receive bindings get OK / DEGRADED / REVOKED with the same rules; REVOKED also clears `is_default` (via `deactivate`) so Task 8's `decide` then holds.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/mail_m365/test_health.py
from datetime import datetime, timezone

from services.style import health
from services.style.graph_source import MailboxAccessDenied
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


def test_revoked_send_receive_mailbox_is_deactivated():
    repo = Repo()
    def probe(b):
        raise MailboxAccessDenied("403")
    health.check_binding(B, repo=repo, probe=probe)
    assert repo.recorded == ["REVOKED"]
    assert repo.deactivated and "withdrawn" in repo.deactivated[0]
```

Adapt the call to the real `check_binding` signature you found; the assertions stay.

- [ ] **Step 2: Run** — `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365/test_health.py -v` → FAIL (no deactivate).

- [ ] **Step 3: Implement** — in `check_binding`, after recording REVOKED: `if binding.role == ROLE_SEND_RECEIVE: repo.deactivate(binding.binding_id, reason="access withdrawn in Microsoft 365")`. In `_default_probe`, for `send_receive` build the client via `services.mail_m365.admin._default_factory(load_settings())(tenant_id)` and call `can_read(binding.mailbox_address)`; `(False, …)` → raise `MailboxAccessDenied`.

- [ ] **Step 4: Run** the new test plus `tests/services/test_style_mailbox.py` → pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/style/health.py tests/services/mail_m365/test_health.py
git commit -m "feat(m365): a mailbox whose access is withdrawn stops being the sender"
```

---

### Task 11: UI — Integrations → Email panel on the real endpoints

**Files (repo `beyond_procwise_ui`, branch `spendiq-ui`):**
- Rewrite: `src/modules/MailConnection/mailboxService.js`
- Create: `src/modules/MailConnection/IntegrationsEmailPanel.jsx`, `src/modules/MailConnection/IntegrationsEmailPanel.render.test.jsx`
- Rewrite: `src/modules/MailConnection/mailbox.contract.test.js`
- Modify: `src/modules/ProcurementHome/HomeNavBar.jsx` (Setup group ~line 298: add "Email integration" item), `src/modules/ProcurementHome/index.jsx` (mount the panel like `BriefSettings` at ~line 3353; show the item only when `GET /admin/integrations/mail` did not return 403; on load, if URL has `mail=connected` or `mail_error=…`, open the panel and show the message), `src/lib/i18n/strings.email.js` (new `mail.admin.*` strings)

**Interfaces:**
- Consumes: Task 7 endpoints at `${AI_API}` (same base `BriefSettings` uses).
- Produces (`mailboxService.js` exports, all async, all throw `Error` whose `.code`/`.message` are the server's `detail.code`/`detail.message` when present, else the string `detail`):
  `getMailIntegration()`, `beginConnect()` → `{authorization_url}`, `getSetupScript(spObjectId?)` → string, `addMailbox(address, controlMailbox)`, `verifyMailbox(id, controlMailbox)`, `testSend(id)`, `makeDefault(id)`, `removeMailbox(id)`, `disconnect()`; plus the existing `subscribeMailbox`, `getMailboxSync` kept for the compose From row, now fed by `getMailIntegration()` (default mailbox → `identities[0]`, `can_send = !!default`).

- [ ] **Step 1: Write the failing contract test**

```js
// src/modules/MailConnection/mailbox.contract.test.js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import * as svc from './mailboxService';

vi.mock('axios');

describe('mailboxService talks to the real endpoints', () => {
  beforeEach(() => { vi.resetAllMocks(); });

  it('reads status from GET /admin/integrations/mail', async () => {
    axios.get.mockResolvedValue({ data: { connected: true, mailboxes: [
      { binding_id: 7, address: 'procurement@cust.com', is_default: true, active: true } ] } });
    const s = await svc.getMailIntegration();
    expect(axios.get.mock.calls[0][0]).toMatch(/\/admin\/integrations\/mail$/);
    expect(s.connected).toBe(true);
    expect(svc.getMailboxSync().identities[0].address).toBe('procurement@cust.com');
    expect(svc.getMailboxSync().can_send).toBe(true);
  });

  it('connect returns Microsoft\'s URL and holds no token', async () => {
    axios.post.mockResolvedValue({ data: { authorization_url: 'https://login.microsoftonline.com/x' } });
    const r = await svc.beginConnect();
    expect(r.authorization_url).toMatch(/^https:\/\/login\.microsoftonline\.com\//);
    expect(JSON.stringify(svc.getMailboxSync())).not.toMatch(/token|secret/i);
  });

  it('passes the server refusal through as given', async () => {
    axios.post.mockRejectedValue({ response: { status: 422, data: { detail: {
      code: 'over_scoped', message: 'We could also read me@cust.com, so access is not restricted' } } } });
    await expect(svc.addMailbox('procurement@cust.com', 'me@cust.com'))
      .rejects.toMatchObject({ code: 'over_scoped', message: expect.stringMatching(/not restricted/) });
  });

  it('no default mailbox means compose cannot send', async () => {
    axios.get.mockResolvedValue({ data: { connected: true, mailboxes: [] } });
    await svc.getMailIntegration();
    expect(svc.getMailboxSync().can_send).toBe(false);
  });
});
```

- [ ] **Step 2: Run** — `npx vitest run src/modules/MailConnection/mailbox.contract.test.js` → FAIL (functions missing).

- [ ] **Step 3: Implement `mailboxService.js`**

```js
/* The company's Microsoft 365 mailboxes, as the server reports them. Nothing here holds a
 * token: Microsoft's approval page returns to the SERVER, which keeps only the tenant id.
 * The admin decides; the server checks they may (403 otherwise) and proves access. */
import axios from 'axios';
import { AI_API } from '../ProcurementHome/useHomeData';

const BASE = '/admin/integrations/mail';
const url = (p = '') => `${AI_API}${BASE}${p}`;

const NOT_CONNECTED = {
  provider: null, provider_label: null, connected: false, can_send: false,
  account: null, identities: [], sent_items: false, threading: null,
  note: 'No company mailbox is connected, so nothing can be sent from here yet.',
};
let MAILBOX = { ...NOT_CONNECTED };
const listeners = new Set();
const clone = (v) => JSON.parse(JSON.stringify(v));
function publish() { listeners.forEach((fn) => fn(clone(MAILBOX))); }

export function subscribeMailbox(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function getMailboxSync() { return clone(MAILBOX); }

function fail(e) {
  const d = e?.response?.data?.detail;
  const err = new Error(typeof d === 'string' ? d : d?.message || 'The request failed.');
  err.code = typeof d === 'object' && d ? d.code : undefined;
  err.status = e?.response?.status;
  throw err;
}

function project(status) {
  const def = (status.mailboxes || []).find((m) => m.is_default && m.active);
  MAILBOX = status.connected ? {
    provider: 'microsoft', provider_label: 'Microsoft 365', connected: true,
    can_send: !!def, account: def ? def.address : null,
    identities: def ? [{ address: def.address, display_name: 'Company mailbox' }] : [],
    sent_items: true, threading: 'conversation',
    note: def ? `Goes out from ${def.address} through Microsoft 365.`
      : 'Microsoft 365 is connected but no default mailbox is chosen yet.',
  } : { ...NOT_CONNECTED };
  publish();
}

export async function getMailIntegration() {
  try { const { data } = await axios.get(url()); project(data); return data; } catch (e) { return fail(e); }
}
export async function beginConnect() {
  try { return (await axios.post(url('/m365/connect'))).data; } catch (e) { return fail(e); }
}
export async function getSetupScript(spObjectId) {
  try {
    const params = spObjectId ? { service_principal_object_id: spObjectId } : undefined;
    return (await axios.get(url('/m365/setup-script'), { params, responseType: 'text' })).data;
  } catch (e) { return fail(e); }
}
async function act(fn) { try { const r = (await fn()).data; await getMailIntegration(); return r; } catch (e) { return fail(e); } }
export const addMailbox = (address, controlMailbox) =>
  act(() => axios.post(url('/mailboxes'), { address, control_mailbox: controlMailbox }));
export const verifyMailbox = (id, controlMailbox) =>
  act(() => axios.post(url(`/mailboxes/${id}/verify`), { control_mailbox: controlMailbox }));
export const testSend = (id) => act(() => axios.post(url(`/mailboxes/${id}/test-send`)));
export const makeDefault = (id) => act(() => axios.patch(url(`/mailboxes/${id}`), { is_default: true }));
export const removeMailbox = (id) => act(() => axios.delete(url(`/mailboxes/${id}`)));
export const disconnect = () => act(() => axios.delete(url()));
```

Then fix every importer of the removed exports (`PROVIDERS`, `capabilitiesFor`, `connectImap`, `requestSendScope`, `__setMailbox`, `getMailbox`): `grep -rn "mailboxService" src`. `MailConnectionModal.jsx` becomes a thin wrapper that renders `IntegrationsEmailPanel` (Google/IMAP options are not shown in v1; their strings stay in `strings.email.js`). `EmailComposeModal.jsx` keeps using `getMailboxSync`/`subscribeMailbox`.

- [ ] **Step 4: Write the panel and its render test**

`IntegrationsEmailPanel.jsx` — built only from `modalStyles.js` constants like `BriefSettings.jsx`. States, in order:
1. `configured === false` → notice "Microsoft 365 is not configured on this server."
2. not connected → explanation (one paragraph: what is asked of IT, that no mail access is granted on Microsoft's page) + **Connect Microsoft 365** (`window.location.assign(authorization_url)`).
3. connected → "Connected to tenant {tenant_id}, approved by {consented_by} on {date}"; collapsible **Restrict access** block showing `getSetupScript()` in a `<pre>` with a Copy button and an optional "Enterprise app Object ID" field that re-fetches the script; **Add shared mailbox** form (address, control mailbox defaulting to the signed-in user's email); mailbox list rows: address, health badge, Verified ✓/✗, **Send test email**, **Make default** (enabled only when `can_be_default`), **Remove**; footer **Disconnect** with a confirm step inside the panel (no `window.confirm`).
Every server refusal shows `err.message` verbatim under the action that caused it.

```jsx
// src/modules/MailConnection/IntegrationsEmailPanel.render.test.jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import * as svc from './mailboxService';
import IntegrationsEmailPanel from './IntegrationsEmailPanel';

vi.mock('./mailboxService');

describe('IntegrationsEmailPanel', () => {
  it('offers Connect when not connected', async () => {
    svc.getMailIntegration.mockResolvedValue({ configured: true, connected: false, mailboxes: [] });
    render(<IntegrationsEmailPanel onClose={() => {}} />);
    expect(await screen.findByRole('button', { name: /connect microsoft 365/i })).toBeTruthy();
  });

  it('says so when the server is not configured', async () => {
    svc.getMailIntegration.mockResolvedValue({ configured: false, connected: false, mailboxes: [] });
    render(<IntegrationsEmailPanel onClose={() => {}} />);
    expect(await screen.findByText(/not configured on this server/i)).toBeTruthy();
  });

  it('shows the server refusal under Add', async () => {
    svc.getMailIntegration.mockResolvedValue({ configured: true, connected: true, tenant_id: 't', mailboxes: [] });
    svc.getSetupScript.mockResolvedValue('Connect-ExchangeOnline');
    svc.addMailbox.mockRejectedValue(Object.assign(new Error('We could also read me@c.com, so access is not restricted'), { code: 'over_scoped' }));
    render(<IntegrationsEmailPanel onClose={() => {}} userEmail="me@c.com" />);
    fireEvent.change(await screen.findByLabelText(/shared mailbox address/i), { target: { value: 'p@c.com' } });
    fireEvent.click(screen.getByRole('button', { name: /add mailbox/i }));
    await waitFor(() => expect(screen.getByText(/not restricted/)).toBeTruthy());
  });

  it('Make default is disabled until a test email was sent', async () => {
    svc.getMailIntegration.mockResolvedValue({ configured: true, connected: true, tenant_id: 't', mailboxes: [
      { binding_id: 1, address: 'p@c.com', active: true, is_default: false, health_state: 'OK',
        verified_at: '2026-10-15', last_test_send_at: null, can_be_default: false } ] });
    svc.getSetupScript.mockResolvedValue('');
    render(<IntegrationsEmailPanel onClose={() => {}} />);
    expect((await screen.findByRole('button', { name: /make default/i })).disabled).toBe(true);
  });
});
```

Run: `npx vitest run src/modules/MailConnection/` → write the panel until all pass. Add the Setup entry + URL-param handling in `ProcurementHome`, then run `npx vitest run src/modules/ProcurementHome/` and `npx vitest run src/lib/i18n/` → pass.

- [ ] **Step 5: Commit (UI repo — check `git status` first; another session has uncommitted edits in `strings.email.js`, so stage only your hunks with `git add -p` there)**

```bash
git add src/modules/MailConnection/ src/modules/ProcurementHome/HomeNavBar.jsx src/modules/ProcurementHome/index.jsx
git add -p src/lib/i18n/strings.email.js
git commit -m "feat(mail): admins connect Microsoft 365 and choose shared mailboxes from Setup"
```

---

### Task 12: Setup guide, local proof, and the live-tenant runbook

**Files:**
- Create: `docs/m365-setup.md` (commit with `git add -f`)
- Modify: `.env.example` (add `M365_CLIENT_ID=`, `M365_CLIENT_SECRET_ARN=`, `PUBLIC_BASE_URL=`, `MAIL_STATE_SIGNING_KEY_ARN=`, `M365_POLL_SECONDS=120`, `M365_POLL=on`, each with a one-line comment)

- [ ] **Step 1: Write `docs/m365-setup.md`** with these sections, concrete:
  1. **Register "Procwise Mail" (once, by Procwise):** Entra admin centre → App registrations → New → "Accounts in any organizational directory (multitenant)" → Redirect URI (Web) = `PUBLIC_BASE_URL/admin/integrations/mail/m365/callback` → **API permissions: remove the default `User.Read`, add none** → Certificates & secrets → new client secret (24 months) → store `{"client_id": "...", "client_secret": "..."}` in Secrets Manager → put the ARN in `M365_CLIENT_SECRET_ARN`. Create a second secret of 48 random bytes for `MAIL_STATE_SIGNING_KEY_ARN`.
  2. **Customer admin steps:** what they will see on Microsoft's page; creating the "Procwise Mailboxes" mail-enabled security group; running the script from the panel; the 30-minute propagation.
  3. **Live-tenant runbook (the release gate):** with a Microsoft 365 developer tenant — (a) connect; (b) confirm Entra shows Procwise Mail with **no** mail permissions; (c) run the script, `Test-ServicePrincipalAuthorization` says InScope for the shared mailbox only; (d) add the shared mailbox with the admin's own mailbox as control → verified; (e) **negative:** temporarily grant `Mail.Read` application permission in Entra and re-verify → must be refused `over_scoped` (proves §2.2 additivity and the probe); revoke it again; (f) test send → in Sent Items; (g) make default; send a real RFQ draft → arrives from the shared mailbox; record whether `Message-ID` was preserved; (h) reply from a supplier test account with a PDF → watcher matches, mail-intake reads the PDF; (i) remove Procwise Mail in Entra → within one health interval the mailbox is REVOKED and a queued draft is held, not sent via SES. Record each result with timestamps.
  4. **Rotation:** client secret rotation steps and that a certificate credential is the follow-up.

- [ ] **Step 2: Local proof against bp_testdb (no tenant needed)** — start the server locally (see `procwise.service` for the command; restart only with Nick's OK if it is the shared running instance), then as an admin token: `GET /admin/integrations/mail` → `configured:false` when env unset; with dummy env values set and an `https://` base URL: `POST …/m365/connect` returns a `login.microsoftonline.com/organizations/v2.0/adminconsent?...` URL whose `redirect_uri` is right; a Buyer token gets 403 on every route; callback with a forged `state` redirects with `mail_error=bad_signature`. Paste the responses into the PR description.

- [ ] **Step 3: Full test run**

Run: `PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/mail_m365 tests/api/test_mail_integration_router.py tests/sql/test_m365_mailbox_sql.py tests/services/test_style_mailbox.py -q` and the UI `npx vitest run src/modules/MailConnection src/modules/ProcurementHome`.
Expected: all pass. Report counts exactly.

- [ ] **Step 4: Commit**

```bash
git add -f docs/m365-setup.md
git add .env.example
git commit -m "docs(m365): registering Procwise Mail, the customer's steps, and the live release gate"
```

- [ ] **Step 5: Stop and report to Nick** — what is proven locally, what waits on the https address / Entra access / test tenant / `mail-intake-v2` merge, and that nothing is merged, pushed or applied to bp_sqldb.

---

## Self-review notes

- Spec §2.1–2.3 → Tasks 3, 4, 5, 6; §3 → Tasks 6, 7, 11; §4.2 → Tasks 1–3; §4.3 → Task 7; §4.4 → Task 11; §4.5 → Tasks 3, 12; §5.1 → Task 8; §5.2 → Task 9; §5.3 → Task 10; §6 rows: declined consent / bad state (5, 7), non-admin (7), over-scoped (6), propagation (6 message), token failure (8 `held`, 10), 429 (4, 9), size limit → surfaces as Graph error → held (8), disconnect with queue (8); §7 → Tasks 4 (allowlist), 5, 7; §8 → every task + 12; §9 → 12.
- Spec §5.1 "replies use the Graph reply path": deliberately simplified to MIME send carrying our `In-Reply-To`/`References` headers (the dispatch already sets thread headers). Outlook threads on those headers; the live runbook step (g)/(h) checks it. If threading fails live, add `/messages/{id}/reply` to the allowlist then.
- Spec §6 size limit: no pre-check; Graph's refusal becomes a held draft with Microsoft's message. Acceptable for v1.
