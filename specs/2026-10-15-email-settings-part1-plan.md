# Email Settings Part 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each customer's use of its company mailbox is governed by a versioned email policy set in Settings; the Email page shows every restriction and how it is guaranteed; a read-only Security reviewer sees the same with evidence; emails inside the company leave the shared mailbox only after the author approves them; Microsoft 365 signs in with a certificate and only a Microsoft administrator can connect.

**Architecture:** One `email_policy` row in `proc.bp_policy` (read and saved by `services/mail_policy.py`, optimistic-locked like `daily_brief.save_settings`, history kept in the row). A thin provider interface (`services/mail_providers/`) wraps the existing `mail_m365` package. Send-time rules (`services/mail_rules.py`) run in `EmailDispatchService.send_draft` for supplier email and in a new internal-email outbox (`proc.bp_internal_email`, `services/internal_email.py`) for forwards, internal replies and product notices. A restrictions model (`services/mail_restrictions.py`) feeds the screen, the Security view and the evidence PDF (`report_export.render_pdf`).

**Tech Stack:** Python 3 / FastAPI / psycopg2 / PyJWT 2.7 + cryptography 41 / WeasyPrint / pytest (BP_Backend); TypeScript gateway (role map); React + vitest (UI); Microsoft Graph v1.0.

**Spec:** `specs/2026-10-15-email-settings-part1-design.md` — read it before starting. It builds on `specs/2026-10-15-m365-shared-mailbox-design.md` (built on branches `m365-mailbox` / `m365-mailbox-ui`, not merged).

## Global Constraints

- Backend branch `email-settings-p1` cut from **`m365-mailbox`** (head `1c729767`) in worktree `/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/email-settings-p1`; UI branch `email-settings-p1-ui` cut from **`m365-mailbox-ui`** (head `1635efb`) in worktree `/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/email-settings-p1`; gateway changes on a branch `email-settings-p1-gw` in a worktree of `/home/muthu/PycharmProjects/beyond-procwaise-Api`. Never touch the shared main checkouts. Never push to `main`; nothing is pushed or merged without Nick.
- Never bare `git stash`; commit only your own paths.
- Tests: `CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest …` from the worktree root (`venv` symlinked, `.env` copied — both untracked). `.env` → **bp_testdb**. Migrations: bp_testdb only; bp_sqldb only with Nick's OK.
- New tables/indexes `bp_` / `ix_bp_<table>_<cols>`.
- **A missing policy value raises** (`PolicyUnavailable`); no code defaults. Defaults live only in the seed migration.
- Every policy save writes a history entry `{version, changed_by, changed_at, field, before, after}`; a save that cannot write it does not apply.
- Policy writes: action `mailbox.bind` (configure). Policy/restriction/evidence reads: new action `mailbox.view` (class `inspect`), allowed to Admin and Security reviewer only.
- Alerts about the email connection itself always come from the Procwise address.
- Plain English for every user-facing string; UI strings through `tOr`; reuse `modalStyles.js` look; the server decides permissions (UI shows controls only on `canConfigure === true`).
- No customer password stored (part 1); our certificate private key lives only in Secrets Manager, referenced by ARN.
- Graph paths reachable stay: mailFolders/messages list, message get (incl. `$select`), `$value`, `sendMail`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Policy changed between draft and send** (admin switches `internal_send` off after a forward was approved) → the send is held under the new policy (pinned in Task 6).
2. **Approval by someone other than the author** (an Admin or another buyer calls approve) → refused; only the author's identity from the token may approve (pinned in Task 6).
3. **Two admins save the policy at once** → the second gets a conflict (409) and nothing is overwritten; history has one entry per applied save (pinned in Task 3).
4. **Recipient address with capitals/spaces or a display name** (`" Finance <FINANCE@YourCompany.com>"`) → classified by the lower-cased bare address; `company_domains` compare lower-cased (pinned in Task 5).
5. **Security reviewer calls a write route directly** (bypassing the UI) → 403 on every write route, and the refusal is audited (pinned in Task 8).

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/migrations/2026-10-16-email-settings.sql` (+ `_rollback.sql`) | email_policy seed, `proc.bp_internal_email`, Security reviewer role + `inspect` class + group mapping |
| `tests/sql/test_email_settings_sql.py` | migration test |
| `src/services/actions.py` (modify) | `mailbox.view`, `mail.internal.create`, `mail.internal.approve` |
| `src/services/mail_providers/__init__.py`, `base.py`, `m365.py` | provider interface + Microsoft adapter |
| `src/services/mail_policy.py` | read / validate / save the email policy with history |
| `src/services/mail_rules.py` | recipient classification + send-time verdicts |
| `src/services/supplier_contact.py` (modify) | `is_known_supplier_email` |
| `src/services/internal_email.py` | internal-email outbox: create, approve, decline, send |
| `src/services/mail_notices.py` | product notices: mailbox (with approval) or Procwise address |
| `src/services/mail_restrictions.py` | restrictions model + evidence HTML |
| `src/services/mail_m365/*` (modify) | certificate login, `wids` admin check, read-scope listing, script uses policy |
| `src/services/email_dispatch_service.py`, `mail_m365/sending.py`, `mail_m365/receiving.py`, `value_digest.py`, `value_query_service.py` (modify) | wiring |
| `src/api/routers/mail_integration.py` (modify), `src/api/routers/internal_email.py` (new), `src/api/main.py` | endpoints |
| gateway `src/modules/user/user-roles.ts` | Security reviewer group ↔ role |
| UI `src/modules/MailConnection/*` | Email page (restrictions, policy, security view), internal approval |
| `docs/m365-setup.md` (`git add -f`), `.env.example` | certificate steps, runbook additions |

---

### Task 1: Migration — policy seed, internal-email outbox, Security reviewer role

**Files:**
- Create: `scripts/migrations/2026-10-16-email-settings.sql`, `scripts/migrations/2026-10-16-email-settings_rollback.sql`
- Modify: `src/services/actions.py` (add three actions)
- Test: `tests/sql/test_email_settings_sql.py`

**Interfaces:**
- Consumes: `proc.bp_policy` (columns `policy_id, policy_name, policy_type, policy_desc, policy_details JSONB, policy_status, version, created_by, created_date, last_modified_by, last_modified_date`; active unique on `(policy_type, policy_name) WHERE policy_status=1`); `role_definition` and `role_assignment` policy rows seeded by `deploy/sql/2026-08-06_guardrail_enforcement.sql` and extended by `deploy/sql/2026-09-18_admin_area_role_mapping.sql` — **read both first** and follow their exact JSON shape.
- Produces:
  - `proc.bp_policy` row: `policy_type='setting'`, `policy_name='email_policy'`, `policy_details = {"policy_identifier":"email_policy","rules":{…defaults…},"history":[]}`, `version=1`.
  - Table `proc.bp_internal_email(internal_email_id BIGSERIAL PK, author TEXT NOT NULL, kind TEXT NOT NULL CHECK (kind IN ('forward','reply','notice')), mailbox_address TEXT, recipients JSONB NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, attachments JSONB NOT NULL DEFAULT '[]', source_ref TEXT, status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','sent','held','declined')), held_reason TEXT, content_hash TEXT NOT NULL, approved_by TEXT, approved_at TIMESTAMPTZ, sent_at TIMESTAMPTZ, message_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now())`; index `ix_bp_internal_email_author_status (lower(author), status)`.
  - `role_definition` gains role `SecurityReviewer` (rank 1, allow `["read","inspect"]`); `Admin.allow` gains `inspect`.
  - `role_assignment.group_to_role` gains `"PROCWISE_SECURITY_REVIEWER": "SecurityReviewer"`.
  - `actions.py`: `"mailbox.view": "inspect"`, `"mail.internal.create": "write"`, `"mail.internal.approve": "write"`.

Seeded rules (exact):

```json
{"internal_send":"with_user_approval","company_domains":[],"external_recipients":"known_suppliers_only",
 "read_scope":"replies_and_suppliers","on_disconnect":"hold","access_level":"read_send",
 "notices_from_mailbox":"off","poll_seconds":120,"restriction_group":"Procwise Mailboxes"}
```

`company_domains` seeds empty: the admin must fill it (Task 3 validation refuses an empty list once `internal_send` is not `off`; the seed is valid because Task 2 treats an empty list as "no internal domains" — every internal-looking address is then "other" and held under `known_suppliers_only`, the safe direction).

- [ ] **Step 1: Write the failing test** `tests/sql/test_email_settings_sql.py`:

```python
import json, os, pytest
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB")
from services.db import get_conn

def _one(sql, *a):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(sql, a); return cur.fetchone()

def test_email_policy_seeded_once_with_safe_defaults():
    row = _one("SELECT policy_details, version FROM proc.bp_policy WHERE policy_name='email_policy' AND policy_status=1")
    d = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    assert d["policy_identifier"] == "email_policy"
    assert d["rules"]["internal_send"] == "with_user_approval"
    assert d["rules"]["on_disconnect"] == "hold"
    assert d["rules"]["read_scope"] == "replies_and_suppliers"
    assert d["history"] == [] or isinstance(d["history"], list)

def test_internal_email_status_check():
    with get_conn() as c:
        c.autocommit = False
        with c.cursor() as cur, pytest.raises(Exception):
            cur.execute("INSERT INTO proc.bp_internal_email(author,kind,recipients,subject,body,content_hash,status)"
                        " VALUES ('a@x','forward','[]','s','b','h','bogus')")
        c.rollback()

def test_security_reviewer_role_defined_and_mapped():
    rd = _one("SELECT policy_details FROM proc.bp_policy WHERE policy_details->>'policy_identifier'='role_definition' AND policy_status=1")[0]
    ra = _one("SELECT policy_details FROM proc.bp_policy WHERE policy_details->>'policy_identifier'='role_assignment' AND policy_status=1")[0]
    rd = rd if isinstance(rd, dict) else json.loads(rd); ra = ra if isinstance(ra, dict) else json.loads(ra)
    roles = {r["role"] if "role" in r else r["name"]: r for r in rd["rules"]["roles"]} if isinstance(rd["rules"].get("roles"), list) else rd["rules"]["roles"]
    assert "inspect" in roles["SecurityReviewer"]["allow"] and "configure" not in roles["SecurityReviewer"]["allow"]
    assert "inspect" in roles["Admin"]["allow"]
    assert ra["rules"]["group_to_role"]["PROCWISE_SECURITY_REVIEWER"] == "SecurityReviewer"
```

(Adjust only the `roles` lookup line to the real `role_definition` JSON shape you read in the 2026-08-06 migration; keep the three assertions.)

- [ ] **Step 2: Run, expect FAIL** (row missing): `… pytest tests/sql/test_email_settings_sql.py -v`
- [ ] **Step 3: Write the migration** — wrapped in `BEGIN; … COMMIT;`, idempotent:
  - email_policy: `INSERT … SELECT … WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name='email_policy' AND policy_status=1)` with the seeded rules above, `created_by='migration:2026-10-16-email-settings'`.
  - `CREATE TABLE IF NOT EXISTS proc.bp_internal_email (…)` exactly as in Produces; `CREATE INDEX IF NOT EXISTS ix_bp_internal_email_author_status …`.
  - role_definition / role_assignment: `UPDATE proc.bp_policy SET policy_details = jsonb_set(…), version = version + 1, last_modified_by='migration:2026-10-16-email-settings', last_modified_date=now() WHERE policy_details->>'policy_identifier'=… AND policy_status=1 AND NOT (<already contains>)` — copy the jsonb_set style of `2026-09-18_admin_area_role_mapping.sql`. Rollback removes exactly these additions and drops the table (comment: "drops any pending internal emails").
  - `actions.py`: add the three entries next to `"mailbox.bind": "configure"`.
- [ ] **Step 4: Apply to bp_testdb, run tests → PASS; apply twice → still PASS; rollback → FAIL; re-apply → PASS.** Then run `tests/guardrails/test_rbac.py` and confirm unchanged results (record any failure that also fails on the base commit).
- [ ] **Step 5: Commit** `git add scripts/migrations/2026-10-16-email-settings*.sql tests/sql/test_email_settings_sql.py src/services/actions.py && git commit -m "feat(mail): email policy, internal-email outbox and a read-only security reviewer role"`

---

### Task 2: Email policy — read, validate, save with history

**Files:**
- Create: `src/services/mail_policy.py`
- Test: `tests/services/test_mail_policy.py`

**Interfaces:**
- Consumes: the Task 1 row; `services.db.get_conn`; `services.agent_actions.record_action` (read its signature first).
- Produces:
  - `class PolicyUnavailable(RuntimeError)`, `class PolicyConflict(RuntimeError)`
  - `@dataclass(frozen=True) class EmailPolicy: version: int; internal_send: str; company_domains: Tuple[str,...]; external_recipients: str; read_scope: str; on_disconnect: str; access_level: str; notices_from_mailbox: str; poll_seconds: int; restriction_group: str; history: Tuple[dict,...]`
  - `FIELDS: Dict[str, Callable[[Any], Any]]` — validators/normalisers (below)
  - `load(conn=None) -> EmailPolicy` — raises `PolicyUnavailable` when the row or any field is missing/invalid
  - `save(changes: Dict[str, Any], *, by: str, expected_version: int, conn=None) -> EmailPolicy` — validates the merged result, writes `rules`, appends one history entry per changed field, bumps `version` with `WHERE … AND version = expected_version` (rowcount 0 → `PolicyConflict`), then `record_action(phase="configure", action_type="email_policy_saved", …)` with the diff; returns the new policy.

Validators (exact):

```python
_RE_GROUP = re.compile(r"[A-Za-z0-9 ._-]{1,64}")
_RE_DOMAIN = re.compile(r"(?=.{1,253}\Z)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")

def _choice(*allowed):
    def v(x):
        if x not in allowed: raise ValueError(f"must be one of {', '.join(allowed)}")
        return x
    return v

def _domains(x):
    if not isinstance(x, list): raise ValueError("must be a list of domains")
    out = []
    for d in x:
        d = str(d).strip().lower().lstrip("@")
        if not _RE_DOMAIN.fullmatch(d): raise ValueError(f"{d!r} is not a domain")
        if d not in out: out.append(d)
    return tuple(out)

def _poll(x):
    x = int(x)
    if not 60 <= x <= 3600: raise ValueError("must be between 60 and 3600 seconds")
    return x

def _group(x):
    x = str(x)
    if not _RE_GROUP.fullmatch(x): raise ValueError("letters, numbers, spaces, . _ - only (max 64)")
    return x

FIELDS = {
    "internal_send": _choice("off", "with_user_approval"),
    "company_domains": _domains,
    "external_recipients": _choice("known_suppliers_only", "any"),
    "read_scope": _choice("replies_and_suppliers", "whole_inbox"),
    "on_disconnect": _choice("hold", "procwise_address"),
    "access_level": _choice("read_send", "read_send_change"),
    "notices_from_mailbox": _choice("off", "with_user_approval"),
    "poll_seconds": _poll,
    "restriction_group": _group,
}
```

Cross-field rule in `save` only (not `load`): if the merged `internal_send` or `notices_from_mailbox` is not `off`, `company_domains` must be non-empty → `ValueError("Add at least one company domain before allowing emails to colleagues")`. Unknown keys in `changes` → `ValueError("unknown setting: <k>")`.

- [ ] **Step 1: Failing tests** (fake connection recording SQL, plus one live round-trip that restores the row):

```python
from services import mail_policy as mp

def test_load_raises_when_a_field_is_missing(fake_conn_with_rules):
    conn = fake_conn_with_rules({"internal_send": "off"})  # others missing
    with pytest.raises(mp.PolicyUnavailable):
        mp.load(conn)

def test_domains_are_lower_cased_trimmed_and_deduplicated():
    assert mp.FIELDS["company_domains"]([" YourCompany.com", "@yourcompany.com"]) == ("yourcompany.com",)

def test_save_refuses_internal_send_without_domains(live_policy):
    with pytest.raises(ValueError, match="company domain"):
        mp.save({"company_domains": []}, by="admin@x", expected_version=live_policy.version)

def test_save_writes_one_history_entry_per_changed_field(live_policy):
    p = mp.save({"company_domains": ["yourcompany.com"], "read_scope": "whole_inbox"},
                by="admin@x", expected_version=live_policy.version)
    new = [h for h in p.history if h["version"] == p.version]
    assert {h["field"] for h in new} == {"company_domains", "read_scope"}
    assert all(h["changed_by"] == "admin@x" and "before" in h and "after" in h for h in new)

def test_concurrent_save_conflicts_and_writes_nothing(live_policy):   # Review Focus 3
    mp.save({"poll_seconds": 300}, by="a@x", expected_version=live_policy.version)
    with pytest.raises(mp.PolicyConflict):
        mp.save({"poll_seconds": 600}, by="b@x", expected_version=live_policy.version)
    assert mp.load().poll_seconds == 300
```

`live_policy` fixture: reads the row (policy_details + version), yields `mp.load()`, and in `finally` restores `policy_details`, `version`, `last_modified_by`, `last_modified_date` exactly. `fake_conn_with_rules` returns a minimal connection whose cursor returns `(policy_id, version, {"policy_identifier":"email_policy","rules":rules,"history":[]})`.

- [ ] **Step 2: Run → FAIL** (module missing).
- [ ] **Step 3: Implement** `load` (one SELECT `policy_id, version, policy_details` where `policy_name='email_policy' AND policy_status=1`; build `EmailPolicy` by running each validator over `rules[k]`; any `KeyError`/`ValueError` → `PolicyUnavailable(f"email policy: {k}: …")`) and `save` (in one transaction on a non-autocommit connection: SELECT … FOR UPDATE, merge, validate, compute changed fields, `UPDATE … SET policy_details = %s::jsonb, version = version+1, last_modified_by=%s, last_modified_date=now() WHERE policy_id=%s AND version=%s`; rowcount != 1 → rollback + `PolicyConflict`; `record_action_or_fail` with `{changes:[{field,before,after}]}` — if the audit write raises, rollback and re-raise so the save does not apply; commit). History entries are stored in `policy_details.history` as JSON (`changed_at` ISO UTC).
- [ ] **Step 4: Run → PASS.** Prove the guard: drop the `AND version=%s` clause, watch `test_concurrent_save_conflicts…` go red, restore.
- [ ] **Step 5: Commit** `feat(mail): the email policy is read strictly and every change keeps its history`

---

### Task 3: Provider interface + policy endpoints + script follows the policy

**Files:**
- Create: `src/services/mail_providers/__init__.py`, `src/services/mail_providers/base.py`, `src/services/mail_providers/m365.py`
- Modify: `src/api/routers/mail_integration.py` (policy routes; `mailbox.view` on reads), `src/services/mail_m365/setup_script.py` (`access_level`, group from policy)
- Test: `tests/services/test_mail_providers.py`, `tests/api/test_mail_policy_router.py`, `tests/services/mail_m365/test_setup_script.py` (extend)

**Interfaces:**
- Consumes: Task 2 `load`, `save`, `PolicyConflict`, `PolicyUnavailable`; existing `admin.status`, `M365Client`, `sending.decide`, `setup_script.render`.
- Produces:
  - `base.py`:
    ```python
    class MailProvider(Protocol):
        name: str                    # "m365"
        label: str                   # "Microsoft 365"
        method: str                  # plain-English connection method
        def size_limit(self) -> int: ...
        def status(self, conn) -> dict: ...
        def prove(self, conn) -> List[dict]: ...      # evidence rows: {check, answer, result, at}
        def send(self, binding, mime: bytes) -> None: ...
    ```
  - `mail_providers.get_provider(conn) -> Optional[MailProvider]` — returns the M365 provider when `mail_integration.provider == "m365"`, else `None`.
  - `m365.M365Provider` with `size_limit() = sending.MAX_GRAPH_MIME_BYTES` (3_000_000 — read the real constant name in `sending.py`), `status()` → `admin.status(...)`, `send()` → `sending.send_via_graph(...)`, `prove()` → Task 7 fills it (returns `[]` here).
  - Routes (prefix `/admin/integrations/mail`):
    - `GET /policy` → `{"policy": {...rules..., "version": n}, "history": [...], "canConfigure": bool}` — gate `mailbox.view`; `canConfigure` = the caller also passes `mailbox.bind` (check with `endpoint_gate.require` in a try; a refusal there must not 403 the read).
    - `PATCH /policy` body `{"changes": {...}, "version": n}` — gate `mailbox.bind`; 409 on `PolicyConflict`, 422 on `ValueError` (message as given), 503 on `PolicyUnavailable`; returns the new policy.
    - Existing `GET ""` (status) switches its gate from `mailbox.bind` to `mailbox.view` and keeps `canConfigure` computed the same way.
  - `setup_script.render(..., group=policy.restriction_group, access_level=policy.access_level)` — `read_send` grants `Application Mail.Read` + `Application Mail.Send`; `read_send_change` grants `Application Mail.ReadWrite` + `Application Mail.Send`. The setup-script route reads the policy and passes both.

- [ ] **Step 1: Failing tests**
  - `test_get_provider_none_when_never_connected`, `test_get_provider_m365_when_connected` (stub `read_integration`).
  - Router through `api.main.app` (copy the `real` fixture pattern from `tests/api/test_mail_integration_router.py`): GET policy as Admin → 200 with `canConfigure: true`; as SecurityReviewer (gate stub allowing `mailbox.view`, refusing `mailbox.bind`) → 200 with `canConfigure: false`; PATCH as SecurityReviewer → 403; PATCH stale version → 409; PATCH bad value → 422 with the validator's message in `detail`.
  - Script: `render(..., access_level="read_send")` contains `Mail.Read` and not `Mail.ReadWrite`; `read_send_change` contains `Mail.ReadWrite`; the group name from the argument appears.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** (route handlers follow the existing `gate(...)` + `_refused`/`JSONResponse` patterns in `mail_integration.py`; read them first).
- [ ] **Step 4: Run → PASS**, plus the whole existing `tests/api/test_mail_integration_router.py` (status gate change must keep its tests green after updating the expected action name).
- [ ] **Step 5: Commit** `feat(mail): a provider interface, the email policy on the Email page, and an IT script that follows it`

---

### Task 4: Microsoft 365 — certificate login and administrator-only connection

**Files:**
- Modify: `src/services/mail_m365/config.py` (`M365_CLIENT_CERT_ARN`), `src/services/mail_m365/graph_client.py` (`token()`), `src/services/mail_m365/consent.py` (code exchange + `wids` check), `src/services/style/graph_source.py` only if `GraphCredentials` must carry the key, `.env.example`
- Test: `tests/services/mail_m365/test_cert_login.py`, extend `tests/services/mail_m365/test_consent.py`

**Interfaces:**
- Consumes: `M365Client.token()` (posts `client_id, client_secret, scope, grant_type`), `consent.verify_id_token(id_token, *, client_id, expected_nonce, transport, now)` (claim checks end with tid/exp/nbf/nonce), `resolve_credentials(...)`.
- Produces:
  - Secret at `M365_CLIENT_CERT_ARN`: JSON `{"client_id": "...", "private_key_pem": "-----BEGIN PRIVATE KEY-----...", "thumbprint_sha1_b64url": "..."}`. `M365_CLIENT_SECRET_ARN` is removed from `load_settings` (settings unconfigured if the cert ARN is missing).
  - `graph_client.client_assertion(client_id: str, tenant_id: str, private_key_pem: str, x5t: str, now: float) -> str`:
    ```python
    def client_assertion(client_id, tenant_id, private_key_pem, x5t, now):
        claims = {"aud": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                  "iss": client_id, "sub": client_id, "jti": uuid.uuid4().hex,
                  "nbf": int(now) - 5, "exp": int(now) + 300}
        return jwt.encode(claims, private_key_pem, algorithm="RS256", headers={"x5t": x5t})
    ```
  - `token()` posts `client_id`, `client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer`, `client_assertion=<above>`, `scope`, `grant_type=client_credentials` — never `client_secret`. The consent code exchange uses the same assertion (with the `organizations`/tenant token URL it already uses).
  - `consent.ADMIN_ROLE_IDS = frozenset({"62e90394-69f5-4237-9190-012177145e10", "e8611ab8-c189-46e8-94e1-60213ab1f814", "158c047a-c907-4556-b7ef-446551a6b5f7", "9b895d92-2cd3-44c7-9d02-a6ac2d5ea5c3"})` (Global Administrator, Privileged Role Administrator, Cloud Application Administrator, Application Administrator).
  - In `verify_id_token`, after the tid/issuer check: `wids = claims.get("wids") or []`; if not `set(map(str.lower, wids)) & ADMIN_ROLE_IDS` → `ConsentNotGiven("not_admin", "Only a Microsoft 365 administrator can connect Procwise")`. Router: add `not_admin` to the fixed consent-code set; UI message (Task 10): "Only a Microsoft 365 administrator can connect. Ask your Microsoft administrator to click Connect."

- [ ] **Step 1: Failing tests**
  - Generate an RSA key in the test (`cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key`); `client_assertion` decodes with the public key, has `aud/iss/sub/exp`, header `x5t`.
  - `token()` with a recording transport posts `client_assertion` and **no** `client_secret` key.
  - `verify_id_token` (existing test helpers that sign id_tokens): token without `wids` → `not_admin`; with `wids=["<some other role guid>"]` → `not_admin`; with Global Administrator → claims returned.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** Load the cert secret through the same egress/Secrets path `resolve_credentials` uses; cache per process like the client cache.
- [ ] **Step 4: Run → PASS** + all of `tests/services/mail_m365` and `tests/api/test_mail_integration_router.py`. Prove the guard: remove the `wids` check → the two `not_admin` tests go red; restore.
- [ ] **Step 5: Commit** `feat(m365): sign in to Microsoft with a certificate; only a Microsoft administrator can connect`

---

### Task 5: Send-time rules for supplier email

**Files:**
- Create: `src/services/mail_rules.py`
- Modify: `src/services/supplier_contact.py` (add `is_known_supplier_email`), `src/services/email_dispatch_service.py` (call the rules before `send_email`), `src/services/mail_m365/sending.py` (`on_disconnect`)
- Test: `tests/services/test_mail_rules.py`, extend `tests/test_email_dispatch_service.py`, extend `tests/services/mail_m365/test_sending.py`

**Interfaces:**
- Consumes: Task 2 `EmailPolicy`, `load`, `PolicyUnavailable`; `proc.bp_supplier(contact_email_1, contact_email_2)`; Task 3 `get_provider`; dispatch at `email_dispatch_service.py` where `send_result = self.email_service.send_email(..., company_mailbox=True)` is called and `held_reason` recorded (`dispatch_payload["held_reason"]` + `record_action(action_type="send_held")`).
- Produces:
  - `supplier_contact.is_known_supplier_email(conn, address: str) -> bool` — `SELECT 1 FROM proc.bp_supplier WHERE lower(trim(contact_email_1)) = %s OR lower(trim(contact_email_2)) = %s LIMIT 1` with the normalised address.
  - `mail_rules.bare_address(raw: str) -> str` — `email.utils.parseaddr`, then strip + lower.
  - `mail_rules.classify(address: str, policy, *, is_supplier: Callable[[str], bool]) -> Literal["internal","supplier","other"]` — domain (after `@`) in `policy.company_domains` → `internal`; else `is_supplier(address)` → `supplier`; else `other`.
  - `@dataclass(frozen=True) class Verdict: allowed: bool; reason: Optional[str]; needs_internal_approval: bool`
  - `mail_rules.check_supplier_email(recipients: List[str], size_bytes: int, policy, *, is_supplier, size_limit: int) -> Verdict`:
    - size over limit → `Verdict(False, f"Too large to send from your company mailbox (limit {size_limit // 1_000_000} MB)", False)`
    - any internal → `Verdict(False, "Emails to colleagues go through 'Send to a colleague' and need your approval", False)`
    - any other, `known_suppliers_only` → `Verdict(False, f"{addr} is not a known supplier", False)` (first offending address)
    - else `Verdict(True, None, False)`
  - Dispatch: when `sending.decide()` says the company mailbox will be used (`kind == "graph"`), load the policy and call `check_supplier_email` **before** `send_email`; a refusal records exactly like an existing hold (`held_reason`, `send_held` audit), no send. `PolicyUnavailable` → held `"Your email policy could not be read; nothing was sent"`. When `decide()` is `ses` (never connected), rules are not applied (spec §11).
  - `sending.decide`: when there is no usable default and `policy.on_disconnect == "procwise_address"` → `Decision("ses", None, None)` instead of held; when the policy cannot be read → held (as today). `hold` keeps today's behaviour.

- [ ] **Step 1: Failing tests**

```python
P = lambda **kw: SimpleNamespace(company_domains=("yourcompany.com",), external_recipients="known_suppliers_only", **kw)
sup = lambda a: a == "quotes@supplier.com"

def test_display_name_and_case_are_normalised():                      # Review Focus 4
    assert mail_rules.bare_address(' Finance <FINANCE@YourCompany.com> ') == "finance@yourcompany.com"
    assert mail_rules.classify("finance@yourcompany.com", P(), is_supplier=sup) == "internal"

def test_supplier_email_to_a_colleague_is_held():
    v = mail_rules.check_supplier_email(["quotes@supplier.com", "finance@yourcompany.com"], 10, P(), is_supplier=sup, size_limit=3_000_000)
    assert not v.allowed and "colleagues" in v.reason

def test_unknown_external_is_held_when_suppliers_only():
    v = mail_rules.check_supplier_email(["someone@gmail.com"], 10, P(), is_supplier=sup, size_limit=3_000_000)
    assert not v.allowed and v.reason == "someone@gmail.com is not a known supplier"

def test_any_external_allowed_when_policy_says_any():
    v = mail_rules.check_supplier_email(["someone@gmail.com"], 10, P(external_recipients="any"), is_supplier=sup, size_limit=3_000_000)
    assert v.allowed

def test_oversize_is_held_with_the_limit():
    v = mail_rules.check_supplier_email(["quotes@supplier.com"], 3_000_001, P(), is_supplier=sup, size_limit=3_000_000)
    assert not v.allowed and "3 MB" in v.reason
```

  Dispatch tests (extend the existing fakes): graph decision + unknown recipient → `sent` false, `held_reason` stored, `send_held` audit, `send_email` never called; ses decision → rules not called; policy unreadable under graph → held with the policy reason. `sending` tests: no default + `on_disconnect="procwise_address"` → `ses`; `hold` → held (existing test still passes).

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** Size: the dispatch already builds the MIME inside `send_email`; pass `size_bytes` as the sum of body + attachment byte lengths × 4/3 + 10_000 header allowance (Graph limit is on the base64 MIME) — the existing in-`sending` exact pre-check stays as the final guard.
- [ ] **Step 4: Run → PASS.** Prove the guards: make `check_supplier_email` always allow → the held tests go red; make `on_disconnect` ignored → its test goes red; restore.
- [ ] **Step 5: Commit** `feat(mail): supplier email follows the email policy at the moment it is sent`

---

### Task 6: Internal-email outbox — create, approve (author only), send

**Files:**
- Create: `src/services/internal_email.py`, `src/api/routers/internal_email.py`
- Modify: `src/api/main.py` (register router in `_AUTHENTICATED_ROUTERS`)
- Test: `tests/services/test_internal_email.py`, `tests/api/test_internal_email_router.py`

**Interfaces:**
- Consumes: Task 1 table; Task 2 `load`; Task 5 `classify`, `bare_address`; Task 3 `get_provider`; `approval_content.content_hash(draft) -> str` (hashes recipients, subject, body, attachments); `MailboxBindingRepository.get_default_sender()`; `services.email_service.EmailService` MIME building (read how `send_email` builds the message and reuse that builder; do not duplicate it — extract a `build_mime(...)` helper if needed).
- Produces (`internal_email.py`):
  - `class InternalEmailRefused(Exception)` with `.code`, `.message`; codes `not_found`, `not_author`, `changed`, `policy_off`, `not_internal`, `no_mailbox`, `already_done`.
  - `create(conn, *, author: str, kind: str, recipients: List[str], subject: str, body: str, attachments: List[dict], source_ref: Optional[str]) -> dict` — normalises recipients with `bare_address`; refuses `not_internal` if any recipient does not classify `internal` (message: "{addr} is outside your company. Use the normal supplier email instead."); refuses `policy_off` when `internal_send == "off"` ("Emails to colleagues are switched off in your email policy"); stores `content_hash`; status `pending`; returns the row as a dict.
  - `list_pending(conn, *, author: str) -> List[dict]` — `lower(author)=lower(%s) AND status='pending'`.
  - `approve_and_send(conn, internal_email_id: int, *, approver: str, provider=None) -> dict` — in order: row exists (`not_found`); `status == 'pending'` (`already_done`); `lower(approver) == lower(author)` (`not_author`, message "Only the person who wrote this email can approve it"); recompute `content_hash` of the stored content equals stored hash (`changed`); **re-load the policy now** — `internal_send` must still be `with_user_approval` (`policy_off`) and every recipient must still classify `internal` (`not_internal`); default sender mailbox exists and provider connected (`no_mailbox` → row `held` with reason); build MIME from the default mailbox address, `provider.send(binding, mime)`; on success `status='sent'`, `approved_by`, `approved_at`, `sent_at`, `message_id`; on send failure `status='held'`, `held_reason` plain; audit `record_action(action_type="internal_email_sent"|"internal_email_held")`.
  - `decline(conn, internal_email_id, *, actor: str)` — author only; `status='declined'`.
  - `edit(conn, internal_email_id, *, author, subject, body, recipients)` — author only, pending only; recomputes `content_hash` (so an approval is always of the content shown).
- Routes (`/mail/internal`, every route `require_user`; identity is the token principal's email — never a body field):
  - `POST ""` body `{kind, recipients, subject, body, source_ref?}` → gate `mail.internal.create`
  - `GET "/pending"` → the caller's pending list
  - `PATCH "/{id}"` → edit
  - `POST "/{id}/approve"` → gate `mail.internal.approve` then `approve_and_send(approver=<principal email>)`
  - `POST "/{id}/decline"`
  - Refusals return `JSONResponse(status, {"detail": {"code","message"}})`: `not_found` 404, `not_author` 403, `changed`/`already_done` 409, `policy_off`/`not_internal`/`no_mailbox` 422.

- [ ] **Step 1: Failing tests** (fake repo/provider; one live-DB test for SQL):

```python
def test_only_the_author_can_approve(svc):                              # Review Focus 2
    e = svc.create(author="buyer@yourcompany.com", kind="forward", recipients=["finance@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        svc.approve_and_send(e["internal_email_id"], approver="admin@yourcompany.com")
    assert r.value.code == "not_author"

def test_policy_switched_off_after_create_holds_the_send(svc, policy):  # Review Focus 1
    e = svc.create(author="b@yourcompany.com", kind="forward", recipients=["f@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    policy.internal_send = "off"
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert r.value.code == "policy_off" and svc.provider.sent == []

def test_external_recipient_refused_at_create(svc):
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        svc.create(author="b@yourcompany.com", kind="forward", recipients=["x@gmail.com"], subject="s", body="b", attachments=[], source_ref=None)
    assert r.value.code == "not_internal"

def test_edit_changes_the_hash_and_approval_covers_the_new_content(svc):
    e = svc.create(author="b@yourcompany.com", kind="reply", recipients=["f@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    e2 = svc.edit(e["internal_email_id"], author="b@yourcompany.com", subject="s2", body="b2", recipients=["f@yourcompany.com"])
    assert e2["content_hash"] != e["content_hash"]
    svc.approve_and_send(e["internal_email_id"], approver="B@YourCompany.com")
    assert svc.provider.sent[0]["subject"] == "s2"

def test_approved_email_goes_from_the_default_shared_mailbox(svc):
    e = svc.create(author="b@yourcompany.com", kind="forward", recipients=["f@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert svc.provider.sent[0]["from"] == "procurement@yourcompany.com"
```

  Router (through `api.main.app`): approve by a different principal → 403 `not_author`; the body cannot set `author` (a body `author` field is ignored — the stored author is the token's email); unauthenticated → 401.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run → PASS.** Prove the guards: remove the author check → `test_only_the_author_can_approve` red; remove the policy re-load → `test_policy_switched_off…` red; restore.
- [ ] **Step 5: Commit** `feat(mail): emails to colleagues leave the shared mailbox only after their author approves`

---

### Task 7: Product notices from the shared mailbox (when the policy allows)

**Files:**
- Create: `src/services/mail_notices.py`
- Modify: `src/services/value_digest.py` (send at ~line 264), `src/services/value_query_service.py` (send at ~line 286)
- Test: `tests/services/test_mail_notices.py`

**Interfaces:**
- Consumes: Task 6 `create`; Task 2 `load`; `EmailService.send_email` (unflagged = Procwise address).
- Produces: `mail_notices.notify(conn, *, for_user: str, recipients: List[str], subject: str, body: str, source_ref: str, email_service) -> dict` →
  - policy `notices_from_mailbox == "with_user_approval"`, Microsoft 365 connected with a default mailbox, and every recipient internal → `internal_email.create(author=for_user, kind="notice", …)`; returns `{"route": "pending_approval", "internal_email_id": …}`.
  - otherwise (policy `off`, not connected, policy unreadable, or any external recipient) → `email_service.send_email(...)` unflagged, exactly as today; returns `{"route": "procwise_address"}`.
  - `signal_alerts.py` and `support_agent.py` are **not** changed (connection alerts always come from the Procwise address).

- [ ] **Step 1: Failing tests:** policy off → SES call made, no internal email; policy on + connected + internal recipient → internal email `pending` for `for_user`, no SES call; policy on + not connected → SES; policy unreadable → SES (a notice is never lost because the policy is unreadable).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** and switch the two callers to `notify(...)`, passing the user the notice is for (the digest's recipient user / the query's asking user — read each caller to find that value).
- [ ] **Step 4: Run → PASS** + the existing value_digest / value_query tests.
- [ ] **Step 5: Commit** `feat(mail): product notices can come from the shared mailbox once the user approves`

---

### Task 8: Read scope — only replies and supplier mail are downloaded

**Files:**
- Modify: `src/services/mail_m365/graph_client.py` (`list_inbox` `$select`; new `get_headers`), `src/services/mail_m365/receiving.py` (`GraphMailSource.iter_messages`, `_replies_from`)
- Test: extend `tests/services/mail_m365/test_graph_client.py`, `tests/services/mail_m365/test_receiving.py`

**Interfaces:**
- Consumes: `list_inbox` (`$select=id,receivedDateTime`), `_url` allowlist (message GET path), `proc.email_thread_map(message_id PK)`, Task 2 `load`, Task 5 `is_known_supplier_email`, `bare_address`.
- Produces:
  - `list_inbox` selects `id,receivedDateTime,from` and each item carries `"from": "<bare address>"`.
  - `M365Client.get_headers(mailbox, message_id) -> Dict[str, str]` — `GET /users/{m}/messages/{id}?$select=internetMessageHeaders` through `_url` (same allowed path as message get); returns a lower-cased-name dict.
  - `receiving.in_scope(item, *, policy, client, mailbox, is_supplier, sent_ids: Callable[[List[str]], bool]) -> bool`:
    - `policy.read_scope == "whole_inbox"` → True
    - `is_supplier(item["from"])` → True
    - else headers = `client.get_headers(...)`; ids = message-ids in `in-reply-to` + `references`; `sent_ids(ids)` (one query: `SELECT 1 FROM proc.email_thread_map WHERE message_id = ANY(%s) LIMIT 1`) → True
    - else False — the message is **never** passed to `get_mime`.
  - Both `iter_messages` and `_replies_from` call `in_scope` before `get_mime`. Policy unreadable → treat as `replies_and_suppliers` (the narrower scope).

- [ ] **Step 1: Failing tests:** an inbox of 3 items (supplier, reply to our Message-ID, unrelated colleague) under `replies_and_suppliers` → `get_mime` called for exactly the first two; under `whole_inbox` → all three; policy unreadable → the narrower result; `get_headers` URL passes the allowlist and a hostile id is quoted.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run → PASS** + the whole `tests/services/mail_m365`. Prove the guard: make `in_scope` return True → the "unrelated colleague is never downloaded" test goes red; restore.
- [ ] **Step 5: Commit** `feat(m365): only replies and supplier mail in the shared inbox are ever downloaded`

---

### Task 9: Restrictions model, Security view data, evidence PDF

**Files:**
- Create: `src/services/mail_restrictions.py`
- Modify: `src/services/mail_providers/m365.py` (`prove()`), `src/api/routers/mail_integration.py` (`GET /restrictions`, `GET /evidence.pdf`)
- Test: `tests/services/test_mail_restrictions.py`, extend `tests/api/test_mail_policy_router.py`

**Interfaces:**
- Consumes: Task 3 provider; Task 2 policy; `admin.status(...)` mailbox views (`address, active, is_default, health_state, verified_at, last_test_send_at, withdrawn`); `M365Client.token()` (decode **our own** access token's `roles` claim with `jwt.decode(token, options={"verify_signature": False})` — it is only read to report what Microsoft granted us); `graph_client.ALLOWED_PATH_PATTERNS`; `report_export.render_pdf(html, css) -> bytes`; `record_action`.
- Produces:
  - `mail_restrictions.build(conn, *, provider, policy, now) -> List[dict]` — items `{id, title, explanation, guarantee: "provider"|"procwise"|"policy", status: "in_force"|"failed"|"not_proven", last_checked_at, evidence: [...]}` for the eight ids in spec §7, in this order: `no_mail_permissions, mailbox_scope, actions_limited, access_level, internal_approval, external_suppliers_only, read_scope, on_disconnect`. Titles and explanations exactly as in the mock-up's Admin artboard (copy the strings).
    - `no_mail_permissions`: token `roles` contains any `Mail.*` → `failed` (evidence answer: the roles found); empty → `in_force`; token failure → `not_proven`.
    - `mailbox_scope`: `in_force` when every active mailbox has `verified_at` within 2 h and `health_state == "OK"`; any REVOKED/withdrawn → `failed`; older than 2 h or DEGRADED → `not_proven`; evidence one row per mailbox (`verified_at`, state).
    - `actions_limited`: always `in_force` (`procwise`), evidence lists the allowed actions in plain words ("list inbox, open message, send").
    - policy items: `in_force` when the setting is on (`internal_send == "with_user_approval"`, `external_recipients == "known_suppliers_only"`, `read_scope == "replies_and_suppliers"`, `on_disconnect == "hold"`, `access_level == "read_send"`); otherwise status `in_force` with explanation saying the admin chose the wider option (e.g. "Your policy allows any outside address") — the screen shows the choice, not a failure.
    - Policy unreadable → every policy item `not_proven` with explanation "Email policy could not be read".
  - `mail_restrictions.evidence_html(restrictions, policy, mailboxes, history, *, generated_at, generated_by) -> str` — static HTML (no external resources: `render_pdf`'s fetcher refuses them), escaped with `html.escape` everywhere.
  - Routes: `GET /admin/integrations/mail/restrictions` → `{"restrictions": [...], "summary": {"in_force": n, "total": 8}, "canConfigure": bool}` (gate `mailbox.view`); `GET /admin/integrations/mail/evidence.pdf` → `application/pdf`, `Content-Disposition: attachment; filename="email-restrictions-<YYYY-MM-DD>.pdf"`, gate `mailbox.view`, plus `record_action(action_type="evidence_report_downloaded")`. Add the PDF route to the output-safety exemptions the same narrow way the connect route was (GET, exact path, 2xx only) if the scrubber touches binary bodies — check `main.py` first.
  - Security reviewer refused on writes (Review Focus 5): a test through `api.main.app` with a principal whose role is `SecurityReviewer` (use the real `endpoint_gate.require` with the role policy seeded in bp_testdb) calls **every** write route (`POST /m365/connect`, `POST /mailboxes`, `POST /mailboxes/{id}/verify`, `POST /mailboxes/{id}/test-send`, `PATCH /mailboxes/{id}`, `DELETE /mailboxes/{id}`, `DELETE ""`, `PATCH /policy`) → all 403, and an audit row exists for each refusal; and both read routes → 200.

- [ ] **Step 1: Failing tests** for each status rule above, the order of ids, HTML escaping of a hostile mailbox address (`<script>` never appears raw), PDF bytes start with `%PDF`, and the Security reviewer write-route sweep.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run → PASS.** Prove the guard: make `no_mail_permissions` ignore `roles` → the "Mail.Read granted → failed" test goes red; restore.
- [ ] **Step 5: Commit** `feat(mail): every restriction is shown with how it is guaranteed, and a dated evidence report`

---

### Task 10: Gateway role map + UI — Email page, Security view, internal approval

**Files:**
- Gateway (worktree on `email-settings-p1-gw`): `src/modules/user/user-roles.ts` (+ its test)
- UI (worktree `email-settings-p1-ui`):
  - Modify: `src/modules/MailConnection/mailboxService.js` (policy, restrictions, evidence, internal-email calls), `src/modules/MailConnection/IntegrationsEmailPanel.jsx` (page structure), `src/modules/MailConnection/consentResult.js` (`not_admin`), `src/modules/ProcurementHome/index.jsx` (probe for Admin **or** Security reviewer), `src/lib/i18n/strings.email.js`, `src/locales/en.json`
  - Create: `src/modules/MailConnection/RestrictionsList.jsx`, `ReachDiagram.jsx`, `PolicyForm.jsx`, `SecurityView.jsx`, `InternalEmailApproval.jsx`, `panelPolicy.js` (pure logic) + tests `restrictions.render.test.jsx`, `securityView.render.test.jsx`, `policyForm.logic.test.js`, `internalEmail.contract.test.js`

**Interfaces:**
- Consumes: Task 3 `GET/PATCH /admin/integrations/mail/policy`; Task 9 `GET /restrictions`, `GET /evidence.pdf`; Task 6 `/mail/internal*`; Task 4 consent code `not_admin`. Mock-ups: https://claude.ai/artifact/PA7WnSrUuSH5GcwyCnJ7R7 (Admin, Security reviewer, Approval artboards) — layout, wording and order come from there.
- Produces:
  - Gateway `ROLE_BY_GROUP.PROCWISE_SECURITY_REVIEWER = "SecurityReviewer"`, `GROUP_FOR_ROLE.SecurityReviewer = "PROCWISE_SECURITY_REVIEWER"`; product role name shown as "Security reviewer".
  - `mailboxService.js` adds `getPolicy()`, `savePolicy(changes, version)`, `getRestrictions()`, `evidenceUrl()`, `createInternalEmail({kind, recipients, subject, body, source_ref})`, `listMyPendingInternal()`, `editInternal(id, …)`, `approveInternal(id)`, `declineInternal(id)` — same error-shape handling as existing functions (`{detail:{code,message}}` and string detail).
  - Email page = mock-up Admin artboard: provider tabs (Microsoft 365 live; Google Workspace and "Other email (IMAP)" shown as "Not set up — coming soon", disabled), `RestrictionsList` (one row per restriction: icon by guarantee, title, explanation, status label "Proven" / "Enforced by Procwise" / "Policy on" or the failed/not-proven text in red/amber, last checked), `ReachDiagram` (green "Can reach" list of active mailboxes, red dashed "Blocked: every other mailbox in your company"), existing mailboxes table, `PolicyForm` (fields from spec §3 with plain labels; "Save as version N+1"; shows "v{n} · changed by {who}, {when}"; 409 → "Someone else changed the policy — reload to see their change"; 422 → message as given), "Download evidence report" button.
  - `SecurityView` = mock-up Security artboard: banner "You are viewing as Security reviewer…", four tiles, "Latest proof, step by step" table from restriction evidence, "Change history" table from policy history, PDF button. Rendered when `canConfigure !== true`; **no** edit control rendered.
  - Home: the Setup → Email entry probes `GET /admin/integrations/mail` when `canUsers` **or** `currentRole` is "SecurityReviewer"; entry shown when the probe succeeds.
  - `InternalEmailApproval` = mock-up Approval artboard; opened from a "Send to a colleague" button in the email view and from the user's pending list (a small "Emails waiting for your approval (n)" item in the Home bar, visible when `listMyPendingInternal()` returns any). Buttons: Approve and send / Edit / Don't send. Refusal messages shown as given.
  - `consentResult.js`: `not_admin` → "Only a Microsoft 365 administrator can connect. Ask your Microsoft administrator to click Connect."

- [ ] **Step 1: Failing tests** (render-to-string like existing panel tests; logic in `panelPolicy.js` unit-tested):
  - Restrictions render: each guarantee label appears; a `failed` item renders its explanation and the failed label; order matches the response.
  - Security view: contains the read-only banner; contains **no** `<button>` other than the PDF download (assert by counting buttons); history rows render before→after.
  - Policy form logic: building `changes` only includes edited fields; 409 → reload message; empty company domains with internal send on → shows the server's 422 message.
  - Internal email contract: every function hits the right path/verb; approve refusal `not_author` surfaces its message.
  - Gateway: `productRole(["PROCWISE_SECURITY_REVIEWER"]) === "SecurityReviewer"` and `toGroup("SecurityReviewer") === "PROCWISE_SECURITY_REVIEWER"`.
- [ ] **Step 2: Run → FAIL** (`npx vitest run src/modules/MailConnection src/modules/ProcurementHome`; gateway: its test command from `package.json`).
- [ ] **Step 3: Implement.** Strings via `tOr` with keys under `mail.settings.*` / `mail.security.*` / `mail.internal.*`; add every key to `strings.email.js` and `en.json`; extend the strings contract test's key scan to these prefixes.
- [ ] **Step 4: Run → PASS**, plus the full UI suite once (known pre-existing: 3 `language.contract` failures).
- [ ] **Step 5: Commit** (UI and gateway separately) `feat(mail): the Email page shows each restriction; a security view; approving emails to colleagues`

---

### Task 11: Docs, local proof, live runbook

**Files:**
- Modify: `docs/m365-setup.md` (`git add -f`), `.env.example`

- [ ] **Step 1: Docs.** Replace client-secret registration with certificate steps (create a self-signed or CA certificate; upload the public `.cer` to the Entra app; store `{"client_id","private_key_pem","thumbprint_sha1_b64url"}` in Secrets Manager; `M365_CLIENT_CERT_ARN`; rotation: upload the new certificate alongside, switch the secret, remove the old one; the per-process cache means a restart or up to 10 minutes). Enable the `wids` optional claim on the id token if Microsoft does not send it by default. New "Email policy" section: each setting in plain words and its default. New "Security reviewer" section: add people to `PROCWISE_SECURITY_REVIEWER`; what they can and cannot do.
- [ ] **Step 2: `.env.example`:** add `M365_CLIENT_CERT_ARN=` (comment), remove `M365_CLIENT_SECRET_ARN`.
- [ ] **Step 3: Runbook additions** (release gate, live test account): (n) certificate login works and no client secret exists on the app; (o) a non-administrator Microsoft user completing Connect is refused `not_admin`; (p) grant `Mail.Read` in Entra → the `no_mail_permissions` line turns red within one check, revoke → green; (q) set `read_scope` to replies-and-suppliers, send an unrelated internal email to the shared mailbox → it is never downloaded (check the poll log shows it skipped); (r) forward a supplier email to a colleague → waits in approvals → approve → arrives from the shared mailbox; a second user cannot approve it; (s) switch `internal_send` off with one pending → approve is refused; (t) Security reviewer account sees the Security view, cannot change anything, and downloads the evidence report; the report matches the screen.
- [ ] **Step 4: Local proof** on a private instance (port 8766, not the shared service): policy GET/PATCH as Admin; PATCH as Security reviewer → 403; restrictions JSON; evidence PDF downloads; internal email create → approve by another identity → 403. Record responses in the report.
- [ ] **Step 5: Full test run** (backend: `tests/services/mail_m365 tests/services/test_mail_*.py tests/services/test_internal_email.py tests/api/test_mail_*.py tests/api/test_internal_email_router.py tests/sql/test_email_settings_sql.py tests/sql/test_m365_mailbox_sql.py tests/test_email_dispatch_service.py`; UI `npx vitest run src/modules/MailConnection src/modules/EmailCompose src/modules/ProcurementHome`) — report exact counts.
- [ ] **Step 6: Commit** `docs(mail): certificate sign-in, the email policy, the security reviewer, and the live checks`

---

## Self-review notes

- Spec §1 rulings → Tasks 2–10; §3 policy → Tasks 1, 2, 3; §4 send rules → Task 5 (supplier) and Task 6 (internal); §5 read scope → Task 8; §6 approvals → Tasks 6, 7, 10; §7 restrictions → Task 9; §8 role → Tasks 1, 9 (server refusal sweep), 10 (gateway + UI); §9 evidence → Task 9; §10 certificate + `wids` + provider interface → Tasks 3, 4; §11 failures → Tasks 5 (policy unreadable → held), 6 (approval write failure: `create` raises, nothing sent), 8 (policy unreadable → narrower scope), 9 (`not_proven`); §12 testing → every task + Task 11.
- §13 open items resolved: supplier emails are `proc.bp_supplier.contact_email_1/2` (Task 5 adds the by-address lookup); `proc.bp_policy` has a `version` column but no history table — history lives in `policy_details.history` (Task 2); PDF via `report_export.render_pdf` (WeasyPrint, data-only fetcher) (Task 9).
- Deliberate choices: internal emails get their own outbox table rather than `proc.draft_rfq_emails` (that table is supplier/RFQ-shaped, single `recipient_email`); `mail.internal.*` actions are class `write` because Buyers author them — the author-only check and the policy re-check are the control. `SecurityReviewer` is rank 1: a person in both a Buyer group and the reviewer group resolves to the higher-ranked role (existing `highest_rank` rule) — the docs say to keep reviewers only in that group.
