# Change request: two NOLOGIN database roles for the email assurance layer

Prepared 2026-10-08. **Nothing in this document has been run.** The only thing executed was a read-only
catalog audit (section 2), on a read-only session, to establish the facts below.

Owner: Nick (approved in principle 2026-10-08, "treat it as a production change").
Reviewer needed: DBA / change control.

## 1. What is being asked

Create two **NOLOGIN group roles** on the RDS cluster `procwisemvpdb01`:

| Role | May do | May not do |
|---|---|---|
| `email_agent_reader` | SELECT on `proc.supplier_response`, `proc.workflow_email_tracking`, and 9 named columns of `proc.bp_supplier` | write anything; read any other table; read `bank_*`, tax or registration columns of `bp_supplier`; DDL; GRANT |
| `email_agent_writer` | SELECT, INSERT, UPDATE on the 8 `email_agent.*` tables; USAGE/SELECT on that schema's sequences | touch anything in `proc`; DELETE; TRUNCATE; DDL |

The exact SQL is `deploy/sql/2026-10-09_email_agent_roles.sql`; the rollback is
`deploy/sql/2026-10-09_email_agent_roles_rollback.sql`. Both were rehearsed on a throwaway Postgres built from
a schema copy of `bp_sqldb`, and 57 tests connect AS the roles and attempt writes, DDL, GRANT and SET ROLE
(`tests/email_evals/test_roles.py`).

Login roles (the ones a service actually connects as) are a **separate, later step** and are not part of this
change. Their passwords never go in the repo.

## 2. What is true on the cluster today (read-only audit, 2026-10-08)

One cluster hosts all nine databases: `bp_sqldb`, `bp_testdb`, `bp_testdb_it`, `postgres`, `rdsadmin`, `ses`,
`uicanvas`, `uicanvas_test`, `uicanvas_test_it`. PostgreSQL 16.11.

**A role is cluster-wide.** Creating `email_agent_reader` creates it for all nine databases at once. Its
*grants* are per-database, so it holds nothing anywhere until a GRANT is run in that database.

| Fact | Result |
|---|---|
| Application login | `procwisedb123`: not a superuser, **is a member of `rds_superuser`** (so it can create roles) |
| Do the four names already exist (`email_agent_reader`, `email_agent_writer`, `email_agent_ro_svc`, `email_agent_rw_svc`)? | **No**, none exist |
| Login-capable roles on the cluster | `procwisedb123`, `rdsadmin`, `rdswriteforwarduser` only |
| `pg_default_acl` (default privileges) | **Empty.** No default grants will leak to a new role |
| Database-level ACL | `NULL` on every database except `rdsadmin`. `NULL` means the built-in default: **PUBLIC has CONNECT and TEMPORARY on every database** |
| PUBLIC on schema `public` | USAGE, **no CREATE** (PG15+ default), in every database |
| PUBLIC on schemas `proc`, `proc_stage`, `canonical`, `email_agent` | **No access** (ACL is NULL = owner only) |
| PUBLIC privileges on any table in any database | **None** (0 rows) |
| Any grantee other than the owner on `proc` tables | **None** |
| Functions in `proc` / `proc_stage` / `canonical` with the default PUBLIC EXECUTE | 19 in `bp_sqldb` and `bp_testdb`; **0 are SECURITY DEFINER** (a SECURITY DEFINER function could write on a caller's behalf; none exists) |
| Other functions with default PUBLIC EXECUTE | 119 more in `bp_sqldb` (extension functions in `public`) |
| `email_agent` schema | exists in `bp_testdb` only; **not** in `bp_sqldb` |
| `proc.supplier_response`, `workflow_email_tracking`, `bp_supplier` | present in `bp_sqldb`, `bp_testdb`, `bp_testdb_it`, `uicanvas*` |

### What PUBLIC can reach, which is what a new role inherits

A new role is automatically a member of PUBLIC. So on **every** database, each role (and any login created in
it) will be able to: `CONNECT`, create `TEMP` tables, `USAGE` schema `public`, and `EXECUTE` the default-grant
functions above. It gets **no** table access, **no** access to `proc` data and no ability to create anything
permanent. The `rdsadmin` database refused our connection, so its PUBLIC state is **not verified**; it has an
explicit ACL (`rdsadmin=CTc/rdsadmin`), which means PUBLIC has no CONNECT there.

## 3. Findings that need a ruling

1. **The reader reaches the 19 `proc` functions.** It is granted USAGE on schema `proc` (needed to read the
   three tables) and those 19 functions carry PUBLIC EXECUTE. None is SECURITY DEFINER, so they run with the
   reader's own (read-only) rights and cannot write. *Recommendation:* accept, and add to the change record
   that a future SECURITY DEFINER function in `proc` would silently widen the reader. A guard test could
   assert none exists; not built yet.
2. **The roles can CONNECT to databases they have no business in** (`bp_sqldb`, `uicanvas`, `ses`, ...). Without
   grants they can do nothing there, but "can connect" is more than "must not reach". The change's own
   revoke statements are in section 5. Two options:
   * **A. Accept PUBLIC CONNECT** (status quo for every role on the cluster). No change to other databases.
   * **B. Close it:** `REVOKE CONNECT ON DATABASE <db> FROM PUBLIC` for databases the roles must not reach,
     then `GRANT CONNECT` back to the three login roles that exist. This changes **other systems'**
     databases, so it is a larger change than the one requested. *Recommendation:* A for this change; raise B
     as its own change for the whole cluster, because it protects against every future role, not just ours.
3. **Which databases get the grants?** The reader needs its three GRANTs in the databases the assurance layer
   reads: `bp_sqldb` (production) and, for testing, `bp_testdb`. The writer needs the `email_agent` schema,
   which exists only in `bp_testdb` today. **Nothing is granted anywhere until the capture schema is applied to
   that database**, which is held back until live verification passes (ruling 2).

## 4. Order of work (each step needs its own approval)

1. **Create both roles NOLOGIN** (the first `DO $$` block of the migration only). Cluster-wide, grants nothing.
   Verify with section 6, queries V1-V3. Stop here and review.
2. In `bp_testdb` only: apply the reader and writer GRANTs. Re-run the effective-privilege comparison in
   `tests/email_evals/test_roles.py::test_applying_the_roles_changes_no_other_roles_privileges` against it.
3. After live verification passes (separate ruling): the same GRANTs in `bp_sqldb`.
4. Create the two login roles with out-of-band passwords:
   `CREATE ROLE email_agent_ro_svc LOGIN PASSWORD :'pw' IN ROLE email_agent_reader;`
   `ALTER ROLE email_agent_ro_svc SET default_transaction_read_only = on;`
   `CREATE ROLE email_agent_rw_svc LOGIN PASSWORD :'pw' IN ROLE email_agent_writer;`
   then set `EMAIL_AGENT_RO_*` / `EMAIL_AGENT_RW_*` in the service environment (secrets store, not the repo).

## 5. Revoke statements

Role-level revoke does **not** work for CONNECT, because the privilege comes from PUBLIC, not from the role.
A `REVOKE ... FROM email_agent_reader` would be a no-op. The working statements are:

```sql
-- Option B only. Run once per database the roles must NOT reach. NOT run.
-- Targets: ses, uicanvas, uicanvas_test, uicanvas_test_it, bp_testdb_it, postgres
REVOKE CONNECT ON DATABASE ses FROM PUBLIC;
GRANT  CONNECT ON DATABASE ses TO procwisedb123, rdswriteforwarduser;
-- (repeat per database; bp_sqldb and bp_testdb are intentionally left reachable)
```

If the roles ever gain a grant they should not hold, per database:

```sql
REVOKE ALL ON ALL TABLES    IN SCHEMA proc FROM email_agent_reader, email_agent_writer;
REVOKE ALL ON SCHEMA proc FROM email_agent_writer;
REVOKE ALL ON SCHEMA email_agent FROM email_agent_reader;
```

`PUBLIC` itself is **not** altered by this change.

## 6. Verification queries (read-only; run after each step)

```sql
-- V1 the roles exist, cannot log in, hold no dangerous attributes
SELECT rolname, rolcanlogin, rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls
  FROM pg_roles WHERE rolname IN ('email_agent_reader','email_agent_writer');
-- V2 membership: nothing is a member of anything powerful
SELECT r.rolname AS member_of FROM pg_auth_members m JOIN pg_roles r ON r.oid = m.roleid
 WHERE m.member IN (SELECT oid FROM pg_roles WHERE rolname IN ('email_agent_reader','email_agent_writer'));
-- V3 proc tables either role can write to or the writer can read (expect ZERO rows; before the roles exist
--     it returns zero rows too, because the role oid is NULL, so run it again after step 2)
WITH r AS (SELECT (SELECT oid FROM pg_roles WHERE rolname='email_agent_reader') AS rd,
                  (SELECT oid FROM pg_roles WHERE rolname='email_agent_writer') AS wr)
SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace, r
 WHERE n.nspname='proc' AND c.relkind='r'
   AND ( has_table_privilege(r.rd, c.oid, 'INSERT')   OR has_table_privilege(r.rd, c.oid, 'UPDATE')
      OR has_table_privilege(r.rd, c.oid, 'DELETE')   OR has_table_privilege(r.rd, c.oid, 'TRUNCATE')
      OR has_table_privilege(r.wr, c.oid, 'SELECT')   OR has_table_privilege(r.wr, c.oid, 'INSERT')
      OR has_table_privilege(r.wr, c.oid, 'UPDATE')   OR has_table_privilege(r.wr, c.oid, 'DELETE') );
-- V4 SECURITY DEFINER functions the reader could reach (expect 0)
SELECT n.nspname, p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 WHERE n.nspname IN ('proc','proc_stage','canonical','email_agent') AND p.prosecdef;
```

## 7. Rollback

`deploy/sql/2026-10-09_email_agent_roles_rollback.sql`: `DROP OWNED BY` then `DROP ROLE`, per database, then
the roles. It fails **loudly** if either role still holds privileges in another database, which is the
intended behaviour: run it in every database that received the GRANTs first. Login roles created in step 4
are the operator's to drop.

## 8. Risk summary

* Creating the roles (step 1) changes no existing privilege: `pg_default_acl` is empty and the roles hold
  nothing. A test compares every other role's effective privileges before and after.
* The cluster-wide scope is the only real blast radius, and it is limited to the PUBLIC defaults in section 2.
* The interim control (a read-only session on the existing application login) stays in force until step 4.
  It is a guardrail, not a boundary, and every draft records which control applied (`read_control`).
* The `rdsadmin` database's PUBLIC state was not verified (connection refused, as expected on RDS).

## 9. How this audit was run

`SELECT`-only catalog queries over a session opened with `default_transaction_read_only=on` and
`readonly=True`, as `procwisedb123`, against all nine databases on 2026-10-08. No object was created,
altered or granted.
