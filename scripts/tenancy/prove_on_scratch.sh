#!/usr/bin/env bash
# Applies the tenant-isolation drafts to a THROWAWAY PostgreSQL 16 container and asserts what they do.
# Needs docker + the postgres:16-alpine image. Never touches any real database (its own container,
# 127.0.0.1:55432, removed at the end). Exit 0 = every assertion held.
set -u
cd "$(dirname "$0")/../.."
N=tenancy_scratch_pg; PORT=55432; FAIL=0
SQL=deploy/sql/tenancy/2026-10-21_tenant_isolation
pg(){ PGPASSWORD="$1" psql -h 127.0.0.1 -p $PORT -U "$2" -d postgres -Atq -v ON_ERROR_STOP=1 "${@:3}"; }
su(){ pg scratch postgres "$@"; }
as_owner(){ ( echo "SET ROLE owner_r;"; cat "$1" ) | su -f - >/dev/null 2>&1; }
ok(){ if [ "$2" = "$3" ]; then echo "  PASS  $1"; else echo "  FAIL  $1  (expected '$3', got '$2')"; FAIL=1; fi; }
app(){ printf "SET ROLE owner_r;\n%s\n" "$1" | pg x app_r 2>&1 | tail -1 | cut -c1-90; }
sup(){ printf "%s\n" "$1" | pg x procwise_readonly 2>&1 | tail -1 | cut -c1-90; }

docker rm -f $N >/dev/null 2>&1
docker run -d --rm --name $N -e POSTGRES_PASSWORD=scratch -p 127.0.0.1:$PORT:5432 postgres:16-alpine >/dev/null || exit 2
trap 'docker rm -f $N >/dev/null 2>&1' EXIT
for i in $(seq 1 40); do su -c "select 1" >/dev/null 2>&1 && break; sleep 1; done
sleep 2

python3 - <<'PY' > /tmp/tenancy_fixture.sql
T=[l.strip() for l in open("tests/tenancy/proc_tables_2026-10-21.txt")]
L={l.strip() for l in open("tests/tenancy/proc_tables_already_labelled_2026-10-21.txt")}
D={l.strip() for l in open("tests/tenancy/proc_tables_label_default_2026-10-21.txt")}
NN={l.strip() for l in open("tests/tenancy/proc_tables_label_notnull_2026-10-21.txt")}
V=[l.strip() for l in open("tests/tenancy/proc_views_2026-10-21.txt")]
print("CREATE ROLE owner_r NOLOGIN; CREATE ROLE app_r LOGIN PASSWORD 'x' IN ROLE owner_r; CREATE ROLE procwise_readonly LOGIN PASSWORD 'x';")
print("CREATE SCHEMA proc AUTHORIZATION owner_r; GRANT USAGE ON SCHEMA proc TO procwise_readonly; SET ROLE owner_r;")
for t in T:
    lab = t in L
    col = ", tenant_id text NOT NULL DEFAULT 'default'" if t in D else (", tenant_id text NOT NULL" if t in NN else (", tenant_id text" if lab else ""))
    print(f"CREATE TABLE proc.{t} (id serial primary key, v text{col});")
    if t in D:
        print(f"INSERT INTO proc.{t}(v) VALUES ('a'),('b');")
    elif t in NN:
        print(f"INSERT INTO proc.{t}(v, tenant_id) VALUES ('a','default'),('b','default');")
    else:
        print(f"INSERT INTO proc.{t}(v{', tenant_id' if lab else ''}) VALUES ('a'{', NULL' if lab else ''}),('b'{', chr(100)||chr(101)||chr(102)||chr(97)||chr(117)||chr(108)||chr(116)' if lab else ''});")
    print(f"GRANT SELECT ON proc.{t} TO procwise_readonly;")
for v in V: print(f"CREATE VIEW proc.{v} AS SELECT * FROM proc.bp_invoice_raw;")
PY
su -f /tmp/tenancy_fixture.sql || { echo "fixture failed"; exit 2; }

echo "part 1 (add label)"
as_owner $SQL"_1_add_label.sql"; as_owner $SQL"_1_add_label.sql"   # twice: must be repeatable
ok "every owned table has tenant_id"      "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id'")" 222
ok "part 1 leaves every owned table with the default label (197 added + 12 old)" "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id' and column_default is not null")" 209
ok "a row written between part 1 and part 6 is labelled, not NULL" "$(printf "SET ROLE owner_r; INSERT INTO proc.bp_invoice_raw(v) VALUES ('window'); select tenant_id from proc.bp_invoice_raw where v='window';\n" | su -f - 2>&1 | tail -1)" default
su -c "delete from proc.bp_invoice_raw where v='window'" >/dev/null
ok "pre-existing NOT NULL tables still take an insert" "$(printf "SET ROLE owner_r; INSERT INTO proc.bp_value_outcome(v) VALUES ('still-works'); select tenant_id from proc.bp_value_outcome where v='still-works';\n" | su -f - 2>&1 | tail -1)" default
su -c "delete from proc.bp_value_outcome where v='still-works'" >/dev/null
echo "part 6 (stamp on insert)"
as_owner $SQL"_6_stamp_on_insert.sql"; as_owner $SQL"_6_stamp_on_insert.sql"   # twice: must be repeatable
one(){ printf "SET ROLE owner_r;\n%s\n" "$1" | su -f - 2>&1 | tail -1 | cut -c1-90; }
ok "a stamp trigger on every owned table"  "$(su -Atc "select count(*) from pg_trigger where tgname='tr_bp_stamp_tenant'")" 220
ok "every default is gone (the trigger replaces them)"           "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id' and column_default is not null")" 0
ok "bound tenant is stamped"               "$(one "select set_config('app.tenant_id','acme',false); insert into proc.bp_account(v) values ('s1'); select tenant_id from proc.bp_account where v='s1';")" acme
ok "bound tenant stamped on a NOT NULL table" "$(one "select set_config('app.tenant_id','acme',false); insert into proc.bp_value_outcome(v) values ('s2'); select tenant_id from proc.bp_value_outcome where v='s2';")" acme
ok "a named tenant is kept, never overwritten" "$(one "select set_config('app.tenant_id','acme',false); insert into proc.bp_account(v,tenant_id) values ('s3','beta'); select tenant_id from proc.bp_account where v='s3';")" beta
ok "nothing bound: stays NULL, not guessed" "$(one "insert into proc.bp_account(v) values ('s4'); select coalesce(tenant_id,'NULL') from proc.bp_account where v='s4';")" NULL
ok "empty tenant counts as none"           "$(one "select set_config('app.tenant_id','',false); insert into proc.bp_account(v) values ('s5'); select coalesce(tenant_id,'NULL') from proc.bp_account where v='s5';")" NULL
ok "system mode with no tenant: NULL"      "$(one "select set_config('app.system','on',false); insert into proc.bp_account(v) values ('s6'); select coalesce(tenant_id,'NULL') from proc.bp_account where v='s6';")" NULL
ok "nothing bound into a NOT NULL table is refused" "$(printf "SET ROLE owner_r;\ninsert into proc.bp_value_outcome(v) values ('s7');\n" | su -f - 2>&1 | grep -c 'null value in column')" 1
ok "multi-row insert stamps every row"     "$(one "select set_config('app.tenant_id','acme',false); insert into proc.bp_account(v) values ('m1'),('m2'),('m3'); select count(*) from proc.bp_account where v like 'm_' and tenant_id='acme';")" 3
ok "upsert path stamps the inserted row"   "$(one "select set_config('app.tenant_id','acme',false); create unique index if not exists u_v on proc.bp_account(v); insert into proc.bp_account(v) values ('u1') on conflict (v) do update set v=excluded.v; select tenant_id from proc.bp_account where v='u1';")" acme
su -c "delete from proc.bp_account where v ~ '^(s[0-9]|m[0-9]|u1)$'; delete from proc.bp_value_outcome where v ~ '^s[0-9]$'; drop index if exists proc.u_v;" >/dev/null
ok "existing rows labelled 'default'"     "$(su -Atc "select count(*) from proc.bp_invoice_raw where tenant_id='default'")" 2
echo "part 2 (relabel + index)"
printf "SET ROLE owner_r; INSERT INTO proc.bp_invoice_raw(v) VALUES ('forgot');\n" | su -f - >/dev/null
su -f - < <( echo "SET ROLE owner_r;"; cat $SQL"_2_backfill_and_index.sql" ) >/dev/null 2>&1
ok "straggler row relabelled"             "$(su -Atc "select count(*) from proc.bp_invoice_raw where tenant_id is null")" 0
ok "NULL in already-labelled owned table fixed" "$(su -Atc "select count(*) from proc.bp_account where tenant_id is null")" 0
ok "shared tables NOT touched (still NULL)" "$(su -Atc "select count(*) from proc.bp_document_type where tenant_id is null")" 1
ok "220 indexes built"                    "$(su -Atc "select count(*) from pg_indexes where schemaname='proc' and indexname like 'ix_bp_%_tenant%'")" 220
echo "part 3 (not null)"
as_owner $SQL"_3_not_null.sql"
ok "220 columns NOT NULL"                 "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id' and is_nullable='NO'")" 220
ok "unlabelled insert refused"            "$(printf "SET ROLE owner_r; INSERT INTO proc.bp_invoice_raw(v) VALUES ('x');\n" | su -f - 2>&1 | grep -c 'null value')" 1
echo "parts 5 + 4 (support log, row-level security)"
as_owner $SQL"_5_support_access.sql"; as_owner $SQL"_4_row_level_security.sql"
ok "RLS forced on 220 tables"             "$(su -Atc "select count(*) from pg_class c join pg_namespace n on n.oid=c.relnamespace where n.nspname='proc' and c.relrowsecurity and c.relforcerowsecurity")" 220
ok "own tenant sees its rows"             "$(app "select set_config('app.tenant_id','default',false); select count(*) from proc.bp_invoice_raw;")" 3
ok "other tenant sees nothing"            "$(app "select set_config('app.tenant_id','other-co',false); select count(*) from proc.bp_invoice_raw;")" 0
ok "forgotten tenant sees nothing"        "$(app "select count(*) from proc.bp_invoice_raw;")" 0
ok "system job sees all"                  "$(app "select set_config('app.system','on',false); select count(*) from proc.bp_invoice_raw;")" 3
ok "cross-tenant insert refused"          "$(app "select set_config('app.tenant_id','other-co',false); insert into proc.bp_invoice_raw(v,tenant_id) values ('x','default');" | grep -c 'row-level security')" 1
ok "cross-tenant update changes 0 rows"   "$(app "select set_config('app.tenant_id','other-co',false); with u as (update proc.bp_invoice_raw set v='h' returning 1) select count(*) from u;")" 0
ok "view cannot read around the policy"   "$(app "select set_config('app.tenant_id','other-co',false); select count(*) from proc.bp_deal_overview;")" 0
ok "support, no reason: sees nothing"     "$(sup "select count(*) from proc.bp_invoice_raw;")" 0
ok "support, short reason refused"        "$(sup "select proc.bp_support_begin('x','default');" | grep -c 'check\|violates\|bp_support_begin')" 1
ok "support with reason reads"            "$(sup "select proc.bp_support_begin('Customer ticket 4471: invoice dispute','default'); select count(*) from proc.bp_invoice_raw;")" 3
ok "support cannot write"                 "$(sup "select proc.bp_support_begin('Customer ticket 4471: invoice dispute','default'); update proc.bp_invoice_raw set v='x';" | grep -c 'permission denied')" 1
ok "support cannot delete its own log"    "$(sup "delete from proc.bp_support_access_log;" | grep -c 'permission denied')" 1
ok "support windows were logged"          "$(su -Atc "select count(*) >= 2 from proc.bp_support_access_log")" t
echo "rollbacks (4, 5, 3, 1)"
for p in 4_row_level_security 5_support_access 3_not_null 6_stamp_on_insert 1_add_label; do as_owner $SQL"_${p}_rollback.sql"; done
ok "no policies left"                     "$(su -Atc "select count(*) from pg_policies where schemaname='proc'")" 0
ok "no RLS left"                          "$(su -Atc "select count(*) from pg_class c join pg_namespace n on n.oid=c.relnamespace where n.nspname='proc' and c.relrowsecurity")" 0
ok "label columns back to the original 25" "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id'")" 25
ok "row data intact"                      "$(su -Atc "select count(*) from proc.bp_invoice_raw")" 3
ok "tables that were NOT NULL before are still NOT NULL after rollback" "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id' and is_nullable='NO'")" 15
ok "no stamp triggers or function left"   "$(su -Atc "select (select count(*) from pg_trigger where tgname='tr_bp_stamp_tenant')+(select count(*) from pg_proc where proname='bp_stamp_tenant')")" 0
ok "after all rollbacks only the 12 original defaults remain" "$(su -Atc "select count(*) from information_schema.columns where table_schema='proc' and column_name='tenant_id' and column_default is not null")" 12
ok "support objects removed"              "$(su -Atc "select (select count(*) from pg_proc where proname='bp_support_begin')+(select count(*) from pg_tables where tablename='bp_support_access_log')")" 0
[ $FAIL -eq 0 ] && echo "ALL ASSERTIONS HELD" || echo "SOME ASSERTIONS FAILED"
exit $FAIL
