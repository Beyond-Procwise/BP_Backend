# Reports Index (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The report builder opens on a list of your saved reports with a Generate button, instead of a blank canvas — and the Generate form on the home page stops throwing away what you typed into it.

**Architecture:** `proc.bp_reports` already stores every save as its own row named `"Title · v3"`, so "report" and "version" are the same thing to the database. We add `report_key` + `version` columns to group versions under a report, backfill them from the existing names, then expose a grouped index endpoint. The report builder (RB6 — a vanilla-JS IIFE inside `engine.js`) gains an **index screen** in front of its composer. The composer, the version-History panel, and the save path all keep working as they do today.

**Tech Stack:** NestJS + raw `dataSource.query` (gateway), jest (gateway tests), vanilla JS inside `engine.js` (report builder), React + react-query + axios (Procurement Home), PostgreSQL (`bp_sqldb`).

**Spec:** `docs/superpowers/specs/2026-07-13-report-builder-design.md` (Stage 1 only. Stage 2 = live data path; Stage 3 = real export. Each gets its own plan once this lands.)

## Global Constraints

- **Repos.** Gateway: `/home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api` (branch `Development`). UI: `/home/muthu/PycharmProjects/beyond_procwise_ui` (branch `spendiq-ui`). This plan document lives in BP_Backend but **no BP_Backend source changes are in Stage 1**.
- **Never push to `main`.** All work stays on the branches above.
- **No Claude attribution in commit messages.** No `Co-Authored-By: Claude`.
- **DB naming:** new tables/columns use the `bp_` prefix; indexes use `ix_bp_<table>_<col>`.
- **Never fabricate data.** If a value isn't in the database, render an empty state — never a plausible-looking number.
- **`proc.bp_reports` has no `CREATE TABLE` in version control** — it exists only in the database. All DDL must be `IF NOT EXISTS` and must not assume the legacy rows' shape.
- **Legacy rows have `definition IS NULL`.** Every query in this feature filters them out. The backfill must leave them alone.
- **Local stack:** start the gateway with `node --experimental-global-webcrypto`. Never run `pkill -f uvicorn` (it kills the agent's own shell).
- **DB access** for migrations/verification uses the credentials in `/home/muthu/PycharmProjects/BP_Backend/.env` (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`).

---

## File Structure

**Gateway** (`beyond_procwaise_api/src/modules/spendiq/`)
- `bp_reports_index.sql` — **create.** Migration: `report_key`, `version`, backfill, index.
- `spendiq.service.ts` — **modify.** Reports section is lines 1073-1107. Rewrite `getReports`, `createReport`; add `getReportsIndex`, `getReport`, `deleteReportByKey`.
- `spendiq.controller.ts` — **modify.** Reports routes are lines 144-157. Add three routes.
- `spendiq.service.spec.ts` — **create.** First test file in the repo. jest is already configured (`package.json:16`) with zero spec files.

**UI** (`beyond_procwise_ui/src/`)
- `modules/SpendIQ/engine.js` — **modify.** RB6 IIFE (lines 4926-5686). Add an index screen; add report open/delete; change `saveVersion` to send a base name + type; make the History panel fetch only the current report's versions.
- `modules/SpendIQ/index.jsx` — **modify.** Pass the `?new=1&name=&type=&period=` deep-link params through to the engine as `window.__RB6_INTENT__`.
- `modules/ProcurementHome/useHomeData.js` — **modify.** Add a `useRecentReports()` hook (react-query, same `q()` pattern as the existing hooks).
- `modules/ProcurementHome/index.jsx` — **modify.** Make the Generate form controlled (line ~839-855); replace the hardcoded `const reportRows = []` (line 394) with the hook.

---

## API contract (produced by Tasks 2-5, consumed by Tasks 6-8)

```
GET    /spendiq/reports/index          -> { data: [{ key, name, type, versions, latestId, latestTs }], total }
GET    /spendiq/reports/:id            -> { id, key, name, type, version, when, snap }   | 404
GET    /spendiq/reports?key=<key>      -> { data: [{ id, name, version, when, snap }], total }   (versions of ONE report, newest first)
GET    /spendiq/reports                -> same, all rows (unchanged behaviour)
POST   /spendiq/reports                <- { name, definition, type? }  -> { status:'ok', id, version }
DELETE /spendiq/reports/key/:key       -> { status:'ok', key, deleted: <n> }    (deletes ALL versions of one report)
DELETE /spendiq/reports/:id            -> { status:'ok', id }                   (one version — unchanged)
```

`name` is now the **base title** (`"Q1 Executive Review"`). The `· vN` suffix is gone: the version number lives in its own column. The UI no longer invents it.

---

## Task 1: Database migration

**Files:**
- Create: `beyond_procwaise_api/src/modules/spendiq/bp_reports_index.sql`

**Interfaces:**
- Produces: columns `proc.bp_reports.report_key` (text), `proc.bp_reports.version` (integer), index `ix_bp_reports_report_key`. Every later gateway task depends on these existing.

- [ ] **Step 1: Write the migration**

Create `beyond_procwaise_api/src/modules/spendiq/bp_reports_index.sql`:

```sql
-- Reports index (Stage 1).
--
-- Until now a saved row WAS a version: every "Save version" wrote a new row named
-- "Title · v3", so there was no way to list distinct reports. report_key groups the
-- versions of one report; version numbers them within that group.
--
-- Legacy rows (the pre-RB6 report_url artifacts) carry definition IS NULL. They are
-- excluded from every query in this feature and are NOT touched by the backfill.

ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS report_key text;
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS version    integer;

-- Backfill: split "Title · v3" into ("Title", 3). Rows with no suffix are version 1.
-- Note the separator is U+00B7 MIDDLE DOT, which is what saveVersion() emitted.
UPDATE proc.bp_reports
   SET report_key = COALESCE(report_key, btrim(regexp_replace(report_name, '\s*·\s*v[0-9]+\s*$', ''))),
       version    = COALESCE(version, NULLIF((regexp_match(report_name, '·\s*v([0-9]+)\s*$'))[1], '')::integer, 1)
 WHERE definition IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_bp_reports_report_key ON proc.bp_reports (report_key);
```

- [ ] **Step 2: Look at the rows before you change them**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT id, report_name, report_type, (definition IS NULL) AS legacy FROM proc.bp_reports ORDER BY id;"
```

Expected: a handful of rows. Note which have `legacy = t` — those must be unchanged at Step 4.

- [ ] **Step 3: Apply the migration**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -f /home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api/src/modules/spendiq/bp_reports_index.sql
```

Expected: `ALTER TABLE`, `ALTER TABLE`, `UPDATE <n>`, `CREATE INDEX`.

- [ ] **Step 4: Verify the backfill**

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT id, report_name, report_key, version, (definition IS NULL) AS legacy FROM proc.bp_reports ORDER BY report_key NULLS FIRST, version;"
```

Expected:
- Every row with `definition IS NOT NULL` has a non-null `report_key` with **no `· vN` suffix**, and an integer `version`.
- Every `legacy = t` row still has `report_key = NULL` and `version = NULL`.
- Rows that were `"X · v1"`, `"X · v2"` now share `report_key = 'X'` with versions 1 and 2.

If any non-legacy row has a null `report_key`, stop — the regex did not match its name. Show the offending name before proceeding.

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api
git add beyond_procwaise_api/src/modules/spendiq/bp_reports_index.sql
git commit -m "feat(reports): group saved rows into reports + versions

A saved row WAS a version — every save wrote a new row named 'Title · vN',
so nothing could list distinct reports. report_key groups them; version
numbers them. Backfilled from the existing names; legacy report_url rows
(definition IS NULL) untouched."
```

---

## Task 2: Gateway — the grouped index query

**Files:**
- Create: `beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts`
- Modify: `beyond_procwaise_api/src/modules/spendiq/spendiq.service.ts` (reports section, ~line 1073)

**Interfaces:**
- Consumes: `report_key` / `version` columns from Task 1.
- Produces: `SpendIqService.getReportsIndex(): Promise<{ data: Array<{key, name, type, versions, latestId, latestTs}>, total: number }>` — consumed by the controller (Task 5), the RB6 index screen (Task 6) and Procurement Home (Task 8).

This is the repo's **first test file**. jest is configured but has never been run against a spec.

- [ ] **Step 1: Write the failing test**

Create `beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts`:

```ts
import { Test } from '@nestjs/testing';
import { getDataSourceToken } from '@nestjs/typeorm';
import { DataSource } from 'typeorm';
import { SpendIqService } from './spendiq.service';

// The service talks to Postgres through raw dataSource.query(sql, params). We stand in
// a fake DataSource that records the SQL it was asked to run and replays a canned result,
// so these tests assert on the shaping logic without needing a database.
function makeService(rows: any[]) {
  const calls: Array<{ sql: string; params: any[] }> = [];
  const dataSource = {
    query: jest.fn(async (sql: string, params: any[] = []) => {
      calls.push({ sql, params });
      return rows;
    }),
  } as unknown as DataSource;
  const service = new SpendIqService(dataSource, dataSource);
  return { service, dataSource, calls };
}

describe('SpendIqService.getReportsIndex', () => {
  it('returns one entry per report, carrying its version count and latest version id', async () => {
    const { service } = makeService([
      { key: 'Q1 Executive Review', name: 'Q1 Executive Review', type: 'Executive',
        versions: '3', latest_id: 42, latest_ts: new Date('2026-07-10T09:00:00Z') },
      { key: 'Compliance Snapshot', name: 'Compliance Snapshot', type: 'Compliance',
        versions: '1', latest_id: 17, latest_ts: new Date('2026-07-04T09:00:00Z') },
    ]);

    const res = await service.getReportsIndex();

    expect(res.total).toBe(2);
    expect(res.data[0]).toEqual({
      key: 'Q1 Executive Review',
      name: 'Q1 Executive Review',
      type: 'Executive',
      versions: 3,                     // numeric, not the '3' Postgres COUNT returns
      latestId: 42,
      latestTs: '2026-07-10T09:00:00.000Z',
    });
  });

  it('excludes legacy rows that have no definition', async () => {
    const { service, calls } = makeService([]);
    await service.getReportsIndex();
    expect(calls[0].sql).toContain('definition IS NOT NULL');
  });

  it('groups by report_key', async () => {
    const { service, calls } = makeService([]);
    await service.getReportsIndex();
    expect(calls[0].sql).toContain('GROUP BY');
    expect(calls[0].sql).toContain('report_key');
  });
});
```

> **Check the constructor first.** Open `spendiq.service.ts` and look at its `constructor(...)`. It injects two datasources (the `bp_sqldb` connection and the `uicanvas` one). If the parameter order or count differs from `new SpendIqService(dataSource, dataSource)` above, fix the helper to match — pass the same fake for every datasource parameter.

- [ ] **Step 2: Run the test and watch it fail**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api
npx jest src/modules/spendiq/spendiq.service.spec.ts
```

Expected: FAIL — `service.getReportsIndex is not a function`.

- [ ] **Step 3: Implement `getReportsIndex`**

In `spendiq.service.ts`, immediately **above** the existing `getReports()` (~line 1072), add:

```ts
  // GET /spendiq/reports/index -> one entry per REPORT (versions collapsed).
  // A row in bp_reports is one VERSION; report_key groups the versions of a report.
  // The index shows the report, its version count, and a pointer to its latest version.
  async getReportsIndex() {
    const rows = await this.dataSource.query(
      `SELECT r.report_key                                     AS key,
              MAX(r.report_type)                               AS type,
              COUNT(*)                                         AS versions,
              MAX(COALESCE(r.updated_ts, r.created_ts))        AS latest_ts,
              (ARRAY_AGG(r.id          ORDER BY COALESCE(r.updated_ts, r.created_ts) DESC, r.id DESC))[1] AS latest_id,
              (ARRAY_AGG(r.report_name ORDER BY COALESCE(r.updated_ts, r.created_ts) DESC, r.id DESC))[1] AS name
         FROM proc.bp_reports r
        WHERE r.definition IS NOT NULL
          AND r.report_key IS NOT NULL
        GROUP BY r.report_key
        ORDER BY MAX(COALESCE(r.updated_ts, r.created_ts)) DESC NULLS LAST
        LIMIT 200`,
    );
    return {
      data: rows.map((r: any) => ({
        key: r.key,
        name: r.name || r.key,
        type: r.type || null,
        versions: Number(r.versions) || 0,   // COUNT(*) arrives as a string
        latestId: r.latest_id,
        latestTs: r.latest_ts ? new Date(r.latest_ts).toISOString() : null,
      })),
      total: rows.length,
    };
  }
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
npx jest src/modules/spendiq/spendiq.service.spec.ts
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api
git add beyond_procwaise_api/src/modules/spendiq/spendiq.service.ts \
        beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts
git commit -m "feat(reports): getReportsIndex — one entry per report, versions collapsed"
```

---

## Task 3: Gateway — fetch one report, and one report's versions

**Files:**
- Modify: `beyond_procwaise_api/src/modules/spendiq/spendiq.service.ts` (`getReports`, ~line 1073)
- Modify: `beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts`

**Interfaces:**
- Produces:
  - `getReport(id: number): Promise<{id, key, name, type, version, when, snap} | null>` — null when not found; the controller (Task 5) turns that into a 404.
  - `getReports(key?: string)` — **unchanged when called with no argument**. With a key, returns only that report's versions.

- [ ] **Step 1: Write the failing tests**

Append to `spendiq.service.spec.ts`:

```ts
describe('SpendIqService.getReport', () => {
  it('returns the snapshot under `snap`, not `definition`', async () => {
    const { service } = makeService([
      { id: 42, report_key: 'Q1 Executive Review', report_name: 'Q1 Executive Review',
        report_type: 'Executive', version: 3, definition: { reportTitle: 'Q1 Executive Review', rows: [] },
        ts: new Date('2026-07-10T09:00:00Z') },
    ]);

    const res = await service.getReport(42);

    expect(res).toEqual({
      id: 42,
      key: 'Q1 Executive Review',
      name: 'Q1 Executive Review',
      type: 'Executive',
      version: 3,
      when: '2026-07-10T09:00:00.000Z',
      snap: { reportTitle: 'Q1 Executive Review', rows: [] },
    });
  });

  it('returns null when the id does not exist', async () => {
    const { service } = makeService([]);
    expect(await service.getReport(999)).toBeNull();
  });
});

describe('SpendIqService.getReports', () => {
  it('filters to one report when given a key', async () => {
    const { service, calls } = makeService([]);
    await service.getReports('Q1 Executive Review');
    expect(calls[0].params).toContain('Q1 Executive Review');
  });

  it('returns every report when given no key', async () => {
    const { service, calls } = makeService([]);
    await service.getReports();
    expect(calls[0].params[0]).toBeNull();
  });
});
```

- [ ] **Step 2: Run and watch them fail**

```bash
npx jest src/modules/spendiq/spendiq.service.spec.ts
```

Expected: FAIL — `service.getReport is not a function`.

- [ ] **Step 3: Implement**

Replace the whole existing `getReports()` method (`spendiq.service.ts` ~1072-1090) with:

```ts
  // GET /spendiq/reports[?key=] -> saved RB6 report versions (proc.bp_reports.definition JSONB).
  // With a key, this is the version history of ONE report — which is what the builder's
  // History panel wants. Without one, every version of every report (unchanged behaviour).
  async getReports(key?: string) {
    const k = key && String(key).trim() ? String(key).trim() : null;
    const rows = await this.dataSource.query(
      `SELECT id, report_key, report_name, report_type, version, definition,
              COALESCE(updated_ts, created_ts) AS ts
         FROM proc.bp_reports
        WHERE definition IS NOT NULL
          AND ($1::text IS NULL OR report_key = $1)
        ORDER BY COALESCE(updated_ts, created_ts) DESC NULLS LAST, id DESC
        LIMIT 200`,
      [k],
    );
    return {
      data: rows.map((r: any) => ({
        id: r.id,
        key: r.report_key,
        name: r.report_name,
        type: r.report_type || null,
        version: r.version,
        when: r.ts ? new Date(r.ts).toISOString() : null,
        snap: r.definition,
      })),
      total: rows.length,
    };
  }

  // GET /spendiq/reports/:id -> one saved version, so opening a report from the index
  // doesn't mean downloading every report's snapshot.
  async getReport(id: number) {
    const [r] = await this.dataSource.query(
      `SELECT id, report_key, report_name, report_type, version, definition,
              COALESCE(updated_ts, created_ts) AS ts
         FROM proc.bp_reports
        WHERE id = $1 AND definition IS NOT NULL`,
      [id],
    );
    if (!r) return null;
    return {
      id: r.id,
      key: r.report_key,
      name: r.report_name,
      type: r.report_type || null,
      version: r.version,
      when: r.ts ? new Date(r.ts).toISOString() : null,
      snap: r.definition,
    };
  }
```

- [ ] **Step 4: Run and watch them pass**

```bash
npx jest src/modules/spendiq/spendiq.service.spec.ts
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api
git add beyond_procwaise_api/src/modules/spendiq/spendiq.service.ts \
        beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts
git commit -m "feat(reports): fetch a single report, and scope version history to one report"
```

---

## Task 4: Gateway — save with a real version number, delete a whole report

**Files:**
- Modify: `beyond_procwaise_api/src/modules/spendiq/spendiq.service.ts` (`createReport`, `deleteReport`)
- Modify: `beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts`

**Interfaces:**
- Produces:
  - `createReport(name, definition, createdBy?, type?): Promise<{status:'ok', id, version}>` — **the gateway now assigns the version number.** Callers send the base title; they no longer append `· vN`.
  - `deleteReportByKey(key: string): Promise<{status:'ok', key, deleted: number}>`
  - `deleteReport(id)` — unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `spendiq.service.spec.ts`:

```ts
describe('SpendIqService.createReport', () => {
  it('assigns the next version number for that report key', async () => {
    const { service, calls } = makeService([{ id: 43, version: 4 }]);

    const res = await service.createReport('Q1 Executive Review', { rows: [] }, 'nick', 'Executive');

    expect(res).toEqual({ status: 'ok', id: 43, version: 4 });
    // The version is computed in SQL from the existing rows, not passed in by the caller.
    expect(calls[0].sql).toContain('MAX(r.version)');
    expect(calls[0].params[0]).toBe('Q1 Executive Review');
  });

  it('stores the report type chosen on the home modal', async () => {
    const { service, calls } = makeService([{ id: 1, version: 1 }]);
    await service.createReport('Savings review', {}, 'nick', 'Savings');
    expect(calls[0].params).toContain('Savings');
  });

  it('falls back to a default name rather than writing an empty one', async () => {
    const { service, calls } = makeService([{ id: 1, version: 1 }]);
    await service.createReport('', {}, 'nick');
    expect(calls[0].params[0]).toBe('Report');
  });
});

describe('SpendIqService.deleteReportByKey', () => {
  it('deletes every version of one report and reports how many went', async () => {
    const { service, calls } = makeService([{ id: 1 }, { id: 2 }, { id: 3 }]);

    const res = await service.deleteReportByKey('Q1 Executive Review');

    expect(res).toEqual({ status: 'ok', key: 'Q1 Executive Review', deleted: 3 });
    expect(calls[0].params[0]).toBe('Q1 Executive Review');
    // Legacy report_url rows must survive even if they somehow share a key.
    expect(calls[0].sql).toContain('definition IS NOT NULL');
  });
});
```

- [ ] **Step 2: Run and watch them fail**

```bash
npx jest src/modules/spendiq/spendiq.service.spec.ts
```

Expected: FAIL — `createReport` returns no `version`; `deleteReportByKey is not a function`.

- [ ] **Step 3: Implement**

Replace the existing `createReport` and `deleteReport` (`spendiq.service.ts` ~1093-1107) with:

```ts
  // POST /spendiq/reports -> persist a new VERSION of a report. body {name, definition, type?}.
  // `name` is the base title: the caller no longer appends "· vN". The version number is
  // derived here from the rows already stored under this report_key, so two browsers saving
  // the same report cannot both decide they are "v3".
  async createReport(name: string, definition: unknown, createdBy?: string, type?: string) {
    const title = (name || '').trim() || 'Report';
    const [row] = await this.dataSource.query(
      `INSERT INTO proc.bp_reports
         (report_name, report_key, version, report_type, creation_date, definition, created_by, created_ts, updated_ts)
       SELECT $1, $1, COALESCE(MAX(r.version), 0) + 1, $4, CURRENT_DATE, $2::jsonb, $3, now(), now()
         FROM proc.bp_reports r
        WHERE r.report_key = $1 AND r.definition IS NOT NULL
       RETURNING id, version`,
      [title, JSON.stringify(definition ?? {}), createdBy || 'ui', type || 'spendiq'],
    );
    return { status: 'ok', id: row?.id, version: row?.version };
  }

  // DELETE /spendiq/reports/:id -> one version.
  async deleteReport(id: number) {
    await this.dataSource.query(`DELETE FROM proc.bp_reports WHERE id = $1 AND definition IS NOT NULL`, [id]);
    return { status: 'ok', id };
  }

  // DELETE /spendiq/reports/key/:key -> the whole report, every version of it.
  // The definition guard keeps the legacy report_url rows out of reach.
  async deleteReportByKey(key: string) {
    const rows = await this.dataSource.query(
      `DELETE FROM proc.bp_reports WHERE report_key = $1 AND definition IS NOT NULL RETURNING id`,
      [key],
    );
    return { status: 'ok', key, deleted: rows.length };
  }
```

> **Note on the INSERT:** `INSERT … SELECT` with an aggregate over an empty set still yields one row (`MAX` of nothing is NULL → `COALESCE(…, 0) + 1` = 1), so the first save of a brand-new report correctly becomes version 1.

- [ ] **Step 4: Run and watch them pass**

```bash
npx jest src/modules/spendiq/spendiq.service.spec.ts
```

Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api
git add beyond_procwaise_api/src/modules/spendiq/spendiq.service.ts \
        beyond_procwaise_api/src/modules/spendiq/spendiq.service.spec.ts
git commit -m "feat(reports): server assigns the version number, and a report can be deleted whole

The browser was inventing 'v3' from its own in-memory list, so two tabs would
both save a v3. The version is now derived in SQL from what is actually stored."
```

---

## Task 5: Gateway — wire the routes

**Files:**
- Modify: `beyond_procwaise_api/src/modules/spendiq/spendiq.controller.ts` (reports routes, lines 144-157)

**Interfaces:**
- Consumes: every service method from Tasks 2-4.
- Produces: the HTTP contract listed at the top of this plan. Tasks 6-8 call these.

- [ ] **Step 1: Add the routes**

In `spendiq.controller.ts`, replace the three existing reports routes (lines 144-157) with:

```ts
  // NOTE ON ROUTE ORDER: NestJS matches in declaration order, so the literal paths
  // 'reports/index' and 'reports/key/:key' MUST come before 'reports/:id' — otherwise
  // ':id' swallows "index" and Number('index') is NaN.
  @Get('reports/index')
  async getReportsIndex() {
    return this.spendIqService.getReportsIndex();
  }

  @Get('reports')
  async getReports(@Query('key') key?: string) {
    return this.spendIqService.getReports(key);
  }

  @Get('reports/:id')
  async getReport(@Param('id') id: string) {
    const report = await this.spendIqService.getReport(Number(id));
    if (!report) throw new NotFoundException(`Report ${id} not found`);
    return report;
  }

  @Post('reports')
  async createReport(@Body() body: { name: string; definition: unknown; createdBy?: string; type?: string }) {
    return this.spendIqService.createReport(body?.name, body?.definition, body?.createdBy, body?.type);
  }

  @Delete('reports/key/:key')
  async deleteReportByKey(@Param('key') key: string) {
    return this.spendIqService.deleteReportByKey(key);
  }

  @Delete('reports/:id')
  async deleteReport(@Param('id') id: string) {
    return this.spendIqService.deleteReport(Number(id));
  }
```

Add `NotFoundException` to the existing `@nestjs/common` import at the top of the file, and make sure `Query`, `Param`, `Body`, `Post`, `Delete` are all imported (most already are — check rather than assume).

- [ ] **Step 2: Build — the compiler is the test here**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api
npx tsc --noEmit -p tsconfig.json
```

Expected: no errors. If it complains about a missing import, add it.

- [ ] **Step 3: Start the gateway and exercise every route**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api
node --experimental-global-webcrypto node_modules/.bin/nest start
```

In another shell — **route order is the thing most likely to be wrong, so prove it**:

```bash
curl -s localhost:3001/spendiq/reports/index | head -c 400; echo
curl -s localhost:3001/spendiq/reports | head -c 400; echo
curl -s "localhost:3001/spendiq/reports?key=<a key from the index>" | head -c 400; echo
curl -s localhost:3001/spendiq/reports/<a latestId from the index> | head -c 400; echo
curl -s -o /dev/null -w '%{http_code}\n' localhost:3001/spendiq/reports/999999   # expect 404
```

Expected: `/reports/index` returns the grouped list — **not** a `NaN`/404 from `:id` swallowing the word "index". If it does, the routes are in the wrong order.

> If these 401 because of `CognitoGuard`, use whatever auth bypass the local stack already uses for the other `/spendiq/*` calls (the UI's dev-auth path) rather than removing the guard.

- [ ] **Step 4: Round-trip a save**

```bash
curl -s -X POST localhost:3001/spendiq/reports \
  -H 'Content-Type: application/json' \
  -d '{"name":"Plan smoke test","definition":{"reportTitle":"Plan smoke test","rows":[]},"type":"Executive"}'
# expect {"status":"ok","id":<n>,"version":1}

# Save it again — the SERVER should number this one v2:
curl -s -X POST localhost:3001/spendiq/reports \
  -H 'Content-Type: application/json' \
  -d '{"name":"Plan smoke test","definition":{"reportTitle":"Plan smoke test","rows":[]},"type":"Executive"}'
# expect {"status":"ok","id":<n+1>,"version":2}

curl -s localhost:3001/spendiq/reports/index | grep -o '"versions":2'   # expect a hit

# Clean up after yourself:
curl -s -X DELETE "localhost:3001/spendiq/reports/key/Plan%20smoke%20test"
# expect {"status":"ok","key":"Plan smoke test","deleted":2}
```

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond-procwaise-Api
git add beyond_procwaise_api/src/modules/spendiq/spendiq.controller.ts
git commit -m "feat(reports): expose the reports index, single-report fetch and whole-report delete"
```

---

## Task 6: Report builder — an index screen in front of the composer

**Files:**
- Modify: `beyond_procwise_ui/src/modules/SpendIQ/engine.js` — RB6 IIFE (4926-5686). Touch points: `MARKUP` (~5630), `mount()` (~5682), `wireControls()` (~5655), `saveVersion()` (~5367), `renderHistory()` (~5370).

**Interfaces:**
- Consumes: `GET /spendiq/reports/index`, `GET /spendiq/reports/:id`, `GET /spendiq/reports?key=`, `POST /spendiq/reports`, `DELETE /spendiq/reports/key/:key` (Task 5), via the existing bridges `window.__SPENDIQ_API__(path, params)` and `window.__SPENDIQ_API_WRITE__(method, path, body)` (defined in `SpendIQ/index.jsx:156-167`).
- Produces: `window.__RB6_INTENT__` is *read* here (set by Task 7). Shape: `{ isNew: true, name?: string, type?: string, period?: string }`.

There is **no test harness in this repo** (no vitest, no jest, zero test files). Verification for this task is driving the real app — see the steps.

- [ ] **Step 1: Add the index screen's state and data access**

In `engine.js`, just below `let STATE = templateState('exec');` / `let VERSIONS=[];` (~line 5361), add:

```js
/* ---- Screens: the builder opens on the reports index, not on a blank canvas ---- */
let screen='index';        // 'index' | 'composer'
let REPORTS=[];            // [{key,name,type,versions,latestId,latestTs}] from GET /spendiq/reports/index
let reportsLoading=true;
let reportsError='';
let currentKey='';         // report_key of the report open in the composer ('' = unsaved)

function api(path,params){ return (typeof window.__SPENDIQ_API__==='function')
  ? window.__SPENDIQ_API__(path,params) : Promise.reject(new Error('no api bridge')); }
function apiWrite(method,path,body){ return (typeof window.__SPENDIQ_API_WRITE__==='function')
  ? window.__SPENDIQ_API_WRITE__(method,path,body) : Promise.reject(new Error('no api bridge')); }

function loadReports(){
  reportsLoading=true; reportsError=''; renderIndex();
  return api('/spendiq/reports/index').then(function(res){
    REPORTS=(res&&Array.isArray(res.data))?res.data:[];
    reportsLoading=false; renderIndex();
  }).catch(function(){
    // Never invent a list. An empty index that says it failed beats a fake one.
    REPORTS=[]; reportsLoading=false; reportsError='Could not load your reports.'; renderIndex();
  });
}

function whenText(iso){ if(!iso) return ''; try{ return new Date(iso).toLocaleString(); }catch(e){ return ''; } }

function openReport(key,latestId){
  const st=document.getElementById('rb-status'); if(st)st.textContent='Opening…';
  api('/spendiq/reports/'+latestId).then(function(r){
    if(!r||!r.snap) throw new Error('empty');
    STATE=JSON.parse(JSON.stringify(r.snap));
    currentKey=key;
    screen='composer'; viewMode='edit';
    VERSIONS=[];                 // history is fetched per report now — see loadHistory()
    renderScreen(); loadHistory();
    if(st)st.textContent='Ready';
  }).catch(function(){ if(st)st.textContent='Could not open that report'; });
}

function deleteReport(key){
  if(!window.confirm('Delete “'+key+'” and all of its versions? This cannot be undone.')) return;
  apiWrite('delete','/spendiq/reports/key/'+encodeURIComponent(key)).then(function(){
    loadReports();
  }).catch(function(){ reportsError='Could not delete that report.'; renderIndex(); });
}

// Start a new report in the composer. `intent` carries the name/type/period typed into the
// home page's Generate form — which until now were collected and thrown away.
function startNewReport(intent){
  const i=intent||{};
  STATE = (i.type==='Executive'||i.type==='Compliance') ? templateState('exec') : blankState();
  STATE.locked=false; STATE.isTemplate=false;
  if(i.name) STATE.reportTitle=i.name;
  if(i.type) STATE.reportType=i.type;
  if(i.period) STATE.period=i.period;
  currentKey=''; VERSIONS=[];
  screen='composer'; viewMode='edit';
  renderScreen();
}

function backToIndex(){ screen='index'; closePalettes(); renderScreen(); loadReports(); }

// Version history is now scoped to the report you have open, rather than every version of
// every report the account has ever saved.
function loadHistory(){
  if(!currentKey){ VERSIONS=[]; renderHistory(); return; }
  api('/spendiq/reports',{key:currentKey}).then(function(res){
    VERSIONS=((res&&res.data)||[]).map(function(row){
      return { id:row.id, name:(row.name||'Report')+' · v'+row.version, when:whenText(row.when), snap:row.snap };
    });
    renderHistory();
  }).catch(function(){ /* leave whatever we have */ });
}
```

- [ ] **Step 2: Render the index**

Add these functions next to the ones above:

```js
const INDEX_MARKUP = `
  <div class="ctrl">
    <strong style="font-size:.95rem">My reports</strong>
    <span class="spacer"></span>
    <button id="rb-generate" class="primary">＋ Generate</button>
  </div>
  <div id="rb-index-list" class="rb-index"></div>`;

function renderIndex(){
  const el=document.getElementById('rb-index-list'); if(!el) return;
  if(reportsLoading){ el.innerHTML='<div class="rb-index-empty">Loading your reports…</div>'; return; }
  if(reportsError){ el.innerHTML='<div class="rb-index-empty">'+escText(reportsError)+'</div>'; return; }
  if(!REPORTS.length){
    el.innerHTML='<div class="rb-index-empty">No saved reports yet. Hit <b>Generate</b> to build your first one.</div>';
    return;
  }
  el.innerHTML=REPORTS.map(function(r){
    const vs=r.versions===1?'1 version':(r.versions+' versions');
    const type=r.type&&r.type!=='spendiq' ? '<span class="rb-index-type">'+escText(r.type)+'</span>' : '';
    return '<div class="rb-index-row" data-open="'+escText(r.key)+'" data-latest="'+r.latestId+'">'
      +   '<div class="rb-index-name">'+escText(r.name)+type+'</div>'
      +   '<div class="rb-index-meta">'+vs+' · '+escText(whenText(r.latestTs))+'</div>'
      +   '<button class="rb-index-del" data-del="'+escText(r.key)+'" title="Delete this report">✕</button>'
      + '</div>';
  }).join('');
}

// One entry point for both screens, so nothing can render the composer's controls while
// the index is showing (or vice versa).
function renderScreen(){
  const root=document.getElementById('rb6-root'); if(!root) return;
  if(screen==='index'){
    root.innerHTML=INDEX_MARKUP;
    document.getElementById('rb-generate').addEventListener('click',function(){ startNewReport(null); });
    const list=document.getElementById('rb-index-list');
    list.addEventListener('click',function(e){
      const del=e.target.closest('[data-del]');
      if(del){ e.stopPropagation(); deleteReport(del.getAttribute('data-del')); return; }
      const row=e.target.closest('[data-open]');
      if(row) openReport(row.getAttribute('data-open'), parseInt(row.getAttribute('data-latest'),10));
    });
    renderIndex();
    return;
  }
  root.innerHTML=MARKUP;
  const per=document.getElementById('rb-period'); if(per) per.value=STATE.period||'';
  wireControls();
  render();
  renderHistory();
}
```

- [ ] **Step 3: Add the "← All reports" button and its styles**

In `MARKUP` (~line 5631), make the back button the **first** control — replace the line

```js
    <button id="rb-new">＋ New report</button>
```

with

```js
    <button id="rb-back">← All reports</button>
    <button id="rb-new">＋ New report</button>
```

In `wireControls()` (~line 5655), add as the first line of the function body:

```js
  { const bk=document.getElementById('rb-back'); if(bk) bk.addEventListener('click',backToIndex); }
```

Find the RB6 stylesheet block in `engine.js` (search for `.rb-hist-row` — the index styles belong beside it) and append:

```css
.rb-index{display:flex;flex-direction:column;gap:8px;padding:16px 0}
.rb-index-row{display:flex;align-items:center;gap:12px;padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:var(--card);cursor:pointer}
.rb-index-row:hover{border-color:#20c3f3}
.rb-index-name{font-weight:700;font-size:.9rem;color:var(--ink);flex:1;display:flex;align-items:center;gap:8px}
.rb-index-type{font-weight:600;font-size:.68rem;padding:2px 8px;border-radius:999px;background:rgba(32,195,243,.12);color:#0d7fa3}
.rb-index-meta{font-size:.74rem;color:var(--ink-3);white-space:nowrap}
.rb-index-del{border:none;background:none;color:var(--ink-3);cursor:pointer;font-size:.9rem;padding:4px 6px;border-radius:6px}
.rb-index-del:hover{background:rgba(216,90,48,.12);color:#d85a30}
.rb-index-empty{padding:32px 16px;text-align:center;color:var(--ink-3);font-size:.82rem}
```

> Use whatever the surrounding rules actually use for card/border/text variables — copy them from the `.rb-hist-row` rule sitting next to it rather than trusting the names above.

- [ ] **Step 4: Save the base title (no more `· vN` in the browser)**

Replace `saveVersion()` (~line 5367) with:

```js
function saveVersion(){
  const st=document.getElementById('rb-status');
  const title=(STATE.reportTitle||'Report').trim()||'Report';
  if(st)st.textContent='Saving…';
  // The server assigns the version number. The browser used to guess it from its own
  // in-memory list, so two tabs would each decide they were saving "v3".
  apiWrite('post','/spendiq/reports',{name:title,definition:snapshot(),type:STATE.reportType||undefined})
    .then(function(r){
      currentKey=title;
      if(st)st.textContent='Saved'+(r&&r.version?(' · v'+r.version):'');
      loadHistory();
    })
    .catch(function(){ if(st)st.textContent='Could not save'; });
}
```

> Note this **removes the optimistic local `VERSIONS.unshift(...)`**. History is now whatever the server says it is, which is the point — the old code showed a version in the panel even when the POST silently failed (`.catch(function(){})`).

- [ ] **Step 5: Boot into the index**

Replace `mount()` (~line 5682) with:

```js
function mount(){
  const root=document.getElementById('rb6-root'); if(!root) return;
  // A deep-link from the home page's Generate form (see SpendIQ/index.jsx) asks for a NEW
  // report and carries the name/type/period that were typed into it. Anything else lands
  // on the index.
  const intent=window.__RB6_INTENT__; window.__RB6_INTENT__=null;
  if(intent&&intent.isNew){ startNewReport(intent); return; }
  screen='index';
  renderScreen();
  loadReports();
}
```

Delete the now-unused `_rbLoaded` flag and its old fetch (they were the only thing populating `VERSIONS` on mount; `loadHistory()` replaces them).

- [ ] **Step 6: Drive it in the browser**

Start the UI and gateway (see `LOCAL_RUN.md` in the UI repo), sign in, and go to `/spendiq?view=reportbuilder`. Confirm, in order:

1. It lands on **My reports** — not a blank canvas.
2. The reports you smoke-tested in Task 5 are listed, each with a version count and a timestamp. If the list is empty, save one from the composer first.
3. Clicking a report opens the composer with that report's title and blocks restored.
4. **History** in the composer lists only *that* report's versions, newest first, and restoring one works.
5. **Save version** shows `Saved · v2` and the History panel gains a row without a page refresh.
6. **← All reports** returns to the index, and the report you just saved shows the new version count.
7. **✕** on a row asks for confirmation, then removes the report; refreshing the page confirms it's really gone.
8. **＋ Generate** opens a fresh composer.

Check the browser console for errors at each step. A silent failure here is the most likely bug — the old code swallowed every API error with `.catch(function(){})`.

- [ ] **Step 7: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/modules/SpendIQ/engine.js
git commit -m "feat(reports): the builder opens on your reports, not a blank canvas

The report builder had no front door: it dropped you on an Untitled report with
your saved work reachable only from a History panel you'd never think to open.
It now opens on an index of your reports — open one, generate one, delete one —
and version history is scoped to the report you actually have open."
```

---

## Task 7: Carry the home page's Generate form into the builder

**Files:**
- Modify: `beyond_procwise_ui/src/modules/SpendIQ/index.jsx` (the mount effect, ~line 145-167)
- Modify: `beyond_procwise_ui/src/modules/ProcurementHome/index.jsx` (`gotoSpend` ~line 246, the Generate form ~line 839-855)

**Interfaces:**
- Produces: `window.__RB6_INTENT__ = { isNew: true, name, type, period }` — consumed by `mount()` from Task 6.

- [ ] **Step 1: Pass the deep-link params into the engine**

In `SpendIQ/index.jsx`, inside the mount effect where the other `window.__SPENDIQ_*` bridges are set (just before `applyView` runs), add:

```jsx
      // The home page's Generate form deep-links here with what the user typed. RB6 reads
      // this on mount and opens a NEW report pre-titled, instead of a blank "Untitled report".
      const qs = new URLSearchParams(search);
      if (qs.get('new') === '1') {
        window.__RB6_INTENT__ = {
          isNew: true,
          name: qs.get('name') || '',
          type: qs.get('type') || '',
          period: qs.get('period') || '',
        };
      }
```

`search` is already in scope (`const { search } = useLocation();`, line 115).

- [ ] **Step 2: Make the Generate form controlled**

In `ProcurementHome/index.jsx`, add state next to the other `useState` declarations near the top of the component:

```jsx
  // The Generate form used to be three uncontrolled inputs whose values were never read —
  // you typed a report name and it was silently discarded.
  const [rptName, setRptName] = useState('');
  const [rptType, setRptType] = useState('Executive');
  const [rptPeriod, setRptPeriod] = useState('This quarter');
```

- [ ] **Step 3: Bind the three inputs and the button**

In the reports modal (~line 839-855), bind each input and rewrite the Generate handler:

```jsx
              <input type="text" placeholder="e.g. Q2 CPO summary"
                value={rptName} onChange={(e) => setRptName(e.target.value)}
                style={css('width:100%;height:34px;font-size:13px;padding:0 10px;border:1px solid rgba(0,0,0,.14);border-radius:8px;background:#fff;color:#1f2933;outline:none;')} />
```

```jsx
              <select value={rptType} onChange={(e) => setRptType(e.target.value)}
                style={css('width:100%;height:34px;font-size:13px;padding:0 8px;border:1px solid rgba(0,0,0,.14);border-radius:8px;background:#fff;color:#1f2933;')}>
                <option>Executive</option><option>Compliance</option><option>Category</option><option>Savings</option>
              </select>
```

```jsx
              <select value={rptPeriod} onChange={(e) => setRptPeriod(e.target.value)}
                style={css('width:100%;height:34px;font-size:13px;padding:0 8px;border:1px solid rgba(0,0,0,.14);border-radius:8px;background:#fff;color:#1f2933;')}>
                <option>This quarter</option><option>This month</option><option>Year to date</option>
              </select>
```

```jsx
            <button onClick={() => navigate('/spendiq?view=reportbuilder&new=1'
                + '&name=' + encodeURIComponent(rptName.trim())
                + '&type=' + encodeURIComponent(rptType)
                + '&period=' + encodeURIComponent(rptPeriod))}
              style={css('height:34px;padding:0 16px;font-size:13px;font-weight:600;color:#fff;background:#20C3F3;border:none;border-radius:8px;cursor:pointer;display:flex;align-items:center;gap:6px;')}>
              <i className="ti ti-plus" style={css('font-size:15px;')} />Generate
            </button>
```

- [ ] **Step 4: Drive it**

From `/home`: click **Reports**, type `Plan task 7 report`, pick Type `Savings` and Period `This month`, click **Generate**.

Expected: the report builder opens **in the composer** (not the index), with the title already reading `Plan task 7 report`. Click **Save version**, then **← All reports** — the report appears in the index, typed `Savings`.

Then go to `/spendiq?view=reportbuilder` **directly** (no query params) and confirm you land on the index instead. That proves the intent is consumed once and doesn't stick around.

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/modules/SpendIQ/index.jsx src/modules/ProcurementHome/index.jsx
git commit -m "fix(reports): Generate stops throwing away the form you just filled in

The name, type and period on the home page's report modal were uncontrolled
inputs that nothing ever read — Generate ignored them and opened an Untitled
report. They now travel with the deep-link and title the new report."
```

---

## Task 8: The home page's "Recent reports" table stops being empty

**Files:**
- Modify: `beyond_procwise_ui/src/modules/ProcurementHome/useHomeData.js`
- Modify: `beyond_procwise_ui/src/modules/ProcurementHome/index.jsx` (line ~394)

**Interfaces:**
- Consumes: `GET /spendiq/reports/index` (Task 5).
- Produces: `useRecentReports(): { reports: Array<{key,name,type,versions,latestId,latestTs}>, isLoading }`

- [ ] **Step 1: Add the hook**

In `useHomeData.js`, below the existing `useHomeData` export, add:

```js
// Saved reports for the home page's "Recent reports" table. This table has been hardcoded
// to [] since it was built, with a comment saying no backend existed — GET /spendiq/reports
// has in fact existed all along and the report builder was already using it.
export function useRecentReports() {
  const reports = useQuery(q('reports', '/spendiq/reports/index'));
  return {
    reports: Array.isArray(reports.data?.data) ? reports.data.data : [],
    isLoading: reports.isLoading,
  };
}
```

- [ ] **Step 2: Use it**

In `ProcurementHome/index.jsx`, update the import on line 23:

```jsx
import { useHomeData, useRecentReports, gbp, num, pct } from './useHomeData';
```

Call it next to the existing `useHomeData()` (~line 68):

```jsx
  const { reports: recentReports } = useRecentReports();
```

Then replace the hardcoded empty list (line ~394):

```jsx
  // No backend exists yet for recent activity or saved reports (see SpendIQ BACKEND_GAPS.md)
  const recents = [];
  const reportRows = [];
```

with:

```jsx
  // No backend exists yet for recent activity (see SpendIQ BACKEND_GAPS.md).
  const recents = [];
  // Saved reports, newest first. Empty until you save one — never seeded with examples.
  const reportRows = recentReports.slice(0, 5).map((r) => ({
    key: r.key,
    name: r.name,
    type: r.type && r.type !== 'spendiq' ? r.type : '—',
    typeBg: 'rgba(32,195,243,.12)',
    typeColor: '#0d7fa3',
    created: r.latestTs ? new Date(r.latestTs).toLocaleDateString('en-GB') : '—',
    latestId: r.latestId,
  }));
```

- [ ] **Step 3: Make the rows open the right report**

The row click, the eye icon and the download icon (lines ~886-897) all currently call `gotoSpend('reportbuilder')`, which opens the builder generically. Point the first two at the actual report, and — since export doesn't exist until Stage 3 — **remove the download icon rather than leave a button that lies**.

Row click and eye icon become:

```jsx
onClick={() => navigate('/spendiq?view=reportbuilder&open=' + encodeURIComponent(row.key) + '&latest=' + row.latestId)}
```

Then in `SpendIQ/index.jsx`, extend the intent block from Task 7 to handle opening:

```jsx
      if (qs.get('open')) {
        window.__RB6_INTENT__ = { isOpen: true, key: qs.get('open'), latestId: Number(qs.get('latest')) };
      }
```

And in `engine.js`'s `mount()` (Task 6, Step 5), handle it before the `isNew` branch:

```js
  if(intent&&intent.isOpen&&intent.latestId){ screen='index'; renderScreen(); openReport(intent.key,intent.latestId); return; }
```

- [ ] **Step 4: Drive it**

Go to `/home`, click **Reports**. Expected: the reports you saved in Tasks 6-7 are listed with their type and date — not the "no reports" empty state. Click one: the builder opens **that** report. Delete every report from the builder's index, come back to `/home` → the table shows the empty state again, not a stale list.

- [ ] **Step 5: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/modules/ProcurementHome/useHomeData.js src/modules/ProcurementHome/index.jsx src/modules/SpendIQ/index.jsx
git commit -m "feat(reports): show saved reports on the home page

The Recent reports table was hardcoded to [] with a comment saying no backend
existed. It did — the report builder was already reading it. Rows now open the
report they name. The download icon is removed until export is real (Stage 3)."
```

---

## Task 9: End-to-end verification on the live stack

**Files:** none — this task changes no code. It exists because the project's standing rule is that a change is proved on the running local server against live `bp_sqldb`, not by tests alone.

- [ ] **Step 1: Bring up the full stack**

Gateway (`node --experimental-global-webcrypto`), UI (`npm run dev`), both against live `bp_sqldb`. BP_Backend is not needed for Stage 1.

- [ ] **Step 2: Walk the exact path from the original complaint**

`/home` → **Reports** → type a name, pick a type and period → **Generate** → the builder opens on a pre-titled new report → **Save version** → **← All reports** → *the report is in the list*.

That is the bug this stage exists to fix. If any step still dead-ends on a blank canvas, the stage is not done.

- [ ] **Step 3: Prove the grouping is real, in the database**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT report_key, COUNT(*) AS versions, MAX(version) AS latest
        FROM proc.bp_reports WHERE definition IS NOT NULL
       GROUP BY report_key ORDER BY MAX(COALESCE(updated_ts, created_ts)) DESC;"
```

Expected: the counts here match the version counts shown in the UI index, exactly. Save the same report twice more and confirm both the SQL and the UI go to 3.

- [ ] **Step 4: Confirm the legacy rows were not harmed**

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT id, report_name, report_key, version FROM proc.bp_reports WHERE definition IS NULL;"
```

Expected: the same rows as before Task 1, still with `report_key = NULL` and `version = NULL`. This feature must not have touched them.

- [ ] **Step 5: Check the console is clean**

With the builder open, the browser console shows no errors and no failed network calls. The pre-existing code swallowed API failures silently (`.catch(function(){})`), so an empty index could mean "no reports" *or* "the call 500'd" — make sure it means the former.

- [ ] **Step 6: Report honestly**

Write up what works and what doesn't. If something in Steps 2-5 failed, say so with the output rather than reporting the stage complete.

---

## What this stage does NOT do

Worth stating so a reviewer doesn't flag them as bugs:

- **The tiles still show demo numbers.** They're still hardcoded in the browser. That is Stage 2 — the numbers move behind a real endpoint, keeping demo *values* but making the data *path* real, with a "Demo data" badge.
- **Export PDF is still `window.print()`.** That is Stage 3. This is why Task 8 removes the download icon rather than wiring it to something that doesn't exist.
- **The period selector still only picks between three hardcoded 2025 slices.** Also Stage 2.
