# Process Monitor Watcher — Design Spec

## Problem

Documents uploaded to the system are tracked in `proc.process_monitor`. Once an upload completes (status = `Completed`), the data extraction pipeline must be triggered automatically. Currently, there is no continuous monitoring of this table — extraction must be kicked off manually.

## Solution

A real-time watcher service (`ProcessMonitorWatcher`) that uses PostgreSQL LISTEN/NOTIFY for instant event-driven processing, backed by a fallback poll for resilience. Records are processed concurrently via a thread pool.

## Table Schema (existing)

```
proc.process_monitor
├── id                 INTEGER        — row identifier
├── process_name       VARCHAR(100)   — e.g. "Local Upload"
├── type               VARCHAR(50)    — e.g. "Upload", "Import"
├── status             VARCHAR(50)    — lifecycle: Completed → Extracting → Extracted / Extraction_Failed
├── file_path          TEXT           — S3 object key (e.g. "documents/po/WADE QUT30789.pdf")
├── start_ts           TIMESTAMP      — extraction start time (set on claim)
├── created_date       TIMESTAMP      — record creation time
├── created_by         VARCHAR(50)
├── lastmodified_date  TIMESTAMP      — last update time
├── end_ts             TIMESTAMP      — extraction end time
├── category           TEXT           — document category (po, invoice, quotes, contract, spend)
├── document_type      TEXT           — file extension (pdf, xlsx, csv, png, jpg)
├── user_id            INTEGER        — uploading user
└── total_count        INTEGER        — document count in batch
```

Note: `proc.process_monitor_bkp` is reference-only and is not used by this service.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 ProcessMonitorWatcher                      │
│                                                           │
│  ┌────────────────┐       ┌──────────────────────┐       │
│  │ Listener Thread │──────▶│                      │       │
│  │ (PG LISTEN)     │      │   Dispatcher          │       │
│  └────────────────┘      │   ThreadPoolExecutor   │──▶ Orchestrator.execute_extraction_flow()
│  ┌────────────────┐      │   (4 workers)          │       │
│  │ Poller Thread   │──────▶│                      │       │
│  │ (60s fallback)  │      └──────────────────────┘       │
│  └────────────────┘                                       │
└──────────────────────────────────────────────────────────┘
         │                                                   
         ▼                                                   
   BackendScheduler (lifecycle management)                   
         │                                                   
         ▼                                                   
   FastAPI app.state (health exposure)                       
```

## Database Trigger

A PostgreSQL trigger fires `pg_notify` whenever a record reaches `status = 'Completed'`:

```sql
CREATE OR REPLACE FUNCTION proc.notify_process_monitor_ready()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'Completed' THEN
        PERFORM pg_notify('process_monitor_ready', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_process_monitor_ready
    AFTER INSERT OR UPDATE ON proc.process_monitor
    FOR EACH ROW
    EXECUTE FUNCTION proc.notify_process_monitor_ready();
```

## Status Transitions

```
Completed ──(claimed by watcher)──▶ Extracting ──(success)──▶ Extracted
                                                  ──(failure)──▶ Extraction_Failed
```

## Claim Pattern (Duplicate Prevention)

Atomic claim via `UPDATE ... RETURNING`:

```sql
UPDATE proc.process_monitor
SET status = 'Extracting', start_ts = now()
WHERE id = %s AND status = 'Completed'
RETURNING *;
```

Only one thread wins the claim. Both the listener and poller use this same function — safe under concurrency.

## Components

### 1. ProcessMonitorWatcher (src/services/process_monitor_watcher.py)

Single class with:

- `start()` / `stop()` — lifecycle management with threading.Event
- `_listen_loop()` — dedicated psycopg2 connection, runs `LISTEN process_monitor_ready`, blocks on `select()` for notifications, dispatches to thread pool
- `_poll_loop()` — every 60s queries for unclaimed `Completed` records, dispatches each
- `_claim_record(record_id)` — atomic claim, returns record dict or None
- `_process_record(record)` — resolves S3 path from `file_path`, calls `orchestrator.execute_extraction_flow(s3_object_key=file_path)`, updates status to `Extracted` or `Extraction_Failed`
- `_ensure_trigger()` — creates the PG trigger if it doesn't exist (idempotent)
- `ThreadPoolExecutor` — max 4 workers for concurrent extraction

### 2. BackendScheduler Integration

- `ProcessMonitorWatcher` created in `BackendScheduler.__init__()` alongside the email watcher
- Exposed via `get_process_monitor_watcher()` method
- `stop()` called during scheduler shutdown

### 3. FastAPI Integration

- `app.state.process_monitor_watcher` set during lifespan startup
- Shutdown handled via BackendScheduler teardown

## Error Handling

| Scenario | Behavior |
|---|---|
| LISTEN connection drops | Listener thread reconnects with exponential backoff (2s → 4s → 8s → max 60s). Poller continues independently. |
| Extraction fails | Record set to `Extraction_Failed`, error logged. Other concurrent extractions unaffected. |
| Duplicate notification | Atomic claim ensures only one thread processes a record. Loser gets zero rows, skips. |
| Service restart | Poller sweeps for any `Completed` records on startup — nothing is lost. |
| Graceful shutdown | Stop event set, threads exit. ThreadPoolExecutor waits for in-flight extractions. |

## Files Changed

| File | Change |
|---|---|
| `src/services/process_monitor_watcher.py` | **New** — ProcessMonitorWatcher service |
| `src/services/backend_scheduler.py` | Add watcher lifecycle management |
| `src/api/main.py` | Expose watcher on app.state |

## Non-Goals

- No retry mechanism for `Extraction_Failed` records (manual review or separate retry service)
- No changes to `proc.process_monitor_bkp` (reference-only)
- No changes to the extraction pipeline itself — uses existing `orchestrator.execute_extraction_flow()`
