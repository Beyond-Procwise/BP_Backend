-- Migration 003: Add performance indexes on proc.routing
--
-- The workflow_id column is used in hot-path queries by
-- ProcessRoutingService (validate_workflow_id, mark_workflow_failed,
-- workflow_has_failed). Without an index these are full table scans.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proc_routing_workflow_id
    ON proc.routing (workflow_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proc_routing_process_status
    ON proc.routing (process_status)
    WHERE process_status IN (0, 1);

-- Index for discrepancy lookups by document
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proc_discrepancy_record_id
    ON proc.bp_discrepancy_data (record_id);
