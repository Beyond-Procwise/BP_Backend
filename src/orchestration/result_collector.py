# src/orchestration/result_collector.py
"""Consumes agent results from Redis Streams and updates workflow state.

Listens on agent:results stream, matches results to workflows,
updates PostgreSQL state, and notifies DAG Scheduler for downstream evaluation.

Spec reference: Section 1.4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from orchestration.message_protocol import ResultMessage
from orchestration.state_manager import StateManager

logger = logging.getLogger(__name__)

RESULT_STREAM = "agent:results"


class ResultCollector:
    def __init__(
        self,
        state_manager: StateManager,
        dag_scheduler: Any,
        workflow_graphs: Dict[str, Any],
    ):
        self._state = state_manager
        self._scheduler = dag_scheduler
        self._graphs = workflow_graphs

    def process_result(
        self,
        result: ResultMessage,
        execution_id: int,
        graph_name: Optional[str] = None,
    ) -> None:
        wf = self._state.get_workflow_execution(execution_id)
        if not wf:
            logger.warning("No workflow execution found for id=%d", execution_id)
            return

        current_round = wf["current_round"]
        workflow_id = wf["workflow_id"]

        # Record node result in PostgreSQL
        self._state.record_node_result(
            execution_id=execution_id,
            node_name=result.node_name,
            round_num=current_round,
            status="completed" if result.status == "SUCCESS" else "failed",
            output_data=result.data,
            pass_fields=result.pass_fields,
            error=result.error,
            duration_ms=result.duration_ms,
        )

        # Merge pass_fields into shared_data on success
        if result.pass_fields and result.status == "SUCCESS":
            self._state.merge_shared_data(execution_id, result.pass_fields)

        # Record observability event
        event_type = "node:completed" if result.status == "SUCCESS" else "node:failed"
        self._state.record_event(
            workflow_id=workflow_id,
            event_type=event_type,
            node_name=result.node_name,
            agent_type=result.agent_type,
            payload={
                "task_id": result.task_id,
                "duration_ms": result.duration_ms,
                "confidence": result.confidence,
                "error": result.error,
            },
            round_num=current_round,
        )

        # Notify DAG Scheduler
        graph = self._graphs.get(graph_name) if graph_name else None
        if graph is None:
            wf_name = wf.get("workflow_name", "")
            graph = self._graphs.get(wf_name)

        if graph is None:
            logger.warning("No graph found for workflow %s", workflow_id)
            return

        if result.status == "SUCCESS":
            self._scheduler.on_node_completed(
                graph=graph,
                execution_id=execution_id,
                workflow_id=workflow_id,
                node_name=result.node_name,
                round_num=current_round,
            )
        else:
            self._scheduler.on_node_failed(
                graph=graph,
                execution_id=execution_id,
                workflow_id=workflow_id,
                node_name=result.node_name,
                round_num=current_round,
            )

        logger.info(
            "Processed result for %s/%s: %s (duration=%dms)",
            workflow_id, result.node_name, result.status, result.duration_ms,
        )
