import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from services.email_watcher import EmailWatcherService
from services.process_monitor_watcher import ProcessMonitorWatcher
from services.promotion_listener_service import PromotionListenerService
from services.uicanvas_bridge import UicanvasBridge

from services.model_training_endpoint import ModelTrainingEndpoint
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


def price_outlier_enabled() -> bool:
    """OFF by default, unlike the other jobs.

    The seeded corpus is ~190,000 lines against ~1,000 findings already open,
    so a loose threshold could bury the Action Centre. Enable only after
    reviewing a dry run.
    """
    import os
    return os.environ.get("PRICE_OUTLIER_ENABLED", "0").strip() in ("1", "true", "True")


def price_outlier_interval_minutes() -> int:
    import os
    try:
        minutes = int(os.environ.get("PRICE_OUTLIER_INTERVAL_MINUTES", "60"))
    except ValueError:
        return 60
    return max(1, minutes)


# Jobs in the same lane never run at the same time; different lanes run in
# parallel. The default is deliberately a SHARED lane, so every job keeps the
# strict one-at-a-time behaviour the old inline loop gave it and concurrency is
# something a job has to opt into. Put a job in its own lane only when nothing
# else writes what it writes.
DEFAULT_JOB_LANE = "pipeline"


class _ScheduledJob:
    def __init__(
        self,
        name: str,
        runner: Callable[[], None],
        interval: timedelta,
        initial_delay: Optional[timedelta] = None,
        *,
        one_shot: bool = False,
        lane: str = DEFAULT_JOB_LANE,
    ) -> None:
        self.name = name
        self.runner = runner
        self.interval = interval
        delay = initial_delay or timedelta(0)
        self.next_run = datetime.now(timezone.utc) + delay
        self._lock = threading.Lock()
        self.one_shot = one_shot
        self.lane = lane

    def due(self, moment: datetime) -> bool:
        with self._lock:
            return moment >= self.next_run

    def mark_executed(self, executed_at: datetime) -> None:
        with self._lock:
            self.next_run = executed_at + self.interval


class BackendScheduler:
    _instance: Optional["BackendScheduler"] = None
    _instance_lock = threading.Lock()
    _poll_seconds: float = 60.0
    TRAINING_JOB_NAME = "context-training-dispatch"

    def __init__(
        self,
        agent_nick,
        *,
        training_endpoint: Optional[ModelTrainingEndpoint] = None,
        orchestrator: Optional[Any] = None,
    ) -> None:
        self.agent_nick = agent_nick
        self._jobs: Dict[str, _ScheduledJob] = {}
        self._lock = threading.Lock()
        # One worker thread per lane, so a slow lane cannot delay another. Its own
        # lock, not _lock: _dispatch must never contend with job registration.
        self._lane_threads: Dict[str, threading.Thread] = {}
        self._lane_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._relationship_scheduler = self._init_relationship_scheduler()
        self._training_endpoint = training_endpoint
        self._email_watcher_service: Optional[EmailWatcherService] = None
        self._email_watcher_lock = threading.Lock()
        self._process_monitor_watcher: Optional[ProcessMonitorWatcher] = None
        self._process_monitor_lock = threading.Lock()
        self._uicanvas_bridge: Optional[UicanvasBridge] = None
        self._promotion_listener: Optional[PromotionListenerService] = None
        self._orchestrator = orchestrator
        self._register_default_jobs()
        self.start()
        # Ensure the email watcher service is running as soon as the scheduler
        # is initialised so replies can be processed even after restarts.
        try:
            self._ensure_email_watcher_service()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to start email watcher service during initialisation")
        try:
            self._ensure_process_monitor_watcher()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to start process monitor watcher during initialisation")
        try:
            self._ensure_uicanvas_bridge()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to start uicanvas bridge during initialisation")
        try:
            self._ensure_promotion_listener()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to start promotion listener during initialisation")

    def _ensure_promotion_listener(self) -> Optional[PromotionListenerService]:
        """Start the daemon that listens for HITL discrepancy resolutions
        and promotes _raw rows to _stg. Enable/disable with
        PROMOTION_LISTENER_ENABLED (default: enabled)."""
        import os
        if os.environ.get("PROMOTION_LISTENER_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("PromotionListenerService disabled by PROMOTION_LISTENER_ENABLED")
            return None
        if self._promotion_listener is None:
            # Drive the rest of the chain on the promotion event (stg->trgt ->
            # deal-linking -> mining) instead of waiting for the deal-assignment
            # timer — see _on_doc_promoted / _run_downstream_chain.
            self._promotion_listener = PromotionListenerService(
                on_promoted=self._on_doc_promoted)
            self._promotion_listener.start()
        return self._promotion_listener

    DOWNSTREAM_CHAIN_JOB_NAME = "downstream-chain"

    def _on_doc_promoted(self, result, payload) -> None:
        """Called by the promotion listener after a _raw -> _stg promotion. Run
        the downstream chain on the event, coalescing bursts (many docs promoting
        at once) into a single one-shot run via a short debounce delay."""
        try:
            import os
            try:
                delay = float(os.environ.get("DOWNSTREAM_CHAIN_DELAY_SECONDS", "5"))
            except ValueError:
                delay = 5.0
            with self._lock:
                pending = self.DOWNSTREAM_CHAIN_JOB_NAME in self._jobs
            if pending:
                return  # a run is already queued; it will pick up this doc too
            self.submit_once(
                self.DOWNSTREAM_CHAIN_JOB_NAME,
                self._run_downstream_chain,
                initial_delay=timedelta(seconds=max(0.0, delay)),
            )
        except Exception:
            logger.exception("scheduling downstream chain failed (non-fatal)")

    def _run_downstream_chain(self) -> None:
        """Event-driven tail of the pipeline: catch up any stranded _raw rows,
        promote eligible _stg rows to _trgt, assign deals, then chain opportunity
        mining when deals changed. Idempotent — the periodic trgt-promotion /
        deal-assignment jobs remain as a backstop."""
        try:
            from src.services.extraction.promotion import promote_pending
            pend = promote_pending()
            if pend.get("promoted") or pend.get("failed"):
                logger.info("downstream chain: pending-raw catch-up %s", pend)
        except Exception:
            logger.exception("downstream chain: pending-raw catch-up failed")
        try:
            from src.services.linking_engine import promote_ready
            prom = promote_ready()
            logger.info("downstream chain: trgt promotion %s", prom)
        except Exception:
            logger.exception("downstream chain: trgt promotion failed")
        try:
            from src.services.deal_assignment_service import assign_deals
            deal_result = assign_deals()
            logger.info("downstream chain: deal assignment %s", deal_result)
        except Exception:
            logger.exception("downstream chain: deal assignment failed")
            return
        try:
            self._chain_opportunity_mining(deal_result)
        except Exception:
            logger.exception("downstream chain: opportunity mining failed")
        # Refresh the knowledge graph so all agents see the new deals/opportunities
        # on the same event — but THROTTLED: a full graph rebuild is expensive, so
        # under sustained uploads we coalesce to at most one rebuild per
        # KG_SYNC_THROTTLE_SECONDS (default 5 min). The periodic kg-sync job remains
        # a backstop, and the next promotion event after the window rebuilds again.
        import os
        import time
        try:
            throttle = float(os.environ.get("KG_SYNC_THROTTLE_SECONDS", "300"))
        except ValueError:
            throttle = 300.0
        now = time.monotonic()
        if now - getattr(self, "_last_kg_sync_at", 0.0) >= throttle:
            try:
                self._run_kg_sync()
                self._last_kg_sync_at = now
            except Exception:
                logger.exception("downstream chain: KG sync failed")
        else:
            logger.debug("downstream chain: KG sync throttled (last run %.0fs ago)",
                         now - getattr(self, "_last_kg_sync_at", 0.0))
        # Duplicate invoices are only visible once the new invoice is in _trgt beside the
        # older one, so this runs at the tail of the chain. Idempotent (a document already
        # carrying the finding is skipped), and never allowed to break the chain.
        #
        # OFF by default. The detector scans the WHOLE corpus, not just what just arrived,
        # so the first automatic run would silently write every historical duplicate at
        # once — on this corpus, 300 critical findings worth £28.4M, reshaping the Value
        # Found headline with nobody having looked. Review the historical set first
        # (scripts/backfill_duplicate_invoices.py, dry run by default), then set
        # DUPLICATE_INVOICE_DETECTOR_ENABLED=1 so new arrivals are flagged as they land.
        if os.environ.get("DUPLICATE_INVOICE_DETECTOR_ENABLED", "0").strip() in ("1", "true", "True"):
            try:
                from src.services.duplicate_invoice_detector import run_detector
                logger.info("downstream chain: duplicate invoices %s new finding(s)",
                            run_detector())
            except Exception:
                logger.exception("downstream chain: duplicate-invoice detection failed")
        else:
            logger.debug("downstream chain: duplicate-invoice detector disabled "
                         "(DUPLICATE_INVOICE_DETECTOR_ENABLED)")

    def _ensure_uicanvas_bridge(self) -> Optional[UicanvasBridge]:
        """Start the uicanvas → bp_sqldb process_monitor bridge.

        The UI uploader writes new procurement docs to uicanvas; the
        renovation pipeline watches bp_sqldb. Bridge transparently mirrors
        rows so end-users don't need to know which DB the pipeline reads.
        Enable/disable with UICANVAS_BRIDGE_ENABLED (default: enabled).
        """
        import os
        if os.environ.get("UICANVAS_BRIDGE_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("UicanvasBridge disabled by UICANVAS_BRIDGE_ENABLED")
            return None
        if self._uicanvas_bridge is None:
            self._uicanvas_bridge = UicanvasBridge()
            self._uicanvas_bridge.start()
        return self._uicanvas_bridge

    @classmethod
    def ensure(
        cls,
        agent_nick,
        *,
        training_endpoint: Optional[ModelTrainingEndpoint] = None,
        orchestrator: Optional[Any] = None,
    ) -> "BackendScheduler":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    agent_nick,
                    training_endpoint=training_endpoint,
                    orchestrator=orchestrator,
                )
            else:
                cls._instance._update_agent(
                    agent_nick,
                    training_endpoint=training_endpoint,
                    orchestrator=orchestrator,
                )
        return cls._instance

    def _update_agent(
        self,
        agent_nick,
        *,
        training_endpoint: Optional[ModelTrainingEndpoint] = None,
        orchestrator: Optional[Any] = None,
    ) -> None:
        if self.agent_nick is agent_nick:
            if training_endpoint is not None:
                self._training_endpoint = training_endpoint
            self._sync_training_job()
            if orchestrator is not None and orchestrator is not getattr(self, "_orchestrator", None):
                self._orchestrator = orchestrator
                try:
                    self._ensure_email_watcher_service()
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to refresh email watcher service after orchestrator update")
            return
        self.agent_nick = agent_nick
        self._relationship_scheduler = self._init_relationship_scheduler()
        if training_endpoint is not None:
            self._training_endpoint = training_endpoint
        if orchestrator is not None:
            self._orchestrator = orchestrator
        # Reinitialise the email watcher service so it reflects the new agent
        # registry/orchestrator context.
        with self._email_watcher_lock:
            watcher = self._email_watcher_service
            self._email_watcher_service = None
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop previous email watcher service")
        self._sync_training_job()
        try:
            self._ensure_email_watcher_service()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to restart email watcher service after agent update")

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="procwise-backend-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        with self._email_watcher_lock:
            watcher = self._email_watcher_service
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop email watcher service")
        with self._process_monitor_lock:
            monitor_watcher = self._process_monitor_watcher
        if monitor_watcher is not None:
            try:
                monitor_watcher.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop process monitor watcher")

    def register_job(
        self,
        name: str,
        runner: Callable[[], None],
        interval: timedelta,
        initial_delay: Optional[timedelta] = None,
        *,
        one_shot: bool = False,
        lane: str = DEFAULT_JOB_LANE,
    ) -> None:
        job = _ScheduledJob(
            name,
            runner,
            interval,
            initial_delay,
            one_shot=one_shot,
            lane=lane,
        )
        with self._lock:
            self._jobs[name] = job

    def submit_once(
        self,
        name: str,
        runner: Callable[[], None],
        *,
        initial_delay: Optional[timedelta] = None,
    ) -> None:
        """Schedule ``runner`` to execute once in the background."""

        self.register_job(
            name,
            runner,
            interval=timedelta(days=365 * 100),
            initial_delay=initial_delay,
            one_shot=True,
        )

    KG_SYNC_JOB_NAME = "kg-sync-dispatch"

    TRGT_PROMOTION_JOB_NAME = "trgt-promotion"

    DEAL_ASSIGNMENT_JOB_NAME = "deal-assignment"
    EXTRACTION_FEEDBACK_JOB_NAME = "extraction-feedback"
    STYLE_STAGING_SWEEP_JOB_NAME = "style-staging-sweep"
    PRICE_OUTLIER_JOB_NAME = "price-outlier-scan"
    ANALYSIS_SWEEP_JOB_NAME = "analysis-sweep"
    ANALYSIS_SWEEP_LANE = "analysis"

    def _register_default_jobs(self) -> None:
        self._sync_training_job()
        # Vendor-baking retired 2026-07-17: the model-sync job rebuilt AgentNick:latest with
        # customer supplier names injected into the system prompt. It targeted :latest (the live
        # pipeline runs :unified), so it never reached the live model, and the governed
        # per-request hint path (bp_prompt -> HINT_STORE -> context_layer) already carries
        # learned vendor knowledge, human-approved. See services/README or git 6622bae history.
        self._register_kg_sync_job()
        self._register_summary_precompute_job()
        self._register_trgt_promotion_job()
        self._register_deal_assignment_job()
        self._register_analysis_sweep_job()
        self._register_extraction_feedback_job()
        self._register_price_outlier_job()
        self._register_value_digest_job()
        self._register_capture_retention_job()
        self._register_style_staging_sweep_job()
        self._register_mailbox_health_job()
        self._register_style_feedback_job()

    def _register_style_staging_sweep_job(self) -> None:
        """Register the style-staging TTL sweep.

        Emails pasted for style compilation sit in proc.bp_style_ingest_staging until the
        compiler consumes them. Someone who pastes five emails and then navigates away
        leaves them there, so this deletes anything past its purge_after. Hourly is well
        inside the 24-hour default TTL, and each run writes an audit row whether or not it
        found anything — a sweep that logged only on a hit would be indistinguishable from
        one that had silently stopped running.

        Interval via STYLE_STAGING_SWEEP_INTERVAL_MINUTES (default 60).
        """
        import os
        if self.STYLE_STAGING_SWEEP_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("STYLE_STAGING_SWEEP_INTERVAL_MINUTES", "60"))
        except ValueError:
            minutes = 60
        self.register_job(
            self.STYLE_STAGING_SWEEP_JOB_NAME,
            self._run_style_staging_sweep,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
        )

    def _run_style_staging_sweep(self) -> None:
        """Purge expired style-ingest staging rows."""
        try:
            from services.style.compiler import sweep_expired_staging

            sweep_expired_staging()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Style staging sweep failed")

    MAILBOX_HEALTH_JOB_NAME = "style-mailbox-health"

    def _register_mailbox_health_job(self) -> None:
        """Register the Mode C mailbox health check.

        A permission grant revoked in the customer's tenant does not tell us so. Without
        this the platform would keep trying to read a mailbox it no longer may — and
        worse, would keep drafting against a profile derived from it. The check marks the
        binding REVOKED, which forces drafting down to the platform baseline with a
        visible flag rather than a silent generic draft.

        Interval via STYLE_MAILBOX_HEALTH_INTERVAL_MINUTES (default 60).
        """
        import os
        if self.MAILBOX_HEALTH_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("STYLE_MAILBOX_HEALTH_INTERVAL_MINUTES", "60"))
        except ValueError:
            minutes = 60
        self.register_job(
            self.MAILBOX_HEALTH_JOB_NAME,
            self._run_mailbox_health_check,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=15),
        )

    def _run_mailbox_health_check(self) -> None:
        """Re-check every active mailbox binding."""
        try:
            from services.style.health import check_all_bindings

            check_all_bindings()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Style mailbox health check failed")

    STYLE_FEEDBACK_JOB_NAME = "style-feedback-sweep"

    def _register_style_feedback_job(self) -> None:
        """Register the style feedback sweep.

        Looks for drafts written back to a bound mailbox that have since been sent, scores
        how far they drifted from what was drafted, and raises a recompile suggestion where
        the drift is sustained. It never recompiles — a human accepts the suggestion, and
        the resulting profile still has to be approved.

        Only Mode C bindings with write-back produce anything: under Mode A the platform
        never touched a mailbox and has no way to learn what was eventually sent.

        Daily by default — this is a trend, not an alert. Interval via
        STYLE_FEEDBACK_INTERVAL_MINUTES.
        """
        import os
        if self.STYLE_FEEDBACK_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("STYLE_FEEDBACK_INTERVAL_MINUTES", "1440"))
        except ValueError:
            minutes = 1440
        self.register_job(
            self.STYLE_FEEDBACK_JOB_NAME,
            self._run_style_feedback_sweep,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=30),
        )

    def _run_style_feedback_sweep(self) -> None:
        """Score sent drafts and raise recompile suggestions where drift is sustained."""
        try:
            from services.style.feedback import (
                StyleFeedbackService,
                capture_sent_drafts,
            )
            from services.style.graph_source import GraphExemplarSource
            from services.style.mailbox import MailboxBindingRepository

            service = StyleFeedbackService()
            for binding in MailboxBindingRepository().list_active():
                if not binding.can_receive_drafts:
                    continue
                try:
                    capture_sent_drafts(
                        GraphExemplarSource(binding), binding=binding
                    )
                    service.suggest_if_sustained(binding.user_ref, "_all")
                except Exception:
                    logger.exception(
                        "Style feedback sweep failed for binding %s", binding.binding_id
                    )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Style feedback sweep failed")

    def _register_trgt_promotion_job(self) -> None:
        """Register the periodic _stg -> _trgt promotion job.

        The _raw -> _stg promotion is event-driven (PromotionListenerService).
        The final _stg -> _trgt step (confidence + link-score gated, via
        linking_engine.promote_ready) had no automatic trigger and only ran on
        a manual API call, leaving _trgt empty. This job closes that gap so
        eligible staged rows flow to the final target tables on a schedule.

        Enable/disable with TRGT_PROMOTION_ENABLED (default: enabled).
        Interval via TRGT_PROMOTION_INTERVAL_MINUTES (default: 15).
        """
        import os
        if os.environ.get("TRGT_PROMOTION_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("trgt promotion job disabled by TRGT_PROMOTION_ENABLED")
            return
        if self.TRGT_PROMOTION_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("TRGT_PROMOTION_INTERVAL_MINUTES", "15"))
        except ValueError:
            minutes = 15
        self.register_job(
            self.TRGT_PROMOTION_JOB_NAME,
            self._run_trgt_promotion,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=3),
        )

    def _register_extraction_feedback_job(self) -> None:
        """Register the extraction feedback-loop proposer job.

        Scans extraction telemetry for recurring per-vendor failures and drafts
        PENDING hint proposals for human approval (propose-only — nothing is
        applied automatically). Enable/disable with EXTRACTION_FEEDBACK_ENABLED
        (default enabled); interval via EXTRACTION_FEEDBACK_INTERVAL_MINUTES
        (default 1440 = daily).
        """
        import os
        if os.environ.get("EXTRACTION_FEEDBACK_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("extraction feedback job disabled by EXTRACTION_FEEDBACK_ENABLED")
            return
        if self.EXTRACTION_FEEDBACK_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("EXTRACTION_FEEDBACK_INTERVAL_MINUTES", "1440"))
        except ValueError:
            minutes = 1440
        self.register_job(
            self.EXTRACTION_FEEDBACK_JOB_NAME,
            self._run_extraction_feedback,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=10),
        )

    def _run_extraction_feedback(self) -> None:
        """Draft pending per-vendor hint proposals from recent telemetry."""
        try:
            from src.services.extraction_feedback.proposer import propose_all
            created = propose_all()
            if created:
                logger.info("extraction feedback: created %d hint proposals", len(created))
        except Exception:
            logger.exception("extraction feedback proposer job failed")

    def _run_trgt_promotion(self) -> None:
        """Catch up stranded _raw rows, then promote confidence/link-gated _stg
        rows into _trgt. The pending catch-up here is the periodic backstop for
        any promotion NOTIFY missed by the event listener."""
        try:
            from src.services.extraction.promotion import promote_pending
            pend = promote_pending()
            if pend.get("promoted") or pend.get("failed"):
                logger.info("trgt promotion: pending-raw catch-up %s", pend)
        except Exception:
            logger.exception("trgt promotion: pending-raw catch-up failed")
        try:
            from src.services.linking_engine import promote_ready
            result = promote_ready()
            logger.info("trgt promotion completed: %s", result)
            self._sync_promoted_to_kg(result)
        except Exception:
            logger.exception("trgt promotion job failed")

    def _sync_promoted_to_kg(self, result: dict) -> int:
        """Push rows that just reached _trgt into the knowledge graph.

        This is where the per-document KG sync belongs, and it used to live in
        process_monitor_watcher — firing when dispatch returned ``promoted``,
        which is _stg promotion, not _trgt. The graph mirrors _trgt, so syncing
        at the earlier point created nodes for documents that had not reached
        final state, and the reconciling rebuild then swept them. The graph
        oscillated for exactly the documents still in flight.

        Failures never propagate. A document is durable in _trgt before this
        runs; the graph is a downstream view and can be rebuilt from _trgt at
        any time, so a KG problem must not fail the promotion job.

        KNOWN GAP: linking_engine reports purchase-order promotions in aggregate
        (`{"doc_type": "purchase_order", "promoted": n}`) with no primary keys,
        so POs promoted here are not individually synced — the full rebuild
        picks them up. Fixing that means returning the PO ids from
        `_promote_purchase_orders`, which is a change to linking_engine rather
        than to this job.
        """
        synced = 0
        for entry in (result or {}).get("details") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("action") != "promoted":
                continue
            doc_type, doc_pk = entry.get("doc_type"), entry.get("doc_pk")
            if not doc_type or not doc_pk:
                continue
            try:
                from src.services.extraction.kg_sync import sync_row_to_kg
                synced += sync_row_to_kg(self.agent_nick, doc_type, str(doc_pk))
            except Exception:
                logger.exception(
                    "KG sync failed for %s pk=%s after _trgt promotion",
                    doc_type, doc_pk,
                )
        if synced:
            logger.info("trgt promotion: %d row(s) synced to the KG", synced)
        return synced

    def _register_deal_assignment_job(self) -> None:
        """Assign documents to deals after _stg->_trgt promotion (look-forward +
        look-back + reconcile). Toggle DEAL_ASSIGNMENT_ENABLED (default on),
        interval DEAL_ASSIGNMENT_INTERVAL_MINUTES (default 15)."""
        import os
        if os.environ.get("DEAL_ASSIGNMENT_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("deal assignment job disabled by DEAL_ASSIGNMENT_ENABLED")
            return
        if self.DEAL_ASSIGNMENT_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("DEAL_ASSIGNMENT_INTERVAL_MINUTES", "15"))
        except ValueError:
            minutes = 15
        self.register_job(
            self.DEAL_ASSIGNMENT_JOB_NAME,
            self._run_deal_assignment,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
        )

    def _register_analysis_sweep_job(self) -> None:
        """Safety net for analysis events the listener path missed. Toggle
        ANALYSIS_SWEEP_ENABLED (default on), interval
        ANALYSIS_SWEEP_INTERVAL_MINUTES (default 15)."""
        import os
        if os.environ.get("ANALYSIS_SWEEP_ENABLED", "1").strip() not in ("1", "true", "True"):
            logger.info("analysis sweep job disabled by ANALYSIS_SWEEP_ENABLED")
            return
        if self.ANALYSIS_SWEEP_JOB_NAME in self._jobs:
            return
        try:
            minutes = int(os.environ.get("ANALYSIS_SWEEP_INTERVAL_MINUTES", "15"))
        except ValueError:
            minutes = 15
        self.register_job(
            self.ANALYSIS_SWEEP_JOB_NAME,
            self._run_analysis_sweep,
            interval=timedelta(minutes=max(1, minutes)),
            initial_delay=timedelta(minutes=5),
            # Its own lane. It shares deal-assignment's 5-minute initial delay and
            # 15-minute interval, so on the shared lane it came due at the same
            # instant and always lost the tie — observed live on 2026-08-01 waiting
            # 4m20s behind a deal-assignment run that did no work at all. Safe to
            # take off the shared lane because bp_analysis* is written only by
            # analysis_store, whose start() is idempotent (ON CONFLICT) and whose
            # freeze() takes FOR UPDATE — it is already built to run alongside the
            # live session listener.
            lane=self.ANALYSIS_SWEEP_LANE,
        )

    def _run_analysis_sweep(self) -> None:
        """Run the three-pass sweep. Logged only when something actually
        happened, so a quiet system stays quiet in the logs."""
        try:
            from src.services import analysis_store  # noqa: PLC0415
            result = analysis_store.sweep()
            if any(result.values()):
                logger.info("analysis sweep: %s", result)
        except Exception:
            logger.exception("analysis sweep job failed")

    VALUE_DIGEST_JOB_NAME = "value-digest"

    def _register_value_digest_job(self) -> None:
        """Email the weekly value digest.

        Registered only when it is switched on, because the alternative is a job that wakes
        every week to decide it has nothing to do. Both switches are checked again at run
        time, so turning it off does not require a restart.
        """
        from src.services.value_digest import _enabled as digest_enabled, recipients
        if not digest_enabled() or not recipients():
            logger.info("value digest job not registered "
                        "(VALUE_DIGEST_ENABLED / VALUE_DIGEST_RECIPIENTS)")
            return
        if self.VALUE_DIGEST_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.VALUE_DIGEST_JOB_NAME,
            self._run_value_digest,
            interval=timedelta(days=7),
            # Not on boot: a restart should not fire an unscheduled digest at whatever hour
            # it happens to be.
            initial_delay=timedelta(hours=1),
        )

    def _run_value_digest(self) -> None:
        try:
            from src.services.value_digest import run_weekly_digest
            run_weekly_digest(agent_nick=self.agent_nick)
        except Exception:
            # run_weekly_digest already swallows its own failures; this is the backstop for
            # anything raised before it gets that far.
            logger.exception("value digest failed")

    CAPTURE_RETENTION_JOB_NAME = "capture-retention"

    def _register_capture_retention_job(self) -> None:
        """Age out the on-disk captures of document and agent content.

        Registered unconditionally, with no enable switch. The other jobs here
        do something; this one stops something accumulating, and a retention
        sweep that can be turned off is a retention policy that quietly is not
        one. The window is tunable via CAPTURE_RETENTION_DAYS; the sweep itself
        is not optional.

        Daily, and on boot: these directories had four months of content in them
        when the job was written, so the first run has real work to do and there
        is no reason to make an operator wait a day for it.
        """
        if self.CAPTURE_RETENTION_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.CAPTURE_RETENTION_JOB_NAME,
            self._run_capture_retention,
            interval=timedelta(days=1),
        )

    def _run_capture_retention(self) -> None:
        try:
            from src.services.capture_retention import purge_all, retention_days
            removed = purge_all()
            total = sum(removed.values())
            if total:
                logger.info(
                    "capture retention: removed %d file(s) older than %d days — %s",
                    total, retention_days(),
                    ", ".join(f"{k}: {v}" for k, v in removed.items() if v),
                )
        except Exception:
            logger.exception("capture retention sweep failed")

    def _register_price_outlier_job(self) -> None:
        """Scan for extreme prices and raise them for review."""
        if not price_outlier_enabled():
            logger.info("price outlier job disabled by PRICE_OUTLIER_ENABLED")
            return
        if self.PRICE_OUTLIER_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.PRICE_OUTLIER_JOB_NAME,
            self._run_price_outlier_scan,
            interval=timedelta(minutes=price_outlier_interval_minutes()),
            initial_delay=timedelta(minutes=10),
        )

    def _run_price_outlier_scan(self) -> None:
        try:
            from src.services.db import get_conn
            from src.services.price_outlier import find_outliers, persist_findings

            with get_conn() as conn:
                with conn.cursor() as cur:
                    findings = find_outliers(cur)
                    written = persist_findings(cur, findings)
                conn.commit()
            logger.info(
                "price outlier scan: %d findings, %d new", len(findings), written)
        except Exception:
            logger.exception("price outlier scan failed")

    # Counts in the assign_deals() result that mean _trgt deals actually changed.
    _DEAL_CHANGE_KEYS = ("forward_linked", "backward_linked", "propagated",
                         "reconciled", "metadata_filled")

    def _run_deal_assignment(self) -> None:
        """Run the deal assignment passes, then chain opportunity mining when the
        deal set actually changed — closing the upload -> extract -> link -> mine
        -> dashboard loop the moment new deals are linked (no separate timer)."""
        try:
            from src.services.deal_assignment_service import assign_deals
            result = assign_deals()
            logger.info("deal assignment completed: %s", result)
        except Exception:
            logger.exception("deal assignment job failed")
            return
        try:
            self._chain_opportunity_mining(result)
        except Exception:
            logger.exception("chained opportunity mining failed")

    def _chain_opportunity_mining(self, deal_result: Any) -> None:
        """Run opportunity mining iff deal-assignment changed something. Mining is
        heavy, so it only fires when new deals were actually linked/updated.
        Set OPPORTUNITY_MINING_ENABLED=0 to disable the chain (default on)."""
        import os
        if os.environ.get("OPPORTUNITY_MINING_ENABLED", "1").strip() in ("0", "false", "False"):
            return
        if self._orchestrator is None:
            logger.debug("opportunity mining chain skipped — no orchestrator wired")
            return
        changed = isinstance(deal_result, dict) and any(
            int(deal_result.get(k) or 0) for k in self._DEAL_CHANGE_KEYS)
        if not changed:
            logger.debug("opportunity mining chain skipped — no deal changes")
            return
        logger.info("deals changed -> chaining opportunity mining")
        self._run_opportunity_mining()

    def _run_opportunity_mining(self) -> None:
        """Run opportunity mining; the miner upserts findings into bp_opportunity."""
        import os
        try:
            workflow = os.environ.get("OPPORTUNITY_MINING_WORKFLOW", "all")
            try:
                min_impact = float(os.environ.get("OPPORTUNITY_MINING_MIN_IMPACT", "100"))
            except ValueError:
                min_impact = 100.0
            result = self._orchestrator.execute_workflow(
                "opportunity_mining",
                {"workflow": workflow, "conditions": {}, "min_financial_impact": min_impact},
            )
            logger.info("opportunity mining completed: %s",
                        (result or {}).get("status") if isinstance(result, dict) else "ok")
        except Exception:
            logger.exception("opportunity mining job failed")

    def _register_kg_sync_job(self) -> None:
        """Register periodic KG sync job (startup + every 6 hours)."""
        if self.KG_SYNC_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.KG_SYNC_JOB_NAME,
            self._run_kg_sync,
            interval=timedelta(hours=6),
            initial_delay=timedelta(minutes=2),
        )

    # Document labels a healthy full rebuild must produce. If every one of these
    # is zero the run did nothing, whatever else it reports.
    _KG_DOCUMENT_LABELS = (
        "Invoice", "InvoiceLine", "PurchaseOrder", "POLine", "Quote", "QuoteLine",
    )

    @staticmethod
    def _kg_full_rebuild_enabled() -> bool:
        """Whether the SCHEDULED job may perform a full rebuild.

        OFF by default, for the same reason the duplicate-invoice detector above
        is: a full rebuild is not an increment. The graph currently holds ~600
        invoices while the _trgt tier holds 12,408, so the first run after the
        source tables were corrected would write roughly 232,000 nodes —
        38,000 documents and 194,000 line items — unattended, on a six-hour
        timer, with nobody looking.

        Set KG_FULL_REBUILD_ENABLED=1 once that first rebuild has been run
        deliberately and its result checked. The per-document path
        (extraction.kg_sync) is unaffected and keeps working either way, so
        newly promoted documents still reach the graph while this is off.
        """
        import os  # imported per-function, as everywhere else in this module

        return os.environ.get("KG_FULL_REBUILD_ENABLED", "0").strip() in (
            "1", "true", "True",
        )

    def _run_kg_sync(self) -> None:
        """Build/refresh the procurement knowledge graph."""
        if not self._kg_full_rebuild_enabled():
            logger.info(
                "KG full rebuild skipped (KG_FULL_REBUILD_ENABLED unset). "
                "Per-document sync is unaffected; run the rebuild deliberately "
                "the first time — see docs/remediation."
            )
            return
        try:
            from services.procurement_kg_builder import ProcurementKGBuilder
            builder = ProcurementKGBuilder(self.agent_nick)
            if not builder._driver:
                logger.error(
                    "KG sync skipped — Neo4j is not reachable. "
                    "Ensure Neo4j is running at the configured URI."
                )
                return
            counts = builder.build_full_graph()

            # A run that loaded no documents is a failure, not a completion.
            # This job logged "KG sync completed: {...}" with every document
            # count at zero for eleven days, because the builder was reading six
            # tables that no longer existed. A success message containing all
            # zeros is worse than an error: it answers the question nobody
            # re-asked.
            loaded = {k: counts.get(k, 0) for k in self._KG_DOCUMENT_LABELS}
            if not any(loaded.values()):
                logger.error(
                    "KG sync loaded NO documents — every one of %s came back 0. "
                    "The source tables are unreadable or empty; the graph has "
                    "not been refreshed. Counts: %s",
                    ", ".join(self._KG_DOCUMENT_LABELS), counts,
                )
            else:
                logger.info("KG sync completed: %s", counts)
            builder.close()
        except Exception:
            logger.exception("KG sync job failed")

    def _register_summary_precompute_job(self) -> None:
        """Register the daily persona-summary precompute job."""
        from config.settings import settings as _settings
        if not bool(getattr(_settings, "enable_summary_precompute", True)):
            logger.info("Summary precompute disabled; skipping job registration")
            return
        hours = int(getattr(_settings, "summary_precompute_interval_hours", 24))
        self.register_job(
            "summary-precompute",
            self._run_summary_precompute,
            interval=timedelta(hours=hours),
            initial_delay=timedelta(minutes=30),
        )

    def _run_summary_precompute(self) -> None:
        try:
            from src.services.summary_agent import precompute_summaries
            counts = precompute_summaries()
            logger.info("summary precompute completed: %s", counts)
        except Exception:
            logger.exception("summary precompute job failed")

    def _training_scheduler_enabled(self) -> bool:
        settings = getattr(self.agent_nick, "settings", None)
        return bool(getattr(settings, "enable_training_scheduler", False))

    def _sync_training_job(self) -> None:
        should_schedule = self._training_scheduler_enabled()
        has_job = self.TRAINING_JOB_NAME in self._jobs

        if should_schedule and not has_job:
            logger.info("Training scheduler enabled; registering automatic dispatch job")
            training_delay = timedelta(minutes=15)
            self.register_job(
                self.TRAINING_JOB_NAME,
                self._run_model_training,
                interval=timedelta(hours=6),
                initial_delay=training_delay,
            )
        elif not should_schedule and has_job:
            logger.info("Training scheduler disabled; removing automatic dispatch job")
            self._deregister_job(self.TRAINING_JOB_NAME)

    def _ensure_email_watcher_service(self) -> EmailWatcherService:
        registry = getattr(self.agent_nick, "agents", None)
        orchestrator = getattr(self, "_orchestrator", None)
        supplier_agent = None
        negotiation_agent = None
        getter = getattr(registry, "get", None)
        if callable(getter):
            try:
                supplier_agent = getter("supplier_interaction")
            except Exception:  # pragma: no cover - defensive
                supplier_agent = None
            try:
                negotiation_agent = getter("negotiation")
            except Exception:  # pragma: no cover - defensive
                negotiation_agent = None
        if supplier_agent is None and isinstance(registry, dict):
            supplier_agent = registry.get("supplier_interaction")
        if negotiation_agent is None and isinstance(registry, dict):
            negotiation_agent = registry.get("negotiation")
        with self._email_watcher_lock:
            if self._email_watcher_service is None:
                registry = getattr(self.agent_nick, "agents", None)
                self._email_watcher_service = EmailWatcherService(
                    agent_registry=registry,
                    orchestrator=orchestrator,
                    supplier_agent=supplier_agent,
                    negotiation_agent=negotiation_agent,
                    process_routing_service=getattr(
                        supplier_agent, "process_routing_service", None
                    ),
                )
            service = self._email_watcher_service
            updater = getattr(service, "update_dependencies", None)
            if callable(updater):
                try:
                    updater(
                        agent_registry=registry,
                        orchestrator=orchestrator,
                        supplier_agent=supplier_agent,
                        negotiation_agent=negotiation_agent,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to update email watcher dependencies")
            else:
                # Fallback for watcher stubs in tests
                if registry is not None:
                    setattr(service, "_agent_registry", registry)
                if orchestrator is not None:
                    setattr(service, "_orchestrator", orchestrator)
        service.start()
        return service

    def get_email_watcher_service(self) -> EmailWatcherService:
        """Expose the active email watcher service instance."""

        return self._ensure_email_watcher_service()

    def _ensure_process_monitor_watcher(self) -> ProcessMonitorWatcher:
        """Create and start the ProcessMonitorWatcher if not already running."""
        with self._process_monitor_lock:
            if self._process_monitor_watcher is None:
                orchestrator = getattr(self, "_orchestrator", None)
                self._process_monitor_watcher = ProcessMonitorWatcher(
                    self.agent_nick,
                    orchestrator=orchestrator,
                )
                self._process_monitor_watcher.start()
            elif self._orchestrator is not None:
                self._process_monitor_watcher.update_orchestrator(self._orchestrator)
            return self._process_monitor_watcher

    def get_process_monitor_watcher(self) -> ProcessMonitorWatcher:
        """Expose the active ProcessMonitorWatcher instance."""
        return self._ensure_process_monitor_watcher()

    def notify_email_dispatch(self, workflow_id: str) -> None:
        workflow_key = (workflow_id or "").strip()
        if not workflow_key:
            return
        service = self._ensure_email_watcher_service()
        try:
            service.notify_workflow(workflow_key)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to notify email watcher for workflow %s", workflow_key
            )

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._poll_seconds):
            self._dispatch_due(datetime.now(timezone.utc))

    def _lane_busy(self, lane: str) -> bool:
        thread = self._lane_threads.get(lane)
        return bool(thread and thread.is_alive())

    def _dispatch_due(self, now: datetime) -> None:
        """Hand every due job to its lane's worker.

        This used to run each job inline, which meant one slow job delayed every
        job after it and stalled this loop entirely — nothing else was even
        evaluated as due while it ran. Dispatching keeps the loop free, so a job
        is late only if its OWN lane is busy.
        """
        with self._lock:
            jobs_snapshot = list(self._jobs.values())
        for job in jobs_snapshot:
            if not job.due(now):
                continue
            self._dispatch(job)

    def _dispatch(self, job: _ScheduledJob) -> bool:
        """Start ``job`` unless its lane is already working.

        A job whose lane is busy is SKIPPED, not dropped: next_run is untouched,
        so it stays due and the next poll picks it up. Marking it executed here
        would silently swallow a run.
        """
        with self._lane_lock:
            if self._lane_busy(job.lane):
                return False
            thread = threading.Thread(
                target=self._execute_job,
                args=(job,),
                name=f"procwise-job-{job.lane}",
                daemon=True,
            )
            self._lane_threads[job.lane] = thread
        thread.start()
        return True

    def _deregister_job(self, name: str) -> None:
        with self._lock:
            self._jobs.pop(name, None)

    def _execute_job(self, job: _ScheduledJob) -> None:
        try:
            configure_gpu()
            job.runner()
        except Exception:
            logger.exception("Backend job %s failed", job.name)
        finally:
            executed_at = datetime.now(timezone.utc)
            job.mark_executed(executed_at)
            if job.one_shot:
                self._deregister_job(job.name)

    def _run_supplier_refresh(self) -> None:
        scheduler = self._relationship_scheduler
        if scheduler is None:
            scheduler = self._init_relationship_scheduler()
            self._relationship_scheduler = scheduler
        if scheduler is None:
            return
        try:
            scheduler.schedule_daily_refresh()
            scheduler.dispatch_due_jobs()
        except Exception:
            logger.exception("Supplier relationship refresh dispatch failed")

    def _run_model_training(self) -> None:
        endpoint = self._resolve_training_endpoint()
        if endpoint is None:
            return
        try:
            endpoint.dispatch(force=False)
        except Exception:
            logger.exception("Model training dispatch failed")

    def _init_relationship_scheduler(self):
        try:
            from services.supplier_relationship_service import SupplierRelationshipScheduler

            scheduler = getattr(self.agent_nick, "supplier_relationship_scheduler", None)
            if scheduler is None:
                scheduler = SupplierRelationshipScheduler(self.agent_nick)
                setattr(self.agent_nick, "supplier_relationship_scheduler", scheduler)
            return scheduler
        except Exception:
            logger.exception("Failed to initialise SupplierRelationshipScheduler")
            return None

    def _resolve_training_endpoint(self) -> Optional[ModelTrainingEndpoint]:
        endpoint = self._training_endpoint
        if isinstance(endpoint, ModelTrainingEndpoint):
            return endpoint
        try:
            endpoint = ModelTrainingEndpoint(self.agent_nick)
            self._training_endpoint = endpoint
            return endpoint
        except Exception:  # pragma: no cover - defensive import/initialisation
            logger.exception("Failed to initialise ModelTrainingEndpoint")
            return None
