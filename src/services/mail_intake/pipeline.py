"""The deterministic orchestrator. Ten stages, each with an explicit transition.

Sequencing, retries, quarantine decisions and dedup are code here, never model
judgement — there is no LLM call anywhere in this package.

Two ordering choices are load-bearing:

* **Raw MIME is stored before anything is parsed.** If parsing then fails, the
  message is still fully reconstructible and the quarantine record points at
  real bytes. Storing after parsing would lose exactly the messages that most
  need investigating.
* **Quarantined bytes go to a different prefix from cleared artifacts.** The
  extraction path is granted a key covering the artifact prefix only, so a
  quarantined document is not merely flagged — it is unreachable.

Emit is stage 10 and is deliberately indirect. The artifact's state row IS the
queue; the EventBus publish is an optimisation on top of it. The bus is
synchronous and non-durable (src/services/event_bus.py), so a process death
mid-handler would otherwise lose a document that had arrived, passed every gate
and then quietly never been read — a silent skip, which spec §1 forbids.

Spec §6.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.services.mail_intake import (
    dedup,
    hidden_text,
    mime_walk,
    repo,
    safety,
    sender_auth,
    store,
    supplier_bind,
    typing_,
)
from src.services.mail_intake.sources.base import RawMessage
from src.services.mail_intake.states import (
    ArtifactState,
    BundleState,
    QuarantineReason,
)

log = logging.getLogger(__name__)

DOCUMENT_INGESTED = "mail_intake.document_ingested"
PIPELINE_VERSION = "mail-intake-1"


@dataclass(frozen=True)
class IngestOutcome:
    bundle_id: str
    artifact_ids: Tuple[str, ...]
    quarantined: Tuple[Tuple[str, QuarantineReason], ...]
    duplicates: Tuple[str, ...]


class IngestPipeline:
    def __init__(
        self,
        object_store: Any,
        bucket: str,
        *,
        size_ceiling_bytes: int = safety.DEFAULT_SIZE_CEILING_BYTES,
        trusted_asserter: Optional[str] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._store = object_store
        self._bucket = bucket
        self._ceiling = size_ceiling_bytes
        self._trusted_asserter = trusted_asserter
        self._bus = event_bus

    # -- stage 1 ---------------------------------------------------------
    def ingest(
        self, cur: Any, binding: Mapping[str, Any], message: RawMessage
    ) -> IngestOutcome:
        owner_ref = binding["owner_ref"]
        binding_id = binding["binding_id"]

        # ORDER: idempotency check, then raw MIME, then parse. Each step earns
        # its position.
        #
        # The insert comes first so a re-delivered message does no work at all.
        # The raw store comes before the parse because a message that cannot be
        # parsed is precisely the one someone will need to look at, and an
        # earlier version had these the other way round: `parse` ran first, and
        # anything it raised other than MalformedMime escaped the handler
        # entirely. A body part declaring an unknown charset raises LookupError
        # straight out of the stdlib — ordinary in real mail — and destroyed the
        # message: no bytes stored, no bundle row, no quarantine record, nothing
        # to investigate. The ordering exists to prevent exactly that, and did
        # not.
        bundle_id = str(uuid.uuid4())
        raw_key = store.raw_mime_key(owner_ref, bundle_id)

        inserted = repo.insert_bundle(
            cur,
            bundle_id=bundle_id,
            binding_id=binding_id,
            owner_ref=owner_ref,
            provider_message_id=message.provider_message_id,
            received_at=message.received_at,
            raw_mime_key=raw_key,
        )
        if inserted is None:
            # Already ingested: a re-delivered provider event or an IMAP
            # re-poll. Stopping here is what makes re-reading safe.
            return IngestOutcome(bundle_id, (), (), ())

        self._store.put_immutable(raw_key, message.raw)

        parsed: Optional[mime_walk.ParsedBundle] = None
        parse_error: Optional[str] = None
        try:
            parsed = mime_walk.parse(message.raw)
        except Exception as exc:
            # Deliberately broad. MalformedMime is the expected failure, but the
            # stdlib parser raises its own exceptions for hostile or merely
            # sloppy input, and any of them means the same thing operationally:
            # we could not read this message, so it is quarantined with the raw
            # bytes already safe on disk.
            parse_error = f"{type(exc).__name__}: {exc}"

        header_from = None
        sender_domain = None
        if parsed is not None:
            auth_preview = sender_auth.evaluate(parsed)
            header_from = auth_preview.header_from
            sender_domain = (
                header_from.rsplit("@", 1)[1].lower()
                if header_from and "@" in header_from
                else None
            )
            repo.update_bundle_from_parse(
                cur,
                bundle_id,
                rfc_message_id=parsed.headers.get("message-id", [None])[0],
                thread_id=parsed.headers.get("thread-index", [None])[0],
                sender_address=header_from,
                sender_display_name=parsed.headers.get("from", [None])[0],
                sender_domain=sender_domain,
                recipients=parsed.headers.get("to", []),
                subject=parsed.headers.get("subject", [None])[0],
                body_text=parsed.body_text,
                body_html=parsed.body_html,
            )

        if parsed is None:
            repo.transition_bundle(
                cur,
                bundle_id,
                BundleState.RECEIVED,
                BundleState.QUARANTINED,
                reason=QuarantineReason.MALFORMED_MIME,
            )
            log.warning(
                "mail_intake bundle quarantined: malformed MIME %s",
                repo.safe_log_fields({"bundle_id": bundle_id}),
            )
            return IngestOutcome(bundle_id, (), (), ())

        repo.transition_bundle(
            cur, bundle_id, BundleState.RECEIVED, BundleState.PARSED
        )

        # -- stages 3 and 4 ---------------------------------------------
        auth = sender_auth.evaluate(parsed, trusted_asserter=self._trusted_asserter)
        bind_result = supplier_bind.bind(
            sender_domain, lambda domain: repo.supplier_candidates(cur, domain)
        )
        repo.insert_sender_auth(cur, bundle_id, auth, bind_result)
        repo.transition_bundle(
            cur, bundle_id, BundleState.PARSED, BundleState.AUTHENTICATED
        )
        repo.transition_bundle(
            cur, bundle_id, BundleState.AUTHENTICATED, BundleState.BOUND
        )

        # -- stage 5 -----------------------------------------------------
        parts = mime_walk.drop_cid_referenced(parsed)

        ready: List[str] = []
        quarantined: List[Tuple[str, QuarantineReason]] = []
        duplicates: List[str] = []

        for part in parts:
            artifact_id, reason, duplicate_of = self._ingest_part(
                cur, binding, bundle_id, part
            )
            if reason is not None:
                quarantined.append((artifact_id, reason))
            elif duplicate_of is not None:
                duplicates.append(artifact_id)
            else:
                ready.append(artifact_id)

        repo.transition_bundle(
            cur, bundle_id, BundleState.BOUND, BundleState.COMPLETE
        )
        return IngestOutcome(
            bundle_id, tuple(ready), tuple(quarantined), tuple(duplicates)
        )

    # -- stages 6 to 9 ---------------------------------------------------
    def _ingest_part(
        self, cur: Any, binding: Mapping[str, Any], bundle_id: str,
        part: mime_walk.ParsedPart,
    ) -> Tuple[str, Optional[QuarantineReason], Optional[str]]:
        owner_ref = binding["owner_ref"]
        detected = typing_.detect(part.content)
        digest = store.sha256_hex(part.content)

        artifact_id = repo.insert_artifact(
            cur,
            bundle_id=bundle_id,
            binding_id=binding["binding_id"],
            owner_ref=owner_ref,
            declared_filename=part.filename,
            declared_mime=part.declared_mime,
            detected_mime=detected,
            size_bytes=len(part.content),
            sha256=digest,
            object_key=None,
            nesting_depth=part.nesting_depth,
            disposition=part.disposition,
            content_id=part.content_id,
        )
        repo.transition_artifact(
            cur, artifact_id, ArtifactState.DETACHED, ArtifactState.TYPED
        )

        verdict = safety.evaluate(
            part.content,
            declared_mime=part.declared_mime,
            detected_mime=detected,
            filename=part.filename,
            size_ceiling_bytes=self._ceiling,
        )
        if not verdict.ok:
            return (
                artifact_id,
                self._quarantine(
                    cur, artifact_id, bundle_id, owner_ref, digest, part.content,
                    ArtifactState.TYPED, verdict.reason, verdict.detail,
                ),
                None,
            )

        hits = hidden_text.scan(part.content, detected)
        if hits:
            return (
                artifact_id,
                self._quarantine(
                    cur, artifact_id, bundle_id, owner_ref, digest, part.content,
                    ArtifactState.TYPED, QuarantineReason.HIDDEN_TEXT,
                    {
                        "techniques": sorted({h.technique for h in hits}),
                        "hit_count": len(hits),
                    },
                ),
                None,
            )

        repo.transition_artifact(
            cur, artifact_id, ArtifactState.TYPED, ArtifactState.GATED
        )

        key = store.artifact_key(owner_ref, digest)
        self._store.put_immutable(key, part.content)
        cur.execute(
            "UPDATE proc.bp_ingest_artifact SET object_key = %s WHERE artifact_id = %s",
            (key, artifact_id),
        )
        repo.transition_artifact(
            cur, artifact_id, ArtifactState.GATED, ArtifactState.STORED
        )
        repo.record_arrival(cur, artifact_id, bundle_id)

        existing = repo.find_existing_artifact(
            cur, owner_ref, digest, None, exclude_artifact_id=artifact_id
        )
        if existing and existing != artifact_id:
            repo.mark_duplicate(
                cur, artifact_id, existing, current=ArtifactState.STORED
            )
            repo.record_arrival(cur, existing, bundle_id)
            return artifact_id, None, existing

        repo.transition_artifact(
            cur, artifact_id, ArtifactState.STORED,
            ArtifactState.READY_FOR_EXTRACTION,
        )
        return artifact_id, None, None

    def _quarantine(
        self, cur: Any, artifact_id: str, bundle_id: str, owner_ref: str,
        digest: str, content: bytes, current: ArtifactState,
        reason: QuarantineReason, detail: Dict[str, Any],
    ) -> QuarantineReason:
        """Retain the bytes under the quarantine prefix, record why, and say so.

        Every quarantine is logged at WARNING with a stable, greppable prefix
        and structured fields. A held document is a thing a person has to act
        on, so it must be discoverable from the logs alone — the operator
        answering "what did we block today, and why" should not need database
        access to get a useful answer.

        The detail dict is byte-derived structural facts only (counts, content
        kinds). It never carries member names, filenames or document text, so
        it is safe to log verbatim.
        """
        key = store.quarantine_key(owner_ref, digest)
        self._store.put_immutable(key, content)
        cur.execute(
            "UPDATE proc.bp_ingest_artifact SET object_key = %s WHERE artifact_id = %s",
            (key, artifact_id),
        )
        repo.transition_artifact(
            cur, artifact_id, current, ArtifactState.QUARANTINED,
            reason=reason, detail=detail,
        )
        log.warning(
            "mail_intake.quarantine reason=%s artifact=%s bundle=%s detail=%s",
            reason.value,
            artifact_id,
            bundle_id,
            detail,
        )
        return reason


def sweep_ready_artifacts(cur: Any, event_bus: Optional[Any] = None,
                          limit: int = 50) -> int:
    """Stage 10. Announce artifacts sitting in the queue. Announce only.

    The durable row is the contract; this is the notification. An artifact
    stays in READY_FOR_EXTRACTION until a consumer moves it, so this sweep
    re-announces the same artifact every time it runs until one does — which is
    the intended behaviour for a safety net, and is why the notification must
    be cheap and repeatable.

    It therefore writes NOTHING. An earlier version appended a derivation row
    per artifact per pass, which on a 60-second job meant one row a minute per
    ready artifact, forever, growing without bound while no consumer existed.
    A derivation records that an extraction RUN happened; this function does not
    run extraction, so it has no business claiming one did. The consumer that
    performs the extraction appends the derivation and transitions the artifact
    out of the queue.
    """
    rows = repo.claim_ready_artifacts(cur, limit=limit)
    published = 0
    for row in rows:
        payload = {
            "artifact_id": row["artifact_id"],
            "bundle_id": row["bundle_id"],
            "object_key": row["object_key"],
            "detected_mime": row["detected_mime"],
            "owner_ref": row["owner_ref"],
        }
        if event_bus is not None:
            event_bus.publish(DOCUMENT_INGESTED, payload)
        published += 1
    return published
