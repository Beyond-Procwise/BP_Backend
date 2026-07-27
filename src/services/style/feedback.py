"""Did the draft survive the person who sent it?

A profile that produces drafts people rewrite before sending is a profile that has the
voice wrong. That is the only honest signal available here, and this module measures it.

**What is stored: a number.** How far the sent version drifted from the draft, plus the
word counts needed to interpret it later. Never the sent text — not in any mode. The
specification allows retaining it outside Modes A and C2; this does not, because a
mode-dependent exception to "we do not keep your mail" is exactly the one that gets
forgotten in a year's time. A score is enough to know a profile is wrong. The text would
only be enough to be embarrassing.

**What is never done: recompiling.** Sustained drift raises a suggestion. A human accepts
it, a fresh profile is compiled as a DRAFT, and they approve that too. Silently improving
someone's writing profile would change how their mail reads without them agreeing to it —
the same failure invariant 6 prevents at approval time, arriving through a side door.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

# A draft is "substantially rewritten" above this. Chosen so ordinary edits — a name
# filled in, a date corrected, a sentence tightened — sit below it, and a wholesale
# rewrite sits above. It is a starting point, not a finding; the first real corpus should
# move it.
DIVERGENCE_THRESHOLD = 0.35

# How many rewritten drafts before it stops being one bad day.
MIN_OBSERVATIONS = 3
DEFAULT_WINDOW_DAYS = 30

OBSERVED_MAILBOX = "mailbox"
OBSERVED_MANUAL = "manual"


def _words(text: str) -> List[str]:
    return re.findall(r"[\w'’]+", (text or "").lower())


def divergence_score(drafted: str, sent: str) -> float:
    """Word-level edit distance, normalised by the longer text. 0.0 identical, 1.0 total.

    Word-level rather than character-level on purpose: changing "30 April" to "1 May"
    should read as one edit, not seven. Normalised by the longer side so deleting half an
    email and rewriting it from scratch score similarly — both mean the draft was wrong.
    """

    a, b = _words(drafted), _words(sent)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    # Levenshtein over word lists, two rows at a time — these are emails, not genomes,
    # but there is no reason to hold an n*m matrix.
    previous = list(range(len(b) + 1))
    for i, word_a in enumerate(a, start=1):
        current = [i]
        for j, word_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (word_a != word_b),
            ))
        previous = current

    return round(previous[-1] / max(len(a), len(b)), 3)


@dataclass
class DivergenceObservation:
    """One drafted-versus-sent comparison. Carries no text."""

    draft_id: int
    user_ref: str
    intent: Optional[str]
    score: float
    drafted_words: int
    sent_words: int
    style_profile_id: Optional[int] = None
    style_profile_version: Optional[int] = None
    observed_via: str = OBSERVED_MAILBOX

    @property
    def substantially_rewritten(self) -> bool:
        return self.score >= DIVERGENCE_THRESHOLD


@dataclass
class RecompileSuggestion:
    """A prompt to a human, never an instruction to the system."""

    suggestion_id: int
    user_ref: str
    intent: str
    observation_count: int
    mean_score: float
    window_days: int
    reason: str
    status: str
    style_profile_id: Optional[int] = None
    style_profile_version: Optional[int] = None


def _audit(action_type: str, *, status: str, summary: str, details: Any = None) -> None:
    try:
        from services.agent_actions import record_action

        record_action(phase="style", action_type=action_type, agent="style_feedback",
                      status=status, summary=summary, details=details)
    except Exception:  # pragma: no cover
        logger.debug("style feedback audit write failed", exc_info=True)


class StyleFeedbackService:
    """Records divergence and raises recompile suggestions."""

    def __init__(self, conn: Optional[Any] = None) -> None:
        self._conn = conn

    def _with_conn(self, fn):
        if self._conn is not None:
            return fn(self._conn)
        with get_conn() as conn:
            result = fn(conn)
            if hasattr(conn, "commit"):
                conn.commit()
            return result

    # -- capture --------------------------------------------------------------------

    def record_divergence(
        self,
        *,
        draft_id: int,
        sent_body: str,
        observed_via: str = OBSERVED_MAILBOX,
    ) -> Optional[DivergenceObservation]:
        """Compare a sent email against the draft it came from, and store the score.

        ``sent_body`` is used and discarded. It is a parameter, never a column: this
        method is the only place the sent text exists, and it does not outlive the call.
        """

        def _load(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT body, style_user_ref, style_intent, style_profile_id, "
                "       style_profile_version "
                "  FROM proc.draft_rfq_emails WHERE id = %s",
                (draft_id,),
            )
            row = cur.fetchone()
            cur.close()
            return row

        row = self._with_conn(_load)
        if not row:
            logger.warning("No draft %s to compare against", draft_id)
            return None

        drafted_body, user_ref, intent, profile_id, profile_version = row
        if not user_ref:
            # Not a style-generated draft — someone typed it. There is no profile to hold
            # responsible, so there is nothing to learn.
            logger.debug("Draft %s has no style provenance; not scoring it", draft_id)
            return None

        score = divergence_score(drafted_body or "", sent_body or "")
        observation = DivergenceObservation(
            draft_id=draft_id,
            user_ref=user_ref,
            intent=intent,
            score=score,
            drafted_words=len(_words(drafted_body or "")),
            sent_words=len(_words(sent_body or "")),
            style_profile_id=profile_id,
            style_profile_version=profile_version,
            observed_via=observed_via,
        )

        def _store(conn):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO proc.bp_style_divergence "
                "(draft_id, user_ref, intent, style_profile_id, style_profile_version, "
                " score, drafted_words, sent_words, observed_via) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (draft_id) DO NOTHING",
                (draft_id, user_ref, intent, profile_id, profile_version,
                 score, observation.drafted_words, observation.sent_words, observed_via),
            )
            cur.close()

        self._with_conn(_store)
        logger.info(
            "Draft %s diverged %.3f from what was sent (%s -> %s words)",
            draft_id, score, observation.drafted_words, observation.sent_words,
        )
        return observation

    # -- detection ------------------------------------------------------------------

    def scope_statistics(
        self, user_ref: str, intent: str, *, window_days: int = DEFAULT_WINDOW_DAYS
    ) -> Dict[str, Any]:
        """Recent divergence for one scope."""

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*), AVG(score), MAX(style_profile_id), MAX(style_profile_version) "
                "  FROM proc.bp_style_divergence "
                " WHERE user_ref = %s AND COALESCE(intent, '_all') = %s "
                "   AND observed_at >= NOW() - make_interval(days => %s)",
                (user_ref, intent, window_days),
            )
            row = cur.fetchone()
            cur.close()
            return row

        count, mean, profile_id, profile_version = self._with_conn(_run)
        return {
            "count": int(count or 0),
            "mean_score": float(mean) if mean is not None else 0.0,
            "style_profile_id": profile_id,
            "style_profile_version": profile_version,
        }

    def suggest_if_sustained(
        self,
        user_ref: str,
        intent: str,
        *,
        window_days: int = DEFAULT_WINDOW_DAYS,
        threshold: float = DIVERGENCE_THRESHOLD,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> Optional[RecompileSuggestion]:
        """Raise a suggestion where drift is sustained. Returns None where it is not.

        "Sustained" is deliberately two conditions: enough observations, and a mean above
        the threshold. Either alone produces noise — one heavily rewritten draft is a bad
        afternoon, and twenty lightly-edited ones are a profile working correctly.
        """

        stats = self.scope_statistics(user_ref, intent, window_days=window_days)
        if stats["count"] < min_observations or stats["mean_score"] < threshold:
            return None

        reason = (
            f"{stats['count']} drafts in the last {window_days} days were edited "
            f"substantially before sending (average change {stats['mean_score']:.0%}). "
            "Recompiling from more recent emails may fit better."
        )

        def _run(conn):
            cur = conn.cursor()
            # ON CONFLICT against the partial unique index: one open suggestion per scope,
            # so a repeated sweep tells the user once rather than building a queue.
            cur.execute(
                "INSERT INTO proc.bp_style_recompile_suggestion "
                "(user_ref, intent, style_profile_id, style_profile_version, "
                " observation_count, mean_score, window_days, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (user_ref, intent) WHERE status = 'pending' DO NOTHING "
                "RETURNING suggestion_id, status",
                (user_ref, intent, stats["style_profile_id"], stats["style_profile_version"],
                 stats["count"], round(stats["mean_score"], 3), window_days, reason),
            )
            row = cur.fetchone()
            cur.close()
            return row

        row = self._with_conn(_run)
        if not row:
            return None  # one is already open

        _audit(
            "recompile_suggested", status="warning",
            summary=f"suggested recompiling the style profile for {user_ref}/{intent}",
            details={"suggestion_id": row[0], **stats, "window_days": window_days},
        )
        return RecompileSuggestion(
            suggestion_id=row[0], user_ref=user_ref, intent=intent,
            observation_count=stats["count"], mean_score=round(stats["mean_score"], 3),
            window_days=window_days, reason=reason, status=row[1],
            style_profile_id=stats["style_profile_id"],
            style_profile_version=stats["style_profile_version"],
        )

    # -- the human's decision -------------------------------------------------------

    def open_suggestion(self, user_ref: str, intent: str) -> Optional[RecompileSuggestion]:
        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT suggestion_id, user_ref, intent, observation_count, mean_score, "
                "       window_days, reason, status, style_profile_id, style_profile_version "
                "  FROM proc.bp_style_recompile_suggestion "
                " WHERE user_ref = %s AND intent = %s AND status = 'pending'",
                (user_ref, intent),
            )
            row = cur.fetchone()
            cur.close()
            return row

        row = self._with_conn(_run)
        if not row:
            return None
        return RecompileSuggestion(
            suggestion_id=row[0], user_ref=row[1], intent=row[2],
            observation_count=row[3], mean_score=float(row[4]), window_days=row[5],
            reason=row[6], status=row[7], style_profile_id=row[8],
            style_profile_version=row[9],
        )

    def resolve_suggestion(
        self, suggestion_id: int, *, status: str, actioned_by: str
    ) -> RecompileSuggestion:
        """Accept or dismiss a suggestion.

        Accepting records a decision. It does **not** recompile, and it does not activate
        anything: the user still pastes fresh emails, a DRAFT is compiled, and they
        approve it. This method moves a flag, nothing more — which is the whole point.
        """

        if status not in ("accepted", "dismissed"):
            raise ValueError("a suggestion is either accepted or dismissed")
        if not actioned_by or not str(actioned_by).strip():
            raise ValueError("resolving a suggestion requires the identity of the person")

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE proc.bp_style_recompile_suggestion "
                "SET status = %s, actioned_by = %s, actioned_at = NOW() "
                "WHERE suggestion_id = %s "
                "RETURNING suggestion_id, user_ref, intent, observation_count, mean_score, "
                "          window_days, reason, status, style_profile_id, style_profile_version",
                (status, str(actioned_by).strip(), suggestion_id),
            )
            row = cur.fetchone()
            cur.close()
            return row

        row = self._with_conn(_run)
        if not row:
            raise ValueError(f"no recompile suggestion with suggestion_id={suggestion_id}")

        _audit(
            "recompile_suggestion_resolved", status="ok",
            summary=f"suggestion {suggestion_id} {status} by {actioned_by}",
            details={"suggestion_id": suggestion_id, "status": status},
        )
        return RecompileSuggestion(
            suggestion_id=row[0], user_ref=row[1], intent=row[2],
            observation_count=row[3], mean_score=float(row[4]), window_days=row[5],
            reason=row[6], status=row[7], style_profile_id=row[8],
            style_profile_version=row[9],
        )


def capture_sent_drafts(
    reader,
    *,
    binding,
    conn: Optional[Any] = None,
    limit: int = 50,
) -> Dict[str, int]:
    """Look for drafts we wrote back that have since been sent, and score them.

    "Where available" is doing real work in the specification's phrasing. This only
    functions for a Mode C binding with draft write-back: under Mode A the platform never
    touched a mailbox, so it has no way to learn what was eventually sent, and the loop
    simply does not run. That is a limitation of the mode, not a gap in this code — and
    worth stating plainly rather than leaving someone to wonder why their scores are empty.

    The sent body is read, compared, and dropped. It is never written anywhere.
    """

    from services.style.graph_source import MailboxAccessDenied, MailboxUnreachable

    service = StyleFeedbackService(conn)

    def _pending(c):
        cur = c.cursor()
        cur.execute(
            "SELECT d.id, d.external_draft_ref "
            "  FROM proc.draft_rfq_emails d "
            "  LEFT JOIN proc.bp_style_divergence v ON v.draft_id = d.id "
            " WHERE d.external_draft_ref IS NOT NULL "
            "   AND d.style_user_ref = %s "
            "   AND v.divergence_id IS NULL "
            " ORDER BY d.id DESC LIMIT %s",
            (binding.user_ref, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return rows

    if conn is not None:
        pending = _pending(conn)
    else:
        with get_conn() as own:
            pending = _pending(own)

    tally = {"checked": 0, "still_draft": 0, "gone": 0, "scored": 0, "errors": 0}
    for draft_id, external_ref in pending:
        tally["checked"] += 1
        try:
            message = reader.get_message(external_ref)
        except (MailboxAccessDenied, MailboxUnreachable):
            # The health check owns deciding what an unreachable mailbox means. This loop
            # is not urgent enough to be worth failing over — it retries next sweep.
            tally["errors"] += 1
            continue
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not read message %s", external_ref)
            tally["errors"] += 1
            continue

        if message is None:
            # Deleted, or moved somewhere we cannot see. Nothing to learn.
            tally["gone"] += 1
            continue
        if message.get("isDraft", True):
            tally["still_draft"] += 1
            continue

        body = (message.get("body") or {}).get("content") or ""
        if not body.strip():
            continue
        if service.record_divergence(
            draft_id=draft_id, sent_body=body, observed_via=OBSERVED_MAILBOX
        ):
            tally["scored"] += 1

    if tally["checked"]:
        logger.info("Style feedback sweep for %s: %s", binding.user_ref, tally)
        _audit("feedback_sweep", status="ok",
               summary=f"checked {tally['checked']} written-back draft(s)", details=tally)
    return tally
