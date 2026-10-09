"""How the email agent is doing, by family, over time (read-only).

One row per (period, family): how many drafts, how many were sent or abandoned, how far people
edited the ones they sent, how often the facts disagreed, and how often the judge flagged a draft. NULL stays NULL: a draft whose
conflicts were never captured is left out of the conflict rate rather than counted as "no conflict".
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

_BUCKETS = ("day", "week", "month")

_SQL = """
SELECT date_trunc('{bucket}', c.captured_at)                                           AS period,
       c.family_id                                                                     AS family_id,
       count(*)                                                                        AS drafts,
       count(*) FILTER (WHERE s.capture_id IS NOT NULL)                                AS sent,
       count(*) FILTER (WHERE a.capture_id IS NOT NULL AND s.capture_id IS NULL)       AS abandoned,
       round(avg(s.edit_distance), 3)                                                  AS mean_edit_distance,
       count(*) FILTER (WHERE s.edit_class = 'fact')                                   AS fact_edits,
       count(*) FILTER (WHERE jsonb_typeof(c.conflicts) = 'array')                     AS conflicts_captured,
       count(*) FILTER (WHERE jsonb_typeof(c.conflicts) = 'array'
                          AND jsonb_array_length(c.conflicts) > 0)                     AS with_conflicts,
       count(*) FILTER (WHERE c.assurance_status = 'needs_review')                     AS needs_review,
       count(*) FILTER (WHERE jsonb_typeof(c.judge -> 'review_flag') = 'object')       AS judge_flagged,
       count(*) FILTER (WHERE jsonb_typeof(c.judge -> 'review_flag') = 'object'
                          AND s.capture_id IS NOT NULL)                                AS judge_flagged_sent,
       count(*) FILTER (WHERE jsonb_typeof(c.judge -> 'review_flag') = 'object'
                          AND s.capture_id IS NOT NULL AND s.edit_distance = 0)        AS judge_flagged_sent_unedited
  FROM email_agent.bp_draft_capture c
  LEFT JOIN email_agent.bp_draft_outcome s ON s.capture_id = c.capture_id AND s.outcome = 'sent'
  LEFT JOIN email_agent.bp_draft_outcome a ON a.capture_id = c.capture_id AND a.outcome = 'abandoned'
 WHERE (%(since)s::timestamptz IS NULL OR c.captured_at >= %(since)s)
   AND (%(family)s::text IS NULL OR c.family_id = %(family)s)
 GROUP BY 1, 2
 ORDER BY 1 DESC, 2
"""


def _rate(part: int, whole: int) -> Optional[float]:
    return round(part / whole, 3) if whole else None


def by_family(conn: Any, *, bucket: str = "week", since: Optional[datetime] = None,
              family: Optional[str] = None) -> List[Dict[str, Any]]:
    if bucket not in _BUCKETS:                      # interpolated into SQL, so only a known word gets in
        raise ValueError(f"bucket must be one of {_BUCKETS}")
    with conn.cursor() as cur:
        cur.execute(_SQL.format(bucket=bucket), {"since": since, "family": family})
        rows = cur.fetchall()
    out = []
    for period, fam, drafts, sent, abandoned, edit, fact_edits, cap, with_c, review, flagged, f_sent, f_unedited in rows:
        out.append({
            "period": period.isoformat() if period else None, "family_id": fam, "drafts": drafts,
            "sent": sent, "abandoned": abandoned,
            "mean_edit_distance": float(edit) if edit is not None else None,
            "fact_edit_rate": _rate(fact_edits, sent),
            "fact_conflict_rate": _rate(with_c, cap), "conflicts_captured": cap,
            "needs_review_rate": _rate(review, drafts),
            # The judge's advisory flag (any criterion <= 2). A flagged draft a person sent with NO edit is the
            # likeliest false flag; the unedited rate is an upper bound on the false-flag rate, not the rate itself.
            "judge_flagged": flagged, "judge_flag_rate": _rate(flagged, drafts),
            "judge_flagged_sent": f_sent, "judge_flagged_sent_unedited": f_unedited,
            "judge_flag_unedited_rate": _rate(f_unedited, f_sent),
        })
    return out
