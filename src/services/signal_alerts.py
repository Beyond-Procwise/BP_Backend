"""Alerts on /health's output signals.

/health reports when triage last finished, how many deals it failed, and how
many points the document index holds. Those exist because findings writes were
silently rejected for 65 days and the index sat empty for five weeks; a signal
nobody watches is the same silence. This reads /health the way an outside probe
would (so procwise not answering is itself an alert), and mails only when an
alert starts or clears.

Runs every five minutes from scripts/health_alerts.py under the user-level
systemd timer procwise-health-alerts.timer (deploy/systemd-user/). Not from
bp-extraction-health: that timer has not fired since May, and reviving it also
revives its stuck-row resets, which is a separate decision. Nothing is mailed
until HEALTH_ALERT_RECIPIENTS is set; until then every pending alert is logged
on each run, and the first mail after it is set carries everything active.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# Triage has run up to ~36h apart on this host (2026-10-05 18:12 -> 10-07 05:48),
# so a day would page on a normal gap. Two days means a missed run, not a slow one.
DEFAULT_TRIAGE_MAX_AGE = timedelta(hours=48)


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def evaluate(health: Optional[dict], now: datetime, *,
             triage_max_age: timedelta = DEFAULT_TRIAGE_MAX_AGE) -> dict[str, str]:
    """The alerts active in this /health body, as {key: message}.

    A signal that is missing or unreadable is an alert, never a pass: the whole
    point is that "could not tell" must not look like "fine"."""
    if health is None:
        return {"api_down": "procwise is not answering /health."}

    alerts: dict[str, str] = {}

    triage = health.get("last_triage_run", "unavailable")
    if triage is None:
        alerts["triage_never_ran"] = "Triage has never finished a run."
    elif not isinstance(triage, dict):
        alerts["triage_unreadable"] = "The last triage run cannot be read from the database."
    else:
        finished = _parse_ts(triage.get("finished_at"))
        if finished is None:
            alerts["triage_unreadable"] = "The last triage run has no readable finish time."
        elif now - finished > triage_max_age:
            hours = int((now - finished).total_seconds() // 3600)
            alerts["triage_stale"] = (
                f"Triage last finished {hours} hours ago ({finished.isoformat()}); "
                f"the limit is {int(triage_max_age.total_seconds() // 3600)} hours.")
        failed = triage.get("failed_deals") or 0
        if failed:
            alerts["triage_failed_deals"] = f"The last triage run failed {failed} deal(s)."

    store = health.get("vector_store")
    points = store.get("points") if isinstance(store, dict) else "unavailable"
    if not isinstance(points, int):
        alerts["index_unreadable"] = "The document index's point count cannot be read."
    elif points == 0:
        alerts["index_empty"] = "The document index holds 0 points; document search returns nothing."

    return alerts


def _load_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001  (missing or corrupt: start from nothing)
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _compose(active: dict[str, str], raised: list[str], cleared: list[str]) -> tuple[str, str]:
    if raised:
        subject = f"[ProcWise] ALERT: {', '.join(raised)}"
    else:
        subject = f"[ProcWise] cleared: {', '.join(cleared)}"
    lines = []
    if raised:
        lines.append("Raised:")
        lines += [f"  - {k}: {active[k]}" for k in raised]
    if cleared:
        lines.append("Cleared:")
        lines += [f"  - {k}" for k in cleared]
    still = sorted(set(active) - set(raised))
    if still:
        lines.append("Still active:")
        lines += [f"  - {k}: {active[k]}" for k in still]
    lines.append("")
    lines.append("Source: GET /health on the procwise host, checked every 5 minutes.")
    return subject, "\n".join(lines)


def run(*, fetch: Callable[[], Optional[dict]], send: Callable[[str, str], bool],
        state_path: Path, now: Optional[datetime] = None,
        triage_max_age: timedelta = DEFAULT_TRIAGE_MAX_AGE) -> dict:
    """Evaluate once; mail the changes since the last notified state.

    State advances only when the mail went (or there was nothing to send), so a
    failed send is retried on the next run instead of being lost."""
    now = now or datetime.now(timezone.utc)
    active = evaluate(fetch(), now, triage_max_age=triage_max_age)
    previous = set(_load_state(state_path).get("active", {}))
    raised = sorted(set(active) - previous)
    cleared = sorted(previous - set(active))

    notified = False
    if raised or cleared:
        subject, body = _compose(active, raised, cleared)
        try:
            notified = bool(send(subject, body))
        except Exception:  # noqa: BLE001
            log.exception("health alert: send failed")
            notified = False
    if notified or not (raised or cleared):
        _save_state(state_path, {"active": active, "updated_at": now.isoformat()})

    return {"active": sorted(active), "raised": raised, "cleared": cleared,
            "notified": notified}


# --- wiring for the timer ---------------------------------------------------

def _fetch_health(url: str) -> Optional[dict]:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.load(resp)
    except Exception as exc:  # noqa: BLE001
        log.warning("health alert: %s unreachable: %s", url, exc)
        return None


def recipients() -> list[str]:
    raw = os.environ.get("HEALTH_ALERT_RECIPIENTS", "")
    return [r.strip() for r in raw.replace(";", ",").split(",") if r.strip()]


def _send_email(subject: str, body: str) -> bool:
    to = recipients()
    if not to:
        # Not configured: the change is still emitted to the journal by the
        # caller. Returning False keeps it pending, so the first mail after
        # recipients are set carries everything still active.
        log.warning("health alert: HEALTH_ALERT_RECIPIENTS not set; not mailed: %s", subject)
        return False
    from types import SimpleNamespace

    from config.settings import Settings
    from src.services.email_service import EmailService

    settings = Settings()
    sender = getattr(settings, "ses_default_sender", None) or "noreply@procwise.co.uk"
    result = EmailService(SimpleNamespace(settings=settings)).send_email(
        subject=subject, body=body, recipients=to, sender=sender)
    return bool(getattr(result, "success", result))


def run_from_env() -> dict:
    url = os.environ.get("HEALTH_ALERT_URL", "http://127.0.0.1:8000/health")
    state = Path(os.path.expanduser(os.environ.get(
        "HEALTH_ALERT_STATE_FILE", "~/.local/state/procwise/health_alerts.json")))
    hours = float(os.environ.get("HEALTH_ALERT_TRIAGE_MAX_AGE_HOURS", "48"))
    return run(fetch=lambda: _fetch_health(url), send=_send_email, state_path=state,
               triage_max_age=timedelta(hours=hours))
