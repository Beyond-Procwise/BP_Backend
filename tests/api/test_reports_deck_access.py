"""The deck download and `readable_deck` are one rule set, so they refuse identically.

`readable_deck` is what 'learn a style from this report' calls. If it ever re-implemented the
download's rules, a report a person may not open could become a pack they can read.
"""
import pytest
from fastapi import HTTPException

from api.routers import reports as rr
from tests.api.test_reports_router import BODY, _P, _awaiting, _release, client  # noqa: F401


def _refusal(call):
    with pytest.raises(HTTPException) as e:
        call()
    return e.value.status_code, e.value.detail


def _job(client):
    client.post("/reports/generate", json=BODY)
    return "rpt-1"


def test_a_released_deck_is_readable_and_is_the_stored_bytes(client):
    job_id = _job(client)
    _release(client.store, job_id)
    job, content, media_type, filename = rr.readable_deck(job_id, _P())
    assert content == b"PK-deck-bytes" and filename.endswith(".pptx") and job["job_id"] == job_id


@pytest.mark.parametrize("status", ["queued", "running", "blocked", "failed"])
def test_a_job_that_was_not_released_has_no_readable_deck(client, status):
    job_id = _job(client)
    client.store.jobs[job_id]["status"] = status
    assert _refusal(lambda: rr.readable_deck(job_id, _P()))[0] == 409


def test_an_unknown_job_is_404(client):
    assert _refusal(lambda: rr.readable_deck("rpt-nope", _P()))[0] == 404


def test_a_released_job_with_no_stored_deck_is_409_not_500(client):
    job_id = _job(client)
    _release(client.store, job_id)
    client.store.decks.pop(job_id)
    assert _refusal(lambda: rr.readable_deck(job_id, _P()))[0] == 409


def test_the_download_and_readable_deck_refuse_an_awaiting_report_identically(client):
    _awaiting(client)
    via_download = client.get("/reports/jobs/rpt-1/deck")
    code, detail = _refusal(lambda: rr.readable_deck("rpt-1", _P()))
    assert (via_download.status_code, via_download.json()["detail"]) == (code, detail)


def test_a_signed_off_deck_that_changed_is_refused_by_both(client):
    _awaiting(client)
    client.signoff.states["rpt-1"] = {"required": True, "state": "signed_off",
                                      "deck_sha256": "0" * 64}
    via_download = client.get("/reports/jobs/rpt-1/deck")
    code, detail = _refusal(lambda: rr.readable_deck("rpt-1", _P()))
    assert code == 409 and "does not match" in detail
    assert (via_download.status_code, via_download.json()["detail"]) == (code, detail)


def test_a_reviewer_may_read_an_awaiting_deck_only_when_review_is_what_they_are_doing(client):
    _awaiting(client)
    client.signoff.may = True            # someone who may sign it off
    assert rr.readable_deck("rpt-1", _P())[1] == b"PK-deck-bytes"            # review: allowed
    code, detail = _refusal(lambda: rr.readable_deck("rpt-1", _P(), allow_review=False))
    assert code == 409 and "awaiting sign-off" in detail
