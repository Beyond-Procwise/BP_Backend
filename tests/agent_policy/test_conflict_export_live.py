"""The conflict history CSV through the whole app (design §3.1 Export). Needs PROCWISE_TEST_LIVE_DB=1."""
import os
from urllib.parse import quote

import pytest

from api.routers import agent_policies as R
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_history as CH
from services.agent_policy.enforcement import MASK
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.test_conflict_endpoints_live import (  # noqa: F401 - fixtures
    NOW, STRANGER, _as, _design_case, client, conn, world)
from tests.agent_policy.test_conflict_history_live import settled_live

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
HEAD = ",".join(f'"{h}"' for h in CH.HEADER)


def _row(text, case):
    [row] = [line for line in text.split("\r\n") if line.startswith(f'"pc_{case}"')]
    return row


def test_the_policy_export_masks_per_caller(client, conn, world, monkeypatch):
    a, b, live_id = settled_live(conn, world, monkeypatch)
    r = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=STRANGER)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == f'attachment; filename="conflict-history-{a}.csv"'
    assert r.text.split("\r\n")[0] == HEAD
    row = _row(r.text, live_id)
    assert f'"Paying the {MASK} refund is fine."' in row and "900" not in row
    assert '"During an action"' in row and '"Person"' in row and '"Approve"' in row
    owner = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=_as(world, world.oa)).text
    assert '"Paying the 900 refund is fine."' in _row(owner, live_id)


def test_the_pair_export_takes_the_pair_in_any_order(client, conn, world):
    a, c, did = _design_case(conn, world)
    pair = "|".join(sorted([a, c], reverse=True))
    r = client.get(f"/agent-policies/conflicts/history.csv?pair={quote(pair)}", headers=STRANGER)
    assert r.status_code == 200, r.text
    first, second = sorted([a, c])
    assert r.headers["content-disposition"] == f'attachment; filename="conflict-history-{first}_{second}.csv"'
    assert '"Waiting for a decision"' in _row(r.text, did)


def test_a_formula_in_a_reason_is_neutralised(client, conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}",
                     reason='=HYPERLINK("http://x.test","go")', limit_text=None, now=NOW)
    text = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=_as(world, world.oa)).text
    assert '"\'=HYPERLINK(""http://x.test"",""go"")"' in _row(text, did)


def test_a_date_in_a_reason_survives(client, conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}",
                     reason="Agreed with Legal on 01/04/2026 under DL/2024/001", limit_text=None, now=NOW)
    text = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=_as(world, world.oa)).text
    assert '"Agreed with Legal on 01/04/2026 under DL/2024/001"' in _row(text, did)
    assert "[withheld]" not in text


@pytest.mark.parametrize("pair", ["FIN-0001", "fin-0001|FIN-0002", "FIN-0001|FIN-0001", "FIN-0001,FIN-0002",
                                  "FIN-0001|FIN-0002|"])
def test_a_bad_pair_is_422(client, world, pair):
    r = client.get(f"/agent-policies/conflicts/history.csv?pair={quote(pair)}", headers=STRANGER)
    assert r.status_code == 422, r.text


def test_an_unknown_policy_is_404(client, world):
    assert client.get("/agent-policies/ZZQ-99999999/conflicts/history.csv", headers=STRANGER).status_code == 404
    assert client.get("/agent-policies/not-a-key/conflicts/history.csv", headers=STRANGER).status_code == 404


def test_below_viewer_is_refused(client, world, monkeypatch):
    monkeypatch.setattr(R, "_role_of", lambda p: "None")
    assert client.get("/agent-policies/conflicts/history.csv?pair=FIN-0001%7CFIN-0002",
                      headers=STRANGER).status_code == 403
    assert client.get("/agent-policies/FIN-0001/conflicts/history.csv", headers=STRANGER).status_code == 403
