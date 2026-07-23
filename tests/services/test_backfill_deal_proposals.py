import scripts.backfill_deal_proposals as bf


def test_backfill_delegates_to_generate(monkeypatch):
    seen = {}
    monkeypatch.setattr(bf, "_generate",
                        lambda batch, session: seen.update(batch=batch) or
                        {"batch_deal_id": batch, "proposal_ids": [1, 2], "ungrouped": [],
                         "members_with_lines": 0, "members_total": 0})
    out = bf.backfill("ANALYSISSET_19072620260719339")
    assert seen["batch"] == "ANALYSISSET_19072620260719339"
    assert out["proposal_ids"] == [1, 2]
