"""The labelling set: built deterministically, leaks nothing, validated, scored, and run through the real stage code.

Everything model-shaped here uses a FAKE model. Those results prove the plumbing and the arithmetic; they say nothing
about how good the real classifier or judge is, and the tests are written so that a fake can never be mistaken for a result.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from evals.email.labelling import build, check, live, metrics, seed

SHEETS = build.HERE / "sheets"
KEY = build.HERE / "key"
CRITERIA = build.rubrics()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("lab")
    files = build.build(d / "sheets", d / "key")
    return d / "sheets", d / "key", files


def rows(p):
    return list(csv.DictReader(Path(p).open(newline="", encoding="utf-8")))


# --- the set itself ----------------------------------------------------------------------------------------------------

def test_the_set_has_the_promised_shape():
    kinds = Counter((i, k) for _, i, k, _ in seed.REQUESTS)
    assert sum(1 for _, i, k, _ in seed.REQUESTS if i == "negotiation_counter" and k == "plain") == 20
    assert sum(1 for _, i, k, _ in seed.REQUESTS if i == "free_prompt" and k == "plain") == 20
    assert sum(1 for *_, k, _ in seed.REQUESTS if k == "ambiguous") == 10 and sum(1 for *_, k, _ in seed.REQUESTS if k == "adversarial") == 5
    assert len({r[0] for r in seed.REQUESTS}) == len(seed.REQUESTS)                       # no duplicate request
    assert Counter(d[3] for d in seed.COUNTER) == {"good": 8, "flawed": 7} and Counter(d[2] for d in seed.FREE) == {"good": 8, "flawed": 6}


def test_every_flawed_draft_names_its_flaw_and_a_real_criterion():
    for d in seed.COUNTER:
        assert (d[3] == "flawed") == bool(d[4]) == bool(d[5])
        assert set(d[5]) <= set(CRITERIA["negotiation_counter"])
    for d in seed.FREE:
        assert (d[2] == "flawed") == bool(d[3]) == bool(d[4])
        assert set(d[4]) <= set(CRITERIA["free_prompt"])


def test_the_rubric_columns_come_from_the_migration():
    text = (build.SQL / "2026-10-08_email_family_v2.sql").read_text()
    for crit in CRITERIA["negotiation_counter"] + CRITERIA["free_prompt"]:
        assert f'"{crit}"' in text


def test_build_is_deterministic(built, tmp_path):
    sheets, keys, files = built
    again = build.build(tmp_path / "s", tmp_path / "k")
    for name, p in files.items():
        assert p.read_bytes() == again[name].read_bytes(), name
    assert (keys / "judge_key.json").read_bytes() == (tmp_path / "k" / "judge_key.json").read_bytes()


def test_the_committed_sheets_and_keys_are_what_the_seed_builds():
    """Editing seed.py without rebuilding would hand out a stale set."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        build.build(t / "s", t / "k")
        for f in ("classifier_requests.csv", "judge_negotiation_counter.csv", "judge_free_prompt.csv", "FOR_LABELLERS.md"):
            assert (SHEETS / f).read_bytes() == (t / "s" / f).read_bytes(), f
        for f in ("classifier_key.json", "judge_key.json"):
            assert (KEY / f).read_bytes() == (t / "k" / f).read_bytes(), f


def test_sheets_are_blank_where_people_must_fill_and_ids_are_unique(built):
    sheets, _, _ = built
    cl = rows(sheets / "classifier_requests.csv")
    assert len(cl) == 55 and len({r["id"] for r in cl}) == 55
    assert all(r["label"] == "" and r["how_sure_1_to_5"] == "" for r in cl)
    for fam, n in (("negotiation_counter", 15), ("free_prompt", 14)):
        js = rows(sheets / f"judge_{fam}.csv")
        assert len(js) == n and len({r["id"] for r in js}) == n
        assert all(r[c + "_1_to_5"] == "" for r in js for c in CRITERIA[fam]) and all(r["overall_1_to_5"] == "" for r in js)


def test_nothing_that_gives_the_answer_is_in_what_labellers_receive(built):
    sheets, keys, _ = built
    handed_out = "\n".join(p.read_text() for p in sheets.iterdir()).lower()
    secret = ["flawed", "intended", "adversarial", "ambiguous", "expected_low"]
    secret += [d[4].lower() for d in seed.COUNTER if d[4]] + [d[3].lower() for d in seed.FREE if d[3]]
    for word in secret:
        assert word not in handed_out, word
    assert not any(p.name.endswith(".json") for p in sheets.iterdir())                 # no key file among the sheets
    assert {p.name for p in keys.iterdir()} == {"classifier_key.json", "judge_key.json"}


def test_neither_position_nor_id_reveals_which_drafts_are_flawed(built):
    sheets, keys, _ = built
    jk = json.loads((keys / "judge_key.json").read_text())
    for fam in ("negotiation_counter", "free_prompt"):
        order = [jk[r["id"]]["kind"] for r in rows(sheets / f"judge_{fam}.csv")]
        longest = max(len(list(g)) for _, g in __import__("itertools").groupby(order))
        assert longest < len(order) // 2 and order != sorted(order) and order != sorted(order, reverse=True), (fam, order)


def test_the_key_covers_every_sheet_row_exactly_once(built):
    sheets, keys, _ = built
    ck, jk = json.loads((keys / "classifier_key.json").read_text()), json.loads((keys / "judge_key.json").read_text())
    assert set(ck) == {r["id"] for r in rows(sheets / "classifier_requests.csv")}
    assert set(jk) == {r["id"] for f in ("negotiation_counter", "free_prompt") for r in rows(sheets / f"judge_{f}.csv")}
    assert {v["intended"] for v in ck.values()} <= set(build.LABELS)
    for v in jk.values():
        assert set(v["expected_low"]) <= set(CRITERIA[v["family"]])


def test_the_labeller_guide_defines_every_label_and_every_criterion():
    guide = (build.HERE / "FOR_LABELLERS.md").read_text()
    for label in build.LABELS:
        assert f"`{label}`" in guide
    for crit in set(CRITERIA["negotiation_counter"] + CRITERIA["free_prompt"]):
        assert crit in guide, crit
    # the two descriptions people read are the two the classifier is shown
    rows_ = live.classifiable_families()
    assert set(rows_) == {"negotiation_counter", "free_prompt"}


# --- the checker -----------------------------------------------------------------------------------------------------------

def _fill_classifier(sheet_rows, label="free_prompt", sure="4"):
    return [{**r, "label": label, "how_sure_1_to_5": sure} for r in sheet_rows]


def test_a_complete_classifier_sheet_passes_and_every_kind_of_mistake_is_named(built):
    sheets, _, _ = built
    base = rows(sheets / "classifier_requests.csv")
    assert check.check_classifier(_fill_classifier(base)) == []
    assert len(check.check_classifier(base)) == 110                                  # blank label and blank confidence on every row
    bad = _fill_classifier(base)
    bad[0]["label"] = "counter"                                                      # not a label
    bad[1]["how_sure_1_to_5"] = "6"
    bad[2]["how_sure_1_to_5"] = "3.5"
    bad[3]["id"] = bad[4]["id"]
    problems = "\n".join(check.check_classifier(bad))
    assert "label must be one of" in problems and "'6'" in problems and "'3.5'" in problems and "duplicate id" in problems


@pytest.mark.parametrize("score,ok", [("1", True), ("5", True), ("0", False), ("6", False), ("", False), ("x", False), ("2.5", False)])
def test_judge_scores_must_be_whole_numbers_one_to_five(built, score, ok):
    sheets, _, _ = built
    r = rows(sheets / "judge_free_prompt.csv")[:1]
    for c in CRITERIA["free_prompt"]:
        r[0][c + "_1_to_5"] = "3"
    r[0]["overall_1_to_5"] = "3"
    r[0]["completeness_1_to_5"] = score
    assert (check.check_judge(r, "free_prompt") == []) is ok


# --- the arithmetic --------------------------------------------------------------------------------------------------------

def test_kappa_known_values():
    assert metrics.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert metrics.cohen_kappa(list("aabb"), list("abbb")) == 0.5
    assert metrics.cohen_kappa(list("abab"), list("baba")) == -1.0
    assert metrics.cohen_kappa([], []) is None and metrics.cohen_kappa(list("aaaa"), list("aaaa")) is None    # nothing to say, never a made-up 1.0


def test_consensus_takes_the_majority_and_leaves_a_tie_undecided():
    c = metrics.consensus({"a": {"1": "x", "2": "x"}, "b": {"1": "x", "2": "y"}, "c": {"1": "y", "2": "z"}})
    assert c["1"]["label"] == "x" and abs(c["1"]["agreement"] - 2 / 3) < 1e-9
    assert c["2"]["label"] is None                                                    # x, y, z: no majority


def test_spearman_known_values():
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert metrics.spearman([1, 2], [1, 2]) is None and metrics.spearman([1, 1, 1], [1, 2, 3]) is None
    assert metrics.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == 1.0                         # ties get average ranks


def test_classifier_report_keeps_a_wrong_answer_an_over_ask_and_a_missed_ask_apart():
    labels = {"a": "free_prompt", "b": "free_prompt", "c": "negotiation_counter", "d": "unclear", "e": "unclear", "f": "free_prompt", "g": None}
    cap = lambda fam, asked=False, **k: {"status": "captured", "family": fam, "asked": asked, "lookup_keys": {}, "latency_s": 1.0, **k}
    preds = {"a": cap("free_prompt"), "b": cap("negotiation_counter"), "c": cap("negotiation_counter", asked=True),
             "d": cap("free_prompt", asked=True), "e": cap("free_prompt"), "f": {"status": "invalid"}, "g": cap("free_prompt")}
    r = metrics.classifier_report(labels, preds)
    assert (r["answered"], r["wrong"], r["over_ask"]) == (2, 1, 1)
    assert r["accuracy_when_answered"] == 0.5 and r["asked_on_unclear"] == 1 and r["answered_on_unclear"] == 1
    assert r["ask_recall_on_unclear"] == 0.5 and r["not_usable_output"] == 1 and r["excluded_no_consensus"] == ["g"]
    assert r["over_ask_rate"] == round(1 / 4, 3) and r["confusion"]["free_prompt"] == {"free_prompt": 1, "negotiation_counter": 1}


def test_a_report_with_nothing_to_score_says_none_not_zero():
    r = metrics.classifier_report({"a": None}, {})
    assert r["accuracy_when_answered"] is None and r["ask_recall_on_unclear"] is None and r["over_ask_rate"] is None


def test_intent_vs_labels_lists_where_the_team_and_i_differ():
    key = {"1": {"intended": "free_prompt", "kind": "plain"}, "2": {"intended": "negotiation_counter", "kind": "ambiguous"},
           "3": {"intended": "free_prompt", "kind": "plain"}, "4": {"intended": "free_prompt", "kind": "plain"}}
    r = metrics.intent_vs_labels(key, {"1": {"label": "free_prompt"}, "2": {"label": "unclear"},
                                       "3": {"label": "free_prompt"}, "4": {"label": "free_prompt"}})
    assert r["agree_rate"] == 0.75 and r["disagreements"] == [{"id": "2", "intended": "negotiation_counter", "team": "unclear", "kind": "ambiguous"}]


def _human_from_key(key, good=5, bad=2):
    return {i: {"overall": good if v["kind"] == "good" else bad} for i, v in key.items()}


def test_a_judge_that_agrees_with_the_team_and_catches_the_flawed_drafts_is_shown_to_separate(built):
    _, keys, _ = built
    key = {i: v for i, v in json.loads((keys / "judge_key.json").read_text()).items() if v["family"] == "free_prompt"}
    human = _human_from_key(key)
    model = {i: {"status": "scored", "scores": {}, "overall": h["overall"], "latency_s": 2.0} for i, h in human.items()}
    r = metrics.judge_report(human, model, key)
    assert r["rank_agreement_overall"] == 1.0 and r["controls_the_model_found"]["separates"] is True
    assert r["controls_the_model_found"]["flawed_scored_a_point_or_more_below_good"] == "6/6"


def test_a_judge_that_gives_everything_a_four_is_shown_NOT_to_separate_even_though_it_never_disagrees_by_much(built):
    _, keys, _ = built
    key = {i: v for i, v in json.loads((keys / "judge_key.json").read_text()).items() if v["family"] == "free_prompt"}
    human = _human_from_key(key)
    model = {i: {"status": "scored", "scores": {}, "overall": 4, "latency_s": 2.0} for i in human}
    r = metrics.judge_report(human, model, key)
    assert r["controls_the_model_found"]["separates"] is False and r["controls_the_team_found"]["separates"] is True
    assert r["controls_the_model_found"]["flawed_scored_a_point_or_more_below_good"] == "0/6"


def test_per_criterion_error_and_unusable_output_are_reported():
    key = {"a": {"kind": "good"}, "b": {"kind": "flawed"}, "c": {"kind": "good"}}
    human = {"a": {"x": 5, "overall": 5}, "b": {"x": 1, "overall": 1}, "c": {"x": 4, "overall": 4}}
    model = {"a": {"status": "scored", "scores": {"x": 4}, "overall": 5}, "b": {"status": "scored", "scores": {"x": 3}, "overall": 2},
             "c": {"status": "invalid", "reason": "not json"}}
    r = metrics.judge_report(human, model, key)
    assert r["compared"] == 2 and r["not_usable_output"] == ["c"]
    assert r["mean_abs_difference"] == 1.5 and r["within_one_point"] == 0.5 and r["mean_abs_difference_by_criterion"] == {"x": 1.5}


# --- the live harness, against a fake -------------------------------------------------------------------------------------------

def _classifier_fake(key, sheet):
    by_text = {r["request"]: key[r["id"]] for r in sheet}

    def ask(system, user):
        k = by_text[user]
        fam = k["intended"] if k["intended"] != "unclear" else "free_prompt"
        conf = 0.92 if k["intended"] != "unclear" else 0.4
        other = "negotiation_counter" if fam == "free_prompt" else "free_prompt"
        return json.dumps({"family_id": fam, "confidence": conf, "lookup_keys": {}, "user_instruction": user,
                           "candidates": [{"family_id": fam, "confidence": conf}, {"family_id": other, "confidence": 1 - conf - 0.05}]})
    return ask


def test_the_classifier_harness_runs_the_real_stage_over_the_sheet_and_scores_a_perfect_fake_perfectly():
    sheet = rows(SHEETS / "classifier_requests.csv")
    key = json.loads((KEY / "classifier_key.json").read_text())
    ticks = iter(range(10_000))
    preds = live.run_classifier(_classifier_fake(key, sheet), {r["id"]: r["request"] for r in sheet}, clock=lambda: next(ticks) * 0.5)
    assert len(preds) == 55 and all(p["status"] == "captured" for p in preds.values())
    assert all(p["latency_s"] == 0.5 for p in preds.values())                       # the injected clock, so latency is really measured
    r = metrics.classifier_report({i: k["intended"] for i, k in key.items()}, preds, key)
    assert r["accuracy_when_answered"] == 1.0 and r["ask_recall_on_unclear"] == 1.0 and r["over_ask_rate"] == 0.0
    assert r["not_usable_output"] == 0


def test_the_classifier_harness_counts_garbage_and_an_unoffered_family_as_unusable_not_as_wrong():
    reqs = {"R-1": "Thank Acme", "R-2": "Thank Brightline", "R-3": "Thank Northgate"}
    replies = iter(["I'm sorry, I cannot do that {",
                    json.dumps({"family_id": "rfq_batch", "confidence": 0.9, "candidates": [], "lookup_keys": {}, "user_instruction": "x"}),
                    json.dumps({"family_id": "free_prompt", "confidence": 0.9, "candidates": [], "lookup_keys": {}, "user_instruction": "Thank Northgate"})])
    preds = live.run_classifier(lambda s, u: next(replies), reqs)
    assert [preds[i]["status"] for i in reqs] == ["invalid", "invalid", "captured"]
    assert "not a configured family" in preds["R-2"]["reason"]
    r = metrics.classifier_report({i: "free_prompt" for i in reqs}, preds)
    assert r["not_usable_output"] == 2 and r["answered"] == 1 and r["wrong"] == 0


def test_the_classifier_is_offered_only_the_two_request_families_with_their_request_descriptions():
    seen = {}
    live.run_classifier(lambda s, u: seen.setdefault("system", s) and "{", {"R-1": "Thank Acme"})
    assert "negotiation_counter:" in seen["system"] and "free_prompt:" in seen["system"]
    assert "rfq_batch" not in seen["system"] and "human_written" not in seen["system"] and "Guardrails" not in seen["system"]


def test_the_judge_harness_scores_through_the_real_stage_and_a_fake_that_follows_the_key_separates():
    sheet = rows(SHEETS / "judge_free_prompt.csv")
    key = {i: v for i, v in json.loads((KEY / "judge_key.json").read_text()).items() if v["family"] == "free_prompt"}
    text_to_id = {r["email"]: r["id"] for r in sheet}

    def ask(system, user):
        k = key[text_to_id[user]]
        low = set(k["expected_low"])
        scores = {c: (1 if c in low else 5) for c in CRITERIA["free_prompt"]}
        return json.dumps({"scores": scores, "rationale": "fake"})

    model = live.run_judge(ask, "free_prompt", {r["id"]: {"email": r["email"], "facts": r["the_request"]} for r in sheet})
    assert all(m["status"] == "scored" for m in model.values())
    human = _human_from_key(key)
    r = metrics.judge_report(human, model, key)
    assert r["controls_the_model_found"]["separates"] is True


def test_the_judge_harness_reports_unusable_output_rather_than_inventing_a_score():
    model = live.run_judge(lambda s, u: "not json at all", "free_prompt", {"J-F-1": {"email": "Dear Alex, hello.", "facts": "x"}})
    assert model["J-F-1"]["status"] == "invalid" and "scores" not in model["J-F-1"]


def test_the_harness_refuses_to_run_without_live(capsys):
    assert live.main(["classifier", "--labels", "a.csv", "--out", "/tmp/never-written.json"]) == 2
    assert "refusing" in capsys.readouterr().err


def test_the_prompts_the_harness_uses_are_the_governed_ones_from_the_migration():
    p = live.prompts_from_migration()
    assert set(p) == {"email_family_classify", "email_brief_plan", "email_draft_judge"}
    assert "{families}" in p["email_family_classify"] and "{rubric}" in p["email_draft_judge"]
