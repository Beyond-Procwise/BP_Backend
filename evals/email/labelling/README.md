# Labelling set for the classifier and the judge

What it is for: two parts of the email assurance layer have only ever met a **fake** model, so their quality is unmeasured.
Both need people's judgement as the reference: **the classifier** (which kind of email does a request ask for?) and **the
judge** (how good is a draft, 1-5 per criterion?). This folder is the material for that. Nothing here is a result.

| item | what | size |
|---|---|---|
| `sheets/classifier_requests.csv` | invented buyer requests; the team picks the kind | 55 (20 + 20 clearly one kind each as I read them, 10 ambiguous, 5 adversarial) |
| `sheets/judge_negotiation_counter.csv` | draft counter emails, scored on 5 criteria | 15 (8 good, 7 deliberately flawed) |
| `sheets/judge_free_prompt.csv` | draft emails from a free-text request, 4 criteria | 14 (8 good, 6 flawed) |
| `sheets/FOR_LABELLERS.md` | the instructions, scale and definitions the team reads | |
| `key/` | my intended labels, request kinds, and which drafts are flawed and why | **never give this to a labeller** |

## Handing it out
Give each person the **`sheets/` folder only** (three CSVs and `FOR_LABELLERS.md`). Order and ids are shuffled by a fixed salt, so
an id or a position reveals nothing. Aim for **at least two people**; one rater cannot be checked against anyone.
A sheet is only usable when `python -m evals.email.labelling.check classifier <file>` (or `... judge <file> <family>`) says OK.

## Why my labels are in a separate key, not on the sheet
The requests are invented by me and the "intended" family is my hypothesis. If it were printed on the sheet the team would be
checking my work instead of giving independent judgement, and any bias of mine would become the gold. Keeping it apart gives a
second, free measurement: `intent_vs_team` lists every request where the team's consensus differs from my intent. Those are
either ambiguous requests or family definitions that need rewriting, which is worth finding out before blaming a model.
The drafts' flaws work the same way: the team scores blind, then the key shows whether **people** found the flawed ones.

## What running it tells you (needs the real model)
`python -m evals.email.labelling.live classifier --labels a.csv b.csv --out r.json --live` (and `judge --family ...`). It refuses
to run without `--live`, so a stand-in can never produce a number that looks like a result. It runs the real stage code with the
governed prompt text read from the migration files, and the drafting agent's own `ask`.

| report field | answers pending item |
|---|---|
| `kappa` | can two people even agree? A low value means the definitions are unclear, not that the model is bad |
| `accuracy_when_answered`, `confusion`, `over_ask_rate`, `ask_recall_on_unclear` | #1 classifier accuracy; low-confidence cases ask rather than guess |
| `lookup_keys_wrong_or_invented` | no invented lookup keys survive |
| `not_usable_output` | #2 the model honours the JSON format |
| `judge_vs_team.mean_abs_difference`, `within_one_point`, `rank_agreement_overall` | #5 judge calibration |
| `controls_the_model_found` vs `controls_the_team_found` | a deliberately bad draft scores low; and the controls are fair |
| `latency_s` | #8 latency |

## Limits, stated
* 55 requests and 29 drafts are enough to expose gross failures, not to rank a model to a decimal place. Treat accuracy as roughly +/- 10 points.
* The requests and drafts are invented, in one house style. Real requests would be better; add them to the sheet (same columns) when
  there are some, and keep a note of which are real.
* The criteria definitions in `FOR_LABELLERS.md` are my wording of the rubric names in the family config. Check they mean what you meant.
* Only two classifiable families exist, so this does not test a third. Adding a family is config plus new rows here.
