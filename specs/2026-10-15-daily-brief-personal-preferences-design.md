# Daily brief — scheduled morning updates, personal preferences and email (design)

Date: 2026-10-15 · Status: **awaiting review** · Builds on: `src/services/daily_brief/` (per-role brief,
pushed BP `edecac1e`, UI `spendiq-ui` `0736c81`)

## 1. What we are building, and why

Today the daily brief ("Today's brief" on Procurement Home) is configured only by an administrator:
which view a role gets, what each view shows, and the cut-offs. A person cannot shape their own.

Nick's request (2026-10-15): *"a configuration structure for the daily report that the user can
complete and then the update will be provided to the user based upon those preferences."* Delivery
ruling: **Home page and email**. Correction (same day): *"since this is a daily update the agents
only run in the morning. There can be a frequency of updates that the user applies if they want
more."*

**Success looks like:** every morning an update is produced for each person; Home shows that
update. A person opens "Customise my brief", chooses what they want to see and how much, may ask for
more updates in the day, and may have them emailed; from then on both the Home card and the email
follow those choices — never showing them anything their role is not
allowed to see.

**Assumption confirmed in chat:** the administrator's settings remain the default and the outer
bounds. A person narrows, reorders and tunes within them; they cannot widen them.

## 2. The rules this design keeps

1. **Produced on a schedule, not live.** An update is produced once each morning by default (more
   often only if the person asks, §3). It is stored; Home shows the person's latest stored update,
   labelled with when it was produced. Opening Home does not recompute it. Two exceptions, so the
   card is never blank or out of step: a person's first visit, and saving new preferences, each
   produce an update immediately.
1a. **One update, two places.** The email carries the same stored update the Home card shows. They
   cannot disagree.
2. **A person never widens their access.** Sections and numbers a person may choose are bounded by
   the administrator's settings for the view they already get. Scope (own vs everyone) is never a
   personal choice.
3. **Nothing is made up.** A preference that no longer fits the administrator's settings is dropped
   and the person is told; it is never "corrected" into something else. A missing or invalid
   administrator setting still refuses the brief (unchanged).
4. **Sent once, never twice.** Each email is claimed in the database before it is sent.
5. **Staff only.** Email goes only to addresses on the company domain, through the same permission
   check the weekly value digest uses.

## 3. What a person can set

Stored in the existing `proc.bp_user_preference` (`user_subject`, `pref_key`, `pref_value JSONB`) —
key `daily_brief`. No new preference table.

```json
{
  "version": 1,
  "sections": ["pipeline", "opportunities"],
  "numbers": {"pipeline_top_n": 10, "critical_aged_min_days": 30},
  "schedule": {
    "times": ["07:00", "13:00"],
    "days": "weekdays",
    "timezone": "Europe/London"
  },
  "email": {
    "send": "morning",
    "skip_when_unchanged": true
  }
}
```

- **`sections`** — the sections to show, in order. Each must be in the person's view's `sections`
  or its new `optional_sections` (§4). Omitted = the view's own list. An empty list is refused at save
  ("choose at least one section").
- **`numbers`** — overrides for cut-offs. Only keys the administrator lists in `person_limits` (§4)
  may appear, each within that key's range. Omitted keys use the administrator's value.
- **`schedule.times`** — when updates are produced, `HH:MM` on a 15-minute boundary, in the
  person's `timezone` (IANA; the panel fills it from the browser the first time). Omitted = one
  update at the administrator's `default_time`. The **first time is the morning update**; further
  times are the extra updates the person asked for. At most `max_per_day` entries, at least
  `min_gap_minutes` apart (§4). The panel offers presets that fill this list — *Each morning*
  (default), *Morning and afternoon*, *Every 2 hours / every 4 hours, 07:00–17:00* — and a custom
  list.
- **`schedule.days`** — `every_day` | `weekdays` (default `weekdays`).
- **`email.send`** — `off` (default) | `morning` (email the morning update only) | `every` (email
  every update).
- **`email.skip_when_unchanged`** — when true, no email is sent if the update's content is identical
  to the last one emailed to this person (§6.4).

**Reset to my team's default** deletes the row.

## 4. What the administrator adds

Three new keys in the existing `daily_brief` policy row (`settings.py` validates them like the rest;
`_KNOWN` grows):

| Key | Shape | Meaning |
|---|---|---|
| `views.<name>.optional_sections` | list of section keys | Extras a reader of this view may switch on. Same scope rule as `sections`: a company-wide section may not be offered in an `own` view. Default `[]`. |
| `person_limits` | `{number_key: [min, max]}` | Which cut-offs a person may change, and within what range. Each range must sit inside `NUMBERS[key]`. A key not listed is not personal. |
| `updates` | `{"default_time": "07:00", "default_timezone": "Europe/London", "max_per_day": 4, "min_gap_minutes": 60, "stale_after_days": 30, "keep_days": 14, "check_minutes": 15}` | The morning time everyone gets unless they choose otherwise; the most updates a person may ask for in a day and how close together; how long a saved sign-in may be used (§6.2); how long stored updates are kept; how often the producer checks (range 5–60). |
| `email_briefs` | `{"enabled": bool}` | Company switch for emailing updates. |

Seed values (migration, both DBs, with rollback): `optional_sections: []` for every view;
`person_limits` for the `*_top_n` keys `[1, 25]`, `critical_aged_min_days [0, 365]`,
`upcoming_horizon_days [1, 365]`, `supplier_email_window_days [1, 90]`, `momentum_window_days
[1, 90]`; `updates` as in the table above; `email_briefs: {"enabled": false}` — **email off until
Nick switches it on**. Scheduled morning updates on Home start as soon as this ships.

The admin settings dialog (`BriefSettings.jsx`) gains: optional sections per view, the person-limit
ranges, the `updates` values, and the email switch.

## 5. How an update is built for a person

`build(principal)` is unchanged in shape and is now called by the producer job (§6), not by the Home
request; one step is added after the view is resolved:

```
rules        = settings.load()                     # admin; refuses if invalid (unchanged)
who          = reader(principal, rules)            # view + scope (unchanged)
prefs        = preferences.load(principal.subject) # may be None
effective, dropped = preferences.apply(rules, who["view"], prefs)
```

`preferences.apply` is a pure function:

- sections = person's list filtered to `view.sections ∪ view.optional_sections`, order kept;
  if the person had no list, or nothing survives the filter, the view's own list.
- numbers = admin values, overlaid with each person value whose key is in `person_limits` and whose
  value is inside the range.
- `dropped` = every preference that did not survive, with a plain reason ("Approvals is no longer
  offered for your view").

The stored update carries `personalised: bool` and `dropped: [...]`; the panel shows `dropped` as a
note. If the preference row cannot be read, the update is built from the team default with
`personalised: false` — the person still gets an update.

## 6. Producing and delivering updates

### 6.1 The producer job

A `BackendScheduler` job `daily_brief_updates`, own lane, interval `updates.check_minutes`. Each run:

1. Select people with a sign-in snapshot (§6.2) whose next due moment — the next entry of their
   `schedule.times` (or `default_time`), in their timezone, on a day `schedule.days` allows — is
   ≤ now and later than their last claimed update.
2. For each: claim (§6.3) → rebuild principal (§6.2) → `build()` → store the update → if this update
   is to be emailed (`email.send`, `email_briefs.enabled`) and not skipped as unchanged (§6.4):
   compose → permission + domain check → send → record the email outcome.

Everyone who has opened Home gets the **morning update with no setup at all**; preferences only
change what is in it, when, how often, and whether it is emailed.

Timing uses `zoneinfo`; a time that does not exist on a clock-change day (e.g. 01:30 when clocks go
forward) is produced at the next valid minute; a time that occurs twice is produced once. If the
service was down at a due moment, the missed update is produced on the next run the same day, and
not at all after the person's next due moment (no catch-up backlog).

### 6.2 Who the update is built as

A background job has no sign-in, and there is no user table: email address, groups and role exist
only on a request. So every authenticated `GET /spendiq/brief` saves a **sign-in snapshot** under
`pref_key = 'daily_brief_identity'`: `{subject, email, username, groups, seen_at}` — nothing else
from the token.

The producer rebuilds a `Principal` from that snapshot, so role, view and "own records" resolve
exactly as they did at the person's last visit. If `seen_at` is older than `stale_after_days`, no
update is produced (outcome `paused_stale`) and on their return the panel says "Your updates were
paused while you were away" — the visit itself produces a fresh one. A role change therefore takes
effect at the person's next visit, and a person who has left stops receiving updates and email
within `stale_after_days`.

### 6.3 Never twice — and where updates are kept

New table `proc.bp_daily_brief_update`:

| column | type | |
|---|---|---|
| `update_id` | bigserial PK | |
| `user_subject` | text | |
| `due_for` | timestamptz | the scheduled moment this update answers (`now` for first-visit / preference-save / test) |
| `kind` | text | `scheduled` · `first_visit` · `preferences_saved` · `test` |
| `produced_at` | timestamptz | null until built |
| `content` | jsonb | the update exactly as Home shows it |
| `content_hash` | text | §6.4 |
| `outcome` | text | `produced` · `paused_stale` · `failed` |
| `email_outcome` | text | null · `sent` · `skipped_unchanged` · `refused` · `failed` |
| `detail` | text | reason in words; never the content |
| `attempts` | int | |

Unique index `ix_bp_daily_brief_update_once` on `(user_subject, due_for)`. An update is claimed with
`INSERT … ON CONFLICT DO NOTHING RETURNING` *before* building; no row returned = another process has
it. `failed` is retried on later runs while `attempts < 3` and before the person's next due moment.
Rows older than `keep_days` are deleted by the same job. `content` holds only what the person was
already entitled to see.

### 6.4 "Nothing has changed"

`content_hash` = SHA-256 of the update's sections with `asOf` removed, keys sorted. With
`skip_when_unchanged`, an update whose hash equals the last one *emailed* to this person is still
stored and shown on Home, but not emailed (`email_outcome = skipped_unchanged`).

### 6.5 The email itself

- Subject: "Your procurement brief — <weekday d Mon>, <HH:MM>".
- Body: real HTML (headings, short tables, links to the matching Home screen) built from the same
  section payloads `briefSections.js` words on screen. The wording is ported to a Python composer
  `daily_brief/email.py` with a test that the section headings match the UI's `en.json` keys.
  A plain-text alternative part is included.
- A section in state `unassigned` / `empty` / failed is worded exactly as on screen.
- Footer: "Change what you receive" (Home, panel open) and **"Stop these emails"** — a signed,
  single-purpose link (`GET /spendiq/brief/unsubscribe?t=…`, HMAC over subject + purpose, no sign-in
  needed) that sets `email.send = "off"` and nothing else. Updates on Home carry on.
- Sending: `EmailService.send_email`; recipient must be on the `SES_DEFAULT_SENDER` domain
  (`value_digest._external` reused, moved to a shared helper); `guardrail.authorize("email.send",
  "communicate", principal, {"purpose": "daily_brief"})` as the rebuilt principal. Refusal →
  `email_outcome = refused` with the reason.

## 7. Endpoints

| Method + path | Who | Does |
|---|---|---|
| `GET /spendiq/brief` | signed in | the person's **latest stored update** (+ `producedAt`, `nextUpdateAt`, `personalised`, `dropped`); saves the sign-in snapshot; produces one immediately only if the person has none yet |
| `GET /spendiq/brief/preferences` | signed in, own only | the person's saved preferences, what their view offers (sections, number ranges, `max_per_day`, `min_gap_minutes`), whether email is enabled company-wide, and status (`active` / `paused_stale` / last email outcome) |
| `PUT /spendiq/brief/preferences` | signed in, own only | validates against §3/§4 as a whole; refuses with reasons (422), never repairs; on success produces a fresh update (`kind = preferences_saved`, at most one per 5 minutes) |
| `DELETE /spendiq/brief/preferences` | signed in, own only | reset to team default |
| `POST /spendiq/brief/preferences/test-email` | signed in, own only | emails the person's latest update now; rate-limited to 3 per hour per person |
| `GET /spendiq/brief/unsubscribe` | signed link | §6.5 |

All writes use the existing `require_user`; the action name `brief.preferences.write` (class
`write`, own records only) is added to the action vocabulary. Nobody — Admin included — reads or
writes another person's preferences through these routes.

## 8. The panel (UI)

`ProcurementHome/MyBriefPanel.jsx`, opened by a **Customise** link on the brief card, in the
existing modal look (as `BriefSettings.jsx`):

- **Sections** — checklist of what the view offers, drag (and keyboard up/down) to reorder.
- **Detail** — a number field per entry in `person_limits`, showing the allowed range.
- **How often** — presets (*Each morning* · *Morning and afternoon* · *Every 2 hours* · *Every 4
  hours* · *Custom*), the morning time, working-day start/end for the "every N hours" presets,
  weekdays / every day, timezone (prefilled). Shows "Next update: 07:00 tomorrow".
- **Email** — Off / Morning update only / Every update (hidden with an explanation when company-wide
  email is off), "skip when nothing has changed", status line, **Send me a test now**.

The brief card itself shows "Updated 07:00 · next 13:00".
- **Reset to my team's default.**
- Save errors show the server's reasons beside the field they name.

All wording through i18n; `en.json` regenerated.

## 9. Testing and proof

- Unit: `preferences.apply` (narrowing, ordering, dropped reasons, never widening — including an
  `own` view trying to add a company-wide section); validation of the admin keys and of
  `schedule.times` against `max_per_day` / `min_gap_minutes`; due-time calculation across presets,
  weekdays, timezones and both UK clock changes; missed-run catch-up stops at the next due moment;
  claim-once under two concurrent claimers; Home reads the stored update and does not recompute;
  skip-when-unchanged; stale snapshot pauses; retention; unsubscribe link tamper-proof.
- **Break the guard on purpose**: remove the scope check in `apply` and show the widening test goes
  red; remove the unique claim and show the double-send test goes red.
- Router: own-only (person B cannot read A's), 422 reasons, rate limit.
- UI: model tests for the panel's state; render test.
- **Live proof on bp_testdb**: open Home as a signed-in user and see a first update; save
  preferences and see the card change; set a schedule time one minute ahead, run the producer and see
  exactly one stored update (and, with email on, one `sent` email to Nick's address); run it again and
  see none.

## 10. Rollout

1. Migration (both DBs, with rollback): `bp_daily_brief_update`; the new admin keys, email **off**;
   action vocabulary row.
2. Backend + UI shipped: Home shows the scheduled morning update and personalisation works
   immediately.
3. Nick switches `email_briefs.enabled` on in the admin dialog when happy with the test email.

## 11. Not in this design

- Teams/Slack delivery; SMS.
- A person creating a new kind of brief, or changing severity grades.
- An administrator editing someone else's preferences.
- Producing updates or sending email for anyone who has never opened Home (there is no identity or
  address for them until they do).
- A live, recomputed-on-open brief. Home shows the latest scheduled update by design.

## 12. Known limits

- A person's role change reaches their updates at their next visit, not instantly (§6.2).
- Between updates, Home is as fresh as the last update — a deal closed at 09:00 still shows in the
  07:00 morning update until the next one. The card says when it was produced.
- "Own records" still depends on `proc.bp_user_buyer_scope`, which has 0 rows in both DBs today;
  until buyers are assigned, a buyer's brief and email say "no deals assigned to you yet".
