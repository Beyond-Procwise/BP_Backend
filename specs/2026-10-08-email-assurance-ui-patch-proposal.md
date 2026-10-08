# PROPOSAL (not applied): email-draft assurance review in the compose modal

Target repo: beyond_procwise_ui, branch spendiq-ui, `src/modules/EmailCompose/`.
Nothing here has been written to that repo. Every snippet is illustrative.

## 0. What I found that shapes the patch (read first)

1. **The compose service is 100% mock.** `composeService.js` has no network call. Drafts live in
   an in-memory Map keyed by `context.key`; the envelope has `draft_id: "drf_<key>"` and **no
   `unique_id`**. Every endpoint in the brief is keyed by `unique_id`. Until the real draft
   envelope carries `unique_id` (see section 8, item 1), none of this can fire. The patch must
   therefore be inert when `draft.unique_id` is absent (same "honestly inert" stance the file
   already takes).
2. **The window bridges are the wrong transport here.** `__SPENDIQ_API_AI__` / `_AI_POST__` are
   assigned in a `useEffect` inside `SpendIQ/index.jsx` (about lines 490-510). The compose modal
   is also opened from Procurement Home (it has an `atHome` branch), where SpendIQ is never
   mounted, so the bridges would be undefined. Recommendation: call `axios` directly against
   `AI_API` (exported from `modules/ProcurementHome/useHomeData.js` and
   `modules/SpendIQ/data/endpoints.js`). That is exactly what the bridges wrap
   (`axios.post(`${AI_API}${path}`, body).then(r => r.data)`).
3. **Auth/identity.** `services/api.js setAuthToken()` sets `axios.defaults.headers.common`
   `Authorization: Bearer <idToken>` and `x-customer-id` at login; a response interceptor
   refreshes on 401. A direct `axios` call inherits both, same as the bridges. The client sends
   NO identity field: `accountability.initiated_by/reviewed_by` must be derived server-side from
   the token (P8 `require_user`). Do not add `reviewed_by` to any request body.
4. **Compose is not on the midnight canvas.** `midnight.css` governs the SpendIQ canvas;
   `emailCompose.css` is portalled to `document.body` and resolves colour only through app
   tokens in `index.css` (`--brand --good --warn --violet --ink* --surface* --line* *-soft
   *-ink`). New CSS must use those tokens only, no hex literals (the file header says so).
5. **Client formats no figure** (BACKEND_CONTRACT 2.2; `composeSurface.contract.test.js` fails on
   `toLocaleString`). So `facts[k].value`, `conflicts[].postgres/supplied`, `changed[].was/now`
   and `confidence` must be shown as backend-rendered strings (see 8.4, 8.5).

## 1. File plan

| File | Change |
|---|---|
| `EmailCompose/assuranceService.js` | NEW. 4 functions, direct axios (see 3). |
| `EmailCompose/assuranceDerive.js` | NEW. Pure derivations: unresolved count, gate reason, fact-by-ref map. Unit-testable, no React. |
| `EmailCompose/AssurancePanel.jsx` | NEW. `BriefPanel`, `AssumptionList`, `ConflictBanner`, `ChangeConfirm` (small, presentational). |
| `EmailCompose/EmailComposeModal.jsx` | EDIT. State, effects, send flow, abandon, token marking. |
| `EmailCompose/composeDerive.js` | EDIT. `sendGate` gets an optional `assurance` argument. |
| `EmailCompose/EmailComposeHost.jsx` | EDIT. Abandon the previous draft when a different key replaces it. |
| `EmailCompose/emailCompose.css` | EDIT. Append a `pq-as-*` block built from existing tokens. |
| `lib/i18n/strings.email.js` + `locales/en.json` | EDIT. New `email.assure.*` keys; regenerate en.json. |
| `EmailCompose/fixtures.js`, `composeService.js` | EDIT (mock only): add `unique_id` + a mock assurance record so the preview and smoke tests render. |
| `EmailCompose/BACKEND_CONTRACT.md` | EDIT. Add the 4 endpoints (it is the handoff doc). |

## 2. State added to `EmailComposeModal`

```js
// PROPOSAL
const [assurance, setAssurance] = useState(null);     // GET body, or null
const [assureState, setAssureState] = useState('idle');// idle|loading|ready|failed|inert
const [briefOpen, setBriefOpen] = useState(false);     // collapsed by default (requirement 1)
const [confirming, setConfirming] = useState(null);    // {id, action} in flight
const [editing, setEditing] = useState(null);          // {id, value}
const [change, setChange] = useState(null);            // preflight result awaiting ack
const [changeAck, setChangeAck] = useState(false);     // checkbox in the change panel
const sentRef = useRef(false);                         // true once send succeeded
const abandonedRef = useRef(false);
```

`assureState` mirrors the existing `checksState` discipline: `stale` is not "ok". While
loading/failed the send gate is CLOSED (fail closed, as `sendGate` already does for checks).
`inert` = no `draft.unique_id` (mock/legacy draft): no calls, no new gate.

### Loading and refresh
- After the draft loads (inside the existing `useEffect([key])`, after `setDraft`), call
  `getAssurance(d.unique_id)`.
- Refresh after every mutation, because facts/conflicts change when the body, tone, deadline or
  attachments change. There are two places to hook, both already central: `runPreflight()` and
  the debounced chain inside `queue()`. Add `refreshAssurance()` next to each. Set
  `assureState='loading'` the instant a mutation starts (like `setChecksState('stale')`).
- Race guard: a monotonically increasing `reqRef`; ignore a response whose number is stale.

## 3. API calls (assuranceService.js)

```js
// PROPOSAL
import axios from 'axios';
import { AI_API } from '../ProcurementHome/useHomeData';
const base = (id) => `${AI_API}/drafts/${encodeURIComponent(id)}`;
export const getAssurance   = (id)       => axios.get(`${base(id)}/assurance`).then(r => r.data);
export const confirmAssumptions = (id, confirmations) =>
  axios.post(`${base(id)}/assumptions/confirm`, { confirmations }).then(r => r.data);
export const preflightDraft = (id)       => axios.post(`${base(id)}/preflight`, {}).then(r => r.data);
export const abandonDraft   = (id, reason) =>
  axios.post(`${base(id)}/abandon`, reason ? { reason } : {}).then(r => r.data);
```

- Auth: inherited from `axios.defaults.headers.common` (Bearer + x-customer-id). No per-call header.
- Name collision: `composeService.preflight(key, docs)` already exists and returns `{checks}`.
  Do NOT reuse the word. Hence `preflightDraft` (the "did the facts change" check) and a
  UI label "change check". BACKEND_CONTRACT 2.5 calls `POST .../preflight` the check-strip
  endpoint; the new endpoint shares the verb and path suffix (see 8.2).
- Errors: axios rejects with `err.response.status` and `err.response.data.detail`. Show only
  `detail` if it is a string, else the i18n generic. Never concatenate raw JSON. The
  output-safety filter may return `"[withheld]"` in a field; render it as-is, do not
  special-case it.
- `assurance` responses replace state wholesale (POST confirm returns the same shape).

## 4. The five display items

### 4.1 Collapsed brief panel
- Where: in the references **rail**, as the first block under `.pq-rail-h` (the rail is already
  "your working context, NOT SENT", the right semantic home). Because the rail can be hidden and
  collapses under the message below 900px, ALSO render a one-line summary chip in the footer row
  (`pq-foot`, next to the "References" button): "Brief · 2 to confirm". Clicking it opens the rail
  and expands the brief (`setRailOpen(true); setBriefOpen(true)`).
- Structure (copy the `.pq-checks-bar` / `.pq-checks-tog` pattern, which is the existing
  collapsible):
```jsx
// PROPOSAL (AssurancePanel.jsx)
<section className="pq-as-brief" aria-label={t('email.assure.briefTitle')}>
  <button type="button" className="pq-as-bar" aria-expanded={open} aria-controls="pq-as-brief-body" onClick={toggle}>
    <span className={`pq-pill ${pillTone}`}>{statusLabel}</span>
    <span className="pq-as-sum">{t('email.assure.briefTitle')}</span>
    <span aria-hidden="true">{open ? '⌃' : '⌄'}</span>
  </button>
  {open && (<div id="pq-as-brief-body" className="pq-as-body">
    {/* goal, key_points[], explicit_ask, deadline, tone_rationale, risks_to_avoid[],
        reasoned{} rows, missing[], accountability line */}
  </div>)}
</section>
```
- Fields render as `<dl>`: label in `var(--font-mono)` 10px uppercase (same as `.pq-prov-h`),
  value in 12px `--ink-2` (same as `.pq-ref-detail`). Lists as `<ul>`.
- Each `reasoned[key]`: row = human label + `value` + a nested `<ul>` of `basis[]`. `key` is a
  machine key; map it with `t('email.assure.reasoned.'+key)` with fallback to the key with
  underscores replaced by spaces (use `tOr`, already used in `lib/valueOutcome.js`). `confidence`
  is shown as a `.pq-tag` (see 8.4).
- `missing[]`: heading "Not known" + list. These are things the agent could not establish.
- Accountability: one muted line, `initiated_by` and `reviewed_by` (or "Not yet reviewed").
- Status chip tone: verified -> `good`, needs_review -> `warn`, unassured -> no tone class
  (default `.pq-pill`). Reuse `.pq-pill`, no new pill.
- `violations[]` and `judge`: list under the brief as `.pq-check` rows (reuse `.pq-check.fail`
  and `.pq-check-ico`), severity text from backend, `judge.status` + `overall` as plain text.

### 4.2 Assumption confirmation (gate)
- Where: **in the message column**, between `.pq-fields` and the body (inside `.pq-msg`), shown
  only when at least one assumption is unresolved, so it cannot hide behind a closed rail.
  Resolved assumptions live in the brief panel (read-only, showing `resolution`).
- Row: `text`, then three buttons using the existing `.pq-btn` at the small size used by
  `.pq-warn-inline .pq-btn` (Confirm / Edit / Reject). Edit swaps in an `<input class="pq-finput">`
  plus Save/Cancel.
- Calls (one assumption per click, so a failure is attributable):
```js
// PROPOSAL
confirmAssumptions(draft.unique_id, [{ id, action: 'confirm' }])
confirmAssumptions(draft.unique_id, [{ id, action: 'edit', value }])
confirmAssumptions(draft.unique_id, [{ id, action: 'reject' }])
  .then(setAssurance).catch(showError).finally(() => setConfirming(null));
```
- Gate: `assuranceDerive.unresolved(assurance)` = `brief.assumptions.filter(a => !a.resolution
  || a.resolution === 'pending')`. Extend `sendGate`:
```diff
-export function sendGate({ checksState, checks = [], orphans = 0, result = null }) {
+export function sendGate({ checksState, checks = [], orphans = 0, result = null, assurance = null }) {
   if (result) return t('email.gate.sent');
   if (orphans) return t('email.gate.orphans');
+  const a = assuranceGate(assurance);            // from assuranceDerive.js, null when fine/inert
+  if (a) return a;                               // "2 assumptions need your answer before this can be sent."
   if (checksState === 'stale') ...
```
  Additive optional arg, so existing callers/tests keep working. The reason reaches the Send
  button through the existing `title={sendBlockedBy || sendTip}` and, because a `title` alone is
  weak for keyboard/screen-reader users, also as visible text in the section header
  ("2 of 3 assumptions still need an answer", `role="status"`, `aria-live="polite"`).
- While `confirming` is set, disable that row's three buttons (`disabled`, `aria-busy`).
- After a Reject the backend may redraft; if `body_template` changed the response is only the
  assurance record, so call `getDraft(context)` + `setDraft` + `paintedRef.current=null` when
  the response's `unique_id` version moved (open question 8.6).

### 4.3 Conflicts before the draft is trusted
- Where: a banner at the TOP of `.pq-msg`, above `.pq-fields`, so it is read before the body.
  Style = `.pq-blocked` (warn-soft, `role="alert"`), the existing "this draft is blocked" look.
- Each conflict row: `fact` label, then two columns "Postgres: {postgres}" / "You supplied:
  {supplied}", then the line "Postgres value used" (en string for `resolution`; see 8.7). Use
  `.pq-ref-detail` typography and `.pq-tag.share` for "Used" on the Postgres side, `.pq-tag.internal`
  for the supplied side. Wording must say the system record wins; the user is informed, not asked
  (no button), because the contract says "Postgres wins".
- Does not by itself gate Send (the conflict is already resolved by the backend), but it counts as
  "needs review" visually; `status` from the backend remains the authority for the pill.

### 4.4 Unverified-fact flags
- Source of truth: `facts[k].source` in {`carried_unverified`, `user_asserted`} plus
  `unverified_figures[]`.
- Inline in the body: tokens are painted imperatively and React must not own them. Follow the
  existing pattern used for aria-labels: a separate effect that touches classes IN PLACE, never
  `paintBody()`:
```js
// PROPOSAL, next to the tokenAria effect
useEffect(() => {
  const el = bodyRef.current; if (!el) return;
  el.querySelectorAll('.pq-tok.ref').forEach((s) => {
    const bad = unverifiedRefIds.has(s.dataset.ref);
    s.classList.toggle('unverified', bad);
    s.setAttribute('aria-label', tokenAria(s) + (bad ? ' ' + t('email.assure.unverifiedAria') : ''));
  });
}, [assurance, tokenAria]);
```
  `unverifiedRefIds` comes from `assuranceDerive.unverifiedRefs(draft.references, assurance)`
  (needs the ref-to-fact link, see 8.3).
- CSS: `.pq-tok.ref.unverified { color: var(--warn-ink); border-color: var(--warn);
  border-bottom-style: dashed; }` (mirrors line 453 `.pq-tok.ref`). Marking is colour AND dashed
  style AND an aria suffix, never colour alone. `.pq-body.nomarks` already strips ref marks; keep
  unverified marks visible even in `nomarks` mode (an explicit exception, because hiding the
  warning with the underlines would defeat it). Plain-text send is untouched: `renderPlain` never
  reads these classes, so the supplier never sees "unverified".
- References rail: in the `draft.references.map` row, add a `.pq-tag.internal` badge
  "Unverified" (or "You stated this") beside the existing tag, using the same `.pq-ref-top` flex.
- Provenance card (`.pq-prov`, shown by `showProv`): add one line under `.pq-prov-meta`:
  fact `label`, `source` as human text, `retrieved_at` (rendered by the backend, see 8.5) and
  `row_id`. Do not print table/column names (the backend sends labels and row ids only).
  Mapping `source`: postgres -> "From your records", user_asserted -> "You stated this
  (unverified)", carried_unverified -> "Carried over, not verified". Note `.pq-prov` is
  `pointer-events:none`, so hover/focus only, which is the existing behaviour.

### 4.5 Send-time change warning
Replace `doSend` with a two-step. Existing function body becomes `reallySend`.
```js
// PROPOSAL
const doSend = () => {
  if (!draft.unique_id) return reallySend();          // inert
  setBusy('send'); setError(null);
  preflightDraft(draft.unique_id).then((p) => {
    if (p.ok && !(p.changed || []).length) return reallySend();
    setChange(p); setChangeAck(false); setBusy(null);
    return getDraft(context).then(setDraft);          // show the NOW values behind the panel
  }).catch((e) => { setError(errText(e)); setBusy(null); });   // fail closed: no send
};
```
- Panel: rendered between the error strip and the checks strip, using `.pq-warn-inline`
  (the regenerate warning's look), `role="alertdialog"`-free: plain `role="alert"` and focus moved
  to its heading. Lists `changed[]` as "{label}: {was} -> {now}".
- `mode === 'shadow'`: shows a checkbox "I have read the changes" (`<input type=checkbox>` with
  `<label>`) and an "Acknowledge and send" `.pq-btn.primary`, disabled until checked; plus
  "Cancel" which only clears `change`. Acknowledge calls `reallySend()`.
- `mode === 'enforce'` or `ok === false`: no acknowledge control at all; show text "Blocked: these
  figures changed since the draft was written" and a "Regenerate" `.pq-btn` that calls
  `doRegenerate()`. Server remains the enforcement point (the send endpoint must also refuse).
- Preflight network failure: Send stays unsent and shows the generic error (fail closed; do not
  send on a failed check).
- Dedupe: the existing send button is `disabled={!!busy}` so a double click cannot fire twice.
- Interaction with the existing 409 handler in `reallySend` is unchanged.

### 4.6 Abandon on close without sending
Close paths today: Esc, backdrop mousedown, the x button, and the Home exit all go through the
`onClose` prop (Home navigates and the host unmounts via route change). Wrap rather than add an
unmount effect:
```js
// PROPOSAL
const closeModal = useCallback(() => {
  if (draft && draft.unique_id && !sentRef.current && !abandonedRef.current) {
    abandonedRef.current = true;
    abandonDraft(draft.unique_id, 'closed_without_sending').catch(() => {});   // fire and forget
  }
  onClose();
}, [draft, onClose]);
```
Replace `onClose` with `closeModal` in the Esc handler, the backdrop `onMouseDown`, `exits`, and
(for Home) wrap `navigate('/home')` with a call to it. `sentRef.current = true` when
`sendDraft` resolves with `result` (including `routed`/awaiting approval). Reasons NOT to use
`useEffect` cleanup: React StrictMode double-mounts in dev (a spurious abandon on mount) and
cleanup cannot tell "sent" from "closed".
- Host case: `EmailComposeHost` replaces `context` when another compose opens with a different
  key, which unmounts the old modal without `onClose`. Edit the Host's `registerComposeHost`
  callback: if the previous context has a live draft id, abandon it first (the host needs the
  modal to report `unique_id` up via a `onDraftLoaded` callback prop, or keep it in a ref).
- Tab close/reload cannot carry the Bearer token through axios reliably; treat that as
  best-effort and do NOT use `navigator.sendBeacon` (it cannot send the Authorization header). A
  server-side sweep of old drafts is the backstop (8.8).
- A failed abandon is silent (no toast); the user is already leaving.

## 5. i18n keys (English source strings)

How it works: `lib/i18n/strings.email.js` holds English only; `scripts/generateEnJson.mjs`
(`npx vite-node scripts/generateEnJson.mjs`) writes `src/locales/en.json`; the translation
service produces all other languages (11 are live) from en.json at runtime. So there is nothing
to write per language; the deliverable is the English source plus a regenerated en.json (CI
fails if stale). `strings.email.contract.test.js` scans `EmailComposeModal.jsx` and
`composeDerive.js` for literal `t('email.x')` and requires each key to exist: add
`AssurancePanel.jsx` and `assuranceDerive.js` to its `files` list, and always call `t()` with a
literal string key (a computed key escapes the test; use `tOr` only for the reasoned-field map).

```
'email.assure.briefTitle': 'How this draft was reasoned',
'email.assure.briefChip': 'Brief',
'email.assure.status.verified': 'Verified',
'email.assure.status.needs_review': 'Needs review',
'email.assure.status.unassured': 'No assurance record',
'email.assure.goal': 'Goal',
'email.assure.keyPoints': 'Key points',
'email.assure.ask': 'The ask',
'email.assure.deadline': 'Deadline',
'email.assure.tone': 'Why this tone',
'email.assure.risks': 'Risks to avoid',
'email.assure.reasonedTitle': 'What the agent worked out',
'email.assure.basis': 'Based on',
'email.assure.confidence': 'Confidence: {level}',
'email.assure.missing': 'Not known',
'email.assure.accountability': 'Started by {initiated}. Reviewed by {reviewed}.',
'email.assure.notReviewed': 'not yet reviewed',
'email.assure.assumptionsTitle': 'Assumptions to confirm',
'email.assure.assumptionsLeft': '{n} assumptions still need your answer',
'email.assure.assumptionsLeftOne': '1 assumption still needs your answer',
'email.assure.confirm': 'Confirm',
'email.assure.edit': 'Edit',
'email.assure.reject': 'Reject',
'email.assure.save': 'Save',
'email.assure.cancel': 'Cancel',
'email.assure.editAria': 'Your value for: {text}',
'email.assure.resolved': 'Resolved: {resolution}',
'email.assure.gate.assumptions': 'Confirm, edit or reject every assumption before this can be sent.',
'email.assure.gate.loading': 'Checking how this draft was reasoned…',
'email.assure.gate.failed': 'How this draft was reasoned could not be loaded, so it cannot be sent yet.',
'email.assure.conflictsTitle': 'Your records and this request disagree',
'email.assure.conflictRecords': 'Your records say',
'email.assure.conflictSupplied': 'The request said',
'email.assure.conflictWins': 'Your records are used.',
'email.assure.unverified': 'Unverified',
'email.assure.youStated': 'You stated this',
'email.assure.unverifiedAria': 'This figure is not verified against your records.',
'email.assure.src.postgres': 'From your records',
'email.assure.src.user_asserted': 'You stated this (unverified)',
'email.assure.src.carried_unverified': 'Carried over, not verified',
'email.assure.retrieved': 'Retrieved {when}',
'email.assure.violationsTitle': 'Problems found',
'email.assure.judge': 'Independent review: {status}',
'email.assure.changeTitle': 'Some figures changed since this draft was written',
'email.assure.changeRow': '{label}: {was} became {now}',
'email.assure.changeRead': 'I have read these changes',
'email.assure.changeSend': 'Acknowledge and send',
'email.assure.changeBlocked': 'Sending is blocked until the draft is regenerated with the current figures.',
'email.assure.changeCancel': 'Do not send',
'email.assure.loadFailed': 'Could not load this draft’s assurance record.',
'email.assure.saveFailed': 'That answer could not be saved. Please try again.',
'email.assure.retry': 'Try again',
```
Check: the `{level}`, `{n}`, `{when}`, `{was}`, `{now}` placeholders must be preserved by the
translation service (it validates placeholder parity server-side). The `email.assure.reasoned.*`
labels depend on the backend's key set (8.4); list them after the backend publishes the keys.

## 6. CSS (appended to emailCompose.css; tokens only)

Reuse as-is: `.pq-pill(.good/.warn)`, `.pq-btn(.primary/.on)`, `.pq-finput`, `.pq-tag(.share/.internal)`,
`.pq-blocked`, `.pq-warn-inline`, `.pq-check(.fail)/.pq-check-ico`, `.pq-ref-top/-detail`,
`.pq-checks-bar/-tog` pattern, `.pq-prov-*`, `.pq-link`. New classes are layout only:
`.pq-as-brief .pq-as-bar .pq-as-body .pq-as-dl .pq-as-assume .pq-as-assume-row .pq-as-conflict
.pq-as-change` plus `.pq-tok.ref.unverified`. Tokens: `--surface-2` backgrounds, `--line` borders,
`--ink-2/--ink-3` text, `--warn/--warn-ink/--warn-soft` for needs-attention, `--good*` for verified,
`--font-mono` for labels, `--radius-sm`. Add the new blocks to the existing `@media (max-width:
900px)` rule so the brief stacks under the message with the rail, and keep
`@media (prefers-reduced-motion)` (line 863) covering any transition. No hex, no new colour.

## 7. States and accessibility

| State | Behaviour |
|---|---|
| inert (no `unique_id`) | Nothing rendered, nothing blocked. |
| loading | Pill "Checking..." (`.pq-pill`), Send closed, reason `email.assure.gate.loading`. |
| failed | `.pq-blocked` "could not load" + Try again button; Send closed (fail closed). |
| unassured | Neutral pill + muted text; does NOT block unless backend `ready:false` (8.6). |
| empty brief parts | A field with no value is omitted, not shown as "-"; `missing[]` empty hides that heading. |
| no unresolved assumptions | Assumption block not rendered in the message column. |

A11y: collapse is a `<button aria-expanded aria-controls>`; assumption block is a
`<section aria-labelledby>` with a live count (`role="status"`); each row's buttons carry
`aria-label` including the assumption text ("Confirm: {text}"); edit input has its own aria-label;
conflict banner `role="alert"` (as `.pq-blocked` already is); the change panel moves focus to its
heading on open and returns focus to Send on cancel; the dialog's focus trap
(`FOCUSABLE` selector) already covers `button`, `input`; the Esc handler's `anyPanel` should
include `change` so Esc closes the change panel before the modal (as it does for other panels);
unverified state is conveyed by dashed underline + aria text, not colour alone; status pill text
is present (not an icon).

## 8. Conflicts, gaps and open questions for the backend

1. **No `unique_id` on the envelope.** BACKEND_CONTRACT 2.1 names `draft_id`; section 3 says
   "prepare returns a fresh `unique_id`". The assurance endpoints use `unique_id`. Need to decide
   which id the UI holds at compose time (and whether the draft already exists in the email table
   before prepare). Required: `draft.unique_id` on the envelope.
2. **Same path, different meaning.** Contract 2.5: `POST /api/v1/negotiation/drafts/{draft_id}/preflight`
   -> `{checks}`. Brief: `POST /drafts/{unique_id}/preflight` -> `{ok, mode, changed}`. Different
   prefix and different response. The UI must treat them as two functions; the backend must
   not alias them.
3. **No ref-to-fact link.** `facts` is keyed by fact `key`; the body tokens use `references[].id`
   (`r1`...). Needs `references[].fact_key` (or `facts[k].ref_id`) so a token can be marked.
   Without it, inline marking (4.4) cannot be done and I would fall back to the references rail
   only.
4. **`confidence` and `reasoned` keys.** Number or label? The client may not format numbers
   (contract 2.2). Ask for `confidence_label` (high/medium/low) or a 0-1 value the client shows
   only as banded text. Also the set of `reasoned` keys (to write the label strings).
5. **Dates/values as strings.** `retrieved_at`, `facts[].value`, `conflicts[].postgres/supplied`
   and `changed[].was/now` must arrive already rendered for the tenant locale.
6. **Semantics of `ready`, `unassured`, `resolution`.** Is `ready=false` the single send gate? Is
   an unassured draft sendable (legacy drafts)? Which `resolution` values mean unresolved
   (assumed null/"pending")? Does a `reject` regenerate the body, and does the response signal
   it? Do non-empty `missing[]` or `violations[]` of severity block also close Send?
7. **`conflicts[].resolution`** is assumed to be a code (e.g. "postgres_wins"); if it is prose,
   it must be English-neutral or the key set published for translation.
8. **Abandon is unreliable by design** (tab close, crash, mobile). Needs a server-side sweep.
9. `unverified_figures[]` shape is unspecified (keys? strings? ids?).
10. Output-safety: any of these fields containing a route/table name will come back
    `"[withheld]"`; the UI shows it as-is, which will look broken but is correct.
11. BACKEND_CONTRACT line 1 claims "UI does not fail open"; the new gate keeps that: failed
    assurance load closes Send. Statement still true.

## 9. Risks

- **Mock/real split.** Everything is gated on `unique_id`; reviewers could think it works in the
  preview when it is inert. Mock fixtures must supply an assurance record to exercise it.
- **Imperative body.** Any edit near `paintBody` could take the caret; the proposal changes only
  classes/aria in a separate effect (the existing safe pattern). Review that effect closely.
- **Double gating.** `checksState`, `orphans`, assumptions, change panel and `result` all close
  Send; ordering in `sendGate` decides which reason shows. Proposal puts assumptions after
  orphans and before stale/failed.
- **StrictMode / HMR** (abandon): handled by close-handler not effect; verify in dev.
- **Race:** assurance responses vs rapid edits (request counter in section 2).
- **Home page:** if the Host remounts via route change, abandon may be skipped (Home exit
  navigates). Covered by wrapping `navigate`.
- **Translation:** 11 languages, long German/Finnish strings in `.pq-as-assume-row`; allow wrap
  (`flex-wrap: wrap`), no fixed widths.
- **Shared working tree:** the UI repo is read-only to me and the owner may have uncommitted
  edits in `EmailComposeModal.jsx`; produce this as a patch against a clean HEAD.

## 10. Tests

Existing tests touched:
- `strings.email.contract.test.js`: add the two new files to its scanned list; en.json must be
  regenerated or "reaches the catalogue" fails.
- `render.smoke.test.jsx`: "draft carries everything its markup reads" must include the mock
  assurance shape; add `unique_id` to the mock draft. Its `sendGate`/`statusPill` assertions
  stay valid (additive arg) but I have not read the whole file; verify.
- `compose.contract.test.js`: unaffected unless fixtures change `getDraft` output; the
  `preflight` tests keep passing because the new function has a different name.
- `SpendIQ/composeSurface.contract.test.js` and `composeSurface` toLocaleString rule: new files
  must not call `toLocaleString`/`Intl`.
- `BACKEND_CONTRACT.md` is "executable" per its own header: update alongside.

New tests (vitest, SSR/pure like the existing ones; there is no jsdom):
- `assuranceDerive.test.js`: unresolved count; `assuranceGate` reasons for loading/failed/needs
  assumptions/ready/unassured/inert; unverified ref mapping.
- Gate: `sendGate` returns the assumption reason while any assumption is unresolved and null
  after all resolved (prove it fails: delete the assurance branch and watch the test go red,
  per the "prove the guard fails" rule).
- `assuranceService.contract.test.js` with a mocked axios: right method/path/body for the four
  calls, `unique_id` URL-encoded, no identity field in any body.
- SSR render tests: brief collapsed by default (`aria-expanded="false"`, body absent); expanded
  shows goal/ask/risks/reasoned basis; assumptions block present with 3 buttons per row;
  conflict banner shows both values and "records are used"; unverified badge in the rail row.
- Send-flow test (extract the decision into a pure function `decideSend(preflightResult)` in
  assuranceDerive.js so it is testable without a DOM): ok+empty -> send; shadow+changed ->
  needs acknowledge; enforce -> blocked, no acknowledge path; network error -> blocked.
- Abandon: pure helper `shouldAbandon({uniqueId, sent, abandoned})`; test sent/closed/double close.
- Supplier artifact: `renderPlain` output for a draft with unverified facts contains no
  "unverified" and no class names (guards that the display layer never leaks into the send).
- CSS contract: new block contains no `#hex` and no `rgb(` (mirror the file-header rule).

## Decision for the UI owner

**Option A, scoped exception:** allow one scoped change set limited to `modules/EmailCompose/` +
`strings.email.js`/`en.json` (about 5 new/edited files, no change to SpendIQ or routing). Pros:
a single reviewable PR, the contract tests travel with it, the backend can be tested end to
end. Cons: it edits the imperative body modal, which is the riskiest file in the module, and it
cannot be fully exercised until the backend puts `unique_id` and the ref-to-fact link on the
draft envelope.

**Option B, hand-off:** the owner applies the proposal themselves, using sections 3-5 and 10
directly; BP_Backend ships the endpoints and I publish the contract (section 8) so nothing in
the UI is blocked on guesses. Pros: no cross-repo edits, the owner controls the modal; cons:
slower.

My recommendation: B for `EmailComposeModal.jsx` (hand-off, because of the caret/imperative
risk) and A for the leaf files (`assuranceService.js`, `assuranceDerive.js`, `AssurancePanel.jsx`,
CSS block, i18n keys, tests), which are additive and only become live once the modal wires
them in. First, answer section 8 items 1, 3 and 6; the rest can be settled during review.

---

# Backend answers to section 8 (added by BP_Backend, 2026-10-08)

The four endpoints exist (`src/api/routers/draft_assurance.py`, 29 + 28 tests). Answers, in the
numbering of section 8:

1. **`unique_id`.** Every backend draft carries `unique_id` (it is the key of the stored draft, the
   approval, the dispatch and the capture row). The compose envelope must pass it through when
   compose stops being a mock. Until it does, the patch is inert, as the proposal already assumes.
   An id with no capture row returns 404 from GET; the UI should treat 404 as `inert`, not as an error.
2. **Preflight naming.** Agreed: two different things. Backend path is `POST /drafts/{unique_id}/preflight`
   returning `{ok, checked, mode, changed[{fact,label,was,now}], ready, reason?}`. It does not alias
   the contract's `/api/v1/negotiation/drafts/{draft_id}/preflight`.
3. **Reference-to-fact link.** NOT built. The backend has `facts` keyed by fact key and no per-token
   link. Needs a decision: either the draft envelope's `references[]` gains `fact_key`, or the view
   gains `facts[k].ref_id`. Until then 4.4's inline marking falls back to the references rail only.
4. **Confidence / reasoned keys.** `brief.reasoned[k]` carries `confidence` (0-1 or null) AND
   `confidence_label` ("high" >= 0.8, "medium" >= 0.5, "low", or null). Show the label. For counters
   `confidence` is null (the negotiation agent states none). `reasoned` keys today: `counter_price`,
   `target_price`, `response_deadline`, `lead_time_request` (counter family); free-text drafts get
   whatever the planner names, so the label map must fall back to the key.
5. **Pre-rendered strings.** NOT built. Values are the raw stored text (e.g. `"132500.0000"`) and
   `retrieved_at` is an ISO-8601 UTC timestamp. Locale formatting of figures and dates is an open item.
6. **Semantics.**
   - `ready` is the single send gate for assumptions and clarification. `ready=false` means an
     assumption is unresolved, was rejected or edited (`needs_redraft=true`), or a clarification
     is open. Violations, conflicts and judge scores do NOT change `ready`.
   - `status="unassured"` is sendable: nothing was checked. Do not gate on it. A draft with no
     capture row at all is `inert` (404).
   - An assumption is unresolved while `resolution` is `null`. After an answer it is
     `{action: "confirm"|"edit"|"reject", by, at, value?}`.
   - `reject` and `edit` do NOT regenerate the body. They set `needs_redraft=true`, `ready=false`;
     regeneration is a separate action and writes a NEW capture row.
   - Today the server enforces readiness only when the family is in `enforce` mode (it is `shadow`),
     so the UI gate is the only gate until a family is switched. The server-side reviewer rule is
     hard: no human reviewer means no send.
7. **`conflicts[].resolution`** is a code (`"postgres_wins"`), not prose.
8. **Abandon sweep.** NOT built. Abandon is best-effort by design; a scheduled sweep of drafts that were
   never sent is not written.
9. **`unverified_figures[]`** is a list of plain strings: the figures that appear in the email text and rest
   only on what the request carried (e.g. `["25", "30"]`). `facts[k].source` is `"postgres"` or
   `"carried_unverified"`; `user_asserted` is not produced yet.
10. **Output safety.** The view carries labels and row ids only; no table or column names, so nothing
    should come back `"[withheld]"`.

Not decided here: Option A vs B in the last section above. That is the UI owner's call.

---

# Addendum 2026-10-08 (tone gaps and the classifier question)

Section 4.2 (assumption confirmation) now also carries two more kinds of item, with no new endpoint
and no new screen. They arrive in `assumptions[]` exactly like the others:

| `id` | When it appears | Reviewer's choices |
|---|---|---|
| `tone:escalation_level`, `tone:recipient_seniority` | There was no stored data to set that tone variable, so a default was assumed. Text says what was assumed ("...assumed to be 1. Confirm, or give the right value."). | Confirm / Edit (a value) / Reject |
| `tone:instruction` | The instruction used a tone word nothing maps (e.g. "brusque"). Text names the word(s) and says it had no effect. | Confirm the tone as set / Edit |
| `clarification` | The classifier could not tell which kind of email this is. The item carries `options` (the two families). | Confirm the one chosen / Edit with `value` = the other option |

- `options` is only present on `clarification`. Render it as the choices for Edit instead of a free input.
- An Edit on any item sets `needs_redraft=true`, `ready=false`: the text rests on the old value.
- Other tone gaps (leverage, warmth, directness, region formality, relationship health, tier) are
  recorded but do NOT ask anything; they are visible in the brief's tone line as "(default)".
- `brief.tone_rationale` lists all eight variables with their source: postgres, user_instruction or default.
- Machine ids contain a colon (`tone:escalation_level`): URL-encode nothing, they travel in the JSON body only.

## Addendum: endpoints now available for a learning-queues screen (2026-10-08)

The backend half exists; the UI is untouched. Everything is under `/email-learning`, authenticated, and every write names the signed-in
person on the server (never in the body).

| screen element | call |
|---|---|
| badge counts per queue | `GET /queues` |
| "Facts a reviewer changed" list; resolve / dismiss | `GET /data-quality`, `POST /data-quality/{id}/decision` `{"action":"resolve"|"dismiss","note":"..."}` |
| "Patterns across reviewers"; accept / dismiss | `GET /review-items`, `POST /review-items/{id}/decision` `{"action":"accept"|"dismiss"}` |
| "My suggested writing rules"; approve / edit / reject | `GET /style-rules`, `POST /style-rules/{id}/decision` `{"action":"approve"|"edit"|"reject","text":"..."}` |
| exemplar candidates; read one; approve / reject | `GET /exemplars`, `GET /exemplars/{id}` (text), `POST /exemplars/{id}/decision` `{"action":"approve"|"reject"}` (Admin; never the author) |
| eval and classifier candidates; export / reject | `GET /eval-candidates`, `GET /classifier-examples`, `POST .../{id}/decision` `{"action":"export"|"reject"}` |
| quality over time | `GET /metrics?bucket=day|week|month&family=...&days=90` |

A 403 means the person may not; a 422 means the item is not in a state that allows the action (already decided, or you are its author);
a 404 means no such item for you.
