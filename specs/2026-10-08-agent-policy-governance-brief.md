# Build prompt: Agent policy governance (update to the existing build)

> Saved verbatim on 2026-10-08 as received. Rulings and the facts about the existing code are in
> `specs/2026-10-08-agent-policy-governance-design.md`, which takes precedence where they differ.

You are updating an existing feature in this product. Do not rebuild it. First read the existing code for policy upload and extraction, the policy list, the policy edit form, and the export to the orchestrator. Then reply with a short plan before you change any code. The plan should list:

1. each existing form field and what happens to it, using the mapping in section 3.1;
2. each data model change and its migration;
3. anything in this brief that conflicts with the existing code.

Also read the existing decision engine's interface: how cases are submitted, how decisions come back, and how it routes to human approvers. Integrate with it as it is. Do not build a second decision service.

Keep everything that already works unless this brief says otherwise. A reference mock of the target form is attached (`hard-policy-form-simple.html`). Match its behaviour, not its styling. Use the product's existing components and styles.

## 1. What the feature is

A company uploads its policy documents. One extraction agent turns them into agent policies. People review and edit each policy in a form and save it as a version. Plain code then converts the saved form, with no model involved, into JSON that the orchestrator enforces on agents.

| Layer | Job | Who touches it |
|---|---|---|
| Extraction agent | Proposes policies from documents. Never enforces, never activates. | Nobody edits its raw output |
| Form | The source of truth. Review, edit, version, activate. | Policy owners and reviewers |
| JSON | Generated from the form. Read by the orchestrator. | Nobody edits it. Administrators can view it read-only. |
| Decision engine (existing) | Where every policy conflict is decided and recorded. Sends a summary to a human approver. | Human approvers |

A policy is either a point where a human must step in or a flat prohibition. It is not a guideline. A policy only qualifies if it can be checked at the moment an agent acts. Everything else in a document is listed as "Not enforceable", with a reason, so nothing is silently dropped.

## 2. Hard requirements

- **One extraction agent.** It reads every document from every business area and produces every policy in one pass. Business area is a field it fills in, not a reason for a second agent.
- **One company policy can yield several agent policies.** Split them so each agent policy has one situation and one outcome. Tiered rules become separate policies. For example, "approval over $500, prohibited over $10,000" becomes two policies, grouped under the same source in the list.
- **Business area and sub-area mean where the policy comes from,** as the document states it. This is separate from what the policy applies to, and one is never inferred from the other. Administrators can edit the taxonomy. Every sub-area list includes "General".
- **Extraction output is never pre-confirmed.** A named person must confirm a policy before it can be Active. Record who confirmed it and when.
- **Nothing the agent produces becomes Active automatically.** This includes changes suggested by learning (section 6).

## 3. Form changes

### 3.1 Map the existing fields

| Existing field | Becomes |
|---|---|
| Name | Name (unchanged) |
| Category | Category (unchanged; keep the existing values) |
| Rule text | **The situation**: one plain-English sentence describing when the policy applies |
| Enforcement | **What happens**: one of three outcomes (section 3.3) |
| Applies to | One line, "Applies to all agents, tools and skills", with an optional **Limit it** control that reveals one free-text field |
| Owner, Effective from, Review by, Status, Change note | Unchanged |
| *(new)* | Business area and sub-area, under Identity |
| *(new)* | Source excerpt, quoted from the document with its section number. Read-only. |
| *(new)* | Example check, with a confirmation box (section 3.2) |
| *(new)* | **How it is enforced**: a three-line plain summary of when it is checked, what the orchestrator needs to know, and what happens next (section 3.7) |

The form sections, in order:

1. Identity
2. The policy: situation, source excerpt, How it is enforced, example check
3. What happens
4. Applies to
5. Governance
6. A health line, on Active policies only
7. A collapsed "Technical view (administrators)" showing the generated JSON, read-only

Do not show these to policy owners: events, hook points, condition rows, "fires when all / any", per-tool or per-agent scope lists, audit settings, or test cases. The agent fills them in. They stay in the data model.

### 3.2 Example check (replaces conditions)

The agent writes the situation sentence and, separately, a structured condition that the orchestrator evaluates. Policy owners check the condition through examples instead of reading it.

- The agent proposes 3 to 6 example inputs. Each numeric boundary needs one case just inside it, one just outside, and one exactly on it. Each list condition needs one value in the list and one outside it.
- Code evaluates every example against the stored condition and shows the result: "A person decides", "Blocked", "Someone is told", or "Nothing happens". The model never decides what an example's result is, so the examples cannot disagree with what will be enforced.
- The reviewer can flip any result. A flipped result means the condition is wrong. In that case:
  - mark the row as changed;
  - block Active;
  - offer "Ask the agent to fix it". This re-runs the agent on this one policy, with the flipped examples as constraints. Show the new situation and examples as a change for the reviewer to accept.
- The checkbox reads "The examples are right. This is what the policy says." Ticking it records `checkedBy` and `checkedAt`. Editing the situation, accepting an agent fix, or flipping an example clears it.

### 3.3 What happens (outcome)

There is one required choice per policy. The agent suggests it from the document's wording and shows "Suggested by the agent because the document says '…'". The owner can change it.

| Outcome | Label | Behaviour | Fields |
|---|---|---|---|
| `approve` | Needs approval: a person decides | The action pauses until someone decides | Who decides (ordered levels), response time |
| `block` | Not allowed | The action is refused. Nobody is asked to approve it. | Who is told (optional) |
| `notify` | Notify only | The action goes ahead and someone is told | Who is told (required) |

- **Response time exists only for `approve`.** It defaults to a company-wide setting, displayed as "Response time: company default, 4 hours. Set a different time". Store an override as an ISO 8601 duration.
- **Who decides is an ordered list of levels.** If a level does not respond in time, the request goes to the next level. If the last level times out, the action is rejected. A timeout never approves anything.
- **`block` has no exception path in this release.** Remove any "approves an exception" wording from the owner field's help text.
- **Switching outcome drops the previous outcome's fields from the saved policy and the JSON.** For example, switching from approve to block removes the levels and response time.

### 3.4 Status and versioning

- **Status values:** Draft, Active, Retired. In the JSON these are `draft`, `live` and `retired`. Map any existing values to these and say how in the plan.
- **Every save creates a new version.** Keep the existing two-step confirm for activating and for retiring a policy. Cancel restores the last saved version.
- **Policy IDs** are stable across versions and across re-extraction of the same source clause. Use a prefix for the business area plus a number, for example `FIN-0012`, and never reuse an ID.

### 3.5 What Active requires

A Draft can be saved with only a name. To save a policy as Active, all of the following must be true:

- name, business area and sub-area, situation, owner, and outcome are filled in;
- `approve`: at least one decider; a custom response time, if set, is at least 1;
- `notify`: at least one person to tell;
- "Limit it", if turned on, has text;
- the examples are confirmed, and no example has been flipped;
- every tool name and field in the hidden condition exists in the orchestrator's registry. If one does not, show "This policy refers to something the orchestrator does not recognise" and route it to an administrator. The example check cannot catch a wrong tool name, so this check must.

- the enforcement contract (section 3.7) is complete:
  - a checkpoint is set;
  - every input is available at that checkpoint;
  - units and currency are stated wherever there are amounts;
  - `approve` and `block` policies have a message to the agent.

Show every problem in one summary and move focus to the first field that needs fixing.

### 3.6 Extraction confidence

The header shows version, status, and **extraction confidence**. Use this name. The learning feature in section 6 has its own "suggestion confidence", and the two must not be confused.

Compute extraction confidence with code, not from the model's opinion of itself:

- **High:** all of these pass:
  - the excerpt appears word for word in the document;
  - every field is filled;
  - the condition uses only names from the registry;
  - the agent's own expected result for each example matches what code computes.
- **Medium:** one of those checks fails.
- **Low:** more than one fails.

When confidence is not High, list the failed checks in the "Check before this goes Active" notice.

- **Inputs:** add the inputs check from section 3.7 to the High checks. Every input must be available at the moment the policy is checked.

### 3.7 Enforcement contract: context, inputs and outputs

Every policy carries a contract. It is everything the orchestrator needs to enforce the policy without guessing, and everything the people involved need to act. The agent fills it in, code checks it against the registry, and the reviewer confirms it in plain words. If any part is missing or cannot be supplied, the policy cannot be Active.

#### Context: when and where it is checked

- **Checkpoint:** the moment the policy is checked. Use the registry's names, for example:
  - before a tool runs;
  - before a message is sent to a customer;
  - before data leaves a system;
  - before a record is written.
- **Actions:** the actions that bring the policy into play, in registry terms (tool names, skill names, message channels) and as one plain phrase, for example "issuing a refund or credit".
- **Applies to:** all agents, tools and skills unless limited (section 3.1).
- **When:** the effective dates. If the policy only applies at certain times, such as outside business hours or during a period close, give the time window and the time zone it is measured in.
- **Units:** currency, and whether amounts include tax. A policy that says "$500" must state the currency and how other currencies are converted, by default at the rate on the day of the action. No unit is ever assumed silently.

#### Inputs: what the orchestrator must know at that moment

- **The list:** every input the condition and the outputs use, with:
  - a plain name ("refund amount", "customer's country", "refunds to this customer in the last 30 days");
  - its registry field (`args.amount`, `customer.country`, `agg.refunds_30d`);
  - its type and unit;
  - where it comes from: the action itself, a lookup (for example the customer record), or a running total over time.
- **Availability check:** code checks every input against the registry's list of what is available at that checkpoint. Lookups and running totals must be registered data sources. If an input is not available, the policy shows "Can't be enforced yet: the orchestrator does not receive *refunds to this customer in the last 30 days* at this point". This blocks Active and creates a task for an administrator. Never let a policy go live that depends on data the orchestrator does not have.
- **When an input is missing at run time:** `onMissingData` (section 7) decides what happens, and the result is logged with the missing input's name.
- **Approval details:** list the inputs the human approver must see to decide, by default every condition input plus the agent's stated reason for the action. Mark any personal or sensitive input so it can be masked to everyone except the approver.

#### Outputs: what happens, and who receives what

For each outcome, the contract states:

1. **To the agent:** what the orchestrator returns. This is a machine-readable result plus a plain reason the agent can use.
   - Results:
     - `allowed`;
     - `paused_for_approval`, with a request ID and the response time;
     - `approved`, `rejected` or `blocked`.
   - Reason codes: the policy ID plus a short code such as `FIN-0012.over_limit`.
   - What the agent may do while paused: wait, or carry on with other work but not retry this action. The default is to not retry.
   - What the agent tells the person it is serving, if anyone. For example: "Your refund needs a manager's approval. You will hear back within 4 hours." The agent suggests this wording and the owner can edit it.
2. **To the human approver** (`approve` only): the approval request. It contains:
   - the action as one plain sentence;
   - the input values from the approval-details list;
   - the agent's stated reason;
   - the policy's situation and the source excerpt word for word;
   - the options (approve or reject, with a reason required on reject);
   - the deadline and what happens on timeout.

   If the existing build allows "approve with changes", keep it, and say in the plan which inputs an approver may change. A changed request is checked against the policy again before it runs.
3. **To people who are told** (`notify`, and optionally `block`): a short message with the action, the policy, the outcome and a link to the record.
4. **To the audit log:**
   - policy ID and version;
   - checkpoint;
   - input values (masked where marked);
   - the outcome, and who decided and when;
   - any reason given;
   - the time taken.

   The learning feature (section 6) and the decision engine (section 4) read from this log.

#### How the reviewer sees it

Keep this to a short block in the form, titled **How it is enforced**, in plain words, for example:

> **Checked when** the agent is about to issue a refund or credit.
> **Needs to know** the refund amount and currency (from the action).
> **Then** the action pauses and the Finance Manager is asked to approve. The agent tells the customer: "Your refund needs a manager's approval…"

- The block is read-only. To change it, the reviewer uses "Ask the agent to fix it" (section 3.2), or edits the customer message directly.
- It is part of what "The examples are right" confirms. Change the checkbox label to "The examples and how it is enforced are right. This is what the policy says."
- If any input is unavailable, show the "Can't be enforced yet" message here, in place of the summary.

## 4. Many documents, inventory and conflicts

Keep all of the following and do not let it regress:

- uploading many documents at once;
- the policy list grouped by source document, with tiered policies grouped under their source;
- an inventory grouped by business area and sub-area;
- the inventory's CSV export, guarded against formula injection: any cell starting with `=`, `+`, `-` or `@` gets a leading `'`.

Replace the current "shares an event and a tool" overlap check with conflict detection, and send every conflict to the decision engine. The decision engine is where the final decision lives. This product detects conflicts and supplies the facts. It never resolves a conflict itself.

### 4.1 Two kinds of conflict

| | Policy conflict | Live conflict |
|---|---|---|
| When it is found | At design time, when a policy is saved or imported | At run time, when an agent action matches several policies |
| What it is | Two policies from different sources whose hidden conditions can both match the same action, with different outcomes | One action matches policies with different outcomes, or with different deciders |
| Who decides | The owners of the policies involved, via the decision engine | The human approver the decision engine routes it to |
| What is decided | How the policies should relate from now on | What happens to this action |

Tiered policies from the same source are not conflicts. When several matching policies all have the same outcome and the same deciders, that is not a conflict either. Apply it normally.

### 4.2 Policy conflicts (design time)

- **Detection:** check for conflicts whenever a policy is saved as Draft or Active, and after every extraction. To prove an overlap, code must find at least one example input that matches both policies. Use the example inputs plus the boundary values from both conditions. If no such input is found, there is no conflict.
- **Case:** send a `policy` conflict case to the decision engine (section 4.4). Show "Conflict sent for decision" on both policies and in the inventory, with a link to the case.
- **Active is still allowed** while a policy conflict is unresolved. Until it is resolved, any action that hits it becomes a live conflict.
- **Applying the decision:** when the decision comes back, apply it in this product. The outcome determines what happens:
  - **Keep both, with a standing rule:** store the rule, for example "Policy A takes priority over Policy B", on both policies' JSON as `conflicts[]`. The orchestrator passes it to the decision engine on live conflicts.
  - **Change one policy:** open that policy as a new Draft version, with the decision summary in the change note, for the owner to edit and save.
  - **Limit one policy:** open a Draft with "Limit it" turned on and the suggested limit filled in.
  - **Retire one policy:** start the normal two-step retire confirmation.

  Never apply a decision silently. Every change still goes through the owner's save.

### 4.3 Live conflicts (run time)

- **The orchestrator holds no precedence rules of its own.** Remove the hard-coded "block, then approve, then notify" rule. When an action matches several policies, the orchestrator pauses the action and sends a `live` conflict case to the decision engine.
- **The decision engine responds** in one of two ways:
  - With an automatic decision, when a standing rule from a policy conflict covers this case.
  - By routing the case to a human approver, with a summary. The action stays paused until the approver decides.
- **Failing safe:**
  - If the decision engine is unreachable, or no decision arrives within the response time, the action is rejected. It is never allowed by default.
  - Matching `notify` policies still send their notifications whatever is decided.
- **Not allowed versus Needs approval:** see Open decisions. My recommendation is that a matching `block` still blocks immediately, and the case goes to the decision engine for the record and for a policy-level fix. Otherwise a conflict becomes a back door around Not allowed.
- **Feedback:** when the same live conflict is decided the same way 5 times by the approver (a company setting), the decision engine, or this product if the engine cannot, raises a policy conflict case. It proposes making that decision a standing rule, so people stop deciding it one action at a time.

### 4.4 Conflict case payload (`policy-conflict/1`)

This product sends facts. The decision engine writes and delivers the summary for the human. Adapt the field names to the engine's existing interface and list the mapping in your plan.

```json
{
  "schema": "policy-conflict/1",
  "caseId": "pc_7f3a",
  "kind": "live",
  "raisedAt": "2026-10-08T09:20:00Z",
  "action": {
    "agent": "support-agent-eu",
    "tool": "refund.issue",
    "plain": "Issue a refund of $12,400 to customer account 4471",
    "args": { "amount": 12400, "currency": "USD" }
  },
  "policies": [
    {
      "id": "FIN-0012", "version": 3, "outcome": "approve",
      "situation": "The agent is about to issue a refund or credit above $500.",
      "deciders": ["Finance Manager", "CFO"],
      "owner": "Chief Financial Officer", "businessArea": "Finance / Refunds and credits",
      "source": { "document": "Finance Payments Policy", "reference": "1.1", "excerpt": "Refunds or credits above $500 need approval from the Finance Manager." }
    },
    {
      "id": "CUS-0004", "version": 1, "outcome": "block",
      "situation": "The agent is about to issue a refund above $10,000.",
      "owner": "Head of Customer Operations", "businessArea": "Customer operations / Refunds",
      "source": { "document": "Customer Refund Standard", "reference": "4.2", "excerpt": "Agents must never process refunds above $10,000." }
    }
  ],
  "overlap": { "example": { "tool": "refund.issue", "amount": 12400 } },
  "standingRules": [],
  "priorDecisions": { "sameConflict": 2, "lastOutcome": "rejected" },
  "options": ["block", "approve_by:CFO", "reject_and_escalate_to_owners"],
  "respondWithin": "PT4H",
  "onTimeout": "reject"
}
```

Rules for the payload:

- **`args`:** send only the fields the matched conditions use. Never send whole payloads or personal data that the conditions do not need.
- **`kind: "policy"` cases:** omit `action`. Make `overlap.example` the input that code found matching both policies, and make `options` the four resolutions in section 4.2. The case goes to the owners of both policies.
- **The decision coming back** must include:
  - `caseId`;
  - `decision`, one of the options;
  - `scope`: `this_action` or `standing_rule`;
  - `decidedBy`;
  - `decidedAt`;
  - `reason`.

  Store every decision against the policies involved and show it in each policy's history.

### 4.5 What the human approver's summary must contain

The decision engine owns the format. Supply what it needs so the summary always states:

1. What the agent was trying to do, in one plain sentence. For policy cases, give the example action that triggers both policies.
2. Each policy involved: its situation sentence, its outcome, its owner, and the source excerpt word for word.
3. Why they conflict, in one line. For example: "One policy needs Finance approval; the other does not allow this at all."
4. Previous decisions on the same conflict.
5. The options, and the response time.

Any generated wording must sit alongside the verbatim policy text, never replace it, so the approver decides on what the policies actually say.

## 5. Extraction agent prompt

Update the agent's instructions so that it:

- reads every document in one pass and returns every policy, plus the "Not enforceable" items with reasons;
- fills business area and sub-area from where the policy comes from, and never from what it applies to;
- splits clauses so each policy has one situation and one outcome;
- returns, for each policy:
  - name and category;
  - business area and sub-area;
  - the situation as one plain sentence;
  - the structured condition, using only names from the orchestrator registry, which is supplied in the prompt;
  - the suggested outcome and the exact phrase behind it;
  - deciders or people to tell, if the document names them;
  - source document, section and excerpt, word for word;
  - 3 to 6 example inputs, each with the agent's own expected result;
  - the enforcement contract (section 3.7):
    - the checkpoint and actions;
    - the time window and time zone, if any;
    - units and currency;
    - every input, with its plain name, registry field and where it comes from;
    - the approval details;
    - a suggested message to the agent, and to the person it serves;
- never sets confirmation or Active;
- returns JSON Lines, one policy per line, so results stream into the list.

When the registry has no name for something the policy needs, the agent must say so in the item, not invent a name. The same applies to an input the orchestrator would need but the registry does not provide at that checkpoint. List it as a missing input so the policy shows "Can't be enforced yet".

Supply the registry in the agent's prompt, including the checkpoints, the actions, and the inputs available at each checkpoint.

## 6. Learning: suggest changes from how people decide

The goal: when people keep approving the same kind of case, suggest a condition change to the owner as an action to review. A suggestion is never applied on its own.

### 6.1 Log

For every firing of an `approve` policy, append a record with:

- policy ID and version;
- time;
- the values that matched the condition;
- which level decided and who decided;
- the decision: approved, rejected, approved with edits, or timed out;
- time taken to decide;
- any later reversal, linked back to the firing (for example, a refund that was clawed back).

Timeouts are excluded from approval rates.

### 6.2 Patterns

Group firings by the values that drive the condition. For a threshold, use bands of the amount; for a list, use each value. For each group, compute:

- n, the number of decisions, and k, the number approved;
- the lower bound of the 95% Wilson score interval for k out of n;
- the number of distinct approvers;
- the number of days the decisions span;
- the number edited, the number reversed, and the median decision time.

### 6.3 Guardrails

Raise a suggestion only when all of the following hold. Make each threshold a company setting with these defaults:

- at least 30 decisions;
- spread over at least 30 days;
- by at least 3 different approvers;
- a Wilson lower bound of at least 0.85;
- no decision in the group was edited or reversed;
- the median decision time is above a floor, by default 30 seconds. Instant approvals are more likely rubber-stamping than judgement.

Never raise a suggestion for a `block` policy. Never raise one for a policy marked "Never suggest changes", which is the default for the Legal and compliance and Security business areas.

### 6.4 The suggestion card

Show the card in the policy form and in a Suggestions queue for owners. It contains:

- the proposed change in plain words, for example "Raise the limit from $500 to $800?" The new value must sit inside the range the evidence covers, never beyond it;
- the evidence in one sentence, the four statistics, and the **suggestion confidence**: High, or "Not enough yet" with the draft button turned off;
- three actions:
  - **Draft this change:** opens the next version as an unsaved draft. The condition and situation are updated, the examples are recomputed, the confirmation is cleared, and the change note is filled in with the evidence. Only the owner's save, followed by the normal activate step, makes it Active.
  - **Not yet:** hides the card until 30 more decisions arrive.
  - **Dismiss:** the same change is not suggested again unless its evidence is materially stronger, by default 30 more decisions.

Once a suggested change is saved, do not suggest it again. Learning only ever suggests loosening an approval policy. Suggestions to tighten one are out of scope for this release.

## 7. JSON (`hard-policy/2`)

Write a pure function that turns form state into this JSON, and validate the result against a JSON Schema checked into the repo.

```json
{
  "schema": "hard-policy/2",
  "id": "FIN-0012",
  "version": 2,
  "status": "live",
  "title": "Refund or credit over $500",
  "category": "Financial",
  "owner": "Chief Financial Officer",
  "businessArea": { "primary": "Finance", "subArea": "Refunds and credits" },
  "effective": { "from": "2026-02-01", "reviewBy": null },
  "scope": { "agents": ["*"], "tools": ["*"], "skills": ["*"] },
  "source": { "document": "Finance Payments Policy", "documentVersion": 1, "reference": "1.1", "excerpt": "Refunds or credits above $500 need approval from the Finance Manager." },
  "context": {
    "checkpoint": "tool.call.before",
    "actions": { "tools": ["refund.issue", "credit.issue"], "plain": "issuing a refund or credit" },
    "timeWindow": null,
    "units": { "currency": "USD", "convertOther": "rate_on_action_date", "amountsIncludeTax": true }
  },
  "inputs": [
    { "name": "Refund amount", "field": "args.amount", "type": "number", "unit": "USD", "from": "action", "showApprover": true, "sensitive": false },
    { "name": "Currency", "field": "args.currency", "type": "string", "from": "action", "showApprover": true, "sensitive": false },
    { "name": "Customer account", "field": "customer.id", "type": "string", "from": "lookup:crm.customer", "showApprover": true, "sensitive": true },
    { "name": "Agent's reason", "field": "agent.reason", "type": "string", "from": "action", "showApprover": true, "sensitive": false }
  ],
  "outputs": {
    "toAgent": {
      "onMatch": "paused_for_approval",
      "reasonCode": "FIN-0012.over_limit",
      "whilePaused": "no_retry",
      "messageForPerson": "Your refund needs a manager's approval. You will hear back within 4 hours."
    },
    "toApprover": { "show": ["args.amount", "args.currency", "customer.id", "agent.reason"], "options": ["approve", "reject"], "reasonRequiredOn": ["reject"] },
    "toNotify": null,
    "audit": { "logInputs": true, "mask": ["customer.id"] }
  },
  "trigger": {
    "plain": "The agent is about to issue a refund or credit above $500.",
    "events": ["tool.call.before"],
    "condition": { "all": [
      { "field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"] },
      { "field": "args.amount", "op": "gt", "value": 500 }
    ] },
    "onMissingData": "fail_closed",
    "setBy": "extraction_agent",
    "checkedBy": "user_8841",
    "checkedAt": "2026-10-08T09:14:00Z"
  },
  "enforcement": {
    "outcome": "approve",
    "intervention": {
      "escalateTo": [ { "type": "role", "name": "Finance Manager" }, { "type": "role", "name": "CFO" } ],
      "sla": { "source": "company_default", "respondWithin": "PT4H", "onTimeout": "escalate_next" }
    }
  },
  "learning": { "eligible": true }
}
```

Rules:

- **Outcome fields:**
  - `block` has no `intervention` and may have `notify`.
  - `notify` requires `notify` and has no `intervention`.
- **Timeout:** `onTimeout` is `escalate_next` when a policy has more than one level. At the last level, or when there is only one level, it is `reject`.
- **Missing data:** `onMissingData` defaults to `fail_closed` for `approve` and `block`. This is a company setting.
- **Orchestrator checks:** the orchestrator refuses a `live` policy that has no `checkedBy`, and refuses any condition that uses a name missing from the registry.
- **Conflict rules:** `conflicts[]` lists standing rules from decided policy conflicts, as `{ "with": "CUS-0004", "rule": "...", "caseId": "pc_...", "decidedAt": "..." }`. The orchestrator passes these to the decision engine and never interprets them itself.
- **Contract:** the contract (`context`, `inputs`, `outputs`) must agree with everything else:
  - every field in `trigger.condition` and `outputs.toApprover.show` appears in `inputs`;
  - every input is registered as available at `context.checkpoint`;
  - `trigger.events` equals `[context.checkpoint]`.

  The schema validation fails if any of these does not hold, and the orchestrator refuses the policy.
- **Learning flag:** `learning.eligible` is false for `block` and for "Never suggest changes" policies.

## 8. Acceptance tests

Write automated tests for each of these:

1. Three documents from three business areas go through one agent call. Every policy has a business area, sub-area, source and extraction confidence.
2. A tiered clause produces two policies, grouped under one source and not flagged as a conflict.
3. Every existing form field maps as in section 3.1. The hidden controls are not visible to policy owners.
4. Approve shows levels and response time; Block and Notify do not. Switching outcome leaves no stale fields in the JSON.
5. The response time default comes from the company setting, and an override is stored as an ISO duration.
6. Levels keep their order. A timeout escalates to the next level, and at the last level it rejects.
7. Example results are computed by code. A flipped example blocks Active. Editing the situation clears the confirmation. Confirming records who and when.
8. A condition that uses an unknown tool name blocks Active.
9. Active is refused with a full list of problems. A Draft saves with only a name. Each save creates a version, and Cancel restores the last one.
10. Extraction confidence is computed from the checks in section 3.6.
11. Conflict detection:
    - Two policies from different sources that can both match the same action, with different outcomes, raise a `policy` case. The case carries a code-found example that matches both.
    - Tiered policies, and same-outcome matches, raise no case.
12. Enforcement contract:
    - An input that is not available at the checkpoint shows "Can't be enforced yet" and blocks Active.
    - Amounts with no stated currency block Active.
    - The JSON validation fails when a condition field is missing from `inputs`.
    - The approval request shows exactly the approval-details inputs, with sensitive ones masked for everyone else.
    - The agent receives the documented result, reason code and message for each outcome.
    - Every firing writes an audit record.
13. A live multi-match pauses the action and sends a `live` case containing only the condition fields. The orchestrator has no precedence logic.
14. If the decision engine times out, or is unreachable, the action is rejected.
15. A returned decision is stored on both policies' history.
16. A "standing rule" decision is written into `conflicts[]`.
17. A "change" or "limit" decision opens a Draft and never saves it.
18. The same live conflict decided the same way 5 times raises a policy case.
19. Learning raises a suggestion only when every guardrail holds. Draft, Not yet and Dismiss behave as specified. There is no suggestion for block or "Never suggest changes" policies, and none after the change is saved. A suggested value never goes beyond the evidence.
20. The generated JSON passes the schema. The orchestrator refuses a live policy that has no `checkedBy`.
21. The CSV export neutralises formula injection.
22. There are no console errors.

## 9. Open decisions

Do not guess these. List them in your plan and wait for an answer.

- **Revised documents.** When a revised policy document is uploaded, map each new clause to an existing policy ID where it matches. Show changed policies as new draft versions. Show removed clauses as "Proposed retire". Confirm this behaviour before building it.
- **Second reviewer.** Do learning suggestions in high-risk categories need a second reviewer besides the owner?
- **Not allowed in a conflict.** When a Not allowed policy is in a live conflict, does it still block immediately, so the case is only for the record and a policy fix? My recommendation is yes. Or can the human approver override it?
- **Who approves a live conflict.** Is it the most senior decider among the matched policies, the policy owners, or a routing rule in the decision engine? My recommendation is to use the engine's own routing, and to fall back to the most senior decider.
- **Lookups and running totals.** Who builds and owns them, for example "refunds to this customer in the last 30 days": the orchestrator team or this product? Until one is registered, policies that need it stay "Can't be enforced yet".
- **Approve with changes.** May approvers change the request, such as lowering the amount, or only approve or reject it?
- **Defaults.** Confirm the company default response time and the learning thresholds.
