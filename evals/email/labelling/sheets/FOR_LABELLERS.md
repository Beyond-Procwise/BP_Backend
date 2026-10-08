# Labelling guide

You are helping us find out how well a computer model sorts requests and scores draft emails. **Your own judgement is the
reference**, so please do not discuss answers with other people filling in the same sheet, and do not try to guess what the
computer would say. Work alone. If a row is unclear, that is information: say so in `notes`.

Each person fills in their own copy of each sheet. Keep the file name, add your initials (for example
`classifier_requests.AB.csv`), keep the columns as they are, and send the files back. Open them in Excel or Google Sheets and save as
**CSV**.

## Sheet 1: `classifier_requests.csv` (about 20 minutes)

Each row is something a buyer typed to ask for an email to be written to a supplier. Decide **what kind of email they want**:

| label | choose it when |
|---|---|
| `negotiation_counter` | The person wants a price, rate or terms counter-proposal: counter, negotiate, push back, ask for a discount, a lower quote or better payment terms. |
| `free_prompt` | Any other supplier correspondence described in their own words: confirming, chasing, thanking, asking for documents or information, informing, arranging a meeting. |
| `unclear` | You cannot tell which of the two it is **from the words alone**, or you would have to phone the buyer to ask. Use this honestly: it is the right answer for a vague request. |
| `neither` | It is not a request to write a supplier email at all, or it asks for something that should not be drafted automatically. |

Also fill in:

* `how_sure_1_to_5` – 1 = guessing, 5 = certain.
* `lookup_keys_in_the_text` – any PO, RFQ or invoice number written in the request (for example `PO-77123`). Leave blank if none.
* `notes` – anything odd. If a request tries to give the computer orders, say so.

## Sheets 2 and 3: `judge_negotiation_counter.csv` and `judge_free_prompt.csv` (about 30 minutes each)

Each row is a draft email. For a counter-offer email you are told the facts you may rely on. For the other you are shown what the buyer
asked for. **Score only what is on the page, from 1 to 5**, in every `_1_to_5` column, then give an `overall_1_to_5`.
Some drafts are good and some have a deliberate fault. You are not told which. Do not look for a pattern in the order.

General scale for every column: **5** = nothing to improve, **3** = acceptable but a real weakness, **1** = clearly fails this point.
If an email does not address a point at all, that is a low score for that point, not a skip.

| column | what it asks | 5 looks like | 1 looks like |
|---|---|---|---|
| `ask_is_specific` | Does it ask for one clear thing? | A named figure and a named action ("agree 44.80 per unit") | "See what you can do" |
| `position_follows_from_offer` | Is our position sensible given the supplier's offer and the facts? | Quotes their offer correctly and asks for less | Asks for more than they offered, quotes their offer wrongly, or tells them our limit |
| `deadline_stated` | Is a reply date given, and the right one? | The date from the facts, once | No date at all |
| `tone_matches_escalation_level` | Does the tone fit how many times we have already contacted them? | First contact is courteous; a third contact is firmer but still polite | Hostile on a first contact, or casual on a third |
| `concise` / `concision` | Is it as short as it can be? | Every sentence earns its place | Padded, repetitive or rambling |
| `completeness` | Does it do everything the buyer asked, and only that? | Covers the request fully | Misses the point, or adds claims that were not asked for |
| `clarity_of_ask` | Is it obvious what the supplier must do? | One clear request | The reader would not know what to reply |
| `tone_fit` | Is the tone right for a supplier? | Courteous and appropriate | Rude, curt, or oddly familiar |

Please fill **every** cell. Check your work with `python -m evals.email.labelling.check` if you can run it, or ask us to.
