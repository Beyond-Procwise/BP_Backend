-- email_tone_rules: the six tone variables, where each comes from, and what "unknown" means.
--
-- FOR REVIEW. NOT APPLIED to any database. Values below that are judgement (the seniority
-- keyword lists, the country-to-formality map, the escalation thresholds) are proposals; each
-- is a row edit, not a deploy, once this is live.
--
-- A GAP (no stored data to derive a variable from) is handled per variable by `on_gap`:
--     default   the unknown_default applies and the source is recorded as "default"
--     assume    as above, AND the default becomes an assumption the reviewer must confirm,
--               so the draft is not ready until a person has said it is right
--   (`ask` is deliberately not offered: it needs a question the reviewer can answer, and the
--    only such question today is the classifier's, which is now answered the same way.)
-- An instruction that uses a tone word nothing maps (see tone_cues) is shown as an assumption
-- too, never ignored. An instruction with no tone word in it ("confirm the price") is a task,
-- not a tone request, and is left alone.
--
-- Three rules the code applies on top of this row:
--   1. Every variable resolves to exactly one of three sources, recorded with it:
--        postgres          derived from a stored row (supplier master, prior contacts)
--        user_instruction  the person's own words matched an override below
--        default           no data, so the variable's unknown_default applies -- never a guess
--   2. user_instruction overrides everything, including a value derived from Postgres.
--   3. A value outside `allowed` is refused when the row is read, not at draft time.

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT
 'EmailToneRules', 'email_tone_rules',
 'Allowed values, derivations, gap handling, unknown defaults and instruction overrides for the email tone variables.',
 $json$
{
 "policy_identifier": "email_tone_rules",
 "required_role": "Admin",
 "rules": {
  "variables": {
   "relationship_tier": {
    "allowed": [
     "strategic",
     "preferred",
     "standard",
     "new"
    ],
    "unknown_default": "standard",
    "on_gap": "default",
    "derive": {
     "kind": "supplier_flag",
     "column": "is_preferred_supplier",
     "map": {
      "true": "preferred",
      "false": "standard"
     }
    }
   },
   "escalation_level": {
    "allowed": [
     1,
     2,
     3,
     4
    ],
    "unknown_default": 1,
    "on_gap": "assume",
    "derive": {
     "kind": "prior_contacts",
     "levels": [
      {
       "min": 0,
       "value": 1
      },
      {
       "min": 1,
       "value": 2
      },
      {
       "min": 3,
       "value": 3
      },
      {
       "min": 5,
       "value": 4
      }
     ]
    }
   },
   "leverage": {
    "allowed": [
     "low",
     "medium",
     "high"
    ],
    "unknown_default": "medium",
    "on_gap": "default"
   },
   "recipient_seniority": {
    "allowed": [
     "junior",
     "manager",
     "senior",
     "executive"
    ],
    "unknown_default": "manager",
    "on_gap": "assume",
    "derive": {
     "kind": "supplier_keyword",
     "column": "contact_role_1",
     "rules": [
      {
       "contains": [
        "chief",
        "ceo",
        "cfo",
        "coo",
        "cpo",
        "president",
        "owner",
        "managing director"
       ],
       "value": "executive"
      },
      {
       "contains": [
        "director",
        "head of",
        "vice president",
        "vp "
       ],
       "value": "senior"
      },
      {
       "contains": [
        "manager",
        "lead",
        "supervisor"
       ],
       "value": "manager"
      },
      {
       "contains": [
        "assistant",
        "coordinator",
        "analyst",
        "clerk",
        "administrator"
       ],
       "value": "junior"
      }
     ]
    }
   },
   "relationship_health": {
    "allowed": [
     "strained",
     "neutral",
     "good"
    ],
    "unknown_default": "neutral",
    "on_gap": "default"
   },
   "region_formality": {
    "allowed": [
     "high",
     "medium",
     "low"
    ],
    "unknown_default": "medium",
    "on_gap": "default",
    "derive": {
     "kind": "supplier_map",
     "column": "country",
     "map": {
      "Germany": "high",
      "Japan": "high",
      "China": "high",
      "India": "high",
      "United Arab Emirates": "high",
      "France": "high",
      "Italy": "high",
      "Spain": "high",
      "United Kingdom": "medium",
      "Ireland": "medium",
      "Netherlands": "medium",
      "Poland": "medium",
      "United States": "low",
      "Australia": "low",
      "Canada": "low"
     }
    }
   },
   "warmth": {
    "allowed": [
     "cool",
     "neutral",
     "warm"
    ],
    "unknown_default": "neutral",
    "on_gap": "default"
   },
   "directness": {
    "allowed": [
     "indirect",
     "balanced",
     "direct"
    ],
    "unknown_default": "balanced",
    "on_gap": "default"
   }
  },
  "tone_cues": "\\b(apolog\\w*|sorry|regret\\w*|appreciat\\w*|thank\\w*|grateful|warm\\w*|friendly|cordial|courteous|polite\\w*|collaborat\\w*|partner\\w*|firm\\w*|stern\\w*|strict\\w*|tough|aggressive|assertive|blunt|brusque|curt|terse|brief|concise|short|urgent\\w*|asap|pressing|soft\\w*|gentl\\w*|light|relaxed|casual\\w*|informal\\w*|formal\\w*|diplomatic|tactful|direct\\w*|candid|neutral|cold|cool|aloof|detached|matter\\w*|businesslike)\\b",
  "instruction_overrides": [
   {
    "match": "\\b(keep it light|gentle|gently|softly|relaxed|low[- ]key)\\b",
    "set": {
     "escalation_level": 1,
     "directness": "indirect"
    }
   },
   {
    "match": "\\b(warm|warmly|friendly|cordial|courteous|polite|politely|appreciative|appreciate|appreciation|thank you|thanks|grateful)\\b",
    "set": {
     "warmth": "warm"
    }
   },
   {
    "match": "\\b(apologetic|apologise|apologize|apology|sorry|regret)\\b",
    "set": {
     "warmth": "warm",
     "directness": "indirect",
     "escalation_level": 1
    }
   },
   {
    "match": "\\b(collaborative|collaboratively|work together|partnership|win[- ]win)\\b",
    "set": {
     "warmth": "warm",
     "relationship_health": "good",
     "directness": "balanced"
    }
   },
   {
    "match": "\\b(push hard|firm|firmly|firm stance|stern|strict|tough|assertive|escalate|final (offer|position))\\b",
    "set": {
     "escalation_level": 3,
     "directness": "direct"
    }
   },
   {
    "match": "\\b(urgent|urgently|asap|time[- ]sensitive|pressing)\\b",
    "set": {
     "escalation_level": 3,
     "directness": "direct"
    }
   },
   {
    "match": "\\b(concise|brief|briefly|short|to the point|blunt|curt|terse|direct|directly|straightforward|candid)\\b",
    "set": {
     "directness": "direct"
    }
   },
   {
    "match": "\\b(neutral|matter[- ]of[- ]fact|businesslike)\\b",
    "set": {
     "warmth": "neutral",
     "directness": "balanced"
    }
   },
   {
    "match": "\\b(cold|cool|aloof|detached)\\b",
    "set": {
     "warmth": "cool"
    }
   },
   {
    "match": "\\b(diplomatic|tactful|tactfully)\\b",
    "set": {
     "directness": "indirect"
    }
   },
   {
    "match": "\\b(helping us elsewhere|good relationship|valued partner)\\b",
    "set": {
     "relationship_health": "good"
    }
   },
   {
    "match": "\\b(formal|formally)\\b",
    "set": {
     "region_formality": "high"
    }
   },
   {
    "match": "\\b(casual|casually|informal|informally)\\b",
    "set": {
     "region_formality": "low"
    }
   }
  ]
 }
}
$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailToneRules');
