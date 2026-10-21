"""Policy documents are documentation; enforced policy lives in the policy engine; extracted figures are records.

Three different kinds of thing reached the answer under the wrong names:

  * policy PDFs (pgvector, collection ``static_policy``) were labelled "Procurement policy", which reads as
    an enforced rule when it is documentation that gets quoted;
  * the governed policy rows (``proc.bp_policy``, the rules the agents are held to) were labelled
    "Your extracted documents" and introduced to the model as "counted records from the customer's own
    documents";
  * when a document and the engine disagree, nothing said which wins.

So: documentation is named for its file and quoted, never stated as fact; enforced policy is stated as fact; and
the enforced policy wins where they differ. The policy lookup also returned the first ten policies alphabetically,
not the ones the question was about.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from services import corpus_facts, source_labels
from services.model_selector import RAGPipeline


@pytest.fixture()
def pipeline() -> RAGPipeline:
    obj = RAGPipeline.__new__(RAGPipeline)
    obj.rag = SimpleNamespace(
        learning_collection="learning",
        primary_collection="procwise_document_embeddings",
        uploaded_collection="uploaded_documents",
        static_policy_collection="static_policy",
    )
    return obj


class TestPolicyPdfsAreLabelledAsDocumentation:
    def test_the_static_policy_collection_is_policy_documentation(self, pipeline):
        label = pipeline._label_for_collection("static_policy")
        assert label == source_labels.POLICY_DOCUMENTATION
        assert label != "Procurement policy"

    def test_other_collections_keep_their_labels(self, pipeline):
        assert pipeline._label_for_collection("procwise_document_embeddings") == "ProcWise knowledge base"
        assert pipeline._label_for_collection("uploaded_documents") == "Uploaded reference"

    def test_a_quoted_excerpt_names_its_source_file(self, pipeline):
        hit = SimpleNamespace(
            payload={
                "collection_name": "static_policy",
                "filename": "Delegation of Authority.pdf",
                "title": "Delegation of Authority",
                "content": "Purchase orders above the limit need the Chief Procurement Officer to approve.",
            },
            combined_score=0.7,
            rerank_score=0.7,
        )
        (item,) = pipeline._prepare_knowledge_items([hit])
        assert item["source_label"] == source_labels.POLICY_DOCUMENTATION
        assert item["document"] == "Delegation of Authority.pdf"

    def test_a_non_policy_document_is_named_as_before(self, pipeline):
        hit = SimpleNamespace(
            payload={"title": "Master Services Agreement", "filename": "msa.pdf", "content": "Term is 24 months."},
            combined_score=0.5,
            rerank_score=0.5,
        )
        (item,) = pipeline._prepare_knowledge_items([hit])
        assert item["document"] == "Master Services Agreement"


class TestGovernedPolicyIsNotLabelledAsExtractedData:
    def test_policy_facts_are_enforced_policy(self):
        assert corpus_facts.fact_source("policies") == (source_labels.ENFORCED_POLICY, source_labels.ENFORCED_POLICY)

    @pytest.mark.parametrize("intent", ["purchase_orders", "invoices", "spend", "suppliers", "quotes", "deals", "findings", None])
    def test_everything_else_stays_extracted_data(self, intent):
        assert corpus_facts.fact_source(intent) == (source_labels.EXTRACTED_RECORDS, "Extracted documents (live)")


class TestThePromptTreatsEachKindDifferently:
    def _prompt(self, pipeline, **kw):
        return pipeline._compose_llm_prompt("Who can approve a PO?", "snippets", "draft", None, "", **kw)

    def test_enforced_policy_is_stated_as_fact_and_not_called_a_customer_document(self, pipeline):
        prompt = self._prompt(pipeline, corpus_context="policies in force:\n  - policy_name=ApprovalThresholdPolicy",
                              corpus_intent="policies")
        assert source_labels.ENFORCED_POLICY_HEADING in prompt
        assert "ApprovalThresholdPolicy" in prompt
        assert "counted records" not in prompt
        assert "customer's own documents" not in prompt

    def test_extracted_facts_keep_the_records_heading(self, pipeline):
        prompt = self._prompt(pipeline, corpus_context="totals:\n  - purchase_orders_total=5042", corpus_intent="purchase_orders")
        assert "Verified data from the customer's own documents" in prompt
        assert source_labels.ENFORCED_POLICY_HEADING not in prompt

    def test_documentation_is_to_be_quoted_and_attributed(self, pipeline):
        prompt = self._prompt(pipeline, has_policy_docs=True)
        assert source_labels.DOCUMENTATION_NOTE in prompt

    def test_no_documentation_note_without_policy_documents(self, pipeline):
        assert source_labels.DOCUMENTATION_NOTE not in self._prompt(pipeline)

    def test_the_engine_wins_when_both_are_present(self, pipeline):
        prompt = self._prompt(pipeline, corpus_context="policies in force:\n  - policy_name=P", corpus_intent="policies",
                              has_policy_docs=True)
        assert source_labels.PRECEDENCE_NOTE in prompt

    def test_no_precedence_note_with_only_one_kind(self, pipeline):
        only_docs = self._prompt(pipeline, has_policy_docs=True)
        only_engine = self._prompt(pipeline, corpus_context="policies in force:\n  - policy_name=P", corpus_intent="policies")
        assert source_labels.PRECEDENCE_NOTE not in only_docs
        assert source_labels.PRECEDENCE_NOTE not in only_engine


_POLICIES = [
    {"policy_name": "AgentLifecycleAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may create or delete an agent."},
    {"policy_name": "ApprovalThresholdPolicy", "policy_type": "approval",
     "policy_desc": "Spend authority gate: amounts at or below the threshold auto-approve; above it escalate."},
    {"policy_name": "DailyBriefSettingsWriteAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may change the daily brief settings."},
    {"policy_name": "DocumentIntakeAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may put documents into the product."},
    {"policy_name": "EntityProfileWriteAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may change the company profile."},
    {"policy_name": "GovernanceAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may edit or reload the rules that govern everything else."},
    {"policy_name": "MailboxAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may change which mailbox is watched."},
    {"policy_name": "ModelAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may retrain or repoint a model."},
    {"policy_name": "PlaybookAuthorityPolicy", "policy_type": "authority", "policy_desc": "Who may author a playbook and who may approve one."},
    {"policy_name": "ReportSignoffPolicy", "policy_type": "approval", "policy_desc": "Which reports need a person's sign-off."},
    {"policy_name": "ZZLateApprovalLimitPolicy", "policy_type": "approval",
     "policy_desc": "Approval limit by role: who may approve a purchase order up to each value."},
]


def _fake_embed(texts):
    """A deterministic stand-in: a text's vector is its bag of 5-letter word stems, hashed into 64 unit-length slots."""
    import hashlib
    import math
    import re

    out = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-z]{4,}", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t).lower()):
            v[int(hashlib.md5(w[:5].encode()).hexdigest(), 16) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _ranked(rows, question, embed):
    """Ask twice: the first call starts the background warm-up of the policy vectors; the second uses them."""
    corpus_facts.relevant_policies(rows, question, embed)
    thread = corpus_facts._policy_warming
    if thread is not None:
        thread.join(timeout=10)
    return corpus_facts.relevant_policies(rows, question, embed)


@pytest.fixture(autouse=True)
def _cold_cache():
    corpus_facts._policy_vectors.clear()
    yield
    corpus_facts._policy_vectors.clear()


class TestThePolicyLookupReturnsTheRelevantPolicies:
    QUESTION = "What is the approval limit by role for approving a purchase order?"

    def test_the_closest_policies_come_back_and_unrelated_ones_do_not(self):
        names = [r["policy_name"] for r in _ranked(_POLICIES, self.QUESTION, _fake_embed)]
        assert names[0] == "ZZLateApprovalLimitPolicy"
        assert "ApprovalThresholdPolicy" in names
        assert "MailboxAuthorityPolicy" not in names
        assert len(names) <= 5

    def test_the_first_question_does_not_wait_for_the_policy_vectors(self):
        """Embedding every policy is slow on a CPU. The first request gets the old list and the cache fills behind it."""
        import threading

        gate, started = threading.Event(), threading.Event()

        def slow(texts):
            if len(texts) > 1:
                started.set()
                gate.wait(timeout=10)
            return _fake_embed(texts)

        first = corpus_facts.relevant_policies(_POLICIES, self.QUESTION, slow)
        assert started.wait(timeout=5)
        assert first == _POLICIES[:10]
        gate.set()
        corpus_facts._policy_warming.join(timeout=10)
        assert corpus_facts.relevant_policies(_POLICIES, self.QUESTION, slow)[0]["policy_name"] == "ZZLateApprovalLimitPolicy"

    def test_without_a_question_the_old_behaviour_stays(self):
        assert corpus_facts.relevant_policies(_POLICIES, "", _fake_embed) == _POLICIES[:10]

    def test_without_an_embedder_the_old_behaviour_stays(self):
        assert corpus_facts.relevant_policies(_POLICIES, self.QUESTION) == _POLICIES[:10]

    def test_a_failing_embedder_falls_back_to_the_old_behaviour(self):
        def broken(texts):
            raise RuntimeError("model unavailable")

        assert _ranked(_POLICIES, self.QUESTION, broken) == _POLICIES[:10]

    def test_each_policy_is_embedded_once(self):
        calls = []

        def counting(texts):
            calls.append(list(texts))
            return _fake_embed(texts)

        _ranked(_POLICIES, self.QUESTION, counting)
        corpus_facts.relevant_policies(_POLICIES, "Who may approve a quote?", counting)
        assert sum(len(c) for c in calls if len(c) > 1) == len(_POLICIES)

    def test_a_relevant_policy_beyond_the_first_ten_is_still_found(self):
        """The old lookup took ten rows in table order and never saw the eleventh."""

        class Cursor:
            description = [("policy_name",), ("policy_type",), ("policy_desc",), ("version",)]

            def execute(self, sql, params):
                self._rows = [(r["policy_name"], r["policy_type"], r["policy_desc"], 1) for r in _POLICIES][: params[0]]

            def fetchall(self):
                return self._rows

        corpus_facts._fetch(Cursor(), "policies", self.QUESTION, _fake_embed)          # starts the warm-up
        corpus_facts._policy_warming.join(timeout=10)
        facts = corpus_facts._fetch(Cursor(), "policies", self.QUESTION, _fake_embed)
        assert facts["policies_in_force"][0]["policy_name"] == "ZZLateApprovalLimitPolicy"
        assert "ZZLateApprovalLimitPolicy" not in [r["policy_name"] for r in corpus_facts._fetch(Cursor(), "policies")["policies_in_force"]]
