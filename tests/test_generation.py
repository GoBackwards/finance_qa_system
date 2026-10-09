import copy
import unittest
from unittest.mock import Mock, patch

from report_search.generation import generate_answer, answer_with_retrieval
from report_search.llm_client import ModelError


TEXT = "Revenue was $10 million."
BUNDLE = {"source_verified": True, "evidence": {"E1": {
    "evidence_id": "E1", "text": TEXT, "kind": "prose",
    "citation": {"document_id": "DEMO_FY2025_annual_report", "source_file": "DEMO_FY2025_annual_report.pdf",
                 "sha256": "a" * 64, "pdf_page": 7, "bbox": [0, 0, 100, 100], "ticker": "DEMO", "fiscal_year": 2025},
    "source_locator": {"element_id": "p7_b2", "char_start": 0, "char_end": len(TEXT)}, "chunk_ids": ["c1"]}},
    "chunks": [{"chunk_id": "c1", "text": TEXT, "evidence_ids": ["E1"]}]}
DRAFT = {"status": "answer", "claims": [{"text": TEXT, "citations": [{"evidence_id": "E1", "quote": TEXT}]}]}
VERDICT = {"answers_question": True, "claim_verdicts": [{"claim_index": 0, "supported": True, "reason": "Directly stated."}]}
CANDIDATES = [{"chunk_id": "c1", "reranker_score": 7.0}]


class Source:
    def __init__(self):
        self.calls = 0

    def build(self, candidates):
        self.calls += 1
        return copy.deepcopy(BUNDLE)


class Model:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return copy.deepcopy(self.response)


class GenerationTests(unittest.TestCase):
    def run_answer(self, *, candidates=None, draft=None, verdict=None, **options):
        source = Source()
        generator = Model(DRAFT if draft is None else draft)
        verifier = Model(VERDICT if verdict is None else verdict)
        result = generate_answer("What was revenue?", CANDIDATES if candidates is None else candidates,
                                 source_store=source, generator=generator, verifier=verifier,
                                 threshold=options.pop("threshold", 6.0), **options)
        return result, source, generator, verifier

    def test_success_has_verified_inline_citation_and_source_trace(self):
        result, source, generator, verifier = self.run_answer()
        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["reference_verified"])
        self.assertTrue(result["semantic_supported"])
        self.assertIn("10 million", result["answer"])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(result["citations"][0]["source_text"], TEXT)
        self.assertNotIn("reranker_score", generator.calls[0][1]["content"])

    def test_low_or_unconfigured_confidence_never_reads_sources_or_calls_model(self):
        for options in ({"threshold": None}, {"candidates": []}, {"candidates": [{"reranker_score": 1.0}]},
                        {"candidates": [{"reranker_score": float("nan")}]}, {"candidates": [{"score": 9.0}]}):
            with self.subTest(options=options):
                result, source, generator, verifier = self.run_answer(**options)
                self.assertEqual(result["answer"], "I'm not sure")
                self.assertEqual(source.calls, 0)
                self.assertFalse(generator.calls or verifier.calls)

    def test_valid_quote_does_not_allow_false_claim(self):
        draft = copy.deepcopy(DRAFT)
        draft["claims"][0]["text"] = "Revenue was $100 million."
        verdict = copy.deepcopy(VERDICT)
        verdict["claim_verdicts"][0]["supported"] = False
        result, _, _, verifier = self.run_answer(draft=draft, verdict=verdict)
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertEqual(result["citations"], [])
        self.assertTrue(verifier.calls)
        self.assertTrue(result["reference_verified"])
        self.assertFalse(result["semantic_supported"])
        self.assertNotIn("100 million", str(result))

    def test_irrelevant_or_incomplete_answer_is_rejected(self):
        verdict = copy.deepcopy(VERDICT)
        verdict["answers_question"] = False
        result, *_ = self.run_answer(verdict=verdict)
        self.assertEqual(result["status"], "abstain")

    def test_unknown_reference_does_not_reach_semantic_verifier(self):
        draft = copy.deepcopy(DRAFT)
        draft["claims"][0]["citations"][0]["evidence_id"] = "E999"
        result, _, _, verifier = self.run_answer(draft=draft)
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertFalse(verifier.calls)

    def test_source_tampering_and_budget_fail_before_generation(self):
        with patch.object(Source, "build", side_effect=ValueError("Changed source hash")):
            result, _, generator, _ = self.run_answer()
            self.assertEqual(result["reason"], "source_verification_failed")
            self.assertFalse(generator.calls)
        result, _, generator, _ = self.run_answer(max_context_chars=10)
        self.assertEqual(result["reason"], "context_budget_exceeded")
        self.assertFalse(generator.calls)

    def test_malformed_source_store_fails_closed(self):
        for value in (None, [], "invalid", {"source_verified": True, "evidence": 1}):
            with patch.object(Source, "build", return_value=value):
                result, _, generator, _ = self.run_answer()
                self.assertEqual(result["answer"], "I'm not sure")
                self.assertFalse(generator.calls)

    def test_provider_errors_and_bad_verdict_fail_closed(self):
        for options in ({"draft": ModelError("model_unavailable")}, {"verdict": ModelError("model_unavailable")},
                        {"verdict": {"answers_question": True, "claim_verdicts": []}},
                        {"draft": {"status": "abstain", "claims": []}}):
            result, *_ = self.run_answer(**options)
            self.assertEqual(result["answer"], "I'm not sure")
            self.assertEqual(result["citations"], [])

    def test_existing_retrieval_stack_is_composed_with_all_answer_checks(self):
        encoder, engine, reranker = Mock(), Mock(), Mock()
        encoder.encode_queries.return_value = [[1.0, 0.0]]
        engine.vector.return_value = [{"chunk_id": "pool"}]
        reranker.rank_candidates.return_value = CANDIDATES
        result = answer_with_retrieval("What was revenue?", encoder=encoder, retrieval_engine=engine,
                                      reranker=reranker, threshold=6.0, source_store=Source(),
                                      generator=Model(DRAFT), verifier=Model(VERDICT))
        self.assertEqual(result["status"], "answer")
        engine.vector.assert_called_once_with([1.0, 0.0], top_k=20)
        reranker.rank_candidates.assert_called_once_with("What was revenue?", [{"chunk_id": "pool"}], top_k=5)

    def test_unconfigured_live_pipeline_does_no_retrieval(self):
        encoder = Mock()
        result = answer_with_retrieval("What was revenue?", encoder=encoder, retrieval_engine=Mock(),
                                      reranker=Mock(), threshold=None)
        self.assertEqual(result["answer"], "I'm not sure")
        encoder.encode_queries.assert_not_called()


if __name__ == "__main__":
    unittest.main()
