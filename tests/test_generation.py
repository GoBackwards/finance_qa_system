import copy
import json
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

    def test_late_generator_is_discarded_before_verification(self):
        class Clock:
            value = 0.0
            def __call__(self):
                return self.value
        clock = Clock()
        class SlowModel(Model):
            def complete_json(self, messages):
                clock.value += 4
                return super().complete_json(messages)
        verifier = Model(VERDICT)
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6,
                                 source_store=Source(), generator=SlowModel(DRAFT), verifier=verifier,
                                 deadline_seconds=3, clock=clock)
        self.assertEqual(result["reason"], "deadline_exceeded")
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertFalse(verifier.calls)
        self.assertFalse(result["audit"]["semantic_verifier_called"])
        self.assertEqual(result["audit"]["generation_elapsed_ms"], 4000)
        self.assertTrue(result["audit"]["generation_budget"]["cooperative_deadline"])

    def test_slow_source_binding_never_starts_generation(self):
        clock = Mock(side_effect=[0, 0, 0, 4, 4, 4, 4])
        generator = Model(DRAFT)
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6,
                                 source_store=Source(), generator=generator, verifier=Model(VERDICT),
                                 deadline_seconds=3, clock=clock)
        self.assertEqual(result["reason"], "deadline_exceeded")
        self.assertFalse(generator.calls)
        self.assertFalse(result["audit"]["generation_called"])

    def test_default_timing_preserves_answer_and_records_all_stages(self):
        result, *_ = self.run_answer()
        stages = [row["stage"] for row in result["audit"]["stage_timings"]]
        self.assertEqual(stages, ["source_binding", "generation", "reference_validation",
                                  "semantic_verification", "verdict_validation", "rendering"])
        self.assertGreaterEqual(result["audit"]["generation_elapsed_ms"], 0)
        self.assertNotIn("generation_budget", result["audit"])

    def test_invalid_deadlines_are_configuration_errors(self):
        for value in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_answer(deadline_seconds=value)

    def test_final_deadline_snapshot_agrees_with_answer_and_elapsed_time(self):
        from report_search.answer_verification import render_verified_answer

        class Clock:
            value = 0.0
            remaining = None

            def __call__(self):
                if self.remaining is not None:
                    self.remaining -= 1
                    if self.remaining <= 0:
                        self.value = 3.1
                return self.value

        # Exercise expiry at the final snapshot, and between the old separate
        # final check and snapshot. One snapshot must decide the public result.
        for advance_after in (2, 3):
            with self.subTest(advance_after=advance_after):
                clock = Clock()

                def render(*args):
                    result = render_verified_answer(*args)
                    clock.value = 2.9
                    clock.remaining = advance_after
                    return result

                with patch("report_search.generation.render_verified_answer", side_effect=render):
                    result, *_ = self.run_answer(deadline_seconds=3, clock=clock)
                snapshot = result["audit"]["generation_budget"]
                self.assertEqual(result["audit"]["generation_elapsed_ms"], snapshot["elapsed_ms"])
                if snapshot["deadline_exceeded"]:
                    self.assertEqual(result["status"], "abstain")
                    self.assertEqual(result["reason"], "deadline_exceeded")
                    self.assertEqual(result["citations"], [])
                else:
                    self.assertEqual(result["status"], "answer")
                self.assertEqual(snapshot["deadline_exceeded"], advance_after == 2)

    def test_late_verifier_and_rendering_never_release_verified_answer(self):
        from report_search.answer_verification import render_verified_answer

        class Clock:
            value = 0.0

            def __call__(self):
                return self.value

        for slow_stage in ("semantic_verification", "rendering"):
            with self.subTest(stage=slow_stage):
                clock = Clock()

                class Verifier(Model):
                    def complete_json(self, messages):
                        if slow_stage == "semantic_verification":
                            clock.value = 4.0
                        return super().complete_json(messages)

                def render(*args):
                    result = render_verified_answer(*args)
                    clock.value = 4.0
                    return result

                generator, verifier = Model(DRAFT), Verifier(VERDICT)
                with patch("report_search.generation.render_verified_answer", side_effect=render):
                    result = generate_answer("What was revenue?", CANDIDATES, threshold=6,
                                             source_store=Source(), generator=generator, verifier=verifier,
                                             deadline_seconds=3, clock=clock)
                self.assertEqual(result["answer"], "I'm not sure")
                self.assertEqual(result["reason"], "deadline_exceeded")
                self.assertEqual(result["citations"], [])
                self.assertFalse(result["semantic_supported"])
                self.assertTrue(result["audit"]["generation_called"])
                self.assertTrue(result["audit"]["semantic_verifier_called"])
                last_stage = result["audit"]["stage_timings"][-1]
                self.assertEqual(last_stage["stage"], slow_stage)
                self.assertEqual(last_stage["status"], "deadline_exceeded")

    def test_expiry_before_generator_preserves_call_flags(self):
        class Clock:
            value = 0.0

            def __call__(self):
                return self.value

        clock = Clock()

        class DelayedIdentity(Model):
            @property
            def identity(self):
                clock.value = 4.0
                return {"backend": "injected"}

        generator, verifier = DelayedIdentity(DRAFT), Model(VERDICT)
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6,
                                 source_store=Source(), generator=generator, verifier=verifier,
                                 deadline_seconds=3, clock=clock)
        self.assertEqual(result["reason"], "deadline_exceeded")
        self.assertFalse(generator.calls or verifier.calls)
        self.assertFalse(result["audit"]["generation_called"])
        self.assertFalse(result["audit"]["semantic_verifier_called"])
        self.assertEqual([stage["stage"] for stage in result["audit"]["stage_timings"]], ["source_binding"])

    def test_unpaired_surrogate_query_stops_before_sources_or_retrieval(self):
        for query in ("What was revenue?\ud800", "\udfff revenue"):
            with self.subTest(query=repr(query)):
                source, generator, verifier = Source(), Model(DRAFT), Model(VERDICT)
                result = generate_answer(query, CANDIDATES, threshold=6, source_store=source,
                                         generator=generator, verifier=verifier)
                self.assertEqual(result["reason"], "invalid_query")
                self.assertEqual(result["answer"], "I'm not sure")
                self.assertEqual(source.calls, 0)
                self.assertFalse(generator.calls or verifier.calls)
                encoder, engine, reranker = Mock(), Mock(), Mock()
                retrieved = answer_with_retrieval(query, encoder=encoder, retrieval_engine=engine,
                                                 reranker=reranker, threshold=6)
                self.assertEqual(retrieved["reason"], "invalid_query")
                encoder.encode_queries.assert_not_called()
                engine.vector.assert_not_called()
                reranker.rank_candidates.assert_not_called()

    def test_unencodable_source_context_fails_before_fingerprinting_or_model_call(self):
        broken = copy.deepcopy(BUNDLE)
        broken["evidence"]["E1"]["text"] += "\ud800"
        with patch.object(Source, "build", return_value=broken):
            result, _, generator, verifier = self.run_answer()
        self.assertEqual(result["reason"], "source_verification_failed")
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertNotIn("context_sha256", result["audit"])
        self.assertFalse(generator.calls or verifier.calls)
        json_ready = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.assertTrue(json_ready)

    def test_valid_multilingual_query_remains_unchanged(self):
        query = "营收是多少？ Résumé 📊"
        generator = Model(DRAFT)
        result = generate_answer(query, CANDIDATES, threshold=6, source_store=Source(),
                                 generator=generator, verifier=Model(VERDICT))
        self.assertEqual(result["status"], "answer")
        self.assertIn(query, generator.calls[0][1]["content"])
        self.assertTrue(result["answer"].encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
