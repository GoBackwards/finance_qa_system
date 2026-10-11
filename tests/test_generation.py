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

    def test_more_passages_than_top5_plus_extras_refuse_before_any_work(self):
        from report_search.generation import MAX_PASSAGES
        from report_search.scoped_retrieval import MAX_SUMMARY_EXTRAS
        self.assertEqual(MAX_PASSAGES, 5 + MAX_SUMMARY_EXTRAS)
        passages = [{"chunk_id": f"c{number}", "reranker_score": 7.0} for number in range(MAX_PASSAGES + 1)]
        result, source, generator, verifier = self.run_answer(candidates=passages)
        self.assertEqual(result["reason"], "too_many_passages")
        self.assertEqual(source.calls, 0)
        self.assertFalse(result["audit"]["generation_called"] or generator.calls or verifier.calls)
        result, *_ = self.run_answer(candidates=passages[:MAX_PASSAGES], threshold=None)
        self.assertEqual(result["reason"], "unconfigured_threshold")  # past the length check

    def test_valid_quote_does_not_allow_false_claim(self):
        # A number absent from the citation is now rejected deterministically.
        draft = copy.deepcopy(DRAFT)
        draft["claims"][0]["text"] = "Revenue was $100 million."
        result, _, _, verifier = self.run_answer(draft=draft)
        self.assertEqual(result["answer"], "I'm not sure")
        self.assertEqual(result["reason"], "claim_number_not_in_evidence")
        self.assertEqual(result["citations"], [])
        self.assertFalse(verifier.calls)
        self.assertTrue(result["reference_verified"])
        self.assertFalse(result["semantic_supported"])
        self.assertNotIn("100 million", str(result))
        # A false claim with grounded numbers still needs the semantic verifier.
        draft["claims"][0]["text"] = "Revenue fell to $10 million because of tariffs."
        verdict = copy.deepcopy(VERDICT)
        verdict["claim_verdicts"][0]["supported"] = False
        result, _, _, verifier = self.run_answer(draft=draft, verdict=verdict)
        self.assertEqual(result["reason"], "unsupported_claim")
        self.assertTrue(verifier.calls)
        self.assertNotIn("tariffs", str(result))

    def test_models_receive_compact_sources_without_provenance_payloads(self):
        result, _, generator, verifier = self.run_answer()
        context = json.loads(generator.calls[0][1]["content"])
        self.assertEqual(context["sources"][0]["evidence"], [["E1", 7, "prose", TEXT]])
        self.assertEqual(context["sources"][0]["report_fiscal_year"], 2025)
        for payload in (generator.calls[0][1]["content"], verifier.calls[0][1]["content"]):
            self.assertNotIn("sha256", payload)
            self.assertNotIn("bbox", payload)
        self.assertEqual(result["citations"][0]["citation"]["sha256"], "a" * 64)

    def test_scope_notes_reach_both_models_and_the_audit(self):
        scope = {"tickers": ["DEMO"], "years": [], "period_mode": "relative_latest", "latest_year": {"DEMO": 2025},
                 "notes": ["'last year' was interpreted as the latest available fiscal year: Demo FY2025."],
                 "report_years": {"DEMO": [2025]}}
        result, _, generator, verifier = self.run_answer(scope=scope)
        for model in (generator, verifier):
            context = json.loads(model.calls[0][1]["content"])
            self.assertEqual(context["query_scope"]["notes"], scope["notes"])
            self.assertNotIn("report_years", context["query_scope"])
        self.assertEqual(result["audit"]["query_scope"]["companies"], ["DEMO"])

    def test_oversized_context_drops_lowest_ranked_passages_instead_of_refusing(self):
        class GrowingSource(Source):
            def build(self, candidates):
                self.calls += 1
                bundle = copy.deepcopy(BUNDLE)
                for index in range(2, len(candidates) + 1):
                    identifier = f"E{index}"
                    bundle["evidence"][identifier] = dict(copy.deepcopy(BUNDLE["evidence"]["E1"]),
                                                          evidence_id=identifier, text="x" * 4000)
                    bundle["chunks"].append({"chunk_id": f"c{index}", "text": "x" * 4000, "evidence_ids": [identifier]})
                return bundle

        candidates = [{"chunk_id": f"c{index}", "reranker_score": 7.0} for index in range(1, 6)]
        source, generator = GrowingSource(), Model(DRAFT)
        result = generate_answer("What was revenue?", candidates, threshold=6, source_store=source,
                                 generator=generator, verifier=Model(VERDICT), max_context_chars=6_000)
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["audit"]["context_passages"], 2)
        self.assertEqual(result["audit"]["context_passages_dropped"], 3)
        self.assertLessEqual(result["audit"]["context_characters"], 6_000)

    def test_typography_only_quote_difference_is_repaired_to_exact_source_text(self):
        source_text = "Revenue was $10 million."
        bundle = copy.deepcopy(BUNDLE)
        bundle["evidence"]["E1"]["text"] = bundle["chunks"][0]["text"] = source_text
        draft = copy.deepcopy(DRAFT)
        draft["claims"][0]["text"] = "Revenue was $10 million."
        draft["claims"][0]["citations"][0]["quote"] = "Revenue was $10 million."
        with patch.object(Source, "build", return_value=bundle):
            result, *_ = self.run_answer(draft=draft)
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["citations"][0]["quote"], source_text)
        self.assertEqual(result["audit"]["citation_repairs"][0]["kind"], "typography")

    def test_claim_period_absent_from_citations_is_rejected_before_review(self):
        draft = copy.deepcopy(DRAFT)
        draft["claims"][0]["text"] = "Revenue is projected to be $10 million in 2027."
        result, _, _, verifier = self.run_answer(draft=draft)
        self.assertEqual(result["reason"], "claim_period_not_in_evidence")
        self.assertFalse(verifier.calls)

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



class SequenceModel(Model):
    """Returns the queued responses in order, recording every request."""

    def __init__(self, *responses):
        super().__init__(None)
        self.responses = list(responses)

    def complete_json(self, messages):
        self.calls.append(messages)
        return copy.deepcopy(self.responses.pop(0))


class RepairRoundTests(unittest.TestCase):
    def test_one_feedback_round_fixes_an_inexact_quote_and_is_fully_rechecked(self):
        bad = copy.deepcopy(DRAFT)
        bad["claims"][0]["citations"][0]["quote"] = "Revenue was 10 million dollars."
        generator, verifier = SequenceModel(bad, DRAFT), Model(VERDICT)
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6, source_store=Source(),
                                 generator=generator, verifier=verifier)
        self.assertEqual(result["status"], "answer")
        self.assertEqual(len(generator.calls), 2)
        feedback = generator.calls[1][-1]["content"]
        self.assertIn("quote_not_in_source", feedback)
        self.assertIn("E1", feedback)
        self.assertEqual(generator.calls[1][-2]["role"], "assistant")
        self.assertEqual(result["audit"]["first_attempt_reason"], "quote_not_in_source")
        self.assertEqual(result["audit"]["generation_attempts"], 2)
        stages = [row["stage"] for row in result["audit"]["stage_timings"]]
        self.assertEqual(stages[:4], ["source_binding", "generation", "reference_validation", "generation_retry"])

    def test_retry_is_single_and_abstentions_are_never_retried(self):
        bad = copy.deepcopy(DRAFT)
        bad["claims"][0]["text"] = "Revenue was $100 million."
        generator, verifier = SequenceModel(bad, bad), Model(VERDICT)
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6, source_store=Source(),
                                 generator=generator, verifier=verifier)
        self.assertEqual(result["reason"], "claim_number_not_in_evidence")
        self.assertEqual(len(generator.calls), 2)
        self.assertFalse(verifier.calls)
        abstaining = SequenceModel({"status": "abstain", "claims": []})
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6, source_store=Source(),
                                 generator=abstaining, verifier=Model(VERDICT))
        self.assertEqual(result["reason"], "generator_abstained")
        self.assertEqual(len(abstaining.calls), 1)


class PackingTests(unittest.TestCase):
    def growing_source(self):
        class GrowingSource(Source):
            def build(self, candidates):
                self.calls += 1
                bundle = {"source_verified": True, "evidence": {}, "chunks": []}
                for index, candidate in enumerate(candidates, 1):
                    identifier = f"E{index}"
                    text = TEXT if candidate["chunk_id"] == "c1" else "x" * 3000
                    bundle["evidence"][identifier] = dict(copy.deepcopy(BUNDLE["evidence"]["E1"]),
                                                          evidence_id=identifier, text=text)
                    bundle["chunks"].append({"chunk_id": candidate["chunk_id"], "text": text,
                                             "evidence_ids": [identifier], "ticker": candidate["ticker"],
                                             "fiscal_year": candidate["fiscal_year"]})
                return bundle
        return GrowingSource()

    def test_packing_never_drops_a_companys_or_years_last_passage(self):
        candidates = [{"chunk_id": "c1", "reranker_score": 7.0, "ticker": "PEP", "fiscal_year": 2022},
                      {"chunk_id": "c2", "reranker_score": 6.9, "ticker": "PEP", "fiscal_year": 2022},
                      {"chunk_id": "c3", "reranker_score": 6.8, "ticker": "PEP", "fiscal_year": 2022},
                      {"chunk_id": "c4", "reranker_score": 6.7, "ticker": "KO", "fiscal_year": 2022}]
        generator = Model(DRAFT)
        result = generate_answer("Coke vs Pepsi?", candidates, threshold=6, source_store=self.growing_source(),
                                 generator=generator, verifier=Model(VERDICT), max_context_chars=8_000)
        context = json.loads(generator.calls[0][1]["content"])
        self.assertEqual([source["ticker"] for source in context["sources"]], ["PEP", "PEP", "KO"])
        self.assertEqual(result["audit"]["context_passages_dropped"], 1)

    def test_gate_is_rechecked_on_the_packed_passages(self):
        # Unsorted caller input: the only high-scoring passage is last and oversized.
        candidates = [{"chunk_id": "c3", "reranker_score": 1.0, "ticker": "PEP", "fiscal_year": 2022},
                      {"chunk_id": "c1", "reranker_score": 1.0, "ticker": "PEP", "fiscal_year": 2022},
                      {"chunk_id": "c2", "reranker_score": 9.0, "ticker": "PEP", "fiscal_year": 2022}]
        generator = Model(DRAFT)
        result = generate_answer("Revenue?", candidates, threshold=6, source_store=self.growing_source(),
                                 generator=generator, verifier=Model(VERDICT), max_context_chars=5_000)
        self.assertEqual(result["reason"], "below_threshold")
        self.assertEqual(result["audit"]["context_passages_dropped"], 1)
        self.assertFalse(generator.calls)


class ModelViewTests(unittest.TestCase):
    def test_header_rows_are_marked_for_the_models(self):
        bundle = copy.deepcopy(BUNDLE)
        bundle["evidence"]["E2"] = dict(copy.deepcopy(BUNDLE["evidence"]["E1"]), evidence_id="E2", kind="row",
                                        text=" | 2024 | 2023", source_locator={"table_id": "t", "row_index": 0,
                                                                              "role": "header"})
        bundle["chunks"][0]["evidence_ids"] = ["E2", "E1"]
        from report_search.generation import model_sources
        units = model_sources(bundle)[0]["evidence"]
        self.assertEqual(units[0][2], "header_row")
        self.assertEqual(units[1][2], "prose")



class AdmissionRetryTests(unittest.TestCase):
    def test_admitted_gap_is_not_retried_away(self):
        admitted = copy.deepcopy(DRAFT)
        admitted["claims"].append({"text": "The R&D share is not provided in the sources.",
                                   "citations": [{"evidence_id": "E1", "quote": TEXT}]})
        generator = SequenceModel(admitted, DRAFT)
        result = generate_answer("What was revenue?", CANDIDATES, threshold=6, source_store=Source(),
                                 generator=generator, verifier=Model(VERDICT))
        self.assertEqual(result["reason"], "claim_admits_missing_evidence")
        self.assertEqual(len(generator.calls), 1)



class PromptTests(unittest.TestCase):
    def test_prompts_name_no_corpus_company(self):
        from pathlib import Path
        from report_search.generation import GENERATOR_SYSTEM, VERIFIER_SYSTEM
        from report_search.query_scope import ScopeResolver
        manifest = Path(__file__).resolve().parents[1] / "data/manifest.json"
        resolver = ScopeResolver(json.loads(manifest.read_text(encoding="utf-8-sig")))
        self.assertEqual(resolver.companies_in_text(GENERATOR_SYSTEM + VERIFIER_SYSTEM), frozenset())

    def test_comparisons_must_state_their_result(self):
        from report_search.generation import GENERATOR_SYSTEM
        self.assertIn("which is higher, larger, lower or better, add one claim that states that result", GENERATOR_SYSTEM)


if __name__ == "__main__":
    unittest.main()
