import copy
import unittest

from report_search.llm_client import ModelError
from report_search.reliable_pipeline import ReliableQAPipeline, reciprocal_rank_pool
from report_search.reranker import apply_scores
from report_search.runtime_support import RetrievalCache


TEXT = "Revenue was $10 million."
CANDIDATE = {"chunk_id": "c1", "document_id": "NVDA_FY2025", "ticker": "NVDA", "fiscal_year": 2025,
             "source_file": "NVDA_FY2025.pdf", "text": TEXT, "score": 0.8, "rank": 1, "evidence": []}
INVENTORY = [{"ticker": ticker, "company": company, "fiscal_year": year}
             for ticker, company in (("NVDA", "NVIDIA"), ("AMD", "Advanced Micro Devices")) for year in (2024, 2025)]
BINDING = {key: {"fingerprint": key + "-v1"} for key in ("corpus", "encoder", "index", "reranker", "query_policy")}
DRAFT = {"status": "answer", "claims": [{"text": TEXT, "citations": [{"evidence_id": "E1", "quote": TEXT}]}]}
VERDICT = {"answers_question": True, "claim_verdicts": [{"claim_index": 0, "supported": True, "reason": "Controlled support."}]}


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


class Encoder:
    def __init__(self): self.queries = []
    def encode_queries(self, queries):
        self.queries.extend(queries)
        return [[1.0, 0.0]]


class Engine:
    def __init__(self, pool=None):
        self.pool = copy.deepcopy([CANDIDATE] if pool is None else pool)
        self.lexical_calls = []
    def vector(self, vector, top_k): return copy.deepcopy(self.pool)
    def bm25(self, query, top_k):
        self.lexical_calls.append((query, top_k))
        return [{**copy.deepcopy(CANDIDATE), "chunk_id": "lexical", "score": -2.0}]


class Reranker:
    def __init__(self, scores=None): self.calls, self.scores = [], list(scores or [7.0])
    def rank_candidates(self, query, pool, top_k):
        self.calls.append((query, copy.deepcopy(pool), top_k))
        score = self.scores[min(len(self.calls) - 1, len(self.scores) - 1)]
        return apply_scores(pool, [score] * len(pool), top_k)


class Source:
    def __init__(self): self.calls, self.broken = 0, False
    def build(self, candidates):
        self.calls += 1
        if self.broken: raise ValueError("source hash changed")
        return {"source_verified": True, "chunks": [], "evidence": {"E1": {
            "evidence_id": "E1", "text": TEXT, "kind": "prose",
            "citation": {"document_id": "NVDA_FY2025", "source_file": "NVDA_FY2025.pdf",
                         "sha256": "a" * 64, "pdf_page": 7},
            "source_locator": {"element_id": "p7_e1", "char_start": 0, "char_end": len(TEXT)},
            "chunk_ids": [candidates[0]["chunk_id"]]}}}


class Client:
    def __init__(self, responses, *, clock=None, delay=0):
        self.responses, self.calls, self.clock, self.delay = list(responses), 0, clock, delay
    def complete_json(self, messages):
        self.calls += 1
        if self.clock: self.clock.advance(self.delay)
        result = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(result, Exception): raise result
        return copy.deepcopy(result)


class ReliablePipelineTests(unittest.TestCase):
    def build(self, **kwargs):
        clock = kwargs.pop("clock", Clock())
        encoder = kwargs.pop("encoder", Encoder())
        engine = kwargs.pop("retrieval_engine", Engine())
        reranker = kwargs.pop("reranker", Reranker())
        source = kwargs.pop("source_store", Source())
        generator = kwargs.pop("generator", Client([DRAFT]))
        verifier = kwargs.pop("verifier", Client([VERDICT]))
        pipeline = ReliableQAPipeline(inventory=INVENTORY, encoder=encoder, retrieval_engine=engine,
            reranker=reranker, source_store=source, generator=generator, verifier=verifier,
            binding=BINDING, threshold=kwargs.pop("threshold", 6.0), clock=clock,
            sleeper=clock.advance, **kwargs)
        return pipeline

    def test_success_has_stage_timings_and_preserves_question(self):
        p = self.build()
        query = "According to NVIDIA's FY2025 annual report, what was revenue?"
        result = p.answer(query)
        self.assertEqual(result["status"], "answer")
        self.assertEqual(p.encoder.queries, [query])
        self.assertEqual(p.reranker.calls[0][0], query)
        self.assertEqual(result["audit"]["runtime"]["reranker_pairs"], 1)
        self.assertIn("semantic_verification", [s["stage"] for s in result["audit"]["runtime"]["stages"]])

    def test_missing_companies_refuse_before_model_or_retrieval(self):
        for query in ("What was Apple's FY2024 gross margin?", "What is Tesla's projected revenue for 2027?"):
            p = self.build()
            result = p.answer(query)
            self.assertEqual(result["reason"], "company_not_in_corpus")
            self.assertFalse(p.encoder.queries)
            self.assertEqual(p.generator.calls, 0)

    def test_oversized_query_refuses_before_embedding(self):
        p = self.build()
        self.assertEqual(p.answer("x" * 10001)["reason"], "invalid_query")
        self.assertFalse(p.encoder.queries)

    def test_future_period_alone_does_not_trigger_corpus_refusal(self):
        p = self.build()
        result = p.answer("What does NVIDIA's FY2025 annual report expect for 2027?")
        self.assertEqual(result["status"], "answer")  # controlled support verdict, not forecast accuracy
        self.assertTrue(p.encoder.queries)

    def test_wrong_company_or_explicit_report_year_is_not_sent_to_generator(self):
        for changes in ({"ticker": "AMD"}, {"fiscal_year": 2024}):
            p = self.build(retrieval_engine=Engine([{**CANDIDATE, **changes}]))
            result = p.answer("According to NVIDIA's FY2025 annual report, what was revenue?")
            self.assertEqual(result["reason"], "retrieved_scope_incomplete")
            self.assertEqual(p.generator.calls, 0)

    def test_comparison_requires_both_companies_in_context(self):
        p = self.build()
        result = p.answer("Compare NVIDIA and AMD revenue.")
        self.assertEqual(result["reason"], "retrieved_scope_incomplete")
        self.assertEqual(p.generator.calls, 0)

    def test_comparison_requires_all_explicit_report_years_without_company(self):
        p = self.build()
        result = p.answer("Compare 2024 annual reports with 2025 annual reports.")
        self.assertEqual(result["reason"], "retrieved_scope_incomplete")
        self.assertEqual(p.generator.calls, 0)

    def test_bare_metric_period_can_use_later_comparative_filing(self):
        p = self.build()
        result = p.answer("What was NVIDIA's FY2020 revenue?")
        self.assertEqual(result["status"], "answer")  # temporal support remains the verifier's responsibility

    def test_low_score_default_does_not_expand(self):
        p = self.build(reranker=Reranker([-2.0]))
        result = p.answer("What was NVIDIA revenue?")
        self.assertEqual(result["reason"], "below_threshold")
        self.assertFalse(p.engine.lexical_calls)

    def test_bounded_fallback_retains_original_query_and_separate_threshold(self):
        p = self.build(reranker=Reranker([-2.0, 7.0]), enable_expansion=True, expansion_threshold=6.5)
        query = "What was NVIDIA's FY2025 revenue?"
        result = p.answer(query)
        self.assertEqual(result["status"], "answer")
        self.assertTrue(result["audit"]["runtime"]["expansion_used"])
        self.assertLessEqual(len(p.engine.lexical_calls), 3)
        self.assertLessEqual(result["audit"]["runtime"]["reranker_pairs"], 40)
        self.assertTrue(all(call[0] == query for call in p.reranker.calls))
        self.assertTrue(all(call[0].startswith(query) for call in p.engine.lexical_calls))
        with self.assertRaises(ValueError): self.build(enable_expansion=True)

    def test_cache_reuses_only_retrieval_and_rechecks_sources_and_models(self):
        clock = Clock()
        cache = RetrievalCache(clock=clock)
        p = self.build(clock=clock, cache=cache)
        first = p.answer("What was NVIDIA revenue?")
        second = p.answer("What was NVIDIA revenue?")
        self.assertEqual(first["status"], "answer")
        self.assertTrue(second["audit"]["runtime"]["cache_hit"])
        self.assertEqual(len(p.encoder.queries), 1)
        self.assertEqual(p.source_store.calls, 2)
        self.assertEqual(p.generator.calls, 2)
        p.source_store.broken = True
        third = p.answer("What was NVIDIA revenue?")
        self.assertEqual(third["reason"], "source_verification_failed")
        self.assertEqual(p.generator.calls, 2)

    def test_transient_errors_retry_once_and_auth_does_not_retry(self):
        p = self.build(generator=Client([ModelError("model_http_429"), DRAFT]))
        result = p.answer("What was NVIDIA revenue?")
        self.assertEqual(result["status"], "answer")
        self.assertEqual(p.generator.calls, 2)
        for error in ("model_http_401", "invalid_model_json", "incomplete_model_response"):
            p = self.build(generator=Client([ModelError(error)]))
            result = p.answer("What was NVIDIA revenue?")
            self.assertEqual(result["status"], "abstain")
            self.assertEqual(p.generator.calls, 1)
        p = self.build(generator=Client([ModelError("model_http_503")]))
        self.assertEqual(p.answer("What was NVIDIA revenue?")["status"], "abstain")
        self.assertEqual(p.generator.calls, 2)

    def test_late_generation_is_discarded_without_verifier_and_audit_is_truthful(self):
        clock = Clock()
        p = self.build(clock=clock, deadline_seconds=3, generator=Client([DRAFT], clock=clock, delay=4))
        result = p.answer("What was NVIDIA revenue?")
        self.assertEqual(result["reason"], "deadline_exceeded")
        self.assertTrue(result["audit"]["generation_called"])
        self.assertEqual(p.verifier.calls, 0)
        self.assertEqual(result["answer"], "I'm not sure")

    def test_slow_cache_write_cannot_release_late_answer(self):
        clock = Clock()
        class SlowCache(RetrievalCache):
            def put(self, *args):
                clock.advance(4)
                return super().put(*args)
        p = self.build(clock=clock, deadline_seconds=3, cache=SlowCache(clock=clock))
        self.assertEqual(p.answer("What was NVIDIA revenue?")["reason"], "deadline_exceeded")

    def test_refusals_are_not_cached(self):
        cache = RetrievalCache()
        bad = copy.deepcopy(VERDICT)
        bad["claim_verdicts"][0]["supported"] = False
        p = self.build(cache=cache, verifier=Client([bad]))
        self.assertEqual(p.answer("What was NVIDIA revenue?")["reason"], "unsupported_claim")
        self.assertEqual(cache.stats["entries"], 0)

    def test_invented_reranker_chunk_refuses_before_source_or_generator(self):
        class BadReranker(Reranker):
            def rank_candidates(self, query, pool, top_k):
                result = super().rank_candidates(query, pool, top_k)
                result[0]["chunk_id"] = "outside-pool"
                return result
        p = self.build(reranker=BadReranker())
        self.assertEqual(p.answer("What was NVIDIA revenue?")["reason"], "retrieval_failed")
        self.assertEqual(p.source_store.calls, 0)

    def test_conflicting_fusion_payloads_fail(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_pool([("vector", [CANDIDATE]), ("lexical", [{**CANDIDATE, "text": "tampered"}])])


if __name__ == "__main__": unittest.main()
