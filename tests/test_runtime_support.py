import copy
import unittest

from report_search.runtime_support import DeadlineExceeded, RequestBudget, RetrievalCache


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def binding():
    return {"corpus": {"sha256": "reports-v1", "chunking": "table-preserving"},
            "encoder": {"revision": "bge-m3-v1", "rendering": "v1"},
            "index": {"revision": "index-v1", "kind": "flat", "top_k": 20},
            "reranker": {"revision": "reranker-v1", "top_k": 5},
            "query_policy": "original-query-v1"}


def passages(text="Revenue was $10."):
    return [{"chunk_id": "NVDA_FY2025:p1:row2", "text": text, "score": 2.0,
             "reranker_score": 2.0, "vector_score": 0.8, "rank": 1,
             "retrieval_approach": "vector_reranker",
             "citations": [{"doc_id": "NVDA_FY2025", "page": 1, "rows": [2]}]}]


class RetrievalCacheTests(unittest.TestCase):
    def test_exact_query_identity_preserves_year_basis_and_whitespace(self):
        cache = RetrievalCache()
        query = "NVIDIA FY2025 GAAP revenue"
        self.assertTrue(cache.put(query, binding(), passages()))
        self.assertEqual(cache.get(query, binding()), passages())
        for other in ("NVIDIA FY2024 GAAP revenue", "NVIDIA FY2025 non-GAAP revenue",
                      "nvidia FY2025 GAAP revenue", " " + query):
            with self.subTest(other=other):
                self.assertIsNone(cache.get(other, binding()))

    def test_all_binding_changes_invalidate_and_dict_order_does_not(self):
        cache = RetrievalCache()
        original = binding()
        cache.put("revenue", original, passages())
        self.assertIsNotNone(cache.get("revenue", dict(reversed(list(original.items())))))
        for field in original:
            changed = copy.deepcopy(original)
            changed[field] = {"changed": "new-revision"}
            with self.subTest(field=field):
                self.assertIsNone(cache.get("revenue", changed))
        extra = dict(original, tenant="different")
        self.assertIsNone(cache.get("revenue", extra))

    def test_binding_and_query_validation(self):
        cache = RetrievalCache()
        bad_bindings = [None, {}, {"corpus": "v1"}]
        for value in ("", {}, None, [], float("nan"), True):
            bad_bindings.append(dict(binding(), encoder=value))
        bad_bindings.extend([dict(binding(), extra=float("inf")),
                             dict(binding(), index={1: "ambiguous-key"}),
                             dict(binding(), extra=("tuple",))])
        for value in bad_bindings:
            with self.subTest(value=value), self.assertRaises(ValueError):
                cache.put("revenue", value, passages())
        for value in (None, "", " \n", 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cache.get(value, binding())

    def test_values_are_isolated_from_writes_and_reads(self):
        cache = RetrievalCache()
        original = passages()
        cache.put("revenue", binding(), original)
        original[0]["citations"][0]["rows"].append(99)
        first = cache.get("revenue", binding())
        self.assertEqual(first[0]["citations"][0]["rows"], [2])
        first[0]["citations"][0]["rows"].append(98)
        self.assertEqual(cache.get("revenue", binding()), passages())

    def test_ttl_is_from_write_and_reads_do_not_extend_expiry(self):
        clock = Clock()
        cache = RetrievalCache(ttl_seconds=10, clock=clock)
        cache.put("revenue", binding(), passages())
        clock.advance(9)
        self.assertIsNotNone(cache.get("revenue", binding()))
        clock.advance(1)
        self.assertIsNone(cache.get("revenue", binding()))
        self.assertEqual(cache.stats["entries"], 0)
        self.assertEqual(cache.stats["bytes"], 0)

    def test_lru_eviction_updates_on_reads(self):
        cache = RetrievalCache(max_entries=2)
        cache.put("a", binding(), passages())
        cache.put("b", binding(), passages())
        cache.get("a", binding())
        cache.put("c", binding(), passages())
        self.assertIsNone(cache.get("b", binding()))
        self.assertIsNotNone(cache.get("a", binding()))
        self.assertIsNotNone(cache.get("c", binding()))
        self.assertEqual(cache.stats["entries"], 2)

    def test_byte_limit_evicts_and_oversized_write_keeps_existing(self):
        probe = RetrievalCache()
        probe.put("a", binding(), passages())
        one_size = probe.stats["bytes"]
        cache = RetrievalCache(max_bytes=one_size * 2 - 1)
        cache.put("a", binding(), passages())
        cache.put("b", binding(), passages())
        self.assertIsNone(cache.get("a", binding()))
        self.assertEqual(cache.stats["bytes"], one_size)
        self.assertFalse(cache.put("b", binding(), passages("x" * one_size * 3)))
        self.assertEqual(cache.get("b", binding()), passages())

    def test_replacing_key_updates_bytes_and_ttl_without_spurious_eviction(self):
        clock = Clock()
        cache = RetrievalCache(max_entries=2, ttl_seconds=10, clock=clock)
        cache.put("a", binding(), passages())
        cache.put("b", binding(), passages())
        clock.advance(8)
        cache.put("a", binding(), passages("Changed source payload"))
        self.assertIsNotNone(cache.get("b", binding()))
        clock.advance(3)
        self.assertIsNone(cache.get("b", binding()))
        self.assertEqual(cache.get("a", binding()), passages("Changed source payload"))

    def test_empty_failure_bad_scores_and_nonreranked_results_never_cached(self):
        cache = RetrievalCache()
        cases = [[], None, {"status": "error"}, [None], passages() * 2]
        for field, value in (("score", float("nan")), ("reranker_score", float("inf")),
                             ("vector_score", float("nan")), ("score", True),
                             ("reranker_score", "2.0"), ("score", 0.9),
                             ("rank", True), ("rank", 1.0), ("rank", 2), ("chunk_id", ""),
                             ("retrieval_approach", "vector"), ("metadata", {"bad": float("inf")})):
            candidate = passages()
            candidate[0][field] = value
            cases.append(candidate)
        ascending = passages() + passages()
        ascending[1].update(chunk_id="second", rank=2, score=3.0, reranker_score=3.0)
        cases.append(ascending)
        missing_score = passages()
        del missing_score[0]["reranker_score"]
        cases.append(missing_score)
        for candidate in cases:
            with self.subTest(candidate=candidate):
                self.assertFalse(cache.put("query", binding(), candidate))
                self.assertIsNone(cache.get("query", binding()))
        self.assertEqual(cache.stats["entries"], 0)

    def test_expanded_reranked_results_preserve_rank_fusion_provenance(self):
        cache = RetrievalCache()
        expanded = passages()
        expanded[0].pop("vector_score")
        expanded[0].update(retrieval_approach="expanded_hybrid_reranker", candidate_score=0.03,
                           candidate_rank=2, retrieval_channels={"vector": 3, "lexical_0": 1})
        self.assertTrue(cache.put("query", binding(), expanded))
        self.assertEqual(cache.get("query", binding()), expanded)
        for field, value in (("candidate_score", float("nan")), ("candidate_score", True),
                             ("candidate_rank", None), ("candidate_rank", 1.0)):
            invalid = copy.deepcopy(expanded)
            invalid[0][field] = value
            self.assertFalse(cache.put("invalid", binding(), invalid))
        invalid = copy.deepcopy(expanded)
        del invalid[0]["candidate_score"]
        self.assertFalse(cache.put("invalid", binding(), invalid))
        mixed = expanded + passages()
        mixed[1].update(chunk_id="second", rank=2)
        self.assertFalse(cache.put("invalid", binding(), mixed))

    def test_clear_and_parameter_validation(self):
        for options in ({"max_entries": 0}, {"max_bytes": True}, {"ttl_seconds": 0},
                        {"ttl_seconds": float("nan")}, {"clock": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RetrievalCache(**options)
        cache = RetrievalCache()
        cache.put("query", binding(), passages())
        cache.clear()
        self.assertEqual(cache.stats["bytes"], 0)
        self.assertEqual(cache.stats["entries"], 0)


class RequestBudgetTests(unittest.TestCase):
    def test_stage_times_and_remaining_budget_are_measured(self):
        clock = Clock()
        budget = RequestBudget(3, clock=clock)
        with budget.stage("vector"):
            clock.advance(0.25)
        with budget.stage("reranker"):
            clock.advance(0.5)
        self.assertEqual(budget.elapsed_seconds, 0.75)
        self.assertEqual(budget.remaining_seconds, 2.25)
        self.assertEqual([row["duration_ms"] for row in budget.timings], [250, 500])
        report = budget.to_dict()
        self.assertFalse(report["deadline_exceeded"])
        self.assertTrue(report["cooperative_deadline"])
        report["stages"][0]["duration_ms"] = 999
        self.assertEqual(budget.timings[0]["duration_ms"], 250)

    def test_expired_stage_never_starts(self):
        clock = Clock()
        budget = RequestBudget(1, clock=clock)
        clock.advance(1)
        started = False
        with self.assertRaises(DeadlineExceeded) as raised:
            with budget.stage("generation"):
                started = True
        self.assertFalse(started)
        self.assertEqual(raised.exception.stage, "generation")
        self.assertEqual(budget.timings, [])

    def test_stage_can_overrun_but_no_later_stage_starts(self):
        clock = Clock()
        budget = RequestBudget(1, clock=clock)
        finished = False
        with self.assertRaises(DeadlineExceeded):
            with budget.stage("gpu"):
                clock.advance(2)
                finished = True
        self.assertTrue(finished)  # Cooperative checks do not pretend to cancel GPU work.
        self.assertEqual(budget.timings[0]["status"], "deadline_exceeded")
        self.assertEqual(budget.remaining_seconds, 0)
        with self.assertRaises(DeadlineExceeded):
            budget.check("generation")
        self.assertTrue(budget.to_dict()["deadline_exceeded"])

    def test_failed_stage_preserves_exception_and_records_sanitized_type(self):
        clock = Clock()
        budget = RequestBudget(1, clock=clock)
        with self.assertRaisesRegex(OSError, "secret-provider-body"):
            with budget.stage("provider"):
                clock.advance(2)
                raise OSError("secret-provider-body")
        self.assertEqual(budget.timings[0]["error_type"], "OSError")
        self.assertNotIn("secret", str(budget.to_dict()))
        with self.assertRaises(DeadlineExceeded):
            budget.check("verifier")

    def test_invalid_budget_parameters_and_stage_names(self):
        for value in (0, -1, True, float("nan"), float("inf"), "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RequestBudget(value)
        with self.assertRaises(ValueError):
            RequestBudget(1, clock=lambda: float("nan"))
        with self.assertRaises(ValueError):
            with RequestBudget(1).stage(""):
                self.fail("invalid stage ran")


if __name__ == "__main__":
    unittest.main()
