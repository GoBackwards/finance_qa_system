"""Hand-computed ranking metrics and source-preservation contracts."""

import copy
import unittest

from report_search.rerank_evaluation import evaluate_pair, summarize


def evidence(text="revenue", *, identifier="e", document="gold", start=0):
    return {"evidence_id": identifier, "kind": "prose", "document_id": document,
            "source_sha256": "hash", "pdf_page": 1, "element_id": identifier,
            "char_start": start, "char_end": start + len(text), "quote": text}


def chunk(identifier, *, document="gold", text="revenue", element="e", start=0, score=.5):
    atom = evidence(text, identifier=element, document=document, start=start)
    atom["text"] = atom.pop("quote")
    return {"chunk_id": identifier, "document_id": document, "text": text, "token_count": 10,
            "score": score, "evidence": [atom],
            "citations": [{"document_id": document, "pdf_page": 1, "sha256": "hash"}]}


def question(identifier="q", *, split="test", category="factual", gold=None):
    gold = gold if gold is not None else [evidence()]
    return {"question_id": identifier, "split": split, "category": category,
            "ground_truth_doc_ids": list(dict.fromkeys(item["document_id"] for item in gold)), "evidence": gold}


def selected(pool, indices):
    result = []
    for rank, index in enumerate(indices, 1):
        item = copy.deepcopy(pool[index])
        item.update(score=10 - rank, rank=rank, retrieval_approach="reranked",
                    vector_score=pool[index]["score"], vector_rank=pool[index].get("rank", index + 1))
        result.append(item)
    return result


class RerankEvaluationTests(unittest.TestCase):
    def test_multi_unit_recall_differs_from_hit_precision_and_accuracy(self):
        q = question(gold=[evidence(), evidence("cost", identifier="cost", document="second")])
        pool = [chunk("wrong", document="noise"), chunk("a"), chunk("a-overlap"),
                chunk("wrong2", document="noise"), chunk("wrong3", document="noise"),
                chunk("b", document="second", text="cost", element="cost")]
        measured = evaluate_pair(q, pool, selected(pool, [5, 1, 3, 4, 0]))
        direct = measured["approaches"]["direct_top5"]["metrics"]
        reranked = measured["approaches"]["reranked_top5"]["metrics"]
        self.assertEqual(direct["document_recall_at_5"], .5)
        self.assertEqual(direct["evidence_recall_at_5"], .5)
        self.assertEqual(direct["evidence_hit_at_5"], 1)
        self.assertEqual(direct["all_evidence_hit_at_5"], 0)
        self.assertEqual(direct["annotated_evidence_precision_at_5"], .4)
        self.assertEqual(direct["document_accuracy_at_1"], 0)
        self.assertEqual(direct["evidence_accuracy_at_1"], 0)
        self.assertEqual(reranked["evidence_recall_at_5"], 1)
        self.assertEqual(reranked["all_evidence_hit_at_5"], 1)
        self.assertEqual(reranked["annotated_evidence_precision_at_5"], .4)
        self.assertEqual(reranked["document_accuracy_at_1"], 1)
        self.assertEqual(reranked["evidence_accuracy_at_1"], 1)
        self.assertEqual(measured["deltas"]["evidence_recall_at_5"], .5)
        self.assertEqual(measured["approaches"]["candidate_top20"]["metrics"]["evidence_recall_at_20"], 1)

    def test_union_coverage_is_not_individual_chunk_precision(self):
        q = question(gold=[evidence("abcdefghi")])
        pool = [chunk("a", text="abcde"), chunk("b", text="defghi", start=3)]
        measured = evaluate_pair(q, pool, selected(pool, [1, 0]))
        for name in ("direct_top5", "reranked_top5"):
            metrics = measured["approaches"][name]["metrics"]
            self.assertEqual(metrics["evidence_recall_at_5"], 1)
            self.assertEqual(metrics["evidence_hit_at_5"], 1)
            self.assertEqual(metrics["all_evidence_hit_at_5"], 1)
            self.assertEqual(metrics["annotated_evidence_precision_at_5"], 0)
            self.assertEqual(metrics["evidence_accuracy_at_1"], 0)
            self.assertEqual(metrics["document_accuracy_at_1"], 1)

    def test_precision_uses_fixed_five_and_empty_pool_has_zero_metrics(self):
        pool = [chunk("one")]
        metrics = evaluate_pair(question(), pool, selected(pool, [0]))["approaches"]["direct_top5"]["metrics"]
        self.assertEqual(metrics["annotated_evidence_precision_at_5"], .2)
        self.assertEqual(metrics["returned_chunks_at_5"], 1)
        self.assertEqual(metrics["context_tokens_at_5"], 10)
        empty = evaluate_pair(question(), [], [])
        self.assertTrue(all(value == 0 for value in empty["approaches"]["direct_top5"]["metrics"].values()))

    def test_gold_document_alone_does_not_give_evidence_accuracy(self):
        pool = [chunk("wrong-text", text="xxxxxxx")]
        metrics = evaluate_pair(question(), pool, pool)["approaches"]["direct_top5"]["metrics"]
        self.assertEqual(metrics["document_accuracy_at_1"], 1)
        self.assertEqual(metrics["evidence_accuracy_at_1"], 0)

    def test_direct_prefix_is_unchanged_and_inputs_are_not_mutated(self):
        pool = [chunk(f"c{i}") for i in range(7)]
        reordered = selected(pool, [6, 4, 3, 2, 1])
        original = copy.deepcopy((pool, reordered))
        measured = evaluate_pair(question(), pool, reordered)
        self.assertEqual(measured["approaches"]["direct_top5"]["chunk_ids"], [f"c{i}" for i in range(5)])
        self.assertEqual(measured["approaches"]["reranked_top5"]["chunk_ids"], ["c6", "c4", "c3", "c2", "c1"])
        self.assertEqual((pool, reordered), original)
        self.assertNotIn("alpha", measured)

    def test_duplicate_outside_and_wrong_sized_selections_fail(self):
        pool = [chunk(f"c{i}") for i in range(6)]
        for candidates, reranked, error in (
            (pool + [pool[0]], pool[:5], "duplicate"),
            (pool, [pool[0]] * 5, "duplicate"),
            (pool, pool[:4] + [chunk("outside")], "outside"),
            (pool, pool[:4], "exactly"),
            (pool, pool, "exactly"),
            ([chunk(f"many{i}") for i in range(21)], [], "at most 20"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                evaluate_pair(question(), candidates, reranked)

    def test_source_payload_changes_are_rejected(self):
        pool = [chunk("one")]
        mutations = [lambda item: item.update(text="new"),
                     lambda item: item.update(document_id="new"),
                     lambda item: item["evidence"][0].update(char_start=1),
                     lambda item: item["citations"][0].update(pdf_page=2),
                     lambda item: item.update(token_count=99),
                     lambda item: item.update(new_payload="unexpected"),
                     lambda item: item.pop("evidence")]
        for mutate in mutations:
            reranked = selected(pool, [0])
            mutate(reranked[0])
            with self.assertRaisesRegex(ValueError, "source payload"):
                evaluate_pair(question(), pool, reranked)

    def test_changed_score_must_preserve_original_vector_score_and_rank(self):
        pool = [chunk("one")]
        reranked = selected(pool, [0])
        evaluate_pair(question(), pool, reranked)
        del reranked[0]["vector_score"]
        with self.assertRaisesRegex(ValueError, "retain the original vector score"):
            evaluate_pair(question(), pool, reranked)
        reranked[0]["score_breakdown"] = {"vector_raw_score": pool[0]["score"]}
        evaluate_pair(question(), pool, reranked)
        for field, value in (("vector_score", .123), ("vector_score", float("nan")), ("vector_rank", 9)):
            altered = selected(pool, [0])
            altered[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "retained original vector"):
                evaluate_pair(question(), pool, altered)

    def test_category_split_macro_aggregation_and_paired_intervals(self):
        pool = [chunk(f"bad{i}", document="noise") for i in range(5)] + [chunk("good")]
        improved = evaluate_pair(question("test", category="summary"), pool, selected(pool, [5, 0, 1, 2, 3]))
        unchanged = evaluate_pair(question("dev", split="dev", category="comparative"), pool, selected(pool, [0, 1, 2, 3, 4]))
        summary = summarize([improved, unchanged], bootstrap_replicates=200, seed=12)
        self.assertEqual(summary["overall"]["question_count"], 2)
        self.assertEqual(summary["overall"]["approaches"]["reranked_top5"]["metrics"]["evidence_recall_at_5"], .5)
        delta = summary["overall"]["paired_deltas"]["evidence_recall_at_5"]
        self.assertEqual(delta["mean_delta_pp"], 50)
        self.assertEqual(delta["improved_questions"], 1)
        self.assertEqual(delta["unchanged_questions"], 1)
        self.assertEqual(delta["worsened_questions"], 0)
        test = summary["splits"]["test"]
        self.assertEqual(test["paired_deltas"]["evidence_recall_at_5"]["ci95_pp"], [100, 100])
        self.assertEqual(test["by_category"]["summary"]["question_count"], 1)
        self.assertEqual(test["by_category"]["factual"]["question_count"], 0)
        self.assertEqual(summary, summarize([improved, unchanged], bootstrap_replicates=200, seed=12))
        self.assertIn("not answer accuracy", summary["metric_definitions"]["evidence_accuracy_at_1"])

    def test_summary_empty_and_invalid_rows_fail_clearly(self):
        empty = summarize([], bootstrap_replicates=1)
        self.assertEqual(empty["overall"]["question_count"], 0)
        self.assertEqual(empty["splits"]["test"]["paired_deltas"], {})
        row = evaluate_pair(question(), [], [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([row, row])
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                summarize([row], bootstrap_replicates=count)
        for field, value in (("split", "train"), ("category", "other"), ("question_id", "")):
            q = question()
            q[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_pair(q, [], [])


if __name__ == "__main__":
    unittest.main()
