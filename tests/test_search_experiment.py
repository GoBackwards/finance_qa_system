"""Evidence metrics, development-only selection, and persisted evaluation contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from report_index.faiss_index import sha256_file
from report_search.experiment import (
    _load_query_cache, aggregate, evaluate_question, paired_interval, run_evaluation,
    select_alpha, write_json,
)


def evidence(document, text="revenue", element="e"):
    return {"evidence_id": document + element, "kind": "prose", "document_id": document,
            "source_sha256": "hash", "pdf_page": 1, "element_id": element,
            "char_start": 0, "char_end": len(text), "quote": text}


def chunk(identifier, document, score, text="revenue"):
    gold = evidence(document, text)
    return {"chunk_id": identifier, "document_id": document, "text": text, "score": score,
            "token_count": 10, "evidence": [{**gold, "text": text}]}


def question(identifier, split="dev", documents=("gold",)):
    return {"question_id": identifier, "category": "factual", "query": "What is revenue?",
            "split": split, "ground_truth_doc_ids": list(documents),
            "evidence": [evidence(document) for document in documents]}


class ExperimentTests(unittest.TestCase):
    def test_document_recall_deduplicates_reports_but_is_not_evidence_recall(self):
        q = question("q", documents=("a", "b"))
        rows = [chunk("a1", "a", 1, "wrong"), chunk("a2", "a", .9, "wrong")]
        measured = evaluate_question(q, rows, (5,))
        self.assertEqual(measured["metrics"]["document_recall_at_5"], .5)
        self.assertEqual(measured["metrics"]["evidence_recall_at_5"], 0)
        self.assertEqual(measured["metrics"]["all_documents_hit_at_5"], 0)
        self.assertEqual(measured["metrics"]["context_tokens_at_5"], 20)

    def test_macro_average_weights_questions_equally(self):
        single = evaluate_question(question("one", documents=("a",)), [chunk("a1", "a", 1)], (5,))
        pair = evaluate_question(question("pair", documents=("b", "c")), [chunk("b1", "b", 1)], (5,))
        self.assertEqual(aggregate([single, pair])["metrics"]["evidence_recall_at_5"], .75)
        self.assertEqual(pair["metrics"]["all_evidence_hit_at_5"], 0)

    def test_tuning_rejects_test_rows_and_misaligned_questions(self):
        row = evaluate_question(question("dev"), [], (5,))
        test = {**row, "split": "test"}
        with self.assertRaisesRegex(ValueError, "development judgments only"):
            select_alpha({0: [row], 1: [test]})
        with self.assertRaisesRegex(ValueError, "same ordered"):
            select_alpha({0: [row], 1: [{**row, "question_id": "different"}]})
        self.assertEqual(select_alpha({0: [row], .5: [row], 1: [row]}), .5)
        self.assertEqual(select_alpha({.25: [row], .75: [row]}), .25)

    def test_full_run_freezes_dev_choice_before_opposite_test_result(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)

            class Engine:
                stats = {"document_count": 2, "chunk_count": 12}

                def vector(self, vector, top_k):
                    self.is_test = bool(vector[0])
                    if self.is_test:
                        frozen = json.loads((output / "tuning.json").read_text())
                        if frozen["selected_alpha"] != 0:
                            raise AssertionError("Test began before development weight was frozen")
                    ids = ["gold"] + [f"bad{i}" for i in range(5)]
                    if not self.is_test:
                        ids.reverse()
                    return [chunk(name, name, 1 - i / 10) for i, name in enumerate(ids)][:top_k]

                def bm25(self, text, top_k):
                    if text != "What is revenue?":
                        raise AssertionError("Retrieval received something other than query text")
                    ids = ["gold"] + [f"bad{i}" for i in range(5)]
                    if self.is_test:
                        ids.reverse()
                    return [chunk(name, name, -10 + i) for i, name in enumerate(ids)][:top_k]

            result = run_evaluation(Engine(), [question("dev"), question("test", "test")],
                                    np.asarray([[0], [1]]), output, alphas=(0, 1),
                                    candidate_k=6, cutoffs=(5,))
            self.assertEqual(result["tuning"]["selected_alpha"], 0)
            test = result["splits"]["test"]
            self.assertEqual(test["vector"]["overall"]["metrics"]["evidence_recall_at_5"], 1)
            self.assertEqual(test["hybrid"]["overall"]["metrics"]["evidence_recall_at_5"], 0)
            self.assertTrue((output / "comparison.md").is_file())
            self.assertEqual(result["per_query_sha256"], sha256_file(output / "per_query.jsonl"))

    def test_query_cache_checks_identity_hash_shape_and_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            contract = {"query_count": 2, "dimension": 2, "model": "pinned"}
            path = output / "query_vectors.npy"
            np.save(path, np.eye(2, dtype=np.float32))
            write_json(output / "query_manifest.json", {"contract": contract, "vectors_file_sha256": sha256_file(path)})
            np.testing.assert_array_equal(_load_query_cache(output, contract), np.eye(2))
            self.assertIsNone(_load_query_cache(output, {**contract, "model": "different"}))
            np.save(path, np.ones((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "checksum"):
                _load_query_cache(output, contract)
            write_json(output / "query_manifest.json", {"contract": contract, "vectors_file_sha256": sha256_file(path)})
            with self.assertRaisesRegex(ValueError, "unit vectors"):
                _load_query_cache(output, contract)

    def test_paired_bootstrap_requires_matching_query_order(self):
        row = evaluate_question(question("q"), [], (5,))
        self.assertEqual(paired_interval([row], [copy.deepcopy(row)])["ci95_pp"], [0, 0])
        with self.assertRaisesRegex(ValueError, "identical question order"):
            paired_interval([row], [{**row, "question_id": "other"}])


if __name__ == "__main__":
    unittest.main()
