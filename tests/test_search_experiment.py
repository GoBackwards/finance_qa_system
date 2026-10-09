"""Evidence metrics, development-only selection, and persisted evaluation contracts."""

import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from report_index.faiss_index import sha256_file
from report_search.experiment import (
    DEFAULT_CUTOFFS, _load_query_cache, aggregate, evaluate_question, main, paired_interval,
    run_evaluation, select_alpha, validate_cutoffs, write_comparison, write_json,
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
    def test_multi_cutoff_recall_uses_exact_rank_prefixes_and_is_monotonic(self):
        ranks = (3, 5, 10, 20, 50)
        documents = tuple(f"gold{rank}" for rank in ranks)
        q = question("multi", documents=documents)
        rows = [chunk(f"row{i}", f"gold{i}" if i in ranks else f"distractor{i}", 1 - i / 100)
                for i in range(1, 52)]
        measured = evaluate_question(q, rows)
        self.assertEqual(DEFAULT_CUTOFFS, ranks)
        metrics = measured["metrics"]
        for position, k in enumerate(ranks, 1):
            self.assertEqual(metrics[f"document_recall_at_{k}"], position / len(ranks))
            self.assertEqual(metrics[f"evidence_recall_at_{k}"], position / len(ranks))
            self.assertEqual(metrics[f"all_evidence_hit_at_{k}"], int(k == 50))
            self.assertEqual(metrics[f"context_tokens_at_{k}"], k * 10)
        # A relevant chunk immediately after k must not receive early credit.
        boundary = evaluate_question(question("boundary", documents=("gold5",)), rows, (3, 5))
        self.assertEqual(boundary["metrics"]["evidence_recall_at_3"], 0)
        self.assertEqual(boundary["metrics"]["evidence_recall_at_5"], 1)

    def test_cutoff_validation_rejects_empty_duplicate_and_invalid_ranks(self):
        for cutoffs in ((), (0, 5), (-1, 5), (True, 5), (3.0, 5), (3, 3, 5)):
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                validate_cutoffs(cutoffs)
        self.assertEqual(validate_cutoffs((50, 3, 5)), (3, 5, 50))
        self.assertEqual(validate_cutoffs((3,)), (3,))
        with self.assertRaisesRegex(ValueError, "Recall@5"):
            validate_cutoffs((3, 10), require_five=True)

    def test_invalid_run_configuration_fails_before_engine_or_output_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "not-created"
            for options in ({"cutoffs": (3, 10)}, {"cutoffs": (3, 5, 50), "candidate_k": 20},
                            {"cutoffs": (5, 5)}, {"cutoffs": ()}, {"alphas": (float("nan"),)},
                            {"candidate_k": True}, {"candidate_k": 100.5}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_evaluation(None, [], [], output, **options)
                self.assertFalse(output.exists())

    def test_cli_rejects_invalid_parameters_before_reading_corpus_or_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "not-created"
            for options in (("--cutoffs", "3", "10"), ("--cutoffs", "5", "5"),
                            ("--cutoffs", "0", "5"), ("--candidate-k", "20"),
                            ("--alphas", "nan"), ("--threads", "0")):
                with self.subTest(options=options), patch.object(Path, "read_text", side_effect=AssertionError("Preflight must precede reads")), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(["--output", str(output), *options])
                    self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

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
            report = (output / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("| Recall@5 |", report)
            self.assertNotIn("Recall@10", report)
            self.assertEqual(result["per_query_sha256"], sha256_file(output / "per_query.jsonl"))

    def test_report_has_dynamic_cutoffs_and_category_context_costs(self):
        rows = [evaluate_question(question("test", "test"), [chunk("gold", "gold", 1)])]
        measured = {"overall": aggregate(rows), "by_category": {
            "factual": aggregate(rows), "comparative": aggregate([]), "summary": aggregate([])}}
        summary = {"cutoffs": list(DEFAULT_CUTOFFS), "benchmark_provenance": {},
                   "corpus": {"document_count": 1, "chunk_count": 1},
                   "tuning": {"selected_alpha": .5, "grid": [{"alpha": .5, **aggregate(rows)}]},
                   "splits": {"test": {name: measured for name in ("vector", "bm25", "hybrid")}},
                   "limitations": ["Fixture benchmark"]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.md"
            write_comparison(path, summary)
            text = path.read_text(encoding="utf-8")
        self.assertIn("| Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |", text)
        self.assertIn("| All evidence@3 | All evidence@5 | All evidence@10 | All evidence@20 | All evidence@50 |", text)
        self.assertIn("| Mean tokens@3 | Mean tokens@5 | Mean tokens@10 | Mean tokens@20 | Mean tokens@50 |", text)
        self.assertIn("| factual | hybrid | 1 | 10 | 10 | 10 | 10 | 10 |", text)
        self.assertNotIn("| summary |", text)

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
