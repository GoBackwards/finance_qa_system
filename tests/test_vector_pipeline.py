"""Integration checks for shard identity, vector mapping, and source evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from report_index.embedding import build_embeddings, sha256_file
from report_index.faiss_index import benchmark_indexes, build_faiss_index
from report_index.pipeline import consolidate_vectors, evaluate_dense, main


HAS_FAISS = importlib.util.find_spec("faiss") is not None


class WorkspaceTemporaryTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp" / "vector_pipeline_tests"
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)


class ConsolidationTests(WorkspaceTemporaryTest):
    def setUp(self):
        super().setUp()
        self.expected = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        self.manifest = {"status": "complete", "fingerprint": "fixture-encoder",
                         "dimension": 2, "vector_count": 3, "shards": []}
        for number, (start, stop) in enumerate([(0, 2), (2, 3)]):
            path = self.folder / f"shard-{number}.npy"
            np.save(path, self.expected[start:stop], allow_pickle=False)
            self.manifest["shards"].append({"start_index": start, "count": stop - start,
                                             "vectors_path": path.name,
                                             "vectors_sha256": sha256_file(path)})

    def test_consolidation_keeps_vector_order_and_reuses_valid_output(self):
        result = consolidate_vectors(self.folder, self.manifest)
        np.testing.assert_array_equal(np.load(result, allow_pickle=False), self.expected)
        mtime = result.stat().st_mtime_ns
        self.assertEqual(consolidate_vectors(self.folder, self.manifest), result)
        self.assertEqual(result.stat().st_mtime_ns, mtime)
        marker = json.loads((self.folder / "vectors_manifest.json").read_text())
        self.assertEqual(marker["sha256"], sha256_file(result))
        self.assertEqual(marker["identity"]["shape"], [3, 2])

    def test_modified_consolidated_output_is_rebuilt_from_verified_shards(self):
        result = consolidate_vectors(self.folder, self.manifest)
        result.write_bytes(b"interrupted or corrupt matrix")
        rebuilt = consolidate_vectors(self.folder, self.manifest)
        np.testing.assert_array_equal(np.load(rebuilt, allow_pickle=False), self.expected)

    def test_corrupt_shard_hash_is_rejected_before_output_commit(self):
        (self.folder / "shard-0.npy").write_bytes(b"corrupt shard")
        with self.assertRaisesRegex(ValueError, "hash|modified"):
            consolidate_vectors(self.folder, self.manifest)
        self.assertFalse((self.folder / "vectors_manifest.json").exists())

    def test_cached_output_does_not_hide_corrupt_input_shard(self):
        consolidate_vectors(self.folder, self.manifest)
        (self.folder / "shard-0.npy").write_bytes(b"corrupt after initial consolidation")
        with self.assertRaisesRegex(ValueError, "hash|modified"):
            consolidate_vectors(self.folder, self.manifest)

    def test_noncontiguous_shard_order_is_rejected(self):
        self.manifest["shards"].reverse()
        with self.assertRaisesRegex(ValueError, "order|contiguous"):
            consolidate_vectors(self.folder, self.manifest)

    def test_incomplete_run_or_missing_rows_is_rejected(self):
        invalid = deepcopy(self.manifest)
        invalid["status"] = "in_progress"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            consolidate_vectors(self.folder, invalid)
        invalid = deepcopy(self.manifest)
        invalid["vector_count"] = 4
        with self.assertRaisesRegex(ValueError, "coverage|count"):
            consolidate_vectors(self.folder, invalid)

    def test_wrong_shard_dimension_or_dtype_is_rejected_even_with_valid_hash(self):
        path = self.folder / "shard-0.npy"
        for bad in (np.ones((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float64)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                np.save(path, bad, allow_pickle=False)
                self.manifest["shards"][0]["vectors_sha256"] = sha256_file(path)
                with self.assertRaisesRegex(ValueError, "shape|dtype"):
                    consolidate_vectors(self.folder, self.manifest)

    def test_shard_path_cannot_escape_embedding_directory(self):
        self.manifest["shards"][0]["vectors_path"] = "../outside.npy"
        with self.assertRaisesRegex(ValueError, "outside|escapes"):
            consolidate_vectors(self.folder, self.manifest)


def citation(document, page):
    return {"document_id": document, "pdf_page": page, "sha256": "fixture-pdf-hash"}


def prose_gold(text="Operating margins improved.", element="paragraph-1", unit="prose-gold"):
    return {"evidence_id": unit, "document_id": "ACME_FY2025", "pdf_page": 7,
            "source_sha256": "fixture-pdf-hash", "element_id": element,
            "char_start": 0, "char_end": len(text), "quote": text}


def corpus_fixture():
    prose = "Operating margins improved."
    row = "Revenue | 100 | 90"
    common = {"document_id": "ACME_FY2025", "ticker": "ACME", "company": "Acme Corporation",
              "fiscal_year": 2025, "source_file": "ACME_FY2025_annual_report.pdf",
              "strategy_id": "table_preserving_fixture"}
    chunks = [
        {**common, "chunk_id": "z-prose", "kind": "prose", "text": prose,
         "token_count": 4, "source_pages": [7],
         "evidence": [{"kind": "prose", "element_id": "paragraph-1", "char_start": 0,
                       "char_end": len(prose), "text": prose, "citation": citation("ACME_FY2025", 7)}]},
        {**common, "chunk_id": "a-table", "kind": "table", "text": row,
         "token_count": 5, "source_pages": [12],
         "evidence": [{"kind": "row", "table_id": "table-1", "row_index": 2,
                       "row_text": row, "citation": citation("ACME_FY2025", 12)}]},
        {**common, "document_id": "OTHER_FY2024", "chunk_id": "m-distractor", "kind": "prose",
         "text": "Unrelated business information.", "token_count": 3, "source_pages": [1], "evidence": []},
    ]
    table_gold = {"evidence_id": "table-gold", "document_id": "ACME_FY2025", "pdf_page": 12,
                  "source_sha256": "fixture-pdf-hash", "table_id": "table-1", "row_index": 2,
                  "row_text": row}
    benchmark = {"questions": [
        {"question_id": "q-prose", "question": "Explain Acme's margins and cash flow.", "type": "prose",
         "evidence": [prose_gold(), prose_gold("Cash flow increased.", "paragraph-missing", "missing-gold")]},
        {"question_id": "q-table", "question": "What was Acme's revenue?", "type": "table",
         "evidence": [table_gold]},
    ]}
    return chunks, benchmark


class FixtureEncoder:
    """Deterministic integration fixture, deliberately not a semantic model."""
    dimension = 2
    max_tokens = 512
    identity = {"model": "deterministic-test-fixture", "revision": "fixture-v1", "dimension": 2}
    revision = "fixture-v1"

    def token_lengths(self, texts):
        return [len(text.split()) + 2 for text in texts]

    def encode_documents(self, texts):
        return np.asarray([[0, 2] if "Revenue |" in text else
                           [2, 0] if "Operating margins improved." in text else [-2, 0]
                           for text in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[0, 1] if "revenue" in text else [1, 0] for text in texts], dtype=np.float32)


@unittest.skipUnless(HAS_FAISS, "FAISS optional dependency is not installed")
class DenseEvidenceIntegrationTests(WorkspaceTemporaryTest):
    def setUp(self):
        super().setUp()
        self.chunks, self.benchmark = corpus_fixture()
        self.source = self.folder / "chunks.jsonl"
        self.source.write_text("".join(json.dumps(chunk) + "\n" for chunk in self.chunks), encoding="utf-8")
        self.embeddings = self.folder / "embeddings"
        self.manifest = build_embeddings(self.source, self.embeddings, FixtureEncoder(), shard_size=2)
        self.vectors = consolidate_vectors(self.embeddings, self.manifest)
        self.index = self.folder / "flat.faiss"
        self.index_manifest = build_faiss_index(self.vectors, self.index, chunks_path=self.source)
        self.queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)

    def test_complete_embedding_to_faiss_to_evidence_path_preserves_mapping(self):
        self.assertEqual(self.manifest["vector_count"], 3)
        self.assertEqual(self.manifest["document_count"], 2)
        self.assertEqual(self.index_manifest["dimension"], 2)
        result = evaluate_dense(self.index, self.source, self.benchmark, self.queries)
        self.assertEqual(result["aggregate"]["recall_at_5"], 0.75)
        self.assertEqual(result["aggregate"]["hit_at_5"], 1)
        self.assertEqual(result["aggregate"]["all_evidence_hit_at_5"], 0.5)
        self.assertEqual(result["aggregate"]["gold_units"], 3)
        self.assertEqual(result["aggregate"]["recovered_units"], 2)
        self.assertEqual(result["by_type"]["table"]["recall_at_5"], 1)
        self.assertEqual(result["average_top5_tokens"], 12)
        self.assertEqual(result["scope"], "full_corpus_unfiltered")
        for number, expected in enumerate(["z-prose", "a-table"]):
            retrieved = result["per_query"][number]["retrieved"]
            self.assertEqual(retrieved[0]["chunk_id"], expected)
            self.assertEqual(retrieved[0]["vector_id"], number)
            self.assertEqual(len(retrieved), 3)  # FAISS -1 padding is never a source document.
            self.assertIsNone(result["per_query"][number]["filters"])
        self.assertEqual(result["per_query"][1]["retrieved"][0]["citations"][0]["pdf_page"], 12)

    def test_ann_agreement_and_gold_evidence_recall_are_separate_metrics(self):
        ann = benchmark_indexes({"flat": self.index}, self.queries, k=5, warmup=0, repeats=1)
        evidence = evaluate_dense(self.index, self.source, self.benchmark, self.queries)
        self.assertEqual(ann["results"][0]["exact_neighbor_recall_at_5"], 1)
        self.assertEqual(ann["effective_k"], 3)
        self.assertEqual(evidence["aggregate"]["recall_at_5"], 0.75)

    def test_nondefault_top_k_labels_metrics_and_returned_context_accurately(self):
        result = evaluate_dense(self.index, self.source, self.benchmark, self.queries, top_k=1)
        self.assertEqual(result["top_k"], 1)
        self.assertEqual(result["aggregate"]["recall_at_1"], 0.75)
        self.assertEqual(result["average_top1_tokens"], 4.5)
        self.assertNotIn("average_top5_tokens", result)

    def test_cli_writes_complete_dimension_counts_and_separate_eval_artifacts(self):
        benchmark_path = self.folder / "benchmark.json"
        benchmark_path.write_text(json.dumps(self.benchmark), encoding="utf-8")
        output = self.folder / "cli-output"
        with patch("report_index.pipeline.DenseEncoder", return_value=FixtureEncoder()):
            status = main(["--chunks", str(self.source), "--output", str(output),
                           "--device", "cpu", "--shard-size", "2", "--threads", "1",
                           "--benchmark", str(benchmark_path)])
        self.assertEqual(status, 0)
        summary = json.loads((output / "index_creation.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["dimension"], 2)
        self.assertEqual(summary["document_count"], 2)
        self.assertEqual(summary["chunk_count"], 3)
        self.assertEqual(summary["vector_count"], 3)
        self.assertEqual(summary["source_sha256"], sha256_file(self.source))
        self.assertEqual(summary["evaluation"]["flat"]["aggregate"]["recall_at_5"], 0.75)
        query_manifest = json.loads((output / "query_manifest.json").read_text())
        self.assertEqual(query_manifest["question_ids"], ["q-prose", "q-table"])
        self.assertEqual(query_manifest["benchmark_sha256"], sha256_file(benchmark_path))
        self.assertEqual(query_manifest["query_vectors_sha256"], sha256_file(output / "query_vectors.npy"))
        self.assertTrue((output / "ann_comparison.json").is_file())
        self.assertTrue((output / "flat_retrieval_results.json").is_file())

    def test_wrong_gold_source_hash_gets_no_credit_for_matching_text_or_row(self):
        benchmark = deepcopy(self.benchmark)
        for question in benchmark["questions"]:
            for gold in question["evidence"]:
                gold["source_sha256"] = "different-pdf-version"
        result = evaluate_dense(self.index, self.source, benchmark, self.queries)
        self.assertEqual(result["aggregate"]["recall_at_5"], 0)
        self.assertEqual(result["aggregate"]["hit_at_5"], 0)

    def test_reordered_source_cannot_reuse_original_vector_mapping(self):
        self.source.write_text("".join(json.dumps(chunk) + "\n" for chunk in reversed(self.chunks)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no longer match"):
            evaluate_dense(self.index, self.source, self.benchmark, self.queries)

    def test_invalid_benchmark_count_or_duplicate_question_ids_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            evaluate_dense(self.index, self.source, self.benchmark, self.queries[:1])
        duplicate = deepcopy(self.benchmark)
        duplicate["questions"][1]["question_id"] = duplicate["questions"][0]["question_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_dense(self.index, self.source, duplicate, self.queries)


if __name__ == "__main__":
    unittest.main()
