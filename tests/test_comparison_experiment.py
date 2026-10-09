"""Query identity and native-vector-to-source evidence bindings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from report_chunker.retrieval import build_index
from report_index.comparison_experiment import prepare_queries, score_index_results
from report_index.faiss_index import sha256_file
from report_search.experiment import json_hash, write_json


class ComparisonExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_cached_query_selection_preserves_order_and_rejects_label_or_model_change(self):
        questions = [{"question_id": i, "query": i + " query", "split": split}
                     for i, split in (("dev", "dev"), ("test-z", "test"), ("test-a", "test"))]
        benchmark = self.root / "benchmark.json"
        write_json(benchmark, {"questions": questions})
        chunks = self.root / "chunks.jsonl"
        chunks.write_bytes(b"corpus\n")
        embedding_path = self.root / "embedding_manifest.json"
        embedding = {"status": "complete", "source_sha256": sha256_file(chunks),
                     "encoder": {"model_id": "fixture", "revision": "pinned"}, "dimension": 2}
        write_json(embedding_path, embedding)
        vectors = np.asarray([[1, 0], [0, 1], [2**-.5, 2**-.5]], dtype=np.float32)
        np.save(self.root / "vectors.npy", vectors)
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "query_vectors.npy", vectors)
        contract = {
            "query_texts_sha256": json_hash([(q["question_id"], q["query"]) for q in questions]),
            "benchmark_sha256": sha256_file(benchmark), "corpus_encoder": embedding["encoder"],
            "corpus_embedding_manifest_sha256": sha256_file(embedding_path),
            "corpus_vectors_file_sha256": sha256_file(self.root / "vectors.npy"),
            "query_count": 3, "dimension": 2,
        }
        write_json(cache / "query_manifest.json", {"contract": contract, "encoder": embedding["encoder"],
                                                   "vectors_file_sha256": sha256_file(cache / "query_vectors.npy")})
        _, selected, path, provenance = prepare_queries(benchmark, cache, embedding_path, chunks, self.root / "run")
        self.assertEqual([q["question_id"] for q in selected], ["test-z", "test-a"])
        np.testing.assert_array_equal(np.load(path), vectors[1:])
        self.assertEqual(provenance["test_queries_file_sha256"], sha256_file(path))
        for target, payload in ((benchmark, {"questions": [{**questions[0], "query": "changed"}, *questions[1:]]}),
                                (embedding_path, {**embedding, "encoder": {"model_id": "wrong-model"}})):
            original = target.read_bytes()
            write_json(target, payload)
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare_queries(benchmark, cache, embedding_path, chunks, self.root / "invalid")
            self.assertFalse((self.root / "invalid").exists())
            target.write_bytes(original)
        metadata = json.loads((cache / "query_manifest.json").read_text(encoding="utf-8"))
        metadata["encoder"] = {"model_id": "self-inconsistent-model"}
        write_json(cache / "query_manifest.json", metadata)
        with self.assertRaisesRegex(ValueError, "query encoder differs"):
            prepare_queries(benchmark, cache, embedding_path, chunks, self.root / "invalid")

    def source_fixture(self):
        chunks, gold = [], []
        for name in ("a", "b"):
            unit = {"kind": "prose", "evidence_id": name, "document_id": name, "source_sha256": "hash",
                    "pdf_page": 1, "element_id": name + "-element", "char_start": 0,
                    "char_end": 7, "quote": "revenue"}
            chunks.append({"chunk_id": "chunk-" + name, "document_id": name, "text": "revenue",
                           "token_count": 1, "evidence": [{**unit, "text": "revenue"}]})
            gold.append(unit)
        source = self.root / "chunks.jsonl"
        source.write_text("".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8", newline="\n")
        database = self.root / "index.sqlite"
        build_index(source, database)
        mapping = [{"vector_id": i, "chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"]}
                   for i, chunk in enumerate(chunks)]
        question = {"question_id": "q", "category": "comparative", "split": "test",
                    "ground_truth_doc_ids": ["a", "b"], "evidence": gold}
        measured = {"ks": [1, 2], "per_query": [{"query_index": 0, "neighbors": {
            "1": {"vector_ids": [0], "scores": [.9]}, "2": {"vector_ids": [1, 0], "scores": [.95, .9]}}}]}
        return database, mapping, question, measured

    def test_native_k_results_get_complete_rows_and_distinct_document_denominators(self):
        database, mapping, question, measured = self.source_fixture()
        result = score_index_results(measured, [question], database, mapping, self.root / "evidence.jsonl")
        metrics = result["overall"]["metrics"]
        self.assertEqual(metrics["document_recall_at_1"], .5)
        self.assertEqual(metrics["evidence_recall_at_1"], .5)
        self.assertEqual(metrics["all_evidence_hit_at_2"], 1)
        self.assertEqual(result["by_category"]["comparative"]["question_count"], 1)

    def test_wrong_mapping_query_order_or_malformed_neighbors_fail(self):
        database, mapping, question, measured = self.source_fixture()
        with self.assertRaisesRegex(ValueError, "payload order"):
            score_index_results(measured, [question], database, mapping[::-1], self.root / "bad.jsonl")
        for update in ({"vector_ids": [0, 0], "scores": [.9, .8]},
                       {"vector_ids": [2], "scores": [.9]},
                       {"vector_ids": [0], "scores": []},
                       {"vector_ids": [0], "scores": [float("nan")]}):
            changed = copy.deepcopy(measured)
            changed["per_query"][0]["neighbors"]["2"] = update
            with self.assertRaises(ValueError):
                score_index_results(changed, [question], database, mapping, self.root / "bad.jsonl")
        changed = copy.deepcopy(measured)
        changed["per_query"][0]["query_index"] = 1
        with self.assertRaisesRegex(ValueError, "query order"):
            score_index_results(changed, [question], database, mapping, self.root / "bad.jsonl")


if __name__ == "__main__":
    unittest.main()
