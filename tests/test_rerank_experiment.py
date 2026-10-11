"""Frozen-candidate extraction and atomic scorer resumption contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from report_index.faiss_index import sha256_file
from report_search.experiment import json_hash, write_json
from report_search.rerank_experiment import checked_score_rows, prepare_candidates, score_worker


def candidates():
    return [{"chunk_id": str(i), "document_id": "doc", "text": "source passage " + str(i),
             "rank": i + 1, "score": 1 - i / 20} for i in range(20)]


class RerankExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source_fixture(self):
        benchmark = self.root / "benchmark.json"
        questions = [{"question_id": "q", "query": "What is revenue?", "answer": "GOLD MUST NOT ENTER MODEL"}]
        write_json(benchmark, {"questions": questions})
        source = self.root / "retrieval"
        source.mkdir()
        np.save(source / "query_vectors.npy", np.ones((1, 1), dtype=np.float32))
        query_manifest = {"vectors_file_sha256": sha256_file(source / "query_vectors.npy"),
                          "contract": {"query_texts_sha256": json_hash([("q", "What is revenue?")])}}
        write_json(source / "query_manifest.json", query_manifest)
        row = {"approach": "vector", "question_id": "q", "query": "What is revenue?", "results": candidates(),
               "answer": "GOLD MUST NOT ENTER MODEL", "judgments": {"private_gold": True}}
        (source / "per_query.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
        write_json(source / "comparison.json", {"benchmark_provenance": {"benchmark_sha256": sha256_file(benchmark),
                                                                        "query_manifest": query_manifest},
                                                "per_query_sha256": sha256_file(source / "per_query.jsonl"), "corpus": {}})
        return benchmark, source

    def test_frozen_top20_extraction_preserves_order_and_excludes_answers_and_judgments(self):
        benchmark, source = self.source_fixture()
        _, manifest = prepare_candidates(benchmark, source, self.root / "output")
        exported = (self.root / "output" / "candidates.jsonl").read_text(encoding="utf-8")
        row = json.loads(exported)
        self.assertEqual(set(row), {"question_id", "query", "candidates"})
        self.assertEqual(row["candidates"], candidates())
        self.assertNotIn("GOLD MUST NOT ENTER MODEL", exported)
        self.assertNotIn("private_gold", exported)
        self.assertEqual(manifest["candidate_file_sha256"], sha256_file(self.root / "output" / "candidates.jsonl"))
        with (source / "per_query.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            prepare_candidates(benchmark, source, self.root / "tampered")

    def test_changed_benchmark_or_query_vectors_are_not_reused(self):
        benchmark, source = self.source_fixture()
        original = benchmark.read_bytes()
        write_json(benchmark, {"questions": []})
        with self.assertRaisesRegex(ValueError, "different frozen benchmark"):
            prepare_candidates(benchmark, source, self.root / "output")
        benchmark.write_bytes(original)
        np.save(source / "query_vectors.npy", np.zeros((1, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "query embeddings differ"):
            prepare_candidates(benchmark, source, self.root / "output")

    def test_checkpoint_rejects_changed_query_candidate_order_or_scores(self):
        inputs = [{"question_id": "q", "candidates": candidates()}]
        row = {"question_id": "q", "candidate_ids": [c["chunk_id"] for c in candidates()], "scores": list(range(20))}
        self.assertEqual(checked_score_rows([row], inputs), [row])
        for field, value in (("question_id", "other"), ("candidate_ids", row["candidate_ids"][::-1]),
                             ("scores", [True] * 20), ("scores", [float("nan")] * 20), ("scores", [1.0])):
            changed = {**row, field: value}
            with self.assertRaises(ValueError):
                checked_score_rows([changed], inputs)

    def test_completed_scoring_checkpoint_reuses_scores_without_loading_model(self):
        benchmark, source = self.source_fixture()
        output = self.root / "output"
        prepare_candidates(benchmark, source, output)

        class FakeReranker:
            calls = []
            identity = {"model": "pinned", "batch_size": 4}

            def __init__(self, **kwargs):
                pass

            def pair_token_lengths(self, pairs):
                return [10] * len(pairs)

            def preflight_pairs(self, pairs):
                return {"pair_count": len(pairs), "overflow_count": 0}

            def score_pairs(self, pairs):
                self.calls.append(len(pairs))
                return list(map(float, range(len(pairs))))

        with patch("report_search.rerank_experiment.BGEReranker", FakeReranker), patch(
                "report_search.rerank_experiment.verify_reference_scores", return_value={"paired_token_ids_equal": True}):
            first = score_worker(output, device="cpu")
            self.assertEqual(FakeReranker.calls, [4, 20])
            FakeReranker.calls.clear()
            second = score_worker(output, device="cpu")
            self.assertEqual(FakeReranker.calls, [])
            self.assertEqual(first["scores_sha256"], second["scores_sha256"])
            metadata = json.loads((output / "scoring_checkpoint.json").read_text(encoding="utf-8"))
            metadata["contract"]["reranker"] = {"model": "changed"}
            write_json(output / "scoring_checkpoint.json", metadata)
            with self.assertRaisesRegex(ValueError, "different inputs/configuration"):
                score_worker(output, device="cpu")


if __name__ == "__main__":
    unittest.main()
