"""Real-index measurement contracts and separate ANN/gold recall semantics."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from report_index.comparison import (
    build_worker, compare_query_results, query_worker, rebuild_parameters, resident_bytes,
)
from report_index.faiss_index import build_faiss_index, manifest_path, search_faiss_index


def normalized(rows, dimension=12, seed=43):
    values = np.random.default_rng(seed).normal(size=(rows, dimension)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


class IndexComparisonTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp" / "index_comparison_tests"
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.vectors = self.folder / "vectors.npy"
        self.queries = self.folder / "queries.npy"
        np.save(self.vectors, normalized(120))
        np.save(self.queries, normalized(3, seed=99))
        self.chunks = self.folder / "chunks.jsonl"
        self.chunks.write_text("".join(json.dumps({"chunk_id": f"c{i}", "document_id": f"d{i % 4}",
                                                   "text": "original source"}) + "\n" for i in range(120)), encoding="utf-8")

    def build(self, kind="flat", **kwargs):
        path = self.folder / f"source-{kind}.faiss"
        manifest = build_faiss_index(self.vectors, path, chunks_path=self.chunks, kind=kind,
                                     threads=2, batch_size=31, **kwargs)
        return path, manifest

    def test_rebuild_preserves_configuration_and_separates_time_scopes(self):
        for kind, options in [("flat", {}), ("ivf", {"nlist": 8, "nprobe": 2, "train_sample_size": 100}),
                              ("hnsw", {"hnsw_m": 8, "ef_construction": 32, "ef_search": 16})]:
            with self.subTest(kind=kind):
                source, old = self.build(kind, **options)
                target = self.folder / f"rebuilt-{kind}.faiss"
                report = build_worker(manifest_path(source), self.vectors, self.chunks, target)
                rebuilt = json.loads(manifest_path(target).read_text())
                self.assertEqual(rebuilt["configuration"], old["configuration"])
                self.assertEqual(rebuilt["vectors_sha256"], old["vectors_sha256"])
                self.assertEqual(report["build_seconds"], report["training_seconds"] + report["add_seconds"])
                self.assertGreaterEqual(report["pipeline_seconds"], report["build_seconds"])
                self.assertEqual(report["index_size_bytes"], target.stat().st_size)
                self.assertEqual(rebuild_parameters(old)["kind"], kind)

    def test_rebuild_rejects_changed_inputs_and_original_destination(self):
        source, _ = self.build()
        with self.assertRaisesRegex(ValueError, "new index path"):
            build_worker(manifest_path(source), self.vectors, self.chunks, source)
        changed = self.folder / "changed.npy"
        np.save(changed, normalized(120, seed=45))
        with self.assertRaisesRegex(ValueError, "vectors checksum"):
            build_worker(manifest_path(source), changed, self.chunks, self.folder / "new.faiss")
        self.chunks.write_text(self.chunks.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "chunks checksum"):
            build_worker(manifest_path(source), self.vectors, self.chunks, self.folder / "new.faiss")

    def test_each_k_matches_an_independent_single_query_search_and_records_samples(self):
        source, _ = self.build()
        report = query_worker(source, self.queries, ks=(3, 10, 50, 150), repeats=5, threads=2)
        queries = np.load(self.queries)
        self.assertEqual(report["ks"], [3, 10, 50, 150])
        for row in report["per_query"]:
            self.assertEqual(len(row["search_ms_samples"]), 5)
            self.assertTrue(all(value > 0 for value in row["search_ms_samples"]))
            for k in report["ks"]:
                scores, ids = search_faiss_index(source, queries[row["query_index"]:row["query_index"] + 1],
                                                k=k, threads=2)
                valid = ids[0] >= 0
                self.assertEqual(row["neighbors"][str(k)]["vector_ids"], ids[0, valid].tolist())
                np.testing.assert_array_equal(row["neighbors"][str(k)]["scores"], scores[0, valid])
        comparison = compare_query_results({"Flat": report})
        self.assertEqual(comparison["results"]["Flat"]["ann_recall"], {str(k): 1.0 for k in report["ks"]})
        self.assertEqual(report["index_storage_mb"], source.stat().st_size / 1_000_000)

    def test_main_memory_metric_is_loaded_rss_delta_not_warmed_or_disk_size(self):
        source, _ = self.build()
        with patch("report_index.comparison.resident_bytes", side_effect=[1_000_000, 3_000_000, 5_000_000, 6_000_000]):
            report = query_worker(source, self.queries, ks=(3,), repeats=1, warmup=0)
        self.assertEqual(report["memory_usage_mb"], 2.0)
        self.assertEqual(report["resident_load_delta_bytes"], 2_000_000)
        self.assertEqual(report["resident_warmed_delta_bytes"], 4_000_000)
        self.assertNotEqual(report["memory_usage_mb"], report["index_storage_mb"])
        self.assertIn("decimal MB", report["memory_units"])

    def test_comparison_uses_id_overlap_and_checks_query_and_corpus_bindings(self):
        flat, _ = self.build()
        reference = query_worker(flat, self.queries, ks=(3, 10), repeats=1)
        altered = copy.deepcopy(reference)
        altered["kind"] = "ivf"
        altered["per_query"][0]["neighbors"]["3"]["vector_ids"] = [1000, 1001, 1002]
        comparison = compare_query_results({"Flat": reference, "IVF": altered})
        self.assertAlmostEqual(comparison["results"]["IVF"]["ann_recall"]["3"], 2 / 3)
        self.assertEqual(comparison["results"]["IVF"]["ann_recall"]["10"], 1.0)
        for field, value in [("vectors_sha256", "wrong"), ("query_vectors_sha256", "wrong"),
                             ("ks", [5]), ("faiss_threads", 1), ("query_count", 100),
                             ("latency_k", 20), ("warmup_passes", 2), ("repeats_per_query", 5)]:
            broken = copy.deepcopy(altered)
            broken[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare_query_results({"Flat": reference, "IVF": broken})
        broken = copy.deepcopy(altered)
        broken["per_query"][0]["query_index"] = 1
        with self.assertRaisesRegex(ValueError, "Query order"):
            compare_query_results({"Flat": reference, "IVF": broken})

    def test_worker_rejects_bad_index_hash_or_query_shape(self):
        source, _ = self.build()
        incorrect_queries = self.folder / "wrong.npy"
        np.save(incorrect_queries, normalized(2, dimension=11))
        with self.assertRaisesRegex(ValueError, "dimension"):
            query_worker(source, incorrect_queries)
        source.write_bytes(source.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            query_worker(source, self.queries)

    def test_query_cli_runs_in_an_independent_process_and_writes_bound_results(self):
        source, _ = self.build()
        output = self.folder / "query.json"
        completed = subprocess.run([sys.executable, "-m", "report_index.comparison", "query", "--index", str(source),
                                    "--queries", str(self.queries), "--output", str(output), "--ks", "3", "10",
                                    "--repeats", "5", "--warmup", "1", "--threads", "2"],
                                   capture_output=True, text=True, check=True)
        report = json.loads(output.read_text())
        self.assertNotEqual(report["process_id"], os.getpid())
        self.assertEqual(report["query_count"], 3)
        self.assertEqual(report["ks"], [3, 10])
        self.assertEqual(json.loads(completed.stdout)["measurement"], "isolated_index_query")
        self.assertGreater(report["rss_baseline_bytes"], 0)

    def test_resident_counter_reports_current_nonzero_integer(self):
        measured = resident_bytes()
        self.assertIsInstance(measured, int)
        self.assertGreater(measured, 0)


if __name__ == "__main__":
    unittest.main()
