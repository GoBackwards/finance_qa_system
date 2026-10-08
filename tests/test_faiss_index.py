"""Real FAISS persistence, exactness, mapping integrity, and ANN comparison."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from report_index.faiss_index import (
    benchmark_indexes, build_faiss_index, load_chunk_mapping, load_faiss_index, manifest_path,
    search_faiss_index, sha256_file, validate_vectors,
)


def normalized(rows=120, dimension=12, seed=123):
    array = np.random.default_rng(seed).standard_normal((rows, dimension)).astype(np.float32)
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    return array


class FaissIndexTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp" / "faiss_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.vectors = normalized()
        self.chunks = self.folder / "chunks.jsonl"
        self.write_chunks(len(self.vectors))

    def write_chunks(self, count):
        with self.chunks.open("w", encoding="utf-8", newline="\n") as stream:
            for index in range(count):
                stream.write(json.dumps({"chunk_id": f"chunk-{index}", "document_id": f"doc-{index % 3}",
                                         "text": "Revenue €10 million", "source_pages": [index + 1]}) + "\n")

    def build(self, kind="flat", **kwargs):
        path = self.folder / f"{kind}.faiss"
        manifest = build_faiss_index(self.vectors, path, kind=kind, chunks_path=self.chunks, **kwargs)
        return path, manifest

    def test_flat_roundtrip_matches_numpy_cosine_and_maps_every_source_row(self):
        path, manifest = self.build()
        queries = normalized(7, 12, seed=91)
        scores, ids = search_faiss_index(path, queries)
        expected = queries @ self.vectors.T
        expected_ids = np.argsort(-expected, axis=1)[:, :5]
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(scores, np.take_along_axis(expected, expected_ids, axis=1), atol=1e-6)
        self.assertEqual(manifest["document_count"], 3)
        self.assertEqual(manifest["vector_count"], len(self.vectors))
        self.assertEqual(manifest["dimension"], 12)
        self.assertEqual(manifest["chunks_sha256"], sha256_file(self.chunks))
        metadata = [json.loads(line) for line in Path(manifest["metadata_path"]).read_text(encoding="utf-8").splitlines()]
        with self.chunks.open("rb") as source:
            for vector_id, row in enumerate(metadata):
                self.assertEqual(row["vector_id"], vector_id)
                source.seek(row["source_jsonl_byte_offset"])
                chunk = json.loads(source.read(row["source_jsonl_byte_length"]))
                self.assertEqual(chunk["chunk_id"], row["chunk_id"])

    def test_npy_memmap_and_batch_boundaries_keep_identical_ids(self):
        vector_path = self.folder / "vectors.npy"
        np.save(vector_path, self.vectors)
        manifest = build_faiss_index(vector_path, self.folder / "disk.faiss", chunks_path=self.chunks, batch_size=7)
        ids = search_faiss_index(self.folder / "disk.faiss", self.vectors[::11], k=1)[1]
        np.testing.assert_array_equal(ids[:, 0], np.arange(0, len(self.vectors), 11))
        self.assertEqual(manifest["vectors_file_sha256"], sha256_file(vector_path))
        self.assertEqual(manifest["vectors_sha256"], validate_vectors(self.vectors)["vectors_sha256"])

    def test_invalid_values_and_unnormalized_vectors_are_rejected(self):
        invalid = [np.empty((0, 12), dtype=np.float32), self.vectors.astype(np.float64),
                   self.vectors[0], self.vectors * 2]
        for value in (0, np.nan, np.inf):
            array = self.vectors.copy()
            array[0] = value
            invalid.append(array)
        for array in invalid:
            with self.subTest(shape=array.shape):
                with self.assertRaises(ValueError):
                    build_faiss_index(array, self.folder / "invalid.faiss")
        self.assertFalse((self.folder / "invalid.faiss").exists())
        with self.assertRaisesRegex(ValueError, "hnsw_m must be at least 2"):
            build_faiss_index(self.vectors, self.folder / "invalid.faiss", kind="hnsw", hnsw_m=1)

    def test_chunk_count_mismatch_and_duplicate_ids_do_not_replace_finished_index(self):
        path, manifest = self.build()
        original_index = path.read_bytes()
        original_manifest = manifest_path(path).read_bytes()
        self.write_chunks(len(self.vectors) - 1)
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            build_faiss_index(self.vectors, path, chunks_path=self.chunks)
        self.assertEqual(path.read_bytes(), original_index)
        self.assertEqual(manifest_path(path).read_bytes(), original_manifest)
        self.chunks.write_text('{"chunk_id":"same","document_id":"doc"}\n' * len(self.vectors), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
            build_faiss_index(self.vectors, path, chunks_path=self.chunks)
        self.assertEqual(path.read_bytes(), original_index)
        self.assertEqual(list(self.folder.glob("*.building-*")), [])

    def test_blank_chunk_lines_cannot_shift_vector_ids(self):
        self.chunks.write_text('\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Blank source chunks line"):
            self.build()

    def test_bad_query_dimension_dtype_and_norm_fail(self):
        path, _ = self.build()
        for queries in (normalized(1, 11), normalized(1, 12).astype(np.float64), normalized(1, 12) * 3):
            with self.assertRaises(ValueError):
                search_faiss_index(path, queries)
        with self.assertRaises(ValueError):
            search_faiss_index(path, normalized(1, 12), k=0)

    def test_tampered_index_is_rejected_before_faiss_deserialization(self):
        path, _ = self.build()
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_faiss_index(path)

    def test_mapping_binds_source_and_rejects_changed_source(self):
        path, manifest = self.build()
        self.assertEqual(len(load_chunk_mapping(path)), len(self.vectors))
        with self.chunks.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Source chunks checksum"):
            load_chunk_mapping(path)
        Path(manifest["metadata_path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Metadata checksum"):
            load_faiss_index(path)

    def test_ivf_adapts_nlist_records_training_and_exhaustive_probe_matches_flat(self):
        path, manifest = self.build("ivf", nlist=256, nprobe=256, train_sample_size=1000)
        self.assertEqual(manifest["configuration"]["nlist"], 3)
        self.assertEqual(manifest["configuration"]["nprobe"], 3)
        self.assertTrue(manifest["configuration"]["adaptive_nlist"])
        self.assertEqual(manifest["configuration"]["training_sample_size"], 120)
        queries = normalized(6, 12, seed=234)
        ids = search_faiss_index(path, queries)[1]
        np.testing.assert_array_equal(ids, np.argsort(-(queries @ self.vectors.T), axis=1)[:, :5])
        with self.assertRaisesRegex(ValueError, "at least 39"):
            build_faiss_index(self.vectors[:30], self.folder / "tiny.faiss", kind="ivf")

    def test_hnsw_seeded_construction_and_benchmark_recall_latency(self):
        flat, _ = self.build()
        hnsw, manifest = self.build("hnsw", hnsw_m=16, ef_search=64, threads=2)
        self.assertEqual(manifest["configuration"]["construction_threads"], 1)
        second = self.folder / "hnsw-again.faiss"
        build_faiss_index(self.vectors, second, kind="hnsw", hnsw_m=16, ef_search=64, threads=2)
        self.assertEqual(hnsw.read_bytes(), second.read_bytes())
        ivf, _ = self.build("ivf", nlist=2, nprobe=1)
        result = benchmark_indexes({"flat": flat, "ivf": ivf, "hnsw": hnsw}, normalized(9, 12, 99),
                                   warmup=1, repeats=2, threads=1)
        self.assertEqual(result["query_count"], 9)
        self.assertEqual(result["results"][0]["exact_neighbor_recall_at_5"], 1)
        for row in result["results"]:
            self.assertGreaterEqual(row["exact_neighbor_recall_at_5"], 0)
            self.assertLessEqual(row["exact_neighbor_recall_at_5"], 1)
            self.assertGreater(row["mean_search_ms"], 0)
            self.assertEqual(len(row["per_query"]), 9)

    def test_benchmark_rejects_different_vector_datasets(self):
        flat, _ = self.build()
        other = self.folder / "other.faiss"
        build_faiss_index(normalized(120, 12, 912), other, kind="hnsw", chunks_path=self.chunks)
        with self.assertRaisesRegex(ValueError, "identical vectors_sha256"):
            benchmark_indexes({"flat": flat, "other": other}, self.vectors[:2])

    def test_flat_baseline_remains_one_with_tied_neighbors(self):
        vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (8, 1))
        path = self.folder / "ties.faiss"
        build_faiss_index(vectors, path)
        result = benchmark_indexes({"flat": path}, vectors[:1], warmup=0, repeats=1)
        self.assertEqual(result["results"][0]["exact_neighbor_recall_at_5"], 1)
        self.assertEqual(result["boundary_tie_query_count"], 1)


if __name__ == "__main__":
    unittest.main()
