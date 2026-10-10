"""Hand-computed score fusion and real FAISS/SQLite corpus-binding checks."""

import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from report_chunker.retrieval import build_index
from report_index.faiss_index import build_faiss_index
from report_search.retrieval import RetrievalEngine, weighted_fusion


def candidate(chunk_id, score):
    return {"chunk_id": chunk_id, "document_id": "doc", "text": chunk_id, "score": score,
            "evidence": [{"citation": {"document_id": "doc", "pdf_page": 1, "sha256": "hash"}}]}


class WeightedFusionTests(unittest.TestCase):
    def test_direction_normalization_and_hand_computed_weighted_union(self):
        dense = [candidate("a", .9), candidate("b", .7), candidate("c", .5)]
        lexical = [candidate("b", -10), candidate("d", -6), candidate("a", -2)]
        before = copy.deepcopy((dense, lexical))
        results = weighted_fusion(dense, lexical, .25, 10)
        self.assertEqual([row["chunk_id"] for row in results], ["b", "d", "a", "c"])
        for actual, expected in zip(results, [.875, .375, .25, 0]):
            self.assertAlmostEqual(actual["score"], expected)
        self.assertEqual((dense, lexical), before)
        self.assertEqual(results[0]["citations"][0]["pdf_page"], 1)
        self.assertEqual(results[1]["score_breakdown"]["vector_raw_score"], None)

    def test_negative_cosines_are_higher_is_better(self):
        results = weighted_fusion([candidate("a", -.2), candidate("b", -.8)], [], .5)
        self.assertEqual([row["score"] for row in results], [.5, 0])

    def test_endpoints_exclude_inactive_candidates_even_for_zero_score(self):
        dense = [candidate("b", .9), candidate("c", .1)]
        lexical = [candidate("a", -2), candidate("z", -1)]
        self.assertEqual([r["chunk_id"] for r in weighted_fusion(dense, lexical, 1)], ["b", "c"])
        self.assertEqual([r["chunk_id"] for r in weighted_fusion(dense, lexical, 0)], ["a", "z"])

    def test_constant_singleton_and_empty_channels(self):
        results = weighted_fusion([candidate("b", -.5), candidate("a", -.5)], [], .5)
        self.assertEqual([r["chunk_id"] for r in results], ["a", "b"])
        self.assertEqual([r["score"] for r in results], [.5, .5])
        self.assertEqual(weighted_fusion([], [candidate("z", -1)], .5)[0]["score"], .5)
        self.assertEqual(weighted_fusion([], []), [])

    def test_duplicate_ids_keep_best_score_once(self):
        results = weighted_fusion([candidate("a", .2), candidate("a", .9), candidate("b", .3)],
                                  [candidate("a", -1), candidate("a", -5), candidate("b", -2)], .5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "a")
        self.assertEqual(results[0]["score_breakdown"]["vector_raw_score"], .9)
        self.assertEqual(results[0]["score_breakdown"]["bm25_raw_score"], -5)

    def test_conflicting_duplicate_payload_is_rejected(self):
        other = candidate("a", 1)
        other["document_id"] = "different"
        with self.assertRaisesRegex(ValueError, "Conflicting payloads"):
            weighted_fusion([candidate("a", .5)], [other])

    def test_invalid_alpha_scores_and_cutoffs(self):
        for alpha in [-.1, 1.1, float("nan"), float("inf"), True, "0.5"]:
            with self.assertRaises(ValueError):
                weighted_fusion([], [], alpha)
        for value in [float("nan"), float("inf"), "a", None]:
            with self.assertRaises(ValueError):
                weighted_fusion([candidate("a", value)], [])
        for top_k in [0, -1, True, .5]:
            with self.assertRaises(ValueError):
                weighted_fusion([], [], top_k=top_k)


class RetrievalEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.chunks = self.directory / "chunks.jsonl"
        self.vector_index = self.directory / "flat.faiss"
        self.bm25_index = self.directory / "bm25.sqlite"
        records = []
        # Physical/vector order intentionally differs from the stable chunk-ID
        # tie order. More identical vectors than top_k exercises cutoff ties.
        for chunk_id, body, document in [("z", "Revenue increased", "doc1"),
                                          ("b", "Manufacturing risk", "doc2"),
                                          ("a", "Revenue risk", "doc1"),
                                          ("c", "Revenue revenue", "doc2")]:
            records.append({"chunk_id": chunk_id, "document_id": document, "text": body,
                            "ticker": "ACME", "fiscal_year": 2024, "company": "Acme",
                            "kind": "prose", "token_count": 2,
                            "evidence": [{"citation": {"document_id": document, "pdf_page": 2,
                                                       "sha256": "hash", "bbox": [0, 0, 50, 50]}}]})
        self.chunks.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        build_index(self.chunks, self.bm25_index)
        vectors = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
        build_faiss_index(vectors, self.vector_index, chunks_path=self.chunks)
        self.query = np.array([1, 0], dtype=np.float32)

    def engine(self):
        return RetrievalEngine(self.chunks, self.vector_index, self.bm25_index, threads=1)

    def test_real_engines_payload_citations_and_global_cutoff_ties(self):
        with self.engine() as engine:
            self.assertEqual(engine.stats["document_count"], 2)
            results = engine.vector(self.query, 2)
            self.assertEqual([r["chunk_id"] for r in results], ["a", "b"])
            self.assertEqual(results[0]["text"], "Revenue risk")
            self.assertEqual(results[0]["citations"][0]["pdf_page"], 2)
            self.assertEqual(engine.bm25("manufacturing", 1)[0]["chunk_id"], "b")
            self.assertEqual(engine.bm25("zzzzunmatchedword", 5), [])
            self.assertEqual(engine.bm25("and the of", 5), [])

    def test_document_scoped_vector_search_is_exact_within_scope(self):
        with self.engine() as engine:
            self.assertEqual(engine.document_ids(), ["doc1", "doc2"])
            scoped = engine.vector(self.query, 5, document_ids=["doc2"])
            # Scoping searches inside doc2; it is not a post-filter of the global Top-K.
            self.assertEqual([r["chunk_id"] for r in scoped], ["b", "c"])
            self.assertEqual({r["document_id"] for r in scoped}, {"doc2"})
            self.assertEqual([r["rank"] for r in scoped], [1, 2])
            tied = engine.vector(self.query, 1, document_ids=["doc1"])
            self.assertEqual([r["chunk_id"] for r in tied], ["a"])
            for invalid in ([], ["missing"], ["doc1", "missing"]):
                with self.assertRaises(ValueError):
                    engine.vector(self.query, 2, document_ids=invalid)

    def test_pure_ranking_endpoints_and_cached_candidate_fusion(self):
        with self.engine() as engine:
            vector, bm25 = engine.vector(self.query, 4), engine.bm25("Revenue", 4)
            for alpha, pure in [(0, bm25), (1, vector)]:
                result = engine.hybrid("Revenue", self.query, top_k=2, candidate_k=4, alpha=alpha)
                self.assertEqual([r["chunk_id"] for r in result], [r["chunk_id"] for r in pure[:2]])
            direct = engine.hybrid("Revenue", self.query, top_k=2, candidate_k=4)
            cached = engine.hybrid("Revenue", self.query, top_k=2, candidate_k=4,
                                   vector_results=vector, bm25_results=bm25)
            self.assertEqual(direct, cached)
            self.assertEqual(len(engine.hybrid("zzzzunmatchedword", self.query, candidate_k=5)), 4)

    def test_source_bytes_change_rejected(self):
        with self.chunks.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.engine()

    def test_bm25_from_other_corpus_rejected(self):
        other = self.directory / "other.jsonl"
        other.write_text(self.chunks.read_text().replace("Manufacturing", "Production"), encoding="utf-8")
        build_index(other, self.bm25_index)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.engine()

    def test_bm25_document_mapping_corruption_rejected(self):
        with closing(sqlite3.connect(self.bm25_index)) as connection:
            connection.execute("UPDATE chunks SET document_id='wrong' WHERE chunk_id='a'")
            connection.commit()
        with self.assertRaisesRegex(ValueError, "order differs"):
            self.engine()

    def test_bm25_payload_identity_corruption_rejected(self):
        with closing(sqlite3.connect(self.bm25_index)) as connection:
            payload = json.loads(connection.execute("SELECT payload FROM chunks WHERE chunk_id='a'").fetchone()[0])
            payload["document_id"] = "wrong"
            connection.execute("UPDATE chunks SET payload=? WHERE chunk_id='a'", (json.dumps(payload),))
            connection.commit()
        with self.engine() as engine:
            with self.assertRaisesRegex(ValueError, "payload identity"):
                engine.vector(self.query, 1)

    def test_invalid_queries_limits_and_closed_session(self):
        with self.engine() as engine:
            for query in [np.array([0, 0]), np.array([1, 1]), np.array([1, 0, 0]),
                          np.array([[1, 0], [1, 0]]), np.array([float("nan"), 0])]:
                with self.assertRaises(ValueError):
                    engine.vector(query)
            with self.assertRaises(ValueError):
                engine.hybrid("Revenue", self.query, top_k=5, candidate_k=4)
            with self.assertRaises(ValueError):
                engine.hybrid("Revenue", self.query, vector_results=[candidate("foreign", 1)])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.bm25("Revenue")


if __name__ == "__main__":
    unittest.main()
