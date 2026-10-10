"""Verified FAISS, BM25, and weighted score-fusion retrieval sessions.

The two raw scores have different scales and directions. Fusion converts
SQLite's negative, lower-is-better BM25 score to its negative, then min-max
normalizes each channel over its own candidate list. Missing candidates score
zero. A nonempty constant-score channel receives one for every candidate.
The endpoints use only the active channel and preserve its ranking exactly.
No question labels, expected documents, or benchmark answers enter retrieval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from report_chunker.retrieval import _search, index_stats
from report_index.faiss_index import (
    load_chunk_mapping, load_faiss_index, manifest_path, search_faiss_index, sha256_file,
)


NORMALIZATION = "per-query candidate-list min-max; negate SQLite BM25; constant list=1; missing=0"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _alpha(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("alpha must be a finite number between zero and one")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("alpha must be a finite number between zero and one")
    return value


def _raw_score(result: dict, channel: str) -> float:
    value = result.get("score_breakdown", {}).get(f"{channel}_raw_score")
    if value is None:
        value = result.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"Missing numeric {channel} score")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {channel} score")
    return value


def _same_chunk(first: dict, second: dict) -> None:
    for key in ("document_id", "text"):
        if key in first and key in second and first[key] != second[key]:
            raise ValueError(f"Conflicting payloads for chunk_id {first['chunk_id']!r}: {key}")


def _deduplicate(results: Iterable[dict], channel: str) -> dict[str, tuple[dict, float]]:
    selected: dict[str, tuple[dict, float]] = {}
    for result in results:
        chunk_id = result.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every candidate needs a nonempty chunk_id")
        score = _raw_score(result, channel)
        if chunk_id in selected:
            previous, previous_score = selected[chunk_id]
            _same_chunk(previous, result)
            if (score <= previous_score if channel == "vector" else score >= previous_score):
                continue
        selected[chunk_id] = (result, score)
    return selected


def _normalized(candidates: dict[str, tuple[dict, float]], channel: str) -> dict[str, float]:
    if not candidates:
        return {}
    scores = {key: score if channel == "vector" else -score
              for key, (_, score) in candidates.items()}
    minimum, maximum = min(scores.values()), max(scores.values())
    if maximum == minimum:
        return dict.fromkeys(scores, 1.0)
    # Division before subtraction also handles otherwise overflowing finite
    # values supplied by a caller; ordinary cosine/BM25 scores are much smaller.
    scale = max(abs(minimum), abs(maximum), 1.0)
    low, span = minimum / scale, maximum / scale - minimum / scale
    return {key: (value / scale - low) / span for key, value in scores.items()}


def _citations(chunk: dict) -> list[dict]:
    values = list(chunk.get("citations", []))
    if chunk.get("citation"):
        values.append(chunk["citation"])
    values.extend(atom["citation"] for atom in chunk.get("evidence", []) if atom.get("citation"))
    return list({json.dumps(value, sort_keys=True): value for value in values}.values())


def weighted_fusion(vector_results: Iterable[dict], bm25_results: Iterable[dict],
                    alpha: float = 0.5, top_k: int = 5) -> list[dict]:
    """Fuse a deduplicated union; alpha is the vector channel's weight.

    Normalization uses the supplied candidate pools, before the top-k cutoff.
    The returned ``score`` is always alpha*dense_norm+(1-alpha)*bm25_norm.
    At alpha=0/1, candidates exclusive to the inactive channel are excluded,
    and the active raw score orders results, preserving pure-channel ties.
    Input payloads and scores are never mutated.
    """
    alpha = _alpha(alpha)
    _positive_integer(top_k, "top_k")
    vector = _deduplicate(vector_results, "vector")
    lexical = _deduplicate(bm25_results, "bm25")
    for chunk_id in vector.keys() & lexical.keys():
        _same_chunk(vector[chunk_id][0], lexical[chunk_id][0])
    vector_norm, bm25_norm = _normalized(vector, "vector"), _normalized(lexical, "bm25")
    keys = lexical.keys() if alpha == 0 else vector.keys() if alpha == 1 else vector.keys() | lexical.keys()
    results = []
    for chunk_id in keys:
        original = (vector.get(chunk_id) or lexical[chunk_id])[0]
        dense, bm25 = vector_norm.get(chunk_id, 0.0), bm25_norm.get(chunk_id, 0.0)
        score = alpha * dense + (1 - alpha) * bm25
        result = dict(original)
        result.update(score=score, retrieval_approach="hybrid", citations=_citations(original),
                      score_breakdown={
                          "vector_raw_score": vector[chunk_id][1] if chunk_id in vector else None,
                          "bm25_raw_score": lexical[chunk_id][1] if chunk_id in lexical else None,
                          "vector_normalized_score": dense, "bm25_normalized_score": bm25,
                          "alpha": alpha, "weighted_score": score, "normalization": NORMALIZATION,
                      })
        results.append(result)
    if alpha == 0:
        results.sort(key=lambda row: (row["score_breakdown"]["bm25_raw_score"], row["chunk_id"]))
    elif alpha == 1:
        results.sort(key=lambda row: (-row["score_breakdown"]["vector_raw_score"], row["chunk_id"]))
    else:
        results.sort(key=lambda row: (-row["score"], row["chunk_id"]))
    for rank, result in enumerate(results[:top_k], 1):
        result["rank"] = rank
    return results[:top_k]


class RetrievalEngine:
    """One resident FAISS index and a read-only SQLite payload/FTS connection.

    Initialization verifies the chunks SHA256 against both indexes, the FAISS
    artifact/compact-map hashes, and every ordered chunk/document mapping in
    SQLite. Full chunk text is loaded only for retrieved candidates. An engine
    is a single-thread-owned session; create another session for another thread.
    """

    def __init__(self, chunks_path: str | Path, vector_index_path: str | Path,
                 bm25_index_path: str | Path, threads: int = 4):
        self.threads = _positive_integer(threads, "threads")
        self.chunks_path = Path(chunks_path).resolve()
        self.vector_index_path = Path(vector_index_path).resolve()
        self.bm25_index_path = Path(bm25_index_path).resolve()
        self.connection = None
        self.index = None
        self.vector_manifest = json.loads(manifest_path(self.vector_index_path).read_text(encoding="utf-8"))
        self.bm25_stats = index_stats(self.bm25_index_path)
        digest = sha256_file(self.chunks_path)
        if any(record.get("chunks_sha256") != digest
               for record in (self.vector_manifest, self.bm25_stats)):
            raise ValueError("Source chunks SHA256 differs from the vector or BM25 index")
        if not self.vector_manifest.get("normalized"):
            raise ValueError("Vector index must use normalized embeddings")
        mapping = load_chunk_mapping(self.vector_index_path, verify_source=False)
        self._mapping = [(entry["chunk_id"], entry["document_id"]) for entry in mapping]
        del mapping
        self._chunk_ids = {chunk_id for chunk_id, _ in self._mapping}
        if len(self._chunk_ids) != len(self._mapping):
            raise ValueError("Vector mapping contains duplicate chunk IDs")
        rows_by_document: dict[str, list[int]] = {}
        for vector_id, (_, document_id) in enumerate(self._mapping):
            rows_by_document.setdefault(document_id, []).append(vector_id)
        self._document_rows = {key: np.asarray(rows, dtype=np.int64) for key, rows in rows_by_document.items()}
        expected_count = len(self._mapping)
        if (self.vector_manifest.get("chunk_count") != expected_count
                or self.bm25_stats.get("chunk_count") != expected_count):
            raise ValueError("Vector and BM25 chunk counts differ")
        documents = {document_id for _, document_id in self._mapping}
        if (len(documents) != self.vector_manifest.get("document_count")
                or len(documents) != self.bm25_stats.get("document_count")
                or sorted(documents) != self.vector_manifest.get("document_ids")):
            raise ValueError("Vector and BM25 document counts or document IDs differ")
        try:
            self.connection = sqlite3.connect(self.bm25_index_path.as_uri() + "?mode=ro", uri=True)
            self.connection.execute("PRAGMA cache_size=-8192")
            cursor = self.connection.execute("SELECT chunk_id,document_id FROM chunks ORDER BY id")
            for expected in self._mapping:
                if cursor.fetchone() != expected:
                    raise ValueError("BM25 chunk/document order differs from vector mapping")
            if cursor.fetchone() is not None:
                raise ValueError("BM25 contains additional unmapped chunks")
            self.index = load_faiss_index(self.vector_index_path)
            self.stats = {
                "chunks_sha256": digest, "chunk_count": expected_count,
                "document_count": len(documents), "document_ids": sorted(documents),
                "dimension": self.index.d, "vector_index_sha256": self.vector_manifest["index_sha256"],
                "mapping_verified": True,
            }
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        self.index = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_open(self) -> None:
        if self.connection is None or self.index is None:
            raise RuntimeError("RetrievalEngine is closed")

    def _payload(self, chunk_id: str, document_id: str) -> dict:
        row = self.connection.execute("SELECT payload,body FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None:
            raise ValueError(f"No source payload for vector chunk {chunk_id}")
        chunk = json.loads(row[0])
        if chunk.get("chunk_id") != chunk_id or chunk.get("document_id") != document_id:
            raise ValueError("SQLite payload identity differs from the verified vector mapping")
        chunk["text"] = row[1]
        return chunk

    def document_ids(self) -> list[str]:
        return sorted(self._document_rows)

    def vector(self, query_vector: np.ndarray, top_k: int = 5, *,
               document_ids: Iterable[str] | None = None) -> list[dict]:
        """Search normalized query embeddings; resolve all cutoff ties by chunk ID.

        Flat search considers the entire corpus, or only ``document_ids`` when
        given (an exact scoped search, not a post-filter of a global Top-K).
        With an approximate index, the tie rule applies to visited candidates.
        """
        self._require_open()
        _positive_integer(top_k, "top_k")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query_vector must contain exactly one query")
        selector, total = None, self.index.ntotal
        if document_ids is not None:
            requested = sorted(set(document_ids))
            unknown = [document_id for document_id in requested if document_id not in self._document_rows]
            if not requested or unknown:
                raise ValueError("document_ids must name documents in this verified corpus")
            selector = np.concatenate([self._document_rows[document_id] for document_id in requested])
            total = int(selector.size)
        wanted = min(top_k, total)
        search_k = min(wanted + 1, total)
        while True:
            scores, ids = search_faiss_index(self.index, query, k=search_k, threads=self.threads,
                                             selector_ids=selector)
            candidates = [(float(score), int(index)) for score, index in zip(scores[0], ids[0]) if index >= 0]
            candidates.sort(key=lambda row: (-row[0], self._mapping[row[1]][0]))
            if (search_k == total or len(candidates) <= wanted
                    or candidates[wanted - 1][0] != candidates[-1][0]):
                break
            search_k = min(total, max(search_k + 1, search_k * 2))
        results = []
        for rank, (score, vector_id) in enumerate(candidates[:wanted], 1):
            chunk = self._payload(*self._mapping[vector_id])
            chunk.update(rank=rank, vector_id=vector_id, score=score, retrieval_approach="vector",
                         citations=_citations(chunk),
                         score_breakdown={"vector_raw_score": score, "bm25_raw_score": None})
            results.append(chunk)
        return results

    def bm25(self, question: str, top_k: int = 5) -> list[dict]:
        """Search the question text without gold-derived metadata filters."""
        self._require_open()
        results = _search(self.connection, question, top_k, None, None)
        for chunk in results:
            chunk.update(retrieval_approach="bm25", citations=_citations(chunk),
                         score_breakdown={"vector_raw_score": None, "bm25_raw_score": chunk["score"]})
        return results

    def hybrid(self, question: str, query_vector: np.ndarray, top_k: int = 5,
               candidate_k: int = 100, alpha: float = 0.5, *,
               vector_results: list[dict] | None = None,
               bm25_results: list[dict] | None = None) -> list[dict]:
        """Fuse candidate pools, optionally reusing caller-cached query results.

        Cached candidates must belong to this query and corpus. The caller
        owns query/cache identity; the engine checks every supplied chunk ID.
        """
        self._require_open()
        _positive_integer(top_k, "top_k")
        _positive_integer(candidate_k, "candidate_k")
        alpha = _alpha(alpha)
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")
        vector_results = ([] if alpha == 0 else self.vector(query_vector, candidate_k)) if vector_results is None else vector_results
        bm25_results = ([] if alpha == 1 else self.bm25(question, candidate_k)) if bm25_results is None else bm25_results
        if any(result.get("chunk_id") not in self._chunk_ids for result in vector_results + bm25_results):
            raise ValueError("Cached candidate does not belong to this verified corpus")
        return weighted_fusion(vector_results[:candidate_k], bm25_results[:candidate_k], alpha, top_k)
