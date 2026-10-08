"""CPU FAISS indexes over complete, normalized chunk embeddings.

Vector IDs are the zero-based physical line numbers of the source chunks JSONL.
The manifest is the commit marker for an index and its compact metadata mapping;
loaders verify the index hash so an interrupted replacement cannot look complete.
ANN neighbor recall measures agreement with Flat, not financial evidence recall.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping
import uuid

import numpy as np


SCHEMA_VERSION = 1
NORM_TOLERANCE = 1e-4
MIN_TRAINING_POINTS_PER_CENTROID = 39


def _faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("Install the project's [vectors] optional dependencies to use FAISS") from error
    return faiss


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _vectors(value: np.ndarray | str | Path, *, dimension: int | None = None) -> np.ndarray:
    array = np.load(value, mmap_mode="r", allow_pickle=False) if isinstance(value, (str, Path)) else value
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Vectors must be a two-dimensional numpy array or .npy file")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("Vectors must contain at least one row and one dimension")
    if array.dtype != np.dtype("float32"):
        raise ValueError("Vectors must have float32 dtype; normalize and convert before indexing")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(f"Vector dimension {array.shape[1]} does not match index dimension {dimension}")
    return array


def validate_vectors(vectors: np.ndarray | str | Path, *, dimension: int | None = None,
                     batch_size: int = 1024) -> dict:
    """Validate all rows in bounded memory; never normalize or drop rows silently."""
    batch_size = _positive_int(batch_size, "batch_size")
    array = _vectors(vectors, dimension=dimension)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": "float32"}, sort_keys=True).encode())
    minimum, maximum = float("inf"), 0.0
    for start in range(0, len(array), batch_size):
        batch = np.ascontiguousarray(array[start:start + batch_size])
        if not np.isfinite(batch).all():
            raise ValueError(f"Vectors contain non-finite values in rows {start}:{start + len(batch)}")
        norms = np.linalg.norm(batch.astype(np.float64), axis=1)
        if np.any(norms == 0):
            raise ValueError(f"Vectors contain a zero row in rows {start}:{start + len(batch)}")
        if np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
            raise ValueError(f"Vectors must be L2 normalized (tolerance {NORM_TOLERANCE})")
        minimum, maximum = min(minimum, float(norms.min())), max(maximum, float(norms.max()))
        digest.update(batch.astype("<f4", copy=False).tobytes(order="C"))
    return {"vector_count": len(array), "dimension": array.shape[1], "dtype": "float32",
            "normalized": True, "norm_tolerance": NORM_TOLERANCE,
            "minimum_norm": minimum, "maximum_norm": maximum,
            "vectors_sha256": digest.hexdigest(),
            "vectors_sha256_format": "sorted JSON shape/dtype followed by row-major little-endian float32 bytes"}


@contextmanager
def _threads(count: int):
    count = _positive_int(count, "threads")
    faiss = _faiss()
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(count)
    try:
        yield faiss
    finally:
        faiss.omp_set_num_threads(previous)


def manifest_path(index_path: str | Path) -> Path:
    return Path(index_path).with_suffix(".manifest.json")


def _json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _metadata(chunks_path: Path, destination: Path, expected_count: int) -> dict:
    documents, chunk_ids = set(), set()
    digest = hashlib.sha256()
    count = 0
    with chunks_path.open("rb") as source, destination.open("w", encoding="utf-8", newline="\n") as target:
        while True:
            offset = source.tell()
            raw = source.readline()
            if not raw:
                break
            digest.update(raw)
            if not raw.strip():
                raise ValueError(f"Blank source chunks line {count + 1}; physical row IDs must not shift")
            chunk = json.loads(raw)
            if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"]:
                raise ValueError(f"Missing chunk_id at source line {count + 1}")
            if chunk["chunk_id"] in chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
            document_id = chunk.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"Missing document_id at source line {count + 1}")
            chunk_ids.add(chunk["chunk_id"])
            documents.add(document_id)
            record = {"vector_id": count, "chunk_id": chunk["chunk_id"], "document_id": document_id,
                      "source_jsonl_byte_offset": offset, "source_jsonl_byte_length": len(raw)}
            for key in ("source_file", "ticker", "fiscal_year", "source_pages", "kind", "token_count"):
                if key in chunk:
                    record[key] = chunk[key]
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
        target.flush()
        os.fsync(target.fileno())
    if count != expected_count:
        raise ValueError(f"Chunk/vector row count mismatch: {count} chunks, {expected_count} vectors")
    return {"chunks_path": str(chunks_path.resolve()), "chunks_sha256": digest.hexdigest(),
            "chunk_count": count, "document_count": len(documents), "document_ids": sorted(documents),
            "metadata_sha256": sha256_file(destination)}


def build_faiss_index(vectors: np.ndarray | str | Path, output_path: str | Path, *,
                      kind: str = "flat", chunks_path: str | Path | None = None,
                      nlist: int = 256, nprobe: int = 16,
                      hnsw_m: int = 32, ef_construction: int = 200, ef_search: int = 64,
                      train_seed: int = 20261008, train_sample_size: int = 100000,
                      threads: int = 1, batch_size: int = 8192) -> dict:
    """Build Flat/IP, IVF/Flat/IP, or HNSW/Flat/IP and atomically publish each file.

    Inputs must already be unit normalized float32 embeddings. Array row i maps
    exactly to source JSONL physical line i. For IVF, effective nlist is reduced
    when necessary to retain >=39 training examples per centroid, and both
    requested/effective settings are recorded. Fewer than 39 rows is rejected.
    HNSW construction uses one thread for reproducibility, regardless of the
    requested search thread count; graph insertion is otherwise nondeterministic.
    """
    started = time.perf_counter()
    kind = kind.lower()
    if kind not in {"flat", "ivf", "hnsw"}:
        raise ValueError("kind must be flat, ivf, or hnsw")
    for name, value in (("nlist", nlist), ("nprobe", nprobe), ("hnsw_m", hnsw_m),
                        ("ef_construction", ef_construction), ("ef_search", ef_search),
                        ("train_sample_size", train_sample_size), ("threads", threads), ("batch_size", batch_size)):
        _positive_int(value, name)
    if kind == "hnsw" and hnsw_m < 2:
        raise ValueError("hnsw_m must be at least 2 for a finite HNSW level multiplier")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int) or not 0 <= train_seed < 2**31:
        raise ValueError("train_seed must be a non-negative signed 32-bit integer")
    array = _vectors(vectors)
    validation_batch_size = min(batch_size, 1024)
    validation = validate_vectors(array, batch_size=validation_batch_size)
    count, dimension = array.shape
    target = Path(output_path).resolve()
    manifest_target = manifest_path(target)
    metadata_target = target.with_suffix(".metadata.jsonl")
    if len({target, manifest_target, metadata_target}) != 3:
        raise ValueError("Index path conflicts with a manifest or metadata sidecar")
    sources = [Path(path).resolve() for path in (vectors, chunks_path) if isinstance(path, (str, Path))]
    if any(source in {target, manifest_target, metadata_target} for source in sources):
        raise ValueError("Index outputs must not overwrite input vectors or chunks")
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".building-" + uuid.uuid4().hex
    temporary_index = target.with_name(target.name + suffix)
    temporary_metadata = metadata_target.with_name(metadata_target.name + suffix)
    temporary_manifest = manifest_target.with_name(manifest_target.name + suffix)
    source_metadata = {"document_count": None, "chunk_count": count,
                       "metadata_path": None, "chunks_path": None}
    effective_threads = 1 if kind == "hnsw" else threads
    config: dict[str, Any] = {"kind": kind, "metric": "inner_product", "similarity": "cosine",
                              "requested_threads": threads, "construction_threads": effective_threads,
                              "train_seed": train_seed, "batch_size": batch_size,
                              "validation_batch_size": validation_batch_size}
    try:
        if chunks_path is not None:
            source_metadata = _metadata(Path(chunks_path), temporary_metadata, count)
            source_metadata["metadata_path"] = str(metadata_target)
        with _threads(effective_threads) as faiss:
            training_seconds = 0.0
            if kind == "flat":
                index = faiss.IndexFlatIP(dimension)
                config.update({"class": "IndexFlatIP", "training_sample_size": 0})
            elif kind == "ivf":
                sample_count = min(count, train_sample_size)
                if sample_count < MIN_TRAINING_POINTS_PER_CENTROID:
                    raise ValueError("IVF requires at least 39 training vectors; use Flat for this sample")
                effective_nlist = min(nlist, sample_count // MIN_TRAINING_POINTS_PER_CENTROID)
                effective_nprobe = min(nprobe, effective_nlist)
                index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, effective_nlist,
                                           faiss.METRIC_INNER_PRODUCT)
                index.cp.seed = train_seed
                index.cp.min_points_per_centroid = MIN_TRAINING_POINTS_PER_CENTROID
                # Avoid an unrecorded second subsampling stage inside FAISS.
                index.cp.max_points_per_centroid = max(index.cp.max_points_per_centroid,
                                                       (sample_count + effective_nlist - 1) // effective_nlist)
                sample_ids = np.sort(np.random.default_rng(train_seed).choice(count, sample_count, replace=False))
                sample_ids_sha256 = hashlib.sha256(sample_ids.astype("<i8", copy=False).tobytes()).hexdigest()
                train_vectors = np.ascontiguousarray(array[sample_ids])
                training_start = time.perf_counter()
                index.train(train_vectors)
                training_seconds = time.perf_counter() - training_start
                # FAISS retains learned centroids, not the caller's training
                # matrix. Release its corpus-sized copy before adding vectors.
                del train_vectors, sample_ids
                index.nprobe = effective_nprobe
                config.update({"class": "IndexIVFFlat", "requested_nlist": nlist, "nlist": effective_nlist,
                               "requested_nprobe": nprobe, "nprobe": effective_nprobe,
                               "requested_training_sample_size": train_sample_size,
                               "training_sample_size": sample_count,
                               "training_sample_ids_sha256": sample_ids_sha256,
                               "training_iterations": index.cp.niter,
                               "minimum_training_points_per_centroid": MIN_TRAINING_POINTS_PER_CENTROID,
                               "adaptive_nlist": effective_nlist != nlist})
            else:
                index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = ef_construction
                index.hnsw.efSearch = ef_search
                index.hnsw.rng = faiss.RandomGenerator(train_seed)
                config.update({"class": "IndexHNSWFlat", "hnsw_m": hnsw_m,
                               "ef_construction": ef_construction, "ef_search": ef_search,
                               "training_sample_size": 0, "construction_seed": train_seed,
                               "single_thread_construction_reason": "deterministic graph insertion"})
            add_started = time.perf_counter()
            for start in range(0, count, batch_size):
                index.add(np.ascontiguousarray(array[start:start + batch_size]))
            add_seconds = time.perf_counter() - add_started
            if index.ntotal != count or index.d != dimension or not index.is_trained:
                raise RuntimeError("FAISS index dimensions/count/training state do not match inputs")
            faiss.write_index(index, str(temporary_index))
            versions = {"faiss": faiss.__version__, "numpy": np.__version__, "python": platform.python_version()}
        vector_source = ({"vectors_path": str(Path(vectors).resolve()), "vectors_file_sha256": sha256_file(vectors)}
                         if isinstance(vectors, (str, Path)) else {"vectors_path": None})
        manifest = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "complete", "index_path": str(target), "manifest_path": str(manifest_target),
                    "index_sha256": sha256_file(temporary_index), "index_size_bytes": temporary_index.stat().st_size,
                    **validation, **vector_source, **source_metadata, "configuration": config,
                    "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "implementation_sha256": sha256_file(__file__),
                    "versions": versions, "training_seconds": training_seconds, "add_seconds": add_seconds,
                    "build_seconds": time.perf_counter() - started,
                    "id_mapping": "FAISS vector ID equals zero-based physical line number in source chunks JSONL",
                    "score_order": "descending cosine similarity",
                    "tie_policy": "returned equal-score neighbors sorted by vector ID; boundary ties retain FAISS candidate set"}
        _json(temporary_manifest, manifest)
        os.replace(temporary_index, target)
        if chunks_path is not None:
            os.replace(temporary_metadata, metadata_target)
        os.replace(temporary_manifest, manifest_target)
        return manifest
    finally:
        # These exact temporary paths were created in the index output directory.
        for temporary in (temporary_index, temporary_metadata, temporary_manifest):
            if temporary.parent == target.parent and temporary.name.endswith(suffix) and temporary.exists():
                temporary.unlink()


def load_faiss_index(index_path: str | Path, *, verify: bool = True):
    """Load a completed trusted local index; verify its manifest binding by default."""
    target = Path(index_path)
    manifest = json.loads(manifest_path(target).read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Index manifest is not a supported complete index")
    if verify and sha256_file(target) != manifest["index_sha256"]:
        raise ValueError("Index checksum does not match its manifest")
    if verify and manifest.get("metadata_path"):
        if sha256_file(manifest["metadata_path"]) != manifest["metadata_sha256"]:
            raise ValueError("Metadata checksum does not match its manifest")
    index = _faiss().read_index(str(target))
    if index.ntotal != manifest["vector_count"] or index.d != manifest["dimension"]:
        raise ValueError("Index count/dimension do not match its manifest")
    if index.metric_type != _faiss().METRIC_INNER_PRODUCT:
        raise ValueError("Index metric must be inner product for normalized embeddings")
    return index


def load_chunk_mapping(index_path: str | Path, *, verify_source: bool = True) -> list[dict]:
    """Load compact pointers, verifying the source before citation-aware lookup.

    Call once when opening a retrieval session, then reuse the mapping and file
    handle. This keeps full-corpus checksum work out of individual query timing.
    """
    manifest = json.loads(manifest_path(index_path).read_text(encoding="utf-8"))
    if not manifest.get("metadata_path"):
        raise ValueError("Index was built without a source chunks mapping")
    if sha256_file(manifest["metadata_path"]) != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum does not match its manifest")
    if verify_source and sha256_file(manifest["chunks_path"]) != manifest["chunks_sha256"]:
        raise ValueError("Source chunks checksum does not match its manifest")
    with Path(manifest["metadata_path"]).open(encoding="utf-8") as stream:
        mapping = [json.loads(line) for line in stream]
    if len(mapping) != manifest["vector_count"] or any(row["vector_id"] != i for i, row in enumerate(mapping)):
        raise ValueError("Metadata row IDs/count do not match the index")
    return mapping


def _ranked_search(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = index.search(queries, k)
    # FAISS has no global ID tie guarantee at the k boundary. We stabilize the
    # order within the returned candidate set and explicitly document that limit.
    for row in range(len(indices)):
        order = np.lexsort((np.where(indices[row] < 0, np.iinfo(np.int64).max, indices[row]), -distances[row]))
        distances[row], indices[row] = distances[row, order], indices[row, order]
    return distances, indices


def search_faiss_index(index_or_path, query_vectors: np.ndarray | str | Path, *,
                       k: int = 5, threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine scores and stable zero-based row IDs, highest score first."""
    k = _positive_int(k, "k")
    index = load_faiss_index(index_or_path) if isinstance(index_or_path, (str, Path)) else index_or_path
    queries = _vectors(query_vectors, dimension=index.d)
    validate_vectors(queries, dimension=index.d)
    if index.metric_type != _faiss().METRIC_INNER_PRODUCT:
        raise ValueError("Expected an inner-product index")
    with _threads(threads):
        return _ranked_search(index, np.ascontiguousarray(queries), k)


def benchmark_indexes(indexfiles: Mapping[str, str | Path], query_vectors: np.ndarray | str | Path,
                      *, k: int = 5, warmup: int = 2, repeats: int = 3, threads: int = 1) -> dict:
    """Compare ANN neighbors with exact Flat and measure single-query search time.

    Loading, embedding, input validation, and warmup are excluded from latency.
    Each query runs alone with the same FAISS CPU thread setting. Raw index.search
    is timed; returned candidates are then tie-sorted outside the timed section.
    Exact-neighbor recall uses ID-set overlap; equivalent boundary ties can reduce
    this metric even when cosine scores are identical, so ties are counted too.
    """
    k, repeats = _positive_int(k, "k"), _positive_int(repeats, "repeats")
    _positive_int(threads, "threads")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not indexfiles:
        raise ValueError("Provide at least one Flat index")
    manifests = {}
    for name, path in indexfiles.items():
        manifests[name] = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    flat_names = [name for name in indexfiles if manifests[name]["configuration"]["kind"] == "flat"]
    if len(flat_names) != 1:
        raise ValueError("Benchmark requires exactly one Flat baseline")
    flat_name = flat_names[0]
    flat_manifest = manifests[flat_name]
    for manifest in manifests.values():
        for field in ("vectors_sha256", "vector_count", "dimension", "chunks_sha256"):
            if manifest.get(field) != flat_manifest.get(field):
                raise ValueError(f"Compared indexes must share identical {field}")
    queries = _vectors(query_vectors, dimension=flat_manifest["dimension"])
    query_stats = validate_vectors(queries)
    queries = np.ascontiguousarray(queries)
    effective_k = min(k, flat_manifest["vector_count"])
    with _threads(threads):
        # Match timed single-query searches exactly. FAISS may dispatch batches
        # and single queries to different SIMD/BLAS/database-parallel kernels;
        # their floating-point reductions can change near-tied neighbor IDs.
        exact_ids = np.empty((len(queries), k), dtype=np.int64)
        boundary_ties, near_boundary_ties, boundary_gaps = [], [], []
        flat_index = load_faiss_index(indexfiles[flat_name])
        try:
            for query_number, query in enumerate(queries):
                single_query = query.reshape(1, -1)
                exact_ids[query_number] = _ranked_search(flat_index, single_query, k)[1][0]
                extra_scores, extra_ids = _ranked_search(flat_index, single_query, k + 1)
                gap = (float(extra_scores[0, k - 1]) - float(extra_scores[0, k])
                       if extra_ids[0, k] >= 0 else None)
                boundary_gaps.append(gap)
                boundary_ties.append(gap == 0.0)
                near_boundary_ties.append(gap is not None and abs(gap) <= 1e-6)
        finally:
            del flat_index
        comparisons = []
        for name, path in indexfiles.items():
            # A large benchmark needs only one resident index at a time. Index
            # loading/checksumming remain outside all search latency samples.
            index = load_faiss_index(path)
            try:
                for _ in range(warmup):
                    for query in queries:
                        index.search(query.reshape(1, -1), k)
                per_query, all_times = [], []
                for query_number, query in enumerate(queries):
                    times = []
                    candidates = None
                    scores = None
                    for _ in range(repeats):
                        start = time.perf_counter_ns()
                        scores, candidates = index.search(query.reshape(1, -1), k)
                        times.append((time.perf_counter_ns() - start) / 1_000_000)
                    candidate_set = {int(value) for value in candidates[0] if value >= 0}
                    truth = {int(value) for value in exact_ids[query_number] if value >= 0}
                    recall = len(candidate_set.intersection(truth)) / effective_k
                    ordering = np.lexsort((candidates[0], -scores[0]))
                    per_query.append({"query_index": query_number,
                                      f"exact_neighbor_recall_at_{k}": recall,
                                      "exact_boundary_tie": boundary_ties[query_number],
                                      "near_boundary_tie": near_boundary_ties[query_number],
                                      "exact_boundary_score_gap": boundary_gaps[query_number],
                                      "mean_search_ms": float(np.mean(times)), "search_ms_samples": times,
                                      "vector_ids": [int(value) for value in candidates[0, ordering] if value >= 0],
                                      "scores": [float(scores[0, position]) for position in ordering if candidates[0, position] >= 0]})
                    all_times.extend(times)
            finally:
                del index
            manifest = manifests[name]
            comparisons.append({"name": name, "kind": manifest["configuration"]["kind"],
                                "vector_count": manifest["vector_count"], "dimension": manifest["dimension"],
                                "index_size_bytes": manifest["index_size_bytes"],
                                "build_seconds": manifest["build_seconds"], "configuration": manifest["configuration"],
                                f"exact_neighbor_recall_at_{k}": float(np.mean([row[f"exact_neighbor_recall_at_{k}"] for row in per_query])),
                                "mean_search_ms": float(np.mean(all_times)),
                                "median_search_ms": float(np.median(all_times)),
                                "p95_search_ms": float(np.percentile(all_times, 95)), "per_query": per_query})
    return {"schema_version": SCHEMA_VERSION, "metric": f"exact_neighbor_recall_at_{k}",
            "metric_definition": "Mean overlap with exact Flat top-k vector IDs divided by min(k, vector count); this is not gold evidence Recall@k",
            "flat_baseline": flat_name, "k": k, "effective_k": effective_k,
            "query_count": len(queries), "query_vectors_sha256": query_stats["vectors_sha256"],
            "faiss_threads": threads, "warmup_passes": warmup, "repeats_per_query": repeats,
            "reference_search_policy": "Flat reference and timed searches both use batch size 1, identical k, and identical FAISS thread count",
            "index_loading_policy": "One resident index at a time; reference Flat released before comparison stages",
            "latency_scope": "Single-query CPU index.search only; excludes embedding, loading, validation, sorting, and warmup",
            "boundary_tie_query_count": sum(boundary_ties),
            "near_boundary_tie_query_count": sum(near_boundary_ties),
            "near_boundary_tie_absolute_tolerance": 1e-6,
            "tie_policy": "ID-set recall can penalize equivalent candidates tied at the kth score",
            "results": comparisons}
