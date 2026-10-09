"""Isolated FAISS build and search measurements, with explicit metric scopes.

Invoke ``python -m report_index.comparison`` once per build or query measurement.
The query worker loads no corpus vectors or chunk payloads: its working-set delta
measures a loaded index and FAISS search scratch, not serialized file size.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import time
from typing import Mapping

import numpy as np

from .faiss_index import (
    _faiss, _positive_int, _threads, _vectors, build_faiss_index,
    load_faiss_index, manifest_path, sha256_file, validate_vectors,
)


SCHEMA_VERSION = 1
DEFAULT_KS = (3, 5, 10, 20, 50)


def _write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, destination)


def resident_bytes() -> int:
    """Current resident bytes, not the process lifetime peak or file size."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    proc_statm = Path("/proc/self/statm")
    if proc_statm.exists():
        return int(proc_statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("Current resident memory measurement requires psutil on this platform") from error
    return int(psutil.Process().memory_info().rss)


def _manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("Expected a complete FAISS manifest")
    return payload


def rebuild_parameters(source: dict) -> dict:
    """Recover every user-configurable build setting from an existing manifest."""
    config = source["configuration"]
    result = {"kind": config["kind"], "threads": config["requested_threads"],
              "train_seed": config["train_seed"], "batch_size": config["batch_size"]}
    if config["kind"] == "ivf":
        result.update(nlist=config["requested_nlist"], nprobe=config["requested_nprobe"],
                      train_sample_size=config["requested_training_sample_size"])
    elif config["kind"] == "hnsw":
        result.update(hnsw_m=config["hnsw_m"], ef_construction=config["ef_construction"],
                      ef_search=config["ef_search"])
    elif config["kind"] != "flat":
        raise ValueError("Unsupported index kind in source manifest")
    return result


def build_worker(source_manifest: str | Path, vectors: str | Path, chunks: str | Path,
                 index_path: str | Path, *, output: str | Path | None = None) -> dict:
    """Freshly rebuild an index with identical vectors, mapping, and settings.

    ``build_seconds`` means FAISS training plus add time. The separately recorded
    pipeline time includes validation, source mapping, serialization, and hashes.
    Embedding and subprocess startup are excluded from both.
    """
    source_path = Path(source_manifest).resolve()
    source = _manifest(source_path)
    target = Path(index_path).resolve()
    if target == source_path or target == Path(source["index_path"]).resolve():
        raise ValueError("Comparison rebuild must use a new index path")
    vectors, chunks = Path(vectors).resolve(), Path(chunks).resolve()
    if sha256_file(vectors) != source.get("vectors_file_sha256"):
        raise ValueError("Rebuild vectors checksum differs from source index")
    if sha256_file(chunks) != source.get("chunks_sha256"):
        raise ValueError("Rebuild chunks checksum differs from source index")
    started = time.perf_counter()
    rebuilt = build_faiss_index(vectors, target, chunks_path=chunks, **rebuild_parameters(source))
    pipeline_seconds = time.perf_counter() - started
    for field in ("vectors_sha256", "vector_count", "dimension", "chunks_sha256", "configuration_sha256"):
        if rebuilt[field] != source[field]:
            raise ValueError(f"Rebuilt index differs from original {field}")
    result = {"schema_version": SCHEMA_VERSION, "measurement": "fresh_build", "process_id": os.getpid(),
              "source_manifest": str(source_path), "source_manifest_sha256": sha256_file(source_path),
              "index_path": str(target), "index_sha256": rebuilt["index_sha256"],
              "configuration": rebuilt["configuration"], "vector_count": rebuilt["vector_count"],
              "dimension": rebuilt["dimension"], "vectors_sha256": rebuilt["vectors_sha256"],
              "chunks_sha256": rebuilt["chunks_sha256"], "versions": rebuilt["versions"],
              "training_seconds": rebuilt["training_seconds"], "add_seconds": rebuilt["add_seconds"],
              "build_seconds": rebuilt["training_seconds"] + rebuilt["add_seconds"],
              "pipeline_seconds": pipeline_seconds, "index_size_bytes": rebuilt["index_size_bytes"],
              "build_time_scope": "FAISS index.train plus index.add; excludes embedding, validation, mapping, serialization, and hashes",
              "pipeline_time_scope": "build_faiss_index call including input validation, metadata mapping, train/add, serialization, and hashes; excludes embedding and preflight input checksum verification",
              "implementation_sha256": sha256_file(__file__)}
    if output is not None:
        _write_json(output, result)
    return result


def _ordered(scores: np.ndarray, indices: np.ndarray) -> dict:
    valid = indices >= 0
    ids, values = indices[valid], scores[valid]
    order = np.lexsort((ids, -values))
    return {"vector_ids": [int(value) for value in ids[order]],
            "scores": [float(value) for value in values[order]]}


def query_worker(index_path: str | Path, query_vectors: str | Path, *,
                 ks=DEFAULT_KS, latency_k: int = 10, warmup: int = 1,
                 repeats: int = 5, threads: int = 4,
                 output: str | Path | None = None) -> dict:
    """Measure an independently loaded index; run in a fresh Python subprocess.

    Recall candidates use an independent single-query search at each requested k.
    The latency loop times exactly k=latency_k, excluding validation, loading,
    embedding, sorting, gold matching, and serialization. All raw samples remain.
    """
    ks = tuple(sorted({_positive_int(k, "k") for k in ks}))
    if not ks:
        raise ValueError("Provide at least one recall cutoff")
    latency_k = _positive_int(latency_k, "latency_k")
    repeats = _positive_int(repeats, "repeats")
    threads = _positive_int(threads, "threads")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    target, queries_path = Path(index_path).resolve(), Path(query_vectors).resolve()
    manifest = _manifest(manifest_path(target))
    # Complete checksum work before the memory baseline: temporary hash buffers
    # must not masquerade as resident index storage.
    if sha256_file(target) != manifest["index_sha256"]:
        raise ValueError("Index checksum does not match its manifest")
    if manifest.get("metadata_path") and sha256_file(manifest["metadata_path"]) != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum does not match its manifest")
    query_file_hash = sha256_file(queries_path)
    queries = np.array(_vectors(queries_path, dimension=manifest["dimension"]), copy=True, order="C")
    query_stats = validate_vectors(queries)
    faiss = _faiss()
    faiss_version = faiss.__version__
    per_query, all_samples = [], []
    with _threads(threads):
        gc.collect()
        rss_baseline = resident_bytes()
        # Validation remains active above; avoid re-reading files between the
        # baseline and the post-load measurement.
        index = load_faiss_index(target, verify=False)
        rss_loaded = resident_bytes()
        try:
            for _ in range(warmup):
                for query in queries:
                    index.search(query.reshape(1, -1), latency_k)
            rss_warmed = resident_bytes()
            for query_number, query in enumerate(queries):
                single = query.reshape(1, -1)
                samples = []
                for _ in range(repeats):
                    started = time.perf_counter_ns()
                    index.search(single, latency_k)
                    samples.append((time.perf_counter_ns() - started) / 1_000_000)
                neighbors = {}
                for k in ks:
                    distances, indices = index.search(single, k)
                    record = _ordered(distances[0], indices[0])
                    # Diagnostic only: strict ID-set recall can penalize exact
                    # or near ties at the kth score. Do not alter candidates.
                    if manifest["configuration"]["kind"] == "flat":
                        extra, extra_ids = index.search(single, k + 1)
                        gap = float(extra[0, k - 1]) - float(extra[0, k]) if extra_ids[0, k] >= 0 else None
                        record.update(boundary_score_gap=gap, boundary_tie=gap == 0.0,
                                      near_boundary_tie=gap is not None and abs(gap) <= 1e-6)
                    neighbors[str(k)] = record
                all_samples.extend(samples)
                per_query.append({"query_index": query_number, "search_ms_samples": samples,
                                  "mean_search_ms": float(np.mean(samples)), "neighbors": neighbors})
            rss_finished = resident_bytes()
        finally:
            del index
    result = {"schema_version": SCHEMA_VERSION, "measurement": "isolated_index_query", "process_id": os.getpid(),
              "index_path": str(target), "index_sha256": manifest["index_sha256"],
              "kind": manifest["configuration"]["kind"], "configuration": manifest["configuration"],
              "vector_count": manifest["vector_count"], "dimension": manifest["dimension"],
              "vectors_sha256": manifest["vectors_sha256"], "chunks_sha256": manifest.get("chunks_sha256"),
              "query_count": len(queries), "query_vectors_path": str(queries_path),
              "query_vectors_file_sha256": query_file_hash, "query_vectors_sha256": query_stats["vectors_sha256"],
              "ks": list(ks), "latency_k": latency_k, "warmup_passes": warmup,
              "repeats_per_query": repeats, "faiss_threads": threads,
              "mean_search_ms": float(np.mean(all_samples)), "median_search_ms": float(np.median(all_samples)),
              "p95_search_ms": float(np.percentile(all_samples, 95)),
              "index_size_bytes": manifest["index_size_bytes"], "index_storage_mb": target.stat().st_size / 1_000_000,
              "rss_baseline_bytes": rss_baseline, "rss_loaded_bytes": rss_loaded,
              "rss_warmed_bytes": rss_warmed, "rss_finished_bytes": rss_finished,
              "resident_load_delta_bytes": rss_loaded - rss_baseline,
              "resident_warmed_delta_bytes": rss_warmed - rss_baseline,
              "memory_usage_mb": (rss_loaded - rss_baseline) / 1_000_000,
              "memory_scope": "Current process resident working-set/RSS increase from FAISS-imported, query-loaded baseline to index loaded; warmed delta is recorded separately; excludes embedding model, corpus input vectors, and chunk payloads",
              "memory_units": "decimal MB = 1,000,000 bytes; process RSS is operating-system dependent, not a native allocation counter or peak build memory",
              "isolation_requirement": "Run each index query worker in a new Python process; do not compare same-process loads",
              "latency_scope": "CPU index.search only, one query at a time at latency_k; excludes embedding, loading, validation, warmup, sorting, payload resolution, and gold scoring",
              "reference_search_policy": "Independent single-query search for each requested k with identical FAISS thread count; no batch-query reference or top-50-prefix substitution",
              "tie_policy": "Equal-score returned candidates sorted by vector ID; exact boundary candidate set remains FAISS-selected; strict ANN ID recall can penalize equivalent boundary ties",
              "near_boundary_tie_absolute_tolerance": 1e-6,
              "versions": {"faiss": faiss_version, "numpy": np.__version__, "python": platform.python_version()},
              "platform": platform.platform(), "implementation_sha256": sha256_file(__file__), "per_query": per_query}
    if output is not None:
        _write_json(output, result)
    return result


def compare_query_results(results: Mapping[str, dict]) -> dict:
    """Compute strict ANN neighbor recall, distinct from gold document/evidence recall."""
    flat_names = [name for name, result in results.items() if result["kind"] == "flat"]
    if len(flat_names) != 1:
        raise ValueError("Comparison requires exactly one Flat baseline")
    flat_name = flat_names[0]
    reference = results[flat_name]
    comparisons = {}
    for name, result in results.items():
        for field in ("vectors_sha256", "chunks_sha256", "vector_count", "dimension", "query_count",
                      "query_vectors_sha256", "ks", "faiss_threads", "latency_k", "warmup_passes",
                      "repeats_per_query"):
            if result[field] != reference[field]:
                raise ValueError(f"Compared query results disagree on {field}")
        rows = result["per_query"]
        if len(rows) != result["query_count"]:
            raise ValueError("Per-query row count does not match query count")
        recalls = {str(k): [] for k in reference["ks"]}
        for number, (truth_row, row) in enumerate(zip(reference["per_query"], rows)):
            if row["query_index"] != number or truth_row["query_index"] != number:
                raise ValueError("Query order does not match cached vector row order")
            for k in reference["ks"]:
                truth = set(truth_row["neighbors"][str(k)]["vector_ids"])
                candidates = set(row["neighbors"][str(k)]["vector_ids"])
                denominator = min(k, reference["vector_count"])
                if len(truth) != denominator:
                    raise ValueError("Flat reference does not contain the expected number of neighbors")
                recalls[str(k)].append(len(truth & candidates) / denominator)
        comparisons[name] = {"ann_recall": {k: float(np.mean(values)) for k, values in recalls.items()},
                             "per_query_ann_recall": recalls,
                             "mean_search_ms": result["mean_search_ms"], "median_search_ms": result["median_search_ms"],
                             "p95_search_ms": result["p95_search_ms"], "memory_usage_mb": result["memory_usage_mb"],
                             "index_storage_mb": result["index_storage_mb"]}
    return {"flat_baseline": flat_name, "query_count": reference["query_count"], "ks": reference["ks"],
            "ann_recall_definition": "Mean per-query intersection of index and exact Flat top-k vector-ID sets divided by min(k, vector count); this is not gold document or evidence Recall@k",
            "flat_boundary_tie_counts": {str(k): sum(row["neighbors"][str(k)]["boundary_tie"] for row in reference["per_query"]) for k in reference["ks"]},
            "flat_near_boundary_tie_counts": {str(k): sum(row["neighbors"][str(k)]["near_boundary_tie"] for row in reference["per_query"]) for k in reference["ks"]},
            "results": comparisons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    build = commands.add_parser("build")
    for flag in ("source-manifest", "vectors", "chunks", "index", "output"):
        build.add_argument("--" + flag, type=Path, required=True)
    query = commands.add_parser("query")
    for flag in ("index", "queries", "output"):
        query.add_argument("--" + flag, type=Path, required=True)
    query.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    query.add_argument("--latency-k", type=int, default=10)
    query.add_argument("--warmup", type=int, default=1)
    query.add_argument("--repeats", type=int, default=5)
    query.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.mode == "build":
        result = build_worker(args.source_manifest, args.vectors, args.chunks, args.index, output=args.output)
    else:
        result = query_worker(args.index, args.queries, output=args.output, ks=args.ks,
                              latency_k=args.latency_k, warmup=args.warmup, repeats=args.repeats, threads=args.threads)
    print(json.dumps({key: result[key] for key in ("measurement", "index_path", "vector_count", "dimension")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
