"""Fresh, isolated FAISS measurements with source-grounded recall diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys

import numpy as np

from report_chunker.retrieval import index_stats
from report_index.comparison import compare_query_results
from report_index.faiss_index import load_chunk_mapping, manifest_path, sha256_file
from report_search.benchmark import validate_benchmark
from report_search.experiment import _load_query_cache, aggregate, evaluate_question, json_hash, write_json


KINDS = ("flat", "ivf", "hnsw")
KS = (3, 5, 10, 20, 50)


def prepare_queries(benchmark_path, cache, embedding_manifest_path, chunks, output):
    """Bind cached query order/model to the unchanged corpus before test selection."""
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    metadata = json.loads((cache / "query_manifest.json").read_text(encoding="utf-8"))
    embedding = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    if embedding.get("status") != "complete" or embedding["source_sha256"] != sha256_file(chunks):
        raise ValueError("Corpus embeddings are incomplete or refer to different chunks")
    questions = benchmark["questions"]
    if (not isinstance(metadata.get("encoder"), dict) or any(
            metadata["encoder"].get(key) != value for key, value in embedding["encoder"].items()
            if key not in {"device", "batch_size", "inference_precision"})):
        raise ValueError("Cached query encoder differs from the corpus encoder")
    contract = {
        "query_texts_sha256": json_hash([(q["question_id"], q["query"]) for q in questions]),
        "benchmark_sha256": sha256_file(benchmark_path), "corpus_encoder": embedding["encoder"],
        "corpus_embedding_manifest_sha256": sha256_file(embedding_manifest_path),
        "corpus_vectors_file_sha256": sha256_file(embedding_manifest_path.parent / "vectors.npy"),
        "query_count": len(questions), "dimension": embedding["dimension"],
    }
    vectors = _load_query_cache(cache, contract)
    if vectors is None:
        raise ValueError("Query cache differs from the benchmark/model/corpus; run the retrieval experiment first")
    selected = [(i, q) for i, q in enumerate(questions) if q["split"] == "test"]
    if not selected:
        raise ValueError("Benchmark has no test questions")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "test_queries.npy"
    np.save(path, np.ascontiguousarray(vectors[[i for i, _ in selected]]), allow_pickle=False)
    provenance = {
        "benchmark_sha256": contract["benchmark_sha256"],
        "question_ids": [q["question_id"] for _, q in selected],
        "test_questions_sha256": json_hash([q for _, q in selected]),
        "test_queries_file_sha256": sha256_file(path),
        "source_query_manifest_sha256": sha256_file(cache / "query_manifest.json"),
        "source_query_vectors_sha256": metadata["vectors_file_sha256"],
        "encoder": metadata["encoder"], "embedding_contract": contract,
        "selection": "Existing test split only, in benchmark order; no re-embedding or parameter selection",
    }
    write_json(output / "test_query_manifest.json", provenance)
    return benchmark, [q for _, q in selected], path, provenance


def score_index_results(measured, questions, bm25_path, mapping, output):
    """Map native FAISS IDs to exact source payloads, outside the timed worker."""
    if len(measured["per_query"]) != len(questions):
        raise ValueError("Measured query count differs from benchmark selection")
    connection = sqlite3.connect(bm25_path.resolve().as_uri() + "?mode=ro", uri=True)
    rows = []
    try:
        # Verify the complete ordered source mapping, not just documents hit by queries.
        cursor = connection.execute("SELECT chunk_id,document_id FROM chunks ORDER BY id")
        for item in mapping:
            if cursor.fetchone() != (item["chunk_id"], item["document_id"]):
                raise ValueError("SQLite payload order differs from FAISS source mapping")
        if cursor.fetchone() is not None:
            raise ValueError("SQLite has extra source payloads")
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for number, (question, query) in enumerate(zip(questions, measured["per_query"])):
                if query["query_index"] != number:
                    raise ValueError("Measured query order is not canonical")
                payloads, metrics, judgments = {}, {}, {}
                for k in measured["ks"]:
                    neighbors = query["neighbors"][str(k)]
                    if (len(neighbors["vector_ids"]) != len(neighbors["scores"])
                            or len(neighbors["vector_ids"]) > k
                            or not np.isfinite(neighbors["scores"]).all()):
                        raise ValueError("Malformed measured neighbor IDs/scores")
                    results = []
                    seen = set()
                    for vector_id, score in zip(neighbors["vector_ids"], neighbors["scores"]):
                        if vector_id == -1:
                            continue
                        if not isinstance(vector_id, int) or not 0 <= vector_id < len(mapping) or vector_id in seen:
                            raise ValueError("Invalid or duplicate returned vector ID")
                        seen.add(vector_id)
                        if vector_id not in payloads:
                            item = mapping[vector_id]
                            found = connection.execute("SELECT payload,body FROM chunks WHERE chunk_id=?",
                                                       (item["chunk_id"],)).fetchone()
                            if found is None:
                                raise ValueError("Missing source payload")
                            chunk = json.loads(found[0])
                            if chunk["document_id"] != item["document_id"] or chunk["chunk_id"] != item["chunk_id"]:
                                raise ValueError("Source payload identity differs from mapping")
                            payloads[vector_id] = {**chunk, "text": found[1], "vector_id": vector_id}
                        results.append({**payloads[vector_id], "score": score, "rank": len(results) + 1})
                    judged = evaluate_question(question, results, cutoffs=(k,))
                    metrics.update(judged["metrics"])
                    judgments.update(judged["judgments"])
                row = {"question_id": question["question_id"], "category": question["category"], "split": "test",
                       "metrics": metrics, "judgments": judgments}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    finally:
        connection.close()
    return {"overall": aggregate(rows),
            "by_category": {category: aggregate(row for row in rows if row["category"] == category)
                            for category in ("factual", "comparative", "summary")},
            "per_query_artifact": str(output), "per_query_sha256": sha256_file(output),
            "search_policy": "Native single-query index.search independently at each k, ties sorted by vector ID; no metadata filters"}


def worker(arguments, environment):
    subprocess.run([sys.executable, "-m", "report_index.comparison", *map(str, arguments)],
                   env=environment, check=True)


def write_report(path, summary):
    model = summary["benchmark_provenance"]["encoder"]["model_id"]
    lines = ["# FAISS index comparison", "",
             f"Fresh builds over {summary['vector_count']:,} normalized {summary['dimension']}-dimensional {model} vectors "
             f"from {summary['document_count']} annual reports. Same {summary['query_count']} test questions and fixed configurations for every index.", "",
             "## Requested comparison", "",
             "**Recall@10 here is ANN neighbor recall:** overlap with exact Flat's top ten vector IDs, averaged over queries. "
             "It measures approximation loss; financial evidence recall is reported separately below.", "",
             "| Index | ANN Recall@10 | Query Latency (ms) | Build Time (s) | Memory Usage (MB) |",
             "|---|---:|---:|---:|---:|"]
    for kind in KINDS:
        row = summary["indexes"][kind]
        query, build = row["query"], row["build"]
        lines.append(f"| {kind.upper() if kind != 'flat' else 'Flat'} | {row['ann_recall']['10']:.2%} | "
                     f"{query['median_search_ms']:.3f} | {build['build_seconds']:.3f} | {query['memory_usage_mb']:.2f} |")
    lines += ["", "Query latency is the median of single-query CPU searches at k=10, four FAISS threads, two warmup passes "
              "and five timed repetitions per query. It excludes embedding, loading, validation, sorting and source-payload lookup. "
              "Build time is training plus vector insertion; end-to-end pipeline times are below. "
              "Memory is the measured process resident-memory increase after loading the index in an isolated process; "
              "MB = 1,000,000 bytes. It is neither serialized file size nor peak construction memory.", "",
              "## ANN neighbor recall across cutoffs", "",
              "| Index | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |", "|---|---:|---:|---:|---:|---:|"]
    for kind in KINDS:
        values = " | ".join(f"{summary['indexes'][kind]['ann_recall'][str(k)]:.2%}" for k in KS)
        lines.append(f"| {kind} | {values} |")
    lines += ["", "Each cutoff uses its own native search with the same k for Flat and the approximate index. "
              "This differs from the nested top-100 ranking prefixes in the [retriever recall sweep](recall_at_k_comparison.md). "
              "Equal-score boundary ties can reduce strict ID-overlap recall even when candidates are equally similar.", "",
              "## Source-evidence recall on the same queries", "",
              "| Index | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |", "|---|---:|---:|---:|---:|---:|"]
    for kind in KINDS:
        metrics = summary["indexes"][kind]["gold_evidence"]["overall"]["metrics"]
        lines.append(f"| {kind} | " + " | ".join(f"{metrics[f'evidence_recall_at_{k}']:.2%}" for k in KS) + " |")
    lines += ["", "These are macro fractions of required canonical source spans or complete table rows recovered. "
              "Flat's ANN recall is 100% by definition, while its evidence recall need not be. "
              "The agent-authored labels are non-exhaustive and await independent human review.", "",
              "## Timing and storage details", "",
              "| Index | Query p95 (ms) | Build pipeline (s) | Index file (MB) | Warmed RAM delta (MB) |",
              "|---|---:|---:|---:|---:|"]
    for kind in KINDS:
        row = summary["indexes"][kind]
        q, b = row["query"], row["build"]
        lines.append(f"| {kind} | {q['p95_search_ms']:.3f} | {b['pipeline_seconds']:.3f} | "
                     f"{q['index_storage_mb']:.2f} | {q['resident_warmed_delta_bytes']/1_000_000:.2f} |")
    construction = "; ".join(f"{kind}: {summary['indexes'][kind]['build']['configuration']['construction_threads']}" for kind in KINDS)
    ivf = summary["indexes"]["ivf"]["build"]["configuration"]
    hnsw = summary["indexes"]["hnsw"]["build"]["configuration"]
    lines += ["", "The build pipeline includes input checks, source mapping, construction and persistence. "
              f"Construction threads are {construction}. HNSW uses one for reproducible graph insertion. "
              "This is a fixed-configuration engineering comparison on one machine, not an equally parallelized build-speed contest or production load test.", "",
              f"Fixed settings: IVF nlist={ivf['nlist']}, nprobe={ivf['nprobe']}; HNSW M={hnsw['hnsw_m']}, "
              f"efConstruction={hnsw['ef_construction']}, efSearch={hnsw['ef_search']}; seed={hnsw['train_seed']}. "
              "These settings came from the earlier index run; no test-based tuning or model re-embedding was performed.", "",
              "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/compare_faiss_indexes.py", "```", "",
              "The script verifies the frozen benchmark/query/corpus contracts, rebuilds each index in a fresh process, "
              "then measures it in a separate fresh process. The parent resolves citations after timing. "
              "Artifacts remain in `data/experiments/faiss_comparison/`; the compact "
              "[result record](../data/evaluation/faiss_comparison_results.json) contains full settings, source/model/index hashes, "
              "memory counters, latency samples, per-query ANN recall and category evidence metrics.", "",
              "References: [FAISS index types and memory](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), "
              "[ties and batching](https://github.com/facebookresearch/faiss/wiki/FAQ), "
              "[thread configuration](https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/retrieval_questions.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/experiments/retrieval"))
    parser.add_argument("--source-indexes", type=Path, default=Path("data/indexes/bge-m3/table_preserving_s512_o50"))
    parser.add_argument("--chunks", type=Path, default=Path("data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/indexes/bm25_s512_o50/retrieval.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/faiss_comparison"))
    args = parser.parse_args(argv)
    if args.output.resolve() == args.source_indexes.resolve():
        parser.error("Use a new experiment directory; original indexes remain source configuration records")
    embedding_manifest = args.source_indexes / "embeddings" / "embedding_manifest.json"
    benchmark, questions, query_path, provenance = prepare_queries(
        args.benchmark, args.query_cache, embedding_manifest, args.chunks, args.output)
    validation = validate_benchmark(benchmark, args.processed)
    vector_path = embedding_manifest.parent / "vectors.npy"
    corpus_sha = sha256_file(args.chunks)
    if index_stats(args.bm25_index)["chunks_sha256"] != corpus_sha:
        raise ValueError("BM25 payload database belongs to another chunk corpus")
    sources = {kind: args.source_indexes / f"{kind}.manifest.json" for kind in KINDS}
    for kind, path in sources.items():
        item = json.loads(path.read_text(encoding="utf-8"))
        if (item["configuration"]["kind"] != kind or item["chunks_sha256"] != corpus_sha
                or item["vectors_file_sha256"] != provenance["embedding_contract"]["corpus_vectors_file_sha256"]):
            raise ValueError("Source index manifests must refer to these exact corpus vectors and chunks")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    builds, queries = {}, {}
    for kind in KINDS:
        folder = args.output / kind
        folder.mkdir(parents=True, exist_ok=True)
        index_path = folder / f"{kind}.faiss"
        print(f"Building fresh {kind} index in an isolated worker", flush=True)
        worker(["build", "--source-manifest", sources[kind], "--vectors", vector_path,
                "--chunks", args.chunks, "--index", index_path, "--output", folder / "build.json"], environment)
        print(f"Measuring {kind} in a separate process", flush=True)
        worker(["query", "--index", index_path, "--queries", query_path, "--output", folder / "query.json",
                "--ks", *KS, "--latency-k", 10, "--warmup", 2, "--repeats", 5, "--threads", 4], environment)
        builds[kind] = json.loads((folder / "build.json").read_text(encoding="utf-8"))
        queries[kind] = json.loads((folder / "query.json").read_text(encoding="utf-8"))
    comparison = compare_query_results(queries)
    mapping = load_chunk_mapping(args.output / "flat" / "flat.faiss")
    indexes = {}
    for kind in KINDS:
        evidence = score_index_results(queries[kind], questions, args.bm25_index, mapping,
                                       args.output / f"{kind}_evidence.jsonl")
        query_summary = {key: value for key, value in queries[kind].items() if key != "per_query"}
        query_summary["per_query_latency_samples_ms"] = [row["search_ms_samples"] for row in queries[kind]["per_query"]]
        indexes[kind] = {"build": builds[kind], "query": query_summary,
                         "ann_recall": comparison["results"][kind]["ann_recall"],
                         "gold_evidence": evidence,
                         "query_artifact": str(args.output / kind / "query.json"),
                         "query_artifact_sha256": sha256_file(args.output / kind / "query.json")}
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "query_count": len(questions), "vector_count": len(mapping), "document_count": len({m['document_id'] for m in mapping}),
        "dimension": provenance["embedding_contract"]["dimension"], "ks": list(KS),
        "benchmark_provenance": provenance, "benchmark_validation": validation,
        "source_configuration_manifests": {kind: {"path": str(path), "sha256": sha256_file(path)} for kind, path in sources.items()},
        "comparison": comparison, "indexes": indexes,
        "implementation_sha256": {path.relative_to(Path(__file__).parents[1]).as_posix(): sha256_file(path) for path in
                                   (Path(__file__), Path(__file__).with_name("comparison.py"),
                                    Path(__file__).with_name("faiss_index.py"),
                                    Path(__file__).parents[1] / "report_search" / "experiment.py",
                                    Path(__file__).parents[1] / "report_chunker" / "retrieval.py")},
    }
    write_json(args.output / "comparison.json", summary)
    write_report(args.output / "comparison.md", summary)
    print(f"Completed: {args.output / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
