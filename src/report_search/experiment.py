"""Evaluate FAISS, BM25, and dev-tuned weighted fusion on source-grounded questions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from report_chunker.retrieval import build_index, index_stats, judge_retrieval
from report_index.faiss_index import sha256_file
from report_search.benchmark import validate_benchmark
from report_search.retrieval import NORMALIZATION, RetrievalEngine, weighted_fusion


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("document_recall", "all_documents_hit", "evidence_recall", "all_evidence_hit", "context_tokens")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def evaluate_question(question, results, cutoffs=(5, 10)):
    """Both recall denominators are independent of chunking and duplicate hits."""
    gold_documents = set(question["ground_truth_doc_ids"])
    if not gold_documents:
        raise ValueError("Questions need gold documents")
    metrics, judgments = {}, {}
    for k in cutoffs:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("cutoffs must be positive integers")
        retrieved_documents = {row["document_id"] for row in results[:k]}
        found = len(gold_documents & retrieved_documents)
        evidence = judge_retrieval(question, results, top_k=k)
        metrics.update({
            f"document_recall_at_{k}": found / len(gold_documents),
            f"all_documents_hit_at_{k}": int(found == len(gold_documents)),
            f"evidence_recall_at_{k}": evidence[f"recall_at_{k}"],
            f"all_evidence_hit_at_{k}": evidence[f"all_evidence_hit_at_{k}"],
            f"context_tokens_at_{k}": sum(row.get("token_count", 0) for row in results[:k]),
        })
        judgments[str(k)] = evidence
    return {"question_id": question["question_id"], "category": question["category"],
            "split": question["split"], "metrics": metrics, "judgments": judgments}


def aggregate(rows):
    rows = list(rows)
    if not rows:
        return {"question_count": 0, "metrics": {}}
    keys = rows[0]["metrics"]
    return {"question_count": len(rows), "metrics": {
        key: sum(row["metrics"][key] for row in rows) / len(rows) for key in keys}}


def select_alpha(dev_rows):
    """Select by dev macro evidence Recall@5; ties favor 0.5, then smaller alpha."""
    if not dev_rows or any(not rows for rows in dev_rows.values()):
        raise ValueError("Each alpha requires nonempty development judgments")
    if any(row["split"] != "dev" for rows in dev_rows.values() for row in rows):
        raise ValueError("Hybrid tuning may consume development judgments only")
    ids = [tuple(row["question_id"] for row in rows) for rows in dev_rows.values()]
    if any(value != ids[0] for value in ids):
        raise ValueError("All alphas must use the same ordered development questions")
    scores = {float(alpha): aggregate(rows)["metrics"]["evidence_recall_at_5"]
              for alpha, rows in dev_rows.items()}
    best = max(scores.values())
    # Roundoff from summing equal fractions must not change a nominal tie.
    tied = [alpha for alpha, score in scores.items() if math.isclose(score, best, abs_tol=1e-12, rel_tol=0)]
    return min(tied, key=lambda alpha: (abs(alpha - 0.5), alpha))


def paired_interval(first, second, key="evidence_recall_at_5", seed=20261008):
    """Paired query bootstrap; exploratory uncertainty, not a population guarantee."""
    if [row["question_id"] for row in first] != [row["question_id"] for row in second]:
        raise ValueError("Paired comparisons need identical question order")
    differences = np.asarray([a["metrics"][key] - b["metrics"][key]
                              for a, b in zip(first, second)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_delta_pp": float(differences.mean() * 100),
            "ci95_pp": [float(low * 100), float(high * 100)], "replicates": 10000, "seed": seed,
            "sampling_unit": "paired question; ignores within-company/template dependence"}


def run_evaluation(engine, questions, query_vectors, output, *, alphas=DEFAULT_ALPHAS,
                   candidate_k=100, cutoffs=(5, 10), provenance=None):
    """Retrieve dev first, persist chosen weight, then evaluate the fixed test policy.

    Only question text/query vectors enter the engine. Labels enter this evaluator
    after retrieval. Results are streamed; complete corpus payloads stay on disk.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if 5 not in cutoffs or candidate_k < max(cutoffs):
        raise ValueError("Recall@5 is required and candidate_k must cover every cutoff")
    alphas = tuple(sorted(set(float(alpha) for alpha in alphas)))
    if not alphas or any(not math.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError("alphas must be finite values in [0,1]")
    if len(query_vectors) != len(questions):
        raise ValueError("One query vector is required per question")
    by_split = {split: [(i, q) for i, q in enumerate(questions) if q["split"] == split]
                for split in ("dev", "test")}
    if not all(by_split.values()) or sum(map(len, by_split.values())) != len(questions):
        raise ValueError("Benchmark must contain nonempty dev and test splits only")
    dev_tuning = {alpha: [] for alpha in alphas}
    reports = {split: {name: [] for name in ("vector", "bm25", "hybrid")} for split in by_split}
    selected = None
    tuning = None
    with (output / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for split in ("dev", "test"):
            for number, (i, question) in enumerate(by_split[split], 1):
                start = time.perf_counter()
                dense = engine.vector(query_vectors[i], top_k=candidate_k)
                vector_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                lexical = engine.bm25(question["query"], top_k=candidate_k)
                lexical_ms = (time.perf_counter() - start) * 1000
                outputs = {"vector": dense[:max(cutoffs)], "bm25": lexical[:max(cutoffs)]}
                if split == "dev":
                    for alpha in alphas:
                        fusion = weighted_fusion(dense, lexical, alpha=alpha, top_k=max(cutoffs))
                        measured = evaluate_question(question, fusion, cutoffs)
                        dev_tuning[alpha].append(measured)
                        handle.write(json.dumps({"approach": "hybrid_dev_candidate", "alpha": alpha,
                                                 **measured, "results": fusion}, ensure_ascii=False) + "\n")
                else:
                    outputs["hybrid"] = weighted_fusion(dense, lexical, alpha=selected, top_k=max(cutoffs))
                for name, results in outputs.items():
                    measured = evaluate_question(question, results, cutoffs)
                    reports[split][name].append(measured)
                    handle.write(json.dumps({"approach": name, "alpha": selected if name == "hybrid" else None,
                                             "query": question["query"], **measured, "results": results,
                                             "candidate_search_ms": {"vector": vector_ms, "bm25": lexical_ms}},
                                            ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{split}: {number}/{len(by_split[split])} {question['question_id']}", flush=True)
            if split == "dev":
                selected = select_alpha(dev_tuning)
                reports["dev"]["hybrid"] = dev_tuning[selected]
                tuning = {
                    "selected_alpha": selected, "candidate_k": candidate_k,
                    "normalization": NORMALIZATION, "selection_split": "dev",
                    "objective": "macro evidence_recall_at_5",
                    "tie_breaker": "closest to 0.5, then smaller alpha (absolute tolerance 1e-12)",
                    "development_questions_sha256": json_hash([q for _, q in by_split["dev"]]),
                    "frozen_before_test_evaluation": True,
                    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "grid": [{"alpha": alpha, **aggregate(rows)} for alpha, rows in dev_tuning.items()],
                }
                write_json(output / "tuning.json", tuning)
                print(f"Frozen development-selected vector weight: {selected}", flush=True)
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_provenance": provenance or {}, "corpus": engine.stats,
        "scope": "all indexed reports, no gold company/year filters or query decomposition",
        "candidate_k_per_channel": candidate_k, "cutoffs": list(cutoffs), "tuning": tuning,
        "metric_definitions": {
            "document_recall": "fraction of distinct gold report IDs among the first k chunks, macro per question",
            "evidence_recall": "fraction of canonical required source spans/rows fully covered by the first k chunks, macro per question",
            "all_evidence_hit": "fraction of questions with every required evidence unit recovered",
            "context_tokens": "sum of existing cl100k_base token_count for the first k chunks, including overlap",
        },
        "splits": {split: {name: {
            "overall": aggregate(rows),
            "by_category": {category: aggregate(row for row in rows if row["category"] == category)
                            for category in ("factual", "comparative", "summary")},
        } for name, rows in methods.items()} for split, methods in reports.items()},
        "test_paired_hybrid_delta": {name: paired_interval(reports["test"]["hybrid"], reports["test"][name])
                                     for name in ("vector", "bm25")},
        "per_query_artifact": "per_query.jsonl", "per_query_sha256": sha256_file(output / "per_query.jsonl"),
        "limitations": [
            "Agent-authored, source-checked labels await independent human semantic review.",
            "Small company-disjoint question split; companies share templates and remain in the searchable corpus.",
            "Gold passages are non-exhaustive; valid unjudged passages receive no evidence credit.",
            "Summary coverage measures selected claims, not answer completeness or generated-answer quality.",
            "A fixed chunk count does not equal a fixed context-token budget; context length is reported.",
            "No test-label tuning; model/chunking were chosen in earlier exploratory work, not on a pristine unseen corpus.",
        ],
    }
    write_json(output / "comparison.json", summary)
    write_comparison(output / "comparison.md", summary)
    return summary


def write_comparison(path, summary):
    alpha = summary["tuning"]["selected_alpha"]
    model = summary["benchmark_provenance"].get("query_manifest", {}).get("encoder", {}).get("model_id", "configured dense encoder")
    lines = ["# Retrieval comparison", "",
             f"All {summary['corpus']['document_count']} reports / {summary['corpus']['chunk_count']:,} chunks. "
             f"{model} + FAISS Flat, SQLite FTS5 BM25, and weighted score fusion (vector weight {alpha:g}).", "",
             "The hybrid weight was selected using development evidence Recall@5 and frozen before test evaluation.", "",
             "## Test results", "",
             "| Category | Approach | Questions | Document Recall@5 | Evidence Recall@5 | All evidence@5 | Evidence Recall@10 | Mean top-5 tokens |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for category in ("overall", "factual", "comparative", "summary"):
        for name in ("vector", "bm25", "hybrid"):
            report = summary["splits"]["test"][name]
            item = report["overall"] if category == "overall" else report["by_category"][category]
            if not item["question_count"]:
                continue
            m = item["metrics"]
            recall10 = f"{m['evidence_recall_at_10']:.2%}" if "evidence_recall_at_10" in m else "n/a"
            lines.append(f"| {category} | {name} | {item['question_count']} | {m['document_recall_at_5']:.2%} | "
                         f"{m['evidence_recall_at_5']:.2%} | {m['all_evidence_hit_at_5']:.2%} | {recall10} | {m['context_tokens_at_5']:.0f} |")
    lines += ["", "Document recall counts distinct gold reports found within the top five **chunks**. "
              "Evidence recall requires the annotated source spans or complete table rows. Both are macro averages.", "",
              "## Development-only weight selection", "",
              "| Vector weight | BM25 weight | Dev evidence Recall@5 |", "|---:|---:|---:|"]
    for row in summary["tuning"]["grid"]:
        lines.append(f"| {row['alpha']:g} | {1-row['alpha']:g} | {row['metrics']['evidence_recall_at_5']:.2%} |")
    lines += ["", "## Limitations", ""] + [f"- {note}" for note in summary["limitations"]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _embedding_contract(args, questions):
    corpus = json.loads(args.embedding_manifest.read_text(encoding="utf-8"))
    if corpus.get("status") != "complete" or corpus["source_sha256"] != sha256_file(args.chunks):
        raise ValueError("Corpus embedding manifest does not match the completed chunk corpus")
    return {"query_texts_sha256": json_hash([(q["question_id"], q["query"]) for q in questions]),
            "benchmark_sha256": sha256_file(args.benchmark), "corpus_encoder": corpus["encoder"],
            "corpus_embedding_manifest_sha256": sha256_file(args.embedding_manifest),
            "corpus_vectors_file_sha256": sha256_file(args.embedding_manifest.parent / "vectors.npy"),
            "query_count": len(questions), "dimension": corpus["dimension"]}


def _load_query_cache(output, contract):
    metadata_path, vectors_path = output / "query_manifest.json", output / "query_vectors.npy"
    if not metadata_path.exists() or not vectors_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract") != contract:
        return None
    if metadata.get("vectors_file_sha256") != sha256_file(vectors_path):
        raise ValueError("Query vector cache checksum changed")
    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.shape != (contract["query_count"], contract["dimension"]) or vectors.dtype != np.float32:
        raise ValueError("Query vector cache shape/dtype mismatch")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("Query vectors must be finite unit vectors")
    return vectors


def encode_queries(args, questions, contract):
    """Run in an isolated process so model memory is released before FAISS loads."""
    from report_index.embedding import DenseEncoder
    if args.device.startswith("cuda:"):
        import torch
        torch.cuda.set_per_process_memory_fraction(0.6, device=args.device)
    original = contract["corpus_encoder"]
    encoder = DenseEncoder(model_id=original["model_id"], revision=original["model_revision"],
                           device=args.device, batch_size=8)
    differences = {key for key, value in original.items()
                   if key not in {"device", "batch_size", "inference_precision"} and encoder.identity.get(key) != value}
    if differences:
        raise ValueError(f"Query encoder differs from corpus encoder: {sorted(differences)}")
    start = time.perf_counter()
    vectors = encoder.encode_queries([q["query"] for q in questions])
    vector_path = args.output / "query_vectors.npy"
    np.save(vector_path, vectors, allow_pickle=False)
    write_json(args.output / "query_manifest.json", {
        "contract": contract, "encoder": encoder.identity,
        "vectors_file_sha256": sha256_file(vector_path),
        "encoding_seconds": time.perf_counter() - start,
        "inputs": "question text only; no gold documents, evidence, answers, or filters",
    })
    _load_query_cache(args.output, contract)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/retrieval_questions.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunks", type=Path, default=Path("data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl"))
    parser.add_argument("--vector-index", type=Path, default=Path("data/indexes/bge-m3/table_preserving_s512_o50/flat.faiss"))
    parser.add_argument("--embedding-manifest", type=Path, default=Path("data/indexes/bge-m3/table_preserving_s512_o50/embeddings/embedding_manifest.json"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/indexes/bm25_s512_o50/retrieval.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/retrieval"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--encode-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    questions = benchmark["questions"]
    contract = _embedding_contract(args, questions)
    if args.encode_only:
        encode_queries(args, questions, contract)
        return 0
    validation = validate_benchmark(benchmark, args.processed)
    write_json(args.output / "benchmark_validation.json", validation)
    print(f"Validated {len(questions)} questions against original source artifacts", flush=True)
    vectors = _load_query_cache(args.output, contract)
    if vectors is None:
        command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "run_retrieval_experiment.py"),
                   "--encode-only", "--benchmark", str(args.benchmark.resolve()),
                   "--chunks", str(args.chunks.resolve()), "--embedding-manifest", str(args.embedding_manifest.resolve()),
                   "--vector-index", str(args.vector_index.resolve()),
                   "--output", str(args.output.resolve()), "--device", args.device]
        subprocess.run(command, check=True)
        vectors = _load_query_cache(args.output, contract)
    if not args.bm25_index.exists() or index_stats(args.bm25_index)["chunks_sha256"] != sha256_file(args.chunks):
        print("Building BM25 index from the same chunk corpus", flush=True)
        build_index(args.chunks, args.bm25_index)
    provenance = {"benchmark_path": str(args.benchmark), "benchmark_sha256": sha256_file(args.benchmark),
                  "validation": validation, "query_manifest": json.loads((args.output / "query_manifest.json").read_text(encoding="utf-8")),
                  "implementation_sha256": {path.relative_to(Path(__file__).parents[1]).as_posix(): sha256_file(path) for path in
                      (Path(__file__), Path(__file__).with_name("retrieval.py"), Path(__file__).with_name("benchmark.py"),
                       Path(__file__).parents[1] / "report_chunker" / "retrieval.py")}}
    with RetrievalEngine(args.chunks, args.vector_index, args.bm25_index, threads=args.threads) as engine:
        if engine.stats["dimension"] != contract["dimension"]:
            raise ValueError("Query and index dimensions differ")
        if engine.vector_manifest.get("vectors_file_sha256") != contract["corpus_vectors_file_sha256"]:
            raise ValueError("FAISS index was not built from this encoder's corpus vectors")
        if engine.vector_manifest.get("configuration", {}).get("kind") != "flat":
            raise ValueError("This experiment requires an exact Flat index to isolate fusion from ANN approximation")
        result = run_evaluation(engine, questions, vectors, args.output, alphas=args.alphas,
                                candidate_k=args.candidate_k, provenance=provenance)
    print(json.dumps({"selected_alpha": result["tuning"]["selected_alpha"], "test": result["splits"]["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
