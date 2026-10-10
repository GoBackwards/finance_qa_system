"""Measure development-only candidate coverage without models or test evaluation.

This is a pre-reranking ablation, not a generated-answer experiment. All four
approaches have a maximum final pool of 20. Lexical recovery is evaluated for
every dev query here. It was not adopted: the bounded-recovery controller that
offered it as an opt-in fallback has since been removed (docs/failure_handling.md).
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from report_chunker.retrieval import _search, index_stats
from report_index.faiss_index import sha256_file
from report_search.experiment import aggregate, evaluate_question, paired_interval, write_json
from report_search.query_policy import plan_query


METHODS = ("original_vector20", "scoped_vector20", "scoped_vector_bm25_original20",
           "scoped_vector_bm25_hints20")
SOURCE_FILES = (
    "scripts/compare_query_expansion.py", "src/report_search/query_policy.py",
    "src/report_search/experiment.py", "src/report_chunker/retrieval.py",
)


def filter_scope(candidates, plan):
    """Apply only recognized report subjects and explicitly requested filings.

    A bare metric period is not a report-year filter: a later report may carry
    historical comparative figures. Textual metric support is checked later.
    """
    constraints = plan["constraints"]
    tickers = set(constraints["requested_tickers"])
    reports = {(r["ticker"], r["fiscal_year"]) for r in constraints.get("requested_reports", [])}
    years = set(constraints["requested_report_years"])
    selected = []
    for candidate in candidates:
        if tickers and candidate.get("ticker") not in tickers:
            continue
        if reports and (candidate.get("ticker"), candidate.get("fiscal_year")) not in reports:
            continue
        if years and not reports and candidate.get("fiscal_year") not in years:
            continue
        selected.append(candidate)
    return selected


def reciprocal_rank_pool(channels, *, top_k=20):
    """Bound candidate union by rank, without mixing cosine and BM25 scores.

    Channels are named ordered lists; every channel contributes once per ID.
    Conflicting source payloads raise instead of silently overwriting text.
    """
    ranked, source_ranks = {}, {}
    for name, candidates in channels:
        seen = set()
        for rank, candidate in enumerate(candidates[:20], 1):
            identifier = candidate["chunk_id"]
            if identifier in seen:
                continue
            seen.add(identifier)
            if identifier in ranked:
                previous = ranked[identifier]
                for field in ("document_id", "text", "evidence", "rows", "source_file", "ticker", "fiscal_year"):
                    if previous.get(field) != candidate.get(field):
                        raise ValueError("Conflicting retrieval source payload")
            else:
                ranked[identifier] = copy.deepcopy(candidate)
                source_ranks[identifier] = {}
            source_ranks[identifier][name] = rank
    scores = {identifier: sum(1.0 / (60 + rank) for rank in ranks.values())
              for identifier, ranks in source_ranks.items()}
    ordered = sorted(ranked, key=lambda identifier: (-scores[identifier], identifier))[:top_k]
    result = []
    for rank, identifier in enumerate(ordered, 1):
        candidate = ranked[identifier]
        candidate.pop("vector_score", None)
        candidate.pop("vector_rank", None)
        candidate.update(score=scores[identifier], rank=rank, retrieval_approach="expanded_rank_fusion",
                         retrieval_channels=source_ranks[identifier])
        result.append(candidate)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_inputs(args):
    manifest_path = args.candidate_run / "candidate_manifest.json"
    candidates_path = args.candidate_run / "candidates.jsonl"
    manifest = read_json(manifest_path)
    if sha256_file(args.benchmark) != manifest["benchmark_sha256"]:
        raise ValueError("Frozen benchmark differs from the source vector ranking")
    if sha256_file(candidates_path) != manifest["candidate_file_sha256"]:
        raise ValueError("Frozen candidate artifact checksum differs")
    full = read_json(args.benchmark)["questions"]
    dev = read_json(args.dev_benchmark)["questions"]
    expected = [q for q in full if q["split"] == "dev"]
    if dev != expected or len(dev) != 50 or any(q["split"] != "dev" for q in dev):
        raise ValueError("Only the exact 50 frozen development questions are permitted")
    dev_by_id = {q["question_id"]: q for q in dev}
    if len(dev_by_id) != 50:
        raise ValueError("Duplicate development question ID")
    vectors, seen = {}, []
    with candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row["question_id"]
            seen.append(identifier)
            # Test rankings are checksummed as part of the frozen artifact, but
            # never enter retrieval, judgments, policy selection or output.
            if identifier not in dev_by_id:
                continue
            pool = row["candidates"]
            if (row["query"] != dev_by_id[identifier]["query"] or len(pool) != 20
                    or len({c["chunk_id"] for c in pool}) != 20
                    or [c["rank"] for c in pool] != list(range(1, 21))):
                raise ValueError("Frozen development query/pool contract differs")
            vectors[identifier] = pool
    if seen != manifest["question_ids"] or len(set(seen)) != len(seen) or set(vectors) != set(dev_by_id):
        raise ValueError("Candidate question identities are incomplete or reordered")
    stats = index_stats(args.bm25_index)
    chunks_sha = sha256_file(args.chunks)
    if chunks_sha != stats["chunks_sha256"] or chunks_sha != manifest["corpus"]["chunks_sha256"]:
        raise ValueError("BM25 and frozen vector sources do not share the same checked chunks")
    for field in ("chunk_count", "document_count"):
        if stats[field] != manifest["corpus"][field]:
            raise ValueError("BM25 and frozen vector source counts differ")
    source_documents = set(manifest["corpus"]["document_ids"])
    records = read_json(args.inventory)
    inventory = [r for r in records if r.get("status") == "validated"
                 and f"{Path(r['relative_path']).stem}-{r['sha256'][:12]}" in source_documents]
    if len(inventory) != len(source_documents):
        raise ValueError("Inventory does not cover the actual indexed reports")
    provenance = {
        "benchmark_sha256": sha256_file(args.benchmark),
        "development_benchmark_sha256": sha256_file(args.dev_benchmark),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_file_sha256": sha256_file(candidates_path),
        "chunks_sha256": chunks_sha, "bm25_index_sha256": sha256_file(args.bm25_index),
        "inventory_sha256": sha256_file(args.inventory), "corpus": manifest["corpus"],
        "source_code_sha256": {p: sha256_file(Path(p)) for p in SOURCE_FILES},
    }
    return dev, vectors, inventory, provenance


def paired(first, second):
    result = paired_interval(first, second, key="evidence_recall_at_20")
    deltas = [a["metrics"]["evidence_recall_at_20"] - b["metrics"]["evidence_recall_at_20"]
              for a, b in zip(first, second)]
    result.update(improved=sum(d > 1e-12 for d in deltas),
                  unchanged=sum(abs(d) <= 1e-12 for d in deltas),
                  worsened=sum(d < -1e-12 for d in deltas))
    return result


def run(args):
    if args.output.resolve() == args.candidate_run.resolve():
        raise ValueError("Output must differ from the frozen candidate directory")
    protected = {p.resolve() for p in (args.benchmark, args.dev_benchmark, args.chunks,
        args.bm25_index, args.inventory, args.candidate_run / "candidates.jsonl",
        args.candidate_run / "candidate_manifest.json")}
    if args.publish and args.publish.resolve() in protected:
        raise ValueError("Published results must not overwrite experiment inputs")
    questions, vectors, inventory, provenance = checked_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output / "per_query.jsonl"
    temporary_path = detailed_path.with_name(detailed_path.name + ".writing")
    reports = {name: [] for name in METHODS}
    assembly_times = {name: [] for name in METHODS}
    hinted_count = 0
    with closing(sqlite3.connect(args.bm25_index.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA cache_size=-8192")
        actual_docs = {row[0] for row in connection.execute("SELECT DISTINCT document_id FROM chunks")}
        if actual_docs != set(provenance["corpus"]["document_ids"]):
            raise ValueError("SQLite document inventory differs from the frozen corpus")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for number, question in enumerate(questions, 1):
                query, identifier = question["query"], question["question_id"]
                original = vectors[identifier]
                started = time.perf_counter()
                plan = plan_query(query, inventory)
                scoped = filter_scope(original, plan)
                scoped_ms = (time.perf_counter() - started) * 1000
                if plan["coverage"]["status"] == "unavailable":
                    raise ValueError("An existing development question unexpectedly requests an absent filing")
                lexical_times, lexical_pools = [], []
                for lexical_query in plan["lexical_queries"][:3]:
                    started = time.perf_counter()
                    lexical = filter_scope(_search(connection, lexical_query, 20, None, None), plan)
                    lexical_times.append((time.perf_counter() - started) * 1000)
                    lexical_pools.append(lexical)
                hinted_count += bool(plan["lexical_hints"])
                started = time.perf_counter()
                original_fusion = reciprocal_rank_pool([("vector", scoped), ("lexical_0", lexical_pools[0])])
                original_fusion_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                hinted_fusion = reciprocal_rank_pool([("vector", scoped),
                    *((f"lexical_{i}", pool) for i, pool in enumerate(lexical_pools))])
                hinted_fusion_ms = (time.perf_counter() - started) * 1000
                pools = dict(zip(METHODS, (original, scoped, original_fusion, hinted_fusion)))
                # Components are shared in this one-pass ablation. None includes
                # vector inference/search, reranking, generation or source checks.
                durations = dict(zip(METHODS, (0.0, scoped_ms,
                    scoped_ms + lexical_times[0] + original_fusion_ms,
                    scoped_ms + sum(lexical_times) + hinted_fusion_ms)))
                row = {"question_id": identifier, "query": query, "split": "dev", "query_plan": plan,
                       "candidate_assembly_ms": durations, "bm25_queries_ms": lexical_times, "approaches": {}}
                for name, pool in pools.items():
                    if len(pool) > 20 or len({c["chunk_id"] for c in pool}) != len(pool):
                        raise ValueError("Candidate budget or unique-ID invariant failed")
                    measured = evaluate_question(question, pool, cutoffs=(5, 20))
                    reports[name].append(measured)
                    assembly_times[name].append(durations[name])
                    row["approaches"][name] = {**measured, "chunk_ids": [c["chunk_id"] for c in pool],
                                              "candidate_count": len(pool)}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if number % 10 == 0:
                    print(f"Development candidate pools: {number}/{len(questions)}", flush=True)
    for path, digest in provenance["source_code_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Implementation changed during evaluation: {path}")
    if sha256_file(args.bm25_index) != provenance["bm25_index_sha256"]:
        raise ValueError("Read-only BM25 artifact changed during evaluation")
    os.replace(temporary_path, detailed_path)
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "development-only candidate-pool ablation before reranking",
        "split": "dev", "question_count": len(questions), "questions_with_lexical_hints": hinted_count,
        "test_questions_evaluated": 0, "labels_sent_to_retrieval": False,
        "candidate_limit": 20, "per_lexical_query_limit": 20, "maximum_lexical_queries": 3,
        "scoring_models_run": [], "generation_calls": 0, "provenance": provenance,
        "approaches": {name: {"overall": aggregate(rows),
            "by_category": {category: aggregate(r for r in rows if r["category"] == category)
                            for category in ("factual", "comparative", "summary")}}
            for name, rows in reports.items()},
        "paired_evidence_recall_at_20": {
            "scope_minus_original_vector": paired(reports[METHODS[1]], reports[METHODS[0]]),
            "original_bm25_fusion_minus_scoped_vector": paired(reports[METHODS[2]], reports[METHODS[1]]),
            "hints_minus_original_bm25_fusion": paired(reports[METHODS[3]], reports[METHODS[2]]),
            "hints_minus_original_vector": paired(reports[METHODS[3]], reports[METHODS[0]]),
        },
        "candidate_assembly_timing_ms": {name: {"median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "mean": float(np.mean(values))}
            for name, values in assembly_times.items() if name != METHODS[0]},
        "timing_scope": "One sequential development pass with one read-only SQLite session; shared component times. Includes query planning, scope filtering, BM25 search and rank fusion as applicable. Excludes vector embedding/search, reranking, model loading, source binding, generation, judgments and file writes. Not independent trials or end-to-end latency.",
        "metric_definitions": {
            "evidence_recall_at_20": "Macro fraction of canonical annotated source units fully covered by the at-most-20 candidate pool, before reranking.",
            "all_evidence_hit_at_20": "Fraction of questions whose complete annotated source units are covered by the candidate pool.",
            "evidence_recall_at_5": "Prefix coverage in the candidate ranking before reranking; not the final cross-encoder Top5 or answer accuracy.",
        },
        "limitations": [
            "Development-only exploratory ablation; no frozen test question is evaluated or used to choose hints.",
            "Every development query is evaluated under recovery variants, rather than only requests rejected by a confidence gate.",
            "Hints and query-policy code are fixed before this run; no grid search or tuning is performed.",
            "Filtering occurs after global Top20 retrieval, so scoped pools may contain fewer than 20 chunks.",
            "Additional overlapping lexical queries can overweight a channel in RRF; gains are not presumed.",
            "Non-exhaustive, agent-authored development labels await independent human review.",
            "No reranking or generation is performed; candidate coverage does not establish final answer quality or hallucination reduction.",
            "Bootstrap intervals sample paired questions and ignore company/template dependence.",
            "Expansion was not adopted; any future use would need its own threshold calibration.",
        ],
        "per_query_artifact": detailed_path.as_posix(), "per_query_sha256": sha256_file(detailed_path),
    }
    write_json(args.output / "comparison.json", summary)
    if args.publish:
        write_json(args.publish, summary)
    for name in METHODS:
        print(name, summary["approaches"][name]["overall"]["metrics"]["evidence_recall_at_20"], flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/retrieval_questions.json"))
    parser.add_argument("--dev-benchmark", type=Path, default=Path("data/evaluation/retrieval_dev_questions.json"))
    parser.add_argument("--candidate-run", type=Path, default=Path("data/experiments/reranker"))
    parser.add_argument("--chunks", type=Path, default=Path("data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/indexes/bm25_s512_o50/retrieval.sqlite"))
    parser.add_argument("--inventory", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/query_expansion"))
    parser.add_argument("--publish", type=Path)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
