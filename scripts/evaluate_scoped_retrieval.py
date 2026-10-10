"""Compare global and scoped retrieval on the frozen 100-question benchmark.

Both paths share one query embedding and the pinned reranker in one process:
  global: vector Top20 over all reports -> reranked Top5 (the original demo path)
  scoped: question-only company/period scope -> per-company exact vector pools
          -> reranked -> balanced, duplicate-aware Top5 (query_scope + scoped_retrieval)
The scoping rules were designed from the live bad-case pool, not from these
labels, and no setting is tuned here; dev and test are both reported. Metrics
are source-selection metrics, not generated-answer accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from report_chunker.retrieval import judge_retrieval
from report_search.experiment import aggregate, evaluate_question, paired_interval
from report_search.scoped_retrieval import SELECTION_VERSION, merge_pools, plan_pools, select_candidates


def _metrics(question, top5, pool):
    measured = evaluate_question(question, top5, (5,))
    metrics = {key: measured["metrics"][key] for key in
               ("evidence_recall_at_5", "all_evidence_hit_at_5", "document_recall_at_5", "context_tokens_at_5")}
    metrics["pool_evidence_recall"] = judge_retrieval(question, pool, top_k=len(pool))[f"recall_at_{len(pool)}"]
    metrics["pool_size"] = len(pool)
    return metrics


def _summary(rows, approach):
    view = [{"question_id": row["question_id"], "metrics": row[approach]} for row in rows]
    return aggregate(view)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/evaluation/retrieval_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/scoped_retrieval_results.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    from report_search.demo_service import _default_factory
    from report_search.query_scope import SCOPE_VERSION

    benchmark_bytes = args.benchmark.read_bytes()
    questions = json.loads(benchmark_bytes)["questions"]
    started = time.perf_counter()
    components = _default_factory(ROOT, args.device)
    setup_seconds = time.perf_counter() - started
    rows = []
    try:
        for question in questions:
            query = question["query"]
            vector = components.encoder.encode_queries([query])[0]
            global_pool = components.retrieval_engine.vector(vector, top_k=20)
            global_top5 = components.reranker.rank_candidates(query, global_pool, top_k=5)
            scope = components.scope_resolver.resolve(query).to_dict()
            pools = plan_pools(scope, components.documents)
            if pools:
                scoped_pool = merge_pools(components.retrieval_engine.vector(
                    vector, top_k=pool["top_k"], document_ids=pool["document_ids"]) for pool in pools)
            else:
                scoped_pool = global_pool
            reranked = components.reranker.rank_candidates(query, scoped_pool, top_k=len(scoped_pool))
            scoped_top5, trace = select_candidates(reranked, scope, top_k=5)
            rows.append({"question_id": question["question_id"], "split": question["split"],
                         "category": question["category"],
                         "scope": {key: scope[key] for key in ("tickers", "years", "period_mode", "report_years")},
                         # Test questions carry no tickers field; derive companies from gold report IDs.
                         "gold_tickers": sorted({doc.split("_FY")[0] for doc in question["ground_truth_doc_ids"]}),
                         "gold_fiscal_years": sorted({int(doc.split("_FY")[1][:4]) for doc in question["ground_truth_doc_ids"]}),
                         "global": _metrics(question, global_top5, global_pool),
                         "scoped": _metrics(question, scoped_top5, scoped_pool),
                         "trace": {key: trace[key] for key in ("quota", "reserved_latest", "suppressed_duplicates")},
                         "global_top5": [row["chunk_id"] for row in global_top5],
                         "scoped_top5": [row["chunk_id"] for row in scoped_top5]})
            print(f"{question['question_id']}: ER@5 {rows[-1]['global']['evidence_recall_at_5']:.2f} -> "
                  f"{rows[-1]['scoped']['evidence_recall_at_5']:.2f}", flush=True)
    finally:
        components.retrieval_engine.close()

    summary = {}
    for split in ("dev", "test", "all"):
        subset = [row for row in rows if split == "all" or row["split"] == split]
        entry = {"question_count": len(subset),
                 "global": _summary(subset, "global")["metrics"], "scoped": _summary(subset, "scoped")["metrics"],
                 "scope_resolved_companies": sum(bool(row["scope"]["tickers"]) for row in subset),
                 "scope_matches_gold_companies": sum(sorted(row["scope"]["tickers"]) == sorted(row["gold_tickers"] or [])
                                                     for row in subset)}
        for key in ("evidence_recall_at_5", "all_evidence_hit_at_5"):
            first = [{"question_id": r["question_id"], "metrics": r["scoped"]} for r in subset]
            second = [{"question_id": r["question_id"], "metrics": r["global"]} for r in subset]
            entry[f"paired_{key}_scoped_minus_global"] = paired_interval(first, second, key)
            deltas = [r["scoped"][key] - r["global"][key] for r in subset]
            entry[f"{key}_improved_unchanged_worsened"] = [sum(d > 0 for d in deltas), sum(d == 0 for d in deltas),
                                                          sum(d < 0 for d in deltas)]
        entry["by_category"] = {
            category: {"question_count": len(group), "global": _summary(group, "global")["metrics"],
                       "scoped": _summary(group, "scoped")["metrics"]}
            for category in ("factual", "comparative", "summary")
            if (group := [row for row in subset if row["category"] == category])}
        summary[split] = entry
    result = {"schema_version": 1, "experiment": "global vs scoped retrieval on the frozen benchmark",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "benchmark": {"path": "data/evaluation/retrieval_questions.json",
                            "sha256": hashlib.sha256(benchmark_bytes).hexdigest(), "question_count": len(questions)},
              "configuration": {"device": args.device, "scope_version": SCOPE_VERSION,
                                "selection_version": SELECTION_VERSION, "global_pool": 20, "scoped_pool": "20 split across resolved companies (>=6 each, <=30 total)",
                                "top_k": 5, "reranker_score": "raw logit, same pinned reranker for both paths"},
              "policy": "Scoping uses only the question text and the corpus inventory; no labels; no tuning on dev or test.",
              "setup_seconds": setup_seconds, "summary": summary, "per_question": rows,
              "limitations": ["Agent-authored, non-exhaustive gold evidence without independent human review.",
                              "Source-selection metrics only; generated answers are not graded here.",
                              "Benchmark questions name their company and year explicitly, so this measures the explicit-scope case, not relative or undated questions."]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        entry = summary[split]
        print(split, "ER@5 global %.4f scoped %.4f" % (entry["global"]["evidence_recall_at_5"], entry["scoped"]["evidence_recall_at_5"]),
              entry["paired_evidence_recall_at_5_scoped_minus_global"]["ci95_pp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
