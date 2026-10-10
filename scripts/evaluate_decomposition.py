"""Ablate comparison decomposition on the frozen benchmark's comparative questions.

Three retrieval variants share the same models, corpus and 5-passage budget:
  scoped:          one query, company-scoped pools, reranked, balanced Top5 (current)
  decomposed_rule: per-company sub-questions from the deterministic rewrite
  decomposed_llm:  per-company sub-questions from the model (validated JSON)
Metrics are source-selection metrics against the frozen labels; nothing is tuned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from report_chunker.retrieval import judge_retrieval
from report_search.decomposition import DECOMPOSITION_VERSION, retrieve_decomposed
from report_search.experiment import aggregate, evaluate_question, paired_interval
from report_search.scoped_retrieval import merge_pools, plan_pools, select_candidates

VARIANTS = ("scoped", "decomposed_rule", "decomposed_llm")
KEYS = ("evidence_recall_at_5", "all_evidence_hit_at_5", "document_recall_at_5", "all_documents_hit_at_5")


def _load_env(path):
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key.startswith("FINANCE_QA_") and value and not os.getenv(key):
                    os.environ[key] = value


def _metrics(question, top5, pool):
    measured = evaluate_question(question, top5, (5,))["metrics"]
    result = {key: measured[key] for key in KEYS}
    result["pool_evidence_recall"] = judge_retrieval(question, pool, top_k=len(pool))[f"recall_at_{len(pool)}"]
    result["pool_size"] = len(pool)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/evaluation/retrieval_questions.json")
    parser.add_argument("--category", default="comparative")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/decomposition_results.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    _load_env(ROOT / ".env")
    from report_search.demo_service import _default_factory
    from report_search.llm_client import ChatCompletionsClient, ModelConfig

    model = ChatCompletionsClient(ModelConfig.from_env())
    benchmark_bytes = args.benchmark.read_bytes()
    questions = [q for q in json.loads(benchmark_bytes)["questions"] if q["category"] == args.category]
    components = _default_factory(ROOT, args.device)
    rows = []
    try:
        for question in questions:
            query = question["query"]
            scope = components.scope_resolver.resolve(query).to_dict()
            pools = plan_pools(scope, components.documents)
            vector = components.encoder.encode_queries([query])[0]
            scoped_pool = merge_pools(components.retrieval_engine.vector(
                vector, top_k=pool["top_k"], document_ids=pool["document_ids"]) for pool in pools)
            reranked = components.reranker.rank_candidates(query, scoped_pool, top_k=len(scoped_pool))
            scoped_top5, _ = select_candidates(reranked, scope, top_k=5)
            row = {"question_id": question["question_id"], "split": question["split"],
                   "scope_tickers": scope["tickers"], "scoped": _metrics(question, scoped_top5, scoped_pool)}
            for variant, decomposer in (("decomposed_rule", None), ("decomposed_llm", model)):
                result = retrieve_decomposed(encoder=components.encoder, retrieval_engine=components.retrieval_engine,
                                             reranker=components.reranker, resolver=components.scope_resolver,
                                             query=query, scope=scope, pools=pools, model=decomposer)
                row[variant] = _metrics(question, result["selected"], result["candidates"])
                row[variant + "_plan"] = {key: result["trace"]["decomposition"].get(key)
                                          for key in ("source", "rejected", "subquestions")}
            rows.append(row)
            print(question["question_id"], " ".join(f"{v}={row[v]['evidence_recall_at_5']:.2f}" for v in VARIANTS), flush=True)
    finally:
        components.retrieval_engine.close()

    summary = {}
    for split in ("all", "dev", "test"):
        subset = [row for row in rows if split == "all" or row["split"] == split]
        entry = {"question_count": len(subset)}
        for variant in VARIANTS:
            entry[variant] = aggregate([{"question_id": r["question_id"], "metrics": r[variant]} for r in subset])["metrics"]
        for variant in VARIANTS[1:]:
            for key in ("evidence_recall_at_5", "all_evidence_hit_at_5"):
                first = [{"question_id": r["question_id"], "metrics": r[variant]} for r in subset]
                second = [{"question_id": r["question_id"], "metrics": r["scoped"]} for r in subset]
                deltas = [r[variant][key] - r["scoped"][key] for r in subset]
                entry[f"{variant}_minus_scoped_{key}"] = dict(
                    paired_interval(first, second, key),
                    improved=sum(d > 0 for d in deltas), unchanged=sum(d == 0 for d in deltas),
                    worsened=sum(d < 0 for d in deltas))
        entry["llm_plan_sources"] = {source: sum(r["decomposed_llm_plan"]["source"] == source for r in subset)
                                     for source in ("model", "rule")}
        summary[split] = entry
    result = {"schema_version": 1, "experiment": "comparison decomposition ablation (retrieval only)",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "benchmark": {"path": "data/evaluation/retrieval_questions.json", "category": args.category,
                            "sha256": hashlib.sha256(benchmark_bytes).hexdigest(), "question_count": len(rows)},
              "configuration": {"decomposition_version": DECOMPOSITION_VERSION, "decomposer_model": model.config.model,
                                "top_k": 5, "device": args.device,
                                "selection": "round-robin across sub-questions (decomposed) vs balanced scoped Top5"},
              "policy": "Designed on the live bad-case pool. The rule variant was fixed before any benchmark run; the model plan prompt v2 and company-first quotas were revised after seeing v1 test results, so decomposed_llm test numbers are not held out. Dev and test both reported.",
              "summary": summary, "per_question": rows,
              "limitations": ["Agent-authored, non-exhaustive gold evidence; source-selection metrics only.",
                              "One decomposition sample per question from a stochastic model."]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for split in ("all", "dev", "test"):
        entry = summary[split]
        print(split, {v: round(entry[v]["evidence_recall_at_5"], 4) for v in VARIANTS},
              {v: round(entry[v]["all_evidence_hit_at_5"], 4) for v in VARIANTS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
