"""Diagnose a raw-logit refusal threshold using development questions only.

Reuses frozen reranker scores; no model is loaded and no generation is run.
The selected threshold is evaluated once on test, without adjusting criteria.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_chunker.retrieval import judge_retrieval
from report_search.confidence import (LIMITATIONS, binding_from_manifests, calibrate_threshold, confidence_gate,
                                      evaluate_threshold, make_gate_policy)
from report_search.reranker import apply_scores


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_frozen_rows(benchmark_path: Path, reranker_run: Path) -> tuple[list[dict], dict, dict]:
    """Bind scores/source payloads/labels to the previous immutable experiment."""
    benchmark = read_json(benchmark_path)
    comparison = read_json(reranker_run / "comparison.json")
    manifest = read_json(reranker_run / "candidate_manifest.json")
    scoring = read_json(reranker_run / "score_manifest.json")
    if comparison["candidate_manifest"] != manifest or comparison["scoring"] != scoring:
        raise ValueError("Reranker manifests differ from the frozen comparison")
    expected = {"per_query.jsonl": comparison["per_query_sha256"],
                "scores.jsonl": scoring["scores_sha256"],
                "candidates.jsonl": manifest["candidate_file_sha256"]}
    for filename, fingerprint in expected.items():
        if sha256_file(reranker_run / filename) != fingerprint:
            raise ValueError(f"Frozen artifact checksum mismatch: {filename}")
    if (scoring["status"] != "complete" or scoring["contract"]["candidate_file_sha256"] != expected["candidates.jsonl"]
            or sha256_file(benchmark_path) != manifest["benchmark_sha256"]):
        raise ValueError("Benchmark or complete scoring contract differs")
    questions, candidates, scores, evaluated = (benchmark["questions"], read_rows(reranker_run / "candidates.jsonl"),
                                               read_rows(reranker_run / "scores.jsonl"), read_rows(reranker_run / "per_query.jsonl"))
    if not len(questions) == len(candidates) == len(scores) == len(evaluated) == manifest["question_count"]:
        raise ValueError("Expected one complete frozen record per benchmark question")
    rows, strategies = [], set()
    for question, candidate, scored, prior in zip(questions, candidates, scores, evaluated):
        identifier = question["question_id"]
        if any(row["question_id"] != identifier for row in (candidate, scored, prior)) or candidate["query"] != question["query"]:
            raise ValueError("Question order/text differs from frozen candidates")
        pool = candidate["candidates"]
        if len(pool) != 20 or scored["candidate_ids"] != [chunk["chunk_id"] for chunk in pool]:
            raise ValueError("Frozen Top-20 candidate IDs differ")
        selected = apply_scores(pool, scored["scores"], top_k=5)
        if selected != prior["reranked_results"]:
            raise ValueError("Selected Top-5 payload differs from frozen reranker result")
        judged = judge_retrieval(question, selected, top_k=5)
        complete = judged["all_evidence_hit_at_5"]
        if (complete != prior["approaches"]["reranked_top5"]["metrics"]["all_evidence_hit_at_5"]
                or prior["split"] != question["split"] or prior["category"] != question["category"]):
            raise ValueError("Recomputed evidence/split differs from frozen evaluation")
        maximum = confidence_gate(selected, 0.0).max_score
        if maximum is None:
            raise ValueError("Invalid raw scores in frozen reranker result")
        rows.append({"question_id": identifier, "split": question["split"], "category": question["category"],
                     "max_score": maximum, "complete_evidence": bool(complete)})
        strategies.update(chunk["strategy_id"] for chunk in pool)
    if len(strategies) != 1:
        raise ValueError("Diagnostic requires a single chunk strategy")
    binding = binding_from_manifests(manifest, scoring, chunk_strategy=next(iter(strategies)))
    provenance = {"benchmark_sha256": manifest["benchmark_sha256"], "source_reranker_run": reranker_run.as_posix(),
                  "reranker_comparison_sha256": sha256_file(reranker_run / "comparison.json"),
                  "source_artifact_sha256": expected, "candidate_count": manifest["pair_count"],
                  "source_validation": "Prior source-validated frozen labels; Top-5 complete evidence recomputed against unchanged source atoms",
                  "inputs_to_gate": "maximum raw reranker logit only; no answer text, gold report IDs, evidence labels or category as runtime features"}
    return rows, binding, provenance


def run_diagnostic(rows: list[dict], binding: dict, *, target_precision: float, min_accepted: int) -> tuple[dict, dict]:
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs across development/test")
    if any(row.get("split") not in ("dev", "test") for row in rows):
        raise ValueError("Unknown evaluation split")
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    if not dev or not test:
        raise ValueError("Both development and held-out test questions are required")
    calibration = calibrate_threshold(dev, target_precision=target_precision, min_accepted=min_accepted)
    policy = make_gate_policy(calibration, binding)
    # Only the now-frozen selected threshold is evaluated on test; no test grid.
    frozen_threshold = policy["threshold"]
    result = {"schema_version": 1, "experiment": "development-only raw reranker threshold diagnostic",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "calibration": calibration,
              "policy": policy, "test": evaluate_threshold(test, frozen_threshold),
              "baseline_no_gate": {split: evaluate_threshold(values, min(row["max_score"] for row in values))
                                   for split, values in (("dev", dev), ("test", test))},
              "by_category": {split: {category: evaluate_threshold([row for row in values if row["category"] == category], frozen_threshold)
                                      for category in ("factual", "comparative", "summary")}
                              for split, values in (("dev", dev), ("test", test))},
              "limitations": list(LIMITATIONS),
              "metric_definitions": {
                  "coverage": "Fraction of questions whose maximum selected raw logit reaches the frozen threshold.",
                  "accepted_complete_evidence_precision": "Fraction of accepted questions with all annotated source units in the selected Top 5; not answer accuracy.",
                  "accepted_incomplete_evidence_rate": "Accepted questions missing any annotated unit divided by accepted questions; not hallucination rate.",
                  "retained_complete_evidence_fraction": "Complete-evidence questions accepted divided by all complete-evidence questions before gating.",
                  "confidence_intervals": "Descriptive Wilson 95% binomial interval; no adjustment for threshold selection or correlated questions.",
              }}
    return result, policy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/retrieval_questions.json"))
    parser.add_argument("--reranker-run", type=Path, default=Path("data/experiments/reranker"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/generation_gate"))
    parser.add_argument("--target-precision", type=float, default=.90,
                        help="Declared diagnostic target for observed DEV complete-evidence precision; not a safety guarantee (default .90)")
    parser.add_argument("--min-accepted", type=int, default=10,
                        help="Minimum accepted DEV questions for an eligible threshold (default 10)")
    parser.add_argument("--publish", type=Path, help="Optional compact result JSON path; detailed score rows stay under output")
    args = parser.parse_args(argv)
    if args.output.resolve() == args.reranker_run.resolve():
        parser.error("Use a separate output directory to preserve the frozen reranker run")
    rows, binding, provenance = load_frozen_rows(args.benchmark, args.reranker_run)
    result, policy = run_diagnostic(rows, binding, target_precision=args.target_precision, min_accepted=args.min_accepted)
    result["provenance"] = provenance
    result["implementation_sha256"] = {"scripts/calibrate_generation_gate.py": sha256_file(Path(__file__)),
                                       "src/report_search/confidence.py": sha256_file(Path(__file__).resolve().parents[1] / "src/report_search/confidence.py")}
    write_json(args.output / "comparison.json", result)
    write_json(args.output / "policy.json", policy)
    write_json(args.output / "per_query.json", rows)
    if args.publish:
        write_json(args.publish, result)
    print(json.dumps({"status": policy["status"], "threshold": policy["threshold"],
                      "development": result["calibration"]["selected_dev"], "test": result["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
