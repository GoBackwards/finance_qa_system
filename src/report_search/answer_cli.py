"""Run the answer safeguards on frozen rankings or caller-supplied Top5."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from report_index.faiss_index import sha256_file
from report_search.confidence import binding_from_manifests, load_gate_policy
from report_search.evidence_binding import SourceEvidenceStore
from report_search.experiment import write_json
from report_search.generation import generate_answer
from report_search.llm_client import strict_json
from report_search.rerank_experiment import checked_score_rows
from report_search.reranker import apply_scores


def _read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def load_cached_context(run: Path, question_id: str):
    """Read only query/source candidates/scores; never pass benchmark labels.

    A policy binds to this historical scored corpus/configuration. Sources are
    subsequently checked against canonical parsed chunks before generation.
    """
    candidate_manifest = _read(run / "candidate_manifest.json")
    score_manifest = _read(run / "score_manifest.json")
    if sha256_file(run / "candidates.jsonl") != candidate_manifest["candidate_file_sha256"]:
        raise ValueError("Candidate artifact changed")
    if sha256_file(run / "scores.jsonl") != score_manifest["scores_sha256"]:
        raise ValueError("Score artifact changed")
    candidate_rows = [strict_json(line) for line in (run / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [strict_json(line) for line in (run / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(candidate_rows) != len(scores) or len(scores) != score_manifest["question_count"]:
        raise ValueError("Incomplete scored candidates")
    identifiers = [row["question_id"] for row in candidate_rows]
    if len(set(identifiers)) != len(identifiers) or identifiers != candidate_manifest["question_ids"]:
        raise ValueError("Question identity/order mismatch")
    checked_score_rows(scores, candidate_rows)
    if question_id not in identifiers:
        raise ValueError("Unknown question ID")
    position = identifiers.index(question_id)
    row = candidate_rows[position]
    pool = row["candidates"]
    if (len(pool) != 20 or len({c["chunk_id"] for c in pool}) != 20
            or [c["rank"] for c in pool] != list(range(1, 21))):
        raise ValueError("Expected original vector Top20")
    strategies = {candidate["strategy_id"] for candidate in pool}
    if len(strategies) != 1:
        raise ValueError("Mixed chunk strategies")
    binding = binding_from_manifests(candidate_manifest, score_manifest, chunk_strategy=strategies.pop())
    return row["query"], apply_scores(pool, scores[position]["scores"], top_k=5), binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question-id", help="Replay one question from the verified cached reranker run")
    source.add_argument("--input", type=Path, help="JSON object with query and the existing pipeline's reranked candidates")
    parser.add_argument("--reranker-run", type=Path, default=Path("data/experiments/reranker"))
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--threshold", type=float, help="Explicit finite raw reranker cutoff; no implicit default")
    gate.add_argument("--gate-policy", type=Path, help="Opt-in diagnostic policy, checked against cached scoring configuration")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-context-chars", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threshold is not None and not math.isfinite(args.threshold):
        parser.error("--threshold must be finite")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be positive")
    if args.input and args.gate_policy:
        parser.error("Cached --gate-policy requires --question-id; use an explicit --threshold for caller-supplied scores")
    if args.output:
        protected = [args.input, args.gate_policy, *(args.reranker_run / name for name in
                     ("candidate_manifest.json", "score_manifest.json", "scores.jsonl", "candidates.jsonl"))]
        if any(path and path.resolve() == args.output.resolve() for path in protected):
            parser.error("--output must not replace an input artifact")
    try:
        if args.question_id:
            query, candidates, binding = load_cached_context(args.reranker_run, args.question_id)
        else:
            supplied = _read(args.input)
            if set(supplied) != {"query", "candidates"}:
                raise ValueError("Input must contain only query and candidates")
            query, candidates = supplied["query"], supplied["candidates"]
        threshold = args.threshold
        if args.gate_policy:
            threshold = load_gate_policy(args.gate_policy, binding)["threshold"]
        result = generate_answer(query, candidates, threshold=threshold,
                                 source_store=SourceEvidenceStore(args.processed),
                                 max_context_chars=args.max_context_chars)
        result["audit"]["threshold_source"] = "diagnostic_policy" if args.gate_policy else "explicit_or_unconfigured"
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        result = {"status": "abstain", "answer": "I'm not sure", "citations": [],
                  "reference_verified": False, "semantic_supported": False,
                  "reason": "input_or_policy_validation_failed", "audit": {"generation_called": False}}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "answer" else 2


if __name__ == "__main__":
    raise SystemExit(main())
