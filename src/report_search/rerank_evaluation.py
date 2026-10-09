"""Paired source-grounded evaluation of vector Top 5 and reranked Top 20.

These are retrieval metrics. No answers are generated or graded, and the
non-exhaustive benchmark evidence is not a complete relevance judgment set.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from report_chunker.retrieval import judge_retrieval
from report_search.experiment import evaluate_question


APPROACHES = ("direct_top5", "reranked_top5", "candidate_top20")
CATEGORIES = ("factual", "comparative", "summary")
RATE_METRICS = (
    "document_recall_at_5", "all_documents_hit_at_5", "evidence_recall_at_5",
    "evidence_hit_at_5", "all_evidence_hit_at_5", "annotated_evidence_precision_at_5",
    "document_accuracy_at_1", "evidence_accuracy_at_1",
)
RANKING_FIELDS = frozenset({
    "score", "rank", "retrieval_approach", "score_breakdown", "rerank_score",
    "reranker_score", "reranker_raw_score", "reranker_normalized_score",
    "vector_score", "vector_rank", "original_score", "original_rank",
})
METRIC_DEFINITIONS = {
    "document_recall_at_5": "Fraction of distinct gold report IDs found in the first 5 chunks; macro per question.",
    "all_documents_hit_at_5": "One if all distinct gold report IDs appear in the first 5 chunks, else zero.",
    "evidence_recall_at_5": "Fraction of canonical gold source units fully covered by the union of the first 5 chunks; macro per question.",
    "evidence_hit_at_5": "One if the first 5 chunks jointly cover any complete canonical gold source unit, else zero.",
    "all_evidence_hit_at_5": "One if the first 5 chunks jointly cover every required evidence unit; an annotated-evidence answerability proxy, not generated-answer accuracy.",
    "annotated_evidence_precision_at_5": "Number of the first 5 chunks that individually cover at least one complete canonical gold source unit, divided by 5. Missing result slots count as nonrelevant; unjudged valid evidence receives no credit.",
    "document_accuracy_at_1": "One if the first chunk's report ID is a gold report ID, else zero; source selection accuracy, not answer accuracy.",
    "evidence_accuracy_at_1": "One if the first chunk individually covers any complete canonical gold source unit, else zero; annotated source selection accuracy, not answer accuracy.",
    "context_tokens_at_5": "Sum of stored cl100k_base chunk token counts in the first 5 chunks, including overlap.",
    "returned_chunks_at_5": "Number of chunks actually returned, at most 5.",
    "candidate_top20": "The unchanged vector Top 20 pool. Its evidence Recall@20 is an upper bound on evidence Recall@5 achievable by selecting from that pool, not necessarily an attainable Top 5 oracle.",
}
LIMITATIONS = [
    "Agent-authored, source-checked labels await independent human semantic review.",
    "Gold evidence is non-exhaustive; valid unjudged passages receive no evidence credit.",
    "Retrieval accuracy and complete evidence coverage do not measure generated-answer accuracy; no answer generator is evaluated.",
    "Summary coverage measures selected source claims, not completeness of a generated summary.",
    "A fixed number of chunks is not a fixed context-token budget; context token counts include overlap.",
    "Paired query bootstrap intervals are exploratory and ignore within-company and question-template dependence.",
]


def _chunk_map(chunks: list[dict], name: str) -> dict[str, dict]:
    by_id = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(f"{name} must contain chunk objects")
        identifier = chunk.get("chunk_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{name} requires nonempty string chunk_id values")
        if identifier in by_id:
            raise ValueError(f"{name} contains duplicate chunk_id {identifier}")
        by_id[identifier] = chunk
    return by_id


def _same_number(actual, expected) -> bool:
    """Reject nonnumeric/NaN provenance; do not silently tolerate changed scores."""
    try:
        return (not isinstance(actual, bool) and not isinstance(expected, bool)
                and math.isfinite(float(actual)) and math.isfinite(float(expected))
                and actual == expected)
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_selection(candidates: list[dict], reranked: list[dict]) -> None:
    if len(candidates) > 20:
        raise ValueError("The candidate pool must contain at most 20 chunks")
    if len(reranked) != min(5, len(candidates)):
        raise ValueError("Reranking must return exactly min(5, candidate_count) chunks")
    by_id = _chunk_map(candidates, "Candidate pool")
    _chunk_map(reranked, "Reranked selection")
    for selected in reranked:
        original = by_id.get(selected["chunk_id"])
        if original is None:
            raise ValueError("A reranked chunk is outside the candidate pool")
        source_payload = {key: value for key, value in original.items() if key not in RANKING_FIELDS}
        selected_payload = {key: value for key, value in selected.items() if key not in RANKING_FIELDS}
        if source_payload != selected_payload:
            raise ValueError(f"Reranking changed source payload for {selected['chunk_id']}")
        # A changed display score must retain the original vector score. Check
        # every retained score/rank alias, even when the display score is stable.
        score_aliases = [(key, selected[key]) for key in ("vector_score", "original_score") if key in selected]
        breakdown = selected.get("score_breakdown", {})
        if not isinstance(breakdown, dict):
            raise ValueError("score_breakdown must be an object")
        if "vector_raw_score" in breakdown:
            score_aliases.append(("score_breakdown.vector_raw_score", breakdown["vector_raw_score"]))
        if any(not _same_number(value, original.get("score")) for _, value in score_aliases):
            raise ValueError("Reranking changed the retained original vector score")
        if selected.get("score") != original.get("score") and not score_aliases:
            raise ValueError("A changed score must retain the original vector score")
        for field in ("vector_rank", "original_rank"):
            if field in selected and selected[field] != original.get("rank", candidates.index(original) + 1):
                raise ValueError("Reranking changed the retained original vector rank")


def _top_five(question: dict, results: list[dict]) -> dict:
    measured = evaluate_question(question, results, (5,))
    judged = measured["judgments"]["5"]
    individually_relevant = []
    for chunk in results[:5]:
        if judge_retrieval(question, [chunk], top_k=1)["hit_at_1"]:
            individually_relevant.append(chunk["chunk_id"])
    metrics = measured["metrics"]
    metrics.update({
        "evidence_hit_at_5": judged["hit_at_5"],
        "annotated_evidence_precision_at_5": len(individually_relevant) / 5,
        "document_accuracy_at_1": int(bool(results) and results[0]["document_id"] in question["ground_truth_doc_ids"]),
        "evidence_accuracy_at_1": int(bool(results) and results[0]["chunk_id"] in individually_relevant),
        "returned_chunks_at_5": len(results[:5]),
    })
    return {"metrics": metrics, "judgments": measured["judgments"],
            "chunk_ids": [chunk["chunk_id"] for chunk in results[:5]],
            "individually_relevant_chunk_ids": individually_relevant}


def evaluate_pair(question: dict, candidates20: list[dict], reranked5: list[dict]) -> dict:
    """Compare an unchanged vector prefix with an integrity-checked selection.

    Inputs are not modified. Candidate order is the original vector ranking.
    A pool with fewer than 20 chunks is supported for small-corpus fixtures;
    production experiments must separately verify their requested pool size.
    Only ranking metadata may change in a reranked chunk; source payloads,
    citations, and evidence atoms must remain byte-for-value equivalent.
    """
    if question.get("split") not in ("dev", "test"):
        raise ValueError("Question split must be dev or test")
    if question.get("category") not in CATEGORIES:
        raise ValueError("Question category must be factual, comparative, or summary")
    if not isinstance(question.get("question_id"), str) or not question["question_id"]:
        raise ValueError("Question needs a nonempty string question_id")
    candidates, selected = list(candidates20), list(reranked5)
    _validate_selection(candidates, selected)
    direct = _top_five(question, candidates[:5])
    reranked = _top_five(question, selected)
    ceiling = evaluate_question(question, candidates, (20,))
    ceiling["metrics"]["evidence_hit_at_20"] = ceiling["judgments"]["20"]["hit_at_20"]
    ceiling["metrics"]["returned_chunks_at_20"] = len(candidates)
    ceiling = {"metrics": ceiling["metrics"], "judgments": ceiling["judgments"],
               "chunk_ids": [chunk["chunk_id"] for chunk in candidates]}
    return {"question_id": question["question_id"], "category": question["category"], "split": question["split"],
            "candidate_count": len(candidates),
            "approaches": {"direct_top5": direct, "reranked_top5": reranked, "candidate_top20": ceiling},
            "deltas": {key: reranked["metrics"][key] - direct["metrics"][key] for key in direct["metrics"]}}


def _aggregate(rows: list[dict], approach: str) -> dict:
    if not rows:
        return {"question_count": 0, "metrics": {}}
    keys = tuple(rows[0]["approaches"][approach]["metrics"])
    if any(set(row["approaches"][approach]["metrics"]) != set(keys) for row in rows):
        raise ValueError("Every question must expose the same approach metrics")
    return {"question_count": len(rows), "metrics": {
        key: sum(row["approaches"][approach]["metrics"][key] for row in rows) / len(rows) for key in keys}}


def _group(rows: list[dict], bootstrap_replicates: int, seed: int) -> dict:
    result = {"question_count": len(rows),
              "approaches": {name: _aggregate(rows, name) for name in APPROACHES}, "paired_deltas": {}}
    if not rows:
        return result
    # The same resampled question indices are used across all metrics. No
    # independence between metrics or between approaches is assumed.
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rows), (bootstrap_replicates, len(rows)))
    for key in RATE_METRICS:
        delta = np.asarray([row["approaches"]["reranked_top5"]["metrics"][key]
                            - row["approaches"]["direct_top5"]["metrics"][key] for row in rows], dtype=float)
        low, high = np.quantile(delta[sampled].mean(axis=1), [.025, .975])
        result["paired_deltas"][key] = {
            "mean_delta_pp": float(delta.mean() * 100), "ci95_pp": [float(low * 100), float(high * 100)],
            "improved_questions": int((delta > 0).sum()), "unchanged_questions": int((delta == 0).sum()),
            "worsened_questions": int((delta < 0).sum()), "replicates": bootstrap_replicates, "seed": seed,
            "sampling_unit": "paired question; ignores within-company/template dependence",
        }
    return result


def summarize(rows: Iterable[dict], *, bootstrap_replicates: int = 10000, seed: int = 20261008) -> dict:
    """Macro means and paired reranked-minus-direct bootstrap deltas.

    Overall results combine dev and test only as a descriptive view. Test
    results are separately reported; this function never selects parameters.
    """
    rows = list(rows)
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    identifiers = [row["question_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate question_id in paired evaluation")
    if any(row["split"] not in ("dev", "test") or row["category"] not in CATEGORIES for row in rows):
        raise ValueError("Rows need valid split and category values")

    def grouped(values):
        group = _group(values, bootstrap_replicates, seed)
        group["by_category"] = {category: _group([row for row in values if row["category"] == category],
                                                 bootstrap_replicates, seed) for category in CATEGORIES}
        return group

    return {"schema_version": 1, "comparison": "vector Top 5 vs the same vector Top 20 reranked to Top 5",
            "delta_direction": "reranked_top5 minus direct_top5", "metric_definitions": dict(METRIC_DEFINITIONS),
            "overall": grouped(rows),
            "splits": {split: grouped([row for row in rows if row["split"] == split]) for split in ("dev", "test")},
            "limitations": list(LIMITATIONS)}
