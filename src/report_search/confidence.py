"""Fail-closed reranker gate and development-only selective retrieval diagnostic.

Raw reranker logits are ranking scores, not answer-correctness probabilities.
Complete annotated evidence in the selected context is a calibration proxy;
passing this gate does not establish that a generated answer is supported.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import copy
import json
import math
from numbers import Real
from pathlib import Path


SCORE_KIND = "reranker_raw_logit"
BINDING_KEYS = frozenset({"reranker", "chunks_sha256", "chunk_strategy", "vector_retrieval",
                          "candidate_depth", "selected_depth", "score_statistic"})
LIMITATIONS = [
    "Raw reranker logits are not calibrated correctness probabilities.",
    "Complete Top-5 annotated evidence is a retrieval proxy, not generated-answer correctness or semantic support.",
    "Gold evidence is non-exhaustive and awaits human semantic review; valid unjudged sources receive no credit.",
    "There are no curated unanswerable or out-of-domain questions; this experiment cannot measure hallucination or OOD refusal rates.",
    "The target uses observed development precision, not a confidence-bound guarantee; adaptive threshold selection and company/template dependence weaken interval interpretation.",
    "Wilson 95% intervals are descriptive binomial intervals; small samples and correlated questions limit their interpretation.",
    "The diagnostic policy is not installed as a production default; an unconfigured application must refuse.",
]


def _finite_number(value) -> bool:
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    threshold: float | None
    max_score: float | None
    candidate_count: int
    score_kind: str = SCORE_KIND

    def to_dict(self) -> dict:
        return asdict(self)


def confidence_gate(candidates: Sequence[Mapping], threshold: float | None) -> GateDecision:
    """Allow only when every raw score is valid and their maximum >= threshold.

    None means no configured policy and always refuses. Any missing or invalid
    score refuses the whole request, even when another candidate scores highly.
    Generic ``score``/normalized hybrid values are deliberately never used.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return GateDecision(False, "invalid_candidates", None, None, 0)
    count = len(candidates)
    if threshold is None:
        return GateDecision(False, "unconfigured_threshold", None, None, count)
    if not _finite_number(threshold):
        return GateDecision(False, "invalid_threshold", None, None, count)
    threshold = float(threshold)
    if not count:
        return GateDecision(False, "empty_candidates", threshold, None, 0)
    if any(not isinstance(candidate, Mapping) or "reranker_score" not in candidate for candidate in candidates):
        return GateDecision(False, "missing_reranker_score", threshold, None, count)
    values = [candidate["reranker_score"] for candidate in candidates]
    if not all(_finite_number(value) for value in values):
        return GateDecision(False, "invalid_reranker_score", threshold, None, count)
    maximum = max(float(value) for value in values)
    allowed = maximum >= threshold
    return GateDecision(allowed, "above_threshold" if allowed else "below_threshold", threshold, maximum, count)


def _validated_rows(rows: Sequence[dict], split: str | None = None) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Calibration rows must be objects")
        identifier = row.get("question_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Calibration requires unique nonempty question_id values")
        seen.add(identifier)
        if row.get("split") not in ("dev", "test") or (split is not None and row["split"] != split):
            raise ValueError(f"Calibration rows must use {split or 'dev/test'} split")
        if not _finite_number(row.get("max_score")):
            raise ValueError("Calibration max_score must be a finite raw reranker logit")
        label = row.get("complete_evidence")
        if not isinstance(label, (bool, int)) or label not in (0, 1):
            raise ValueError("complete_evidence must be binary")
        result.append(dict(row, max_score=float(row["max_score"]), complete_evidence=bool(label)))
    return result


def _wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    midpoint = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, midpoint - half), min(1.0, midpoint + half)]


def evaluate_threshold(rows: Sequence[dict], threshold: float | None) -> dict:
    """Evaluate a fixed policy, with None representing refusal for every query."""
    rows = _validated_rows(rows)
    if threshold is not None and not _finite_number(threshold):
        raise ValueError("Threshold must be finite or None for refusal-only")
    accepted = [row for row in rows if threshold is not None and row["max_score"] >= threshold]
    complete = sum(row["complete_evidence"] for row in accepted)
    total_complete = sum(row["complete_evidence"] for row in rows)
    count, accepted_count = len(rows), len(accepted)
    return {
        "threshold": float(threshold) if threshold is not None else None,
        "question_count": count, "accepted_count": accepted_count, "refused_count": count - accepted_count,
        "coverage": accepted_count / count if count else None,
        "complete_evidence_count": complete, "incomplete_evidence_count": accepted_count - complete,
        "accepted_complete_evidence_precision": complete / accepted_count if accepted_count else None,
        "accepted_incomplete_evidence_rate": (accepted_count - complete) / accepted_count if accepted_count else None,
        "accepted_complete_evidence_precision_wilson95": _wilson(complete, accepted_count),
        "total_complete_evidence_count": total_complete,
        "retained_complete_evidence_fraction": complete / total_complete if total_complete else None,
        "refused_complete_evidence_count": total_complete - complete,
    }


def calibrate_threshold(dev_rows: Sequence[dict], *, target_precision: float, min_accepted: int) -> dict:
    """Maximize dev coverage subject to a declared observed precision target.

    Every unique development maximum is a candidate inclusive threshold. Ties
    move together; no test row is accepted by this API. If none meets the
    target/count criteria, the returned threshold is None (refuse all).
    """
    if not _finite_number(target_precision) or not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    if isinstance(min_accepted, bool) or not isinstance(min_accepted, int) or min_accepted < 1:
        raise ValueError("min_accepted must be a positive integer")
    rows = _validated_rows(dev_rows, "dev")
    grid = [evaluate_threshold(rows, threshold) for threshold in sorted({row["max_score"] for row in rows}, reverse=True)]
    eligible = [row for row in grid if row["accepted_count"] >= min_accepted
                and row["accepted_complete_evidence_precision"] >= target_precision]
    selected = max(eligible, key=lambda row: row["accepted_count"]) if eligible else evaluate_threshold(rows, None)
    return {
        "status": "diagnostic_threshold_selected" if eligible else "refusal_only_no_eligible_threshold",
        "threshold": selected["threshold"], "score_kind": SCORE_KIND,
        "criterion": {"target_observed_complete_evidence_precision": float(target_precision),
                      "minimum_accepted_dev_questions": min_accepted,
                      "selection": "maximum development coverage satisfying both criteria; inclusive threshold; score ties stay together"},
        "selected_dev": selected, "dev_tradeoff_grid": grid,
        "selection_split": "dev", "production_certified": False,
    }


def _validate_binding(binding: dict) -> None:
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        raise ValueError("Gate binding must specify model, corpus, retrieval, and selection fingerprints")
    if not isinstance(binding["reranker"], dict) or not isinstance(binding["vector_retrieval"], dict):
        raise ValueError("Gate model and retrieval bindings must be objects")
    for field in ("model_id", "model_revision", "model_weights_sha256", "implementation_sha256",
                  "inference_precision", "document_rendering", "tokenizer_config_sha256"):
        if not isinstance(binding["reranker"].get(field), str) or not binding["reranker"][field]:
            raise ValueError(f"Missing reranker binding {field}")
    if binding["reranker"].get("score_type") != "raw_logit" or binding["reranker"].get("normalized") is not False:
        raise ValueError("Gate binding requires unnormalized reranker logits")
    for field in ("chunks_sha256", "chunk_strategy"):
        if not isinstance(binding[field], str) or not binding[field]:
            raise ValueError(f"Missing gate binding {field}")
    for field in ("candidate_depth", "selected_depth"):
        if isinstance(binding[field], bool) or not isinstance(binding[field], int) or binding[field] < 1:
            raise ValueError(f"Invalid {field}")
    if binding["selected_depth"] > binding["candidate_depth"] or binding["score_statistic"] != "maximum_selected_reranker_score":
        raise ValueError("Unsupported confidence score statistic or selection")


def binding_from_manifests(candidate_manifest: dict, score_manifest: dict, *, chunk_strategy: str) -> dict:
    """Derive compatibility binding from independently verified active artifacts.

    The caller must first check actual candidate/score/corpus/index checksums;
    this helper checks the two manifest contracts agree, not file contents.
    chunk_strategy must come from the actual selected candidate source payload.
    """
    if (score_manifest.get("status") != "complete" or candidate_manifest.get("candidate_depth") != 20
            or score_manifest["contract"]["candidate_file_sha256"] != candidate_manifest["candidate_file_sha256"]):
        raise ValueError("Active candidate and score contracts differ or are incomplete")
    identity = score_manifest["contract"]["reranker"]
    reranker = {key: value for key, value in identity.items() if key not in ("device", "batch_size", "packages")}
    vector = candidate_manifest["vector_query_manifest"]["contract"]
    binding = {"reranker": reranker, "chunks_sha256": candidate_manifest["corpus"]["chunks_sha256"],
               "chunk_strategy": chunk_strategy, "candidate_depth": 20, "selected_depth": 5,
               "score_statistic": "maximum_selected_reranker_score",
               "vector_retrieval": {"index": "FlatIP", "encoder": vector["corpus_encoder"],
                                    "corpus_vectors_file_sha256": vector["corpus_vectors_file_sha256"],
                                    "corpus_embedding_manifest_sha256": vector["corpus_embedding_manifest_sha256"]}}
    _validate_binding(binding)
    return binding


def make_gate_policy(calibration: dict, binding: dict) -> dict:
    """Package a diagnostic selection without silently enabling it in an app."""
    _validate_binding(binding)
    policy = {"schema_version": 1, "status": calibration["status"], "threshold": calibration["threshold"],
              "score_kind": SCORE_KIND, "selection_split": "dev", "production_certified": False,
              "criterion": copy.deepcopy(calibration["criterion"]), "binding": copy.deepcopy(binding)}
    _validate_policy(policy, binding)
    return policy


def _validate_policy(policy: dict, expected_binding: dict) -> None:
    _validate_binding(expected_binding)
    if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1 or policy.get("score_kind") != SCORE_KIND):
        raise ValueError("Unsupported gate policy schema or score kind")
    _validate_binding(policy.get("binding"))
    if policy["binding"] != expected_binding:
        raise ValueError("Gate policy binding does not match the active model/corpus/retrieval configuration")
    if (policy.get("selection_split") != "dev" or policy.get("production_certified") is not False
            or "threshold" not in policy):
        raise ValueError("Gate policy must be an explicit development-only diagnostic")
    threshold = policy["threshold"]
    expected_status = "refusal_only_no_eligible_threshold" if threshold is None else "diagnostic_threshold_selected"
    if policy.get("status") != expected_status or (threshold is not None and not _finite_number(threshold)):
        raise ValueError("Gate policy has an invalid threshold or status")
    criterion = policy.get("criterion", {})
    if not isinstance(criterion, dict):
        raise ValueError("Gate policy criteria must be an object")
    precision, count = criterion.get("target_observed_complete_evidence_precision"), criterion.get("minimum_accepted_dev_questions")
    if not _finite_number(precision) or not 0 < precision <= 1 or isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Gate policy has invalid development criteria")


def load_gate_policy(path: str | Path, expected_binding: dict) -> dict:
    """Load only with an independently supplied active-configuration binding.

    Never construct expected_binding by copying it from the policy being loaded;
    it must describe the actually running reranker, corpus and retrieval path.
    This is a compatibility check, not a cryptographic signature/authenticity test.
    """
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_policy(policy, expected_binding)
    return policy
