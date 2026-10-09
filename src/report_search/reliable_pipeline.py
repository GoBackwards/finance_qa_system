"""Bounded recovery and observability around the existing evidence-bound QA.

Expansion is opt-in and has a separately configured raw-logit threshold.
Neither a retry nor a cache hit bypasses source and answer verification.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

from report_search.confidence import confidence_gate
from report_search.evidence_binding import SourceEvidenceStore
from report_search.generation import generate_answer
from report_search.llm_client import ChatCompletionsClient, ModelConfig, ModelError
from report_search.query_policy import plan_query
from report_search.rerank_evaluation import _validate_selection
from report_search.runtime_support import DeadlineExceeded, RequestBudget


PIPELINE_VERSION = "bounded-recovery-v1"
TRANSIENT_ERRORS = frozenset({"model_http_429", "model_http_500", "model_http_502",
                             "model_http_503", "model_http_504", "model_unavailable"})


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _refusal(reason):
    return {"status": "abstain", "answer": "I'm not sure", "citations": [],
            "reference_verified": False, "semantic_supported": False, "reason": reason,
            "audit": {"generation_called": False, "semantic_verifier_called": False}}


def failure_category(reason):
    if reason is None:
        return None
    if reason in {"company_not_in_corpus", "report_not_in_corpus"}:
        return "corpus_coverage"
    if reason == "deadline_exceeded":
        return "latency_budget"
    if reason in {"empty_candidates", "below_threshold", "retrieved_scope_incomplete"}:
        return "insufficient_retrieval"
    if reason in {"unsupported_claim", "question_not_answered", "generator_abstained"}:
        return "unsupported_answer"
    if reason.startswith("model_") or reason in {"generation_failed", "semantic_verifier_failed"}:
        return "provider_or_configuration"
    if reason in {"invalid_query", "invalid_threshold", "unconfigured_threshold"}:
        return "request_configuration"
    if reason == "source_verification_failed" or "citation" in reason or "quote" in reason or "evidence_id" in reason:
        return "source_or_reference_integrity"
    if reason == "retrieval_failed":
        return "retrieval_runtime"
    return "validation_failure"


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


def scope_complete(candidates, plan):
    constraints = plan["constraints"]
    if not set(constraints["requested_tickers"]) <= {c.get("ticker") for c in candidates}:
        return False
    required = {(r["ticker"], r["fiscal_year"]) for r in constraints.get("requested_reports", [])}
    if required:
        return required <= {(c.get("ticker"), c.get("fiscal_year")) for c in candidates}
    return set(constraints["requested_report_years"]) <= {c.get("fiscal_year") for c in candidates}


def verify_reranking(pool, selected):
    """Reject invented candidates, changed source payloads and invalid logits."""
    _validate_selection(pool, selected)
    if [c.get("rank") for c in selected] != list(range(1, len(selected) + 1)):
        raise ValueError("Invalid final reranker ranks")
    if selected and not confidence_gate(selected, -1e300).allowed:
        raise ValueError("Invalid reranker scores")
    scores = [c["reranker_score"] for c in selected]
    if scores != sorted(scores, reverse=True) or any(c.get("score") != c["reranker_score"] for c in selected):
        raise ValueError("Invalid reranker order or display scores")


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


class BudgetedSourceStore:
    def __init__(self, store, budget):
        self.store, self.budget = store, budget

    def build(self, candidates):
        with self.budget.stage("source_binding"):
            return self.store.build(candidates)


class BudgetedClient:
    """One optional retry for transient errors, within the shared deadline.

    Actual HTTP clients get the remaining budget as their socket timeout.
    Socket timeouts and stage checks are not hard cancellation of an in-flight
    operation; late responses are discarded and subsequent stages do not run.
    """
    def __init__(self, client, budget, stage, *, retries=1, sleeper=time.sleep):
        self.client, self.budget, self.stage_name = client, budget, stage
        self.retries, self.sleeper, self.attempts = retries, sleeper, 0

    @property
    def identity(self):
        return getattr(self.client, "identity", {"backend": "injected"})

    def complete_json(self, messages):
        with self.budget.stage(self.stage_name):
            for attempt in range(self.retries + 1):
                self.budget.check(self.stage_name)
                client = self.client
                if isinstance(client, ChatCompletionsClient):
                    remaining = self.budget.remaining_seconds
                    client = ChatCompletionsClient(replace(client.config,
                        timeout_seconds=min(client.config.timeout_seconds, remaining)))
                self.attempts += 1
                try:
                    result = client.complete_json(messages)
                    self.budget.check(self.stage_name)
                    return result
                except ModelError as error:
                    if str(error) not in TRANSIENT_ERRORS or attempt == self.retries:
                        raise
                    self.budget.check(self.stage_name)
                    # Never reset the request clock or perform an unbounded retry.
                    if self.budget.remaining_seconds <= 0.25:
                        raise
                    self.sleeper(0.1)
            raise ModelError("model_unavailable")


class ReliableQAPipeline:
    """Single-session controller; inject the existing encoder/index/reranker.

    ``binding`` must be derived from active verified artifacts, never from a
    cache entry. Query, model/corpus binding, inventory, thresholds, expansion
    policy, and code version form the cache namespace. No answers are cached.
    """
    def __init__(self, *, inventory, encoder, retrieval_engine, reranker, binding,
                 threshold=None, expansion_threshold=None, enable_expansion=False,
                 cache=None, source_store=None, generator=None, verifier=None,
                 deadline_seconds=30.0, latency_warning_seconds=3.0,
                 clock=time.monotonic, sleeper=time.sleep):
        self.inventory = copy.deepcopy(inventory)
        self.encoder, self.engine, self.reranker = encoder, retrieval_engine, reranker
        self.threshold, self.expansion_threshold = threshold, expansion_threshold
        self.enable_expansion = enable_expansion
        if type(enable_expansion) is not bool:
            raise ValueError("enable_expansion must be a boolean")
        if enable_expansion and confidence_gate([], expansion_threshold).reason != "empty_candidates":
            raise ValueError("Expansion needs its own explicit finite threshold; the old vector policy is not reused")
        if not isinstance(binding, dict) or not all(binding.get(k) for k in ("corpus", "encoder", "index", "reranker", "query_policy")):
            raise ValueError("Supply a complete verified corpus/model/index/query-policy cache binding")
        self.binding = {**copy.deepcopy(binding), "pipeline_version": PIPELINE_VERSION,
                        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "query_policy_sha256": hashlib.sha256(Path(__file__).with_name("query_policy.py").read_bytes()).hexdigest(),
                        "inventory_sha256": _hash(inventory), "threshold": threshold,
                        "expansion_threshold": expansion_threshold, "enable_expansion": enable_expansion}
        _hash(self.binding)
        if isinstance(latency_warning_seconds, bool) or not isinstance(latency_warning_seconds, (int, float)) or not math.isfinite(latency_warning_seconds) or latency_warning_seconds <= 0:
            raise ValueError("latency_warning_seconds must be finite and positive")
        RequestBudget(deadline_seconds, clock=clock)
        self.deadline_seconds, self.latency_warning_seconds = deadline_seconds, latency_warning_seconds
        self.clock, self.sleeper = clock, sleeper
        self.cache = cache
        self.source_store = source_store if source_store is not None else SourceEvidenceStore()
        self.generator, self.verifier = generator, verifier

    def answer(self, query):
        budget = RequestBudget(self.deadline_seconds, clock=self.clock)
        audit = {"pipeline_version": PIPELINE_VERSION, "cache_hit": False, "expansion_used": False,
                 "candidate_depth": 20, "selected_depth": 5, "reranker_pairs": 0,
                 "binding_sha256": _hash(self.binding), "retry_attempts": {}}
        generator = verifier = None

        def finish(result):
            audit.update(budget.to_dict())
            audit["slow_request"] = budget.elapsed_seconds > self.latency_warning_seconds
            audit["latency_warning_seconds"] = self.latency_warning_seconds
            audit["failure_category"] = failure_category(result.get("reason"))
            if generator is not None:
                audit["retry_attempts"]["generator"] = generator.attempts
            if verifier is not None:
                audit["retry_attempts"]["verifier"] = verifier.attempts
            result.setdefault("audit", {})["generation_called"] = bool(generator and generator.attempts)
            result["audit"]["semantic_verifier_called"] = bool(verifier and verifier.attempts)
            result.setdefault("audit", {})["runtime"] = audit
            return result

        try:
            if not isinstance(query, str) or not query.strip() or len(query) > 10_000:
                return finish(_refusal("invalid_query"))
            with budget.stage("query_policy"):
                plan = plan_query(query, self.inventory)
            audit["query_plan"] = plan
            if plan["coverage"]["status"] == "unavailable":
                reason = "company_not_in_corpus" if plan["coverage"]["missing_tickers"] else "report_not_in_corpus"
                return finish(_refusal(reason))
            gate = confidence_gate([], self.threshold)
            if gate.reason in {"invalid_threshold", "unconfigured_threshold"}:
                return finish(_refusal(gate.reason))
            candidates = None
            if self.cache is not None:
                with budget.stage("cache_lookup"):
                    try:
                        candidates = self.cache.get(query, self.binding)
                    except (ValueError, TypeError, KeyError, OSError):
                        audit["cache_error"] = "invalid_cache_entry"
                audit["cache_hit"] = candidates is not None
            threshold = self.threshold
            if candidates is None:
                with budget.stage("embedding"):
                    vector = self.encoder.encode_queries([query])[0]
                with budget.stage("vector_retrieval"):
                    pool = filter_scope(self.engine.vector(vector, top_k=20), plan)
                candidates = []
                if pool:
                    with budget.stage("reranking"):
                        candidates = self.reranker.rank_candidates(query, pool, top_k=5)
                        verify_reranking(pool, candidates)
                    audit["reranker_pairs"] += len(pool)
                weak = not confidence_gate(candidates, threshold).allowed or not scope_complete(candidates, plan)
                if weak and self.enable_expansion:
                    audit["expansion_used"] = True
                    channels = [("vector", pool)]
                    with budget.stage("lexical_fallback"):
                        for number, lexical_query in enumerate(plan["lexical_queries"][:3]):
                            lexical = self.engine.bm25(lexical_query, top_k=20)
                            channels.append((f"lexical_{number}", filter_scope(lexical, plan)))
                        expanded = reciprocal_rank_pool(channels)
                    candidates = []
                    if expanded:
                        with budget.stage("fallback_reranking"):
                            candidates = self.reranker.rank_candidates(query, expanded, top_k=5)
                            verify_reranking(expanded, candidates)
                        audit["reranker_pairs"] += len(expanded)
                        for candidate in candidates:
                            candidate["candidate_score"] = candidate.pop("vector_score", None)
                            candidate["candidate_rank"] = candidate.pop("vector_rank", None)
                            candidate["retrieval_approach"] = "expanded_hybrid_reranker"
                    threshold = self.expansion_threshold
            elif any(c.get("retrieval_approach") == "expanded_hybrid_reranker" for c in candidates):
                audit["expansion_used"] = True
                threshold = self.expansion_threshold
            # Recheck constraints on cache hits as well as freshly retrieved text.
            if filter_scope(candidates, plan) != candidates or not scope_complete(candidates, plan):
                return finish(_refusal("retrieved_scope_incomplete"))
            gate = confidence_gate(candidates, threshold)
            if not gate.allowed:
                result = _refusal(gate.reason)
                result["audit"]["confidence_gate"] = gate.to_dict()
                return finish(result)
            # Lazy construction means scope/score refusals need no provider configuration.
            generator_client = self.generator if self.generator is not None else ChatCompletionsClient(ModelConfig.from_env())
            verifier_client = self.verifier if self.verifier is not None else ChatCompletionsClient(ModelConfig.from_env(verifier=True))
            generator = BudgetedClient(generator_client, budget, "generation", sleeper=self.sleeper)
            verifier = BudgetedClient(verifier_client, budget, "semantic_verification", sleeper=self.sleeper)
            result = generate_answer(query, candidates, threshold=threshold,
                source_store=BudgetedSourceStore(self.source_store, budget), generator=generator, verifier=verifier)
            budget.check("answer_release")
            if result["status"] == "answer" and self.cache is not None:
                try:
                    with budget.stage("cache_write"):
                        self.cache.put(query, self.binding, candidates)
                except (ValueError, TypeError, KeyError, OSError):
                    audit["cache_error"] = "cache_write_failed"
            budget.check("answer_release")
            return finish(result)
        except DeadlineExceeded:
            return finish(_refusal("deadline_exceeded"))
        except ModelError as error:
            return finish(_refusal(str(error)))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, RuntimeError):
            return finish(_refusal("invalid_query" if not isinstance(query, str) or not query.strip() else "retrieval_failed"))
