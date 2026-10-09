"""Bounded retrieval caching and cooperative request deadlines.

Cache entries contain reranked passages only, never generated answers. A caller
must bind all corpus/model/index/query-policy versions to its exact query. The
byte limit counts the immutable serialized payload and digest, not Python object
overhead; the independent entry limit also bounds per-entry overhead.

Deadlines are checked at stage boundaries. They cannot interrupt an active CPU,
GPU, or blocking network operation; no background workers are started here.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import threading
import time


REQUIRED_CACHE_BINDINGS = frozenset({"corpus", "encoder", "index", "reranker", "query_policy"})


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _json_bytes(value):
    """Reject values whose JSON representation loses identity or admits NaNs."""
    def validate(item):
        if item is None or isinstance(item, (str, bool)):
            return
        if isinstance(item, int):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError("Cache values must contain only finite JSON values and string keys")

    try:
        validate(value)
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, UnicodeError, OverflowError, RecursionError) as error:
        raise ValueError("Cache values must be valid finite JSON") from error


def _cache_key(query, binding):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Cache query must be a nonempty string")
    if not isinstance(binding, dict) or not REQUIRED_CACHE_BINDINGS.issubset(binding):
        raise ValueError("Cache binding requires corpus, encoder, index, reranker, and query_policy")
    for name in REQUIRED_CACHE_BINDINGS:
        value = binding[name]
        if not ((isinstance(value, str) and value.strip()) or (isinstance(value, dict) and value)):
            raise ValueError(f"Cache binding {name} must be a nonempty version string or configuration object")
    return hashlib.sha256(_json_bytes({"version": 1, "query": query, "binding": binding})).digest()


def _valid_candidates(candidates):
    if not isinstance(candidates, list) or not candidates:
        return False
    identifiers = set()
    for position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            return False
        identifier = candidate.get("chunk_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            return False
        identifiers.add(identifier)
        approach = candidate.get("retrieval_approach")
        if approach not in {"vector_reranker", "expanded_hybrid_reranker"}:
            return False
        if approach != candidates[0].get("retrieval_approach"):
            return False
        rank = candidate.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != position:
            return False
        for field in ("score", "reranker_score"):
            score = candidate.get(field)
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                return False
        if candidate["score"] != candidate["reranker_score"]:
            return False
        if position > 1 and candidates[position - 2]["reranker_score"] < candidate["reranker_score"]:
            return False
        score_field = "candidate_score" if approach == "expanded_hybrid_reranker" else "vector_score"
        rank_field = "candidate_rank" if approach == "expanded_hybrid_reranker" else "vector_rank"
        if approach == "expanded_hybrid_reranker" and not {score_field, rank_field} <= candidate.keys():
            return False
        if score_field in candidate:
            score = candidate[score_field]
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                return False
        if rank_field in candidate:
            rank = candidate[rank_field]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                return False
    return True


@dataclass(frozen=True)
class _Entry:
    payload: bytes
    expires_at: float
    size: int


class RetrievalCache:
    """Thread-safe LRU/TTL store whose reads return independent passage copies.

    Binding completeness is a caller responsibility: include content hashes or
    immutable revisions, search parameters, rendering versions, and all query
    policies. Required top-level components prevent accidentally unbound keys.
    Mutating a model/index while retaining its binding is unsupported.
    """

    def __init__(self, *, max_entries=128, max_bytes=64_000_000, ttl_seconds=300,
                 clock=time.monotonic):
        _positive_integer(max_entries, "max_entries")
        _positive_integer(max_bytes, "max_bytes")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = _positive_seconds(ttl_seconds, "ttl_seconds")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock = clock
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def _now(self):
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(now):
            raise ValueError("clock must return finite monotonic seconds")
        return float(now)

    def _expire(self, now):
        for key in [key for key, entry in self._entries.items() if now >= entry.expires_at]:
            self._bytes -= self._entries.pop(key).size

    def get(self, query, binding):
        key = _cache_key(query, binding)
        with self._lock:
            self._expire(self._now())
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            payload = entry.payload
        return json.loads(payload)

    def put(self, query, binding, candidates):
        """Return False for empty, malformed, failed, or oversized results.

        Bad query/binding configuration raises ValueError. A rejected write
        never replaces a previously valid result and never evicts other keys.
        """
        key = _cache_key(query, binding)
        if not _valid_candidates(candidates):
            return False
        try:
            payload = _json_bytes(candidates)
        except ValueError:
            return False
        size = len(key) + len(payload)
        if size > self.max_bytes:
            return False
        with self._lock:
            now = self._now()
            expires_at = now + self.ttl_seconds
            if not math.isfinite(expires_at):
                raise ValueError("Cache expiry must be finite")
            self._expire(now)
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old.size
            while self._entries and (len(self._entries) >= self.max_entries
                                     or self._bytes + size > self.max_bytes):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.size
            self._entries[key] = _Entry(payload, expires_at, size)
            self._bytes += size
        return True

    @property
    def stats(self):
        with self._lock:
            self._expire(self._now())
            return {"entries": len(self._entries), "bytes": self._bytes,
                    "hits": self._hits, "misses": self._misses,
                    "byte_measurement": "serialized_payload_plus_digest"}

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0


class DeadlineExceeded(RuntimeError):
    def __init__(self, stage="request"):
        self.stage = stage
        super().__init__("request_deadline_exceeded")


class RequestBudget:
    """Measure stages and stop later work after a cooperative deadline expires."""

    def __init__(self, timeout_seconds, *, clock=time.monotonic):
        self.timeout_seconds = _positive_seconds(timeout_seconds, "timeout_seconds")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock = clock
        self._started_at = self._now()
        self._deadline = self._started_at + self.timeout_seconds
        if not math.isfinite(self._deadline):
            raise ValueError("Request deadline must be finite")
        self._expired = False
        self._timings = []

    def _now(self):
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(now):
            raise ValueError("clock must return finite monotonic seconds")
        return float(now)

    def _check_at(self, now, stage):
        if self._expired or now >= self._deadline:
            self._expired = True
            raise DeadlineExceeded(stage)

    @property
    def elapsed_seconds(self):
        return max(0.0, self._now() - self._started_at)

    @property
    def remaining_seconds(self):
        return max(0.0, self._deadline - self._now()) if not self._expired else 0.0

    @property
    def timings(self):
        return [dict(item) for item in self._timings]

    def check(self, stage="request"):
        self._check_at(self._now(), stage)

    @contextmanager
    def stage(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Stage name must be a nonempty string")
        started_at = self._now()
        self._check_at(started_at, name)
        try:
            yield self
        except BaseException as error:
            ended_at = self._now()
            self._timings.append({"stage": name, "duration_ms": max(0.0, ended_at - started_at) * 1000,
                                  "status": "failed", "error_type": type(error).__name__})
            if ended_at >= self._deadline:
                self._expired = True
            raise
        else:
            ended_at = self._now()
            expired = self._expired or ended_at >= self._deadline
            self._timings.append({"stage": name, "duration_ms": max(0.0, ended_at - started_at) * 1000,
                                  "status": "deadline_exceeded" if expired else "completed"})
            self._check_at(ended_at, name)

    def to_dict(self):
        now = self._now()
        return {"timeout_seconds": self.timeout_seconds,
                "elapsed_ms": max(0.0, now - self._started_at) * 1000,
                "remaining_ms": max(0.0, self._deadline - now) * 1000 if not self._expired else 0.0,
                "deadline_exceeded": self._expired or now >= self._deadline,
                "cooperative_deadline": True, "stages": self.timings}
