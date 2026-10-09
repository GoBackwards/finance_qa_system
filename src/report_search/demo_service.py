"""Thread-owned retrieval resources for a rerun-based demo UI.

Only model/index/source-store resources are reused. Queries, rankings, answers
and provider credentials are never cached by this service. All SQLite/model
operations run on one dedicated worker, including initialization and cleanup.
"""
from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
from queue import Queue
import re
import secrets
import threading
import time

from report_search.generation import generate_answer
from report_search.llm_client import ChatCompletionsClient, ModelConfig, ModelError


class DemoError(RuntimeError):
    """A safe public code, optionally retaining already retrieved evidence."""

    def __init__(self, code, *, partial_result=None):
        self.code = code
        self.partial_result = partial_result
        super().__init__(code)


@dataclass
class DemoComponents:
    encoder: object
    retrieval_engine: object
    reranker: object
    source_store: object
    stats: dict | None = None


def _valid_query(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 10_000:
        return False
    try:
        query.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _default_factory(root, device):
    # Imports and native resources stay on the owning worker. Existing index
    # loaders verify hashes and ordered SQLite/FAISS chunk mappings themselves.
    from report_index.embedding import DenseEncoder, sha256_file
    from report_search.evidence_binding import SourceEvidenceStore
    from report_search.reranker import BGEReranker
    from report_search.retrieval import RetrievalEngine

    index_root = root / "data/indexes/bge-m3/table_preserving_s512_o50"
    embedding_path = index_root / "embeddings/embedding_manifest.json"
    metadata = json.loads(embedding_path.read_text(encoding="utf-8"))
    encoder_meta = metadata["encoder"]
    inventory = json.loads((root / "data/manifest.json").read_text(encoding="utf-8-sig"))
    cache = root / ".cache/huggingface"
    # Explicit local snapshots avoid implicit downloads or global environment
    # changes from a shared UI process. Model preparation remains an offline step.
    encoder_snapshot = cache / ("models--" + encoder_meta["model_id"].replace("/", "--")) / "snapshots" / encoder_meta["model_revision"]
    encoder = DenseEncoder(model_id=encoder_meta["model_id"], revision=encoder_meta["model_revision"],
                           device=device, batch_size=1, cache_dir=cache, local_model_path=encoder_snapshot)
    engine = RetrievalEngine(root / "data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl",
                             index_root / "flat.faiss", root / "data/indexes/bm25_s512_o50/retrieval.sqlite",
                             threads=4)
    try:
        for key in ("model_id", "model_revision", "dimension", "pooling", "normalized",
                    "query_instruction", "tokenizer_config_sha256", "document_rendering"):
            if encoder.identity.get(key) != encoder_meta.get(key):
                raise ValueError("Encoder/index identity mismatch")
        if (metadata.get("status") != "complete" or metadata.get("normalized") is not True
                or metadata.get("dimension") != engine.stats["dimension"]
                or metadata.get("source_sha256") != engine.stats["chunks_sha256"]
                or metadata.get("vector_count") != engine.stats["chunk_count"]
                or sha256_file(embedding_path.parent / "vectors.npy") != engine.vector_manifest.get("vectors_file_sha256")):
            raise ValueError("Embedding/index identity mismatch")
        document_ids = {Path(row["relative_path"]).stem + "-" + row["sha256"][:12]
                        for row in inventory if row.get("status", "validated") == "validated"}
        if document_ids != set(engine.stats["document_ids"]):
            raise ValueError("Inventory/index identity mismatch")
        from report_search.reranker import MODEL_ID, MODEL_REVISION
        reranker_snapshot = cache / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
        reranker = BGEReranker(device=device, batch_size=2, cache_dir=cache,
                              local_model_path=reranker_snapshot)
        # Startup is paid once on the first search, outside warm stage timings.
        encoder._load_model()
        reranker._load_model()
        return DemoComponents(encoder, engine, reranker, SourceEvidenceStore(root / "data/processed"),
                              {**engine.stats, "encoder_model": encoder.identity["model_id"],
                               "reranker_model": reranker.identity["model_id"], "device": device})
    except BaseException:
        engine.close()
        raise


class _UnconfiguredModel:
    identity = {"backend": "unconfigured"}

    def complete_json(self, messages):
        raise ModelError("model_not_configured")


class DemoService:
    """Reuse thread-owned resources without retaining session request data.

    ``worker_factory(root_dir, device)`` optionally returns DemoComponents for
    tests or an alternate resource loader. It is always called on the worker.
    ``answer`` accepts only an unmodified result from this service's ``search``.
    """

    def __init__(self, root_dir, *, device="cpu", worker_factory=None):
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:\d+", device):
            raise DemoError("invalid_device")
        self.root_dir = Path(root_dir).resolve()
        self.device = device
        self._factory = worker_factory or _default_factory
        self._components = None
        self._worker_id = None
        self._closed = False
        self._cleanup_failed = False
        self._lock = threading.Lock()
        self._secret = secrets.token_bytes(32)
        self._jobs = Queue()
        # A ThreadPoolExecutor shuts itself down before application atexit
        # handlers, so its owner-thread cleanup can no longer be submitted.
        # This owned daemon remains available for close() to drain and join.
        self._thread = threading.Thread(target=self._worker_loop, name="finance-qa-demo", daemon=True)
        self._thread.start()

    def _worker_loop(self):
        self._worker_id = threading.get_ident()
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    break
                future, operation, args = job
                try:
                    if future.set_running_or_notify_cancel():
                        try:
                            future.set_result(operation(*args))
                        except BaseException as error:
                            future.set_exception(error)
                finally:
                    # Do not retain the last query, provider config, or result
                    # while this shared worker waits for the next request.
                    del future, operation, args, job
        finally:
            try:
                self._close_worker()
            except Exception:
                self._cleanup_failed = True

    def _submit(self, operation, *args):
        if threading.get_ident() == self._worker_id:
            raise DemoError("reentrant_request")
        with self._lock:
            if self._closed:
                raise DemoError("service_closed")
            future = Future()
            self._jobs.put((future, operation, args))
        try:
            return future.result()
        except DemoError:
            raise
        except Exception:
            raise DemoError("request_failed") from None

    def _resources(self):
        self._worker_id = threading.get_ident()
        if self._components is None:
            try:
                components = self._factory(self.root_dir, self.device)
                if not isinstance(components, DemoComponents):
                    raise TypeError("Expected DemoComponents")
                self._components = components
            except Exception:
                raise DemoError("startup_failed") from None
        return self._components

    def _signature(self, result):
        payload = json.dumps({"query": result["query"], "selected": result["selected"]},
                             sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def search(self, query):
        if not _valid_query(query):
            raise DemoError("invalid_query")
        return self._submit(self._search, query, time.perf_counter())

    def _search(self, query, queued_at):
        started = time.perf_counter()
        timings = {"queue_wait_ms": (started - queued_at) * 1000}
        partial = {"query": query, "candidates": [], "selected": [], "timings": timings, "stats": {}}
        stage, stage_started = "startup", started
        try:
            components = self._resources()
            timings["startup_ms"] = (time.perf_counter() - stage_started) * 1000
            partial["stats"] = deepcopy(components.stats or getattr(components.retrieval_engine, "stats", {}))
            stage, stage_started = "query_embedding", time.perf_counter()
            vector = components.encoder.encode_queries([query])[0]
            timings["query_embedding_ms"] = (time.perf_counter() - stage_started) * 1000
            stage, stage_started = "vector_search", time.perf_counter()
            partial["candidates"] = deepcopy(components.retrieval_engine.vector(vector, top_k=20))
            timings["vector_search_ms"] = (time.perf_counter() - stage_started) * 1000
            stage, stage_started = "reranking", time.perf_counter()
            partial["selected"] = components.reranker.rank_candidates(query, partial["candidates"], top_k=5)
            timings["reranking_ms"] = (time.perf_counter() - stage_started) * 1000
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            partial["_signature"] = self._signature(partial)
            return deepcopy(partial)
        except Exception:
            timings[stage + "_ms"] = (time.perf_counter() - stage_started) * 1000
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            raise DemoError(stage + "_failed", partial_result=deepcopy(partial)) from None

    def answer(self, search_result, *, threshold=None, generator_config=None,
               verifier_config=None, deadline_seconds=30):
        if any(config is not None and not isinstance(config, ModelConfig)
               for config in (generator_config, verifier_config)):
            raise DemoError("invalid_model_configuration")
        try:
            snapshot = deepcopy(search_result)
            signature = snapshot.get("_signature")
            if (not isinstance(signature, str) or not _valid_query(snapshot.get("query"))
                    or not hmac.compare_digest(signature, self._signature(snapshot))):
                raise ValueError("Invalid search result")
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise DemoError("invalid_search_result") from None
        return self._submit(self._answer, snapshot, threshold, generator_config,
                            verifier_config, deadline_seconds)

    def _answer(self, snapshot, threshold, generator_config, verifier_config, deadline_seconds):
        components = self._resources()
        # Fresh clients keep credentials and model choices local to this call.
        generator = ChatCompletionsClient(generator_config) if generator_config is not None else _UnconfiguredModel()
        effective_verifier = verifier_config if verifier_config is not None else generator_config
        verifier = ChatCompletionsClient(effective_verifier) if effective_verifier is not None else _UnconfiguredModel()
        try:
            return generate_answer(snapshot["query"], snapshot["selected"], threshold=threshold,
                                   source_store=components.source_store, generator=generator, verifier=verifier,
                                   deadline_seconds=deadline_seconds)
        except Exception:
            raise DemoError("answer_failed") from None

    def close(self):
        """Drain queued requests, then close SQLite on its owning worker."""
        if threading.get_ident() == self._worker_id:
            raise DemoError("reentrant_request")
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
        self._thread.join()
        if self._cleanup_failed:
            raise DemoError("close_failed")

    def _close_worker(self):
        if self._components is not None:
            self._components.retrieval_engine.close()
            self._components = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
