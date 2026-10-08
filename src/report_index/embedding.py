"""Pinned dense encoders and resumable, lossless-input embedding shards.

Token limits are checked with the model tokenizer, after adding document
metadata and special tokens. No passage, table row, or fallback page is ever
silently truncated or omitted. Heavy ML imports are lazy so audits and tests
can run without loading model weights.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Iterator

import numpy as np


MODEL_SPECS = {
    "BAAI/bge-m3": {"dimension": 1024, "max_tokens": 8192,
                    "revision": "5617a9f61b028005a4858fdac845db406aefb181",
                    "query_instruction": ""},
    "BAAI/bge-large-en-v1.5": {"dimension": 1024, "max_tokens": 512,
                              "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                              "query_instruction": "Represent this sentence for searching relevant passages: "},
}
RENDERING_VERSION = "company-ticker-fiscal-year-v1"
TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt",
                   "added_tokens.json"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def render_document(chunk: dict) -> str:
    """Deterministic retrieval context; the source text itself is unchanged."""
    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Chunk {chunk.get('chunk_id')!r} has no nonempty text")
    metadata = []
    if chunk.get("company"):
        metadata.append(f"Company: {chunk['company']}")
    if chunk.get("ticker"):
        metadata.append(f"Ticker: {chunk['ticker']}")
    if chunk.get("fiscal_year") is not None:
        metadata.append(f"Annual report fiscal year: FY{chunk['fiscal_year']}")
    return "\n".join(metadata) + "\n\n" + text if metadata else text


def normalized_float32(vectors, *, count: int, dimension: int) -> np.ndarray:
    """Reject invalid encoder output and normalize in full precision."""
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (count, dimension):
        raise ValueError(f"Expected embeddings {(count, dimension)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Embedding output contains NaN or infinity")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Embedding output contains a zero or invalid norm")
    return np.ascontiguousarray(result / norms, dtype=np.float32)


class DenseEncoder:
    """Single-device FlagEmbedding adapter with immutable model revisions."""

    def __init__(self, model_id: str = "BAAI/bge-m3", revision: str | None = None,
                 device: str = "cuda:0", batch_size: int = 8,
                 cache_dir: str | Path | None = None,
                 local_model_path: str | Path | None = None):
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unsupported model: {model_id}; choose {list(MODEL_SPECS)}")
        spec = MODEL_SPECS[model_id]
        revision = revision or spec["revision"]
        if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("revision must be an immutable 40-character Hugging Face commit SHA")
        if not re.fullmatch(r"cpu|cuda:\d+", device):
            raise ValueError("device must explicitly select cpu or one CUDA device, e.g. cuda:0")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.model_id, self.revision = model_id, revision.lower()
        self.dimension, self.max_tokens = spec["dimension"], spec["max_tokens"]
        self.query_instruction = spec["query_instruction"]
        self.device, self.batch_size = device, batch_size
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parents[2] / ".cache" / "huggingface")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_model_path = Path(local_model_path).resolve() if local_model_path else None
        self._model = None
        self._snapshot_path = self._snapshot(weights=False)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self._snapshot_path), local_files_only=True,
                                                      trust_remote_code=False)
        file_hashes = {name: sha256_file(self._snapshot_path / name) for name in TOKENIZER_FILES
                       if (self._snapshot_path / name).is_file()}
        self.identity = {
            "model_id": self.model_id, "model_revision": self.revision,
            "dimension": self.dimension, "max_tokens": self.max_tokens,
            "device": self.device, "batch_size": self.batch_size,
            "implementation_sha256": sha256_file(Path(__file__)),
            "pooling": "cls", "normalized": True, "output_dtype": "float32",
            "inference_precision": "float16" if device.startswith("cuda:") else "float32",
            "query_instruction": self.query_instruction,
            "document_rendering": RENDERING_VERSION,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_files_sha256": file_hashes,
            "tokenizer_config_sha256": _json_hash(file_hashes),
            "special_tokens_included": True,
            "packages": {name: _package_version(name) for name in
                         ("FlagEmbedding", "transformers", "torch", "tokenizers", "numpy")},
        }

    def _snapshot(self, *, weights: bool) -> Path:
        if self._local_model_path is not None:
            if not self._local_model_path.is_dir():
                raise ValueError(f"Missing local model snapshot: {self._local_model_path}")
            if (re.fullmatch(r"[a-fA-F0-9]{40}", self._local_model_path.name)
                    and self._local_model_path.name.lower() != self.revision):
                raise ValueError("Local snapshot directory does not match the requested model revision")
            return self._local_model_path
        from huggingface_hub import snapshot_download
        patterns = list(TOKENIZER_FILES)
        if weights:
            # M3 publishes PyTorch weights. Selecting this format avoids duplicate
            # ONNX/safetensors downloads; both supported models publish it.
            patterns += ["pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt"]
        return Path(snapshot_download(repo_id=self.model_id, revision=self.revision,
                                      cache_dir=str(self.cache_dir), allow_patterns=patterns))

    def _load_model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel, FlagModel
            import torch
            if self.device.startswith("cuda:") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; explicitly choose --device cpu")
            model_path = self._snapshot(weights=True)
            kwargs = dict(use_fp16=self.device.startswith("cuda:"), devices=[self.device],
                          normalize_embeddings=True, pooling_method="cls", batch_size=self.batch_size,
                          query_max_length=self.max_tokens, passage_max_length=self.max_tokens,
                          trust_remote_code=False, cache_dir=str(self.cache_dir),
                          query_instruction_for_retrieval=None)
            if self.model_id == "BAAI/bge-m3":
                self._model = BGEM3FlagModel(str(model_path), return_dense=True,
                                            return_sparse=False, return_colbert_vecs=False, **kwargs)
            else:
                self._model = FlagModel(str(model_path), **kwargs)
        return self._model

    def token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self.tokenizer(texts, add_special_tokens=True, truncation=False,
                                 padding=False, return_attention_mask=False,
                                 return_token_type_ids=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        lengths = self.token_lengths(texts)
        overflowing = [(i, length) for i, length in enumerate(lengths) if length > self.max_tokens]
        if overflowing:
            raise ValueError(f"Refusing to truncate {len(overflowing)} inputs above {self.max_tokens} model tokens: "
                             f"{overflowing[:5]}")
        model = self._load_model()
        kwargs = dict(batch_size=self.batch_size, max_length=self.max_tokens)
        if self.model_id == "BAAI/bge-m3":
            encoded = model.encode(texts, return_dense=True, return_sparse=False,
                                   return_colbert_vecs=False, **kwargs)["dense_vecs"]
        else:
            encoded = model.encode(texts, **kwargs)
        return normalized_float32(encoded, count=len(texts), dimension=self.dimension)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_instruction + text for text in texts])


class TokenLimitError(ValueError):
    def __init__(self, audit: dict):
        self.audit = audit
        super().__init__(f"Refusing to truncate or skip {audit['overflow_count']} chunks above "
                         f"{audit['max_model_tokens']} model tokens. See preflight.json; "
                         f"examples: {audit['overflow_examples'][:3]}")


def _iter_chunks(path: Path) -> Iterator[tuple[int, int, dict]]:
    with path.open("rb") as handle:
        ordinal = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                raise ValueError(f"Blank JSONL line in {path} at byte {offset}; vector IDs require one chunk per line")
            try:
                chunk = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid JSON in {path} at byte {offset}") from exc
            if not isinstance(chunk, dict) or not chunk.get("chunk_id") or not chunk.get("document_id"):
                raise ValueError(f"Missing chunk_id/document_id in {path} at byte {offset}")
            yield ordinal, offset, chunk
            ordinal += 1


def _token_lengths(encoder, texts: list[str]) -> list[int]:
    if hasattr(encoder, "token_lengths"):
        lengths = encoder.token_lengths(texts)
    else:
        batch = encoder.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False,
                                  return_attention_mask=False, return_token_type_ids=False)
        lengths = [len(row) for row in batch["input_ids"]]
    if len(lengths) != len(texts) or any(not isinstance(n, int) or n < 1 for n in lengths):
        raise ValueError("Tokenizer returned invalid lengths")
    return lengths


def preflight_chunks(chunks_path: str | Path, output_dir: str | Path, encoder,
                     on_progress: Callable[[dict], None] | None = None) -> dict:
    """Audit every input before encoding; write and raise on any token overflow."""
    chunks_path, output_dir = Path(chunks_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(chunks_path)
    identity = {"source_sha256": source_sha, "encoder": encoder.identity,
                "document_rendering": RENDERING_VERSION}
    fingerprint = _json_hash(identity)
    audit_path = output_dir / "preflight.json"
    if audit_path.exists():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            if existing["overflow_count"]:
                raise TokenLimitError(existing)
            if on_progress:
                on_progress({"phase": "preflight", "count": existing["chunk_count"], "resumed": True})
            return existing
    counts, kinds, overflows = Counter(), Counter(), Counter()
    chunk_ids, document_ids = set(), set()
    overflow_examples, pending = [], []
    max_length, total_tokens = 0, 0

    def inspect_batch():
        nonlocal max_length, total_tokens
        lengths = _token_lengths(encoder, [render_document(chunk) for chunk in pending])
        for chunk, length in zip(pending, lengths):
            counts["chunks"] += 1
            kinds[chunk.get("kind", "unknown")] += 1
            document_ids.add(chunk["document_id"])
            max_length = max(max_length, length)
            total_tokens += length
            if length > encoder.max_tokens:
                overflows[chunk.get("kind", "unknown")] += 1
                if len(overflow_examples) < 30:
                    overflow_examples.append({key: chunk.get(key) for key in
                                              ("chunk_id", "document_id", "kind", "source_pages")}
                                             | {"model_tokens": length})
        if on_progress:
            on_progress({"phase": "preflight", "count": counts["chunks"],
                         "max_actual_model_tokens": max_length, "overflow_count": sum(overflows.values())})
        pending.clear()

    for _, _, chunk in _iter_chunks(chunks_path):
        if chunk["chunk_id"] in chunk_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
        chunk_ids.add(chunk["chunk_id"])
        pending.append(chunk)
        if len(pending) == 128:
            inspect_batch()
    if pending:
        inspect_batch()
    if not counts["chunks"]:
        raise ValueError("Cannot embed an empty chunk file")
    if sha256_file(chunks_path) != source_sha:
        raise ValueError("Chunk input changed during token preflight")
    audit = {"schema_version": "1.0", "status": "failed" if overflows else "passed",
             "fingerprint": fingerprint, "source_chunks_path": str(chunks_path), **identity,
             "document_count": len(document_ids), "chunk_count": counts["chunks"],
             "kinds": dict(kinds), "max_model_tokens": encoder.max_tokens,
             "max_actual_model_tokens": max_length, "total_model_tokens": total_tokens,
             "mean_model_tokens": total_tokens / counts["chunks"],
             "overflow_count": sum(overflows.values()), "overflow_by_kind": dict(overflows),
             "overflow_examples": overflow_examples, "truncated_chunks": 0, "skipped_chunks": 0}
    _write_json(audit_path, audit)
    if overflows:
        raise TokenLimitError(audit)
    return audit


def _within(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes output directory: {relative}")
    return result


def _validate_shard(root: Path, shard: dict, dimension: int) -> None:
    vectors_path = _within(root, shard["vectors_path"])
    metadata_path = _within(root, shard["metadata_path"])
    if (not vectors_path.is_file() or not metadata_path.is_file()
            or sha256_file(vectors_path) != shard["vectors_sha256"]
            or sha256_file(metadata_path) != shard["metadata_sha256"]):
        raise ValueError(f"Embedding shard missing or modified: {shard['vectors_path']}")
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (shard["count"], dimension) or vectors.dtype != np.float32:
        raise ValueError(f"Invalid embedding shard shape/dtype: {vectors_path}")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError(f"Embedding shard has invalid or unnormalized vectors: {vectors_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata_count = sum(1 for line in handle if line.strip())
    if metadata_count != shard["count"]:
        raise ValueError(f"Embedding shard metadata count mismatch: {metadata_path}")


def build_embeddings(chunks_path: str | Path, output_dir: str | Path, encoder, shard_size: int = 512,
                     on_progress: Callable[[dict], None] | None = None) -> dict:
    """Create normalized float32 NPY shards with stable vector-to-source mappings.

    A checkpoint is committed only after both shard files are durably written.
    Re-running verifies hashes and reuses complete shards; source, tokenizer,
    model, precision, or shard-size changes require a new output directory.
    """
    run_started = time.perf_counter()
    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    chunks_path, output_dir = Path(chunks_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, manifest_path = output_dir / "embedding_checkpoint.json", output_dir / "embedding_manifest.json"
    source_sha = sha256_file(chunks_path)
    identity = {"source_sha256": source_sha, "encoder": encoder.identity,
                "document_rendering": RENDERING_VERSION, "shard_size": shard_size}
    fingerprint = _json_hash(identity)
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Embedding output identity differs from this input/model/configuration; use a new output directory")
    audit = preflight_chunks(chunks_path, output_dir, encoder, on_progress)
    if audit["source_sha256"] != source_sha:
        raise ValueError("Chunk input changed before embedding")
    if checkpoint is None:
        checkpoint = {"schema_version": "1.0", "fingerprint": fingerprint, **identity,
                      "shards": [], "vector_count": 0, "encoding_seconds": 0.0}
        _write_json(checkpoint_path, checkpoint)
    expected_start = 0
    for shard in checkpoint["shards"]:
        if shard["start_index"] != expected_start:
            raise ValueError("Embedding checkpoint contains non-contiguous shards")
        _validate_shard(output_dir, shard, encoder.dimension)
        expected_start += shard["count"]
    if expected_start != checkpoint["vector_count"] or expected_start > audit["chunk_count"]:
        raise ValueError("Embedding checkpoint count mismatch")
    if on_progress:
        on_progress({"phase": "embedding", "count": expected_start,
                     "total": audit["chunk_count"], "resumed": bool(expected_start)})
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    pending = []

    def write_shard():
        shard_started = time.perf_counter()
        texts = [render_document(record[2]) for record in pending]
        # Validate again at the encoding boundary, including for pluggable encoders.
        lengths = _token_lengths(encoder, texts)
        if max(lengths) > encoder.max_tokens:
            raise ValueError("Input exceeds the model token limit after successful preflight; input may have changed")
        vectors = normalized_float32(encoder.encode_documents(texts), count=len(texts), dimension=encoder.dimension)
        number = len(checkpoint["shards"])
        vectors_path = shard_dir / f"vectors-{number:06d}.npy"
        metadata_path = shard_dir / f"chunks-{number:06d}.jsonl"
        vectors_tmp = vectors_path.with_suffix(".npy.tmp")
        with vectors_tmp.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        metadata_tmp = metadata_path.with_suffix(".jsonl.tmp")
        with metadata_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for (ordinal, offset, chunk), text, length in zip(pending, texts, lengths):
                record = {key: chunk.get(key) for key in ("chunk_id", "document_id", "source_file", "ticker",
                                                        "fiscal_year", "company", "kind", "source_pages", "strategy_id")}
                record.update(vector_id=ordinal, source_jsonl_offset=offset,
                              source_chunk_sha256=_json_hash(chunk), model_token_count=length,
                              rendered_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(vectors_tmp, vectors_path)
        os.replace(metadata_tmp, metadata_path)
        shard = {"start_index": pending[0][0], "count": len(pending),
                 "vectors_path": vectors_path.relative_to(output_dir).as_posix(),
                 "metadata_path": metadata_path.relative_to(output_dir).as_posix(),
                 "vectors_sha256": sha256_file(vectors_path), "metadata_sha256": sha256_file(metadata_path),
                 "encoding_and_write_seconds": time.perf_counter() - shard_started}
        checkpoint["shards"].append(shard)
        checkpoint["vector_count"] += len(pending)
        checkpoint["encoding_seconds"] = checkpoint.get("encoding_seconds", 0) + shard["encoding_and_write_seconds"]
        _write_json(checkpoint_path, checkpoint)
        if on_progress:
            on_progress({"phase": "embedding", "count": checkpoint["vector_count"],
                         "total": audit["chunk_count"], "shard": number,
                         "elapsed_seconds": time.perf_counter() - run_started})
        pending.clear()

    for record in _iter_chunks(chunks_path):
        if record[0] < expected_start:
            continue
        pending.append(record)
        if len(pending) == shard_size:
            write_shard()
    if pending:
        write_shard()
    if sha256_file(chunks_path) != source_sha:
        raise ValueError("Chunk input changed during embedding; no complete manifest written")
    if checkpoint["vector_count"] != audit["chunk_count"]:
        raise ValueError("Final vector count does not match input chunk count")
    manifest = {"schema_version": "1.0", "status": "complete", "fingerprint": fingerprint,
                **identity, "source_chunks_path": str(chunks_path),
                "dimension": encoder.dimension, "dtype": "float32", "normalized": True,
                "document_count": audit["document_count"], "chunk_count": audit["chunk_count"],
                "vector_count": checkpoint["vector_count"], "shards": checkpoint["shards"],
                "max_actual_model_tokens": audit["max_actual_model_tokens"],
                "mean_model_tokens": audit["mean_model_tokens"],
                "encoding_and_write_seconds": checkpoint.get("encoding_seconds", 0),
                "timing_semantics": {
                    "encoding_and_write_seconds": "Sum of committed shard times; includes token checks, encoding, shard writes, and first-shard model initialization/weight download when needed",
                    "last_run_seconds": "Wall time of this invocation, including input hashing, preflight, resume validation, and any newly encoded shards",
                },
                "last_run_seconds": time.perf_counter() - run_started,
                "truncated_chunks": 0, "skipped_chunks": 0,
                "preflight_path": "preflight.json", "preflight_sha256": sha256_file(output_dir / "preflight.json")}
    _write_json(manifest_path, manifest)
    return manifest
