"""Pinned FlagEmbedding cross-encoder with lossless financial chunk inputs.

The reranker receives only a question and the existing rendered source chunk.
Every complete pair is checked against the model's 8,192-token limit before
weights are loaded or inference starts. It never splits a table, drops a row,
truncates a passage, or uses benchmark answers as a model input.
"""
from __future__ import annotations

import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Sequence

from report_index.embedding import RENDERING_VERSION, render_document


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_WEIGHTS_SHA256 = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
MAX_PAIR_TOKENS = 8192
TOKENIZER_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "sentencepiece.bpe.model",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_scores(scores: Sequence[float], count: int) -> list[float]:
    try:
        values = list(scores)
    except TypeError as exc:
        raise ValueError("Reranker output must contain one finite numeric score per pair") from exc
    if len(values) != count:
        raise ValueError(f"Expected {count} reranker scores, got {len(values)}")
    if any(isinstance(score, bool) or not isinstance(score, Real)
           or not math.isfinite(float(score)) for score in values):
        raise ValueError("Reranker output contains a nonnumeric or nonfinite score")
    return [float(score) for score in values]


def apply_scores(candidates: Sequence[dict], scores: Sequence[float], top_k: int = 5) -> list[dict]:
    """Select by raw logits, then original vector rank, then stable chunk ID.

    Source payloads, including all table rows and citations, are deep-copied
    intact. Only retrieval scores/ranks and the approach marker are changed.
    Missing original ranks use input order, which must be vector order.
    """
    _positive_integer(top_k, "top_k")
    values = _finite_scores(scores, len(candidates))
    identifiers = set()
    prepared = []
    for position, (candidate, score) in enumerate(zip(candidates, values), start=1):
        if not isinstance(candidate, dict):
            raise ValueError("Candidates must be chunk dictionaries")
        chunk_id = candidate.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Each candidate must have a nonempty chunk_id")
        if chunk_id in identifiers:
            raise ValueError(f"Duplicate candidate chunk_id: {chunk_id}")
        identifiers.add(chunk_id)
        vector_rank = candidate.get("vector_rank", candidate.get("rank", position))
        _positive_integer(vector_rank, "original vector rank")
        vector_score = candidate.get("vector_score", candidate.get("score"))
        if vector_score is not None:
            vector_score = _finite_scores([vector_score], 1)[0]
        prepared.append((candidate, score, vector_rank, vector_score))
    prepared.sort(key=lambda item: (-item[1], item[2], item[0]["chunk_id"]))
    results = []
    for rank, (candidate, score, vector_rank, vector_score) in enumerate(prepared[:top_k], start=1):
        result = copy.deepcopy(candidate)
        result.update(score=score, rank=rank, retrieval_approach="vector_reranker",
                      reranker_score=score, vector_score=vector_score, vector_rank=vector_rank)
        results.append(result)
    return results


class PairTokenLimitError(ValueError):
    """Complete preflight audit attached when any question/passage overflows."""

    def __init__(self, audit: dict):
        self.audit = audit
        super().__init__(f"Refusing to truncate {audit['overflow_count']} query/passage pairs above "
                         f"{audit['max_model_tokens']} model tokens: {audit['overflow_examples'][:5]}")


def _bounded_batch(forward, batch_size: int, oom_type, clear_cache) -> tuple[list[float], int]:
    """Retry CUDA allocation failures only; a one-pair failure must terminate."""
    while True:
        try:
            return forward(batch_size), batch_size
        except oom_type:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            clear_cache()


def _safe_flag_single_gpu(flag, sentence_pairs, *, batch_size=None, query_max_length=None,
                          max_length=None, normalize=None, device=None, **kwargs):
    """FlagReranker single-device inference without its unbounded error retry.

    FlagEmbedding 1.3.5's implementation catches every RuntimeError and can
    reduce batch size to zero. This adapter retains its tokenizer/model and
    raw-logit scoring, while retrying only CUDA OOM with a positive batch size.
    """
    import torch
    batch_size = flag.batch_size if batch_size is None else batch_size
    max_length = flag.max_length if max_length is None else max_length
    query_max_length = flag.query_max_length if query_max_length is None else query_max_length
    normalize = flag.normalize if normalize is None else normalize
    device = flag.target_devices[0] if device is None else device
    if normalize:
        raise ValueError("The lossless reranker adapter returns raw logits only")
    if kwargs:
        raise ValueError(f"Unsupported extra reranker tokenization arguments: {sorted(kwargs)}")
    if not sentence_pairs:
        return []
    if flag.use_fp16 and device != "cpu":
        flag.model.half()
    flag.model.to(device)
    flag.model.eval()
    # Passing both text sequences preserves the model's complete paired-input
    # special-token format. Never let Transformers truncate either sequence.
    tokenized = flag.tokenizer([pair[0] for pair in sentence_pairs],
                               [pair[1] for pair in sentence_pairs],
                               add_special_tokens=True, truncation=False, padding=False)
    items = [{key: values[index] for key, values in tokenized.items()}
             for index in range(len(sentence_pairs))]
    if any(len(item["input_ids"]) > max_length for item in items):
        raise ValueError("Complete reranker input exceeds max_length; refusing to truncate")
    order = sorted(range(len(items)), key=lambda index: (-len(items[index]["input_ids"]), index))
    ordered = [items[index] for index in order]
    ordered_scores = []
    start = 0
    with torch.inference_mode():
        while start < len(ordered):
            def forward(size):
                inputs = flag.tokenizer.pad(ordered[start:start + size], padding=True,
                                            return_tensors="pt").to(device)
                return flag.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

            scores, batch_size = _bounded_batch(forward, batch_size, torch.cuda.OutOfMemoryError,
                                                torch.cuda.empty_cache)
            expected_count = min(batch_size, len(ordered) - start)
            ordered_scores.extend(_finite_scores(scores, expected_count))
            start += expected_count
    scores = [0.0] * len(items)
    for original_index, score in zip(order, ordered_scores):
        scores[original_index] = score
    return scores


class BGEReranker:
    """Single-device, lazy-weight adapter for the immutable BGE reranker snapshot.

    Construction loads only tokenizer/configuration files. Preflight can run
    without model weights. ``score_pairs`` verifies the official safetensors
    SHA256 and loads FlagReranker only after all supplied pairs pass preflight.
    The caller can impose a CUDA memory budget in its inference subprocess.
    """

    def __init__(self, model_id: str = MODEL_ID, revision: str | None = None,
                 device: str = "cuda:0", batch_size: int = 4,
                 cache_dir: str | Path | None = None,
                 local_model_path: str | Path | None = None):
        if model_id != MODEL_ID:
            raise ValueError(f"Unsupported reranker: {model_id}; expected {MODEL_ID}")
        revision = revision or MODEL_REVISION
        if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("revision must be an immutable 40-character Hugging Face commit SHA")
        if revision.lower() != MODEL_REVISION:
            raise ValueError("This adapter verifies the pinned model revision only")
        if not re.fullmatch(r"cpu|cuda:\d+", device):
            raise ValueError("device must select cpu or one explicit CUDA device, e.g. cuda:0")
        _positive_integer(batch_size, "batch_size")
        self.model_id, self.revision = model_id, revision.lower()
        self.max_tokens = MAX_PAIR_TOKENS
        # FlagEmbedding otherwise silently caps queries at 3/4 of max_length.
        self.query_max_length = self.max_tokens
        self.device, self.batch_size = device, batch_size
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parents[2] / ".cache" / "huggingface")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_model_path = Path(local_model_path).resolve() if local_model_path else None
        self._model = None
        self._snapshot_path = self._snapshot(weights=False)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self._snapshot_path), local_files_only=True, trust_remote_code=False)
        self._tokenizer_hashes = {name: _sha256(self._snapshot_path / name) for name in TOKENIZER_FILES}
        self.identity = {
            "model_id": self.model_id, "model_revision": self.revision,
            "snapshot_revision": self._snapshot_path.name,
            "model_weights_file": "model.safetensors",
            "model_weights_sha256": MODEL_WEIGHTS_SHA256,
            "weight_verification": "sha256_checked_before_model_load",
            "implementation_sha256": _sha256(Path(__file__)),
            "max_tokens": self.max_tokens, "query_max_length": self.query_max_length,
            "special_tokens_included": True, "truncation_policy": "reject_complete_pair",
            "device": self.device, "batch_size": self.batch_size,
            "inference_precision": "float16" if self.device.startswith("cuda:") else "float32",
            "inference_backend": "FlagReranker_with_bounded_cuda_oom_retry",
            "score_type": "raw_logit", "normalized": False,
            "query_instruction": "", "passage_instruction": "",
            "document_rendering": RENDERING_VERSION,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_files_sha256": self._tokenizer_hashes,
            "tokenizer_config_sha256": _json_hash(self._tokenizer_hashes),
            "packages": {name: _package_version(name) for name in
                         ("FlagEmbedding", "transformers", "torch", "tokenizers", "safetensors")},
        }

    def _snapshot(self, *, weights: bool) -> Path:
        if self._local_model_path is not None:
            snapshot = self._local_model_path
        else:
            from huggingface_hub import snapshot_download
            patterns = list(TOKENIZER_FILES)
            if weights:
                patterns.append("model.safetensors")
            snapshot = Path(snapshot_download(repo_id=self.model_id, revision=self.revision,
                                             cache_dir=str(self.cache_dir), allow_patterns=patterns))
        if not snapshot.is_dir() or snapshot.name.lower() != self.revision:
            raise ValueError("Local model snapshot directory must exist and match the pinned revision")
        required = list(TOKENIZER_FILES) + (["model.safetensors"] if weights else [])
        missing = [name for name in required if not (snapshot / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete model snapshot; missing {missing}")
        return snapshot

    def _pair_lengths(self, pairs: Sequence[tuple[str, str]]) -> list[dict]:
        for pair in pairs:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or any(not isinstance(text, str) or not text.strip() for text in pair)):
                raise ValueError("Each reranker pair must contain a nonempty query and passage string")
        lengths = []
        for start in range(0, len(pairs), 64):
            batch = pairs[start:start + 64]
            options = dict(add_special_tokens=False, truncation=False, padding=False,
                           return_attention_mask=False, return_token_type_ids=False)
            query_ids = self.tokenizer([pair[0] for pair in batch], **options)["input_ids"]
            passage_ids = self.tokenizer([pair[1] for pair in batch], **options)["input_ids"]
            if len(query_ids) != len(batch) or len(passage_ids) != len(batch):
                raise ValueError("Tokenizer returned the wrong number of inputs")
            for query, passage in zip(query_ids, passage_ids):
                # This is the same special-token assembly used by FlagReranker's
                # prepare_for_model, with no truncation permitted here.
                complete = self.tokenizer.build_inputs_with_special_tokens(query, passage)
                lengths.append({"query_tokens": len(query), "passage_tokens": len(passage),
                                "pair_tokens": len(complete)})
        return lengths

    def pair_token_lengths(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        return [item["pair_tokens"] for item in self._pair_lengths(pairs)]

    def preflight_pairs(self, pairs: Sequence[tuple[str, str]]) -> dict:
        """Audit all pairs and reject any overflow before model inference."""
        lengths = self._pair_lengths(pairs)
        overflowing = [{"pair_index": index, **item} for index, item in enumerate(lengths)
                       if item["pair_tokens"] > self.max_tokens
                       or item["query_tokens"] > self.query_max_length]
        audit = {
            "status": "failed" if overflowing else "passed", "pair_count": len(lengths),
            "max_model_tokens": self.max_tokens, "query_max_length": self.query_max_length,
            "max_pair_tokens": max((item["pair_tokens"] for item in lengths), default=0),
            "max_query_tokens": max((item["query_tokens"] for item in lengths), default=0),
            "max_passage_tokens": max((item["passage_tokens"] for item in lengths), default=0),
            "mean_pair_tokens": sum(item["pair_tokens"] for item in lengths) / len(lengths) if lengths else 0.0,
            "overflow_count": len(overflowing), "overflow_examples": overflowing[:20],
            "special_tokens_included": True, "truncated_pairs": 0,
        }
        if overflowing:
            raise PairTokenLimitError(audit)
        return audit

    def _load_model(self):
        if self._model is None:
            model_path = self._snapshot(weights=True)
            actual_hashes = {name: _sha256(model_path / name) for name in TOKENIZER_FILES}
            if actual_hashes != self._tokenizer_hashes:
                raise ValueError("Tokenizer/configuration changed after preflight initialization")
            if _sha256(model_path / "model.safetensors") != MODEL_WEIGHTS_SHA256:
                raise ValueError("Reranker weights do not match the pinned official SHA256")
            import torch
            if self.device.startswith("cuda:") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; explicitly choose device='cpu'")
            from FlagEmbedding import FlagReranker

            class LosslessFlagReranker(FlagReranker):
                compute_score_single_gpu = _safe_flag_single_gpu

            self._model = LosslessFlagReranker(
                str(model_path), use_fp16=self.device.startswith("cuda:"), devices=[self.device],
                batch_size=self.batch_size, max_length=self.max_tokens,
                query_max_length=self.query_max_length, normalize=False,
                query_instruction_for_rerank=None, passage_instruction_for_rerank=None,
                trust_remote_code=False, cache_dir=str(self.cache_dir))
        return self._model

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return one raw higher-is-better relevance logit per unchanged pair."""
        pairs = list(pairs)
        self.preflight_pairs(pairs)
        if not pairs:
            return []
        model = self._load_model()
        scores = model.compute_score(
            pairs, batch_size=self.batch_size, max_length=self.max_tokens,
            query_max_length=self.query_max_length, normalize=False)
        return _finite_scores(scores, len(pairs))

    def rank_candidates(self, query: str, candidates: Sequence[dict], top_k: int = 5) -> list[dict]:
        _positive_integer(top_k, "top_k")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string")
        # Validate IDs/ranks before expensive inference and never change payloads.
        apply_scores(candidates, [0.0] * len(candidates), top_k)
        pairs = [(query, render_document(candidate)) for candidate in candidates]
        return apply_scores(candidates, self.score_pairs(pairs), top_k)
