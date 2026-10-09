"""Answer quoted questions in one session with evidence checks and bounded recovery.

Startup verifies artifacts and loads models outside per-request deadlines.
Cache entries live in this process only; cached passages are verified again.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_search.llm_client import ModelConfig, ModelError
from report_search.query_policy import POLICY_VERSION, plan_query


def _refusal(reason, plan=None):
    return {"status": "abstain", "answer": "I'm not sure", "citations": [],
            "reference_verified": False, "semantic_supported": False, "reason": reason,
            "audit": {"generation_called": False, "semantic_verifier_called": False,
                      "query_plan": plan}}


def _build_pipeline(args, inventory, configs):
    from report_index.embedding import DenseEncoder, sha256_file
    from report_search.evidence_binding import SourceEvidenceStore
    from report_search.llm_client import ChatCompletionsClient
    from report_search.reliable_pipeline import ReliableQAPipeline
    from report_search.reranker import BGEReranker
    from report_search.retrieval import RetrievalEngine
    from report_search.runtime_support import RetrievalCache

    metadata = json.loads(args.embedding_manifest.read_text(encoding="utf-8"))
    encoder_meta = metadata["encoder"]
    encoder = DenseEncoder(model_id=encoder_meta["model_id"], device=args.device)
    engine = RetrievalEngine(args.chunks, args.vector_index, args.bm25_index, threads=4)
    try:
        for key in ("model_id", "model_revision", "dimension", "pooling", "normalized",
                    "query_instruction", "tokenizer_config_sha256", "document_rendering"):
            if encoder.identity.get(key) != encoder_meta.get(key):
                raise ValueError("Query encoder does not match indexed document encoder")
        if (metadata.get("status") != "complete" or not metadata.get("normalized")
                or metadata.get("dimension") != engine.stats["dimension"]
                or metadata.get("source_sha256") != engine.stats["chunks_sha256"]
                or sha256_file(args.embedding_manifest.parent / "vectors.npy") != engine.vector_manifest.get("vectors_file_sha256")):
            raise ValueError("Embedding metadata does not match active index")
        document_ids = {Path(row["relative_path"]).stem + "-" + row["sha256"][:12]
                        for row in inventory if row.get("status", "validated") == "validated"}
        if document_ids != set(engine.stats["document_ids"]):
            raise ValueError("Inventory does not match the active verified corpus")
        reranker = BGEReranker(device=args.device)
        binding = {"corpus": engine.stats, "encoder": encoder.identity,
                   "index": {"vector": engine.vector_manifest, "bm25": engine.bm25_stats},
                   "reranker": reranker.identity, "query_policy": {"version": POLICY_VERSION}}
        pipeline = ReliableQAPipeline(inventory=inventory, encoder=encoder, retrieval_engine=engine,
            reranker=reranker, binding=binding, threshold=args.threshold,
            expansion_threshold=args.expansion_threshold, enable_expansion=args.enable_expansion,
            cache=RetrievalCache(), source_store=SourceEvidenceStore(args.processed),
            generator=ChatCompletionsClient(configs[0]), verifier=ChatCompletionsClient(configs[1]),
            deadline_seconds=args.deadline_seconds, latency_warning_seconds=args.latency_warning_seconds)
        encoder._load_model()
        reranker._load_model()
        return pipeline, engine
    except BaseException:
        engine.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+", help="One or more individually quoted questions")
    parser.add_argument("--threshold", type=float, help="Explicit raw reranker cutoff; default refuses")
    parser.add_argument("--enable-expansion", action="store_true", help="Opt in to bounded lexical recovery")
    parser.add_argument("--expansion-threshold", type=float, help="Separately validated recovery cutoff")
    parser.add_argument("--deadline-seconds", type=float, default=30.0)
    parser.add_argument("--latency-warning-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0", help="cpu or one explicit CUDA device")
    defaults = {"manifest": "data/manifest.json", "processed": "data/processed",
        "chunks": "data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl",
        "vector-index": "data/indexes/bge-m3/table_preserving_s512_o50/flat.faiss",
        "embedding-manifest": "data/indexes/bge-m3/table_preserving_s512_o50/embeddings/embedding_manifest.json",
        "bm25-index": "data/indexes/bm25_s512_o50/retrieval.sqlite"}
    for name, default in defaults.items():
        parser.add_argument("--" + name, type=Path, default=Path(default))
    parser.add_argument("--output", type=Path, help="Write new UTF-8 JSON file; default prints JSON")
    args = parser.parse_args(argv)
    for name in ("threshold", "expansion_threshold", "deadline_seconds", "latency_warning_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or (name.endswith("seconds") and value <= 0)):
            parser.error(f"--{name.replace('_', '-')} must be finite" + (" and positive" if name.endswith("seconds") else ""))
    if args.enable_expansion and args.expansion_threshold is None:
        parser.error("--enable-expansion requires a separate --expansion-threshold")
    if not re.fullmatch(r"cpu|cuda:\d+", args.device):
        parser.error("--device must be cpu or cuda:N")
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must be a new file in an existing directory")
    results, plans = [None] * len(args.queries), [None] * len(args.queries)
    try:
        inventory = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        for position, query in enumerate(args.queries):
            if not query.strip() or len(query) > 10_000:
                results[position] = _refusal("invalid_query")
                continue
            plan = plan_query(query, inventory)
            plans[position] = plan
            if plan["coverage"]["status"] == "unavailable":
                reason = "company_not_in_corpus" if plan["coverage"]["missing_tickers"] else "report_not_in_corpus"
                results[position] = _refusal(reason, plan)
            elif args.threshold is None:
                results[position] = _refusal("unconfigured_threshold", plan)
    except (OSError, ValueError, TypeError, KeyError):
        results = [_refusal("invalid_query_or_inventory") for _ in args.queries]
    pending = [position for position, result in enumerate(results) if result is None]
    startup_seconds, pipeline, engine = 0.0, None, None
    if pending:
        try:
            configs = ModelConfig.from_env(), ModelConfig.from_env(verifier=True)
            started = time.monotonic()
            try:
                with redirect_stdout(sys.stderr):
                    pipeline, engine = _build_pipeline(args, inventory, configs)
            finally:
                startup_seconds = time.monotonic() - started
            for position in pending:
                with redirect_stdout(sys.stderr):
                    results[position] = pipeline.answer(args.queries[position])
        except (ModelError, OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
            reason = str(error) if isinstance(error, ModelError) else "startup_or_runtime_failed"
            for position in pending:
                if results[position] is None:
                    results[position] = _refusal(reason, plans[position])
        finally:
            if engine is not None:
                engine.close()
    report = {"startup_seconds": startup_seconds,
        "timing_semantics": "Artifact verification and model loading precede request deadlines; cache is in-process only.",
        "cache_stats": pipeline.cache.stats if pipeline is not None else None,
        "results": [{"query": query, **result} for query, result in zip(args.queries, results)]}
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    else:
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if all(result["status"] == "answer" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
