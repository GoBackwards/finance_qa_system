"""Compare frozen vector Top 5 with BGE reranking of its unchanged Top 20 pool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from report_index.embedding import render_document
from report_index.faiss_index import sha256_file
from report_search.benchmark import validate_benchmark
from report_search.experiment import json_hash, write_json
from report_search.rerank_evaluation import evaluate_pair, summarize
from report_search.reranker import BGEReranker, apply_scores


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare_candidates(benchmark_path, retrieval_run, output):
    """Extract only source payloads and query text from the verified vector run.

    The frozen exact-vector Top 100 ranking already contains its Top 20 prefix.
    Reusing it gives both approaches exactly the same candidates and preserves
    the earlier baseline, including its cutoff-tie policy. No gold labels enter
    the candidate file or model input.
    """
    benchmark = read_json(benchmark_path)
    source = read_json(retrieval_run / "comparison.json")
    benchmark_sha = sha256_file(benchmark_path)
    if source["benchmark_provenance"]["benchmark_sha256"] != benchmark_sha:
        raise ValueError("Vector rankings refer to a different frozen benchmark")
    source_path = retrieval_run / "per_query.jsonl"
    if sha256_file(source_path) != source["per_query_sha256"]:
        raise ValueError("Vector ranking file checksum changed")
    query_manifest = read_json(retrieval_run / "query_manifest.json")
    if (query_manifest != source["benchmark_provenance"]["query_manifest"]
            or sha256_file(retrieval_run / "query_vectors.npy") != query_manifest["vectors_file_sha256"]):
        raise ValueError("Vector query embeddings differ from the measured retrieval run")
    questions = benchmark["questions"]
    expected_queries = [(q["question_id"], q["query"]) for q in questions]
    if json_hash(expected_queries) != query_manifest["contract"]["query_texts_sha256"]:
        raise ValueError("Benchmark query order/text differs from cached query vectors")
    by_id = {q["question_id"]: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError("Duplicate benchmark question IDs")
    candidates = {}
    for record in read_rows(source_path):
        if record["approach"] != "vector":
            continue
        identifier = record["question_id"]
        if identifier in candidates or identifier not in by_id:
            raise ValueError("Duplicate or unknown vector ranking question")
        question = by_id[identifier]
        if record.get("query") != question["query"]:
            raise ValueError("Vector ranking query text differs from benchmark")
        selected = record["results"][:20]
        if len(selected) != 20 or len({c["chunk_id"] for c in selected}) != 20:
            raise ValueError("Every question needs 20 distinct vector candidates")
        if [c["rank"] for c in selected] != list(range(1, 21)):
            raise ValueError("Vector candidates are not an unchanged ranked prefix")
        scores = [c["score"] for c in selected]
        if not np.isfinite(scores).all() or scores != sorted(scores, reverse=True):
            raise ValueError("Vector candidates have invalid or unordered scores")
        candidates[identifier] = {"question_id": identifier, "query": question["query"], "candidates": selected}
    if candidates.keys() != by_id.keys():
        raise ValueError("Missing vector rankings for benchmark questions")
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidates.jsonl"
    with candidate_path.open("w", encoding="utf-8", newline="\n") as stream:
        for question in questions:
            stream.write(json.dumps(candidates[question["question_id"]], ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1, "benchmark_sha256": benchmark_sha,
        "retrieval_run": str(retrieval_run), "retrieval_summary_sha256": sha256_file(retrieval_run / "comparison.json"),
        "source_rankings_sha256": source["per_query_sha256"], "candidate_file_sha256": sha256_file(candidate_path),
        "question_ids": [q["question_id"] for q in questions], "question_count": len(questions),
        "candidate_depth": 20, "pair_count": len(questions) * 20,
        "corpus": source["corpus"], "vector_query_manifest": query_manifest,
        "selection": "First 20 unchanged candidates from the existing exact-vector ranking; direct baseline is first 5",
        "model_inputs": "Query text and render_document(original candidate) only; no answers, labels, or expected documents",
    }
    write_json(output / "candidate_manifest.json", manifest)
    return benchmark, manifest


def checked_score_rows(rows, candidates):
    if len(rows) > len(candidates):
        raise ValueError("Score checkpoint contains extra questions")
    for row, candidate in zip(rows, candidates):
        if (row.get("question_id") != candidate["question_id"]
                or row.get("candidate_ids") != [c["chunk_id"] for c in candidate["candidates"]]):
            raise ValueError("Score checkpoint query/candidate order changed")
        scores = row.get("scores", [])
        if (len(scores) != len(candidate["candidates"])
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scores)
                or not np.isfinite(scores).all()):
            raise ValueError("Score checkpoint contains malformed scores")
    return rows


def verify_reference_scores(reranker, pairs, adapter_scores):
    """Check guarded inference against stock FlagEmbedding on the same loaded model."""
    from FlagEmbedding import FlagReranker
    flag = reranker._load_model()
    for query, passage in pairs:
        query_ids = flag.tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]
        passage_ids = flag.tokenizer(passage, add_special_tokens=False, truncation=False)["input_ids"]
        stock_ids = flag.tokenizer.prepare_for_model(query_ids, passage_ids, truncation="only_second",
                                                     max_length=reranker.max_tokens, padding=False)["input_ids"]
        lossless_ids = flag.tokenizer(query, passage, add_special_tokens=True, truncation=False)["input_ids"]
        if stock_ids != lossless_ids:
            raise ValueError("Lossless paired tokens differ from the stock FlagEmbedding input format")
    reference = FlagReranker.compute_score_single_gpu(
        flag, pairs, batch_size=reranker.batch_size, max_length=reranker.max_tokens,
        query_max_length=reranker.query_max_length, normalize=False, device=reranker.device)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    actual = np.asarray(adapter_scores, dtype=np.float64)
    if reference.shape != actual.shape or not np.allclose(reference, actual, rtol=1e-3, atol=1e-3):
        raise ValueError("Guarded reranker scores differ from stock FlagEmbedding beyond FP16 tolerance")
    return {"pair_count": len(pairs), "paired_token_ids_equal": True,
            "max_absolute_logit_difference": float(np.max(np.abs(reference - actual))),
            "relative_tolerance": 1e-3, "absolute_tolerance": 1e-3,
            "same_loaded_model": True, "input_pairs_sha256": json_hash(pairs)}


def score_worker(output, *, device="cuda:0", batch_size=4, preflight_only=False):
    """Model-only child process with complete-pair validation and atomic checkpoints."""
    candidate_manifest = read_json(output / "candidate_manifest.json")
    if sha256_file(output / "candidates.jsonl") != candidate_manifest["candidate_file_sha256"]:
        raise ValueError("Candidate source file checksum changed")
    candidates = list(read_rows(output / "candidates.jsonl"))
    if device.startswith("cuda:"):
        import torch
        torch.cuda.set_per_process_memory_fraction(0.6, device=device)
    reranker = BGEReranker(device=device, batch_size=batch_size)
    pairs = [(row["query"], render_document(chunk)) for row in candidates for chunk in row["candidates"]]
    lengths = reranker.pair_token_lengths(pairs)
    audit = reranker.preflight_pairs(pairs)
    preflight = {"adapter_audit": audit, "pair_count": len(pairs), "max_pair_tokens": max(lengths),
                 "mean_pair_tokens": float(np.mean(lengths)),
                 "pairs_above_512": sum(length > 512 for length in lengths),
                 "pairs_above_1024": sum(length > 1024 for length in lengths),
                 "candidate_file_sha256": candidate_manifest["candidate_file_sha256"], "reranker": reranker.identity,
                 "truncated_pairs": 0, "skipped_pairs": 0,
                 "policy": "Score each complete query plus unchanged source chunk; fail before inference if any pair exceeds model capacity"}
    write_json(output / "preflight.json", preflight)
    print(f"Preflight: {len(pairs)} pairs, maximum {max(lengths)} model tokens, no truncation", flush=True)
    if preflight_only:
        return preflight
    contract = {"candidate_file_sha256": candidate_manifest["candidate_file_sha256"], "reranker": reranker.identity,
                "worker_implementation_sha256": sha256_file(__file__), "warmup_pairs": 4, "gpu_memory_fraction": .6}
    checkpoint_path = output / "scoring_checkpoint.json"
    completed, previous_warmups, reference_checks = [], [], []
    if checkpoint_path.exists():
        saved = read_json(checkpoint_path)
        if saved["contract"] != contract:
            raise ValueError("Scoring checkpoint belongs to different inputs/configuration; use a new output directory")
        completed = checked_score_rows(saved["rows"], candidates)
        previous_warmups = saved.get("model_load_and_warmup_seconds", [])
        reference_checks = saved.get("reference_checks", [])
    if len(completed) < len(candidates):
        started = time.perf_counter()
        warmup_scores = reranker.score_pairs(pairs[:4])
        reference_checks.append(verify_reference_scores(reranker, pairs[:4], warmup_scores))
        if device.startswith("cuda:"):
            torch.cuda.synchronize(device)
        previous_warmups.append(time.perf_counter() - started)
        del pairs
        for number in range(len(completed), len(candidates)):
            row = candidates[number]
            query_pairs = [(row["query"], render_document(chunk)) for chunk in row["candidates"]]
            if device.startswith("cuda:"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            scores = reranker.score_pairs(query_pairs)
            if device.startswith("cuda:"):
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            completed.append({"question_id": row["question_id"], "candidate_ids": [c["chunk_id"] for c in row["candidates"]],
                              "scores": scores, "rerank_ms": elapsed_ms})
            write_json(checkpoint_path, {"contract": contract, "rows": completed,
                                        "model_load_and_warmup_seconds": previous_warmups,
                                        "reference_checks": reference_checks})
            print(f"Reranked {number + 1}/{len(candidates)} {row['question_id']} ({elapsed_ms:.1f} ms)", flush=True)
    checked_score_rows(completed, candidates)
    score_path = output / "scores.jsonl"
    with score_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in completed:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    times = [row["rerank_ms"] for row in completed]
    manifest = {"status": "complete", "contract": contract, "question_count": len(completed),
                "pair_count": sum(len(row["scores"]) for row in completed), "scores_sha256": sha256_file(score_path),
                "mean_rerank_ms": float(np.mean(times)), "median_rerank_ms": float(np.median(times)),
                "p95_rerank_ms": float(np.percentile(times, 95)), "model_load_and_warmup_seconds": previous_warmups,
                "reference_checks": reference_checks,
                "timing_scope": "One synchronized scoring pass per 20-candidate query after warmup, including tokenization/transfer/model scoring; excludes retrieval, model loading, disk checkpointing, ranking and evaluation",
                "truncated_pairs": 0, "skipped_pairs": 0}
    write_json(output / "score_manifest.json", manifest)
    return manifest


def evaluate_scores(benchmark, output, candidate_manifest, validation):
    manifest = read_json(output / "score_manifest.json")
    candidates = list(read_rows(output / "candidates.jsonl"))
    scores = list(read_rows(output / "scores.jsonl"))
    if (manifest.get("status") != "complete" or sha256_file(output / "scores.jsonl") != manifest["scores_sha256"]
            or manifest["contract"]["candidate_file_sha256"] != sha256_file(output / "candidates.jsonl")):
        raise ValueError("Scoring artifacts are incomplete or changed")
    if len(scores) != len(candidates) or len(candidates) != len(benchmark["questions"]):
        raise ValueError("Every benchmark question must have one complete score record")
    checked_score_rows(scores, candidates)
    rows = []
    with (output / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for question, candidate, scored in zip(benchmark["questions"], candidates, scores):
            if question["question_id"] != candidate["question_id"] or question["query"] != candidate["query"]:
                raise ValueError("Question/candidate order differs at evaluation")
            reranked = apply_scores(candidate["candidates"], scored["scores"], top_k=5)
            row = evaluate_pair(question, candidate["candidates"], reranked)
            row["rerank_ms"] = scored["rerank_ms"]
            row["reranked_results"] = reranked
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    summary = summarize(rows)
    summary.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": candidate_manifest, "benchmark_validation": validation,
        "preflight": read_json(output / "preflight.json"), "scoring": manifest,
        "per_query_artifact": "per_query.jsonl", "per_query_sha256": sha256_file(output / "per_query.jsonl"),
        "latency_by_split": {split: {"median_rerank_ms": float(np.median([r['rerank_ms'] for r in rows if r['split'] == split])),
                                      "p95_rerank_ms": float(np.percentile([r['rerank_ms'] for r in rows if r['split'] == split], 95))}
                             for split in ("dev", "test")},
        "policy": "Fixed requested BGE reranker, original vector Top 20 candidates and Top 5 output; no tuning on dev or test labels",
        "implementation_sha256": {path.relative_to(Path(__file__).parents[1]).as_posix(): sha256_file(path) for path in
                                   (Path(__file__), Path(__file__).with_name("reranker.py"),
                                    Path(__file__).with_name("rerank_evaluation.py"),
                                    Path(__file__).with_name("experiment.py"),
                                    Path(__file__).parents[1] / "report_chunker" / "retrieval.py")},
    })
    write_json(output / "comparison.json", summary)
    write_report(output / "comparison.md", summary)
    return summary


def write_report(path, summary):
    test = summary["splits"]["test"]
    lines = ["# BGE reranker experiment", "",
             f"The same {test['question_count']} frozen test questions compare direct vector Top 5 with vector Top 20 "
             "reranked by `BAAI/bge-reranker-v2-m3` to Top 5. Development results are recorded separately. "
             "The model and candidate depth are fixed; no labels select parameters.", "",
             "| Approach | Evidence Recall@5 | Annotated Precision@5 | Evidence accuracy@1 | Complete evidence@5 | Document Recall@5 |",
             "|---|---:|---:|---:|---:|---:|"]
    columns = ("evidence_recall_at_5", "annotated_evidence_precision_at_5", "evidence_accuracy_at_1", "all_evidence_hit_at_5", "document_recall_at_5")
    for name, label in (("direct_top5", "Vector Top 5"), ("reranked_top5", "Vector Top 20 → rerank Top 5")):
        metrics = test["approaches"][name]["metrics"]
        lines.append(f"| {label} | " + " | ".join(f"{metrics[key]:.2%}" for key in columns) + " |")
    lines.append("| Change (percentage points) | " + " | ".join(f"{test['paired_deltas'][key]['mean_delta_pp']:+.2f}" for key in columns) + " |")
    ceiling = test["approaches"]["candidate_top20"]["metrics"]["evidence_recall_at_20"]
    lines += ["", f"The unchanged Top 20 pool has **{ceiling:.2%} evidence Recall@20**. This bounds the evidence recall "
              "available to selection; a reranker cannot retrieve missing candidates, and five chunks may not cover every unit available in twenty.", "",
              "**Accuracy@1** means that the first passage individually covers at least one annotated evidence unit. "
              "Precision@5 is the fraction of five selected passages that each cover an annotated unit. "
              "Evidence Recall@5 credits complete source units recovered by the union of five passages. "
              "Complete evidence@5 requires every annotated unit. These are retrieval metrics; **generated-answer accuracy is not measured**.", "",
              "## Category results", "",
              "| Category | Approach | Evidence Recall@5 | Precision@5 | Evidence accuracy@1 | Complete evidence@5 |",
              "|---|---|---:|---:|---:|---:|"]
    for category, values in test["by_category"].items():
        for approach in ("direct_top5", "reranked_top5"):
            metrics = values["approaches"][approach]["metrics"]
            lines.append(f"| {category} | {approach} | " + " | ".join(f"{metrics[key]:.2%}" for key in columns[:4]) + " |")
    lines += ["", "## Paired changes and runtime", "",
              "| Metric | Change (pp) | Exploratory 95% interval (pp) | Improved / unchanged / worsened queries |",
              "|---|---:|---:|---:|"]
    for key in columns[:4]:
        value = test["paired_deltas"][key]
        lines.append(f"| {key} | {value['mean_delta_pp']:+.2f} | [{value['ci95_pp'][0]:+.2f}, {value['ci95_pp'][1]:+.2f}] | "
                     f"{value['improved_questions']} / {value['unchanged_questions']} / {value['worsened_questions']} |")
    timing = summary["latency_by_split"]["test"]
    preflight = summary["preflight"]
    device = summary["scoring"]["contract"]["reranker"]["device"]
    lines += ["", f"Added reranker latency: test median **{timing['median_rerank_ms']:.1f} ms**, p95 **{timing['p95_rerank_ms']:.1f} ms** per twenty-candidate query. "
              f"One scoring pass on `{device}` per question after warmup; includes tokenization, transfer and synchronized inference, excludes vector retrieval/model loading. "
              "This is an observed experiment cost, not a production load benchmark.", "",
              f"All {preflight['pair_count']:,} development/test pairs fit the reranker's 8,192-token combined limit; "
              f"maximum {preflight['max_pair_tokens']:,}, mean {preflight['mean_pair_tokens']:.1f}. "
              f"The default 512-token limit would truncate {preflight['pairs_above_512']:,} pairs; 1,024 would truncate {preflight['pairs_above_1024']:,}. "
              "No passage, table row, query or citation was truncated or skipped. Input text uses the existing company/ticker/fiscal-year prefix and original chunk text.", "",
              "The reference answers never enter the reranker. Raw cross-encoder logits rank candidates; they are not correctness probabilities. "
              "Score ties preserve original vector rank. Models and tokenizers are pinned and source/candidate/query/model artifacts are checksummed.", "",
              "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/run_reranker_experiment.py", "```", "",
              "The script extracts the unchanged vector Top 20 from the verified prior run, preflights complete pairs, "
              "scores in an isolated process with atomic resumable checkpoints, and evaluates source evidence after inference. "
              "Use `--preflight-only` to audit token lengths without loading model weights, or `--device cpu` for CPU inference. "
              "Use a new `--output` directory when changing configuration. Full rankings, score records and checkpoints remain under "
              "`data/experiments/reranker/`. See the [machine-readable results](../data/evaluation/reranker_results.json).", "",
              "## Limitations", ""] + [f"- {value}" for value in summary["limitations"]]
    lines += ["- BGE's authors recommend 1,024 tokens based on fine-tuning length, while the architecture supports 8,192. "
              "This experiment keeps longer source passages intact; the longer setting is not a claim of equal ranking quality at every length.", "",
              "Sources: [pinned model](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e), "
              "[author input-length guidance](https://huggingface.co/BAAI/bge-reranker-v2-m3/discussions/9)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/retrieval_questions.json"))
    parser.add_argument("--retrieval-run", type=Path, default=Path("data/experiments/retrieval_k_sweep"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/reranker"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--score-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not re.fullmatch(r"cpu|cuda:\d+", args.device):
        parser.error("device must be cpu or one explicit CUDA device, e.g. cuda:0")
    if args.output.resolve() == args.retrieval_run.resolve():
        parser.error("Use a separate output directory to preserve the source retrieval experiment")
    if args.score_worker:
        score_worker(args.output, device=args.device, batch_size=args.batch_size, preflight_only=args.preflight_only)
        return 0
    benchmark, candidates = prepare_candidates(args.benchmark, args.retrieval_run, args.output)
    validation = validate_benchmark(benchmark, args.processed)
    write_json(args.output / "benchmark_validation.json", validation)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "run_reranker_experiment.py"),
               "--score-worker", "--output", str(args.output.resolve()), "--device", args.device, "--batch-size", str(args.batch_size)]
    if args.preflight_only:
        command.append("--preflight-only")
    subprocess.run(command, check=True, env=environment)
    if not args.preflight_only:
        result = evaluate_scores(benchmark, args.output, candidates, validation)
        print(json.dumps(result["splits"]["test"]["approaches"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
