"""Reproduce source-grounded retrieval probes without changing the frozen benchmark.

Model stages run in separate processes to keep encoder and reranker weights from
co-residing. This is a small diagnostic case study, never an accuracy benchmark.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from report_index.embedding import sha256_file


CHUNKS = Path("data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl")
INDEX = Path("data/indexes/bge-m3/table_preserving_s512_o50/flat.faiss")
BM25 = Path("data/indexes/bm25_s512_o50/retrieval.sqlite")
CASES = [
    {"case_id": "nvda_buybacks", "ticker": "NVDA", "year": 2025,
     "query": "According to its cash-flow statement, how much cash did NVIDIA pay for share buybacks in FY2025?",
     "hint": "NVIDIA FY2025 cash flow payments related to repurchases of common stock",
     "table_page": 148, "rows": [31]},
    {"case_id": "msft_scope", "ticker": "MSFT", "year": 2024,
     "query": "What was Microsoft's company-wide gross margin percentage in FY2024?",
     "hint": "Microsoft FY2024 consolidated income statements gross margin total revenue",
     "table_page": 48, "rows": [4, 9]},
    {"case_id": "msft_operating_income", "ticker": "MSFT", "year": 2024,
     "query": "What was Microsoft's operating income in FY2024?",
     "hint": "Microsoft FY2024 income statements operating income",
     "table_page": 48, "rows": [13]},
    {"case_id": "cmi_net_sales", "ticker": "CMI", "year": 2021,
     "query": "What net sales did Cummins report for FY2021?",
     "hint": "Cummins FY2021 total revenue net sales",
     "table_page": 32, "rows": [0]},
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sync():
    import torch
    torch.cuda.synchronize()


def source_question(case):
    directory = Path(f"data/processed/{case['ticker']}/{case['ticker']}_FY{case['year']}_annual_report")
    table = next(json.loads(line) for line in (directory / "tables.jsonl").open(encoding="utf-8")
                 if json.loads(line)["table_id"].endswith(f":p{case['table_page']}:table1"))
    if not table["structured_reliable"]:
        raise ValueError("Probe requires accepted table rows")
    citation = table["citation"]
    evidence = []
    for row in case["rows"]:
        evidence.append({"kind": "table", "document_id": citation["document_id"],
            "source_sha256": citation["sha256"], "pdf_page": case["table_page"],
            "table_id": table["table_id"], "row_index": row,
            "evidence_id": f"{table['table_id']}:row:{row}",
            "row_text": " | ".join(cell or "" for cell in table["rows"][row]),
            "row_cells": table["rows"][row]})
    return {"question_id": case["case_id"], "query": case["query"], "split": "diagnostic",
            "category": "factual", "ground_truth_doc_ids": [citation["document_id"]],
            "evidence": evidence, "source_context": table["context"],
            "source_table_sha256": sha256_file(directory / "tables.jsonl")}


def rrf(pools, limit=20):
    """Equal-channel RRF over unique IDs; source payload is retained unchanged."""
    scores, candidates = {}, {}
    for pool in pools:
        seen = set()
        for rank, chunk in enumerate(pool, 1):
            identifier = chunk["chunk_id"]
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (60 + rank)
            candidates.setdefault(identifier, chunk)
    result = []
    for rank, identifier in enumerate(sorted(scores, key=lambda key: (-scores[key], key))[:limit], 1):
        chunk = deepcopy(candidates[identifier])
        chunk.update(rank=rank, score=scores[identifier], retrieval_approach="diagnostic_rrf")
        result.append(chunk)
    return result


def encode(output):
    import torch
    from report_index.embedding import DenseEncoder
    torch.cuda.set_per_process_memory_fraction(0.60, 0)
    started = time.perf_counter()
    encoder = DenseEncoder(device="cuda:0", batch_size=4)
    encoder.encode_queries(["Warmup: annual report revenue"])
    sync()
    load_s = time.perf_counter() - started
    texts = [case[key] for case in CASES for key in ("query", "hint")]
    vectors, times = [], []
    for query in texts:
        sync()
        started = time.perf_counter()
        vectors.append(encoder.encode_queries([query])[0])
        sync()
        times.append((time.perf_counter() - started) * 1000)
    np.save(output / "query_vectors.npy", np.asarray(vectors, dtype=np.float32))
    write(output / "encoding.json", {"queries": texts, "identity": encoder.identity,
        "load_and_warmup_s": load_s, "per_query_warm_encode_ms": times})


def retrieve(output):
    from report_search.retrieval import RetrievalEngine
    from report_chunker.retrieval import _search
    vectors = np.load(output / "query_vectors.npy")
    started = time.perf_counter()
    engine = RetrievalEngine(CHUNKS, INDEX, BM25)
    setup_s = time.perf_counter() - started
    engine.vector(vectors[0], 20)
    results = []
    with engine:
        for i, case in enumerate(CASES):
            original, expanded, times = [], [], {}
            for label, vector in (("original", vectors[2*i]), ("expanded", vectors[2*i+1])):
                started = time.perf_counter()
                pool = engine.vector(vector, 100)
                times[f"{label}_vector100_payload_ms"] = (time.perf_counter() - started) * 1000
                if label == "original":
                    original = pool
                else:
                    expanded = pool
            # Company is explicit in every query. Do not require report-year
            # equality: a later comparative report may support the requested FY.
            scoped = lambda pool: [c for c in pool if c["ticker"] == case["ticker"]]
            started = time.perf_counter()
            bm25_original = _search(engine.connection, case["query"], 20, case["ticker"], None)
            bm25_hint = _search(engine.connection, case["hint"], 20, case["ticker"], None)
            times["scoped_bm25_two_queries_ms"] = (time.perf_counter() - started) * 1000
            pools = {"vector20": original[:20], "expanded_vector20": expanded[:20],
                "scoped_vector_bm25_hint20": rrf([scoped(original[:20]), bm25_original, bm25_hint]),
                "scoped_expanded_vector_union20": rrf([scoped(original[:20]), scoped(expanded[:20])])}
            results.append({"case_id": case["case_id"], "query": case["query"],
                "lexical_hint": case["hint"], "timings_ms": times, "pools": pools,
                "original_vector100": original, "expanded_vector100": expanded})
    write(output / "candidates.json", {"index_setup_verify_load_s": setup_s,
        "corpus": engine.stats, "cases": results})


def rerank(output):
    import torch
    from report_index.embedding import render_document
    from report_search.reranker import BGEReranker, apply_scores
    torch.cuda.set_per_process_memory_fraction(0.60, 0)
    payload = read(output / "candidates.json")
    started = time.perf_counter()
    model = BGEReranker(device="cuda:0", batch_size=4)
    model.score_pairs([("Warmup: annual report revenue", "Annual report: revenue was 100 million.")])
    sync()
    load_s = time.perf_counter() - started
    for row in payload["cases"]:
        row["reranked"], row["reranker_timings_ms"], row["pair_token_preflight"] = {}, {}, {}
        for label, pool in row["pools"].items():
            pairs = [(row["query"], render_document(chunk)) for chunk in pool]
            row["pair_token_preflight"][label] = model.preflight_pairs(pairs)
            sync()
            started = time.perf_counter()
            scores = model.score_pairs(pairs)
            sync()
            row["reranker_timings_ms"][label] = (time.perf_counter() - started) * 1000
            row["reranked"][label] = apply_scores(pool, scores, 5)
        print("Reranked", row["case_id"], flush=True)
    payload["reranker"] = {"identity": model.identity, "load_and_warmup_s": load_s}
    write(output / "rankings.json", payload)


def summarize(output):
    from report_search.experiment import evaluate_question
    from report_search.confidence import confidence_gate
    payload = read(output / "rankings.json")
    encoding = read(output / "encoding.json")
    cases = []
    for index, (case, row) in enumerate(zip(CASES, payload["cases"])):
        question = source_question(case)
        # Alternative source groups live only in this diagnostic artifact. The
        # frozen benchmark and its published numbers remain unchanged.
        support_groups = [question]
        if case["case_id"] in ("msft_scope", "msft_operating_income"):
            alternative = dict(case, table_page=35,
                rows=[0, 1] if case["case_id"] == "msft_scope" else [2])
            support_groups.append(source_question(alternative))
        if case["case_id"] == "cmi_net_sales":
            directory = Path("data/processed/CMI/CMI_FY2021_annual_report")
            table = next(json.loads(line) for line in (directory / "tables.jsonl").open(encoding="utf-8")
                         if json.loads(line)["table_id"].endswith(":p61:table1"))
            quote = table["source_text"].splitlines()[0]
            if "24,021" not in quote or "2021" not in table["caption"] or "In millions" not in table["caption"]:
                raise ValueError("Alternative statement source lost its amount, period or units")
            alternative = deepcopy(question)
            alternative["evidence"] = [{"kind": "table_source", "document_id": table["citation"]["document_id"],
                "source_sha256": table["citation"]["sha256"], "pdf_page": 61,
                "table_id": table["table_id"], "evidence_id": table["table_id"] + ":source:net_sales",
                "char_start": 0, "char_end": len(quote), "quote": quote}]
            alternative["source_context"] = table["context"]
            support_groups.append(alternative)
        measurements = {}
        for label, pool in row["pools"].items():
            ranked = row["reranked"][label]
            group_support = [{"evidence_ids": [e["evidence_id"] for e in group["evidence"]],
                "candidate20_complete": bool(evaluate_question(group, pool, cutoffs=(20,))["metrics"]["all_evidence_hit_at_20"]),
                "reranked5_complete": bool(evaluate_question(group, ranked, cutoffs=(5,))["metrics"]["all_evidence_hit_at_5"])}
                for group in support_groups]
            measurements[label] = {
                "candidate20": evaluate_question(question, pool, cutoffs=(5, 20)),
                "reranked5": evaluate_question(question, row["reranked"][label], cutoffs=(5,)),
                "top5_chunk_ids": [chunk["chunk_id"] for chunk in row["reranked"][label]],
                "top5_titles": [{"document_id": c["document_id"], "score": c["score"],
                    "pages": c["source_pages"], "text": c["text"][:700]} for c in row["reranked"][label]],
                "reranker_ms": row["reranker_timings_ms"][label],
                "max_reranker_logit": max(c["reranker_score"] for c in ranked),
                "alternative_support_groups": group_support,
                "any_inspected_support_group_at20": any(g["candidate20_complete"] for g in group_support),
                "any_inspected_support_group_at5": any(g["reranked5_complete"] for g in group_support),
                "configured_gate": confidence_gate(ranked, 6.359375 if label == "vector20" else None).to_dict()}
        baseline_allowed = measurements["vector20"]["configured_gate"]["allowed"]
        ranks = {}
        for label in ("original_vector100", "expanded_vector100"):
            ranks[label] = [c["rank"] for c in row[label]
                if evaluate_question(question, [c], cutoffs=(1,))["metrics"]["evidence_recall_at_1"] > 0]
        cases.append({"case_id": case["case_id"], "query": case["query"], "hint": case["hint"],
            "source_question": question, "measurements": measurements, "target_ranks_in_top100": ranks,
            "inspected_support_groups": support_groups,
            "conditional_expansion_diagnostic": {
                "original_threshold": 6.359375, "baseline_allowed": baseline_allowed,
                "would_attempt_expansion_if_enabled": not baseline_allowed,
                "expanded_policy_threshold": None,
                "outcome": "keep_original_context_for_verification" if baseline_allowed else "refuse_unconfigured_expanded_policy",
                "note": "Existing original-policy diagnostic threshold is not certified and cannot be transferred to altered candidate pools."},
            "warm_embedding_ms": encoding["per_query_warm_encode_ms"][index*2],
            "timings_ms": row["timings_ms"]})
    summary = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "four selected diagnostic probes; exploratory, no generalization claim",
        "case_count": len(cases), "human_reviewed": False, "generation_calls": 0,
        "model_identity": {"encoder": encoding["identity"], "reranker": payload["reranker"]["identity"]},
        "corpus": payload["corpus"], "cases": cases,
        "cold_setup_s": {"encoder_load_and_warmup": encoding["load_and_warmup_s"],
            "index_verify_load": payload["index_setup_verify_load_s"],
            "reranker_load_and_warmup": payload["reranker"]["load_and_warmup_s"]},
        "limitations": ["Source-specific row coverage is non-exhaustive; other passages may answer correctly.",
            "Alternative-support groups are separately inspected diagnostic annotations, not exhaustive judgments or generated-answer accuracy; frozen labels are unchanged.",
            "Hints are predefined case-study interventions, not a deployed policy or tuned test metric.",
            "Reranking always uses the unchanged original question and unchanged candidate source text.",
            "Every final pool is at most 20. Equal-channel RRF constant=60; ties use chunk ID.",
            "Scope limits company only, allowing later reports that disclose historical fiscal years.",
            "One GPU pass after warmup; timings include tokenization and transfer but exclude queueing, HTTP generation and verification.",
            "Encoder, FAISS and reranker stages are separate processes; summed warm times are component estimates, not measured end-to-end latency.",
            "CMI probe reuses frozen development question dev-019; no frozen test labels are tuned."],
        "source_hashes": {str(path): sha256_file(path) for path in [Path(__file__), CHUNKS, INDEX, BM25,
            output / "query_vectors.npy", output / "candidates.json", output / "rankings.json"]}}
    write(output / "comparison.json", summary)
    for row in cases:
        print(row["case_id"], row["target_ranks_in_top100"],
              {name: (v["candidate20"]["metrics"]["evidence_recall_at_20"],
                      v["reranked5"]["metrics"]["evidence_recall_at_5"])
               for name, v in row["measurements"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/bad_cases_retrieval"))
    parser.add_argument("--stage", choices=["all", "encode", "retrieve", "rerank", "summarize"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for stage in ("encode", "retrieve", "rerank", "summarize"):
            subprocess.run([sys.executable, str(Path(__file__)), "--stage", stage,
                            "--output", str(args.output)], check=True)
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
