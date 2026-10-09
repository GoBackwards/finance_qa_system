"""Validate and merge source-authored dev/test labels; export reusable triplets."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_search.benchmark import merge_benchmarks, validate_benchmark
from report_search.experiment import write_json
from report_index.faiss_index import sha256_file


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "evaluation"
    benchmark = merge_benchmarks([directory / f"retrieval_{split}_questions.json" for split in ("dev", "test")])
    questions = benchmark["questions"]
    for split in ("dev", "test"):
        counts = Counter(q["category"] for q in questions if q["split"] == split)
        if counts != {"factual": 20, "comparative": 15, "summary": 15}:
            raise ValueError(f"Unexpected benchmark category/split composition: {split}: {counts}")
    validation = validate_benchmark(benchmark, root / "data" / "processed")
    benchmark["provenance"].update({
        "agent_curated": True, "human_reviewed": False, "status": "needs_human_review",
        "split_method": "20 companies per split, four per industry in each split; no company shared between dev and test",
        "freeze_rule": "Source-authored labels frozen before retrieval; dev-only alpha tuning; test labels never select settings",
    })
    benchmark["summary"] = validation
    destination = directory / "retrieval_questions.json"
    write_json(destination, benchmark)
    with (directory / "retrieval_triplets.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for q in questions:
            record = {key: q[key] for key in ("question_id", "split", "category", "query", "ground_truth_doc_ids", "answer")}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(directory / "retrieval_benchmark_validation.json", {
        **validation, "benchmark_sha256": sha256_file(destination),
        "triplets_sha256": sha256_file(directory / "retrieval_triplets.jsonl"),
        "source_parts": benchmark["provenance"]["source_parts"],
    })
    print(json.dumps({"questions": len(questions), "validation": validation,
                      "benchmark_sha256": sha256_file(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
