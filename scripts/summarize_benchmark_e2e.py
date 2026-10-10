"""Summarize end-to-end answer grades on the frozen 100-question benchmark.

Both systems answered every benchmark question live (scripts/run_live_usage.py):
``baseline_head`` is commit 6b61b5e, before the live bad-case fixes (vector
Top20 -> rerank Top5 -> prompt v1), ``current`` is the fixed pipeline. Grades are
agent-authored and adversarially re-checked (data/evaluation/benchmark_e2e_grades.json).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("baseline_head", "current")
GROUPS = ("all", "factual", "comparative", "summary", "dev", "test")


def _paired(first, second, seed=20261008):
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_delta_pp": float(differences.mean() * 100), "ci95_pp": [float(low * 100), float(high * 100)],
            "improved": int((differences > 0).sum()), "unchanged": int((differences == 0).sum()),
            "worsened": int((differences < 0).sum()), "replicates": 10000, "seed": seed}


def _metrics(rows, system):
    grades = [row[system] for row in rows]
    total_points = sum(row["points_total"] for row in rows)
    return {
        "question_count": len(rows),
        "correct": sum(g["outcome"] == "correct" for g in grades),
        "partially_correct": sum(g["outcome"] == "partially_correct" for g in grades),
        "wrong": sum(g["outcome"] == "wrong" for g in grades),
        "false_abstention": sum(g["outcome"] == "false_abstention" for g in grades),
        "accuracy": sum(g["outcome"] == "correct" for g in grades) / len(rows),
        "correct_or_partial_rate": sum(g["outcome"] in ("correct", "partially_correct") for g in grades) / len(rows),
        "answered_rate": sum(g["outcome"] != "false_abstention" for g in grades) / len(rows),
        "answers_with_error": sum(g["contains_error"] for g in grades),
        "answer_point_coverage": sum(min(g["points_supported"], row["points_total"]) for g, row in zip(grades, rows)) / total_points,
        "precision_of_released_answers": (sum(g["outcome"] == "correct" for g in grades)
                                          / max(1, sum(g["outcome"] != "false_abstention" for g in grades))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", type=Path, default=ROOT / "data/evaluation/benchmark_e2e_grades.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data/experiments/benchmark_e2e")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/benchmark_e2e_results.json")
    args = parser.parse_args(argv)

    questions = {q["question_id"]: q for q in
                 json.loads((ROOT / "data/evaluation/retrieval_questions.json").read_text(encoding="utf-8"))["questions"]}
    grades_doc = json.loads(args.grades.read_text(encoding="utf-8"))
    rows = []
    for grade in grades_doc["grades"]:
        q = questions[grade["question_id"]]
        rows.append({**grade, "category": q["category"], "split": q["split"]})
    rows.sort(key=lambda row: row["question_id"])
    runs = {system: {json.loads(line)["id"]: json.loads(line) for line in
                     (args.runs_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
            for system, name in (("baseline_head", "baseline"), ("current", "current"))}

    summary = {}
    for group in GROUPS:
        subset = [row for row in rows if group == "all" or row["category"] == group or row["split"] == group]
        entry = {system: _metrics(subset, system) for system in SYSTEMS}
        entry["paired_accuracy_current_minus_baseline"] = _paired(
            [row["current"]["outcome"] == "correct" for row in subset],
            [row["baseline_head"]["outcome"] == "correct" for row in subset])
        entry["paired_correct_or_partial_current_minus_baseline"] = _paired(
            [row["current"]["outcome"] in ("correct", "partially_correct") for row in subset],
            [row["baseline_head"]["outcome"] in ("correct", "partially_correct") for row in subset])
        entry["better"] = dict(Counter(row["better"] for row in subset))
        summary[group] = entry

    result = {
        "schema_version": 1,
        "experiment": "end-to-end answers on the frozen 100-question benchmark: commit 6b61b5e vs the bad-case-fixed pipeline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "data/evaluation/retrieval_questions.json (40 factual, 30 comparative, 30 summary; dev/test company-disjoint)",
        "systems": {system: {"prompt_version": next(iter(runs[system].values()))["settings"]["prompt_version"],
                             "generator_model": next(iter(runs[system].values()))["settings"]["generator_model"],
                             "released_answers": sum((r.get("result") or {}).get("status") == "answer" for r in runs[system].values()),
                             "abstain_reasons": dict(Counter((r.get("result") or {}).get("reason") for r in runs[system].values()
                                                             if (r.get("result") or {}).get("status") != "answer").most_common()),
                             "diagnostic_gate_accepts": sum(r.get("diagnostic_gate_accepts") is True for r in runs[system].values())}
                    for system in SYSTEMS},
        "settings": {"threshold": "-1e6 (gate open; both systems)", "runs": "one live run per system"},
        "grading": grades_doc["provenance"],
        "summary": summary,
        "per_question": rows,
        "limitations": [
            "Grades are agent-authored and adversarially re-checked; no independent human review.",
            "Benchmark labels are agent-authored and non-exhaustive.",
            "One live run per system; gpt-4.1-mini sampling adds per-question noise.",
            "The fixes were designed on the separate 50-question live pool, not on these benchmark questions.",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for group in GROUPS:
        b, c = summary[group]["baseline_head"], summary[group]["current"]
        p = summary[group]["paired_accuracy_current_minus_baseline"]
        print(f"{group:12s} n={b['question_count']:3d} accuracy {b['accuracy']:.0%} -> {c['accuracy']:.0%} "
              f"({p['mean_delta_pp']:+.1f} pp, CI {p['ci95_pp'][0]:+.1f}..{p['ci95_pp'][1]:+.1f}) | "
              f"correct+partial {b['correct_or_partial_rate']:.0%} -> {c['correct_or_partial_rate']:.0%} | "
              f"answered {b['answered_rate']:.0%} -> {c['answered_rate']:.0%} | wrong {b['wrong']} -> {c['wrong']} | "
              f"points {b['answer_point_coverage']:.1%} -> {c['answer_point_coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
