"""Summarize end-to-end answers to the 30 comparative benchmark questions with and without decomposition.

Every system ran live through DemoService (scripts/run_live_usage.py, gate
open): ``current`` is the scoped pipeline of Table A10, ``decomposed_rule`` and
``decomposed_model`` split each comparison into per-company sub-questions
(deterministic rewrite or model plan), retrieve and rerank each one in its
company's reports, and answer the original question from the combined Top5.
The ``_v22`` systems repeat all three with generator prompt v2.2, which asks
for an explicit comparison result. Grades are agent-authored, blind side by
side within each prompt version, and adversarially re-checked
(data/evaluation/decomposition_e2e_grades.json).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from report_search.experiment import evaluate_question
from summarize_benchmark_e2e import _metrics, _paired

SYSTEMS = ("current", "decomposed_rule", "decomposed_model",
           "current_v22", "decomposed_rule_v22", "decomposed_model_v22")
GROUPS = ("all", "dev", "test")


def _retrieval(question, record, store):
    """Re-score the run's own Top5 against the gold units using the stored chunk payloads."""
    selected = []
    for row in (record.get("search") or {}).get("selected") or []:
        payload, body = store.execute("SELECT payload,body FROM chunks WHERE chunk_id=?", (row["chunk_id"],)).fetchone()
        selected.append(dict(json.loads(payload), text=body))
    metrics = evaluate_question(question, selected, (5,))["metrics"]
    return {key: metrics[key] for key in ("evidence_recall_at_5", "all_evidence_hit_at_5")}


def _median_timings(records):
    """Median search-stage and end-to-end times; requests that loaded the models are excluded."""
    warm = [record for record in records
            if ((record.get("search") or {}).get("timings") or {}).get("startup_ms", 0) < 1000]
    keys = sorted({key for record in warm for key in record["search"]["timings"] if key.endswith("_ms")})
    medians = {key: round(statistics.median(record["search"]["timings"].get(key, 0.0) for record in warm), 1)
               for key in keys if key not in ("startup_ms", "queue_wait_ms")}
    medians["end_to_end_wall_ms"] = round(statistics.median(record["wall_ms"] for record in warm), 1)
    medians["warm_requests"] = len(warm)
    return medians


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", type=Path, default=ROOT / "data/evaluation/decomposition_e2e_grades.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data/experiments/benchmark_e2e")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/decomposition_e2e_results.json")
    args = parser.parse_args(argv)

    questions = {q["question_id"]: q for q in
                 json.loads((ROOT / "data/evaluation/retrieval_questions.json").read_text(encoding="utf-8"))["questions"]}
    grades_doc = json.loads(args.grades.read_text(encoding="utf-8"))
    systems = [system for system in SYSTEMS if system in grades_doc["grades"][0]]
    graded = {grade["question_id"] for grade in grades_doc["grades"]}
    runs = {}
    for system in systems:  # current.jsonl holds all 100 benchmark questions; keep the graded ones
        records = (json.loads(line) for line in
                   (args.runs_dir / f"{system}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        runs[system] = {record["id"]: record for record in records if record["id"] in graded}
    rows = []
    store = sqlite3.connect((ROOT / "data/indexes/bm25_s512_o50/retrieval.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        for grade in grades_doc["grades"]:
            q = questions[grade["question_id"]]
            row = {**grade, "split": q["split"]}
            for system in systems:
                row[system] = {**grade[system], "retrieval": _retrieval(q, runs[system][grade["question_id"]], store)}
            rows.append(row)
    finally:
        store.close()
    rows.sort(key=lambda row: row["question_id"])

    summary = {}
    for group in GROUPS:
        subset = [row for row in rows if group == "all" or row["split"] == group]
        entry = {}
        for system in systems:
            entry[system] = _metrics(subset, system)
            for key in ("evidence_recall_at_5", "all_evidence_hit_at_5"):
                entry[system][key] = sum(row[system]["retrieval"][key] for row in subset) / len(subset)
        for system in systems[1:]:
            entry[f"paired_accuracy_{system}_minus_current"] = _paired(
                [row[system]["outcome"] == "correct" for row in subset],
                [row["current"]["outcome"] == "correct" for row in subset])
            entry[f"paired_correct_or_partial_{system}_minus_current"] = _paired(
                [row[system]["outcome"] in ("correct", "partially_correct") for row in subset],
                [row["current"]["outcome"] in ("correct", "partially_correct") for row in subset])
        for key in ("best", "best_v22"):
            if key in subset[0]:
                entry[key] = dict(Counter(row[key] for row in subset))
        summary[group] = entry

    result = {
        "schema_version": 1,
        "experiment": "end-to-end answers to the comparative benchmark questions: scoped vs decomposed retrieval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "data/evaluation/retrieval_questions.json, comparative category (15 dev, 15 test)",
        "systems": {system: {"settings": {key: value for key, value in next(iter(runs[system].values()))["settings"].items()
                                          if key != "device"},
                             "released_answers": sum((r.get("result") or {}).get("status") == "answer" for r in runs[system].values()),
                             "abstain_reasons": dict(Counter((r.get("result") or {}).get("reason") for r in runs[system].values()
                                                             if (r.get("result") or {}).get("status") != "answer").most_common()),
                             "median_timings_ms": _median_timings(runs[system].values())}
                    for system in systems},
        "grading": grades_doc["provenance"],
        "summary": summary,
        "per_question": rows,
        "limitations": [
            "Grades are agent-authored and adversarially re-checked; no independent human review.",
            "30 questions from 3 comparison templates; one live run per system, so gpt-4.1-mini sampling adds noise.",
            "The rule decomposition was fixed before any benchmark run; the model prompt v2 was revised after seeing "
            "test retrieval results, so decomposed_model test numbers are not held-out evidence.",
            "Generator prompt v2.2 (explicit comparison result) was written after the v2.1 grades of these same "
            "questions showed missing verdicts, so the v2.2 rows are not held-out evidence.",
            "Each prompt version was graded in its own blind pass; comparisons across passes include grader variance.",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for group in GROUPS:
        entry = summary[group]
        line = "\n      ".join(f"{system:22s} acc {entry[system]['accuracy']:.0%} c+p {entry[system]['correct_or_partial_rate']:.0%} "
                          f"ans {entry[system]['answered_rate']:.0%} wrong {entry[system]['wrong']} "
                          f"pts {entry[system]['answer_point_coverage']:.1%} EvR@5 {entry[system]['evidence_recall_at_5']:.1%}"
                          for system in systems)
        print(f"{group:4s} n={entry['current']['question_count']}:\n      {line}")
        for system in systems[1:]:
            p = entry[f"paired_accuracy_{system}_minus_current"]
            print(f"      {system} - current accuracy {p['mean_delta_pp']:+.1f} pp "
                  f"(CI {p['ci95_pp'][0]:+.1f}..{p['ci95_pp'][1]:+.1f}; {p['improved']}/{p['unchanged']}/{p['worsened']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
