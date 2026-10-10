"""Summarize live-usage runs, their verified grades and the typical bad-case table.

Inputs are the raw run records written by run_live_usage.py (gitignored under
data/experiments/live_usage), the agent grades in
data/evaluation/live_usage_grades.json and the authored typical-case table.
The output is the committed, compact data/evaluation/live_bad_case_results.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
GOOD = {"correct_answer", "acceptable_clarification", "correct_abstention"}
# Each run used the code at that point of the fix history (see docs/live_bad_cases.md).
RUNS = ("baseline", "iteration1", "iteration2", "iteration3", "after_fix")
GRADED = ("baseline", "iteration2", "iteration3", "after_fix")
RELEASED_BAD = {"wrong_answer", "partially_correct", "unsafe_answer"}


def _records(path):
    return {json.loads(line)["id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _answer_text(result):
    text = html.unescape(result.get("answer") or "")
    return text.split("\n\n[^")[0].strip()


def _run_summary(records):
    results = [record.get("result") or {} for record in records.values()]
    audits = [result.get("audit") or {} for result in results]
    timings = [result.get("audit", {}).get("generation_elapsed_ms") for result in results]
    contexts = [audit.get("context_characters") for audit in audits if audit.get("context_characters")]
    retrieval = [(record.get("search") or {}).get("timings", {}).get("total_ms") for record in records.values()]
    first = next(iter(records.values()))["settings"]
    return {
        "question_count": len(records),
        "settings": {key: first[key] for key in ("threshold", "deadline_seconds", "device", "generator_model",
                                                 "verifier_model", "prompt_version")},
        "answered": sum(result.get("status") == "answer" for result in results),
        "abstained": sum(result.get("status") == "abstain" for result in results),
        "reasons": dict(Counter(result.get("reason") or "answer" for result in results).most_common()),
        "context_characters_median": median(contexts) if contexts else None,
        "context_characters_max": max(contexts) if contexts else None,
        "generation_ms_median": median(t for t in timings if t is not None),
        # The first question also pays one-time model loading.
        "retrieval_ms_median": median(t for t in retrieval[1:] if t is not None),
        "retried_generations": sum(audit.get("generation_attempts", 1) > 1 for audit in audits),
        "citation_repairs": dict(Counter(item["kind"] for audit in audits
                                         for item in audit.get("citation_repairs") or []).most_common()),
        "diagnostic_gate_accepts": sum(record.get("diagnostic_gate_accepts") is True for record in records.values()),
    }


def _grade_summary(grades, records):
    by_id = {grade["id"]: grade for grade in grades}
    outcomes = Counter(grade["outcome"] for grade in grades)
    bad = [grade for grade in grades if grade["is_bad_case"]]
    gate = {"correct_released": 0, "correct_released_gate_accepts": 0,
            "bad_released": 0, "bad_released_gate_blocks": 0}
    for qid, grade in by_id.items():
        accepts = records[qid].get("diagnostic_gate_accepts") is True
        released = (records[qid].get("result") or {}).get("status") == "answer"
        if released and grade["outcome"] in GOOD:
            gate["correct_released"] += 1
            gate["correct_released_gate_accepts"] += accepts
        if released and grade["outcome"] in RELEASED_BAD:
            gate["bad_released"] += 1
            gate["bad_released_gate_blocks"] += not accepts
    return {"outcomes": dict(outcomes.most_common()), "good": sum(outcomes[key] for key in GOOD),
            "bad_cases": len(bad), "bad_by_stage": dict(Counter(grade["failure_stage"] for grade in bad).most_common()),
            "bad_by_severity": dict(Counter(grade["severity"] for grade in bad).most_common()),
            "verifier_disagreements": sum(not grade.get("verifier_agrees", True) for grade in grades),
            "diagnostic_gate_6_359375": gate}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/evaluation/live_usage_questions.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data/experiments/live_usage")
    parser.add_argument("--grades", type=Path, default=ROOT / "data/evaluation/live_usage_grades.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/evaluation/live_bad_case_table.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/live_bad_case_results.json")
    args = parser.parse_args(argv)

    question_bytes = args.questions.read_bytes()
    questions = {q["id"]: q for q in json.loads(question_bytes)["questions"]}
    runs = {name: _records(args.runs_dir / f"{name}.jsonl") for name in RUNS}
    grades_doc = json.loads(args.grades.read_text(encoding="utf-8"))
    grades = {name: {grade["id"]: grade for grade in grades_doc[name]} for name in GRADED}
    cases = json.loads(args.cases.read_text(encoding="utf-8"))

    transitions = Counter()
    per_question = []
    for qid, question in questions.items():
        row = {"id": qid, "persona": question["persona"], "category": question["category"],
               "expected_behavior": question["expected_behavior"], "query": question["query"]}
        for name, records in runs.items():
            result = records[qid].get("result") or {}
            entry = {"status": result.get("status"), "reason": result.get("reason"),
                     "diagnostic_gate_accepts": records[qid].get("diagnostic_gate_accepts")}
            if result.get("status") == "answer":
                entry["answer"] = _answer_text(result)[:600]
                entry["cited_pages"] = sorted({f"{c['citation']['ticker']} FY{c['citation']['fiscal_year']} p{c['citation']['pdf_page']}"
                                               for c in result.get("citations") or []})
            if name in grades:
                grade = grades[name][qid]
                entry.update({key: grade[key] for key in ("outcome", "is_bad_case", "failure_stage", "failure_type",
                                                          "severity", "root_cause")})
            row[name] = entry
        transitions[(row["baseline"]["outcome"], row["after_fix"]["outcome"])] += 1
        per_question.append(row)
    fixed = [r["id"] for r in per_question if r["baseline"]["is_bad_case"] and not r["after_fix"]["is_bad_case"]]
    regressed = [r["id"] for r in per_question if not r["baseline"]["is_bad_case"] and r["after_fix"]["is_bad_case"]]
    still = [r["id"] for r in per_question if r["baseline"]["is_bad_case"] and r["after_fix"]["is_bad_case"]]

    result = {
        "schema_version": 1,
        "experiment": "live bad-case pool: real questions through the demo product path, before and after fixes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_pool": {"path": "data/evaluation/live_usage_questions.json",
                          "sha256": hashlib.sha256(question_bytes).hexdigest(), "question_count": len(questions),
                          "by_persona": dict(Counter(q["persona"] for q in questions.values())),
                          "by_category": dict(Counter(q["category"] for q in questions.values()).most_common()),
                          "by_expected_behavior": dict(Counter(q["expected_behavior"] for q in questions.values()))},
        "grading": grades_doc["provenance"],
        "runs": {name: _run_summary(records) for name, records in runs.items()},
        "grades": {name: _grade_summary(list(grades[name].values()), runs[name]) for name in grades},
        "before_after": {"fixed": fixed, "still_bad": still, "regressed": regressed,
                         "outcome_transitions": [{"baseline": a, "after_fix": b, "count": n}
                                                 for (a, b), n in sorted(transitions.items(), key=lambda x: -x[1])]},
        "typical_cases": cases["cases"],
        "per_question": per_question,
        "limitations": [
            "50 hand-designed questions; counts describe this pool, not production rates.",
            "Gold answers and grades are agent-authored and adversarially re-checked by a second agent; no independent human review.",
            "Each configuration was run once; gpt-4.1-mini output varies between runs, so single-question changes can include sampling variation.",
            "The confidence gate was opened (threshold -1e6) to observe every stage; the dev-calibrated gate decision is recorded separately.",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({name: result["grades"][name]["outcomes"] for name in result["grades"]}, indent=1))
    print("fixed", len(fixed), "still_bad", len(still), "regressed", len(regressed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
