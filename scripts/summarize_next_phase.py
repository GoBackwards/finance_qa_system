"""Summarize the six next-phase directions (Table A13) from replay runs and blind grades.

Answer-side arms replay generation 5 times per question on identical recorded
contexts (scripts/replay_generation.py); the all-six arm retrieves fresh
contexts first. Factual and comparative answers are graded deterministically
(scripts/autograde_answers.py); summary answers (2 repeats) and the 50 live
questions are graded blind by agents with a skeptic re-check
(data/evaluation/next_phase_grades.json). Writes data/evaluation/next_phase_results.json.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from autograde_answers import grade  # noqa: E402
from report_chunker.retrieval import judge_retrieval  # noqa: E402

RUNS = ROOT / "data/experiments/next_phase"
ARMS = {"r0_baseline": "baseline (commit 435b7ae: context extras v1, prompt v2.2)",
        "r1_d4": "+ 4 column-year and footnoted-row checks",
        "r2_d4d1": "+ 1 deterministic comparison verdict",
        "r3_d4d1d3": "+ 3 citation repair (longest verbatim block, cross-claim calculations)",
        "r4_answer_side": "+ 6 verifier calibration and 5's 3,072-token generator limit",
        "r5_all": "+ 2 and 5 context extras v3 (fresh retrieval)"}
GOOD = {"correct_answer", "acceptable_clarification", "correct_abstention"}


def bootstrap(deltas):
    deltas = np.asarray(deltas, dtype=float)
    sums = deltas[np.random.default_rng(20261008).integers(0, len(deltas), (10000, len(deltas)))].sum(axis=1)
    low, high = np.quantile(sums, [0.025, 0.975])
    return {"delta_questions": round(float(deltas.sum()), 2), "ci95": [round(float(low), 2), round(float(high), 2)],
            "increased": int((deltas > 0).sum()), "decreased": int((deltas < 0).sum())}


def main():
    questions = {q["question_id"]: q for q in json.loads((ROOT / "data/evaluation/retrieval_questions.json")
                                                         .read_text(encoding="utf-8"))["questions"]}
    runs = {arm: [json.loads(line) for line in (RUNS / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
            for arm in ARMS}
    per_question = {arm: {} for arm in ARMS}
    arms = {}
    for arm, records in runs.items():
        outcomes = Counter()
        for record in records:
            q = questions[record["id"]]
            outcome = grade(q, record["result"])
            per_question[arm].setdefault(record["id"], []).append(outcome)
            outcomes[(q["category"], outcome if outcome else record["result"].get("status"))] += 1
        entry = {"label": ARMS[arm], "generations": len(records)}
        for category in ("factual", "comparative", "summary"):
            total = sum(v for (c, _), v in outcomes.items() if c == category)
            entry[category] = {o: round(v / total, 4) for (c, o), v in outcomes.items() if c == category}
        if arm != "r0_baseline":
            ids = sorted(q for q in per_question[arm] if questions[q]["category"] != "summary")
            for metric, value in (("correct", "correct"), ("wrong", "wrong"), ("refused", "false_abstention")):
                entry[f"paired_{metric}_vs_baseline"] = bootstrap(
                    [np.mean([o == value for o in per_question[arm][q]])
                     - np.mean([o == value for o in per_question["r0_baseline"][q]]) for q in ids])
        arms[arm] = entry

    # Final-context retrieval of the two context sets (strict judge, repo SQLite payloads).
    store = sqlite3.connect((ROOT / "data/indexes/bm25_s512_o50/retrieval.sqlite").as_uri() + "?mode=ro", uri=True)

    def recall(records):
        contexts, values = {}, {"all": [], "factual": [], "comparative": [], "summary": [], "tokens": []}
        for record in records:
            contexts.setdefault(record["id"], record["search"]["selected"])
        for qid, selected in contexts.items():
            chunks = []
            for row in selected:
                payload, body = store.execute("SELECT payload, body FROM chunks WHERE chunk_id=?",
                                              (row["chunk_id"],)).fetchone()
                chunks.append(dict(json.loads(payload), text=body))
            judged = judge_retrieval(questions[qid], chunks, top_k=len(chunks))
            share = judged["recovered_units"] / judged["gold_units"]
            values["all"].append(share)
            values[questions[qid]["category"]].append(share)
            values["tokens"].append(sum(int(c.get("token_count", 0)) for c in chunks))
        return {key: round(100 * statistics.mean(v), 2) if key != "tokens" else round(statistics.mean(v))
                for key, v in values.items()}

    retrieval = {"extras_v1 (r0-r4 contexts)": recall(runs["r0_baseline"]), "extras_v3 (r5 contexts)": recall(runs["r5_all"])}

    grades = json.loads((ROOT / "data/evaluation/next_phase_grades.json").read_text(encoding="utf-8"))
    summaries = grades["summary_rep0"] + grades["summary_rep1"]
    summary = {}
    for system in ("base", "all"):
        counts = Counter(row[system]["outcome"] for row in summaries)
        summary[system] = {"outcomes": dict(counts), "fully_correct": round(counts["correct"] / len(summaries), 4),
                           "points": round(sum(r[system]["points_supported"] for r in summaries)
                                           / sum(r["points_total"] for r in summaries), 4)}
    by_question = {}
    for row in summaries:
        by_question.setdefault(row["question_id"], []).append(
            (row["all"]["outcome"] == "correct") - (row["base"]["outcome"] == "correct"))
    summary["paired_fully_correct"] = bootstrap([np.mean(v) for v in by_question.values()])
    live = {system: dict(Counter(row[system]["outcome"] for row in grades["live"])) for system in ("base", "all")}
    for system in ("base", "all"):
        live[system + "_good"] = sum(v for k, v in live[system].items() if k in GOOD)
    live["paired_good"] = bootstrap([(r["all"]["outcome"] in GOOD) - (r["base"]["outcome"] in GOOD) for r in grades["live"]])

    result = {"schema_version": 1, "experiment": "next-phase improvement directions (docs/next_phase_plan.md)",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "runs": "data/experiments/next_phase/*.jsonl (gitignored); gpt-4.1-mini generator and verifier; gate open",
              "arms": arms, "retrieval_final_context_strict_evr": retrieval,
              "summary_blind": summary, "live_blind": live, "grading": grades["provenance"],
              "limitations": grades["limitations"]}
    (ROOT / "data/evaluation/next_phase_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                                                   encoding="utf-8")
    for arm, entry in arms.items():
        paired = entry.get("paired_correct_vs_baseline", {})
        print(f"{arm:16s} fact {entry['factual'].get('correct', 0):.1%}/{entry['factual'].get('wrong', 0):.1%} "
              f"comp {entry['comparative'].get('correct', 0):.1%}/{entry['comparative'].get('partially_correct', 0):.1%}/"
              f"{entry['comparative'].get('wrong', 0):.1%} | correct {paired.get('delta_questions', '-')} {paired.get('ci95', '')}")
    print("retrieval", retrieval, "| summaries", summary, "| live", live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
