"""Summarize corpus v2 (parser header-order fix) against corpus v1 (Table A14).

Corpus v2 re-parses all 200 reports with same-line table headers read left to
right, then re-chunks, re-embeds and rebuilds BM25 in separate folders
(report_search.demo_service.CORPORA); v1 is untouched. Both answer runs use
this checkout's code: v1 replays the recorded A13 all-six contexts, v2 searches
its own indexes afresh. Factual and comparative answers are graded
deterministically (scripts/autograde_answers.py); the live pool is graded blind
by agents (data/evaluation/corpus_v2_grades.json). Writes
data/evaluation/corpus_v2_results.json.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from autograde_answers import grade  # noqa: E402
from report_chunker.retrieval import judge_retrieval  # noqa: E402
from report_search.demo_service import CORPORA  # noqa: E402
from report_search.evidence_binding import column_years  # noqa: E402
from summarize_next_phase import GOOD, bootstrap  # noqa: E402

RUNS = ROOT / "data/experiments/corpus_v2"
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _jsonl(path):
    # split("\n"), not splitlines(): chunk text can contain U+2028.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def _first_seen(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def header_order(processed):
    """Per table with >= 2 bbox-bound year columns: does the caption text list the years in column order?"""
    status = {}
    for path in sorted((ROOT / processed).glob("*/*/tables.jsonl")):
        for number, table in enumerate(_jsonl(path)):
            columns = column_years(table)
            if len(columns) < 2:
                continue
            years = set(columns.values())
            text = _first_seen(int(y) for item in table.get("context") or () if item.get("position") == "above"
                               for y in YEAR.findall(item.get("text") or "") if int(y) in years)
            order = _first_seen(columns[column] for column in sorted(columns))
            if set(text) == set(order):
                status[(path.parent.name, number)] = text == order
    return status


def main():
    v1, v2 = CORPORA["v1"], CORPORA["v2"]
    # 1. What the re-parse changed.
    order1, order2 = header_order(v1["processed"]), header_order(v2["processed"])
    transitions = Counter(f"{'agree' if order1[key] else 'disagree'}->{'agree' if order2[key] else 'disagree'}"
                          for key in order1.keys() & order2.keys())
    chunks1 = {(c["document_id"], c["chunk_index"]): c for c in _jsonl(ROOT / v1["chunks"])}
    chunks2 = {(c["document_id"], c["chunk_index"]): c for c in _jsonl(ROOT / v2["chunks"])}
    changed = [key for key in chunks1.keys() & chunks2.keys() if chunks1[key]["text"] != chunks2[key]["text"]]
    reordered = [key for key in changed
                 if YEAR.findall(chunks1[key]["text"].split("\n| ")[0]) != YEAR.findall(chunks2[key]["text"].split("\n| ")[0])]
    corpus = {"chunks": [len(chunks1), len(chunks2)], "chunks_with_changed_text": len(changed),
              "documents_with_changed_text": len({key[0] for key in changed}),
              "chunks_with_reordered_year_headers": len(reordered),
              "reordered_by_company": dict(Counter(chunks1[key]["ticker"] for key in reordered).most_common()),
              "header_order_tables_checked": len(order1),
              "header_order_disagreements": {"v1": sum(not v for v in order1.values()),
                                             "v2": sum(not v for v in order2.values())},
              "header_order_transitions": dict(transitions)}

    # 2. Gold evidence still resolvable, and final-context recall per corpus.
    questions = {q["question_id"]: q for q in json.loads((ROOT / "data/evaluation/retrieval_questions.json")
                                                         .read_text(encoding="utf-8"))["questions"]}
    by_document = {name: {} for name in ("v1", "v2")}
    for name, chunks in (("v1", chunks1), ("v2", chunks2)):
        for chunk in chunks.values():
            by_document[name].setdefault(chunk["document_id"], []).append(chunk)
    resolvable = {}
    for name in ("v1", "v2"):
        total = 0
        for q in questions.values():
            documents = {unit["document_id"] for unit in q["evidence"]}
            pool = [chunk for document in documents for chunk in by_document[name].get(document, [])]
            total += judge_retrieval(q, pool, top_k=len(pool))["recovered_units"]
        resolvable[name] = total
    corpus["gold_units"] = sum(len(q["evidence"]) for q in questions.values())
    corpus["gold_units_resolvable"] = resolvable

    runs = {"v1": _jsonl(RUNS / "v1_control.jsonl"), "v2": _jsonl(RUNS / "v2_all.jsonl")}

    def recall(records, corpus_name):
        store = sqlite3.connect((ROOT / CORPORA[corpus_name]["bm25"]).as_uri() + "?mode=ro", uri=True)
        contexts, values = {}, {"top5": [], "all": [], "factual": [], "comparative": [], "summary": [], "tokens": []}
        for record in records:
            contexts.setdefault(record["id"], record["search"]["selected"])
        for qid, selected in contexts.items():
            chunks = []
            for row in selected:
                payload, body = store.execute("SELECT payload, body FROM chunks WHERE chunk_id=?",
                                              (row["chunk_id"],)).fetchone()
                chunks.append(dict(json.loads(payload), text=body, context_reason=row.get("context_reason")))
            top5 = [chunk for chunk in chunks if not chunk.get("context_reason")]
            judged = judge_retrieval(questions[qid], chunks, top_k=len(chunks))
            share = judged["recovered_units"] / judged["gold_units"]
            values["all"].append(share)
            values[questions[qid]["category"]].append(share)
            values["top5"].append(judge_retrieval(questions[qid], top5, top_k=len(top5))["recovered_units"]
                                  / judged["gold_units"] if top5 else 0.0)
            values["tokens"].append(sum(int(c.get("token_count", 0)) for c in chunks))
        return {key: round(100 * statistics.mean(v), 2) if key != "tokens" else round(statistics.mean(v))
                for key, v in values.items()}

    retrieval = {name: recall(runs[name], name) for name in runs}

    # Exposure: which delivered contexts contain a chunk whose year headers changed order.
    position = {name: {chunk["chunk_id"]: key for key, chunk in chunks.items()}
                for name, chunks in (("v1", chunks1), ("v2", chunks2))}
    reordered = set(reordered)
    live_questions = {q["id"]: q for q in json.loads((ROOT / "data/evaluation/live_usage_questions.json")
                                                     .read_text(encoding="utf-8"))["questions"]}
    exposure = {}
    for pool, (old, new), category in (
            ("benchmark", ("v1_control.jsonl", "v2_all.jsonl"), lambda qid: questions[qid]["category"]),
            ("live", ("v1_control_live.jsonl", "v2_live.jsonl"), lambda qid: "live")):
        contexts = {}
        for name, file in (("v1", old), ("v2", new)):
            for record in _jsonl(RUNS / file):
                contexts.setdefault(name, {}).setdefault(record["id"], [position[name][row["chunk_id"]]
                                                                        for row in record["search"]["selected"]])
        counts = Counter()
        for qid, keys in contexts["v2"].items():
            counts[(category(qid), "questions")] += 1
            counts[(category(qid), "identical_context")] += keys == contexts["v1"][qid]
            if any(key in reordered for key in keys):
                counts[(category(qid), "with_reordered_year_headers")] += 1
                exposure.setdefault(pool + "_exposed_ids", []).append(qid)
        exposure[pool] = {f"{c}:{k}": v for (c, k), v in sorted(counts.items())}
    corpus["exposure"] = exposure

    # 3. Answers: deterministic grades, paired by question.
    per_question, arms = {name: {} for name in runs}, {}
    for name, records in runs.items():
        outcomes = Counter()
        for record in records:
            q = questions[record["id"]]
            outcome = grade(q, record["result"])
            per_question[name].setdefault(record["id"], []).append(outcome or record["result"].get("status"))
            outcomes[(q["category"], outcome if outcome else record["result"].get("status"))] += 1
        entry = {"generations": len(records)}
        for category in ("factual", "comparative", "summary"):
            total = sum(v for (c, _), v in outcomes.items() if c == category)
            entry[category] = {o: round(v / total, 4) for (c, o), v in outcomes.items() if c == category}
        arms[name] = entry
    ids = sorted(q for q in per_question["v2"] if questions[q]["category"] != "summary")
    paired = {metric: bootstrap([sum(o == value for o in per_question["v2"][q]) / len(per_question["v2"][q])
                                 - sum(o == value for o in per_question["v1"][q]) / len(per_question["v1"][q])
                                 for q in ids])
              for metric, value in (("correct", "correct"), ("wrong", "wrong"), ("refused", "false_abstention"))}
    moved = {q: {name: dict(Counter(per_question[name][q])) for name in runs} for q in ids
             if Counter(per_question["v1"][q]) != Counter(per_question["v2"][q])
             and abs(per_question["v1"][q].count("correct") - per_question["v2"][q].count("correct")) >= 3}

    result = {"schema_version": 1, "experiment": "corpus v2: parser header-order fix (docs/next_phase_plan.md, 4b)",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "corpora": CORPORA, "runs": "data/experiments/corpus_v2/*.jsonl (gitignored); gpt-4.1-mini; gate open",
              "corpus_change": corpus, "retrieval_strict_evr": retrieval, "answers": arms,
              "paired_v2_vs_v1": paired, "questions_moved_by_3_or_more_correct": moved}
    grades_path = ROOT / "data/evaluation/corpus_v2_grades.json"
    if grades_path.exists():
        grades = json.loads(grades_path.read_text(encoding="utf-8"))
        live = {name: dict(Counter(row[name]["outcome"] for row in grades["live"])) for name in ("v1", "v2")}
        for name in ("v1", "v2"):
            live[name + "_good"] = sum(v for k, v in live[name].items() if k in GOOD)
        live["paired_good"] = bootstrap([(r["v2"]["outcome"] in GOOD) - (r["v1"]["outcome"] in GOOD)
                                         for r in grades["live"]])
        result.update(live_blind=live, grading=grades["provenance"])
    (ROOT / "data/evaluation/corpus_v2_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                                                 encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("corpus_change", "retrieval_strict_evr", "answers", "paired_v2_vs_v1")},
                     indent=1))
    print("moved:", json.dumps(moved))
    if "live_blind" in result:
        print("live:", result["live_blind"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
