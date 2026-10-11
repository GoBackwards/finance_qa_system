"""Deterministic grades for factual and comparative benchmark answers, averaged over repeats.

Each gold answer point's figures (any scale, display rounding) must appear in
the released answer; a comparative point without figures must be stated as a
verdict naming the gold winner. A conflicting figure (same order of magnitude,
different value) or a verdict for the other company makes the answer wrong.
Summary answers are not graded here (outcome None). This port agreed with all
230 blind agent grades it was checked against on fully-correct vs not (factual +
comparative); its prototype agreed with all 310.

Usage: autograde_answers.py BASE.jsonl [NEW.jsonl] - with two files, prints a
paired comparison by question (each question's mean over its repeats).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
NUMBER = re.compile(r"(?<![\w.])\(?\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
                    r"(%|percent|thousand|million|billion|trillion)?", re.I)
DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+\d{1,2},\s+\d{4}")
COMPARE = re.compile(r"\b(higher|more|greater|larger|exceed\w*|outpac\w*|bigger|lower|less|smaller)\b", re.I)


def answer_body(answer):
    lines = [line for line in html.unescape(answer or "").split("\n")
             if not line.strip().startswith(("[^", ">"))]
    return re.sub(r"\[\^\d+\]", "", "\n".join(lines))


def numbers(text):
    out = []
    for match in NUMBER.finditer(DATE.sub(" ", text)):
        value, unit = float(match[1].replace(",", "")), (match[2] or "").lower()
        if unit in ("%", "percent"):
            out.append(("pct", value, 1.0, 0))
        elif not (1900 <= value <= 2100 and "," not in match[1] and not unit):  # bare years are periods
            decimals = len(match[1].split(".")[1]) if "." in match[1] else 0
            out.append(("amt", value, SCALE.get(unit, 1.0), decimals))
    return out


def _pairs(gold, answer):
    for gold_scale in ({gold[2]} if gold[2] != 1 else {1, 1e3, 1e6}):
        for answer_scale in ({answer[2]} if answer[2] != 1 else {1, 1e3, 1e6}):
            yield gold[1] * gold_scale, answer[1] * answer_scale, max(
                0.5 * 10 ** -answer[3] * answer_scale, 0.5 * 10 ** -gold[3] * gold_scale) * 1.0001


def matches(gold, found):
    if gold[0] == "pct":
        return any(a[0] == "pct" and abs(a[1] - gold[1]) <= 0.051 + 0.006 * abs(gold[1]) for a in found)
    return any(x > 0 and abs(x - y) <= tolerance
               for a in found if a[0] == "amt" for x, y, tolerance in _pairs(gold, a))


def conflicts(gold, found):
    if gold[0] == "pct":
        return any(a[0] == "pct" and gold[1] > 0 and 0.5 <= a[1] / gold[1] <= 2.0 and abs(a[1] - gold[1]) > 0.5
                   for a in found)
    return any(x > 0 and 0.5 <= y / x <= 2.0 and abs(y - x) / x > 0.006
               for a in found if a[0] == "amt" for x, y, _ in _pairs(gold, a))


def verdict_named(point, text, winner_only=True):
    winner = re.match(r"\s*([A-Z][\w&.\-]*)", point)
    if not winner:
        return False
    name = winner[1]
    for sentence in re.split(r"(?<=[.;])\s+", text):
        found = COMPARE.search(sentence)
        position = sentence.find(name)
        if found and position >= 0:
            lower = found[1].lower() in ("lower", "less", "smaller")
            if (position > found.start()) if lower else (position < found.start()):
                return True
    return False


def grade(question, result):
    """'correct' | 'partially_correct' | 'wrong' | 'false_abstention' | None (summary)."""
    if question["category"] == "summary":
        return None
    if (result or {}).get("status") != "answer":
        return "false_abstention"
    text = answer_body(result.get("answer"))
    found, supported, conflict = numbers(text), 0, False
    for point in question["answer_points"]:
        gold = [g for g in numbers(point["claim"]) if not (g[0] == "amt" and g[1] < 10 and g[1].is_integer()
                                                           and g[2] == 1)]  # small counts are not figures
        if gold:
            ok = all(matches(g, found) for g in gold)
            conflict |= not ok and any(conflicts(g, found) for g in gold if not matches(g, found))
        elif question["category"] == "comparative" and COMPARE.search(point["claim"]):
            ok = verdict_named(point["claim"], text)
        else:
            ok = False
        supported += bool(ok)
    if supported == len(question["answer_points"]):
        return "correct"
    return "wrong" if conflict or not supported else "partially_correct"


def load(path, questions):
    grades = defaultdict(list)
    for record in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()):
        question = questions.get(record["id"])
        if question is not None:
            grades[record["id"]].append((grade(question, record.get("result")), (record.get("result") or {}).get("status")))
    return grades


def summarize(name, grades, questions):
    print(f"== {name}")
    for category in ("factual", "comparative", "summary"):
        ids = [q for q in grades if questions[q]["category"] == category]
        if not ids:
            continue
        reps = [g for q in ids for g in grades[q]]
        answered = sum(status == "answer" for _, status in reps) / len(reps)
        line = f"  {category:12s} n={len(ids):3d} reps={len(reps):4d} answered {answered:6.1%}"
        if category != "summary":
            share = lambda outcome: sum(o == outcome for o, _ in reps) / len(reps)
            line += (f" | correct {share('correct'):6.1%} partial {share('partially_correct'):6.1%} "
                     f"wrong {share('wrong'):6.1%} refused {share('false_abstention'):6.1%}")
        print(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/evaluation/retrieval_questions.json")
    args = parser.parse_args(argv)
    questions = {q["question_id"]: q for q in json.loads(args.benchmark.read_text(encoding="utf-8"))["questions"]}
    loaded = [load(path, questions) for path in args.runs]
    for path, grades in zip(args.runs, loaded):
        summarize(path.name, grades, questions)
    if len(loaded) == 2:
        first, second = loaded
        for metric, test in (("correct", lambda o: o == "correct"), ("wrong", lambda o: o == "wrong"),
                             ("refused", lambda o: o == "false_abstention")):
            ids = sorted(q for q in set(first) & set(second) if questions[q]["category"] != "summary")
            mean = lambda grades, q: np.mean([test(o) for o, _ in grades[q]])
            deltas = np.array([mean(second, q) - mean(first, q) for q in ids])
            boot = deltas[np.random.default_rng(20261008).integers(0, len(ids), (10000, len(ids)))].sum(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            print(f"{metric:8s} {args.runs[1].name} - {args.runs[0].name}: {deltas.sum():+.2f} questions "
                  f"(of {len(ids)}; 95% CI {low:+.2f} to {high:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
