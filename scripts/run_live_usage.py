"""Run real questions through the demo's product path with a live model.

Each question uses ``DemoService.search`` (query scope -> company-scoped BGE-M3
vector pools -> BGE reranker -> balanced Top5, plus context extras for
company-scoped questions; global Top20 when no company is recognized) and
``DemoService.answer`` (gate -> source binding -> generator -> citation repair +
reference/grounding checks -> semantic verifier). The only addition is an
observer that records the generator/verifier JSON so failures can be diagnosed;
it never changes requests or responses. Credentials are read from the
environment or a local .env file and are never written to the output.

Without ``--corpus`` the run uses the DemoService default (corpus v2); Tables
A9-A12 were measured on ``--corpus v1``. Code from ``--src`` that predates
corpus versions always reads corpus v1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# --src runs another code version (e.g. `git archive` of an earlier commit's
# src/) against this checkout's data, indexes and models for before/after runs.
_SRC = Path(sys.argv[sys.argv.index("--src") + 1]) if "--src" in sys.argv[:-1] else ROOT / "src"
sys.path.insert(0, str(_SRC.resolve()))

DIAGNOSTIC_THRESHOLD = 6.359375  # data/evaluation/generation_gate_results.json (dev-only)
PREVIEW_CHARS = 400


def load_env_file(path):
    """Fill unset FINANCE_QA_* variables from a dotenv file; values are never printed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key.startswith("FINANCE_QA_") and value and not os.getenv(key):
            os.environ[key] = value


class CallLog:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = []

    def take(self):
        with self._lock:
            calls, self.calls = self.calls, []
        return calls

    def add(self, record):
        with self._lock:
            self.calls.append(record)


def recording_client_class(base_class, log):
    class RecordingClient(base_class):
        def complete_json(self, messages):
            system = messages[0]["content"] if messages else ""
            role = ("verifier" if system.startswith("You independently review")
                    else "decomposer" if system.startswith("You split a comparison") else "generator")
            started = time.perf_counter()
            record = {"role": role, "model": self.config.model,
                      "input_characters": sum(len(m.get("content", "")) for m in messages)}
            try:
                output = super().complete_json(messages)
                record["output"] = output
                return output
            except Exception as error:
                record["error"] = str(error)
                raise
            finally:
                record["duration_ms"] = (time.perf_counter() - started) * 1000
                log.add(record)
    return RecordingClient


def _chunk_view(chunk, full_text=False):
    text = chunk.get("text") or ""
    pages = sorted({c.get("pdf_page") for c in chunk.get("citations") or [] if c.get("pdf_page") is not None})
    view = {key: chunk.get(key) for key in ("rank", "chunk_id", "ticker", "fiscal_year", "kind",
                                            "vector_rank", "vector_score", "reranker_score")
            if chunk.get(key) is not None}
    if "vector_score" not in view and chunk.get("score") is not None:
        view["vector_score"] = chunk["score"]
    view["pdf_pages"] = pages
    view["text" if full_text else "preview"] = text if full_text else text[:PREVIEW_CHARS]
    return view


def _gate_decision(selected, threshold):
    scores = [c.get("reranker_score") for c in selected]
    if not scores or not all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores):
        return None
    return max(scores) >= threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/evaluation/live_usage_questions.json")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output; existing ids are skipped")
    parser.add_argument("--label", required=True, help="Run label, e.g. baseline or after_fix")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Raw reranker cutoff passed to the gate (use a very low value to observe generation)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-seconds", type=float, default=None)
    parser.add_argument("--ids", nargs="*", help="Optional subset of question ids")
    parser.add_argument("--category", help="Only questions of this category (e.g. comparative in the benchmark file)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--decompose", choices=("rule", "model"),
                        help="Split comparisons into per-company sub-questions (deterministic rewrite or model)")
    parser.add_argument("--src", type=Path, default=ROOT / "src",
                        help="Package source directory to import (default: this checkout's src)")
    parser.add_argument("--corpus", help="Corpus version (report_search.demo_service.CORPORA); "
                                         "default: the DemoService default")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold):
        parser.error("--threshold must be finite")

    load_env_file(args.env_file)
    deadline = args.deadline_seconds
    if deadline is None:
        deadline = float(os.getenv("FINANCE_QA_DEADLINE_SECONDS") or 30)

    from report_search import demo_service
    from report_search.generation import PROMPT_VERSION
    from report_search.llm_client import ChatCompletionsClient, ModelConfig, ModelError

    try:
        generator_config = ModelConfig.from_env()
        verifier_config = ModelConfig.from_env(verifier=True)
    except ModelError as error:
        parser.error(f"Model configuration unavailable: {error}")
    log = CallLog()
    demo_service.ChatCompletionsClient = recording_client_class(ChatCompletionsClient, log)

    # Live-pool questions carry "id"; the frozen benchmark (retrieval_questions.json) uses "question_id".
    questions = [dict(q, id=q.get("id", q.get("question_id")))
                 for q in json.loads(args.questions.read_text(encoding="utf-8"))["questions"]]
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
    if args.ids:
        wanted = set(args.ids)
        questions = [q for q in questions if q["id"] in wanted]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(todo)} questions to run ({len(done)} already recorded)", flush=True)

    options = {"decompose_comparisons": True} if args.decompose else {}
    corpora = getattr(demo_service, "CORPORA", None)  # None: --src code that predates corpus versions (v1 only)
    if args.corpus and args.corpus not in (corpora or ("v1",)):
        parser.error(f"--corpus must be one of {sorted(corpora or ('v1',))} for this source tree")
    if args.corpus and corpora:
        options["corpus"] = args.corpus
    service = demo_service.DemoService(ROOT, device=args.device, **options)
    corpus = getattr(service, "corpus", "v1")
    try:
        for number, question in enumerate(todo, 1):
            record = {"id": question["id"], "label": args.label, "query": question["query"],
                      "run_at_utc": datetime.now(timezone.utc).isoformat(),
                      "settings": {"threshold": args.threshold, "deadline_seconds": deadline,
                                   "device": args.device, "generator_model": generator_config.model,
                                   "verifier_model": verifier_config.model,
                                   "prompt_version": PROMPT_VERSION, "decompose": args.decompose,
                                   "corpus": corpus}}
            log.take()
            started = time.perf_counter()
            try:
                search = (service.search(question["query"], decomposer_config=generator_config)
                          if args.decompose == "model" else service.search(question["query"]))
                record["search"] = {
                    "timings": search["timings"],
                    "candidates": [_chunk_view(c) for c in search["candidates"]],
                    "selected": [_chunk_view(c, full_text=True) for c in search["selected"]],
                    "retrieval_trace": search.get("retrieval_trace"),
                }
                record["diagnostic_gate_accepts"] = _gate_decision(search["selected"], DIAGNOSTIC_THRESHOLD)
                result = service.answer(search, threshold=args.threshold, generator_config=generator_config,
                                        verifier_config=verifier_config, deadline_seconds=deadline)
                record["result"] = result
            except demo_service.DemoError as error:
                record["error"] = error.code
                if error.partial_result is not None:
                    record["partial_search"] = {"timings": error.partial_result.get("timings")}
            record["model_calls"] = log.take()
            record["wall_ms"] = (time.perf_counter() - started) * 1000
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            outcome = record.get("result", {}).get("status") or record.get("error")
            reason = record.get("result", {}).get("reason", "")
            print(f"[{number}/{len(todo)}] {question['id']}: {outcome} {reason} "
                  f"({record['wall_ms'] / 1000:.1f}s)", flush=True)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
