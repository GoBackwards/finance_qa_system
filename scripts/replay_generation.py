"""Re-run only the answer stage on fixed contexts, several times per question.

Retrieval is deterministic, so answer-side changes can be compared on
identical passages; repeating generation averages out model sampling (two
runs of the same configuration differed by about 10 fully-correct answers per
100). Contexts come from a recorded run (``--contexts``, the ``search.selected``
passages of run_live_usage.py records) or from a fresh ``DemoService.search``
(``--fresh-search``, loads the models). Output records keep the
run_live_usage.py shape plus ``rep``. Credentials come from the environment or
.env and are never written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
_SRC = Path(sys.argv[sys.argv.index("--src") + 1]) if "--src" in sys.argv[:-1] else ROOT / "src"
sys.path.insert(0, str(_SRC.resolve()))
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_usage import load_env_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--contexts", type=Path, help="run_live_usage.py JSONL whose search.selected is replayed")
    source.add_argument("--fresh-search", action="store_true", help="retrieve with DemoService first (GPU)")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/evaluation/retrieval_questions.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=-1e6)
    parser.add_argument("--deadline-seconds", type=float, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--corpus", default="v1", choices=("v1", "v2"),
                        help="corpus the contexts were retrieved from; v1 reproduces Tables A1-A13")
    parser.add_argument("--src", type=Path, default=ROOT / "src")
    args = parser.parse_args(argv)
    load_env_file(ROOT / ".env")

    try:
        from report_search.demo_service import CORPORA
    except ImportError:  # --src code that predates corpus versions reads corpus v1 only
        CORPORA = {"v1": {"processed": "data/processed", "bm25": "data/indexes/bm25_s512_o50/retrieval.sqlite"}}
    if args.corpus not in CORPORA:
        parser.error(f"--corpus {args.corpus} is not available in this source tree")
    from report_search.evidence_binding import SourceEvidenceStore
    from report_search.generation import PROMPT_VERSION, generate_answer
    from report_search.llm_client import ChatCompletionsClient, ModelConfig
    from report_search.query_scope import ScopeResolver

    questions = {q.get("id", q.get("question_id")): q["query"]
                 for q in json.loads(args.questions.read_text(encoding="utf-8"))["questions"]}
    if args.contexts:
        contexts = {r["id"]: r["search"]["selected"] for r in map(json.loads, args.contexts.read_text(
            encoding="utf-8").splitlines()) if r.get("search")}
    else:
        from report_search.demo_service import DemoService
        options = {"corpus": args.corpus} if "v2" in CORPORA else {}  # older code has no corpus argument
        with DemoService(ROOT, device=args.device, **options) as service:
            contexts = {qid: service.search(query)["selected"]
                        for qid, query in questions.items() if not args.ids or qid in args.ids}
    ids = [qid for qid in contexts if qid in questions and (not args.ids or qid in args.ids)]

    store = sqlite3.connect((ROOT / CORPORA[args.corpus]["bm25"]).as_uri() + "?mode=ro", uri=True,
                            check_same_thread=False)
    lock, local = threading.Lock(), threading.local()

    def candidates(selected):
        out = []
        for row in selected:
            with lock:
                payload, body = store.execute("SELECT payload, body FROM chunks WHERE chunk_id=?",
                                              (row["chunk_id"],)).fetchone()
            chunk = dict(json.loads(payload), text=body)
            chunk.update({key: row[key] for key in ("rank", "reranker_score", "vector_score", "vector_rank",
                                                     "context_reason") if key in row})
            out.append(chunk)
        return out

    resolver = ScopeResolver(json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8-sig")))
    generator_config, verifier_config = ModelConfig.from_env(), ModelConfig.from_env(verifier=True)
    done = set()
    if args.output.exists():
        done = {(r["id"], r["rep"]) for r in map(json.loads, args.output.read_text(encoding="utf-8").splitlines())}

    def run(job):
        qid, rep = job
        if not hasattr(local, "store"):
            local.store = SourceEvidenceStore(ROOT / CORPORA[args.corpus]["processed"], cache_documents=8)
        query = questions[qid]
        result = generate_answer(query, candidates(contexts[qid]), threshold=args.threshold,
                                 source_store=local.store, generator=ChatCompletionsClient(generator_config),
                                 verifier=ChatCompletionsClient(verifier_config),
                                 deadline_seconds=args.deadline_seconds, scope=resolver.resolve(query).to_dict(),
                                 company_resolver=resolver.entities_in_text)
        record = {"id": qid, "rep": rep, "label": args.label, "query": query,
                  "settings": {"prompt_version": PROMPT_VERSION, "generator_model": generator_config.model,
                               "corpus": args.corpus,
                               "threshold": args.threshold, "deadline_seconds": args.deadline_seconds},
                  "search": {"selected": [{key: row.get(key) for key in ("chunk_id", "rank", "reranker_score",
                                                                         "context_reason") if key in row}
                                          for row in contexts[qid]]},
                  "result": result}
        with lock, args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result.get("status")

    jobs = [(qid, rep) for rep in range(args.reps) for qid in ids if (qid, rep) not in done]
    print(f"{len(jobs)} generations ({len(ids)} questions x {args.reps} reps, {len(done)} already recorded)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for number, status in enumerate(pool.map(run, jobs), 1):
            if number % 50 == 0 or number == len(jobs):
                print(f"[{number}/{len(jobs)}] last: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
