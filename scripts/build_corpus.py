"""Build the demo corpus (report_search.demo_service.CORPORA) from the downloaded PDFs.

Runs the existing stages in order: parse, parse audit, 512/50 table-preserving
chunks, the SQLite FTS5 chunk store, BGE-M3 embeddings with the exact FAISS
index, and finally fetches the pinned reranker snapshot the demo loads. Every
output path comes from CORPORA, so a build never writes into another corpus
version's folders. Parsing, chunk parts and embeddings resume, so an
interrupted build continues when the same command is run again; the chunk store
and the FAISS index are rebuilt on every run (a few minutes), so stop the demo
before rerunning a build. The parse audit goes to an ignored file; the committed
data/parsed_validation_report_v2.json is the recorded audit of the published v2.

Corpus v1, behind Tables A1-A13, was parsed before the table header-order fix
and cannot be rebuilt with this parser (docs/system_architecture.md, Table A14).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from report_search.demo_service import CORPORA, DEFAULT_CORPUS  # noqa: E402

AUDIT_REPORTS = {"v2": "data/experiments/parsed_validation_report_v2.json"}


def run(script, *args):
    command = [sys.executable, f"scripts/{script}", *map(str, args)]
    print("==", " ".join(command[1:]), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N, for the embedding stage")
    parser.add_argument("--workers", type=int, default=4, help="parallel parse and chunk workers")
    parser.add_argument("--batch-size", type=int, default=8, help="embedding batch size")
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="CUDA allocator cap for embedding (default 0.6; 0.9 suits an 8 GB GPU)")
    args = parser.parse_args(argv)
    if args.corpus not in AUDIT_REPORTS:
        parser.error(f"corpus {args.corpus} is frozen: it was parsed before the table header-order fix")
    paths = CORPORA[args.corpus]
    chunks = Path(paths["chunks"])

    run("parse_reports.py", "--output", paths["processed"], "--workers", args.workers)
    run("verify_parsed.py", "--root", paths["processed"], "--output", AUDIT_REPORTS[args.corpus],
        "--source-manifest", "data/manifest.json")
    run("run_chunking_experiment.py", "--input", paths["processed"], "--output", chunks.parents[1],
        "--sizes", "512", "--overlaps", "50", "--workers", args.workers)

    from report_chunker.retrieval import build_index
    print("== chunk store", paths["bm25"], flush=True)
    summary = build_index(ROOT / chunks, ROOT / paths["bm25"])
    print(f"   {summary['chunk_count']} chunks from {summary['document_count']} reports", flush=True)

    memory = ["--gpu-memory-fraction", args.gpu_memory_fraction] if args.gpu_memory_fraction else []
    run("build_vector_index.py", "--chunks", chunks, "--output", paths["vectors"], "--device", args.device,
        "--batch-size", args.batch_size, "--index-types", "flat", *memory)

    from report_search.reranker import BGEReranker
    print("== reranker snapshot", flush=True)
    BGEReranker(device="cpu")._snapshot(weights=True)
    start = ("the demo uses it by default" if args.corpus == DEFAULT_CORPUS
             else f"start the demo with FINANCE_QA_CORPUS={args.corpus}")
    print(f"Corpus {args.corpus} is ready; {start}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
