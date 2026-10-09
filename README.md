# Financial report Q&A corpus

This project contains 200 annual-report PDFs for 40 companies, FY2021–FY2025, and a local, page-grounded parsing pipeline. Source provenance is in `data/manifest.json`; collection details are in [data/README.md](data/README.md).

The parser uses **PyMuPDF for native text** and **pdfplumber for tables**. It adapts DeepDoc's separation of layout analysis, table extraction, and text assembly. It uses transparent geometric heuristics; it does not run DeepDoc's trained layout models or OCR. See [the source review and design notes](docs/deepdoc_reference.md).

## Run

Python 3.11+ is required. From the project root, install the package in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The workspace's `.venv` is already configured. Parse the full downloaded corpus:

```powershell
.\.venv\Scripts\python.exe scripts/parse_reports.py --workers 4
```

Useful scoped runs:

```powershell
# One company/year
.\.venv\Scripts\python.exe scripts/parse_reports.py --tickers NVDA --years 2025

# Specific PDF pages; written to a separate partial-output folder
.\.venv\Scripts\python.exe scripts/parse_reports.py --tickers NVDA --years 2025 --pages 140-150

# Try table extraction on every page, bypassing the native-text candidate gate
.\.venv\Scripts\python.exe scripts/parse_reports.py --tickers MSFT --years 2025 --table-mode all

# A PDF outside the collection
.\.venv\Scripts\python.exe scripts/parse_reports.py --input path/to/report.pdf

# Tests and output-integrity audit
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_parsed.py --source-manifest data/manifest.json
```

`--table-mode auto` is the default: pages with at least six native text lines containing digits are scanned for tables. `all` tries every page; `off` extracts text only. Even `all` cannot guarantee every table is detected. `--force` reparses matching cached outputs. Processing uses separate worker processes because PDF engine state must not be shared between threads.

Successful runs are resumable. The parser checks the PDF hash, source metadata, parser code, configuration, dependency versions, and output-file hashes before reuse. The completion marker is committed after the output files. Do not run two parser commands against the same output folder concurrently.

## Output

```text
data/processed/
├── parse_manifest.json
└── NVDA/
    └── NVDA_FY2025_annual_report/
        ├── document.json
        ├── pages.jsonl
        ├── tables.jsonl
        └── document.md
```

| File | Contents |
|---|---|
| `document.json` | Source URLs and hash, ticker/year, page count, parser settings, versions, output hashes, extraction counts, and review warnings |
| `pages.jsonl` | One record per PDF page: ordered text/table elements, original text lines, removed margins, table references, and quality flags |
| `tables.jsonl` | One record per detected table: rows/cells and coordinates when accepted, Markdown or original-text fallback, nearby captions/units, source evidence, strategy, and warnings |
| `document.md` | Human-readable text and tables under explicit PDF-page headings |
| `parse_manifest.json` | Corpus run registry, including failures and partial runs |

Every paragraph and table carries a citation:

```json
{
  "document_id": "NVDA_FY2025_annual_report-<hash-prefix>",
  "source_file": "NVDA_FY2025_annual_report.pdf",
  "sha256": "<full-source-sha256>",
  "ticker": "NVDA",
  "fiscal_year": 2025,
  "pdf_page": 144,
  "bbox": [40.0, 90.0, 550.0, 720.0]
}
```

Coordinates above are illustrative. Actual boxes are in PDF points, relative to the top-left of the displayed, cropped page. **`pdf_page` is always 1-based PDF position**, not the printed page numeral. This matters for bundled proxies, covers, and publisher inserts. `pdf_page_label` is optional PDF metadata; it is not recognition of printed numerals. Keep these citation fields when building retrieval chunks.

## Parsing decisions

1. Extract native lines, font sizes, directions, and bounding boxes. Flag duplicated drawing operations without discarding the source lines.
2. Compare marginal text across the document. Label recurring headers/footers conservatively using position, font size, and normalized text; protect body paragraphs and prominent titles. Removed material remains in `source_lines` and `removed_margins` for audit.
3. Extract ruled tables, then search constrained regions for borderless tables with aligned numeric columns. Preserve displayed values as strings, including parentheses, currencies, percentages, and dashes. Do not infer units or convert values to floats.
4. Compare numeric-token counts in the inferred cells with native source text. If numbers were split, added, or lost, empty the usable `rows`/`cells` and emit original text as the table's readable fallback. Preserve rejected cells under `candidate_rows`/`candidate_cells` for debugging only. Keep table regions indivisible and preserve overlapping native lines separately for audit.
5. Order page regions using whitespace and column geometry, then join compatible lines within each region. Paragraphs never merge across PDF pages; page references remain explicit.
6. Emit review warnings for sparse text, suspect encoding, uncertain tables, rotation, or extraction failures. A table failure keeps native text available.

## Quality limits

`complete: true` means the configured extraction finished; it does **not** mean that every page or table is accurate. Inspect `review_pages`, page `warnings`, and `table_scan`. The integrity audit checks bookkeeping and provenance, not semantic accuracy.

The audit writes `data/parsed_validation_report.json`, including `sparse_pages` and `table_error_pages` for follow-up. With `--source-manifest`, it also requires full coverage of the downloaded corpus and compares source hashes, tickers, fiscal years, and page counts. Omit that option when auditing an intentionally limited subset.

- Scanned/image-only content requires a later OCR stage. Sparse covers may be intentionally empty; the parser flags both cases for review.
- Complex magazine spreads, sidebars, charts, text embedded in figures, and unusual typography may need learned layout analysis. Column ordering is heuristic.
- Borderless statements can split into sections. Multiline/spanning cells and physical currency-symbol columns can need reconciliation. Preserve `caption`, `context`, and `source_text` alongside rows when indexing tables.
- `structured_reliable: false` means structured cells failed the numeric-integrity check: use the original-text fallback and review the table. Never index rejected `candidate_rows`/`candidate_cells` as facts. `true` only indicates that this check passed; it does not guarantee correct column/year associations, units, signs, or table boundaries. Footnote markers can conservatively trigger fallback too.
- Rotated pages retain display-aligned text coordinates and are flagged. Rotated-table structure extraction is currently skipped with an explicit error warning.
- Header/footer recurrence is conservative and cannot guarantee removal of all marginal text. Original lines remain available.
- Table `confidence` is a ranking heuristic, not a calibrated accuracy probability. Candidate limits fail visibly instead of silently truncating a table search.
- The table stage limits pages to 4,000 extracted words and 2,500 visible vector edges, and limits borderless candidates to 12. A limit failure marks `table_scan: "error"` and retains native page text. `--table-mode all` bypasses the numeric-page gate, not these complexity limits.

These outputs provide source-grounded input for chunking and retrieval. Answer generation is a later stage.

The next stage is available in [the chunking experiment](docs/chunking_experiment.md): nine table-preserving size/overlap combinations, reusable chunks, a source-grounded starter benchmark, and local BM25 Recall@5 measurements. Run `scripts/run_chunking_experiment.py` after installing the `experiments` dependency extra.

The [dense embedding and FAISS stage](docs/vector_index.md) adds pinned BGE models, model-token length audits, resumable GPU/CPU embeddings, an exact Flat index, optional IVF/HNSW comparisons, and vector-to-source citations. Run `scripts/build_vector_index.py` after installing the `vectors` dependency extra.

The [retrieval experiment](docs/retrieval_experiment.md) compares FAISS Flat, BM25, and normalized weighted score fusion on a [100-question benchmark](data/evaluation/retrieval_questions.json). It contains 40 factual, 30 comparative, and 30 summary questions with answers and page-grounded evidence. Development and test questions use different companies; hybrid weights are selected on development evidence recall before test scoring. Run `scripts/run_retrieval_experiment.py` with the existing chunks and dense index. See the [measured comparison](docs/retrieval_comparison.md). These agent-authored labels have been source-checked and still need independent human review.

The [Recall@3/5/10/20/50 sweep](docs/recall_at_k_comparison.md) extends that experiment without changing its labels or candidate pool. The [FAISS index comparison](docs/faiss_index_comparison.md) measures fresh Flat/IVF/HNSW builds, single-query latency, resident RAM, and ANN neighbor Recall@10 against exact Flat. It also reports source-evidence recall separately. Run `scripts/compare_faiss_indexes.py` to reproduce the isolated index measurements.

The [BGE reranker stage](docs/reranker.md) uses FlagEmbedding's pinned `bge-reranker-v2-m3` to select five passages from the unchanged vector Top 20. On the 50 frozen test questions, evidence Recall@5 increased from 37.33% to 48.00%, with 388.8 ms median added GPU scoring time per query. See the [measured comparison](docs/reranker_comparison.md) and [machine-readable results](data/evaluation/reranker_results.json). Run `scripts/run_reranker_experiment.py` to reproduce or resume scoring. Accuracy here measures source selection; generated-answer accuracy remains unevaluated. The reranker stage was validated with 212 tests.

The [generation constraints](docs/generation_constraints.md) add source paragraph/table citations, an explicit raw-score refusal threshold, exact reference validation, and a separate model review of every claim. Failed checks return **"I'm not sure"** without exposing an unverified draft. Use `scripts/answer_question.py` with a configured OpenAI-compatible endpoint and an explicitly selected threshold or diagnostic policy. The [development-only gate diagnostic](data/evaluation/generation_gate_results.json) and [offline integration checks](data/evaluation/generation_checks.json) are recorded separately; no live generated-answer accuracy has been measured. That stage passed 287 automated tests and 11 controlled HTTP integration cases.

The [failure-handling stage](docs/failure_handling.md) adds recognized company/report scope checks, optional bounded lexical recovery, a passage cache that preserves answer verification, transient-provider retries, request budgets and stage timings. Run `scripts/answer_reliably.py` with quoted questions, a configured endpoint and an explicitly evaluated score threshold; `scripts/inspect_query.py` previews query planning without models. The [development candidate comparison](data/evaluation/query_expansion_results.json) found Evidence Recall@20 of 74.67% for vector plus original-query BM25 versus 72.67% after adding financial hints, so expansion remains opt-in. The [30-case behavior catalog](data/evaluation/failure_cases.json) and [118 controlled regression checks](data/evaluation/failure_handling_checks.json) distinguish intended behavior from observed retrieval results. The full suite passes 358 tests; live answer quality and production latency remain unevaluated.

## Validated corpus run — 2026-10-08

- All 200 reports, 40 tickers, and 23,280 PDF pages processed; no document-level failures.
- 13,141 detected table regions: 12,055 passed numeric consistency checks; 1,086 use original-text fallbacks.
- All 1,639,474 active native text lines accounted for. Source-manifest coverage, metadata, citation references, and artifact hashes passed with zero integrity errors.
- 43 parser tests pass (127 tests across the project after adding chunking, embeddings, and retrieval). Rendered spot checks covered complex NVIDIA spreads, Microsoft headings/bullets/statements, and financial statements from Coca-Cola, Walmart, and Caterpillar. These are sample checks, not a corpus-wide semantic-accuracy benchmark.
- 324 pages have sparse-text or encoding warnings that can require OCR/review. Table extraction produced explicit warnings on 313 pages; native text remains available. Other layout/table review flags are recorded separately.

The machine-readable results are in [data/parsed_validation_report.json](data/parsed_validation_report.json). A real cached rerun of NVIDIA FY2025 reused the verified artifacts successfully.
