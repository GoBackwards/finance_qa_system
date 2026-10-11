# PDF parsing

The parser turns the 200 annual-report PDFs (40 companies, FY2021–FY2025) into page-grounded text and tables. Source provenance is in `data/manifest.json`; collection details are in [data/README.md](../data/README.md).

The parser uses **PyMuPDF for native text** and **pdfplumber for tables**. It adapts DeepDoc's separation of layout analysis, table extraction, and text assembly. It uses transparent geometric heuristics; it does not run DeepDoc's trained layout models or OCR. See [the source review and design notes](deepdoc_reference.md). Its output feeds the [table-preserving chunker](chunking_experiment.md); the rest of the pipeline is described in [system_architecture.md](system_architecture.md).

## Run

Python 3.11+ is required. The parser needs only the base dependencies. From the project root, install the package in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Parse the full downloaded corpus into the demo corpus folder (corpus v2; `scripts/build_corpus.py` runs this together with the later stages):

```powershell
.\.venv\Scripts\python.exe scripts/parse_reports.py --output data/processed_v2 --workers 4
```

Always pass `--output`. The parser's own default is `data/processed`, the folder of corpus v1 behind Tables A1–A13; this parser's code fingerprint differs from v1's, so every v1 report would be re-parsed and overwritten, and v1 can only be rebuilt with the parser from commit `435b7ae` (Table A14). `scripts/parse_reports.py` therefore refuses to run without `--output` or with `--output data/processed`. Likewise, `verify_parsed.py` without `--root` and `--output` audits v1 and overwrites its recorded audit, `data/parsed_validation_report.json`.

Useful scoped runs:

```powershell
# One company/year
.\.venv\Scripts\python.exe scripts/parse_reports.py --output data/processed_v2 --tickers NVDA --years 2025

# Specific PDF pages; a partial parse, so keep it out of data/processed_v2 (the chunker rejects partial parses)
.\.venv\Scripts\python.exe scripts/parse_reports.py --output tmp/parsed_pages --tickers NVDA --years 2025 --pages 140-150

# Try table extraction on every page, bypassing the native-text candidate gate
.\.venv\Scripts\python.exe scripts/parse_reports.py --output tmp/parsed_table_mode_all --tickers MSFT --years 2025 --table-mode all

# A PDF outside the collection
.\.venv\Scripts\python.exe scripts/parse_reports.py --output tmp/parsed_other --input path/to/report.pdf

# Parser tests (base dependencies only) and output-integrity audit
.\.venv\Scripts\python.exe -m unittest tests.test_layout tests.test_pipeline tests.test_tables -v
.\.venv\Scripts\python.exe scripts/verify_parsed.py --root data/processed_v2 `
  --output data/parsed_validation_report_v2.json --source-manifest data/manifest.json
```

The full suite (`-m unittest discover -s tests`) also needs the `experiments`, `vectors` and `demo` extras.

`--table-mode auto` is the default: pages with at least six native text lines containing digits are scanned for tables. `all` tries every page; `off` extracts text only. Even `all` cannot guarantee that every table is detected. `--force` reparses matching cached outputs. Processing uses separate worker processes because PDF engine state must not be shared between threads.

Successful runs are resumable. Before reusing an output, the parser checks the PDF hash, source metadata, parser code, configuration, dependency versions, and output-file hashes. The completion marker is committed after the output files. Do not run two parser commands against the same output folder concurrently.

## Output

Corpus v2 is laid out as below under `data/processed_v2/`; corpus v1 uses the same layout under `data/processed/`.

```text
data/processed_v2/
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

Coordinates above are illustrative. Actual boxes are in PDF points, relative to the top-left of the displayed, cropped page. **`pdf_page` is always the 1-based PDF position**, not the printed page numeral. This matters for bundled proxies, covers, and publisher inserts. `pdf_page_label` is optional PDF metadata, not recognition of printed numerals. Keep these citation fields when building retrieval chunks.

## Parsing decisions

1. Extract native lines, font sizes, directions, and bounding boxes. Flag duplicated drawing operations without discarding the source lines.
2. Compare marginal text across the document. Label recurring headers/footers conservatively using position, font size, and normalized text; protect body paragraphs and prominent titles. Removed material remains in `source_lines` and `removed_margins` for audit.
3. Extract ruled tables, then search constrained regions for borderless tables with aligned numeric columns. Preserve displayed values as strings, including parentheses, currencies, percentages, and dashes. Do not infer units or convert values to floats.
4. Compare numeric-token counts in the inferred cells with native source text. If numbers were split, added, or lost, empty the usable `rows`/`cells` and emit original text as the table's readable fallback. Preserve rejected cells under `candidate_rows`/`candidate_cells` for debugging only. Keep table regions indivisible and preserve overlapping native lines separately for audit.
5. Order page regions using whitespace and column geometry, then join compatible lines within each region. Paragraphs never merge across PDF pages; page references remain explicit.
6. Emit review warnings for sparse text, suspect encoding, uncertain tables, rotation, or extraction failures. A table failure keeps native text available.

## Quality limits

`complete: true` means the configured extraction finished; it does **not** mean that every page or table is accurate. Inspect `review_pages`, page `warnings`, and `table_scan`. The integrity audit checks bookkeeping and provenance, not semantic accuracy.

The audit writes the file named by `--output` (`data/parsed_validation_report.json` is the recorded v1 audit and the default; the command above writes the recorded v2 audit, `data/parsed_validation_report_v2.json`), including `sparse_pages` and `table_error_pages` for follow-up. With `--source-manifest`, it also requires full coverage of the downloaded corpus and compares source hashes, tickers, fiscal years, and page counts. Omit that option when auditing an intentionally limited subset.

- Scanned/image-only content requires a later OCR stage. Sparse covers may be intentionally empty; the parser flags both cases for review.
- Complex magazine spreads, sidebars, charts, text embedded in figures, and unusual typography may need learned layout analysis. Column ordering is heuristic.
- Borderless statements can split into sections. Multiline/spanning cells and physical currency-symbol columns can need reconciliation. Preserve `caption`, `context`, and `source_text` alongside rows when indexing tables.
- `structured_reliable: false` means structured cells failed the numeric-integrity check: use the original-text fallback and review the table. Never index rejected `candidate_rows`/`candidate_cells` as facts. `true` only indicates that this check passed; it does not guarantee correct column/year associations, units, signs, or table boundaries. Footnote markers can also trigger the fallback conservatively.
- Table `context` items (captions, units and year headers above a table) are grouped into lines (y-centres within 2 pt) and each line is read left to right. Earlier output, which the published corpus v1 was built from, ordered them by exact y, so same-line year headers could print as '2023 2024' over columns 2024 | 2023; corpus v2 is re-parsed with the fix ([architecture Table A14](system_architecture.md)).
- Rotated pages retain display-aligned text coordinates and are flagged. Table-structure extraction on rotated pages is skipped with an explicit error warning.
- Header/footer recurrence is conservative and cannot guarantee removal of all marginal text. Original lines remain available.
- Table `confidence` is a ranking heuristic, not a calibrated accuracy probability. Candidate limits fail visibly instead of silently truncating a table search.
- The table stage limits pages to 4,000 extracted words and 2,500 visible vector edges, and limits borderless candidates to 12. A limit failure marks `table_scan: "error"` and retains native page text. `--table-mode all` bypasses the numeric-page gate, not these complexity limits.

These outputs provide the source-grounded input for [chunking](chunking_experiment.md) and retrieval.

## Validated corpus run — 2026-10-08

- All 200 reports, 40 tickers, and 23,280 PDF pages processed; no document-level failures.
- 13,141 detected table regions: 12,055 passed numeric consistency checks; 1,086 use original-text fallbacks.
- All 1,639,474 active native text lines accounted for. Source-manifest coverage, metadata, citation references, and artifact hashes passed with zero integrity errors.
- The 43 parser tests passed at the time of this run. Rendered spot checks covered complex NVIDIA spreads, Microsoft headings/bullets/statements, and financial statements from Coca-Cola, Walmart, and Caterpillar. These are sample checks, not a corpus-wide semantic-accuracy benchmark.
- 324 pages have sparse-text or encoding warnings that can require OCR/review. Table extraction produced explicit warnings on 313 pages; native text remains available. Other layout/table review flags are recorded separately.

The machine-readable results are in [data/parsed_validation_report.json](../data/parsed_validation_report.json). A real cached rerun of NVIDIA FY2025 reused the verified artifacts successfully.
