# Chunk-size and overlap experiment

The experiment compares **512, 1,024, and 2,048 tokens**, crossed with **50, 100, and 200 tokens of prose overlap**, for nine configurations. These are starting points to measure, not a universal optimum. The smaller size favors focused facts; larger sizes can retain longer explanations and table context. Libraries expose both size and overlap as tuning parameters, and their length units differ: this implementation explicitly counts BPE tokens. See [LlamaIndex's chunk-size discussion](https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/) and [LangChain's splitter parameters](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

The diagonal pairs 512/50, 1024/100, and 2048/200 each have approximately 10% requested overlap. A complete grid also tests whether increasing overlap helps at a fixed size. For 512-token chunks, 200-token overlap is 39%, so extra duplication should be justified by measured retrieval improvement.

## Run and reproduce

From the repository root:

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[experiments]"
.\.venv\Scripts\python.exe scripts/run_chunking_experiment.py --workers 4
```

The workspace environment already has the required tokenizer. Its `cl100k_base` vocabulary is cached under `.cache/tiktoken`; a fresh installation needs network access once to obtain that vocabulary. No embedding API, paid service, or LLM answer generation is used in this experiment.

To change the grid:

```powershell
.\.venv\Scripts\python.exe scripts/run_chunking_experiment.py --sizes 512 1024 2048 --overlaps 50 100 200
```

Document chunk parts are cached against the parsed-source hashes, chunker and validator code, tokenizer version, and configuration grid. `--force` rebuilds those parts. The CLI regenerates aggregate files and retrieval measurements. `--keep-indexes` retains the SQLite retrieval databases; otherwise they are removed after evaluation and can be rebuilt from the retained chunks.

Outputs are under `data/experiments/chunking/`:

| Output | Purpose |
|---|---|
| `comparison.md` | Requested human-readable comparison table |
| `comparison.json` | Full statistics, metric deltas, confidence intervals, controls, and artifact paths |
| `chunking_manifest.json` | Source and chunk-part fingerprints for reproducibility |
| `table_preserving_s512_o50/chunks.jsonl` | Reusable chunks for one configuration; analogous folders exist for the other eight |
| `statistics.json` within each configuration | Counts, actual lengths, overlaps, overflows, and preservation validation |
| `retrieval_results.json` within each configuration | Unfiltered BM25 top-five results and evidence judgments for every question |
| `retrieval_filtered_diagnostic.json` | Separately labeled company/year-filtered diagnostic, not the primary score |

## Table preservation

Prose is grouped in reading order and split near paragraph or sentence boundaries. It can cross PDF pages, while every fragment retains its original element ID, exact character range, source hash, page number, and bounding box. Overlap stays within a prose stream and does not carry through a table or into another document. Unicode-safe slicing preserves the original text exactly.

Recognized tables are separate from prose:

1. A table that fits is kept together. Larger tables split only **between complete source rows**.
2. Chunks repeat verified source header rows and available captions, date headings, and units. Neutral `Column N` labels are positional scaffolding when source header rows are not identifiable; they are not invented financial meanings.
3. Body rows are not duplicated for overlap. Requested overlap applies to prose; repeated table context is measured separately.
4. A row larger than the remaining budget is kept whole and flagged as oversized. Content is never silently truncated to meet a nominal size.
5. Tables with unreliable extracted structure remain atomic original-text chunks. Their rejected diagnostic cells are never indexed as facts.
6. A page whose table extraction failed is retained as an atomic page chunk. This protects possible rows when there is no trustworthy structure to split.

Independent validation checks every source prose character, exact source offsets, complete original row cells, table/page fallback coverage, source citations, and unique chunk IDs for every document and configuration. Tables that the upstream PDF parser missed entirely can still appear as prose; this stage cannot guarantee the structure of an undetected table. Missing OCR text also remains an upstream limitation.

## Retrieval evaluation

The initial retriever is **SQLite FTS5 BM25**, using Porter stemming over `unicode61`, fixed stopwords, `k1=1.2`, `b=0.75`, and equally weighted body and company/ticker/year metadata. Only the question text enters the primary search, across all 200 reports. The same settings and deterministic tie-breaking apply to every configuration. [SQLite documents its BM25 scoring](https://sqlite.org/fts5.html#the_bm25_function).

The [starter benchmark](chunking_benchmark.md) contains 40 agent-curated questions and 48 required source-evidence units. It was grounded before viewing experiment scores. Reference answers, gold quotes, and target source identifiers are never added to the retrieval index or primary search query.

For each question:

```text
evidence Recall@5 = required source units covered by the top 5 chunks
                   ------------------------------------------------
                              all required source units
```

A table unit requires its complete canonical row. A prose unit requires the union of retrieved source intervals to cover the full gold span. Duplicate overlap and repeated headers do not count twice. The comparison reports the macro average across questions and its change in percentage points from 512/50. It also records narrative/table subsets, Hit@5, all-required-evidence success, and paired bootstrap uncertainty estimates.

Forty questions provide a pilot measurement, not a production-quality ranking of chunkers. Gold judgments are not exhaustive: another valid passage can remain unjudged. Larger chunks deliver more context at fixed top-k, so average retrieved tokens is reported too. Re-run the same benchmark with the eventual embedding/hybrid retriever and a fixed context-token budget; then validate the choice on independently written, held-out questions. Keep filtered and unfiltered scores separate.

Chunk sizes use `cl100k_base` only as a reproducible unit. After selecting an embedding model, check its own tokenizer and input limit, including captions and metadata. Oversized atomic tables/pages need a supported model or a separate retrieval route, rather than silent embedding truncation.

## Measured corpus results — 2026-10-08

The complete run processed all **200 reports** in all nine configurations and evaluated the same 40 questions against each full-corpus BM25 index. Recall below is macro source-evidence Recall@5; delta is percentage points relative to 512/50.

| Target tokens | Prose overlap | Chunks | Average tokens | Recall@5 | Delta (pp) | Average top-5 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 50 | 56,525 | 370.3 | 50.00% | 0.00 | 2,039.5 |
| 512 | 100 | 59,720 | 378.8 | 46.25% | -3.75 | 2,075.1 |
| 512 | 200 | 69,498 | 397.4 | 46.25% | -3.75 | 2,141.8 |
| 1,024 | 50 | 38,311 | 518.8 | 35.00% | -15.00 | 3,625.8 |
| 1,024 | 100 | 38,858 | 527.4 | 32.50% | -17.50 | 3,724.5 |
| 1,024 | 200 | 40,188 | 545.9 | 37.50% | -12.50 | 3,799.9 |
| 2,048 | 50 | 31,278 | 623.6 | 30.00% | -20.00 | 6,271.6 |
| 2,048 | 100 | 31,359 | 629.5 | 30.00% | -20.00 | 6,209.6 |
| 2,048 | 200 | 31,548 | 641.6 | 27.50% | -22.50 | 6,457.1 |

Use **512/50 as the provisional BM25 baseline**. It achieved the highest observed recall, with prose recall of 65% and table recall of 35%. Increasing its overlap to 200 generated about 23% more chunks without increasing recall. The 40-question sample remains small: the baseline's bootstrap 95% recall interval is 35%–65%, and the paired intervals for the other 512-token settings include zero improvement. These measurements do not establish a generally optimal setting or predict an embedding retriever's ranking.

All **1,800 document/configuration validations passed**, with zero missing source characters or missing/split recognized table rows. The 512-, 1,024-, and 2,048-token groups contain 400, 97, and 4 oversized chunks respectively, independent of overlap; these preserve whole unverified tables or failed-table-scan pages. Average lengths are below the nominal sizes because table boundaries and short prose runs are retained. The complete suite passed **84 tests**.

The generated [comparison report](../data/experiments/chunking/comparison.md) additionally records average character length and oversize counts. [Machine-readable results](../data/experiments/chunking/comparison.json) include uncertainty intervals and artifact locations; the benchmark and individual retrieval results remain available for the next embedding/hybrid experiment. Generated chunk and retrieval artifacts are intentionally excluded from Git by `data/experiments/`.
