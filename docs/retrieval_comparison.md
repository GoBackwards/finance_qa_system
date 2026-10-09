# Retrieval comparison

All 200 reports / 56,525 chunks. BAAI/bge-m3 + FAISS Flat, SQLite FTS5 BM25, and weighted score fusion (vector weight 0.5).

The hybrid weight was selected using development evidence Recall@5 and frozen before test evaluation.

## Test results

| Category | Approach | Questions | Document Recall@5 | Evidence Recall@5 | All evidence@5 | Evidence Recall@10 | Mean top-5 tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | vector | 50 | 87.00% | 37.33% | 26.00% | 50.00% | 1807 |
| overall | bm25 | 50 | 73.00% | 11.33% | 10.00% | 15.33% | 2159 |
| overall | hybrid | 50 | 86.00% | 40.33% | 32.00% | 45.33% | 1970 |
| factual | vector | 20 | 100.00% | 65.00% | 65.00% | 80.00% | 1881 |
| factual | bm25 | 20 | 100.00% | 25.00% | 25.00% | 30.00% | 1794 |
| factual | hybrid | 20 | 100.00% | 80.00% | 80.00% | 80.00% | 1810 |
| comparative | vector | 15 | 63.33% | 20.00% | 0.00% | 33.33% | 1212 |
| comparative | bm25 | 15 | 50.00% | 0.00% | 0.00% | 0.00% | 2392 |
| comparative | hybrid | 15 | 60.00% | 10.00% | 0.00% | 20.00% | 1766 |
| summary | vector | 15 | 93.33% | 17.78% | 0.00% | 26.67% | 2302 |
| summary | bm25 | 15 | 60.00% | 4.44% | 0.00% | 11.11% | 2412 |
| summary | hybrid | 15 | 93.33% | 17.78% | 0.00% | 24.44% | 2386 |

Document recall counts distinct gold reports found within the top five **chunks**. Evidence recall requires the annotated source spans or complete table rows. Both are macro averages.

Hybrid improves test evidence Recall@5 by **3.00 percentage points** over vector retrieval, but the exploratory paired-query 95% bootstrap interval is **−4.33 to +11.33 points**. This small benchmark does not establish a reliable advantage over vector retrieval. Its gain over BM25 is **29.00 points** (interval **+18.33 to +40.67**). The bootstrap uses 10,000 draws, seed 20261008, and does not account for dependence between questions sharing a company or template.

The overall hybrid gain comes from factual questions. Vector retrieval is stronger on comparative evidence at both cutoffs and on overall Recall@10. Every approach has **0% complete-evidence recovery at five chunks for comparative and summary questions**. Finding the right report is much easier than finding all the passages needed for an answer. Per-company retrieval for comparative questions and retrieval/reranking of several summary subtopics are useful next experiments; the current benchmark has not evaluated those changes.

Measured run: 2026-10-09 UTC (2026-10-08 Pacific). See [machine-readable results](../data/evaluation/retrieval_results.json) for source, benchmark, model, query-vector and implementation hashes, category metrics, and paired intervals. Full per-query results remain local in `data/experiments/retrieval/per_query.jsonl`. The benchmark is [here](../data/evaluation/retrieval_questions.json), and the [methodology](retrieval_experiment.md) describes reproduction and limitations.

## Development-only weight selection

| Vector weight | BM25 weight | Dev evidence Recall@5 |
|---:|---:|---:|
| 0 | 1 | 23.33% |
| 0.25 | 0.75 | 32.00% |
| 0.5 | 0.5 | 46.67% |
| 0.75 | 0.25 | 45.33% |
| 1 | 0 | 38.67% |

## Limitations

- Agent-authored, source-checked labels await independent human semantic review.
- Small company-disjoint question split; companies share templates and remain in the searchable corpus.
- Gold passages are non-exhaustive; valid unjudged passages receive no evidence credit.
- Summary coverage measures selected claims, not answer completeness or generated-answer quality.
- A fixed chunk count does not equal a fixed context-token budget; context length is reported.
- No test-label tuning; model/chunking were chosen in earlier exploratory work, not on a pristine unseen corpus.
