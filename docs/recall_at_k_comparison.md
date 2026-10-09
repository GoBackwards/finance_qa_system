# Recall@3, @5, @10, @20 and @50 comparison

All 200 reports / 56,525 chunks. BAAI/bge-m3 + FAISS Flat, SQLite FTS5 BM25, and weighted score fusion (vector weight 0.5).

The hybrid weight was selected using development evidence Recall@5 and frozen before test evaluation.

Each cutoff uses a prefix of the same ranking. Increasing k does not retune the hybrid weight or change its candidate pool.

The sweep preserves the earlier @5/@10 results exactly. Hybrid leads overall evidence recall at @3 and @5; vector leads at @10, @20 and @50. At @50, mean retrieved context grows to about 17,005 tokens for vector, 18,850 for hybrid and 21,676 for BM25. Higher recall therefore comes with more context to process. These measurements do not establish an optimal generation context length or answer quality.

See the [machine-readable record](../data/evaluation/recall_at_k_results.json) for development/test category metrics and hashes, and the separate [FAISS comparison](faiss_index_comparison.md) for approximation recall, search latency, construction time and measured resident RAM. The benchmark and query embeddings are unchanged. The test set contains 50 questions; 50 development questions select the 0.5 vector / 0.5 BM25 weight at @5.

## Test evidence recall

| Category | Approach | Questions | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | vector | 50 | 26.00% | 37.33% | 50.00% | 60.00% | 77.67% |
| overall | bm25 | 50 | 10.67% | 11.33% | 15.33% | 28.67% | 40.00% |
| overall | hybrid | 50 | 31.00% | 40.33% | 45.33% | 56.67% | 75.00% |
| factual | vector | 20 | 45.00% | 65.00% | 80.00% | 80.00% | 100.00% |
| factual | bm25 | 20 | 25.00% | 25.00% | 30.00% | 55.00% | 80.00% |
| factual | hybrid | 20 | 70.00% | 80.00% | 80.00% | 85.00% | 100.00% |
| comparative | vector | 15 | 13.33% | 20.00% | 33.33% | 53.33% | 70.00% |
| comparative | bm25 | 15 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| comparative | hybrid | 15 | 3.33% | 10.00% | 20.00% | 40.00% | 56.67% |
| summary | vector | 15 | 13.33% | 17.78% | 26.67% | 40.00% | 55.56% |
| summary | bm25 | 15 | 2.22% | 4.44% | 11.11% | 22.22% | 26.67% |
| summary | hybrid | 15 | 6.67% | 17.78% | 24.44% | 35.56% | 60.00% |

## Test document recall

| Category | Approach | Questions | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | vector | 50 | 83.00% | 87.00% | 92.00% | 96.00% | 100.00% |
| overall | bm25 | 50 | 61.00% | 73.00% | 82.00% | 94.00% | 98.00% |
| overall | hybrid | 50 | 79.00% | 86.00% | 92.00% | 96.00% | 99.00% |
| factual | vector | 20 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| factual | bm25 | 20 | 95.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| factual | hybrid | 20 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| comparative | vector | 15 | 56.67% | 63.33% | 80.00% | 86.67% | 100.00% |
| comparative | bm25 | 15 | 36.67% | 50.00% | 66.67% | 80.00% | 93.33% |
| comparative | hybrid | 15 | 56.67% | 60.00% | 73.33% | 86.67% | 96.67% |
| summary | vector | 15 | 86.67% | 93.33% | 93.33% | 100.00% | 100.00% |
| summary | bm25 | 15 | 40.00% | 60.00% | 73.33% | 100.00% | 100.00% |
| summary | hybrid | 15 | 73.33% | 93.33% | 100.00% | 100.00% | 100.00% |

## Test complete evidence coverage

| Category | Approach | Questions | All evidence@3 | All evidence@5 | All evidence@10 | All evidence@20 | All evidence@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | vector | 50 | 18.00% | 26.00% | 38.00% | 44.00% | 64.00% |
| overall | bm25 | 50 | 10.00% | 10.00% | 12.00% | 22.00% | 32.00% |
| overall | hybrid | 50 | 28.00% | 32.00% | 32.00% | 44.00% | 60.00% |
| factual | vector | 20 | 45.00% | 65.00% | 80.00% | 80.00% | 100.00% |
| factual | bm25 | 20 | 25.00% | 25.00% | 30.00% | 55.00% | 80.00% |
| factual | hybrid | 20 | 70.00% | 80.00% | 80.00% | 85.00% | 100.00% |
| comparative | vector | 15 | 0.00% | 0.00% | 13.33% | 33.33% | 46.67% |
| comparative | bm25 | 15 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| comparative | hybrid | 15 | 0.00% | 0.00% | 0.00% | 26.67% | 33.33% |
| summary | vector | 15 | 0.00% | 0.00% | 6.67% | 6.67% | 33.33% |
| summary | bm25 | 15 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| summary | hybrid | 15 | 0.00% | 0.00% | 0.00% | 6.67% | 33.33% |

## Test retrieved context size

| Category | Approach | Questions | Mean tokens@3 | Mean tokens@5 | Mean tokens@10 | Mean tokens@20 | Mean tokens@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | vector | 50 | 1073 | 1807 | 3562 | 6952 | 17005 |
| overall | bm25 | 50 | 1295 | 2159 | 4340 | 8670 | 21676 |
| overall | hybrid | 50 | 1201 | 1970 | 3879 | 7743 | 18850 |
| factual | vector | 20 | 1120 | 1881 | 3725 | 7246 | 17078 |
| factual | bm25 | 20 | 1082 | 1794 | 3704 | 7487 | 18576 |
| factual | hybrid | 20 | 1112 | 1810 | 3609 | 7306 | 17880 |
| comparative | vector | 15 | 710 | 1212 | 2336 | 4540 | 12312 |
| comparative | bm25 | 15 | 1427 | 2392 | 4700 | 9345 | 23414 |
| comparative | hybrid | 15 | 1085 | 1766 | 3369 | 6644 | 16228 |
| summary | vector | 15 | 1373 | 2302 | 4571 | 8973 | 21601 |
| summary | bm25 | 15 | 1448 | 2412 | 4828 | 9571 | 24073 |
| summary | hybrid | 15 | 1436 | 2386 | 4750 | 9423 | 22765 |

Document recall counts distinct gold reports found within the first k **chunks**. Evidence recall requires the annotated source spans or complete table rows. Both are macro averages. Complete evidence coverage is the fraction of questions whose required evidence is fully recovered. Context size sums the retrieved chunks' cl100k_base token counts, including overlap.

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
