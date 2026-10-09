# FAISS index comparison

Fresh builds over 56,525 normalized 1024-dimensional BAAI/bge-m3 vectors from 200 annual reports. Same 50 test questions and fixed configurations for every index.

## Requested comparison

**Recall@10 here is ANN neighbor recall:** overlap with exact Flat's top ten vector IDs, averaged over queries. It measures approximation loss; financial evidence recall is reported separately below.

| Index | ANN Recall@10 | Query Latency (ms) | Build Time (s) | Memory Usage (MB) |
|---|---:|---:|---:|---:|
| Flat | 100.00% | 21.997 | 0.192 | 231.67 |
| IVF | 100.00% | 0.672 | 1.398 | 233.59 |
| HNSW | 95.00% | 0.101 | 25.442 | 247.15 |

On this sample, IVF reproduces Flat's top-ten neighbors while reducing median index-search latency by about 33 times. HNSW is faster again, with 95% neighbor recall and 48% source-evidence Recall@10 versus 50% for Flat/IVF. These are fixed-setting measurements on 50 queries, not guarantees for other workloads. Keep Flat as the exact reference when investigating approximation loss.

Query latency is the median of single-query CPU searches at k=10, four FAISS threads, two warmup passes and five timed repetitions per query. It excludes embedding, loading, validation, sorting and source-payload lookup. Build time is training plus vector insertion; end-to-end pipeline times are below. Memory is the measured process resident-memory increase after loading the index in an isolated process; MB = 1,000,000 bytes. It is neither serialized file size nor peak construction memory.

## ANN neighbor recall across cutoffs

| Index | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| flat | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| ivf | 100.00% | 100.00% | 100.00% | 100.00% | 99.92% |
| hnsw | 93.33% | 93.60% | 95.00% | 94.60% | 93.68% |

Each cutoff uses its own native search with the same k for Flat and the approximate index. This differs from the nested top-100 ranking prefixes in the [retriever recall sweep](recall_at_k_comparison.md). Equal-score boundary ties can reduce strict ID-overlap recall even when candidates are equally similar.

This run had no exact Flat boundary ties at the measured cutoffs. HNSW omitted 25 of the 500 exact top-ten neighbors across eight questions; its replacements had lower cosine scores, so the measured loss reflects approximation. IVF omitted two of 2,500 exact neighbors at @50, while matching all exact neighbors at smaller cutoffs.

## Source-evidence recall on the same queries

| Index | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| flat | 26.00% | 37.33% | 50.00% | 60.00% | 77.67% |
| ivf | 26.00% | 37.33% | 50.00% | 60.00% | 77.67% |
| hnsw | 27.00% | 37.33% | 48.00% | 57.00% | 72.67% |

These are macro fractions of required canonical source spans or complete table rows recovered. Flat's ANN recall is 100% by definition, while its evidence recall need not be. The agent-authored labels are non-exhaustive and await independent human review.

## Timing and storage details

| Index | Query p95 (ms) | Build pipeline (s) | Index file (MB) | Warmed RAM delta (MB) |
|---|---:|---:|---:|---:|
| flat | 26.757 | 5.498 | 231.53 | 234.71 |
| ivf | 1.101 | 7.093 | 233.03 | 233.73 |
| hnsw | 0.280 | 30.725 | 246.90 | 247.36 |

The build pipeline includes input checks, source mapping, construction and persistence. Construction threads are flat: 4; ivf: 4; hnsw: 1. HNSW uses one for reproducible graph insertion. This is a fixed-configuration engineering comparison on one machine, not an equally parallelized build-speed contest or production load test.

Fixed settings: IVF nlist=256, nprobe=16; HNSW M=32, efConstruction=200, efSearch=64; seed=20261008. These settings came from the earlier index run; no test-based tuning or model re-embedding was performed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/compare_faiss_indexes.py
```

The script verifies the frozen benchmark/query/corpus contracts, rebuilds each index in a fresh process, then measures it in a separate fresh process. The parent resolves citations after timing. Artifacts remain in `data/experiments/faiss_comparison/`; the compact [result record](../data/evaluation/faiss_comparison_results.json) contains full settings, source/model/index hashes, memory counters, latency samples, per-query ANN recall and category evidence metrics.

References: [FAISS index types and memory](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [ties and batching](https://github.com/facebookresearch/faiss/wiki/FAQ), [thread configuration](https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster).
