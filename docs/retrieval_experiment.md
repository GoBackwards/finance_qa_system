# Dense, BM25 and weighted hybrid retrieval

This experiment compares three retrieval approaches on the same **200 reports and 56,525 table-preserving chunks** from the 512-token / 50-token-overlap configuration. It evaluates retrieval of source documents and supporting evidence; it does not generate or grade answers.

| Approach | Implementation | Ranking |
|---|---|---|
| A: vector | BAAI/bge-m3 dense embeddings, 1,024 dimensions, FAISS Flat IP | Cosine similarity of normalized query and passage vectors |
| B: BM25 | SQLite FTS5, `porter unicode61`, `k1=1.2`, `b=0.75` | Lower raw SQLite BM25 score is better |
| C: hybrid | Union of up to 100 candidates from each approach | Weighted sum of normalized scores |

All approaches search the full corpus without gold company/year filters or query decomposition. Only question text enters query encoding and lexical search. Answers, evidence annotations and expected document IDs are evaluator inputs. Passages have the established company/ticker/fiscal-year prefix; BM25 indexes these metadata fields alongside the same chunk text, with equal body/metadata weights. The lexical query uses safely quoted, distinct terms after the fixed stopword filter.

The runner requires an exact Flat index so approximate-neighbor loss does not confound fusion. The existing [vector-index experiment](vector_index.md) covers IVF/HNSW separately. The [retrieval engine](../src/report_search/retrieval.py) verifies source hashes, index manifests, dimensions, and the complete ordered chunk/document mapping before searching.

## Benchmark and review status

The [full benchmark](../data/evaluation/retrieval_questions.json) contains **100 questions: 40 factual, 30 comparative and 30 summary**. Development and test each contain 50 questions: 20 factual, 15 comparative and 15 summary. Each split has 20 companies, four per industry, with no company shared across splits. Labels cover FY2021–FY2025, **190 required evidence units and 101 source reports**, as recorded in the [validation record](../data/evaluation/retrieval_benchmark_validation.json). The other reports remain searchable distractors.

These are individually **agent-authored, source-checked labels awaiting independent human review**. The source authors inspected original parsed documents and native PDF text without consulting ranked retrieval results. This is a draft benchmark, not a claim that 100 questions have been manually labeled by a human. Provenance is preserved in the [development](../data/evaluation/retrieval_dev_questions.json) and [test](../data/evaluation/retrieval_test_questions.json) parts.

Four development items explicitly identify reused evidence from the earlier 40-question pilot (`dev-013`, `dev-015`, `dev-016`, `dev-032`). Test gold spans have no overlap with that pilot. Separate agents checked all 30 comparative answers and sampled summaries against the original sources before viewing retrieval results; this remains agent review, not human certification.

The compact [triplet export](../data/evaluation/retrieval_triplets.jsonl) stores `query`, `ground_truth_doc_ids`, and `answer`, plus question ID, category and split. Document IDs are a **list**, since comparative questions require multiple reports. Use the full JSON for evidence-based evaluation:

| Field | Contract |
|---|---|
| `question_id`, `query`, `question` | Unique ID and question text; the two text aliases must match |
| `category`, `type`, `split` | Matching factual/comparative/summary labels, and dev/test assignment |
| `ground_truth_doc_ids` | Distinct report IDs matching the evidence documents |
| `answer` | Source-grounded reference answer; never indexed or submitted to a retriever |
| `answer_points` | Claims linked to one or more required `evidence_ids` |
| `evidence` | Exact original prose spans, accepted complete table rows, or an original fallback-table text span |
| `provenance`, `pdf_source_validation` | Authoring/review status and source-check details |

Every evidence unit carries document identity, source SHA-256 and a 1-based **PDF page number**. Prose uses `element_id`, exact `quote`, and original `[char_start, char_end)` offsets. Structured tables use `table_id`, zero-based `row_index`, and the complete unchanged `row_cells`/`row_text`. `table_source` evidence uses an exact span of an unstructured table's original `source_text`; unverified candidate cells are never treated as accepted rows. Existing atomic table/page fallbacks remain intact in the chunk corpus.

Comparisons identify the reported metric and units and use the companies' stated fiscal years; fiscal year-end dates can differ. Summary questions specify bounded themes and source-linked answer points. The validator requires two to four distinct evidence units and at least three claims for a summary, and at least two source documents for a comparison.

[Benchmark validation](../src/report_search/benchmark.py) checks schema, claim links, exact source text/rows, original citations, parsed-artifact checksums and split isolation. **A passing validation does not establish semantic answer accuracy, complete relevance judgments or independent human review.**

## Score fusion and development tuning

Raw cosine scores are theoretically in `[-1, 1]`; SQLite BM25 scores use a different scale and the opposite ranking direction. SQLite explicitly negates its BM25 result, assigning better matches lower scores. [SQLite FTS5 documentation](https://sqlite.org/fts5.html#the_bm25_function).

For each query, retrieve up to 100 candidates per channel and deduplicate their union by `chunk_id`. Define dense score `d` as raw cosine and lexical score `b` as **negative raw SQLite BM25**. Normalize each channel separately over its own candidate list, before selecting the final top results:

```text
normalized(s) = (s - min(candidate_scores)) / (max(candidate_scores) - min(candidate_scores))
hybrid_score  = alpha * normalized_dense + (1 - alpha) * normalized_bm25
```

A candidate absent from a channel receives zero for that channel. A nonempty channel whose scores are all tied assigns one to every candidate. An empty channel contributes zero. The normalization is relative to this query's candidate pool; scores are not probabilities or comparable confidence measures across queries. Candidate depth can change the normalized values and final ranking.

At `alpha=0`, only BM25 candidates participate; at `alpha=1`, only vector candidates participate. These endpoints preserve the active raw-score ranking. Other weights sort by descending fused score, then `chunk_id` for deterministic ties. Returned payloads retain citations and a `score_breakdown` containing raw scores, normalized scores and weights. This follows the general normalized linear-fusion pattern described in [Elasticsearch's linear retriever documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever); this project implements it locally with its explicit edge-case policies above.

The development grid is `alpha = 0, 0.25, 0.5, 0.75, 1`. Select the highest **development macro evidence Recall@5**. Ties within an absolute tolerance of `1e-12` choose the weight closest to `0.5`, then the smaller weight. Persist the selected policy in `tuning.json` **before test retrieval evaluation**, then measure that fixed policy on test questions. Reference answers do not participate in this optimization. Do not revise labels or settings in response to test performance while continuing to call the same test set held out.

## Metrics and interpretation

The [evaluator](../src/report_search/experiment.py) reports overall and per-category metrics at **5 and 10 retrieved chunks**, macro-averaged over questions:

- **Document Recall@k:** distinct gold reports represented among the first `k` chunks divided by the number of gold reports. This is not retrieval of `k` distinct documents. Five chunks from one report recover only half of a two-report comparison's required documents.
- **Evidence Recall@k:** required canonical evidence units fully recovered divided by required units. Original prose intervals may be covered by the union of several retrieved chunks; a structured table unit requires its complete row. Fallback-table spans require matching original fallback text and source identity. Duplicate or overlapping hits do not increase credit.
- **All documents / all evidence hit:** the fraction of questions for which every required report or every required evidence unit is recovered.
- **Context tokens:** total existing `cl100k_base` token counts of retrieved chunks, including duplicated overlap. A fixed number of chunks is not a fixed token budget.

Document recall can succeed while the relevant passage is missing; evidence recall exposes that difference. Gold locations are non-exhaustive, so other valid answer passages may receive no evidence credit. Summary coverage measures selected claims, not completeness of an unrestricted summary or generated-answer quality.

Test hybrid-minus-baseline evidence Recall@5 differences include paired question-bootstrap intervals: 10,000 resamples, seed `20261008`, percentile 95% bounds. These are exploratory intervals: questions share companies, templates and source patterns, and the bootstrap does not model that dependence. Companies are disjoint between label splits but all remain in the searchable corpus. The model and chunking configuration were also chosen in earlier exploratory work; this is not an untouched corpus or a production-quality generalization estimate.

## Reproduction

Run from the repository root with the project's `.venv`, downloaded/parsed reports, the chosen chunk file, and the existing BGE-M3 Flat index. See [vector setup](vector_index.md#installation-and-execution) for optional dependencies and index creation. SQLite must support FTS5.

```powershell
# Merge the frozen parts, validate original source evidence, and export triplets.
.\.venv\Scripts\python.exe scripts/prepare_retrieval_benchmark.py

# Encode queries, build/reuse BM25, tune on development, then evaluate test.
.\.venv\Scripts\python.exe scripts/run_retrieval_experiment.py --device cuda:0 --threads 4
```

Use `--device cpu` when CUDA is unavailable. Query embeddings are created in a separate process so model memory is released before index loading. Cached query vectors are reused only when the benchmark, query texts, encoder contract, corpus vectors and cached-vector hash match. A missing BM25 index or different chunk-source hash triggers a rebuild. Runtime validation rejects inconsistent corpus mappings or a non-Flat vector index.

The [runner](../scripts/run_retrieval_experiment.py) exposes these defaults:

| Argument | Default |
|---|---|
| `--benchmark` | `data/evaluation/retrieval_questions.json` |
| `--processed` | `data/processed` |
| `--chunks` | `data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl` |
| `--vector-index` | `data/indexes/bge-m3/table_preserving_s512_o50/flat.faiss` |
| `--embedding-manifest` | `data/indexes/bge-m3/table_preserving_s512_o50/embeddings/embedding_manifest.json` |
| `--bm25-index` | `data/indexes/bm25_s512_o50/retrieval.sqlite` |
| `--output` | `data/experiments/retrieval` |
| `--device`, `--threads` | `cuda:0`, `4` |
| `--candidate-k`, `--alphas` | `100`, `0 0.25 0.5 0.75 1` |

Use a separate output directory for a distinct experiment; rerunning in an existing output directory rewrites its evaluation records. The runner writes `benchmark_validation.json`, `query_vectors.npy`, `query_manifest.json`, `tuning.json`, `per_query.jsonl`, `comparison.json`, and `comparison.md`. Per-query records contain rankings, citation/evidence judgments and score breakdowns. Manifests and the comparison JSON record source/model/artifact hashes and implementation fingerprints. Large vectors, indexes and detailed run artifacts remain local rather than being committed.

## Measured results

The completed run is recorded in [retrieval_comparison.md](retrieval_comparison.md), with the compact machine-readable record in [retrieval_results.json](../data/evaluation/retrieval_results.json). The comparison includes category tradeoffs and uncertainty rather than assuming that hybrid retrieval wins. Full rankings and judgments are retained locally in `data/experiments/retrieval/per_query.jsonl` for error analysis.
