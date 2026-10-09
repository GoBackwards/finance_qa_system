# Failure handling for financial-report Q&A

The application now has a bounded recovery controller around retrieval, reranking and [evidence-bound generation](generation_constraints.md). It records where a request failed, attempts only configured recovery, and returns `I'm not sure` when it cannot release a verified answer. The public response does not expose a partially checked draft.

| Failure type | Example and diagnosis | Handling | What to measure |
|---|---|---|---|
| Retrieval miss | A revenue question uses different vocabulary from the filing. A terminology mismatch is a hypothesis, not an established cause for every missed answer. | Preserve the original question; optional BM25 recovery adds at most two financial hints, combines bounded candidate lists by reciprocal rank, then reranks once. | Evidence Recall@5/@20 on development questions; improved and worsened queries; additional time and pairs scored. |
| Irrelevant retrieval | A gross-margin question returns sales text, another company, or the wrong requested filing. | Check recognized company and explicit filing-year scope; keep the existing table-preserving chunks and cross-encoder; require claim-level metric/basis verification. | Correct company/report coverage, annotated evidence precision, complete comparison coverage, false refusals. |
| Unsupported answer | The draft invents a forecast, swaps GAAP for non-GAAP, or attaches a correct quote to a false claim. | Apply the raw-logit gate, canonical source binding, exact quote checks and a separate semantic review. Refuse missing essential parts. | With a configured model and human labels: unsupported-answer acceptance, answer correctness, refusal precision and answerable-query coverage. |
| High latency or dependency failure | Retrieval, reranking or either model call consumes the request budget; the provider times out or rate-limits. | Bound candidate work and retries, cache only ranked passages, share one request deadline, and record stage timings. | End-to-end p50/p95, individual stage p50/p95, cache-hit ratio, timeout/retry rate and deadline refusals. |

Apple and Tesla are outside the present 40-company corpus. Their examples therefore exercise corpus coverage refusal. A genuine retrieval-quality example must use an available source or first add the missing reports. No new reports are downloaded by this feature.

## Request path and limits

[`ReliableQAPipeline`](../src/report_search/reliable_pipeline.py) accepts the existing encoder, retrieval engine and reranker. Its `answer(query)` method performs these stages:

1. [`plan_query`](../src/report_search/query_policy.py) identifies recognized subjects, explicit filing years and bounded lexical hints. An explicitly unavailable company/report can stop the request before embedding or generation.
2. Retrieve vector Top 20 and filter recognized source scope. Rerank the remaining pool to Top 5 using the **original** question.
3. If confidence or requested-source coverage is insufficient and recovery is enabled, search BM25 for the original question plus at most two hinted queries. Each search returns at most 20 candidates. Reciprocal-rank fusion uses `sum(1 / (60 + rank))` to select at most 20 candidates for one additional rerank to Top 5. At most 40 query–passage pairs are reranked per uncached request.
4. Require the selected set to cover the requested subjects/filings and pass the configured raw-reranker-score threshold. Reconstruct canonical source paragraphs/table rows, generate a cited draft, check every quote, and obtain the separate semantic verdict.
5. Check the request budget before releasing an answer. A cache entry can be written after generation and both verification stages succeed; it contains the selected passages, not the answer. The deadline is checked again after the cache write, so a late request can still be refused.

Recovery is **disabled by default**. Enabling it requires its own explicit `expansion_threshold`; the earlier vector-only diagnostic threshold is not installed automatically. Thresholds must be evaluated for the actual corpus, score model and retrieval policy, including source filtering. Raw reranker logits are not answer-correctness probabilities.

Financial synonyms are search hints, not accounting identities. For example, Costco's total revenue includes membership fees; replacing that requested value with a net-sales subtotal would be wrong. Adding terms to this project's OR-based FTS query can change ranking in either direction, so lexical expansion must be measured before promotion. SQLite documents the underlying [FTS5 Boolean query behavior](https://sqlite.org/fts5.html#fts5_boolean_operators).

A bare metric period such as “FY2024 revenue” is not automatically a hard report-year filter: a later filing may report that value in a comparative column. “According to the FY2024 annual report” does request a particular filing. The parser uses deterministic names, aliases and patterns; unrecognized companies and ambiguous grammar remain limitations. A covered source scope establishes availability, not that the answer is supported.

Future statements are also evidence-dependent. Catalog case F017 points to a real paragraph in NVIDIA's FY2025 report, PDF page 59, describing an expected certification date in March 2027. That is eligible for an evidence-bound answer preserving “expected”; it does not justify answering an unrelated future-revenue prediction.

## Integration

Use [`answer_reliably.py`](../scripts/answer_reliably.py) for live retrieval in one session. It verifies the active inventory, chunks, indexes and embedding identities before loading the models. Configure the existing model environment variables and supply a threshold evaluated for this retrieval policy:

```powershell
# Set $threshold to your evaluated raw reranker cutoff before running.
.\.venv\Scripts\python.exe scripts/answer_reliably.py `
  "According to Intel's FY2024 annual report, what was total revenue?" `
  "According to Intel's FY2024 annual report, what was total revenue?" `
  --threshold $threshold --deadline-seconds 30
```

Quoted questions share one pipeline and an in-process passage cache. `startup_seconds` covers artifact verification and model loading separately from request deadlines. `--device cpu` selects CPU inference; the default is `cuda:0`. To opt into recovery, add both `--enable-expansion` and an independently validated `--expansion-threshold`. The CLI returns exit code 2 if any question abstains and 0 if all receive answers. Without a threshold, it refuses before loading models; unavailable recognized companies/reports also stop early.

[`inspect_query.py`](../scripts/inspect_query.py) previews scope and hints without models or a provider:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_query.py "What is Tesla's projected revenue for 2027?"
```

The earlier frozen-ranking answer CLI retains its original behavior. The new controller also exposes a Python API for verified resident retrieval components:

```python
from report_search.reliable_pipeline import ReliableQAPipeline
from report_search.runtime_support import RetrievalCache

pipeline = ReliableQAPipeline(
    inventory=active_report_inventory,
    encoder=encoder,
    retrieval_engine=engine,
    reranker=reranker,
    binding=active_artifact_binding,
    threshold=chosen_raw_logit_threshold,
    enable_expansion=False,
    cache=RetrievalCache(max_entries=128, max_bytes=64_000_000, ttl_seconds=300),
    deadline_seconds=30,
    latency_warning_seconds=3,
)
result = pipeline.answer("According to Intel's FY2024 annual report, what was total revenue?")
```

`active_report_inventory` contains `company`, `ticker` and integer `fiscal_year` for reports actually present in the verified active index. When supplied, only `status="validated"` rows count. Derive this inventory from active indexed documents, rather than assuming that every downloaded report was indexed. `active_artifact_binding` requires `corpus`, `encoder`, `index`, `reranker` and `query_policy` fingerprints independently obtained from those active artifacts; do not copy a binding from the cache itself. The pipeline adds its implementation digest, inventory digest, thresholds and recovery configuration. API credentials must never be included in cache bindings.

Treat the session's components and configuration as fixed. Recreate the pipeline and its binding after changing active artifacts or models.

The existing `FINANCE_QA_BASE_URL`, `FINANCE_QA_MODEL`, `FINANCE_QA_API_KEY` and optional verifier-model settings configure the generator and reviewer. Alternatively inject objects implementing `complete_json(messages)`. No provider is selected implicitly. Leaving `threshold=None` keeps generation disabled.

## Caching, retries and deadlines

[`RetrievalCache`](../src/report_search/runtime_support.py) is an in-memory LRU cache with entry, serialized-byte and TTL limits. Keys include the exact query and binding; changing case, whitespace, fiscal year, basis or a fingerprint makes a different key. Reads return independent copies and do not extend TTL. The byte limit counts serialized passage data plus key digests, not total Python process memory. Invalid, empty or oversized results are not stored. Cache hits still undergo scope checks, the confidence gate, canonical source verification, generation and semantic review.

The model wrapper makes at most one retry per model stage for transport unavailability or HTTP 429/500/502/503/504, using the remaining shared budget. It does not retry authentication failures, malformed outputs, unsupported claims or bad citations as transient failures. Retrying a generation request can incur another provider charge even if an earlier response was lost. Retry behavior is bounded recovery, not exactly-once execution.

The deadline is **cooperative**. Monotonic checks run before and after stages; network calls receive a socket timeout limited by the remaining budget. An active synchronous CPU/GPU call or blocking read cannot be forcibly cancelled by this controller. A late result is discarded and later stages do not start. The three-second setting marks a slow request; it is not a measured production latency promise or a hard three-second cutoff. Model initialization outside `answer()` is outside its timing scope.

`result["audit"]["runtime"]` records the query plan, cache hit, recovery use, scored pair count, provider attempt counts, stage durations, remaining budget, slow-request flag and failure category. Credentials and raw provider error bodies are not included. Failure categories distinguish missing corpus coverage, insufficient retrieval, unsupported answers, source/reference integrity, provider/configuration errors and latency exhaustion.

## Evaluation and remaining work

The [development candidate comparison](../data/evaluation/query_expansion_results.json) ran on the same **50 development questions**, with a maximum pool of 20 for every approach. It reused checksum-verified vector rankings, searched the matching BM25 index read-only and applied the production query-policy/scope/fusion functions. No encoder, reranker or generator was run, and no held-out test question was evaluated.

| Candidate pool, before reranking | Evidence Recall@20 | Complete annotated evidence@20 |
|---|---:|---:|
| Original vector Top 20 | 71.33% | 52.00% |
| Vector Top 20 with requested-source filtering | 71.33% | 52.00% |
| Filtered vector + original-query BM25, RRF Top 20 | 74.67% | 56.00% |
| Filtered vector + original-query BM25 + bounded hints, RRF Top 20 | 72.67% | 54.00% |

Hints were generated for 26 questions. Compared with original-query fusion, hints improved one question, worsened two and left 47 unchanged: **-2.00 percentage points** in mean evidence Recall@20, with a descriptive paired-bootstrap 95% interval of [-7.00, +2.00] points. Net-earnings expansion displaced Salesforce's annotated income evidence; top-line expansion displaced evidence in a Microsoft/Oracle comparison. The policy was not retuned after inspecting these results. Recovery remains opt-in: these results do not justify enabling automatic synonym expansion by default.

Candidate assembly had medians of 224.7 ms for original-query fusion and 378.8 ms with hints in this one sequential SQLite pass. These are shared component measurements, excluding vector embedding/search, reranking, model loading, generation, source verification and judgment. They are not end-to-end latency or independent production trials. Full per-query judgments, paired deltas, timing scope and artifact/code fingerprints are recorded in the result file and its ignored detailed artifact.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/compare_query_expansion.py --publish data/evaluation/query_expansion_results.json
```

[`failure_cases.json`](../data/evaluation/failure_cases.json) contains **30 agent-authored behavior specifications**, marked `measured=false`. They cover the four original examples, metric and accounting-basis confusion, incomplete comparisons, supported versus unsupported future statements, empty/invalid scores, source/citation changes, provider failures, deadlines and cache identity. Existing development-question references provide source context; F017 additionally includes a checked source paragraph and artifact hash. The catalog is not a set of 30 observed failures or 30 live-model successes.

The full suite passes **358 tests**. The [failure regression results](../data/evaluation/failure_handling_checks.json) separately record **118 passing controlled checks** of branching, budgets, CLI preconditions and validation against prescribed model responses. Reproduce them with `python scripts/run_failure_checks.py --publish data/evaluation/failure_handling_checks.json`. They cannot measure whether a real model detects unsupported claims. The prior [generation checks](../data/evaluation/generation_checks.json) remain controlled tests, and the prior held-out retrieval results remain unchanged. No live answer-quality or production latency experiment is claimed for this feature.

Before changing defaults, extend the development experiment through reranking and recalibrate the active gate; do not select policy on the frozen test questions. Once an endpoint and independent human labels are available, evaluate answerable and unanswerable cases together, including wrong-year/basis citations, unsupported forecasts and partial comparisons. Report answer quality, refusal tradeoffs and end-to-end latency separately from retrieval recall.
