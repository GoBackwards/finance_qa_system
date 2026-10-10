# Failure handling for financial-report Q&A

Every question in the Streamlit demo and the live runner goes through [`DemoService`](../src/report_search/demo_service.py). Its `search` method retrieves and selects evidence, and its `answer` method runs the [evidence-bound generation checks](generation_constraints.md). Each request ends in one of three ways:

- an answer whose claims passed exact-reference, grounding and verifier checks;
- the public abstention `I'm not sure` with a machine-readable reason (a rejected draft is never shown);
- for caller errors and infrastructure failures, a `DemoError` with a sanitized code.

An earlier bounded-recovery controller and its command-line runner have been removed. [`failure_cases.json`](../data/evaluation/failure_cases.json) and [`failure_handling_checks.json`](../data/evaluation/failure_handling_checks.json) were written for that controller and are kept as historical records; they do not describe the current path.

| Failure type | Example and diagnosis | Handling now (module) | What to measure |
|---|---|---|---|
| Retrieval miss | A revenue question uses different vocabulary from the filing. In the live audit, dense-only search missed "critical audit matter" (C05) and "BBB" (C11). A terminology mismatch is a hypothesis, not an established cause for every missed answer. | Each resolved company gets its own exact vector pool, so a frequently named company cannot crowd out another. The whole pool is reranked against the original question (`scoped_retrieval.py`, `retrieval.py`, `reranker.py`). Comparisons can opt into per-company sub-questions (`decomposition.py`, off by default). Missing evidence ends in abstention: the generator is told to abstain, a claim that admits a gap is rejected and not retried, and the verifier must find the question fully answered (`generation.py`, `answer_verification.py`). No BM25 fallback or synonym expansion runs. | Evidence Recall@5/@20 and pool evidence recall on the frozen benchmark; false refusals on the live pool. Scoped pools raised pool evidence recall from 71.33% to 90.33% (dev) and from 60.00% to 73.67% (test), but the Top-5 Evidence Recall gain is within noise ([Table A6](system_architecture.md)). 13 of 50 live questions are still false refusals, not all of them retrieval misses. |
| Irrelevant retrieval | A gross-margin question returns sales text, another company, or the wrong filing. Live examples: "coke vs pepsi 2022" retrieved only PepsiCo (B01), "last year" was answered from FY2023 (A13), and FY2024 tables outranked the requested FY2023 (B05). | `query_scope.py` resolves companies (aliases, tickers, typos, Chinese names) and explicit, relative or missing periods against the inventory. Each company's pool is limited to its resolved report years. `scoped_retrieval.select_candidates` reserves Top-5 slots for each compared company and for the latest report of an undated question, and skips cross-year near-duplicates. Both models receive the resolved scope and its notes. `check_claim_grounding` rejects a claim whose period, corpus company or qualifier is not found in its own cited sources. The verifier is told to check entity, period, metric and basis. | Document Recall@5, complete evidence@5, and answers with the wrong period, company or basis. Scoping raised DocR@5 from 90% to 100% on both benchmark splits (Table A6). It addressed live failure types 4, 6, 7 and 9 in [live_bad_cases.md](live_bad_cases.md). |
| Unsupported answer | The draft invents a forecast, swaps GAAP for non-GAAP, or attaches a correct quote to a false claim. Live example: an FY2025 actual was released as a 2027 projection, even though the verifier's own reason expressed doubt (D05, baseline). | A fail-closed raw-logit gate with no default threshold (`confidence.py`), then hash-checked source binding (`evidence_binding.py`). Claims must carry exact quotes, and derived values are allowed only as calculations the application recomputes (`generation.py`). `answer_verification.py` repairs citations to exact source spans, validates every quote strictly, and checks that each year, date, number, company and qualifier occurs in the claim's own cited sources. A mechanical failure gets one corrective retry. A separate verifier call follows, and `validate_semantic_verdict` rejects a `supported` verdict whose reason expresses doubt. | Wrong and unsafe released answers, false refusals, and the stage that rejected each draft. Live pool, baseline → final run (prompt v2.1): wrong answers 5 → 1, unsafe 1 → 0, false refusals 23 → 13 ([live_bad_cases.md](live_bad_cases.md)). Benchmark, prompt v2.1: 73.9% of released answers fully correct, with 9 wrong among 69 answered (Table A10). |
| High latency or dependency failure | The first search loads both models. A provider stalls, times out, rate-limits or returns malformed JSON. A reranker input would need truncation. Another session's request is occupying the shared worker. | One worker owns the models, the FAISS index and the SQLite connection and runs every search and answer; each search stage is timed (`demo_service.py`). A failing search stage raises `DemoError("<stage>_failed")` and keeps the evidence already retrieved. Generation runs under a cooperative budget, 30 seconds by default in `DemoService.answer` (`runtime_support.RequestBudget`, `generation.py`). Each model call has a 45-second socket timeout, a 2,000,000-byte response limit, no redirects and no retry, and any provider error abstains with a sanitized code (`llm_client.py`). | Per-stage p50/p95, queue wait, and the rates of `deadline_exceeded` and provider errors. Final live run: median retrieval 0.60 s, median generation 3.7 s. The 250 live-pool requests had no deadline or provider failure; the current pipeline's 100-question benchmark run had 1 `deadline_exceeded` and 3 model-call abstentions. |

Apple, Tesla and a short list of other well-known companies outside the 40-company corpus are recognized by the scope resolver, and a note tells both models that no report is available. No request is stopped before retrieval. When the question names no corpus company, the search is global, and preventing a wrong answer depends mainly on the generator abstaining and on the verifier. The grounding check also resolves these out-of-corpus names (`entities_in_text`), so a claim that names one must cite text that names it too. In the final live run, all seven questions that expected a refusal ended as `generator_abstained`: Apple, Tesla, an uncovered FY2026, a 2027 forecast, investment advice, a prompt injection and unreported emissions.

## Request path

1. **Validate and queue.** `search` rejects a question that is not a string, is blank, exceeds 10,000 characters or is not valid Unicode (`invalid_query`). It then hands the work to a single worker thread. On first use, the worker loads the encoder, index and reranker, checks the encoder, embedding, inventory and index identities, and records `startup_ms`.
2. **Scope.** [`ScopeResolver`](../src/report_search/query_scope.py) resolves corpus companies, out-of-corpus companies, periods, comparison intent and forecast intent. It writes a short note for each interpretation.
3. **Scoped retrieval.** [`plan_pools`](../src/report_search/scoped_retrieval.py) gives each resolved company an exact FAISS search over its own reports, restricted to the resolved report years when there are any: max(6, ⌈20/n⌉) candidates per company for n companies, reduced so that these pools hold at most 30 in total. Undated questions add a 4-candidate pool restricted to each company's latest report. Pools are merged by chunk ID. Without a resolved company, the global vector Top 20 is used. With `decompose_comparisons=True`, a comparison of two or more companies uses per-company sub-questions instead: each is searched in its company's reports with 5 to 10 candidates (at most about 40 reranked pairs in total; 20 for a two-company comparison) and reranked against its own wording, and the Top 5 gives each company an equal share first.
4. **Rerank and balanced Top 5.** The BGE cross-encoder scores every pooled candidate against the original question and refuses to truncate any input. `select_candidates` reserves max(1, ⌊5/n⌋) slots for each compared company while slots remain, and keeps latest-report slots for undated questions. It also skips a passage that repeats, with identical figures, one already selected from another of the same company's reports, unless only such repeats remain. The query and the selected passages are HMAC-signed.
5. **Gate.** `answer` rejects a result whose query or selected passages were changed or not produced by this service instance (`invalid_search_result`). It recomputes the scope from the signed question and calls [`generate_answer`](../src/report_search/generation.py). [`confidence_gate`](../src/report_search/confidence.py) requires valid raw reranker scores whose maximum reaches an explicitly configured threshold. No threshold is installed by default. The dev-calibrated 6.359375 would have kept only 4 of the 25 correct or acceptable live answers (case 12 in [live_bad_cases.md](live_bad_cases.md)).
6. **Source binding.** [`SourceEvidenceStore`](../src/report_search/evidence_binding.py) rebuilds each chunk from hash-checked parsed sources and compares it with the retrieved payload. If the compact model context exceeds 100,000 characters, passages are dropped one at a time, lowest-ranked first. A passage that is the last one for its company or report year is never dropped; if nothing can be dropped, the request abstains (`context_budget_exceeded`). The gate is then rechecked on the passages that remain.
7. **Generation.** A single JSON-mode call uses prompt `evidence-bound-claims-v2.2` and includes the scope notes. Each claim carries exact quotes, and derived values are allowed only as `calculation` objects.
8. **Citation repair and deterministic grounding.** [`canonicalize_citations`](../src/report_search/answer_verification.py) re-points retyped or misattributed quotes to exact source spans. `validate_draft` then requires every quote to match verbatim and recomputes each calculation. `check_claim_grounding` requires each claim's years, dates, numbers, companies and qualifiers to occur in its own cited sources, and rejects claims that admit missing evidence. A mechanical failure gets one corrective retry, and every check then runs again.
9. **Verifier.** A separate stateless call judges each claim and whether the question is fully answered. It uses the configured verifier model, or the generator model if none is set. `validate_semantic_verdict` requires a verdict for every claim and rejects a `supported` verdict whose reason expresses doubt. Rendering takes page numbers and locators from the verified bundle, never from the model.

## Error and abstention codes

`DemoError` means the request could not run. Search errors carry a `partial_result` with the timings and any candidates retrieved before the failing stage. An abstention means the request ran but nothing could be released. Its `audit` records the gate decision, the stage timings, and whether the generator and verifier were called.

| Stage | Signal | Codes | Usually indicates |
|---|---|---|---|
| Input and lifecycle | `DemoError` | `invalid_query`, `invalid_device`, `invalid_model_configuration`, `invalid_search_result`, `service_closed`, `reentrant_request`, `close_failed` | Caller error, a tampered search result or a failed shutdown; no retrieval or generation ran |
| Search | `DemoError` with `partial_result` | `startup_failed`, `query_scope_failed`, `query_embedding_failed`, `vector_search_failed`, `decomposed_retrieval_failed`, `reranking_failed` | Dependency or component failure; a failed startup is retried on the next search |
| Unexpected exception | `DemoError` | `request_failed`, `answer_failed` | A defect outside the checked branches |
| Gate | abstention | `unconfigured_threshold`, `invalid_threshold`, `below_threshold`, `empty_candidates`, `missing_reranker_score`, `invalid_reranker_score`, `expected_reranked_top5` | No configured policy, or weak evidence |
| Source binding | abstention | `source_verification_failed`, `context_budget_exceeded` | Changed source artifacts, or evidence that cannot be packed into the context budget |
| Model call | abstention | `model_not_configured`, `model_unavailable`, `model_http_<status>`, `incomplete_model_response`, `model_response_too_large`, `invalid_model_response_framing`, `invalid_model_choices`, `model_refusal_or_tool_call`, `empty_model_response`, `invalid_model_json`, `invalid_model_request`, `generation_failed`, `semantic_verifier_failed` | Provider or configuration failure in the generator or verifier call |
| Draft | abstention | `generator_abstained`; `quote_not_in_source`, `invalid_quote`, `unknown_evidence_id`, `duplicate_citation`, `missing_or_excess_citations`, schema codes such as `invalid_draft_schema`, calculation codes such as `calculation_result_mismatch` | Missing evidence, or a draft that could not be bound to its sources |
| Grounding | abstention | `claim_admits_missing_evidence`, `claim_period_not_in_evidence`, `claim_number_not_in_evidence`, `claim_entity_not_in_evidence`, `claim_qualifier_mismatch` | A retrieval miss (admitted gap), or an irrelevant or unsupported claim |
| Verifier | abstention | `unsupported_claim`, `question_not_answered`, `verifier_reason_contradicts_verdict`, `incomplete_verifier_coverage`, `invalid_verifier_json`, `invalid_verifier_schema`, `verification_context_budget_exceeded` | An unsupported or incomplete answer |
| Budget | abstention | `deadline_exceeded` | The generation layer, from source binding to rendering, exceeded its budget |

How the callers surface these codes:

- **Streamlit app.** A search error is shown together with the evidence retrieved so far, and no answer is generated. An answer-stage `DemoError` becomes the refusal `answer_unavailable`. Common abstention reasons are explained in plain language.
- **[`run_live_usage.py`](../scripts/run_live_usage.py).** It records the code, the partial timings and every generator, verifier and decomposer call.

## Deadlines, timeouts and retries

- **Retrieval has no deadline.** `search` records `queue_wait_ms`, `startup_ms`, `query_scope_ms`, `query_embedding_ms`, `vector_search_ms`, `reranking_ms` and `total_ms` (the decomposed path adds `decomposition_ms`), including the timing of the stage that failed. One worker runs every search and answer, including provider calls, so a slow request delays other sessions; `queue_wait_ms` shows that delay.
- **Generation has a cooperative budget.**
  - *Configuration.* `DemoService.answer` defaults to 30 seconds. Streamlit's budget field starts from `FINANCE_QA_DEADLINE_SECONDS` (default 30), and `run_live_usage.py` takes `--deadline-seconds`, falling back to the same variable and then to 30. A direct `generate_answer` call has no budget unless `deadline_seconds` is passed.
  - *Checks.* [`RequestBudget`](../src/report_search/runtime_support.py) starts inside `generate_answer`, after retrieval. It checks a monotonic clock before and after each stage: source binding, generation, reference validation, the retry and its revalidation, semantic verification, verdict validation and rendering.
  - *On expiry.* An expired budget prevents all later stages, and a result that finishes late is replaced by `deadline_exceeded`. The budget cannot interrupt an active model call or source read. `audit.generation_budget` and `audit.stage_timings` record every stage.
- **Model calls have a fixed socket timeout.** [`ModelConfig`](../src/report_search/llm_client.py) defaults to `timeout_seconds=45`, and neither Streamlit nor the live runner changes it. urllib applies it per socket operation (connect and each read), not to the whole call, and it is not shortened to fit the remaining budget. One stalled call can therefore outlast the 30-second budget before its result is discarded. Calls are non-streaming, with at most 2,048 output tokens and a 2,000,000-byte response. Any of the following fails the call:
  - a redirect;
  - incomplete HTTP framing;
  - a `finish_reason` other than `stop`;
  - a tool call;
  - a provider refusal.
- **No transport retry.** The client sends each request once. A timeout or connection error (`model_unavailable`), HTTP 429 or 5xx (`model_http_<status>`), or a malformed response abstains immediately. Credentials and provider error bodies never appear in codes or audits.
- **One corrective generation round.** A mechanical check failure in the first draft triggers this round: a quote, schema or calculation failure, or an ungrounded year, date, number, company or qualifier. The generator is told the specific failure once, and the revised draft goes through every check again. Abstentions, admitted gaps and verifier verdicts are never retried. The retry is a second provider call within the same budget; `audit.generation_attempts` and `audit.first_attempt_reason` record it.
- **Decomposer fallback.** With opt-in model decomposition, a failed or invalid plan falls back to the deterministic per-company rewrite, and the trace records why.
- **No request cache.** `DemoService` keeps only the resident models, the index and the source store; queries, rankings and answers are not cached. `RetrievalCache` in `runtime_support.py` is no longer used by the request path and remains only as a tested utility. `SourceEvidenceStore` keeps up to four reconstructed source documents and invalidates them when a source file changes.

The final live run used a GPU, gpt-4.1-mini as generator and verifier, prompt v2.1, an open gate (threshold −1e6), a 30-second budget and one sequential client. Median retrieval was 0.60 s, median generation was 3.7 s, and 11 of 50 questions used the corrective retry. None of the five 50-question live runs ended in `deadline_exceeded` or a model-call code. The current pipeline's 100-question benchmark run (Table A10) did: one `deadline_exceeded` and three model-call abstentions (`incomplete_model_response` twice, `invalid_model_json` once). These are not load tests or service-level measurements.

## Scope, synonyms, periods and future statements

Financial synonyms are search hints, not accounting identities. For example, Costco's total revenue includes membership fees, so replacing that requested value with a net-sales subtotal would be wrong. Adding terms to this project's OR-based FTS query can change ranking in either direction, so lexical expansion must be measured before promotion. SQLite documents the underlying [FTS5 Boolean query behavior](https://sqlite.org/fts5.html#fts5_boolean_operators). The request path therefore uses neither a BM25 channel nor synonym hints. The prompt requires the stated basis (GAAP or non-GAAP, segment or consolidated) to be kept, the verifier is told to check metric and basis, and the qualifier check rejects a claim that drops or adds a basis qualifier such as non-GAAP, organic or "excluding".

For each resolved company, periods are resolved as follows:

- **Explicit year with its own report.** The year filters the search to that report: "FY2024 revenue" searches the FY2024 report only.
- **Explicit year without its own report.** The year is searched in the one or two following reports where it appears as a prior-year comparative, and a note says so. This applies, for example, to a year before the earliest report.
- **Uncovered year.** A year beyond the latest report, or more than two years before the earliest, is reported as not covered and is not mapped to another year; if no requested year is usable, the company's reports are searched without a year filter. A year beyond the latest report also sets forecast intent.
- **Relative period.** "Last year", "latest" and "current" resolve to each company's latest report.
- **No period.** The question searches all of a company's reports plus a small latest-report pool, and a note tells the models to answer for the latest fiscal year and state it.

Matching is deterministic and conservative: an unknown name stays unrecognized, and an ambiguous typo is not guessed. Restricting a year to its own report can lock out a later filing; in live case C01, an explicit "2022" excluded the later buyout report. A covered scope establishes which reports were searched, not that the answer is supported.

Future statements are also evidence-dependent. Case F017 in the historical catalog points to a real paragraph in NVIDIA's FY2025 report, PDF page 59, which says certification of fiscal 2025 PSU payouts is expected by March 2027. That paragraph is eligible for an evidence-bound answer that preserves "expected"; it does not justify answering an unrelated future-revenue prediction. The current path has three safeguards here:

- the scope resolver flags forecast intent and uncovered future years in a note;
- the prompt allows a forward-looking claim only when the quoted text is itself forward-looking;
- year grounding rejects a 2027 claim unless 2027 appears in its cited passages.

Live case D05 ("how much revenue is AMD gonna make in 2027?") moved from an unsafe answer to a correct refusal.

## Query-expansion evaluation (development only)

The request path does not use lexical expansion, based on the [development candidate comparison](../data/evaluation/query_expansion_results.json). All four candidate pools were built for the same **50 development questions**, each capped at 20 candidates. The comparison reused checksum-verified vector rankings and searched the matching BM25 index read-only. It applied the hints from [`query_policy.py`](../src/report_search/query_policy.py), a requested-source post-filter of the global vector Top 20, and reciprocal-rank fusion (`sum(1 / (60 + rank))`); the filter and the fusion now live in the script (the result file's code fingerprints predate that move). This filter predates the current exact scoped search (Table A6). No encoder, reranker or generator was run, and no held-out test question was evaluated.

| Candidate pool, before reranking | Evidence Recall@20 | Complete annotated evidence@20 |
|---|---:|---:|
| Original vector Top 20 | 71.33% | 52.00% |
| Vector Top 20 with requested-source filtering | 71.33% | 52.00% |
| Filtered vector + original-query BM25, RRF Top 20 | 74.67% | 56.00% |
| Filtered vector + original-query BM25 + bounded hints, RRF Top 20 | 72.67% | 54.00% |

Hints were generated for 26 questions. Compared with original-query fusion, hints improved one question, worsened two and left 47 unchanged. The net change was **-2.00 percentage points** in mean evidence Recall@20, with a descriptive paired-bootstrap 95% interval of [-7.00, +2.00] points. Net-earnings expansion displaced Salesforce's annotated income evidence, and top-line expansion displaced evidence in a Microsoft/Oracle comparison. The policy was not retuned after these results were inspected. The +3.33-point gain of original-query fusion over the filtered vector pool is not statistically clear either (interval [-3.33, +10.00]). The results do not justify automatic synonym expansion, and none runs in the request path.

Candidate assembly in this one sequential SQLite pass had a median of 224.7 ms for original-query fusion and 378.8 ms with hints. These are shared component measurements. They exclude vector embedding and search, reranking, model loading, generation, source verification and judgment. They are not end-to-end latency or independent production trials. The result file records the paired deltas, timing scope and artifact and code fingerprints; its ignored per-query artifact holds the full per-query judgments.

Reproduce from the repository root. This needs the ignored local artifacts: the frozen candidate run, the chunk file and the BM25 index.

```powershell
.\.venv\Scripts\python.exe scripts/compare_query_expansion.py --publish data/evaluation/query_expansion_results.json
```

[`inspect_query.py`](../scripts/inspect_query.py) previews the evaluated query-policy plan (hints and explicit-report coverage) without models or a provider:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_query.py "What is Tesla's projected revenue for 2027?"
```

The preview shows that hint policy, not the request path's scope. The resolver's interpretation appears in each search result's `scope` field and in the demo's interpreted-scope panel.

## Verification and remaining work

Unit and integration tests cover these branches with controlled model responses. They include sanitized stage errors with partial evidence, signed search results, startup retry, the single corrective retry, provider failures and the final deadline snapshot. The relevant files are `test_demo_service.py`, `test_generation.py`, `test_answer_verification.py`, `test_citation_repair.py`, `test_claim_grounding.py`, `test_llm_client.py`, `test_runtime_support.py`, `test_query_scope.py`, `test_scoped_retrieval.py` and `test_decomposition.py`. These tests establish control flow, not how often a real model fails. Live behaviour is measured in [live_bad_cases.md](live_bad_cases.md), on 50 designed questions that also drove the fixes.

Open items:

- **Latency and transport.**
  - The budget covers generation only.
  - Socket timeouts are not tied to the remaining budget.
  - Transient 429/5xx responses are not retried.
- **Retrieval.** Lexical-cue misses (C05, C11) are candidates for scoped BM25 fusion or neighbour-chunk expansion. Given the result above, these would need to be measured through reranking before being enabled.
- **Reranking and generation.** Six live failures come from reranking, and a segment subtotal can still be labelled as a company total (B12).
- **Labels.** Gold labels still lack independent human review.
