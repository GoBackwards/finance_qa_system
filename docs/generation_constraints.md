# Evidence-bound generation and refusal

The generation stage accepts up to 13 selected passages (in the demo, the scoped and balanced Top 5 plus up to 8 context extras; more abstains with `too_many_passages`), binds them to verified parsed sources, drops low-ranked passages that do not fit the context budget (never a company's or report year's last passage), and releases an answer only after both citation checks and a separate semantic review pass. It uses a configurable OpenAI-compatible Chat Completions endpoint. If a required check fails, its public answer is exactly **"I'm not sure"**; the rejected draft is not returned.

This reduces unsupported answers but does not guarantee factual correctness. Source parsing and model review can both be wrong. When this stage was first built, no live generation endpoint was configured, so its results measured a retrieval-based gate diagnostic and automated contracts. The later [live bad-case audit](live_bad_cases.md) ran the generator and verifier with gpt-4.1-mini on 50 graded questions; its numbers describe that designed pool, not production accuracy.

## Request flow

1. **Confidence gate.** Require valid raw `reranker_score` values and an explicitly configured threshold. If every selected score is below that threshold, stop before reading sources or calling either model. Equality passes. Empty candidates, missing/nonfinite scores, and an unconfigured threshold also refuse.
2. **Source binding.** Verify parsed artifact hashes, reconstruct canonical table-preserving chunks, and compare the retrieved source payload against them. Build request-local IDs such as `E1` for original paragraph spans, complete table rows, and table context. Preserve source warnings, table headings, units, and surrounding chunk text.
3. **Constrained draft.** Ask the generator for individually checkable claims, each carrying evidence IDs and exact supporting quotes. Missing essential evidence or incompatible comparison bases must produce an abstention. Since prompt `evidence-bound-claims-v2` (now v2.2), a value that differs across reports is answered with both values and their report years, and a false premise in the question is corrected; the verifier may still reject conflicting evidence. v2.2 also asks a "which is higher" comparison to state its result in a claim citing both sides ([system_architecture.md](system_architecture.md), Table A11). Derived values (margins, growth, differences) are allowed only through a `calculation` object whose cited operands and result the application recomputes; see [live bad cases](live_bad_cases.md).
4. **Deterministic reference checks.** Reject malformed or oversized JSON, extra/duplicate keys, unknown evidence IDs, missing or duplicate citations, altered quotes, and model-written citation markers. Every quote must occur exactly in its registered source unit. Since the live bad-case audit, a pre-pass may re-point a quote to an exact source span, for example when it differs in typography (NBSP, ligatures, dot leaders, hyphens, trailing punctuation), quotes selected table cells, uses an ellipsis inside one unit (the full covering span is then shown), cites the wrong evidence ID when the quote matches exactly one other unit (first in the same chunk, otherwise anywhere in the request's sources; bare figures only within the same report), or runs across adjacent parser fragments (split into per-unit citations); the strict check then runs unchanged, and each claim's years, dates, numbers and qualifiers must occur in its citations. When a company resolver is supplied, as in the demo and the live runner, named companies must also occur there; `scripts/answer_question.py` does not supply one ([system architecture](system_architecture.md)).
5. **Independent semantic request.** In a new stateless call, review every claim against the full source context and the original question. Require complete claim coverage, all `supported` verdicts to be true, and `answers_question` to be true. Missing, malformed, truncated, unavailable, or negative review results refuse.
6. **Application-controlled rendering.** Add citation markers and source labels only after both checks pass. Render model/source text with Markdown and HTML punctuation escaped. Document identity, PDF page number, and source locators come from the verified registry, never from a generated page number.

The second call is independent of the generator's conversation and reasoning, but it can use the same model by default. Its errors may therefore be correlated with the generator's errors. A positive semantic verdict is a fallible model judgment, not an entailment proof. Exact citation matching establishes where a quote came from; it does not establish whether the whole claim follows from it.

## Citation trace

Every returned citation contains the supporting `quote`, the full bound `source_text`, its document/PDF-page metadata, the original `source_locator`, and contributing chunk IDs. `quote_char_start` and `quote_char_end` locate the quote within `source_text`; the end offset is exclusive. For a paragraph fragment, the source locator retains its original element-relative character range. These are not character offsets into the complete PDF.

| Source kind | Original locator retained |
|---|---|
| Prose | Paragraph `element_id`, `char_start`, `char_end` |
| Structured table row | `table_id`, zero-based original `row_index`, row role |
| Table source fallback | `table_id`, source-text character range |
| Table context | `table_id`, context role, and context index when applicable |

The response includes inline footnote references plus a source label and supporting quote. PDF page numbers are **one based** and identify the PDF page, which can differ from a report's printed page number. Structured rows remain complete. A paragraph split during chunking retains its bound span; `source_text` does not imply the remainder of that paragraph was retrieved. Fallback text remains traceable but does not prove a reliable table column association.

`reference_verified` and `semantic_supported` are separate fields. For example, a draft claiming "$100 million" can cite a real paragraph saying "$10 million": its reference exists, but semantic review must reject the false claim. The failed response exposes neither that draft nor its citations; internal reason codes and audit fields record which gate failed.

## Confidence policy and measured diagnostic

The confidence statistic is the maximum selected **raw BGE reranker logit**. These scores are neither cosine similarity nor calibrated answer-correctness probabilities; a sigmoid would not make them calibrated probabilities. The application default is `threshold=None`, which means refusal until an explicit policy is chosen.

The diagnostic selects a threshold using only the existing 50 development questions. It maximizes development coverage subject to at least 10 accepted questions and at least 90% observed complete annotated-evidence coverage among accepted questions. The selected value is frozen before the separate 50-question test evaluation. The policy binds the model/revision, inference settings, corpus, chunk strategy, and retrieval/selection configuration; loading it with a different active binding fails.

| Frozen threshold: 6.359375 | Development | Test |
|---|---:|---:|
| Accepted questions | 11 / 50 | 15 / 50 |
| Coverage | 22.00% | 30.00% |
| Accepted with complete annotated evidence | 10 / 11 | 11 / 15 |
| Complete-evidence proportion among accepted | 90.91% | 73.33% |
| Accepted with incomplete annotated evidence | 1 / 11 | 4 / 15 |

The 90% development target did not hold on test. Before gating, 18 test questions had complete annotated evidence; the gate retained 11 and refused seven. A high score alone therefore neither establishes complete support nor avoids unnecessary refusal. The reference and semantic checks remain required after the gate.

These are **retrieval proxy measurements**, not answer accuracy or hallucination rates. All benchmark questions were designed to be answerable somewhere in the corpus; there is no curated unanswerable or out-of-domain set. Gold evidence is nonexhaustive and still awaits human semantic review. The small sample and company/template dependence limit uncertainty estimates. This policy is explicitly diagnostic and is not installed as a production default. See the [machine-readable diagnostic](../data/evaluation/generation_gate_results.json) for criteria, split-specific results, intervals, and provenance.

Reproduce the diagnostic without loading models or sending a generation request:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_generation_gate.py --target-precision 0.90 --min-accepted 10 --publish data/evaluation/generation_gate_results.json
```

The output directory `data/experiments/generation_gate/` contains `comparison.json`, `policy.json`, and the detailed per-question diagnostic. Generated artifacts stay outside Git; the compact published result is versioned.

## Configure and run generation

Set the endpoint explicitly. `FINANCE_QA_BASE_URL` is the API base, normally ending in `/v1`; the client appends `/chat/completions`.

| Environment variable | Purpose |
|---|---|
| `FINANCE_QA_BASE_URL` | Explicit compatible endpoint; remote endpoints require HTTPS, while local loopback HTTP is supported |
| `FINANCE_QA_MODEL` | Generator model identifier |
| `FINANCE_QA_API_KEY` | Provider credential; may be empty for a local endpoint that does not require authentication |
| `FINANCE_QA_VERIFIER_MODEL` | Optional different review model on the same endpoint; defaults to the generator model |
| `FINANCE_QA_TOKEN_PARAMETER` | Optional `max_completion_tokens` (default) or compatible `max_tokens` |

For example, after setting the provider credential in the environment:

```powershell
$env:FINANCE_QA_BASE_URL = 'https://your-provider.example/v1'
$env:FINANCE_QA_MODEL = 'your-generator-model'
$env:FINANCE_QA_VERIFIER_MODEL = 'your-review-model'

.\.venv\Scripts\python.exe scripts/answer_question.py --question-id dev-001 --gate-policy data/experiments/generation_gate/policy.json --output data/experiments/generation/dev-001.json
```

The runner uses the chosen benchmark question's frozen reranked Top 5 and its checked source artifacts. Benchmark answers and gold evidence labels are not sent to the generator or reviewer. The diagnostic gate can refuse a selected question before any model request. Omitting a gate policy leaves generation disabled through the default refusal policy.

For application integration, call `report_search.generation.generate_answer(query, reranked_top5, ...)` with the already selected chunks, a verified source store, and an explicitly selected threshold. `answer_with_retrieval()` is an older helper that composes the existing encoder, the global (unscoped) vector Top 20, and reranker Top 5 APIs before these generation checks; the demo instead uses `DemoService`'s scoped, balanced selection. A reusable `SourceEvidenceStore` caches reconstructed source documents. When using a policy file, call `load_gate_policy()` with a binding independently derived from the active artifacts; copying the expected binding from that same policy would defeat its compatibility check.

The CLI also accepts `--input` with a JSON object containing only `query` and the existing pipeline's full reranked `candidates`. This mode accepts an explicit `--threshold`; historical cached policies require `--question-id`. Successful answers return exit code 0; refusals return exit code 2 and still write the structured output when requested.

The client requests JSON mode and validates schemas locally. Compatible endpoints must implement that request format; unsupported JSON mode causes refusal rather than an unconstrained fallback. The transport rejects redirects, incomplete responses, tool calls, provider refusals, and malformed JSON. Credentials and raw provider error bodies are not included in result audits. Calls are nonstreaming with a bounded output and timeout. The context guard measures characters, not model-specific tokens, so an endpoint can still reject a context that exceeds its own token limit.

For the upstream API formats, see [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and the [distinction between JSON mode and schema-constrained structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs). Local schema checks remain required regardless of provider support.

## Verification scope

Automated tests cover low/unconfigured confidence, no-call refusals, canonical source tampering, exact quote matching, false claims attached to true quotes, incomplete comparisons/reviews, malformed provider responses, missing citations, Markdown/HTML injection, and citation offsets. They establish deterministic behavior with controlled model responses. Real generation quality was later measured with live gpt-4.1-mini answers and agent grading ([live bad cases](live_bad_cases.md); [system architecture](system_architecture.md) Tables A9–A11); independent human review and a curated out-of-domain set are still missing.

The local HTTP integration regression passed 11 controlled cases over real retrieved sources when it was recorded at commit 873038a, with an earlier `generation.py` (the record lists the implementation hashes). It tested a supported quotation with its source trace, a false claim attached to a real quote, invented IDs, altered quotes, incomplete answers, low/unconfigured/empty retrieval, provider failure, malformed review output, and output truncation. The server returned prescribed drafts and verdicts, so this was not an LLM accuracy experiment. Rerun `scripts/run_generation_checks.py` to check the current code; see the [recorded checks](../data/evaluation/generation_checks.json). Source-binding smoke checks additionally covered 40 real chunks and 607 evidence entries across prose, tables, table fallbacks, and page fallbacks.
