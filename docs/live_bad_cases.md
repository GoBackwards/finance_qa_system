# Live bad cases: real use, root causes, fixes and results

Earlier audits ([classic_bad_cases.md](classic_bad_cases.md)) injected controlled faults; no live model answered a question. This audit used the system for real. Fifty realistic questions went through the demo's product path (`DemoService.search` → `DemoService.answer`) with a live `gpt-4.1-mini` generator and verifier. Every outcome was graded against source-verified gold answers. The pipeline was then fixed and rerun on the same questions, and the fixed version was graded again.

## Process

1. **Question pool** ([`live_usage_questions.json`](../data/evaluation/live_usage_questions.json)): four simulated user personas each wrote 11–14 questions. A is an analyst doing numeric lookups, B compares companies and asks for margins, C asks qualitative and summary questions, and D is a casual user (typos, Chinese, "last year", companies outside the corpus, forecasts, advice, prompt injection, false premises). Each gold answer was located in the parsed reports with an exact excerpt and PDF page. An independent agent then re-checked it; 17 of 50 were corrected and none rejected. Of the 50 questions, 38 expect an answer, 5 accept an answer or a clarification, and 7 expect a refusal.
2. **Live runs** ([`run_live_usage.py`](../scripts/run_live_usage.py)): the runner records the retrieved candidates and Top-5, every generator draft and verifier verdict, the reason codes and the timings. The confidence gate was opened (threshold −1e6) so every stage could be observed. What the dev-calibrated gate (6.359375) would have done is recorded separately for each question.
3. **Grading**: one grader agent per batch of questions classified each outcome and diagnosed the failing stage and root cause from the logged evidence. An independent skeptic agent then re-checked every grade and could change it. There was no human review. Files:
   - grades: [`live_usage_grades.json`](../data/evaluation/live_usage_grades.json);
   - run summaries, transitions, gate analysis and per-question outcomes: [`live_bad_case_results.json`](../data/evaluation/live_bad_case_results.json), built by [`summarize_live_usage.py`](../scripts/summarize_live_usage.py);
   - machine-readable version of the table below: [`live_bad_case_table.json`](../data/evaluation/live_bad_case_table.json).
4. **Fix and rerun** in several rounds. Each round was validated by the full unit-test suite (410 tests at the last commit before the audit; the published suite has 588, 587 passing and one skipped on Windows), adversarial code reviews, and an independent fact-check of these documents. The fact-check found that two worked examples in prompt v2 reused live-pool answers (case 13), so the final run used prompt v2.1 with fictional examples. Prompt v2.2 then added an explicit comparison-result rule, measured in [architecture Table A11](system_architecture.md#a11-comparison-decomposition-per-company-sub-questions-30-comparative-benchmark-questions-12-live-comparisons), and the shipped v2.3 adds a verifier calibration sentence; the full 50-question pool was re-run with v2.3 and the other next-phase directions in [Table A13](system_architecture.md#a13-next-phase-directions-comparison-verdict-value-checks-citation-repair-context-extras-v3).

| Run | Code state | Answered | Good outcomes | Bad cases |
|---|---|---:|---:|---:|
| baseline | Original demo path, prompt v1 | 21 | 17 | 33 |
| iteration 1 | Scoping, compact context, citation repair, grounding, calculations | 24 | — | — |
| iteration 2 | + cross-fragment quote repair, one corrective retry, review hardening | 32 | 30 | 20 |
| iteration 3 | + qualifier check, more quote repairs, no retry after an admitted gap, verifier-doubt rule | 30 | 30 | 20 |
| **final** | Same code, prompt v2.1 (no leaked examples) | **30** | **32** | **18** |

A good outcome is a correct answer, an acceptable clarification, or a correct refusal. From baseline to final:

| Outcome | Baseline | Final |
|---|---:|---:|
| Correct answers | 10 | 21 |
| Acceptable clarifications | 1 | 4 |
| Correct refusals | 6 | 7 |
| False refusals | 23 | 13 |
| Partially correct | 4 | 4 |
| Wrong | 5 | **1** |
| Unsafe | 1 | **0** |

15 baseline bad cases were fixed and none regressed. Eight of the 18 remaining bad cases changed form. Four false refusals are now released partially correct answers (B01, B04, B08, C05; B04 high severity). C01 and C07 moved from partially correct to refused, and C11 from wrong to refused. B12 is the only one that became a wrong answer: it was refused in the baseline (see "Remaining failures").

Two single-question reruns were made after a check produced a false positive, and are recorded in the grades file. C02 (iteration 3) tripped a verdict-doubt pattern on "including assumed debt". D07 (final run) tripped the qualifier check because a year stood between the qualifier and the figure. Both checks were fixed and covered by regression tests before the rerun.

Model context shrank from a median of 65,479 to 11,708 characters (maximum 402,553 → 23,495). Median per-question time was 4.0 s generation and 0.65 s retrieval in the baseline, and 3.7 s and 0.60 s in the final run; 11 of 50 final questions used the corrective retry.

## Typical bad cases

Results are the graded outcomes, baseline → final. "Replay" means a recorded draft was re-checked offline with the final code.

| Failure Type | Bad Case | Root Cause | Mitigation Strategy |
|---|---|---|---|
| **1. Context assembly overflow** | A07 "What did Deere earn in fiscal 2023? net income and diluted EPS". Refused with `context_budget_exceeded`. Eight questions failed this way (A07, A10, B03, B04, B07, B09, D03, D14). | Each evidence unit was serialized three times with its full citation object (SHA-256, bbox, locator). The A07 context was 176,543 characters for 250 units (one fragmented page alone was 154 units), against a 100,000 budget, so the request was refused before generation. | Models now get a compact view, each unit once as `[id, page, kind, text]`, while provenance stays local for validation. Over budget, the lowest-ranked passage that is not a company's or report year's last is dropped, and the gate is rechecked on what remains. Median context fell 65,479 → 11,708 characters. **A07, A10, B03, D03: false refusal → correct.** |
| **2. Exact-quote mismatch (typography)** | C04 "Can you summarize Starbucks' 'Back to Starbucks' turnaround plan…". A correct draft was refused with `quote_not_in_source`. | The source contains non-breaking spaces (`$1.0\xa0billion`, `September\xa028`). The model typed ordinary spaces, so the exact-substring check failed. Ligatures (ﬁ/ﬀ, in C05) and a NUL in place of an NBSP (D13) failed the same way. | A citation-repair pre-pass matches quotes under NFKC, whitespace, dot-leader, hyphen and trailing-punctuation normalization, then rewrites each quote to the **exact source span**. The strict validator is unchanged. **C04: false refusal → correct.** D13 also needed the recency scoping of case 6 (its draft used FY2022) and is now correct; C05 is partially correct because its second critical audit matter was not retrieved (see "Remaining failures"). |
| **3. Quote split across parser fragments** | D08 "costco membership fee revenue 2024?". The quote "Membership fees . . . $ 4,828" spans two units. | The parser stores one visual line as adjacent units ("… $" \| "4,828"), and the model quotes across the boundary or cites the neighbouring ID. | A quote that uniquely covers consecutive units is split into exact per-unit citations. A wrong ID is rebound only to a unique exact match in the same report, never to another company's table. Mechanical failures get one corrective retry. **D08: false refusal → correct; A09: false refusal → correct after one retry.** |
| **4. Comparison entity missing** | B01 "coke vs pepsi 2022 — which one ran the higher operating margin, and how far apart were they?". All 20 candidates were PepsiCo, so the generator correctly refused. | A single global vector search with no entity resolution: "coke" was never linked to Coca-Cola, "HD" was never resolved to Home Depot (B02), and the more frequent company crowded out the other. | A query-scope resolver handles names, aliases, typos, tickers and Chinese names. Each company gets an exact FAISS search restricted to its reports, and the Top-5 reserves slots per company. On the benchmark, DocR@5 rose 90% → 100% ([architecture A6](system_architecture.md)). **B01: false refusal → partially correct.** Both companies' margins (25.4%, 13.3%) are now found and cited, but the final run's retry dropped the 12.1-point gap. |
| **5. Calculation refused** | B06 "Caterpillar earned a bit more than Deere in FY2023, but whose net profit margin was actually higher?" and B04 "how much did intel's revenue shrink from 2021 to 2023 in % terms, and what did AMD's revenue do over the same 2 years". Margin and growth questions could not be answered (B06, B10 and B11 were refused). B04's observed baseline refusal was the context overflow of case 1, with the ban as a further blocker. | Prompt v1 forbade derived values ("Do not introduce calculated values"), and model arithmetic is unreliable. | A claim may carry a `calculation` over cited operand cells. The application recomputes it in Decimal, allows only scale constants, checks sign and direction, and renders the formula. **B04: false refusal → partially correct.** Intel's −31.4% is released as an application-recomputed figure (also with the leak-free prompt, case 13). AMD's FY2021 value was in the Top-20 but lost at reranking, so AMD's span is wrong. B06 is still refused: the final retry packed the comparison into one calculation claim. |
| **6. Relative period resolved to a stale year** | A13 "What was Microsoft's operating cash flow last year?" was answered for FY2023. D07 "how much $ did starbuks make last yr" was answered for FY2021. | Nothing resolved "last year", so the model picked a year itself. The typo "starbuks" also weakened matching. | The resolver maps "last year / latest / current / 去年" to each company's latest report and uses it as a retrieval filter. Both models receive the interpretation, and fuzzy matching handles typos. **A13: wrong → acceptable (FY2025, $136,162M). D07: wrong → acceptable (FY2025, $1,856.7M "including noncontrolling interests").** |
| **7. Undated question answered from an old report** | D12 "What is Home Depot's revenue?" was answered for FY2022 without noting that it was stale. | No period was given, and the reranker preferred older tables. | Undated questions get a dedicated latest-report pool, reserved latest-year slots, and a note telling the model to answer for the latest fiscal year and say so. **D12: partially correct → acceptable (FY2025, $164,683M).** |
| **8. Historical figure presented as a forecast** | D05 "how much revenue is AMD gonna make in 2027? just give me a number". The baseline released "AMD is projected to make $34,639 million … in 2027". | The generator relabelled FY2025 actuals as a projection. The verifier wrote "This is a projection for 2025, not for 2027; however …" and still returned supported=true. Nothing checked the year. | Deterministic grounding: every year, date, number and company in a claim must occur in its cited text. Forecast intent and out-of-coverage years become scope notes. A "supported" verdict whose own reason expresses doubt is rejected. **D05: unsafe → correct refusal.** |
| **9. Answer for the wrong period** | B05 "TXN vs ADI fiscal 2023 operating margin. Which one held up better compared with the prior year?" was answered for FY2024 vs FY2023, with the conclusion reversed. | Without year scoping, FY2024 reports outranked FY2023 ones, and the verifier did not compare periods. | An explicit year restricts each company's pool to that year's report, and both models receive the requested period. **B05: wrong → correct (ADI 27.3% → 31.1% held up better than TI 50.6% → 41.8%).** |
| **10. Qualifier dropped or invented** | Iteration 2 released two wrong answers and one partial answer this way. "Organic Net Revenue, a non-GAAP financial measure" was restated as Mondelez net revenues (B12, wrong). A consolidated total was labelled "excluding financing businesses" (B07, wrong). "Including noncontrolling interests" was dropped (D07, partially correct). Its third wrong answer, A06, is case 13. | The quotes were exact and the numbers grounded, so only the meaning was wrong. The verifier missed the change of basis. | A per-figure qualifier check: a non-GAAP/organic/adjusted, noncontrolling, constant-currency or pro-forma qualifier on a quoted figure must stay with that figure in the claim, and "excluding …" may only come from the sources. **Replay: all three released drafts are now rejected.** In the final run, D07's retry kept the qualifier (acceptable) and B07 was refused. It does not catch B12's final draft, which mislabels segment figures (see below). |
| **11. Partial answer released with an admitted gap** | C01 "How did Emerson end up owning all of AspenTech? Walk me through what they put in back in 2022 and then what they paid for the rest of it later". The baseline answer said "no separate mention … is provided". In iteration 2, B09's retry deleted such an admission and released a partial comparison. | The generator answered the convenient part, and the verifier passed it. | Claims that admit missing evidence are rejected, and that failure is never retried. **C01: partially correct → refused.** The verifier found the question unanswered, because the year filter excluded the FY2025 buyout report. **B09: iteration-2 partial → refused**; Broadcom's R&D row was in the Top-20 but reranked out. Both outcomes are safer, but both are still false refusals. |
| **12. Over-refusal by the calibrated confidence gate** | In the final run the 6.359375 raw-logit gate would accept only 4 of 50 questions. Of the 25 correct or acceptable released answers it would keep only 4, refusing for example A07 (6.23), D13 (3.00), B05 (2.46), A11 (2.17), D03 (1.75) and A06 (1.18). | Raw reranker logits are uncalibrated. Table rows and scoped pools score low even when the answer is correct; on the benchmark, dev precision of 90.9% fell to 73.3% on test. | No default threshold is installed: the gate refuses until an operator configures one. Release is decided by exact references, deterministic grounding and the verifier. The gate would also block all 5 bad released answers, but only by refusing almost everything. |
| **13. Evaluation leakage through prompt examples** | Found during the fact-check of this document. Prompt v2's scale example "6,690 in millions as $6.69 billion" is A06's wrong sub-row figure (Oracle R&D excluding stock compensation). Its calculation example "Intel's revenue fell 31.4%" with operands 54,228 and 79,024 is B04's gold answer. | The worked examples were written from observed live failures. | Prompt v2.1 uses a fictional "Example Corp". The prompts were scanned for any figure from the question pool, and the final run was repeated in full. **A06 "oracle R&D spend FY24, in billions": wrong in the baseline and iteration 2 ($6,690M), partially correct in iteration 3 → correct ($8,915M) without the leaked example. B04's Intel calculation stayed correct.** |

## Remaining failures (final run)

18 of 50 are still bad: 13 false refusals, 4 partially correct answers and 1 wrong answer. Grouped by the stage the graders identified:

- **Reranking (6: A08, B02, B04, B09, B10, B11).** The reranker prefers prose containing the query words (often non-GAAP letters) over the statement or table row that holds the figure. Candidates: per-metric sub-queries for margin and ratio questions, and a per-page cap so slices of one table cannot fill the Top-5.
- **Generation (5: A03, B01, B06, B12, D14).**
  - **B12 is the one wrong answer (high severity):** AMEA segment figures were labelled as Mondelez totals and the wrong span was answered. Candidates: carry segment headings into table chunks, and check that a claim's span matches the requested span.
  - B01's retry dropped the requested gap.
  - D14 refused a false premise instead of correcting it.
- **Retrieval (5: B07, B08, C05, C07, C11).**
  - Dense-only scoped search misses lexical cues (C05 "critical audit matter", C11 "BBB").
  - C07's maturity schedule is split across dot-leader chunks.
  - Candidates: scoped BM25 fusion and neighbour-chunk expansion.
- **Verifier (1: A12).** The verifier returned supported=false while its own reason affirms the claim.
- **Query understanding (1: C01).** An explicit "2022" locked out "what they paid … later".
- **Not counted as failures:** calendar vs fiscal year for January year-ends (A01's NVIDIA "2024" is answered as FY2024, graded acceptable), and answering Chinese questions in English (D09, D10).

## Limitations

- **The final results are measured on the same 50 questions that drove the fixes.** They show that the fixes work on these cases, not that they generalize. The only independent check is the frozen 100-question benchmark: retrieval in [architecture Table A6](system_architecture.md#a6-retrieval--re-ranking-global-vs-scoped-pools-frozen-benchmark-same-reranker) and end-to-end answers in [Table A10](system_architecture.md#a10-end-to-end-answers-on-the-frozen-100-question-benchmark-original-vs-current-pipeline) (accuracy 34% → 51%, paired +17.0 points, 95% CI +8.0 to +26.0).
- The 50 questions were designed, so the counts describe this pool, not production rates.
- Gold answers and grades are agent-authored and adversarially re-checked, without independent human review.
- Each configuration ran once, and gpt-4.1-mini sampling varies, so single-question changes can include noise; for example, B01 was correct in iterations 2 and 3.
- Raw run records are gitignored under `data/experiments/live_usage/`.

## Reproduce

```powershell
# Baseline: the code at commit 6b61b5e (prompt v1), run against this checkout's data and indexes
New-Item -ItemType Directory -Force tmp/head_code | Out-Null
git archive --output tmp/head_code.tar 6b61b5e src
tar -xf tmp/head_code.tar -C tmp/head_code
.\.venv\Scripts\python.exe scripts/run_live_usage.py --src tmp/head_code/src --label baseline --threshold -1000000 `
  --output data/experiments/live_usage/baseline.jsonl

# Final configuration: the recorded run used prompt v2.1 (commit 960bcc1; extract it as above and pass --src to
# rerun it exactly). This checkout ships prompt v2.3.
.\.venv\Scripts\python.exe scripts/run_live_usage.py --label after_fix --threshold -1000000 `
  --output data/experiments/live_usage/after_fix.jsonl

# Summarize the five recorded runs, then run the regression suite
.\.venv\Scripts\python.exe scripts/summarize_live_usage.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

`summarize_live_usage.py` reads all five recorded runs (`baseline`, `iteration1`–`iteration3`, `after_fix`) from the gitignored `data/experiments/live_usage/` and fails if one is missing. Only `baseline` and `after_fix` can be regenerated as above. `iteration1`–`iteration3` were intermediate working-tree states recorded for the audit and were never committed. The runner fills unset `FINANCE_QA_*` variables from `.env` (`--env-file`) and skips question ids already in its output file, so move a recorded file aside before regenerating it. Each configuration ran once and model sampling varies, so a rerun will not reproduce the graded outcomes exactly.
