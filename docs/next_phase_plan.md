# Next phase: raising evidence recall and fully-correct answers

> **Outcome.** All six directions were implemented in reduced form and measured in [system_architecture.md, Table A13](system_architecture.md). Factual and comparative fully correct answers rose by +10.2 of 70 (CI +5.2 to +15.8), and comparisons went from 36.7% to 66.0%. Final-context recall rose from 75.8% to 85.2%, and summaries went from 41.7% to 51.7% fully correct. The live pool was unchanged (34 good of 50). The estimates below are the design-phase projections, and the parser header-order fix (4b) is being built as a separate corpus version.

**Short answer.** The best next step is mostly not more recall. 24 of the 36 benchmark answers that are not fully correct already have every gold unit in their context. So the largest gains come from CPU-only fixes at the answer layer, in this order:

1. A deterministic "which is higher" sentence for comparisons.
2. Value-binding checks for the column year and footnoted rows.
3. Citation repair.

After those come two cheap rule changes to the context extras. Colab Pro does not speed up any of these. Its best uses are offline measurement (exact reranker scores, encoding, a judge, drafting a fresh test set), a parser re-embed, and a gated reranker experiment later.

**Baseline** (commit 435b7ae, Table A12):
- **Final-context Evidence Recall (EvR):** 75.83 strict / 86.83 eq.
- **Benchmark answers:** 64 of 100 fully correct, 17 partial, 3 wrong, 16 false refusals.
- **Live:** 34 of 50 good, 3 wrong.
- **Cost:** 3,046 tokens per question (1.44x the Top-5 base of 2,115); latency p50 5.6 s, p95 14.7 s; 26% of questions use the corrective retry.

## What the diagnoses agree on

- **Generation is now the main limit on answers.**
  - 24 of the 36 answers that are not fully correct have eq-complete evidence.
  - 12 of those 24 are comparative partials that only leave out which company is higher. None of the 10 answered test-template comparatives states this verdict.
  - P(correct | complete evidence) = 54/78 = 69%. So recall work alone is worth about +4 correct answers (at most +6).
- **The remaining recall gap is mostly summary prose:** 23 of the 30 eq-missing units, or 7.67 of 13.17 eq points. Separately, 21 of the 51 strict misses (11.0 points) are label artifacts that the eq judge already credits.
- **CPU-only rule changes to `context_extras` recover most of what is left,** with no re-index. One exception: stage B needs a higher `MAX_EXTRAS` cap (see rank 5).
- **Measurement is the binding constraint.**
  - Replicate-like runs with identical Top-5 contexts differed by about 10 fully-correct flips per 100 questions.
  - That is an upper bound on sampling noise. The 70-question pair also differed in prompt version (v2.1 vs v2.2) and grading pass.
  - So one run per arm cannot reliably detect a gain under about 9 questions (80% power).
- **Colab Pro:** use it for exact reranker scoring, encoding, an open-LLM judge, drafting a fresh question set, the parser re-embed, and a gated reranker experiment later. It does not speed up the main fixes, which are CPU code plus API calls.

## Projected effect

These are in-sample, single-run estimates; expect them to shrink out of sample. Verdict synthesis is mechanical and should transfer. The retrieval rules were designed on these residuals, so plan for up to about half of their gain.

| Metric | Now | After ranks 1-4 (+4b) | After ranks 1-6 |
|---|---|---|---|
| Benchmark fully correct | 64 | about 79-80 (+1 with repair level D) | about 83 (range 77-87) |
| Comparatives fully correct | 9/30 | about 22/30 | about 22-23/30 |
| Benchmark wrong | 3 | 1 (dev-021) | 1 |
| Benchmark false refusals | 16 | about 12 | about 10 |
| EvR strict / eq | 75.83 / 86.83 | 84.33 / 89.83 | 87.67 / 93.17 (needs `MAX_EXTRAS` 10) |
| Live good / wrong | 34 / 3 | about 38 / about 1 | about 38-39 / about 1 |
| Benchmark tokens per question (vs Top-5 base 2,115) | 3,046 (1.44x) | 3,459 (1.64x); live about +450-600 | 4,269 (2.02x) |
| Corrective-retry rate | 26% | about 18% | about 18-20% |

**How the fully-correct figures add up**

*After ranks 1-4:* 64 + 10 + 2.5 + 1.3 + 1.7 ≈ 79.5.
- verdict +10;
- extras stage A +2.5 (3 newly complete questions × 0.85);
- citation repair A-C +1.3 (dev-047 0.85; test-021 date 0.45);
- value binding +1.7.

*After ranks 5-6:* + 2.55 + 0.6 + 0.6 ≈ 83.
- stage B +2.55 (dev-039, dev-041, test-043 × 0.85);
- truncation +0.6 (dev-044);
- verifier +0.6 (dev-024, a comparative).

*Why 0.85 per completed question:* ranks 1-4 lift P(correct | complete) from 69% to about 86%. Do the answer side first.

**Retry rate:** 10 first drafts would pass without a retry (dev-018, dev-020, dev-036, dev-047, test-013, test-016, test-021, test-034, test-044, test-050), and rank 4 adds 2 retries (dev-017 and test-028 were released at attempt 0).

**Live:** in-sample fact predicates go 47 → 52 of 58. One of the five, B09's 8,818, is already in the delivered context (the "line" predicate misses it), so the real gain is +4 (B12, A03, C05, C11).

## Measurement prerequisites (before any answer-level claim)

**M1. Generation replay harness** (local, CPU + API)
- `scripts/replay_generation.py`, about 100 lines. It rebuilds the recorded `search.selected` passages from the SQLite payloads and calls `generate_answer` R times per question.
  - It loads no encoder or reranker, so Streamlit can stay up.
  - The scratch `residual_answers/replay.py` reproduces the recorded check outcomes on 150 of 150 records.
- **New contexts.** For rank 2 and rank 5 contexts, build the extras on CPU and order them by the M4 reranker scores, as production does.
- **`--drafts K`** stores the raw generator JSON. Post-processing fixes (ranks 1 and 3) can then be compared on identical drafts. Fixes that trigger a retry or change the verifier input still need new calls.
- **Cost:** about $0.0056 per question, so about $6 per A/B (100 questions × R = 5 × 2 arms). With the 50 live questions it is about $8.
- **Noise:**
  - single-run paired SD ≈ 3.2 questions, so the minimum detectable effect (MDE) is about 8.9; with R = 5 it is about 4.0;
  - two identical configurations gave +13.3 points (bootstrap CI [+3.3, +26.7]);
  - A12's comparative gain is about +16.7, not +23.3, once the base is averaged over its two draws.
- **Optional:** `FINANCE_QA_TEMPERATURE` in `llm_client.py`, about 6 lines, unset by default (no temperature is sent today). Test T = 0 as its own arm.

**M2. Automatic grading**
- **Deterministic CPU grader** for factual and comparative answers, about 150 lines.
  - It takes the figures from the gold answer_points, normalises scale and display precision, and requires a comparative verdict to name the winner.
  - Agreement with the blind grades: fully correct 310/310, 4-class 301/310.
  - The 140 "held-out" grades are held out by run, not by question (same 70 questions), so re-validate it on M3.
- **Summaries:** a pinned open-LLM judge on Colab (vLLM, 32B AWQ on an A100-40GB, or a 14B model on an L4 if no A100 is assigned; 2-5 CU). Use it only if kappa ≥ 0.8 against the 120 existing summary grades.
- **Human spot-check** (about 2-2.5 h): every grade that differs between arms, every "wrong", and 30 summaries.

**M3. Fresh held-out set** (drafted on Colab, about 5 CU; human review about 6 h)
- About 120 questions from the 71 report-years that no benchmark or live question touches: 42 dev-company and 29 test-company. There are 253 same-year pairs within the splits (178 dev, 75 test).
- **Mix:**
  - 40 factual;
  - 35 comparative, at least 10 of them ratio, growth or multi-year;
  - 45 summary in at least 3 phrasings, half without the word "summar";
  - at least 20% written by hand;
  - about 20 live-style questions with fact predicates written before any run.
- **Format:** v2 gold with equivalence `alternatives`; answers in the M2 grader contract; fictional prompt examples only.
- **Size:** a sign test needs at least 6 improved and 0 worsened questions. If stage A's 11 per 100 halves out of sample, about 110 questions are needed.
- **Single use, for real.**
  - Freeze configurations (version strings plus a hash) and pre-register all arms (ranks 1-6, 4b and their ablations).
  - Run once. Per-rank tuning happens on dev, then the test split once, never on M3.
- **Training-pool rule.**
  - M3's dev-company report-years are reserved only until the confirmation run; they may then join reranker training.
  - Rank 8 is evaluated on the test split plus the test-company part of M3, which never enters training. Otherwise the 66-report training pool would shrink to 24 reports.

**M4. Colab session 0: score and encode, no training** (about 1-2 CU on a T4; up to about 4 on an L4 or at older rates)
- **Export** about 9k (query, `render_document`) pairs on the laptop CPU:
  - 2,000 benchmark pool pairs (827 of their scores are unknown today);
  - 1,000 live pool pairs;
  - about 1.5k candidate extras from stages A, B and C4;
  - about 4.5k depth-50 pairs.
- **Encode** the 50 live queries and about 90 summary theme strings with the pinned BGE-M3.
- **Score** with the repo's own reranker class, which hard-checks the pinned revision and weights hash.
- **Reproducibility gate** before trusting results:
  - re-scored stored scores within 0.05 logit of the 1,173 known ones;
  - re-encoded benchmark queries within about 3e-4 of the stored vectors;
  - K = 5 reproduces the recorded Top-5 on at least 96 of 100 questions.
- **What it unlocks:**
  - exact equal-token Top-K controls;
  - the true order of new extras;
  - AUC of useful vs useless extras;
  - an exact live slot replay;
  - dense per-theme sub-queries;
  - the measured conversion of a depth-50 pool.

**M5. Log full-pool reranker scores and the query vector in run records**
- About 5-10 lines in `demo_service._search` (the full pool is already scored there, then discarded) and in `run_live_usage.py`. Keep them out of the signed Streamlit payload.
- Test that the selected lists are byte-identical before and after.
- After this, re-selection within the logged pool is exact on CPU. New candidates (anchors, extra theme hits, neighbours, deeper pools) still need one scoring pass.

**M6. Gold v2 with equivalence alternatives**
- Covers the 21 eq-covered strict misses and the 2 prose equivalents (dev-003, test-016). About a 60-line script plus about 1 h of review.
- Report strict v1, strict v2 and eq side by side.

## Ranked plan

### 1. Deterministic comparison verdict (local)

**What changes**
- One function in `answer_verification.py` (about 60 lines), about 5 lines in `generation.py` after the verifier passes, and about 80 lines of tests. Behind a flag.
- **Fires when** all three hold:
  - 2 or more companies are in scope;
  - the question asks "which/whose/who" with a comparative word, "did X or Y ... more", or "compare/vs";
  - no released claim already states a verdict.
- **How it works:**
  1. Take one currency amount per company from the released, verified claims. When a claim covers both companies, attribute each amount to the ticker of the cited unit that contains its digits.
  2. Normalise thousand/million/billion with `_claim_numbers`.
  3. Require a currency or scale marker on both amounts and the scoped report year. The prototype checked neither, and it fired on live B02 (a correct answer, unit-less 14.2 vs 13.38).
  4. Append the neutral sentence "X's $A is higher than Y's $B", citing both claims.
- v1.1 would accept percentages that are verified calculation results.
- An optional missing-side retry covers live B06; it is the only part that needs an LLM call.

**Targets**
- The 12 comparative partials: dev-022, dev-025, dev-032, test-021, test-022, test-023, test-024, test-026, test-029, test-031, test-033, test-034.
- Live B07.
- test-028 gets the right direction, but its figure stays wrong until rank 4.

**Expected gain**
- `verdict_v2.py`: direction correct in 13 of 13 benchmark cases; it does not fire on the 11 answers that already have a verdict.
- 9 × 0.95 + 3 × 0.5 ≈ +10 (range +9 to +12). test-021, test-029 and test-031 also omit year-end dates, hence the 0.5; rank 3 lifts test-021.
- Live +1.

**Cost:** about 140 lines, 0 API calls, +0 ms.

**Risks**
- It inherits wrong operands (test-028, or a segment figure as in B12), so ship it with ranks 2 and 4.
- The trigger regex was checked on the same 32 multi-company answers.

**Evaluation**
- `verdict_v2.py` as a regression gate, with B02 added as a must-not-fire case.
- M1 draft bank (K = 5) on the 30 benchmark comparatives and the 12 live comparisons.
- In the M3 confirmation run: fire rate, direction accuracy, and no firing on non-comparisons.

### 2. Context extras v2, stage A (local)

**What changes** (about 70-90 lines in `scoped_retrieval.py`, behind `context-extras-v2` and per-rule flags)
- (a) `_label_terms` drops every resolved company's name. Today it drops only the current pool's names, so test-028 keeps "starbucks".
- (b) Terms and row labels are plural-stemmed.
- (c) New slot sort, in order:
  1. statement-like tables (3 or more statement row families) first, with "segment" tables demoted;
  2. best single-row coverage;
  3. union overlap;
  4. vector rank.

  Depth 20, 1 slot per company.
- (d) **Income-statement anchor.**
  - Fires on non-summary questions with generic income-statement terms.
  - For each scoped company-year, it appends the table after a "CONSOLIDATED STATEMENTS OF INCOME/OPERATIONS/EARNINGS" title, found in the last 75% of the report. At most 2 years per company, otherwise the latest only.
  - The title is found in 159 of 200 reports.
  - Precompute a document→anchor map offline, rather than walking `engine.neighbour` over about 280 chunks per document at query time.
- (e) For single-company non-summary questions, add the scoped raw-question BM25 top-1.
- (f) For summary themes, skip a top-1 hit that is already selected and take the next one.
- Stage A stays within the current `MAX_EXTRAS = 8`: at most 7 extras per question in replay.

**Targets**
- Eq units: dev-027 and test-035 (anchor), test-015 (BM25), dev-038, dev-039 and test-042 (themes).
- About 17 strict units.
- Live: B12 (consolidated 26,042), the A03 split note, C05 and C11. B09 is not a recall target; see rank 3.

**Expected gain** (CPU replay, re-run for this review: `p6_stage.py`, 0 eq units lost)
- **EvR:** strict 75.83 → 84.33, eq 86.83 → 89.83.
  - By group (eq): factual 92.5, comparative 98.3, summary 77.8.
  - Test split eq: 81.7 → 85.3. The anchor alone adds +1.0 eq on dev and on test.
- **Tokens:** 3,459 per question; by group 3,213 / 3,410 / 3,836.
- **Live tokens:** the anchor fires on 21 of 50 live questions (49 chunks), +448 tokens per live question.
- **Slot precision:** "segment" picks fall from 7 to about 2 on the benchmark and from 7 to 3 on live.
- **Answers:** newly complete dev-027, test-015 and test-035 ≈ +2.5 (dev-027 and test-035 also need rank 1). B12 should move from wrong to correct or partial.

**Risks**
- **Overfitting.** The anchor is as in-sample as the rejected synonym map. It was built from dev-027, test-035 and B10-B12, and its trigger set (revenue, sale, income, earning, profit, margin, eps, expense, cost, operating) contains the same sales/earnings vocabulary. Judge it on M3 like every other rule.
- **Coverage.** In the 41 reports without the title, the anchor silently falls back to the slot.
- **Wrong columns.** Full statements bring prior-year columns, so ship with rank 4.
- **Token gate.** At 1.64x the Top-5 base, stage A sits above the roadmap's 1.6x gate (roadmap section 8). Decide before the M3 run whether to re-set the gate or apply it per category; do not decide after seeing results.

**Evaluation**
- Unit tests.
- Replay by split and category (strict v1/v2/eq, 0-loss, tokens).
- M4 exact live slot replay.
- R = 5 end-to-end A/B on 100 benchmark + 50 live questions (M1 with M4 scores). Gates:
  - wrong-in-any-run does not rise;
  - p95 latency rises by less than 300 ms;
  - count the "segment" passages cited in released answers.

### 3. Citation repair v2 (local)

**What changes** (in `_repair_citation` and `check_claim_grounding`; about 135 lines plus 120 lines of tests)
- (a) **Quote repair levels A-C:**
  - case-insensitive matching, with control characters as wildcards and normalised escapes, bullets and ligatures;
  - a unique rebind elsewhere in the bundle;
  - splitting a quote at a sentence break or ellipsis.

  Level D (trim to the longest exact block when it is at least 70% of the quote) has its own flag.
- (b) **Period-end date auto-citation.**
  - Only on `claim_period_not_in_evidence`, when the date equals the cover-page fiscal-year-end of an in-bundle report (resolved for 199 of 200 reports).
  - At most one citation per company and date.
  - Never on entity failures: that variant releases test-027's wrong Lowe's date.
- (c) **Cross-claim calculation grounding:** accept a number that equals a verified calculation result of another claim in the same draft, and carry that claim's citations.

**Expected gain** (measured on recorded drafts: `eval_repair2.json`, `date_autocite.py`)
- **A-C:**
  - dev-047 passes (≈ +0.85);
  - test-050 goes from refusal to partial (its evidence is missing);
  - test-021's and test-034's first drafts pass with both year-ends cited. With rank 1, test-021 rises from 0.5 to about 0.95 (+0.45).
  - Total ≈ +1.3.
- **Level D:** only D passes dev-042 and dev-048 (+about 1.0) and live C07 (refusal → partial).
- **Live:** B09 goes partial → correct (cross-claim grounding; the first draft had both verdicts and correct shares), +0.8.
- **Retries avoided:** 10 benchmark first drafts no longer need the retry (listed above), plus live C01.
- **Wrong answers:** +0 measured.
- The verifier was not replayed; it accepted 120 of 122 drafts that passed the checks.

**Risks**
- D-trim can cut the supporting clause.
- About 10 newly released first drafts were never graded (±1).

**Evaluation**
- `replay_all.py`, `eval_repair2.py` and `date_autocite.py` as gates: every attempt that passes today must still pass with identical claim text.
- M1 draft bank.
- D-trim false-pass rate in the M3 run.

**Optional add-on:** a "fiscal year ended <date>" unit per scoped report (about 20 tokens), for test-027 and test-032. With rank 1: +1 to +2.

### 4. Value-binding checks with one targeted retry (local; ship with rank 2)

**What changes**
- **Column-year check**, ported from `colyear.py`, about 130 lines; about 20 of them expose cell bounding boxes in `SourceEvidenceStore`.
  - Bind each claimed table number to the nearest year header by x-position, and skip it when that header is ambiguous.
  - Compare the column's year rank with the claim's.
- **S2 footnote-row check**, about 30 lines: when the cited row label has a marker such as "(1)" and a same-label unmarked row from the same report is in the bundle, retry pointing to it.
- About 10 lines of retry hints in `_retry_feedback`, e.g. "value X is under <year>; the <claim year> column holds Y".
- About 120 lines of tests.
- There is only one corrective retry. A flag on the retry's output withholds the answer.

**Targets:** dev-017 and test-028 (released wrong answers), test-025 (drafts), live A06.

**Expected gain**
- The check flags all 3 targets. Among the other ~99 checkable released claims it flags 0, in-sample after 3 tuning rounds. S2 flags only A06 among 217 released claims.
- Benchmark wrong answers 3 → 1.
- +1.7 fully correct: dev-017 0.7, test-028 0.7 (needs rank 1), test-025 0.3.
- Live wrong −1 (A06).
- Cost: +2 retries on the benchmark.

**Risks**
- It covers only 47% of released claims.
- A failed retry turns a wrong answer into a refusal (+0-2 refusals).

**Evaluation**
- `eval_colyear.py` and `segcheck.py` as gates.
- Hand-label 50 table claims from the M1 reruns.
- R = 5 on dev-017, test-028, test-025 and A06.

### 4b. Parser header-order fix (laptop CPU plus one re-embed; before the M3 run)

- **What changes:** in `src/report_parser/tables.py` `_context`, group same-line tokens and order them by x. Today the sort is by (y, x), so '2023' at y=84.54 lands before '2024' at y=84.70 although the columns run 2024 | 2023.
- **Scope:** 175 of 5,517 year-header tables in 37 documents (every CAT report, INTC FY2024/25, EMR).
- **What it takes:**
  - re-parse those documents, re-chunk, rebuild the BM25 store;
  - one full re-embed: about 25 min on the laptop with Streamlit stopped, or 1-3 CU on Colab;
  - FAISS and manifest rebuild.
- **Value:** the root fix for dev-017-class wrong columns on every CAT statement. Rank 4 only catches them at answer time and stays needed for garbled headers (test-028, test-025).
- **Gates:** retrieval replay with 0 eq units lost, and a fresh benchmark run.
- Decoupled from the rank 9 bundle; there is no reason to wait for paragraph packing.

### 5. Extras v2 stage B plus the truncation fix (local; ship together)

**Retrieval** (about 25 lines, under the existing summary trigger)

*Step B1 (neighbours first, fits the current cap):*
- Add the best 3 of the ±1 prose neighbours of the Top-5 prose chunks, ranked by query vector × stored chunk vector. Read the stored vector with `index.reconstruct`.
- Measured: eq 91.50, strict 86.00, 3,869 tokens per question (summary 5,204; 1.83x the base).
- Completes dev-039 and dev-041. At most 7 extras.

*Step B2 (2 unselected BM25 hits per theme, C3):*
- Measured: eq 93.17, strict 87.67, 4,269 tokens (summary 6,537; 2.02x).
- Also completes test-043 and adds units in dev-043, test-041, test-046, test-047 and test-050.
- **Requires raising `MAX_EXTRAS` to 10 for summary-trigger questions.** C3 gives 9-10 extras on 26 of 30 summary questions. Kept at 8, eq falls to 92.50 and test-043 is lost (`adv_cap.py`).
- Variant C4 (3 hits per theme): eq 94.17 at 4,654 tokens, and it needs a cap of 13.
- B2's 2-hits rule alone completes no further non-correct question. Its value is partial-evidence units at +403 tokens per question.

**Generation** (`llm_client.py` / `generation.py`, about 10 lines plus one prompt sentence)
- Generator `max_output_tokens` 2048 → 3072, about 17.6 s at about 175 tokens/s.
- Retry on `incomplete_model_response` only when at least 15 s of the 30 s generation deadline remain. The deadline covers generator, retry and verifier; the 45 s figure is only the socket timeout. Otherwise the retry ends in `deadline_exceeded`.
- Prompt rule: quote the shortest contiguous supporting sentence(s), about 300 characters or fewer; quote the whole row for tables. Use fictional examples only.

**Expected gain**
- B1 + B2 completions: dev-039, dev-041, test-043 ≈ +2.5. B1 alone ≈ +1.7.
- dev-044: +0.6.
- False refusals −1 to −3.
- API cost about +$0.001 per question.

**Risks**
- 1.8-2.0x the Top-5 base fails the 1.6x gate unless it is re-set beforehand.
- Template overfit: the summary trigger fires on only 1 of 50 live questions.
- The prompt change shifts every answer.

**Evaluation**
- Ablate B1 / B1+B2 / C4 on dev, then run test once.
- R = 5 on the 30 summaries. Count `quote_not_in_source`, truncations, `deadline_exceeded`, output length and p95.

### 6. Verifier calibration and partial release (local)

**What changes**
- 1-2 sentences in `VERIFIER_SYSTEM`: no plausibility judgments against outside knowledge.
- About 15 lines of partial release:
  1. drop the rejected claims;
  2. re-check scope coverage;
  3. release the rest, otherwise abstain.
- Zero-code extra arm: gpt-4.1 as the verifier via `FINANCE_QA_VERIFIER_MODEL` (about +$0.008 per question, +1-2 s).

**Targets:** dev-024 and A11. Across 231 calls the verifier rejected at least 4 correct drafts and caught 0 of the 16 released wrong answers.

**Expected gain:** +0.6 benchmark, +0.6 live (estimated).

**Evaluation**
- 3 arms, each R = 5.
- 20 injected wrong claims (wrong column, wrong company, reversed comparison). Each arm must reject at least as many as today's verifier does on the same 20.

### 7. Retrieval legs measured with M4 data (Colab outputs, then local CPU)

- **Dense per-theme sub-queries.**
  - Targets test-036 ×2, test-038, test-041 p26 and test-044 ×2.
  - Ceiling 2.0 eq points; expected +0.6 to +1.0 (speculative).
  - In production it adds a batched encode of about 3 strings per question; measure the latency.
- **Depth-50 pool for single-company and summary questions.**
  - After stages A and B, about 2-3 real units remain.
  - It adds +0.6-0.8 s of reranking, which **fails the p95 < +300 ms gate**. Adopt only with an explicit gate change and at least +1 eq on dev with 0 loss.
- **Reranker-picked shortlist of extras.**
  - Only if a scorer reaches AUC ≥ 0.75 on useful vs useless extras.
  - Today's v2-m3: row-label extras 0.28 (bench) / 0.45 (live); all extras 0.53 / 0.48, which is chance.

### 8. Table-aware reranker: bake-off, then a fine-tune only if gated (Colab)

**Bake-off** (5-8 CU at the March-2026 A100 rate, about 2x at older rates)
- Candidates on the M4 pairs: v2-m3 (control), Qwen3-Reranker-0.6B, jina-reranker-v3 (CC BY-NC), v2-gemma, and 4B/8B models as teachers.
- Metrics on dev:
  - table-vs-narrative pairwise accuracy (24.2% today);
  - prose-vs-prose accuracy (71.9% today);
  - useful-extra AUC.

**Fine-tune** (20-40 CU, about 2x at older rates; full fine-tune of v2-m3, about 1 h per epoch on an A100 or about 3 h on an L4)
- Only if no 0.6B drop-in passes the gates.
- About 10K synthetic queries from the 24 dev-benchmark-only reports. After the M3 run, add M3's 42 dev-company report-years. Exclude live-touched reports.
- Hard negatives:
  - segment vs consolidated tables;
  - tables that share "Net income";
  - the same row from another year;
  - MD&A prose without the value.
- Remove false negatives with the eq rule.

**Value**
- Mainly tokens: about −1,000 useless extra tokens per question at C4 recall, landing near 3,700.
- Recall headroom is about 1-1.5 eq points (dev-021, test-038, test-039); at most +1 benchmark correct answer.

**Deployment**
- Same VRAM, about 580 ms of reranking.
- Re-pin `MODEL_ID`, `MODEL_REVISION` and `MODEL_WEIGHTS_SHA256`.
- Recalibrate any raw-logit gate threshold. None is installed by default; A8's dev value is 6.36.

**Ship gates**
- Pairwise ≥ 60%; prose-vs-prose ≥ 71.9%; AUC ≥ 0.75.
- 0 eq units lost at equal or lower tokens.
- Then the test split and the test-company part of M3 only, plus the Chinese live questions.

### 9. Re-embed bundle (later)

- **Contents:**
  - contextual section breadcrumbs in `render_document` (bump the rendering version);
  - paragraph-aligned chunk packing;
  - BGE-M3 sparse weights, kept only if they beat FTS5 BM25.
- **Value:** recall 0 to +1 eq (headers +0.33 in the CPU proxy).
- **Cost:** 1-2 weeks of work. The re-embed takes 6-10 min on an A100 (1-3 CU with session overhead), against about 25 min on the laptop.
- It breaks chunk-id comparability with earlier runs. Pilot on 38 dev reports, and do it only after ranks 1-6 and 4b are confirmed.

## Not worth it now

| Option | Evidence |
|---|---|
| Fine-tune BGE-M3 (15-25 CU) | Only about 6 eq units (2.0 points) lie beyond a depth-50 pool; multilingual risk; full re-index for every checkpoint |
| ColBERT multi-vectors | About 48 GB in fp16; the cross-encoder already models the query-passage interaction |
| Larger reranker at inference on the 8 GB GPU | v2-gemma: about 4x the FLOPs (rerank p50 about 2.3 s); 4B-9B models do not fit beside BGE-M3 without int4 |
| Reranked Top-K instead of extras | Best-case Top-8 without extras reaches 84.17 eq, below 86.83 with extras |
| Deeper row-label slot (depth 30/50) | −0.5 strict, 0 eq |
| v2-m3 score floor on extras | Row-label AUC 0.28; a floor at 0 drops useful extras in dev-022, dev-027, dev-029, dev-030 |
| Synonym map, second slot per company | Anchor-based packages reach the same eq. Both are equally in-sample, and the anchor is the more structural of the two |
| Hard segment exclusion; S1 keyword segment check | Slot: −1 strict, 0 eq. S1: 1 true of 9 flags, and it blocks 5 correct answers |
| Dropping same_table / title_table | same_table: −3 live facts (A08 ×2, B09). title_table: −1.5 eq and −1 live fact (B07) |
| "risk" cue in theme queries (S6); table_fallback row labels (F1) | S6: −1.33 points. F1: 0 eq, −1 strict |
| Item 1A-only theme search (S4) | In-sample detector; 7 gold units lie outside Item 1A; loses test-038 |
| Date auto-citation on entity failures; hard "wrong year-end" check | The first releases test-027's wrong date. The second has 3 true vs 4 false flags; it also flagged test-021's correct "Sunday closest to October 31" |
| LLM verdict retry for every comparison | Earlier retries dropped good claims (test-031, B09); deterministic synthesis is 13/13 offline |
| Stronger generator as a first move | Speculative +4 to +10, about 5x tokens, duplicates ranks 1-4, cannot fix dev-017 |
| All ±1 summary neighbours (N1/N6) | +3.33-4.0 points at 2.0-2.3x summary tokens; best-3 keeps most of the gain |

## Schedule and Colab budget

| When | Local (CPU + API) | Colab Pro |
|---|---|---|
| Days 1-3 | M1, M2, M5 | M4 session 0 (about 1-2 CU on a T4) |
| Weeks 1-2 | Ranks 1 and 3 (A-C); ranks 2 and 4 together; M6 | M3 drafting (about 5 CU on an A100-40GB; a 14B model on an L4 if no A100); summary judge calibration (2-5 CU) |
| Weeks 2-3 | Rank 4b parser fix; ranks 5 and 6; R = 5 A/Bs (about $6-8 each) | Rank 4b re-embed if the laptop GPU is busy (1-3 CU) |
| Week 3 | M3 review (about 6 h); pre-register all arms and the token-gate decision; one confirmation run | Retrieval for M3 if needed (about 1 CU) |
| Later | Rank 7 legs; rank 9 bundle | Rank 8 bake-off (5-8 CU); fine-tune only if gated (20-40 CU) |

- **Colab total at the March-2026 rates:** about 15-25 CU without the fine-tune and about 35-65 CU with it. At older published rates (A100 at 11-15 CU/h) it is roughly double, so a fine-tune month can use most of the 100 CU.
- A100 assignment is not guaranteed. Checkpoint jobs at 4 h or less.
- Keep API keys in Colab Secrets, never upload `.env`, and pin model revisions in every notebook.

## Ship gates (every direction)

- **Retrieval:**
  - eq gain with 0 units lost, on dev and then test;
  - in the M3 run, beat the exact equal-token Top-K control by at least +2 eq;
  - report tokens per category;
  - p95 latency rises by less than 300 ms;
  - the token-gate decision is pre-registered.
- **Answers:**
  - R = 5 mean with a two-level bootstrap;
  - wrong-in-any-run must not rise;
  - `deadline_exceeded` must not rise;
  - every auto-graded "wrong" is reviewed by hand.
- **Out of sample:** publish each lever's shrinkage (M3 gain divided by in-sample gain) next to Table A12.

## Caveats

- **Everything is in-sample.** The extras rules, the anchor, and the verdict, column-year and repair prototypes were designed after reading these residuals. The test split has been inspected.
- **Noise and grading.** Figures come from single gpt-4.1-mini runs (±1-3 questions), and the grades are agent-authored.
- **Judge coverage.** The eq judge covers table units only. The live fact predicates are in-sample regexes; one (B09) is a known false miss.
- **Colab rates** come from secondary sources; check the in-notebook resource panel.
- **Prompts.** Every prompt or judge example must be fictional.

**Supporting analysis.** The CPU replays and prototypes behind these estimates (verdict synthesis, column-year check, citation repair levels, extras v2 stages, noise and grader checks) ran in a temporary working folder and are not in the repository; each rank should be re-implemented as tested code before its numbers are quoted as results.
