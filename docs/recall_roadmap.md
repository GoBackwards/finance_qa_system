# Recall roadmap: financial annual-report QA (reviewed)

> **Outcome.** The quick wins Q1–Q3 were implemented in a reduced form as `context_extras`:
> - **Q1** became the term slot only: question words matched against table row labels. T2v and LTY, which needed the metric lexicon, were dropped.
> - **Q2** became theme slots without the Item 1A filter.
> - **Q3** became the table rules only.
>
> Measured live with blind A/B grading, benchmark Evidence Recall rose from 57.5% to 75.8% and answer accuracy from 50% to 64%; see [system_architecture.md, Table A12](system_architecture.md). The numbers below are the design-phase estimates.

All gains below are macro Evidence Recall (EvR) points on the 100-question benchmark. "Strict" uses the repo's `judge_retrieval`. "Eq" uses the automated equivalence judge, which covers table units only: a unit counts as found if a chunk from the same report holds the claim figure with a row-label word within 400 characters.

Every benchmark number here is an in-sample design estimate. The diagnoses read both dev and test failures, so neither split is held out.

## 1. Where recall is lost today

On the scoped baseline, macro EvR@5 is 57.5% and pool recall@20 is 82.0%, so 42.5 points are missing. Each diagnosis rebuilt the pools exactly on CPU; the table splits the 42.5 points by unit.

| Bucket | Units | Macro pts | What it is |
|---|---|---|---|
| Pool miss, factual + comparative | 16 | 10.00 | 13 of the 16 have a chunk with the same figure, label and period already in the pool, and 9 of those are in the Top-5. Only 3 are true misses: test-008, test-015 and test-034 (ROK). |
| Pool miss, summary | 24 | 8.00 | Blended three-theme query 10; boundary straddle 9; gold next to a pool chunk 4; gold outside Item 1A 1. |
| Pool to Top-5, A: label-only | 17 | 10.33 | The same figure, from another location, is already in the Top-5. |
| Pool to Top-5, B: comparative statement row outranked | 13 | 6.50 | Against narrative prose, the reranker ranks a gold table higher in only 24.2% of pairs; vector order does so in 66.7%. |
| Pool to Top-5, C: per-company quota | 4 | 2.00 | Dropping the quota is net-neutral: +3.8 / −3.0 units. |
| Pool to Top-5, D: summary prose outranked | 15 | 5.00 | Reranker 71.9% vs vector 71.2% pairwise on prose against prose. |
| Pool to Top-5, E: unit split across chunks | 2 | 0.67 | |

**Label artifacts.**
- The automated equivalence judge accounts for 12.5 points: the baseline goes from 57.5 to 70.0, and comparative from 31.7 to 60.0.
- A hand check that also covers prose puts it at about 15.8 points.

**Scope resolution** causes none of the 16 pool misses and 1 of the 58 live facts (C01).

**Live after_fix run.** Of 58 answer facts:
- 32 are in the Top-5;
- 17 are in the pool only, 15 of them carried by tables;
- 8 are dense misses;
- 1 is out of scope.

4 of the 18 bad cases (A12, B01, B06, D14) had every fact in the Top-5. Tables are selected 17.3% of the time, against 32.5% for prose.

**Reading.** Most of the loss happens at selection:
- the reranker demotes statement tables;
- one blended query drops whole summary themes;
- 512-token boundaries split units.

Reserved, structure-aware context slots placed outside the reranker are the cheapest lever.

### Quick-win package, measured

Measured on CPU from the stored pools. The full package is `merged_bundle.py`; this review re-measured it and its leave-one-out attribution with `attrib.py`. Units lost against the baseline: 0 in every row.

| Variant | EvR strict | EvR eq | Dev / test strict | Fact. / Comp. / Summ. strict | Chunks | Tokens |
|---|---|---|---|---|---|---|
| Baseline Top-5 | 57.50 | 70.00 | 63.67 / 51.33 | 80.0 / 31.7 / 53.3 | 5.00 | 2,115 |
| Q1 alone (T2v + LTY + term slot) | 69.00 | 80.00 | 76.67 / 61.33 | 85.0 / 63.3 / 53.3 | 6.45 | 2,638 |
| Q2 alone (theme slots) | 64.83 | 77.33 | 67.67 / 62.00 | 80.0 / 31.7 / 77.8 | 5.45 | 2,331 |
| Q3 alone (same-table, title link, table rules) | 62.50 | 73.00 | 67.67 / 57.33 | 82.5 / 45.0 / 53.3 | 6.45 | 2,593 |
| **Package Q1 + Q2 + Q3** | **79.33** | **88.33** | 82.67 / 76.00 | 85.0 / 73.3 / 77.8 | 8.18 | 3,264 (1.54×) |
| Package + Q3 prose ±1 (summary only) | 82.33 | 91.33 | 85.33 / 79.33 | 85.0 / 73.3 / 87.8 | 10.21 | 4,232 (2.00×) |
| Control: reranked Top-8 (Monte Carlo; hard bounds 59.0–79.5) | 69.04 | – | – / 58.35 | – | 8 | 3,192 |
| Control: reranked Top-10 (Monte Carlo; bounds 59.5–80.5) | 73.15 | – | – / 63.33 | – | 10 | 3,868 |

**Attribution. The gains are not additive.**

| Direction | Alone, strict / eq | Marginal inside the package (leave-one-out), strict / eq |
|---|---|---|
| Q1 | +11.5 / +10.0 | +9.5 / +8.0 |
| Q2 | +7.33 / +7.33 | +7.33 / +7.33 |
| Q3 | +5.0 / +3.0 | +3.0 / +1.0 |

- The standalone gains sum to 23.83, but the package gains 21.83, because Q1 and Q3 recover 2.0 points of the same units.
- Under the eq judge, the package adds nothing on factual questions (90.0 in every row). Its eq gain is comparative +11.0 macro and summary +7.33 macro.
- At about the same mean token count, the package beats the Monte Carlo estimate for reranked Top-8 by about 10 points. That margin rests on imputed scores, because 44% of pool scores are unknown. Under the gold-best bound, Top-8 reaches 79.5, so the margin is only settled by the scoring pass in Step 0.
- Comparatives in the package also use more tokens than Top-8 would: 10.53 chunks and 3,713 tokens.

### What is still missing after the package

| Group | Strict missing | Under the eq judge |
|---|---|---|
| Factual | 6 units | 4: dev-003 and test-016 (prose, answer already in context), test-008 and test-015 (true misses). dev-007 and test-019 are eq-covered. |
| Comparative | 16 units | 2: dev-021 (INTC p65 table_source, class B) and dev-027 (ADBE FY2022 p54 row 16, class C). Both test-034 units are eq-covered. |
| Summary | 20 units | 20 (prose; not covered by the eq judge): 14 never reach the pool, 6 are in the pool. |

**Headroom left on this benchmark.**
- After the package: 11.67 eq points, of which about 9.7 are real.
- After the package plus prose ±1: 8.67 eq points (17 units), of which 6.67 are real.

Every direction below competes for these same points.

## 2. Ranked roadmap

Directions are ordered by expected gain per unit of cost and risk. "Ceiling" means the gain if every unit a leg brings into the pool were converted into the context. "Expected" assumes 30–50% conversion, which is what the decomposition run and the stored reranker scores suggest.

| # | Direction | Tier | Gain (benchmark) | Confidence | Cost |
|---|---|---|---|---|---|
| 0 | Fix measurement: equivalence `alternatives`, one full-pool reranker scoring pass, a fresh held-out set | Prerequisite | No recall change. Baseline 57.5 → 70.0 under eq. | measured | 2–4 days; about 3 min of GPU |
| 1 | Q1. Row-label table slots per company and per company-year | Quick win | Alone +11.5 strict / +10.0 eq. Inside the package +9.5 / +8.0. Live facts 32 → 45–50 of 58, in-sample. | measured, but the rule was written after the diagnoses | 1–2 days, no GPU |
| 2 | Q2. Summary theme slots in a soft Item 1A range | Quick win | +7.33 strict and eq, alone and inside the package. Summary 53.3 → 77.8; test summary 31.1 → 66.7. | measured | 1–2 days, no GPU |
| 3 | Q3. Structure-aware neighbour stitching | Quick win | Table rules alone +5.0 / +3.0 eq; inside the package +3.0 / +1.0 eq. Prose ±1 adds +3.0 more at 2× tokens. | measured | about 1 day, no GPU |
| – | **Q1 + Q2 + Q3 package** | – | **+21.83 strict / +18.33 eq** (not the sum of rows 1–3) | measured | – |
| 4 | M1. Scoped lexical union legs, plus depth 20 per company for comparisons | Medium | Pool 82.0 → 91.7 at 28.7 candidates. Eq ceiling after the package +2.33 (+2.0 after prose ±1). Expected +0.7 to +1.2. | pool measured, Top-K estimated | 0.5–1 day; about 1.4× reranker pairs |
| 5 | M2. Section and statement typing at parse time, with soft typed pools | Medium | Summary pool 73.3 → 82.2 (top-10) or 90.0 (top-20). Eq ceiling after the package +1.0 / +2.67 (+0.33 / +1.33 after prose ±1). Expected +0.3 to +1.3. | pool measured, Top-K estimated | 1–2 days; payload rebuild, no re-embed |
| 6 | M3. Evidence-completeness loop with a facet planner and scope fixes | Medium | Comparisons fully correct: 6 → 9–10 of 30, by counterfactual substitution of a different intervention. | estimated, n=30, in-sample prompt | 2–3 days; runs on triggers only |
| 7 | M4. Reranker-side changes: table-aware input, fusion, K=8, dev-trained head | Medium | Eq ceiling over the package +3.0 (+1.67 after prose ±1). Measured fusion on a similar package: +0.84 strict / −1.2 eq. Main value is fewer tokens. | speculative / Monte Carlo | 1–2 days; about 1 min of GPU |
| 8 | L1. Parser, chunker and contextual headers, with one re-embed (plus a BGE-M3 sparse leg) | Larger bet | Shares the 6.7 real points left on the benchmark. Mainly fixes wrong-column answers and lets the slots shrink. | speculative | 1–2 weeks; about 25 min of GPU |
| 9 | L2. Fine-tune the cross-encoder (LoRA) on dev-company pairs | Larger bet | Same shared headroom | speculative | 2–3 days plus hours of GPU; needs LoRA on 8 GB |
| 10 | L3. Dense row-level child index | Larger bet | 0 under eq on the benchmark after the package; may help with live vocabulary gaps (B07) | speculative | 2 min of GPU, 256–511 MB |

Rows 4–10 cannot add more than the remaining headroom: 8.67 eq points after the package plus prose ±1, 2 of which are prose label artifacts. The M1 + M2 (top-20) ceiling is +3.33 at that point. Their value has to be shown on the fresh held-out set and on live replays, not on this benchmark.

## 3. Step 0: fix the measurement first

**Equivalence-aware gold.**
- Write a versioned copy of `retrieval_questions.json` that adds `alternatives`. `judge_retrieval` and `benchmark.py` already read the field. Keep the v1 numbers.
- Cover the 13 pool-miss and 17 rank-loss units that have an equivalent location (same figure, label and period), including the prose units dev-003 and test-016.
- Report every variant under both judges. Without this, every direction is rewarded for finding one particular labelled table.

**One full-pool reranker scoring pass.** Score these pairs once with the pinned reranker:
- the union of all candidate legs: about 3,760 benchmark pairs, which include the current 2,000, about 75 s;
- the 1,000 live pool pairs, whose scores are not stored, about 20 s;
- a table-aware re-render of the current pools: about 2,000 pairs, about 40 s.

With model load (about 45 s), the pass takes about 3 minutes on the RTX 4060 Laptop (8 GB). Today 44% of pool scores are imputed, so every fusion and Top-K number is a Monte Carlo range, and the comparison with the Top-8 control is unsettled. This pass needs your approval, with Streamlit stopped first.

**A fresh held-out set** (spec in section 8).

**Trace provenance.** Record a `slot` field (`row_label_table`, `term`, `theme`, `stitch`, `bm25_leg`) in `retrieval_trace`, so each leg's contribution and conversion can be audited.

**Slots and the confidence gate.** Two code facts constrain how slots can be added:
- `confidence_gate` refuses the whole request with `missing_reranker_score` when any passed candidate lacks `reranker_score`, and `generation.py` re-checks the gate on the packed passages.
- `select_candidates` sorts by `reranker_score`.

So slot chunks must either be scored (8 extra pairs or fewer, about 150 ms) or the gate must ignore `slot`-tagged candidates. Decide this before building Q1–Q3. Computing the gate maximum over the reranker-selected Top-5 only keeps the A8 threshold semantics unchanged.

## 4. Quick wins (days, no re-index)

### Q1. Row-label table slots, appended outside the reranker

**Change**
- New `src/report_search/metric_intent.py`: question-term extraction, a generic row-label lexicon, and the operation (value, ratio, growth or compare).
- New `reserve_table_slots()` in `scoped_retrieval.py`, called from `DemoService._search` after `select_candidates`. For each company:
  - **Term slot.** One table chunk from the company's depth-20 restricted vector order, choosing the best overlap between question terms and row labels (the first cell of `rows`). Comparisons search only 10 per company today, so this needs the per-company search raised to 20. That shares M1's search but reranks nothing extra. Depth was chosen on dev: 8, 9 and 8 dev units at depths 10, 20 and 30, a one-unit difference.
  - **T2v.** Up to 2 lexicon-matched table chunks from the current dense pool.
  - **LTY.** 1 table chunk per (company, scoped report year), from FTS5 BM25 over the lexicon sub-query, filtered by ticker and year.
- A hard cap of 3 slot chunks per company-year. It never binds on the benchmark: of 97 company-years, 61 got 1, 24 got 2 and 12 got 3. When the budget is tight, the packer drops slots after the neighbours.
- Apply the gate rule from Step 0.

**Targets**
- Rank-loss B, and rank-loss A on comparatives.
- Live C1, ratio operands: B02, B09, B10, B11, B12.
- Live C2, multi-year endpoints: B04, B12.

**Gain, deterministic**
- Q1 alone: +11.5 strict / +10.0 eq. Comparative 31.7 → 63.3; test comparative 33.3 → 66.7. Factual +5.0 strict, but +0 under eq: these are label-only units.
- By component, alone: term slot +8.0 (15 units; +1.31 chunks per non-summary question, +333 tokens averaged over 100 questions); T2v +7.0 (12 units); LTY +1.5 (3 units).
- Inside the package: Q1 adds +9.5 strict / +8.0 eq, and the term slot alone +3.5 / +5.0 (comparative 61.7 → 73.3).
- Context: comparatives go from 5 to 8.2 chunks with Q1 alone (2,986 tokens), and to 10.5 chunks in the full package (3,713 tokens, 2.04× their baseline).
- Live, on the recorded pools: T2v gives 45 of 58 facts; same-table stitching + T2v + LTY gives 50 of 58, and evidence-complete cases go from 6 to 13 of 20. This replay is in-sample: the lexicon is the live diagnosis lexicon minus 2 entries, and several remaining phrases map to single live cases ("were as follows" C07, "Moody's Fitch S&P outlook" C11, "critical audit matter" C05, "stock split" A03).

**Cost:** about 200 lines plus the gate change. BM25 queries take milliseconds.

**Risks**
- Segment tables feed the known generation errors: a segment subtotal taken as the total (dev-003, B12) and the wrong period column (test-028, test-025). Prefer statement-titled or total/consolidated rows only if dev shows it helps.
- Lexicon gaps for company-specific labels: ROK 'Total sales (d)', CAT ME&T. Do not add entries from test or live failures.

**Evaluate:** ablate the term slot, T2v and LTY on dev; run test once. Then run the fresh set's ratio, growth and multi-year items and a new live set. Use end-to-end blind grades to confirm that wrong answers do not rise.

### Q2. Summary theme slots in a soft Item 1A range

**Change**
- New `src/report_search/theme_plan.py`:
  - a generic theme splitter, designed on the 15 dev phrasings;
  - an Item 1A chunk-range detector that ignores cross-references (found in 29 of 30 summary documents; fails on INTC FY2024);
  - one FTS5 BM25 Top-1 per theme inside Item 1A, falling back to the whole report when no section is found.
- The slot is skipped when the theme's top chunk is already selected.
- Phase 2: one dense sub-query per theme, reranked against its own wording, with one slot per theme, following the pattern of `select_round_robin`. Measuring it needs the encoder on GPU.
- Any model plan must use fictional prompt examples only.

**Targets:** summary P3 (10 units), R4 (6) and rank-loss D (15).

**Gain, measured:** +22 units, 0 lost, which is +7.33 strict and eq, alone and inside the package.
- Summary 53.3 → 77.8 (dev 75.6 → 88.9; test 31.1 → 66.7).
- 6.5 summary chunks, 3,151 tokens instead of 2,430.
- The generic splitter reproduces the hand split (70 of 90).

**Cost:** about 150 lines; 3 FTS5 queries per question.

**Risks**
- 10.67 of Q2's 14.67 macro points come from the test split, where all 15 questions share one template, so coverage of real user wording is overstated.
- The section restriction must stay soft: 7 of 87 gold units lie outside Item 1A (dev-043 u2, dev-046 ×3, test-038 ×3).

**Evaluate:** tune on dev; run test once. On the fresh set, use summaries in at least 3 phrasings, including questions without an enumerated list and themes in MD&A, Human Capital and CAM sections.

### Q3. Structure-aware neighbour stitching at context assembly

**Change:** add `assemble_context()` after selection, looking neighbours up by chunk id ± 1 within the same document.
- **(a) Same table.** Attach c±1 when it has the same `table_id`.
- **(b) Orphaned title.** If a selected chunk carries a statement title, or is not a table, attach the table chunk on the same or next page.
- **(c) Summary prose.** A ±1,000-character window first; the full ±1 chunk optional, chosen on dev.
- **(d) Schedule tail.** An 'as follows:' or dot-leader tail rule (live C07, in-sample).
- Before merging adjacent chunks into one passage, check `SourceEvidenceStore`'s hash checks. The existing citation repair already handles a quote split across adjacent parser fragments.

**Targets**
- 9 of the 12 missed comparative tables carry no title; in 5 of them, a title-bearing chunk on the same page ranks 1–8.
- Live C3 split tables (A08 ×2, B09).
- Summary P1 (9), R1 (3) and rank-loss E (2).

**Gain, measured**
- Alone: title link +4.0 (7 units); table rules (a)+(b) +3.5 (6 units); same-table alone +1.5 (2 units, 0 under eq); all together +5.0 strict / +3.0 eq (9 / 6 units).
- Inside the package: +3.0 strict / +1.0 eq.
- Full prose ±1 on top of the package: +3.0 (summary 77.8 → 87.8), but summary tokens go from 3,188 to 6,416 (2.64× the summary baseline). This fails the 1.6× token gate in section 8.
- The ±1,000-character window recovered 9 units against 16 for full ±1, on the Top-5 alone.
- Live: +3 facts at +0.4 chunks per case.

**Cost:** about 80 lines. Primary-key lookups take under 1 ms.

**Risks**
- The adjacent table may be a different statement.
- Part of the R2 gain is non-exhaustive labels (dev-039).

**Evaluate:** report EvR together with tokens and the rates of `quote_not_in_source` and `context_budget_exceeded`.

## 5. Medium (new channel, metadata rebuild, or a GPU measurement)

### M1. Scoped lexical union legs, plus depth 20 per company for comparisons

**Change** in `plan_pools` and `merge_pools`. Every leg is a union, never a replacement.
- Comparisons get 20 vector candidates per company: per-company depth 20 and `MAX_POOL` 30 → 40. No raw-question BM25.
- Single-company questions add a scoped raw-question BM25 top-5.
- Summary questions add a whole-document theme BM25 top-3.

**Gain, measured on the pool**
- 82.0 → 91.7 at 28.7 mean candidates: comparative 80.0 → 93.3 at 40, factual 90.0 → 92.5 at 22.8, summary 73.3 → 88.9 at 25.2. 0 units lost.
- The 95.8 at about 38 candidates quoted earlier is a different union: it adds ±1 neighbours, LTY and the Item 1A legs of M2 and Q2.

**After the package**
- Units M1 brings into the pool that the package still misses under eq: test-015, test-038 u1/u2, dev-043 u2 and test-042 u1. Ceiling +2.33, or +2.0 after prose ±1.
- test-008 (BM25 rank 14) is not reached by a top-5 leg.
- Comparative depth 20 adds only label-only rows that the package already covers under eq (dev-026, dev-030, dev-033, test-028, test-030).
- Expected +0.7 to +1.2 eq at 30–50% conversion.
- Live: pool coverage 49 → 52 of 58 facts (C11, C05, A03).

**Cost:** about 30 lines. Reranker pairs go from 20 to about 29 per question. Scaling the measured p50 (389 ms at 20 pairs) linearly gives about 560 ms, and about 780 ms for comparisons.

**Risks**
- More unlabelled prose for the reranker to prefer; dev-030 lost its row this way.
- If conversion is low, use a reserved BM25 Top-1 slot instead (unmeasured).
- Comparative depth 20 is low priority under eq.

**Evaluate:** after the Step 0 scoring pass, report conversion per leg on dev, with and without M4.

### M2. Parse-time section and statement typing, with soft typed pools

**Change**
- A structure pass after `report_parser`, prototyped in `s2_structure.py` (25 s of CPU for 200 reports). It produces `item_code`, `section_path`, `statement_type`, `table_seq` and `period_columns`.
- Store these in the payload and as SQLite columns, with a validator.
- Soft typed pools:
  - Item 1A dense top-k for risk questions;
  - the primary income statement per company-year for metric questions (replaces Q1's lexicon gating);
  - Human Capital and CAM routing.
- Segment and period breadcrumbs for the generator.

**Gain**
- Measured, summary pool: current ∪ Item 1A top-10 = 82.2 (+2.9 candidates); ∪ top-20 = 90.0 (+9.8; test 55.6 → 81.1). A deeper-vector control with +10 candidates gave 82.2.
- After the package, the units M2 brings into the pool that the package misses:
  - top-10: dev-041, test-041 and test-047 → ceiling +1.0;
  - top-20: those plus test-036 u2, test-042 u1, test-044 ×2 and test-046 → ceiling +2.67, or +1.33 after prose ±1.
- Expected +0.3 to +1.3 eq.
- Live: the naive regex router reached 41 of 58 facts.
- The main value is robustness to non-template wording and precision for Q1's slots.

**Cost:** 1–2 days. Payload and SQLite rebuild only; no re-embedding.

**Risks**
- Detection gaps: the Item 1A heading was found in 174 of 200 reports, and the income statement, balance sheet and cash-flow statement were all tagged in about 120 of 200 (116–122 depending on the regex).
- Typed pools must stay soft.

**Evaluate:** freeze the detection rules before looking at labels; compare against a deeper-vector control of equal size.

### M3. Evidence-completeness loop with a facet planner

**Change**
- Add a separate retrieve-then-regenerate branch for `generator_abstained` and `claim_admits_missing_evidence`.
- Do not remove these reasons from `_NOT_RETRIED`. `generation.py` documents that retrying on the same evidence "only taught the model to delete the admission and answer partially".
- Find the missing facets: scope companies and years with no bound unit, or a `missing` field in the generator output.
- Retrieve once more for those facets only (Q1 slots, stitching, and per company × operand × endpoint-year sub-queries), add 3 chunks or fewer, and regenerate once within `RequestBudget`. The second draft must pass the same quote, grounding and verifier checks.
- Scope fixes ride along: fiscal-year-end-aware mapping (B10) and sequence cues that widen to the latest report (C01; the FY2025 chunk is at BM25 rank 2).

**Targets**
- 13 of 100 benchmark questions abstained for missing evidence, 11 of them comparisons (v2.1 run).
- Live: B02, B04, B09, B10, B11, B12, C01, A08.

**Gain, a counterfactual on stored blind grades** (30 comparisons, prompt v2.2; the trigger fired on 12)
- Substituting the always-on rule-decomposition answer: correct 6 → 9; correct or partial 8 → 14; wrong 4 → 5.
- Substituting the model-plan answer: correct 6 → 10; wrong stays at 4.
- Always-on model decomposition by itself scored 9 correct with 2 wrong.
- This measures a full Top-5 replacement, not the proposed ≤3-chunk second pass, and prompt v2.2 was written on these same questions.
- On retrieval, decomposition plus the T2v/LTY/stitching legs reaches 65.0% comparative (7.8 chunks), against 60.0% for the same legs on the scoped Top-5. The gain is dev-only (test 73.3 → 70.0), and the full package (73.3%) already exceeds it. So use decomposition only as the second pass.

**Cost:** about 250–370 lines. On triggers only: +0.5 s of retrieval and +2–5 s of generation.

**Risks**
- It can turn an honest abstention into a partial or wrong answer.
- Deadline pressure.
- Prompts must use fictional examples only.
- n = 30, so the confidence interval is wide.

### M4. Reranker-side changes, measured on the full-pool scores

**Change**
- Add `render_for_rerank()`, which changes only the reranker pair, not the payload:
  - drop the '| Column 1 |' scaffold, which all 14,515 table chunks carry;
  - fold the period lines into one 'Columns: FY…' line;
  - prepend the statement title, taking an orphaned title from the neighbouring chunk;
  - render rows as 'Label: v1 | v2';
  - merge cells that hold only '$' (13 of the 33 lost table units).
- Then test z-score fusion 0.5/0.5, K=8 for comparisons, and a logistic head fit on dev pools only.

**Targets after the package**
- In the pool, missed: class B 5 (dev-021, dev-025, dev-032, dev-035, test-034 DOV); class C 2 (dev-022 NVDA, dev-027 ADBE); class A 1 (dev-029); class D 5; class E 1 (test-037).
- Under eq, only dev-021, dev-027 and the 6 summary units count, so the ceiling is +3.0 (+1.67 after prose ±1).

**Gain**
- Fusion alone: +5.95 (Monte Carlo; worst case +4.17).
- On a similar package (the reranking package: term slot at the current depth + table rules + prose ±1): +0.84 strict and −1.2 under eq.
- K=8 for comparisons on top of that fused package: +2.85 strict (Monte Carlo).
- The dev-trained head reaches 59.86 on test.
- Realistic eq gain over the package: −1 to +1.
- The main value is letting Q1 shrink once the reranker ranks tables itself, which cuts tokens.

**Ship criteria:** on dev, table-vs-narrative pairwise accuracy rises from 24.2% toward the vector score's 66.7%, prose-vs-prose stays at 71.9% or above, and the change gains over the package under the eq judge.

## 6. Larger bets

### L1. Parser, chunker and contextual headers, with one combined re-embed

**Parser**
- Capture the statement title beyond the 55-pt caption window (`report_parser/tables.py`).
- Bind year lines to value columns by x-position: tables with a period header go from 21.8% to 47.2% of 12,055.
- Render a real header row.
- Merge '$' cells (3,063 tables).
- Restore or flag the 2,105 unlabelled numeric rows (1,627 tables), including ROK 'Total sales'.

**Chunker**
- Pack whole paragraphs and break at headings once a chunk has at least 300 tokens: +5% chunks without heading breaks, about +15% with them.
- Today 84 of 90 summary units are fully inside one chunk and 6 are split. All 101 prose gold units fit one chunk under the paragraph-aligned (+5%) and heading-≥100-token (+47%) variants; the ≥300-token variant was not checked per unit.
- Add a 12-token breadcrumb and a table header in `render_document`, and bump `RENDERING_VERSION`.
- Emit BGE-M3 sparse weights in the same pass. This is feasible: FlagEmbedding 1.3.5 `BGEM3FlagModel` is installed and `sparse_linear.pt` is in the local snapshot. `embedding.py` passes `return_sparse=False` today, and the identity checks would need extending. Do not store ColBERT vectors.

**Gain, speculative.**
- On the benchmark it shares the 6.7 real points left after Q1–Q3 and prose ±1.
- In the CPU proxy, embedded headers gave +3 / −1 comparative and −2 summary units (net +0.33).
- The main value is fewer wrong-column answers (test-028, test-025, dev-017, dev-034, B12) and smaller slots.

**Cost**
- An exact dev pilot of 4.7 min of GPU (38 reports, 11,083 chunks).
- A full re-embed of about 25–28 min (A2: 1,437 s today).
- Then rebuild FAISS and FTS and every chunk-id-keyed artifact.

**Risks**
- Headers make statements look alike.
- Mixing neighbour vectors at α=1.0 took the pool from 82.0 to 71.7, and mixing in the previous prose chunk (α=0.25) took the comparative pool from 80 to 65.
- D01 and D02 use global search and need the full corpus re-embedded.

### L2. Fine-tune the cross-encoder on dev-company synthetic pairs

- **Pairs:** template questions built from row labels and period headers; a fixed revenue / net sales / total sales map; hard negatives (comprehensive income, cash flows, TSR graph, boilerplate); risk-heading paraphrases.
- **Feasibility:** the GPU is an RTX 4060 Laptop with 8 GB. Fully fine-tuning about 568M parameters with fp32 AdamW needs about 9 GB before activations, so use LoRA or an 8-bit optimizer, gradient checkpointing and a capped training length. New pinned weights mean a new `MODEL_WEIGHTS_SHA256`.
- **Gain:** speculative, inside the shared headroom.
- **Abort** if dev prose-vs-prose pairwise accuracy drops below 71.9%, or if the Chinese live questions regress.

### L3. Dense row-level child index (after L1's period binding)

- **Change:** embed about 124,800 header-bound row texts (126,703 parsed rows), at 256–511 MB. Search them per company and return the parent chunks.
- **Gain:** 0 under eq on the benchmark. Its targets (test-029, test-034, test-035, test-025) are already recovered or eq-covered by the package. It may still help with live vocabulary gaps (B07 CAT ME&T).
- **Status:** the row-label BM25 leg gained only label-only units (+6), so use it only as an index for LTY.

## 7. What not to do

| Idea | Evidence against |
|---|---|
| Replace the vector pool with RRF or BM25 fusion | +7 / −16 labelled units (comparative −14); at depth 20, +7 / −8. In A5's reranked NVIDIA probes, every fused pool pushed the exact row out of the Top-5. Plain fusion's pool-only EvR@20 was +3.33 pp; it does not help after reranking. |
| Raw-question BM25 for templated comparisons | 9 of the 12 missed comparative rows rank 93–279 in scope. |
| MMR, per-page or per-table caps | MMR 57.19 / 57.10 / 54.68; the per-page cap loses 24 units (49.49); the per-table cap has no effect. |
| Drop or shrink the per-company quota | +3.8 / −3.0 units (net +0.4); a 1-slot quota gives 0.0. |
| Hard Item 1A filter | 7 gold units lie outside Item 1A, including dev-046 ×3, which are in the Top-5 today. |
| Scope-resolver work as a recall lever | 0 of 16 pool misses and 1 of 58 live facts. Keep it only inside M3. |
| Deeper pools or a larger K without slots | About 50% conversion; dev-032's gold reranked 10th of 20. Top-10 (Monte Carlo) 73.15 at 3,868 tokens, against 79.33 at 3,264, pending the scoring pass. |
| Row sub-chunks without headers | Size does not separate hits from misses (Fisher p=1.0). |
| Broad query rewriting, HyDE or synonym bags | Financial hints cost −2.00 pp EvR@20 against plain fusion (A5). True vocabulary misses are 4 units or fewer. The synonym bag reached only rank 69 for test-034. |
| Shipped rule decomposition as the live fix | Live facts in the Top-5: 21 to 22 of 40. decomposed_rule_v22 gave 0 correct answers and 11 false abstentions. |
| Mixing neighbour vectors into stored embeddings | α=1.0: 82.0 → 71.7; previous prose at α=0.25: comparative pool 80 → 65. |
| Store ColBERT multi-vectors | About 50 GB in fp16 (56,525 chunks × 431 tokens × 1,024 dimensions). |
| Add lexicon or regex entries from test or live failures | ROK 'Total sales (d)' and CAT ME&T would overfit. The live replay is already in-sample. |

## 8. Evaluation plan

1. **Judges.** Report strict and eq EvR for every variant, using v2 gold with `alternatives`.
2. **Protocol.**
   - Tune on dev only; freeze the configuration with `SELECTION_VERSION` and a config hash.
   - Run the benchmark test split once, as a secondary read, because it has been inspected.
   - Use the fresh held-out set as the primary read.
3. **Fresh held-out set.**
   - About 60 questions on the 73 report-years that neither the benchmark nor the live gold touches: 20 factual; 20 comparative, including ratio, growth and multi-year; 20 summary in at least 3 phrasings, with themes outside Item 1A.
   - The set is year-disjoint, not company-disjoint, since all 40 companies appear in the benchmark. Report dev-company and test-company subsets separately.
   - Gold-label everything, with all equivalent locations, before any run.
4. **Live.**
   - Replay the 58-fact table on the recorded pools; this is in-sample.
   - Then run a new live set of at least 20 cases and grade it blind in the bad-case table format.
   - Stop Streamlit (ask first) before any GPU run.
5. **Metrics**
   - Pool recall.
   - EvR@5 for the reranked Top-5 only, and EvR at the delivered context (slots included).
   - Complete-evidence rate.
   - End-to-end grades; wrong answers must not rise.
   - Tokens per question, overall and per category.
   - `context_budget_exceeded` and `missing_reranker_score` rates.
   - Latency p50 and p95, and reranker pairs per question.
6. **Controls and statistics**
   - Equal-token reranked Top-K, using the real scores from the Step 0 pass.
   - Deeper-vector control at equal pool size.
   - Per-unit 0-loss check.
   - Leave-one-direction-out attribution, since gains are not additive.
   - Paired bootstrap confidence intervals by question.
7. **GPU budget, one approved session.**
   - About 3,760 benchmark pairs (about 75 s, including the current 2,000).
   - 1,000 live pairs (about 20 s).
   - Table-aware re-score (about 40 s).
   - Model load (about 45 s).
   - The L1 dev pilot (4.7 min) comes later, under its own approval.
8. **Ship gates for Q1–Q3, on the fresh set**
   - EvR at the delivered context improves by at least 10 points under eq. The benchmark package gives +18.3.
   - 0 units lost.
   - Wrong answers do not increase.
   - Mean tokens stay at or below 1.6× today: the package is at 1.54×. Full prose ±1 is at 2.00× and fails, so ship the ±1,000-character window instead.
   - p95 latency rises by under 300 ms.

**Suggested order**
- **Week 1:** Step 0 (gold and gate decision), Q1, Q3 table rules.
- **Week 2:** Q2, the Q3 prose window, the GPU scoring pass, then M4 and M1's single-company and summary legs. Comparative depth 20 last.
- **Week 3:** M2 (summary is the main remaining gap: 6.67 of the 9.7 real eq points after the package), M3.
- **Then:** the L1 dev pilot.

**Measurement scripts.** The CPU prototypes behind these numbers (pool rebuilds, the Q1-Q3 package simulation, leave-one-out attribution) were written in a temporary working folder and are not part of the repository; Step 0 should re-implement the package as tested code before any number here is quoted as a result.
