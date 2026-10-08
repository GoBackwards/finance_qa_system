# Chunking retrieval pilot

The benchmark in [`data/evaluation/chunking_questions.json`](../data/evaluation/chunking_questions.json) contains **40 source-grounded analyst questions**, one for each company in the downloaded corpus. It is an agent-curated development set for a controlled chunking experiment, not an independently human-labeled, held-out benchmark.

The questions were written after inspecting the source passages and tables, before seeing retrieval scores. Do not index the questions' reference answers, evidence quotes, or judgment metadata. They are evaluation labels.

## Coverage

| Dimension | Coverage |
| --- | --- |
| Companies | 40, one question per company |
| Sectors | 5, eight questions per sector |
| Source fiscal years | FY2021–FY2025, eight questions per year |
| Narrative questions | 20 |
| Table questions | 20 |
| Questions requiring multiple evidence units | 8 |
| Required evidence units | 48: 23 narrative spans and 25 table rows |

Questions explicitly name the company and the source annual report's fiscal year. This distinguishes, for example, a value in the FY2021 report from the same historical value repeated in a later filing. Questions cover financial-statement amounts, business segments, revenue recognition, supply chains, distribution operations, and operating strategy. Several require two rows, two passages, or two tables on different pages.

Only readable narrative passages and accepted structured table rows were selected. The pilot does not cover rejected table candidates, image-only evidence, chart interpretation, unanswerable queries, or cross-company synthesis. These are gaps to address in a later benchmark.

## Stable source judgments

Chunk IDs depend on chunk size and overlap, so they cannot be the ground truth. Every required evidence unit instead identifies the unchanged source document and its 1-based PDF page:

- A narrative unit stores `element_id`, its exact `quote`, and half-open character offsets `[char_start, char_end)` in the **original parsed element text**. Unicode characters and source whitespace count as stored; offsets must not be applied to a normalized copy.
- A table unit stores `table_id`, a **0-based** `row_index`, complete `row_cells`, a readable `row_text`, the row bounding box, and the table's caption and surrounding header context. Its canonical key is `table_id + ":row:" + row_index`.
- Both kinds include the source PDF's SHA-256, document ID, filename, ticker, fiscal year, and bounding box. Re-parsing that changes element IDs requires revalidating the labels, even when the PDF hash is unchanged.
- `alternatives`, when present, lists verified equivalent source locations for a unit. Recovering one alternative satisfies that unit; it does not add another item to the denominator. Equivalent passages elsewhere may remain unjudged.

Table labels come from `rows` on tables with `structured_reliable: true`. They never use `candidate_rows` from failed table extraction. Dates and units were checked separately from the numeric values: Amazon's selected table places 2021 before 2022, whereas many other tables place the current year first. Costco's FY2024 and FY2023 columns cover 52 and 53 weeks, respectively.

Units can fall outside the parser's immediate table-caption window. Costco's selected statement places its units in the page heading; Salesforce places them in a preceding sentence. Deere's cash-flow summary inherits the MD&A section's units statement from PDF page 30. These contexts are retained in the labels for answer review. They are not extra scored units, so this retrieval metric alone does not certify that every retrieved table chunk includes sufficient date and unit context.

## Recall@5 definition

Retrieve the top five ranked chunks for each question. For each required unit:

1. A narrative unit is recovered when the union of source intervals present in the five chunks covers its entire quoted interval within the same source element. This allows a passage split at a chunk boundary to be recovered by two retrieved chunks. Disconnected fragments with a missing interval do not qualify.
2. A table unit is recovered only when a retrieved chunk contains the complete canonical row. Partial row text or a matching number elsewhere does not qualify.
3. Repeated overlap does not count the same evidence unit twice. Any listed equivalent source location may satisfy its original unit.

For a question with `m` required units, evidence Recall@5 is `recovered_units / m`. Average that value across the 40 questions to obtain **macro evidence Recall@5**. Also report narrative and table subsets, plus the fraction of questions for which all required evidence was recovered. A question that recovers one of two necessary rows has recall 0.5, but does not count as complete evidence success.

This is a source-evidence retrieval metric. It differs from recall over arbitrarily labeled chunk IDs and from answer correctness. A correct answer supported by an unjudged passage may receive a lower score. An evidence hit also does not demonstrate that an answer generator will interpret periods, signs, units, or accounting definitions correctly.

## Experimental controls

Keep corpus coverage, questions, tokenizer, retrieval model, ranking parameters, and `k=5` fixed while comparing chunk sizes and overlaps. A full 3×3 experiment with target sizes 512/1024/2048 tokens and overlaps 50/100/200 tokens separates size effects from overlap effects. A diagonal-only comparison confounds both changes.

Use identical provenance rules for every group. Table rows remain atomic, and repeated headers or context must not create extra evidence units. Oversized rows need an explicit overflow rule instead of silent splitting. Report actual chunk lengths and oversize counts; target length alone does not describe the resulting corpus.

The initial local BM25 evaluation is a lexical baseline. Results should be identified as BM25 results, not embedding retrieval results. Re-run the unchanged benchmark after selecting an embedding model or hybrid retriever. If company/year filters are used, label those results separately from retrieval across all 200 reports; do not infer filters from hidden gold metadata without reporting that advantage.

## Validation and interpretation

All 48 labels were checked against the stored parsed source. All 23 narrative spans also matched whitespace-normalized native text extracted directly from the corresponding raw-PDF bounding boxes. The 25 selected table rows were compared with native raw-PDF text, and their year-column order and units were inspected. This is agent review, not independent human validation.

Treat scores as exploratory measurements for this corpus and retriever. Forty selected questions are too few to establish a broadly optimal chunk size. Inspect per-question wins and losses, annotate additional equivalent evidence, and add independently written questions before choosing a production configuration. Once scores influence implementation or question selection, this set remains a development set; use a separate held-out set for final evaluation.
