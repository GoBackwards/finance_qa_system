# BGE reranker experiment

The same 50 frozen test questions compare direct vector Top 5 with vector Top 20 reranked by `BAAI/bge-reranker-v2-m3` to Top 5. Development results are recorded separately. The model and candidate depth are fixed; no labels select parameters.

| Approach | Evidence Recall@5 | Annotated Precision@5 | Evidence accuracy@1 | Complete evidence@5 | Document Recall@5 |
|---|---:|---:|---:|---:|---:|
| Vector Top 5 | 37.33% | 10.40% | 14.00% | 26.00% | 87.00% |
| Vector Top 20 → rerank Top 5 | 48.00% | 14.00% | 26.00% | 36.00% | 90.00% |
| Change (percentage points) | +10.67 | +3.60 | +12.00 | +10.00 | +3.00 |

The unchanged Top 20 pool has **60.00% evidence Recall@20**. This bounds the evidence recall available to selection; a reranker cannot retrieve missing candidates, and five chunks may not cover every unit available in twenty.

**Accuracy@1** means that the first passage individually covers at least one annotated evidence unit. Precision@5 is the fraction of five selected passages that each cover an annotated unit. Evidence Recall@5 credits complete source units recovered by the union of five passages. Complete evidence@5 requires every annotated unit. These are retrieval metrics; **generated-answer accuracy is not measured**.

## Category results

| Category | Approach | Evidence Recall@5 | Precision@5 | Evidence accuracy@1 | Complete evidence@5 |
|---|---|---:|---:|---:|---:|
| factual | direct_top5 | 65.00% | 13.00% | 20.00% | 65.00% |
| factual | reranked_top5 | 75.00% | 15.00% | 40.00% | 75.00% |
| comparative | direct_top5 | 20.00% | 8.00% | 6.67% | 0.00% |
| comparative | reranked_top5 | 33.33% | 13.33% | 26.67% | 13.33% |
| summary | direct_top5 | 17.78% | 9.33% | 13.33% | 0.00% |
| summary | reranked_top5 | 26.67% | 13.33% | 6.67% | 6.67% |

## Paired changes and runtime

| Metric | Change (pp) | Exploratory 95% interval (pp) | Improved / unchanged / worsened queries |
|---|---:|---:|---:|
| evidence_recall_at_5 | +10.67 | [+3.67, +18.34] | 11 / 37 / 2 |
| annotated_evidence_precision_at_5 | +3.60 | [+1.20, +6.40] | 11 / 37 / 2 |
| evidence_accuracy_at_1 | +12.00 | [-2.00, +26.00] | 10 / 36 / 4 |
| all_evidence_hit_at_5 | +10.00 | [+2.00, +18.00] | 5 / 45 / 0 |

Added reranker latency: test median **388.8 ms**, p95 **546.3 ms** per twenty-candidate query. One scoring pass on `cuda:0` per question after warmup; includes tokenization, transfer and synchronized inference, excludes vector retrieval/model loading. This is an observed experiment cost, not a production load benchmark.

All 2,000 development/test pairs fit the reranker's 8,192-token combined limit; maximum 1,566, mean 464.4. The default 512-token limit would truncate 1,062 pairs; 1,024 would truncate 9. No passage, table row, query or citation was truncated or skipped. Input text uses the existing company/ticker/fiscal-year prefix and original chunk text.

The reference answers never enter the reranker. Raw cross-encoder logits rank candidates; they are not correctness probabilities. Score ties preserve original vector rank. Models and tokenizers are pinned and source/candidate/query/model artifacts are checksummed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/run_reranker_experiment.py
```

The script extracts the unchanged vector Top 20 from the verified prior run, preflights complete pairs, scores in an isolated process with atomic resumable checkpoints, and evaluates source evidence after inference. Use `--preflight-only` to audit token lengths without loading model weights, or `--device cpu` for CPU inference. Use a new `--output` directory when changing configuration. Full rankings, score records and checkpoints remain under `data/experiments/reranker/`. See the [machine-readable results](../data/evaluation/reranker_results.json).

## Limitations

- Agent-authored, source-checked labels await independent human semantic review.
- Gold evidence is non-exhaustive; valid unjudged passages receive no evidence credit.
- Retrieval accuracy and complete evidence coverage do not measure generated-answer accuracy; no answer generator is evaluated.
- Summary coverage measures selected source claims, not completeness of a generated summary.
- A fixed number of chunks is not a fixed context-token budget; context token counts include overlap.
- Paired query bootstrap intervals are exploratory and ignore within-company and question-template dependence.
- BGE's authors recommend 1,024 tokens based on fine-tuning length, while the architecture supports 8,192. This experiment keeps longer source passages intact; the longer setting is not a claim of equal ranking quality at every length.

Sources: [pinned model](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e), [author input-length guidance](https://huggingface.co/BAAI/bge-reranker-v2-m3/discussions/9).

## Interpretation

Evidence Recall@5 improved by 10.67 percentage points on the 50 frozen test questions. The exploratory paired interval is +3.67 to +18.34 points. The accuracy@1 interval crosses zero (-2 to +26 points), and summary-question accuracy@1 fell from 13.33% to 6.67% despite better Top-5 recall. This supports using reranking for further evaluation, while leaving its effect on generated-answer accuracy unmeasured.

The adapter passed the same-model stock FlagEmbedding smoke check: identical paired token IDs and zero absolute logit difference on four warmup pairs. All 2,000 complete pairs were scored without truncation. The project suite passes 212 tests. See the [integration and reproduction guide](reranker.md).
