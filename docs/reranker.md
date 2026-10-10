# BGE reranker integration

The experiment compares **direct vector Top 5** with **vector Top 20 → BGE reranker → Top 5**. See the [measured comparison](reranker_comparison.md) and [machine-readable results](../data/evaluation/reranker_results.json) for outcomes, category breakdowns, uncertainty, and runtime.

## Fixed comparison

The 100-question benchmark retains its existing 50 development and 50 test questions. The model, candidate depth, and output depth are fixed; neither split selects reranker parameters. Test results are reported separately.

The runner verifies the previous exact-vector experiment's benchmark, query embeddings, and ranking checksums, then takes the first 20 chunks from each frozen Top 100 ranking. Direct Top 5 is that same ranking's first five chunks. No extra filtering, query rewriting, or evidence injection changes the candidate pool. Recall@20 measures the evidence available to selection: a reranker cannot recover absent candidates, and five chunks may not cover everything available in twenty.

Each model input contains only the question and the original chunk text with its existing company, ticker, and fiscal-year prefix. Reference answers and gold document IDs never enter the model. Selection preserves complete chunk text, table rows, evidence offsets, and page citations. Raw relevance logits determine order; ties preserve original vector rank. Logits are not answer-correctness probabilities.

## Model and complete inputs

The adapter uses FlagEmbedding's `FlagReranker` with the pinned [BAAI/bge-reranker-v2-m3 snapshot](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e):

- Revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Weight file: `model.safetensors`, verified before loading against SHA256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.
- Input limit: 8,192 tokens for the complete question/passage pair, including special tokens. Overflow rejects the run before inference; inputs are never silently truncated.
- CUDA inference uses FP16; CPU inference uses FP32. The default batch size is four.

The [recorded preflight](../data/evaluation/reranker_results.json) covers 2,000 pairs: maximum length 1,566 tokens, with nine exceeding 1,024. BGE's authors [recommend 1,024 based on fine-tuning length while confirming support for 8,192](https://huggingface.co/BAAI/bge-reranker-v2-m3/discussions/9). This experiment uses the larger limit to preserve those passages; support for longer inputs does not guarantee equal ranking quality at every length.

Installed FlagEmbedding 1.3.5 can retry arbitrary runtime errors while reducing its batch size to zero. The adapter retains its tokenizer, model, paired-input format, and raw-logit scoring, but retries only CUDA allocation failures and stops if one pair cannot fit. A smoke check compares complete token IDs and scores with stock FlagEmbedding on the same loaded model before scoring the benchmark. The recorded run matched token IDs exactly and had zero absolute logit difference across four warmup pairs (`rtol=1e-3, atol=1e-3`); this is a smoke check, not an exhaustive equivalence proof. See the [official usage](https://huggingface.co/BAAI/bge-reranker-v2-m3#using-flagembedding).

## Metrics

| Metric | Meaning |
|---|---|
| Evidence Recall@5 | Fraction of required source units fully covered by the union of five selected chunks, averaged per question. |
| Document Recall@5 | Fraction of distinct gold report IDs represented among five chunks, averaged per question. |
| Evidence accuracy@1 | Fraction of questions whose first chunk individually covers at least one annotated source unit. |
| Annotated Precision@5 | Fraction of five selected chunks that each individually cover an annotated source unit. |
| Complete evidence@5 | Fraction of questions for which all annotated source units are covered. |

Accuracy@1 measures source selection. **Generated-answer accuracy is not measured:** no answer generator or answer grader is evaluated. Gold source units are nonexhaustive, so useful unjudged passages receive no evidence credit. The agent-authored benchmark still needs independent human semantic review. Paired bootstrap intervals describe uncertainty on these questions and ignore dependence between company and question templates.

## Run and reuse

Use the project environment with its `vectors` and `experiments` dependencies. The [retrieval experiment](retrieval_experiment.md), frozen rankings under `data/experiments/retrieval_k_sweep/`, and parsed source documents must already exist.

```powershell
# Audit complete input lengths without loading model weights.
.\.venv\Scripts\python.exe scripts/run_reranker_experiment.py --preflight-only

# Run or resume the default CUDA experiment.
.\.venv\Scripts\python.exe scripts/run_reranker_experiment.py

# CPU uses a separate output directory because its precision differs.
.\.venv\Scripts\python.exe scripts/run_reranker_experiment.py --device cpu --output data/experiments/reranker_cpu
```

Model weights and tokenizer files download to `.cache/huggingface/` when absent. Scoring runs in an isolated process so it does not keep the dense encoder or FAISS index resident. The CUDA worker caps its allocation at 60% of device memory. Measured reranker latency includes tokenization, transfers, and synchronized inference after warmup; it excludes retrieval, model loading, checkpoint writes, ranking, and evaluation.

For application use, pass the vector retriever's unchanged Top 20 results:

```python
from report_search.reranker import BGEReranker

reranker = BGEReranker(device="cuda:0", batch_size=4)
# vector_top20 contains the original ranked chunk dictionaries and citations.
top5 = reranker.rank_candidates(query, vector_top20, top_k=5)
```

Selected chunks retain `vector_rank` and `vector_score`, and add `reranker_score`; `rank` and `score` describe the final ranking. The adapter also exposes `preflight_pairs()` and `score_pairs()` for explicit question/passage pairs. The demo path (`DemoService`) instead reranks per-company pools when the question names companies, then selects a balanced Top 5; see [system architecture](system_architecture.md).

Generated candidates, preflight audits, atomic scoring checkpoints, scores, and full evaluation records remain under `data/experiments/reranker/`. Resume checks bind scores to candidate content/order, model and tokenizer identity, package versions, device, batch size, and implementation hashes. Use a fresh `--output` directory after configuration, dependency, or scoring-code changes. PDFs, model weights, and generated caches remain excluded from Git.
