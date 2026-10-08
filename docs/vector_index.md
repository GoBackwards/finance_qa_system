# Dense embeddings and FAISS indexes

The first dense baseline uses the existing **512-token / 50-token-overlap** chunks, **BAAI/bge-m3 dense embeddings**, and **FAISS IndexFlatIP**. The source corpus has 200 annual reports; the chosen configuration contains 56,525 chunks. One vector represents one complete chunk, and its original document, page citations, and evidence mappings remain available.

## Model choice

| Model | Dense dimensions | Maximum model tokens | Query instruction | Role here |
|---|---:|---:|---|---|
| `BAAI/bge-m3` | 1,024 | 8,192 | None | Initial baseline; accommodates preserved financial tables and fallback pages |
| `BAAI/bge-large-en-v1.5` | 1,024 | 512 | Documented retrieval prefix | Optional English-only comparison after checking input lengths |

These are model specifications, not measured financial-retrieval rankings. See the official [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) and [BGE-large model card](https://huggingface.co/BAAI/bge-large-en-v1.5). The English model can be competitive on short English passages; the current corpus's intact tables make M3 the more practical first choice. Actual retrieval quality must be measured on the same questions.

The previous chunk sizes use `cl100k_base`. These are **not the embedding model's tokens**. Every complete embedding input is checked with its own pinned tokenizer, including metadata and special tokens. Any overflow fails the run and writes `preflight.json`; nothing is silently truncated, skipped, or split. This preserves the chunking experiment's source evidence. The 400 explicitly oversized chunks at 512/50 already exceed that nominal BPE target, and ordinary 512-BPE-token prose may also exceed BGE-large's 512-token limit.

Model revisions are immutable and downloaded as local snapshots before FlagEmbedding loads them:

```text
BAAI/bge-m3: 5617a9f61b028005a4858fdac845db406aefb181
BAAI/bge-large-en-v1.5: d4aa6901d3a41ba39fb536a557fa166f842b0e09
```

The tokenizer-only audit on 2026-10-08 checked every final input, including its metadata and special tokens:

| Model | Chunks checked | Mean model tokens | Longest input | Inputs exceeding model limit |
|---|---:|---:|---:|---:|
| BGE-M3 | 56,525 | 431.14 | 7,921 | 0 |
| BGE-large-en-v1.5 | 56,525 | 381.24 | 5,230 | 14,704 (26.01%) |

BGE-large overflow includes 14,044 prose chunks, 306 structured table chunks, 95 table fallbacks, and 259 page fallbacks. Its audit intentionally stops before embedding. A later BGE-large comparison must introduce and document a model-aware segmentation strategy; silently truncating these inputs would invalidate the evidence-preservation guarantee. The source chunk SHA-256 for both audits is `d9dc54b432956aec565f3ede839e0062ee90d2c7166f48eaf22e733f6da474c6`.

Passages receive the same deterministic company/ticker/fiscal-year prefix before their unchanged chunk text. Queries contain only the analyst question, plus BGE-large's official query instruction when that model is selected. Evaluation labels, reference answers, source IDs, and gold evidence are never embedding inputs. M3 sparse and ColBERT outputs are disabled in this stage.

## Installation and execution

The project uses a local `.venv`. Install its optional vector dependencies with `uv`. For this Windows RTX 4060 environment, install the CUDA wheel first:

```powershell
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv -e ".[experiments,vectors]"
```

For CPU-only environments, omit the CUDA-specific first command and use `--device cpu`; inference will be slower. Exact package versions are recorded with the model identity. Model downloads are cached in `.cache/huggingface/` under the workspace.

Build the exact baseline and evaluate the existing starter questions:

```powershell
.\.venv\Scripts\python.exe scripts/build_vector_index.py --device cuda:0 --batch-size 8
```

Perform a tokenizer-only audit without downloading model weights:

```powershell
.\.venv\Scripts\python.exe scripts/build_vector_index.py --model BAAI/bge-large-en-v1.5 --preflight-only
```

After the Flat baseline, compare IVF and HNSW using the same saved embeddings:

```powershell
.\.venv\Scripts\python.exe scripts/build_vector_index.py --device cuda:0 --index-types flat ivf hnsw --threads 4
```

Completed embedding shards are reused only after checking source/model/tokenizer/configuration fingerprints and artifact hashes. Source changes require a separate output directory. An interrupted run resumes from its last durable shard. CUDA runs use FP16 model inference; vectors are normalized and persisted in float32. CPU inference uses FP32. Floating-point inference can vary across hardware or library versions; saved-vector hashes identify the exact matrix used in an index comparison.

CUDA allocation defaults to 60% of device memory (`--gpu-memory-fraction 0.6`). FlagEmbedding can reduce the inference batch when necessary; this never shortens inputs. Unused GPU cache is released before CPU index work. The full corpus run exercised checkpoint resumption after host memory pressure at 49,664 saved vectors; model, source inputs, precision, and requested batch size remained unchanged. `runtime_restart.json` records that boundary and allocator change.

## Index design and comparisons

All vectors and query vectors are L2-normalized, so inner product is cosine similarity. Flat performs exact nearest-neighbor search without a training stage. At 56,525 vectors and 1,024 dimensions, the float32 vector payload alone occupies about 220.8 MiB. This is small enough to make Flat a useful baseline. [FAISS metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances), [index selection guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

Optional indexes start with these explicit experimental parameters:

| Index | Initial settings | Relevant tradeoff |
|---|---|---|
| Flat IP | Exact exhaustive search | Reference neighbors; no approximation loss |
| IVF Flat IP | `nlist=256`, `nprobe=16`, seeded training | Probe more clusters to increase recall at higher search cost |
| HNSW Flat IP | `M=32`, `efConstruction=200`, `efSearch=64` | More graph links/search work can improve recall while increasing memory or latency |

IVF uses a deterministic training sample and adapts the number of lists for small test corpora; both requested and effective settings are recorded. HNSW uses one construction thread for reproducibility. Search thread count is explicit. These defaults are starting points; later sweeps can vary `nprobe` and `efSearch`. [FAISS index parameters](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

Keep two metrics separate:

- **Gold-evidence Recall@5:** the existing benchmark's canonical source spans/rows recovered in the top five chunks. This measures retrieval usefulness against those labels.
- **ANN neighbor Recall@5:** overlap with exact Flat's five vector neighbors. This measures approximation loss, even when those neighbors do not answer the question.

The comparison also records index build time, index bytes, warmed search p50/p95 latency, and query-embedding time separately. Embedding costs are shared across indexes. Search timing excludes model inference and artifact loading; a 40-question pilot is not a production load test. Fixed top-k can retrieve different token budgets, so source-token context totals are recorded as well.

## Output records

Default output: `data/indexes/bge-m3/table_preserving_s512_o50/`.

| Artifact | Contents |
|---|---|
| `index_creation.json` | Model/revision, dimension, distinct report count, chunk/vector count, index records, evidence metrics |
| `embeddings/preflight.json` | Actual model-token lengths, overflow counts, source/model identity |
| `embeddings/embedding_manifest.json` | Complete embedding run and resumable shard provenance |
| `embeddings/vectors.npy` | Shared normalized float32 vector matrix in original chunk order |
| `flat.faiss` | Exact FAISS index; optional `ivf.faiss` and `hnsw.faiss` |
| `flat.manifest.json` | Index settings, dimension/count, training/build measurements, software and artifact hashes |
| `flat.metadata.jsonl` | Vector IDs mapped to chunk IDs and source JSONL byte offsets |
| `query_manifest.json` / `query_vectors.npy` | Benchmark query order, model identity, saved query embeddings |
| `flat_retrieval_results.json` | Per-question evidence matches and original document/page citations |
| `ann_comparison.json` | Neighbor overlap against Flat and measured search latency |

`document_count` means distinct annual reports, not vector count. `vector_count` and `chunk_count` must match. An index with a mismatched source hash is rejected before source payloads are read. The original chunks remain authoritative; vector metadata does not replace page-level evidence.

Large PDFs, parsed text, embedding shards, and FAISS indexes are excluded from Git. Code, dependency pins, corpus provenance, benchmark labels, and documentation are versioned. The local repository's origin is `https://github.com/GoBackwards/finance_qa_system.git`.

## Measured index run — 2026-10-08

The full run produced **56,525 normalized float32 vectors of dimension 1,024 from 200 annual reports**, with zero truncated or skipped chunks. The actual longest input was 7,921 M3 tokens. All three persisted indexes contain the same vectors and original source mappings. The source and matrix hashes, model revision, settings, and measurements are saved in the versioned [result record](vector_index_results.json); full per-query judgments remain in the generated artifact directory.

| Index | Size (MiB) | Build and validation (s) | Search median (ms) | Search p95 (ms) | Neighbor Recall@5 vs Flat | Gold-evidence Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| Flat IP | 220.8 | 12.37 | 31.85 | 40.08 | 100% | 48.75% |
| IVF Flat IP | 222.2 | 15.52 | 0.95 | 1.45 | 100% | 48.75% |
| HNSW Flat IP | 235.5 | 46.31 | 0.21 | 0.45 | 100% | 48.75% |

Measurements use the same **40 agent-curated development questions**, four FAISS CPU search threads, two warmup passes, and three timed single-query searches per question. Search latency excludes embedding, loading, and validation. Build time includes validation and artifact writing, but excludes model inference. HNSW construction uses one thread; IVF uses 256 lists and 16 probes. These are measurements on this laptop, not a production latency or approximation-recall guarantee. The approximate indexes matched every exact top-five neighbor in this pilot; that does not imply perfect agreement for other questions.

The prior BM25 baseline scored **50.00%** overall, versus dense retrieval's **48.75%**. Dense table recall was **57.5%** (BM25: 35%), while dense prose recall was **40%** (BM25: 65%). This motivates a later hybrid experiment; it does not establish that combining the retrievers will improve results. The small development set and non-exhaustive source judgments remain limitations. Retain Flat as the exact reference when tuning approximate indexes or evaluating new questions.

The independent artifact audit checked all vector norms/dimensions, persisted index sizes and counts, hashes, source mappings, and every retrieved evidence judgment. All **127 tests passed**, including a regression for near-tie arithmetic differences between batch and single-query Flat search. Exact-neighbor references now use the same single-query shape and thread settings as timed searches. Indexes are loaded one at a time during benchmarking to reduce peak memory.
