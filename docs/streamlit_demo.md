# Financial report Q&A demo

The local Streamlit app searches the existing annual-report corpus, reranks the best 20 vector matches to five passages, and shows their document and PDF-page sources. A configured generation endpoint can produce a cited answer through the existing confidence, source-binding, reference-validation, and semantic-review checks.

Retrieval works without a generation endpoint. The app has no canned-answer mode. A refused answer displays **"I'm not sure"** while the retrieved passages remain available for inspection. Retrieved passages are search results; their presence does not mean they support an answer.

## Start the app

Run these commands from the repository root using the project's Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo,vectors]"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). The checked-in Streamlit configuration binds to loopback, uses port 8501, runs headlessly, disables usage telemetry, and disables file watching. Stop the server with Ctrl+C in its launch terminal.

The default inference device is `cpu`. On this project's CUDA-capable machine, select the GPU before starting the server:

```powershell
$env:FINANCE_QA_DEVICE = 'cuda:0'
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The device is a server setting. Restart the server to change it; there is no browser control that creates a second set of resident models. CUDA mode requires a working CUDA-enabled PyTorch installation. CPU and CUDA use different inference precision, so a threshold measured with one configuration should not be assumed calibrated for the other.

The first search loads and verifies the indexes and model weights. Later searches reuse those resources. The app uses the locally prepared snapshots and does not download models during a request.

## Required local artifacts

The repository contains code and compact experiment results. Large source and model artifacts are excluded from Git and must already be prepared:

| Artifact | Location |
|---|---|
| Report inventory | `data/manifest.json` |
| Original PDFs | `data/raw/<TICKER>/<TICKER>_FY<YEAR>_annual_report.pdf` |
| Parsed source documents | `data/processed/<TICKER>/<REPORT>/` |
| Table-preserving chunks | `data/experiments/chunking/table_preserving_s512_o50/chunks.jsonl` |
| Flat index, metadata, embedding manifest and vectors | `data/indexes/bge-m3/table_preserving_s512_o50/` |
| SQLite chunk payload store | `data/indexes/bm25_s512_o50/retrieval.sqlite` |
| Pinned encoder and reranker snapshots | `.cache/huggingface/` |

The demo uses the vector path; the SQLite store supplies the matching chunk payloads. It does not enable query expansion or hybrid retrieval. See [vector indexing](vector_index.md), [reranking](reranker.md), and [generation safeguards](generation_constraints.md) for the preparation pipeline and artifact contracts.

## Search and inspect evidence

1. Enter a financial-report question and submit the search form. The original question goes to BGE-M3, Flat retrieval selects 20 candidates, and the BGE reranker selects five.
2. Inspect the selected passages and, when useful, the full initial candidate list. Check fiscal-year headings, units, table warnings, and whether a passage concerns the whole company or a business segment.
3. Open the public report at the cited PDF page, download the verified local PDF, or inspect the local page preview. Page numbers are one-based PDF positions and can differ from printed report page numbers. Browser support for opening a remote PDF at a page varies; the local preview identifies the requested page directly.
4. To request generation, enable **Generate a cited answer** and configure the endpoint and an explicit threshold in the sidebar before submitting. The app displays a verified answer or a refusal, together with diagnostic information.

Only the completed result is displayed. A rejected draft is never shown as an answer. A citation includes a source paragraph or table-row locator and its supporting quote; the application supplies the document and page identity.

## Configure generation

The sidebar accepts an API base URL, generator model, password-masked API key, optional verifier model, completion-token parameter, confidence threshold, and generation deadline. These settings and results belong to the browser session. Editing them does not modify the server's environment or another session's settings.

Environment variables can provide the initial settings when launching the server:

| Variable | Meaning and default |
|---|---|
| `FINANCE_QA_DEVICE` | `cpu` by default; set `cuda:0` before launch to select the first GPU. |
| `FINANCE_QA_BASE_URL` | Compatible API base, normally ending in `/v1`. The client appends `/chat/completions`. Empty means no configured endpoint. |
| `FINANCE_QA_MODEL` | Generator model identifier; empty by default. |
| `FINANCE_QA_API_KEY` | Provider credential. It may be empty for a local endpoint that does not require authentication. |
| `FINANCE_QA_VERIFIER_MODEL` | Review model on the same endpoint; blank uses the generator model in a separate request. |
| `FINANCE_QA_TOKEN_PARAMETER` | `max_completion_tokens` by default; select `max_tokens` if the compatible endpoint requires it. |
| `FINANCE_QA_THRESHOLD` | Explicit finite raw reranker cutoff. Blank leaves generation in refusal-only mode. |
| `FINANCE_QA_DEADLINE_SECONDS` | Positive generation-layer budget; default 30 seconds. |

[`.env.example`](../.env.example) documents these names. **The application does not automatically load `.env` files.** Set variables in the launch environment or enter generation settings in the sidebar. For example:

```powershell
$env:FINANCE_QA_BASE_URL = 'https://your-provider.example/v1'
$env:FINANCE_QA_MODEL = 'your-generator-model'
$env:FINANCE_QA_VERIFIER_MODEL = 'your-review-model'
$env:FINANCE_QA_DEADLINE_SECONDS = '30'
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Supply the actual credential and chosen threshold through the sidebar or your launch environment. No provider or model is chosen implicitly. Remote endpoints require HTTPS; local loopback HTTP is supported. The endpoint must accept JSON-mode Chat Completions requests and the selected completion-token parameter.

The threshold compares the highest selected raw reranker logit. It is not a probability of correctness. The historical `6.359375` diagnostic is documented in [generation constraints](generation_constraints.md); the demo does not install it as a default or certify it for this server's configuration. Passing a score threshold still requires source and claim verification. Newly calculated numerical values are outside the current generator's supported behavior.

The generation deadline starts after retrieval. Checks at stage boundaries discard a late result and prevent later stages, but cannot interrupt an active synchronous API call. Provider socket timeouts are separate. A 30-second generation budget is not an end-to-end response-time guarantee.

## Resource and session behavior

One shared service owns the resident encoder, FAISS index, reranker, and source store. A dedicated worker creates, uses, and closes its SQLite connection and serializes model work. Queries, generated answers, and provider credentials are not stored in that shared resource cache; session settings are supplied for each request.

This design follows Streamlit's requirement that [globally cached resources be thread-safe](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource) and its [pattern of keeping Streamlit commands outside custom worker threads](https://docs.streamlit.io/develop/concepts/design/multithreading). It also preserves the existing retrieval engine's SQLite thread ownership. A request may wait for another session's work; this local demo has not been load-tested as a multi-user service.

Source actions resolve documents through the checked report catalog rather than accepting arbitrary filesystem paths. The preview and download are bound to the report identity and local PDF hash. Source passages still need careful interpretation: a verified file or exact quotation does not establish that a claim follows from it.

## Validation scope

The completed regression run covered **410 tests: 409 passed and one Windows symlink-privilege test was skipped**. Resolved-path confinement is also covered without needing that privilege. This includes ten service tests, thirteen source-catalog tests, and seven Streamlit AppTest flows. The suite ran in an isolated snapshot that preserved unrelated workspace edits.

A real Chrome browser exercised resident BGE-M3 and BGE reranker models on the local RTX 4060 GPU, using the existing 200-report corpus. Both Microsoft and NVIDIA queries returned 20 candidates and five selected passages. The browser displayed a final answer and its cited source through two prescribed localhost responses (one generator call, one reviewer call); no paid or live LLM was used. Opening the cited page did not regenerate the answer. The downloaded Microsoft PDF matched the manifest's SHA256. See [recorded demo checks](../data/evaluation/streamlit_demo_checks.json).

Live generation quality remains unvalidated because no external model endpoint is configured. The controlled responses establish integration and refusal behavior, not generated-answer accuracy or reviewer reliability. The single warm browser timing is an interactive smoke measurement, including browser synchronization; it is not a latency benchmark or service-level guarantee.
