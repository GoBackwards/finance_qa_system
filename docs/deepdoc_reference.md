# DeepDoc reference for the financial-report parser

Research checked on 2026-10-08. These are architectural observations and project design decisions, not copied RAGFlow code or a claim that the local parser runs DeepDoc models.

## Version matters

RAGFlow's current main branch uses Go and places the PDF pipeline under `internal/deepdoc/parser/pdf`. The reviewed snapshot is [commit ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c](https://github.com/infiniflow/ragflow/commit/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c). Old links to `main/deepdoc/parser/pdf_parser.py` therefore return 404. For the older Python module those links refer to, this review also pins [v0.27.2 / a024bea0cd93f39e6652a42bf84dd20c55bc560b](https://github.com/infiniflow/ragflow/tree/a024bea0cd93f39e6652a42bf84dd20c55bc560b/deepdoc).

## What to learn from DeepDoc

1. **Separate text acquisition from document structure.** The current pipeline extracts embedded characters, renders each page, chooses an OCR path according to character quality, then applies document-layout analysis (DLA) and table-structure recognition (TSR). Clean embedded characters can be matched to detected text boxes; empty/garbled pages need recognition. Assembly across pages happens after page-local processing. [Current parser](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/parser.go#L169)

2. **Keep layout types and region identities.** The Python layout recognizer distinguishes text, title, figure, figure caption, table, table caption, header, footer, reference, and equation. It matches text boxes to visual regions by overlap, retains region IDs, and can add placeholders for figures with no text. Its header/footer filtering has page-margin guards and learns repeated discarded text. Do not interpret every reference or footnote in a financial report as disposable. [Python layout recognizer](https://github.com/infiniflow/ragflow/blob/a024bea0cd93f39e6652a42bf84dd20c55bc560b/deepdoc/vision/layout_recognizer.py#L32)

3. **Determine reading order before joining paragraphs.** Current column detection combines whitespace-gutter evidence with clustering and guards against treating narrow table cells as prose columns. Later ordering is page, column, vertical position, then horizontal position. Merging uses geometry and layout information; vertical merging stays within a page and column. [Column detection](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/layout/combined_column.go#L13), [text ordering and merging](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/layout/layout.go#L415)

4. **Find running matter while line geometry is intact.** Current header/footer removal combines page-margin position, explicitly tagged regions, page-number patterns, repeated text across pages, alternating odd/even headers, and stable local runs. Whitespace can expand its margin zones. A separate numeric-sequence check helps distinguish page numbers from arbitrary numbers. The parser runs this before text merging, which could otherwise absorb a header into the body. [Header/footer logic](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/layout/toc_header_footer.go#L802), [pipeline order](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/parser.go#L514)

5. **Treat tables as structured regions.** DeepDoc detects a table region, crops it, predicts rows/columns/headers/spanning cells, and associates text with those structures. Its Python table builder can represent row spans and column spans in HTML. The current pipeline can evaluate table orientation, transform coordinates back to the page, retain cell annotations, and replace table text in the prose stream. This is substantially more than extracting a page into plain text. [Python TSR](https://github.com/infiniflow/ragflow/blob/a024bea0cd93f39e6652a42bf84dd20c55bc560b/deepdoc/vision/table_structure_recognizer.py#L30), [current table extraction](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/table_extract.go#L68)

6. **Make provenance part of every result.** Current sections and table items carry text/rows alongside page positions, bounding boxes, layout type, and optional table/image data. A section may have multiple source positions. Coordinates must be transformed consistently between rendered image pixels, table crops, and PDF points. [Shared result types](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/type/types.go#L142)

## Decisions for this repository

The local first stage uses PyMuPDF for native text and geometry and pdfplumber for table extraction. It adopts the separation of acquisition, layout, structured tables, cleanup, reading order, and provenance. This is a geometric baseline informed by DeepDoc, not a port or neural-layout implementation.

- Preserve the physical, 1-based PDF page number as the citation key. Store `pdf_page_label` separately when PDF label metadata is available; it can differ because of covers and introductory pages. This field reads PDF metadata, not printed folio numerals through OCR, and is not guaranteed to match the visible page number. Keep page width/height and PDF-point bounding boxes using a documented origin.
- Infer columns per page or local region. Full-width headings should separate reading-order bands. Exclude recognized tables from prose-column detection, and avoid merging across a gutter or through a table.
- Detect recurring headers/footers conservatively using margin position plus text/position recurrence. Preserve excluded text with a reason so mistakes are auditable. A one-off body heading, financial footnote, or table row must not disappear merely because it is near a margin.
- Preserve table rows and cells, empty cells, captions, units, source page, and bounding boxes. Export a readable representation for inspection, but use structured cells for retrieval and downstream numeric handling. Suppress duplicate prose only when a table extraction has passed acceptance checks.
- Keep source text, cleaned text, extraction method, warnings, and a stable document identifier. Mark image-only/garbled pages for OCR or review; never silently treat an empty extraction as a successful page.
- Validate a sample of actual reports containing two-column prose, financial tables, headings, and footnotes. Compare reading order and table values with rendered pages. Document-level success counts alone cannot demonstrate extraction accuracy.

These choices deliberately do not copy every upstream heuristic. Current Go column assignment can force all pages to one column when the document's most common prediction is one; annual reports with marketing pages followed by a differently formatted 10-K need page-local treatment. [Upstream majority behavior](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/layout/combined_column.go#L46)

## Model requirements and limits

Running DeepDoc itself requires trained OCR, layout, and TSR models plus their inference runtime. The Python snapshot imports ONNX Runtime, downloads model assets when absent, and supports CPU inference as well as supported accelerator paths. Current Go code uses an in-process, cgo-gated ONNX backend and checks model availability. Installing PyMuPDF and pdfplumber alone does not provide these neural stages. [Python OCR runtime](https://github.com/infiniflow/ragflow/blob/a024bea0cd93f39e6652a42bf84dd20c55bc560b/deepdoc/vision/ocr.py#L70), [current native analyzer](https://github.com/infiniflow/ragflow/blob/ef150bc2cd2c4b6ccb6be72014e1d8bc382bff6c/internal/deepdoc/parser/pdf/inference/native_analyzer/native_analyzer.go#L1)

An often-repeated description of DeepDoc mentions XGBoost paragraph merging. In v0.27.2 the Python class does load a concatenation model, but `_concat_downward` sorts and immediately returns, making its later learned-merge branch unreachable. Do not claim that the reviewed version executes that branch. The separate naive vertical merge is called by its bounding-box parsing path. [Python parser initialization and merge implementation](https://github.com/infiniflow/ragflow/blob/a024bea0cd93f39e6652a42bf84dd20c55bc560b/deepdoc/parser/pdf_parser.py#L1115)

Current local limitations:

- No OCR or neural layout/TSR models run. Image-only or garbled native text requires review and a later OCR stage; chart content is not reconstructed.
- Tables on rotated PDF pages are not extracted. The parser retains native source text and reports the unsupported orientation when table extraction is attempted.
- Table detection favors numeric financial tables. The default `auto` mode uses a numeric-text candidate gate; `all` scans every selected page but still applies conservative table acceptance rules. Qualitative, sparse, or unusually formatted tables can remain as text.
- Spanning-cell and multirow-header reconstruction is not guaranteed. Empty/null cells and source evidence are retained for review rather than converted into a claimed complete table structure.
- Reading order and margin removal use geometry and recurrence heuristics. Unusual mixed layouts can require correction; warnings are review signals, not proof that every problematic page was detected.

Neural-layout/OCR integration remains a separately testable enhancement; no DeepDoc model weights or source code are bundled here.
