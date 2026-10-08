# Annual-report PDF corpus

This corpus contains 200 annual-report PDFs: 40 companies, with one report for each of FY2021 through FY2025. Collected on October 8, 2026.

## Layout

```text
data/raw/<TICKER>/<TICKER>_FY<YEAR>_annual_report.pdf
```

| Group | Companies | PDFs |
| --- | ---: | ---: |
| Semiconductors | 8 | 40 |
| Software and Cloud Services | 8 | 40 |
| Retail and Consumer Brands | 8 | 40 |
| Food, Beverages, and Household Products | 8 | 40 |
| Industrial and Machinery Companies | 8 | 40 |
| Total | 40 | 200 |

## Sources and verification

- `manifest.csv` and `manifest.json` record the actual download URL, resolved URL, source archive, company, fiscal year, file path, page count, size, and SHA-256 checksum.
- `sources_*.json` contain the source mappings and available fallback links used by the collector.
- `validation_report.json` records inventory checks, duplicate checks, company identity, fiscal-period evidence with PDF page numbers, and text-quality checks. Automated review flags are distinguished from manual resolutions.
- The PDFs are downloaded files, without page removal, recompression, or edits. The corpus includes annual reports, annual reports bundled with proxy materials, and standalone annual reports on Form 10-K.
- Company investor-relations sites and SEC-hosted PDFs were used where reachable. Annreports and Stocklight copies fill gaps or replace documents with unusable text encoding. Some mirrors include publisher cover pages.

## Fiscal years and page citations

Filenames use the company's fiscal-year designation, not the filing or download year. For example, Target, Home Depot, and Lowe's can end a fiscal year in the following calendar year. Cadence FY2021 ended January 1, 2022. A mirror's URL may likewise contain the filing year rather than the fiscal year.

For Q&A ingestion, preserve the filename, ticker, fiscal year, SHA-256, and **1-based PDF page number** on every text chunk. Printed page labels can differ from PDF page numbers because of covers, proxy statements, and publisher material. Keep the two page-number fields separate if printed labels are extracted. Retain page boundaries during extraction, and exclude publisher front matter and proxy sections when they are outside the question's scope.

Checks confirm that each PDF opens and contains extractable report text. Table layouts, scanned covers, and occasional letter spacing still require normal extraction handling; these checks do not certify every extracted table cell. No OCR or Q&A index has been generated.

## Reproduce or resume

Use Python with `pypdf` and `pypdfium2`, plus Poppler's `pdfinfo` on `PATH`. Run from the project root:

```shell
python scripts/collect_reports.py --only-missing --workers 4
python scripts/verify_corpus.py
```

Existing validated PDFs are retained. Failed downloads appear in the manifest. `--replace TICKER:YEAR` downloads a replacement only after its source mapping has been updated; the existing file stays in place until the new PDF passes the download checks. Public source URLs may change over time.
